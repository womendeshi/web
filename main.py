import io
import hashlib
import json
import logging
import os
import re
from typing import Dict, Any, List, Optional
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import jieba
import PyPDF2
from openai import OpenAI

try:
    # swagger-ui-bundle-v2 的模块名仍然叫 swagger_ui_bundle
    from swagger_ui_bundle import swagger_ui_path

    _HAVE_SWAGGER_BUNDLE = True
except Exception:
    swagger_ui_path = None  # type: ignore
    _HAVE_SWAGGER_BUNDLE = False

try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore


app = FastAPI(title="AI Resume Analyzer", version="0.1.0", docs_url=None)
logger = logging.getLogger("uvicorn.error")

_cors_origins_raw = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
if _cors_origins_raw in {"*", ""}:
    _cors_allow_origins = ["*"]
else:
    _cors_allow_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResumeUploadResponse(BaseModel):
    resume_id: str
    raw_text: str
    structured_info: Dict[str, Any]


class MatchRequest(BaseModel):
    resume_id: Optional[str] = None
    resume_text: Optional[str] = None
    job_description: str


class MatchByIdRequest(BaseModel):
    job_description: str


class MatchResponse(BaseModel):
    resume_id: Optional[str]
    job_keywords: List[str]
    scores: Dict[str, float]
    analysis: Dict[str, Any]


# 简单内存缓存：实际部署时可以替换为 Redis
RESUME_CACHE: Dict[str, Dict[str, Any]] = {}


def _redis_cache_enabled() -> bool:
    return os.getenv("USE_REDIS_CACHE", "0").lower() in {"1", "true", "yes"} and redis is not None


_redis_client_singleton = None


def _get_redis_client():
    global _redis_client_singleton
    if not _redis_cache_enabled():
        return None
    if _redis_client_singleton is not None:
        return _redis_client_singleton

    redis_url = os.getenv("REDIS_URL")
    try:
        if redis_url:
            _redis_client_singleton = redis.Redis.from_url(redis_url, decode_responses=True)
        else:
            host = os.getenv("REDIS_HOST", "127.0.0.1")
            port = int(os.getenv("REDIS_PORT", "6379"))
            password = os.getenv("REDIS_PASSWORD")
            db = int(os.getenv("REDIS_DB", "0"))
            _redis_client_singleton = redis.Redis(
                host=host,
                port=port,
                password=password,
                db=db,
                decode_responses=True,
            )
        return _redis_client_singleton
    except Exception as e:
        logger.error("Redis init failed (fallback to memory): %s", e)
        _redis_client_singleton = None
        return None


def _cache_ttl_seconds() -> int:
    try:
        return int(os.getenv("CACHE_TTL_SECONDS", "86400"))
    except Exception:
        return 86400


def _redis_get_json(key: str) -> Optional[Dict[str, Any]]:
    client = _get_redis_client()
    if not client:
        return None
    try:
        raw = client.get(key)
        if not raw:
            return None
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception as e:
        logger.error("Redis get failed (key=%s): %s", key, e)
        return None


def _redis_set_json(key: str, value: Dict[str, Any], ttl_seconds: int) -> None:
    client = _get_redis_client()
    if not client:
        return
    try:
        client.set(key, json.dumps(value, ensure_ascii=False), ex=ttl_seconds)
    except Exception as e:
        logger.error("Redis set failed (key=%s): %s", key, e)


def _resume_cache_key(resume_id: str) -> str:
    return f"resume:{resume_id}"


def _match_cache_key(resume_id: str, job_description: str) -> str:
    salt = f"ai={int(_llm_matching_enabled())}|model={os.getenv('DOUBAO_MATCH_MODEL', '')}"
    h = hashlib.sha256((salt + "||" + job_description).encode("utf-8")).hexdigest()
    return f"match:{resume_id}:{h}"


def cache_get_resume(resume_id: str) -> Optional[Dict[str, Any]]:
    # 优先 Redis（多实例共享），再回退内存
    data = _redis_get_json(_resume_cache_key(resume_id)) if _redis_cache_enabled() else None
    if data and "raw_text" in data and "structured_info" in data:
        return data
    return RESUME_CACHE.get(resume_id)


def cache_put_resume(resume_id: str, raw_text: str, structured_info: Dict[str, Any]) -> None:
    payload = {"raw_text": raw_text, "structured_info": structured_info}
    RESUME_CACHE[resume_id] = payload
    if _redis_cache_enabled():
        _redis_set_json(_resume_cache_key(resume_id), payload, _cache_ttl_seconds())


def cache_get_match(resume_id: str, job_description: str) -> Optional[Dict[str, Any]]:
    if _redis_cache_enabled():
        return _redis_get_json(_match_cache_key(resume_id, job_description))
    return None


def cache_put_match(resume_id: str, job_description: str, payload: Dict[str, Any]) -> None:
    if _redis_cache_enabled():
        _redis_set_json(_match_cache_key(resume_id, job_description), payload, _cache_ttl_seconds())

if _HAVE_SWAGGER_BUNDLE and swagger_ui_path:
    # 自托管 Swagger UI 静态资源，避免 Edge Tracking Prevention 拦截外网 CDN
    app.mount(
        "/static/swagger-ui",
        StaticFiles(directory=swagger_ui_path),
        name="swagger-ui",
    )


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    swagger_js_url = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"
    swagger_css_url = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"
    swagger_favicon_url = "https://fastapi.tiangolo.com/img/favicon.png"

    if _HAVE_SWAGGER_BUNDLE:
        swagger_js_url = "/static/swagger-ui/swagger-ui-bundle.js"
        swagger_css_url = "/static/swagger-ui/swagger-ui.css"
        # swagger-ui-bundle 内通常带 favicon 文件，存在则用本地的
        swagger_favicon_url = "/static/swagger-ui/favicon-32x32.png"

    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url="/docs/oauth2-redirect",
        swagger_js_url=swagger_js_url,
        swagger_css_url=swagger_css_url,
        swagger_favicon_url=swagger_favicon_url,
    )


@app.get("/docs/oauth2-redirect", include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


def _llm_enabled() -> bool:
    # 仅表示豆包可用（已配置 key）；具体功能由各自开关控制
    return os.getenv("DOUBAO_API_KEY") not in {None, ""}


def _llm_extraction_enabled() -> bool:
    return os.getenv("ENABLE_LLM_EXTRACTION", "0").lower() in {"1", "true", "yes"} and _llm_enabled()


def _llm_matching_enabled() -> bool:
    return os.getenv("ENABLE_LLM_MATCHING", "0").lower() in {"1", "true", "yes"} and _llm_enabled()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    full_text = "\n".join(texts)
    full_text = re.sub(r"\s+", " ", full_text)
    return full_text.strip()


def extract_basic_info(text: str) -> Dict[str, Any]:
    # 电话
    phone_match = re.search(r"(1[3-9]\d{9})", text)
    phone = phone_match.group(1) if phone_match else None

    # 邮箱
    email_match = re.search(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text
    )
    email = email_match.group(0) if email_match else None

    # 姓名（极简启发式）
    name = None
    first_200 = text[:200]
    name_label_match = re.search(r"(姓名[:：]?\s*([^\s，,。]{2,4}))", first_200)
    if name_label_match:
        name = name_label_match.group(2)
    if not name:
        title_match = re.search(r"([\u4e00-\u9fa5]{2,4})(?:简历|求职)", first_200)
        if title_match:
            name = title_match.group(1)

    # 地址
    address = None
    address_match = re.search(
        r"(?:现居住地|所在地|地址|现居)[:：]?\s*([^\s，,。]{2,20})", text
    )
    if address_match:
        address = address_match.group(1)

    # 学历（简单匹配最高学历关键词）
    education_levels = ["博士", "硕士", "研究生", "本科", "大专", "专科", "高中"]
    education = None
    for level in education_levels:
        if level in text:
            education = level
            break

    # 求职意向
    expected_position = None
    jd_match = re.search(
        r"(?:求职意向|期望职位|目标岗位)[:：]?\s*([^\n，,。]{2,30})", text
    )
    if jd_match:
        expected_position = jd_match.group(1).strip()

    # 工作年限（例如 “3年工作经验”）
    years_of_experience = None
    years_match = re.search(r"(\d+)\s*年(?:工作)?经验", text)
    if years_match:
        try:
            years_of_experience = int(years_match.group(1))
        except ValueError:
            years_of_experience = None

    # 技能关键字：基于简单词典 + 统计
    skill_dict = [
        "Python",
        "Java",
        "C++",
        "Go",
        "JavaScript",
        "TypeScript",
        "Django",
        "Flask",
        "FastAPI",
        "Spring",
        "MySQL",
        "PostgreSQL",
        "Redis",
        "Kafka",
        "Docker",
        "Kubernetes",
        "Linux",
        "Pandas",
        "NumPy",
        "PyTorch",
        "TensorFlow",
    ]
    skills_found = []
    for kw in skill_dict:
        if kw.lower() in text.lower():
            skills_found.append(kw)

    info = {
        "name": name,
        "phone": phone,
        "email": email,
        "address": address,
        "education": education,
        "expected_position": expected_position,
        "years_of_experience": years_of_experience,
        "skills": list(set(skills_found)),
    }
    return info


def extract_basic_info_ai(text: str) -> Optional[Dict[str, Any]]:
    """可选：调用豆包大模型做关键信息抽取，失败时返回 None。"""
    if not _llm_extraction_enabled():
        return None

    api_key = os.getenv("DOUBAO_API_KEY")
    # 默认使用一个通用对话模型，名称可通过环境变量覆盖
    model_name = os.getenv("DOUBAO_EXTRACT_MODEL", os.getenv("DOUBAO_MODEL", "doubao-1-5-pro-32k-250115"))
    base_url = os.getenv(
        "DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"
    )

    system_prompt = (
        "你是一个简历解析助手。请从给定的中文或英文简历文本中抽取关键信息，"
        "严格以 JSON 格式输出，不要包含任何多余说明文字。\n"
        "JSON 字段包括：\n"
        "name: 姓名（字符串）\n"
        "phone: 电话号码（字符串）\n"
        "email: 邮箱（字符串）\n"
        "address: 地址（字符串）\n"
        "expected_position: 求职意向/期望职位（字符串）\n"
        "years_of_experience: 工作年限（数字，年）\n"
        "education: 学历背景（字符串，例如 本科 计算机科学 与 技术）\n"
        "skills: 技能列表（字符串数组，例如 [\"Python\", \"Django\"]）\n"
        "如果某项无法确定，请填 null。"
    )

    user_prompt = (
        "下面是一份候选人的完整简历文本，请按要求抽取信息并只以 JSON 返回：\n\n"
        f"{text}"
    )

    try:
        logger.info("Doubao extraction: calling model=%s base_url=%s", model_name, base_url)
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
    except Exception as e:  # 网络或 SDK 错误直接回退
        logger.error("Doubao 调用失败: %s", e)
        return None

    try:
        choice = resp.choices[0]
        content = choice.message.content
        # OpenAI SDK 返回的 content 通常是字符串
        content_text = str(content)

        data = json.loads(content_text)
        # 只保留我们关心的字段
        extracted = {
            "name": data.get("name"),
            "phone": data.get("phone"),
            "email": data.get("email"),
            "address": data.get("address"),
            "expected_position": data.get("expected_position"),
            "years_of_experience": data.get("years_of_experience"),
            "education": data.get("education"),
            "skills": data.get("skills") or [],
        }
        logger.info("Doubao extraction OK: model=%s", model_name)
        return extracted
    except Exception as e:
        logger.error("解析 Doubao 返回 JSON 失败: %s", e)
        return None


def _safe_json_loads_maybe(content_text: str) -> Optional[Dict[str, Any]]:
    """
    兼容模型输出可能带 ```json ... ``` 或多余文字的情况：
    - 先尝试直接 json.loads
    - 再截取第一个 { 到最后一个 } 的子串
    """
    try:
        data = json.loads(content_text)
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        pass

    start = content_text.find("{")
    end = content_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(content_text[start : end + 1])
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def score_resume_match_ai(
    *,
    resume_text: str,
    resume_info: Dict[str, Any],
    job_description: str,
    job_keywords: List[str],
) -> Optional[Dict[str, Any]]:
    """可选：调用豆包模型对简历与 JD 做语义匹配评分，失败返回 None。"""
    if not _llm_matching_enabled():
        return None

    api_key = os.getenv("DOUBAO_API_KEY")
    model_name = os.getenv("DOUBAO_MATCH_MODEL", os.getenv("DOUBAO_MODEL", "doubao-1-5-pro-32k-250115"))
    base_url = os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")

    # 控制输入长度，避免 token 过大
    resume_text_trim = resume_text[:8000]
    job_desc_trim = job_description[:6000]

    system_prompt = (
        "你是招聘简历与岗位匹配评估助手。你的任务是根据简历内容和岗位描述给出匹配度评分。\n"
        "请严格只输出 JSON，不要包含任何解释性文本或 Markdown。\n"
        "评分范围均为 0 到 1 的小数，保留 2 位小数。\n"
        "输出字段：\n"
        "semantic_similarity_score: 简历与 JD 整体语义相似度\n"
        "skill_match_score: 技能匹配度\n"
        "experience_relevance_score: 工作经历/年限/方向相关性\n"
        "overall_score: 综合评分（你综合上述维度给出）\n"
        "matched_keywords: 命中的关键词列表（字符串数组）\n"
        "missing_keywords: 缺失的关键词列表（字符串数组）\n"
        "comment: 一句话简评（字符串）\n"
        "如果无法判断，请给出保守分数并说明原因。"
    )

    user_payload = {
        "resume_info": resume_info,
        "resume_text": resume_text_trim,
        "job_description": job_desc_trim,
        "job_keywords": job_keywords,
    }

    try:
        logger.info("Doubao matching: calling model=%s base_url=%s", model_name, base_url)
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            temperature=0.1,
        )
    except Exception as e:
        logger.error("Doubao matching 调用失败: %s", e)
        return None

    try:
        content_text = str(resp.choices[0].message.content or "")
        data = _safe_json_loads_maybe(content_text)
        if not data:
            logger.error("Doubao matching 返回非 JSON: %s", content_text[:2000])
            return None

        # 归一化字段
        out = {
            "semantic_similarity_score": data.get("semantic_similarity_score"),
            "skill_match_score": data.get("skill_match_score"),
            "experience_relevance_score": data.get("experience_relevance_score"),
            "overall_score": data.get("overall_score"),
            "matched_keywords": data.get("matched_keywords") or [],
            "missing_keywords": data.get("missing_keywords") or [],
            "comment": data.get("comment"),
        }
        logger.info("Doubao matching OK: model=%s", model_name)
        return out
    except Exception as e:
        logger.error("解析 Doubao matching 返回失败: %s", e)
        return None


def extract_keywords_from_jd(jd_text: str, top_k: int = 15) -> List[str]:
    # 使用 jieba 分词 + 简单 TF 计数
    words = [w for w in jieba.cut(jd_text) if len(w.strip()) > 1]
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [w for w, _ in sorted_words[:top_k]]
    return keywords


def compute_match_scores(
    resume_info: Dict[str, Any],
    resume_text: str,
    job_keywords: List[str],
) -> Dict[str, Any]:
    resume_text_lower = resume_text.lower()
    resume_skills = set([s.lower() for s in resume_info.get("skills", [])])

    # 技能匹配度：JD 关键词中出现在简历技能或文本中的比例
    matched_keywords = []
    for kw in job_keywords:
        if kw.lower() in resume_skills or kw.lower() in resume_text_lower:
            matched_keywords.append(kw)

    skill_match_score = (
        len(matched_keywords) / len(job_keywords) if job_keywords else 0.0
    )

    # 工作经验匹配：基于工作年限的简单评分
    years = resume_info.get("years_of_experience")
    exp_score = 0.5
    if years is not None:
        if years >= 5:
            exp_score = 1.0
        elif years >= 3:
            exp_score = 0.8
        elif years >= 1:
            exp_score = 0.6
        else:
            exp_score = 0.3

    overall_score = round(
        0.7 * skill_match_score + 0.3 * exp_score, 4
    )

    analysis = {
        "matched_skills": matched_keywords,
        "missing_skills": [kw for kw in job_keywords if kw not in matched_keywords],
        "comment": "基于关键词和工作年限的启发式匹配，仅供参考。",
    }

    scores = {
        "skill_match_score": round(skill_match_score, 4),
        "experience_match_score": round(exp_score, 4),
        "overall_score": overall_score,
    }

    return {
        "scores": scores,
        "analysis": analysis,
    }


@app.post("/api/resume/upload", response_model=ResumeUploadResponse)
async def upload_resume(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="文件内容为空")

    try:
        text = extract_text_from_pdf(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 解析失败: {e}")

    if not text:
        raise HTTPException(status_code=400, detail="未从 PDF 中提取到文本")

    # 先用规则方法抽取，保证即使没有大模型也有结果
    info = extract_basic_info(text)

    # 如配置了大模型，则尝试用大模型再抽一遍，成功则覆盖/补充字段
    ai_info = extract_basic_info_ai(text)
    if ai_info:
        merged = info.copy()
        for k, v in ai_info.items():
            if v not in (None, "", [], {}):
                merged[k] = v
        info = merged
    else:
        if _llm_extraction_enabled():
            logger.info("Doubao extraction fallback to rules")
    resume_id = str(uuid4())
    cache_put_resume(resume_id=resume_id, raw_text=text, structured_info=info)

    return ResumeUploadResponse(
        resume_id=resume_id,
        raw_text=text,
        structured_info=info,
    )


@app.post("/api/resume/{resume_id}/match", response_model=MatchResponse)
async def match_uploaded_resume(resume_id: str, req: MatchByIdRequest):
    """
    更贴合题目流程：上传简历后，使用岗位描述对该 resume_id 做匹配评分。
    请求体只需要 job_description。
    """
    if not req.job_description:
        raise HTTPException(status_code=400, detail="job_description 不能为空")

    # 先查“匹配结果缓存”（同一 resume_id + 同一 JD 直接返回）
    cached_match = cache_get_match(resume_id, req.job_description)
    if cached_match:
        return MatchResponse(**cached_match)

    cached = cache_get_resume(resume_id)
    if not cached:
        raise HTTPException(status_code=404, detail="未找到对应的 resume_id")

    resume_text = cached["raw_text"]
    resume_info = cached["structured_info"]

    job_keywords = extract_keywords_from_jd(req.job_description)
    result = compute_match_scores(resume_info, resume_text, job_keywords)

    ai_match = score_resume_match_ai(
        resume_text=resume_text,
        resume_info=resume_info,
        job_description=req.job_description,
        job_keywords=job_keywords,
    )
    if ai_match:
        result["scores"]["semantic_similarity_score"] = float(
            ai_match.get("semantic_similarity_score") or 0.0
        )
        result["scores"]["ai_skill_match_score"] = float(
            ai_match.get("skill_match_score") or 0.0
        )
        result["scores"]["ai_experience_relevance_score"] = float(
            ai_match.get("experience_relevance_score") or 0.0
        )
        result["scores"]["ai_overall_score"] = float(ai_match.get("overall_score") or 0.0)

        result["analysis"]["matched_skills"] = ai_match.get("matched_keywords") or result[
            "analysis"
        ].get("matched_skills")
        result["analysis"]["missing_skills"] = ai_match.get("missing_keywords") or result[
            "analysis"
        ].get("missing_skills")
        if ai_match.get("comment"):
            result["analysis"]["comment"] = ai_match["comment"]
        result["analysis"]["ai_matching_enabled"] = True
    else:
        if _llm_matching_enabled():
            logger.info("Doubao matching fallback to rules")
        result["analysis"]["ai_matching_enabled"] = False

    resp_payload = {
        "resume_id": resume_id,
        "job_keywords": job_keywords,
        "scores": result["scores"],
        "analysis": result["analysis"],
    }

    # 写入匹配缓存（Redis）
    cache_put_match(resume_id, req.job_description, resp_payload)

    return MatchResponse(
        resume_id=resume_id,
        job_keywords=job_keywords,
        scores=result["scores"],
        analysis=result["analysis"],
    )


@app.get("/health")
async def health_check():
    return JSONResponse({"status": "ok"})


@app.on_event("startup")
async def _startup_log_routes():
    try:
        import main as _main_mod  # noqa: F401

        logger.info("Loaded module file: %s", getattr(_main_mod, "__file__", None))
    except Exception:
        logger.exception("Failed to log loaded module file")

    paths = []
    for r in app.routes:
        p = getattr(r, "path", None)
        if p:
            paths.append(p)
    logger.info("Registered paths: %s", sorted(set(paths)))


@app.get("/debug/routes")
async def debug_routes():
    out = []
    for r in app.routes:
        out.append(
            {
                "path": getattr(r, "path", None),
                "name": getattr(r, "name", None),
                "methods": sorted(list(getattr(r, "methods", []) or [])),
            }
        )
    return {"routes": out}

