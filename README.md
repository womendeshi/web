# AI 赋能的智能简历分析系统（后端示例）

本项目是在本地目录下使用 **FastAPI + Python** 实现的一个简历解析与匹配 RESTful API，方便后续部署到阿里云 Serverless（函数计算 + API 网关）。

当前实现功能：

- **简历上传与解析**：上传 PDF 简历，解析全文并做基本清洗。
- **关键信息提取**：从简历文本中启发式抽取姓名、电话、邮箱、地址、学历、工作年限、技能等。
- **简历评分与岗位匹配**：根据岗位描述文本提取关键词，并与简历进行匹配，输出匹配度评分及分析结果。
- **缓存机制**：默认进程内存缓存；可选启用 Redis，对已解析简历和匹配结果做缓存，避免重复处理。

> 说明：目前的“AI 能力”主要是基于规则 + 关键词的启发式算法，方便在 24 小时内快速落地。后续可以替换为阿里云灵积（DashScope）或其它大模型做更精细的抽取与匹配。

---

## 本地运行

### 1. 安装依赖

建议使用虚拟环境（例如 `venv`）。在项目根目录（本文件所在目录）执行：

```bash
pip install -r requirements.txt
```

### 2. 启动服务

```bash
python -m uvicorn main:app --reload --port 8000
```

启动成功后，服务默认监听在：

- 接口地址：`http://127.0.0.1:8000`
- Swagger 文档：`http://127.0.0.1:8000/docs`

---

## 接口说明

### 1. 上传简历并解析

- **URL**：`POST /api/resume/upload`
- **请求类型**：`multipart/form-data`
- **参数**：
  - `file`：PDF 简历文件
- **返回示例**：

```json
{
  "resume_id": "abc123",
  "raw_text": "......",
  "structured_info": {
    "name": "张三",
    "phone": "13800000000",
    "email": "xx@xx.com",
    "address": "上海市",
    "education": "本科",
    "expected_position": "Python 开发工程师",
    "years_of_experience": 3,
    "skills": ["Python", "Django", "MySQL"]
  }
}
```

### 2. 简历与岗位匹配评分

- **URL**：`POST /api/resume/{resume_id}/match`
- **请求类型**：`application/json`
- **请求体**：

```json
{
  "job_description": "岗位职责：...... 任职要求：......"
}
```

- **返回示例**：

```json
{
  "resume_id": "abc123",
  "job_keywords": ["Python", "Django", "MySQL"],
  "scores": {
    "skill_match_score": 0.82,
    "experience_match_score": 0.75,
    "overall_score": 0.8
  },
  "analysis": {
    "matched_skills": ["Python", "Django", "MySQL"],
    "missing_skills": ["Flask"],
    "comment": "基于关键词和工作年限的启发式匹配，仅供参考。"
  }
}
```

### 3. 健康检查

- **URL**：`GET /health`
- **返回**：

```json
{ "status": "ok" }
```

---

## 后续如何部署到阿里云 Serverless（思路简述）

1. 使用 **阿里云函数计算（Function Compute）** 创建 Python 运行时函数，将本项目代码打包上传。
2. 在函数入口中引入 `main:app`（FastAPI 应用），配合 HTTP 触发器或通过 API 网关转发。
3. 如需 Redis 缓存，可创建阿里云 Redis 实例，在代码中将当前的内存缓存 `RESUME_CACHE` 替换为 Redis 操作。
4. 前端可以做一个简单的上传页面（HTML/JS），部署到 GitHub Pages 或 OSS 静态网站，前端通过 HTTP 调用后端 Serverless API。

如你需要，我可以在下一步继续帮你写：

- 面向函数计算的入口文件示例
- 部署说明（`fun deploy` / 控制台配置）
- 简单前端页面示例

---

## Redis 缓存（加分项）

默认不需要 Redis，也能正常运行（仅使用进程内存缓存）。

如需启用 Redis 缓存（推荐用于多实例/Serverless 场景），设置环境变量：

```bash
export USE_REDIS_CACHE=1
export REDIS_URL="redis://:password@host:6379/0"
export CACHE_TTL_SECONDS=86400
```

也可以不用 `REDIS_URL`，改用：

```bash
export USE_REDIS_CACHE=1
export REDIS_HOST="127.0.0.1"
export REDIS_PORT=6379
export REDIS_PASSWORD=""
export REDIS_DB=0
```

缓存内容包括：

- `resume:{resume_id}`：简历解析结果（`raw_text` + `structured_info`）
- `match:{resume_id}:{hash(job_description)}`：匹配评分结果

---

## 前端（GitHub Pages 部署）

项目内置了一个最小前端（静态页面），位于 `web/`：

- 上传 PDF → 调用后端 `/api/resume/upload`
- 填写 JD → 调用后端 `/api/resume/{resume_id}/match`

### 本地打开

直接用浏览器打开 `web/index.html` 也可以使用（注意浏览器可能对 `file://` 有限制，推荐用本地静态服务）。

例如用 Python 起一个静态服务（在项目根目录执行）：

```bash
python -m http.server 5173
```

然后访问：

- `http://127.0.0.1:5173/web/`

在页面里把 **API Base URL** 填为后端地址（本地开发默认 `http://127.0.0.1:8000`）。

### GitHub Pages 部署

已提供工作流：`.github/workflows/pages.yml`，会在推送到 `main` 分支时自动把 `web/` 部署到 GitHub Pages。

操作步骤（在 GitHub 仓库里）：

1. 仓库 Settings → Pages → Build and deployment：
   - Source 选择 **GitHub Actions**
2. 推送代码到 `main` 分支
3. 等 Actions 跑完后，Pages 会生成一个可访问的前端 URL

> 注意：GitHub Pages 的域名与后端不同源，需要后端允许跨域（CORS）。
> 本项目后端已默认允许所有来源（`CORS_ALLOW_ORIGINS="*"`），你也可以按需改成只允许指定域名：
>
> ```bash
> export CORS_ALLOW_ORIGINS="https://<你的GitHub用户名>.github.io"
> ```

