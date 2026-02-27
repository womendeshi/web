function $(id) {
  return document.getElementById(id);
}

function now() {
  return new Date().toISOString();
}

function log(line) {
  const el = $("log");
  el.textContent = `[${now()}] ${line}\n` + el.textContent;
}

function getApiBase() {
  const v = $("apiBase").value.trim();
  return v ? v.replace(/\/+$/, "") : "http://127.0.0.1:8000";
}

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { raw: text };
  }
  if (!res.ok) {
    const detail = data && data.detail ? data.detail : text;
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  return data;
}

async function health() {
  const api = getApiBase();
  log(`Health check: ${api}/health`);
  const data = await fetchJson(`${api}/health`, { method: "GET" });
  log(`Health OK: ${JSON.stringify(data)}`);
}

async function upload() {
  const api = getApiBase();
  const file = $("pdfFile").files[0];
  if (!file) {
    alert("请选择一个 PDF 文件");
    return;
  }

  $("btnUpload").disabled = true;
  try {
    log(`Uploading: ${file.name} (${file.size} bytes)`);
    const fd = new FormData();
    fd.append("file", file);
    const data = await fetchJson(`${api}/api/resume/upload`, {
      method: "POST",
      body: fd,
    });

    $("resumeId").textContent = data.resume_id || "-";
    $("structuredInfo").textContent = JSON.stringify(data.structured_info, null, 2);
    $("rawText").textContent = data.raw_text || "-";
    log(`Upload OK, resume_id=${data.resume_id}`);
  } finally {
    $("btnUpload").disabled = false;
  }
}

async function match() {
  const api = getApiBase();
  const resumeId = $("resumeId").textContent.trim();
  if (!resumeId || resumeId === "-") {
    alert("请先上传简历，拿到 resume_id");
    return;
  }
  const jobDesc = $("jobDesc").value.trim();
  if (!jobDesc) {
    alert("请填写岗位需求描述（JD）");
    return;
  }

  $("btnMatch").disabled = true;
  try {
    log(`Matching resume_id=${resumeId}`);
    const data = await fetchJson(`${api}/api/resume/${encodeURIComponent(resumeId)}/match`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_description: jobDesc }),
    });
    $("matchResult").textContent = JSON.stringify(data, null, 2);
    log(`Match OK, overall_score=${data?.scores?.overall_score ?? "n/a"}`);
  } finally {
    $("btnMatch").disabled = false;
  }
}

function initDefaults() {
  $("apiBase").value = localStorage.getItem("apiBase") || "http://127.0.0.1:8000";
  $("apiBase").addEventListener("change", () => {
    localStorage.setItem("apiBase", $("apiBase").value.trim());
  });

  $("btnHealth").addEventListener("click", () => health().catch((e) => log(`Health ERROR: ${e.message}`)));
  $("btnUpload").addEventListener("click", () => upload().catch((e) => log(`Upload ERROR: ${e.message}`)));
  $("btnMatch").addEventListener("click", () => match().catch((e) => log(`Match ERROR: ${e.message}`)));
}

initDefaults();

