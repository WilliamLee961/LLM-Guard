const queryEl = document.getElementById("query");
const difficultyEl = document.getElementById("difficulty");
const engineEl = document.getElementById("engine");
const simulateUnsafeEl = document.getElementById("simulateUnsafe");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
const examplesEl = document.getElementById("examples");

const summaryEl = document.getElementById("summary");
const inputGuardEl = document.getElementById("inputGuard");
const outputGuardEl = document.getElementById("outputGuard");
const rawAnswerEl = document.getElementById("rawAnswer");
const finalAnswerEl = document.getElementById("finalAnswer");

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function riskClass(level) {
  const normalized = (level || "").toUpperCase();
  if (normalized === "HIGH") return "risk-high";
  if (normalized === "MEDIUM") return "risk-medium";
  if (normalized === "LOW") return "risk-low";
  return "risk-none";
}

function collectRisks(profile) {
  const levels = profile?.risk_levels || {};
  return Object.entries(levels).filter(([, level]) => String(level).toUpperCase() !== "NONE");
}

function renderGuardBlock(container, data, title) {
  const pred = Number(data?.pred_violation || 0);
  const profile = data?.risk_profile || {};
  const risks = collectRisks(profile);
  const violationClass = pred === 1 ? "violation-blocked" : "violation-pass";
  const violationText = pred === 1 ? "触发拦截" : "通过";

  const riskTags = risks.length
    ? risks
        .slice(0, 10)
        .map(([cat, level]) => `<span class="risk-tag ${riskClass(level)}">${escapeHtml(cat)}:${escapeHtml(level)}</span>`)
        .join("")
    : '<span class="risk-tag risk-none">NO_RISK</span>';

  container.innerHTML = `
    <div><strong>${escapeHtml(title)}</strong></div>
    <div>违规判定：<span class="${violationClass}">${violationText}</span></div>
    <div>整体等级：${escapeHtml(profile?.overall_risk_level || "L1")}</div>
    <div class="risk-tag-wrap">${riskTags}</div>
    <details style="margin-top:8px;">
      <summary>查看 router_ctx</summary>
      <pre class="mono-block">${escapeHtml(JSON.stringify(data?.router_ctx || {}, null, 2))}</pre>
    </details>
    <details>
      <summary>查看 risk_profile</summary>
      <pre class="mono-block">${escapeHtml(JSON.stringify(profile, null, 2))}</pre>
    </details>
  `;
}

function renderSummary(result) {
  const blocked = result?.blocked_stage || "none";
  const fallback = result?.used_fallback ? "yes" : "no";
  summaryEl.innerHTML = `
    <div class="metric">
      <span class="label">请求引擎</span>
      <span class="value">${escapeHtml(result?.requested_engine || "-")}</span>
    </div>
    <div class="metric">
      <span class="label">实际引擎</span>
      <span class="value">${escapeHtml(result?.engine_used || "-")}</span>
    </div>
    <div class="metric">
      <span class="label">阻断阶段</span>
      <span class="value">${escapeHtml(blocked)}</span>
    </div>
    <div class="metric">
      <span class="label">处理动作</span>
      <span class="value">${escapeHtml(result?.action || "-")}</span>
    </div>
    <div class="metric">
      <span class="label">降级兜底</span>
      <span class="value">${fallback}</span>
    </div>
    <div class="metric">
      <span class="label">耗时</span>
      <span class="value">${escapeHtml(String(result?.latency_ms ?? "-"))} ms</span>
    </div>
  `;
}

function setBusy(busy) {
  runBtn.disabled = busy;
  runBtn.textContent = busy ? "运行中..." : "运行 Guard 流程";
}

async function runGuardedFlow() {
  const query = queryEl.value.trim();
  if (!query) {
    statusEl.textContent = "请输入用户问题";
    return;
  }

  setBusy(true);
  statusEl.textContent = "正在请求后端并执行输入/输出 Guard...";

  try {
    const resp = await fetch("/api/guarded-chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        difficulty: difficultyEl.value,
        engine: engineEl.value,
        simulate_unsafe_output: simulateUnsafeEl.checked
      })
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${text}`);
    }

    const result = await resp.json();
    renderSummary(result);
    renderGuardBlock(inputGuardEl, result.input_guard, "输入侧检测");
    renderGuardBlock(outputGuardEl, result.output_guard, "输出侧检测");
    rawAnswerEl.textContent = result?.model?.raw_answer || "(输入侧已拦截，无原始模型输出)";
    finalAnswerEl.textContent = result?.final_answer || "-";

    if (result.used_fallback) {
      statusEl.textContent = `请求完成。真实引擎不可用，已自动降级到 demo 引擎。原因：${result.fallback_reason || "unknown"}`;
    } else {
      statusEl.textContent = "请求完成。";
    }
  } catch (err) {
    statusEl.textContent = `请求失败：${err.message}`;
  } finally {
    setBusy(false);
  }
}

async function loadExamples() {
  try {
    const resp = await fetch("/api/examples");
    if (!resp.ok) {
      return;
    }
    const data = await resp.json();
    const examples = data.examples || [];
    examplesEl.innerHTML = "";
    examples.forEach((item) => {
      const btn = document.createElement("button");
      btn.className = "chip";
      btn.type = "button";
      btn.textContent = item.label || "示例";
      btn.addEventListener("click", () => {
        queryEl.value = item.text || "";
      });
      examplesEl.appendChild(btn);
    });
  } catch (_err) {
    // ignore
  }
}

runBtn.addEventListener("click", runGuardedFlow);
loadExamples();
