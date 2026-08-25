const SAMPLES = {
  auto: [
    { label: "Quotes threshold", q: "How many competitive quotes are required above 50,000 SAR?" },
    { label: "إجازة سنوية", q: "كم يوم الإجازة السنوية للموظف؟" },
  ],
  search: [
    { label: "Vendor onboarding", q: "vendor onboarding KYC security review" },
    { label: "بحث MFA", q: "ابحث عن متطلبات المصادقة متعددة العوامل" },
  ],
  qa: [
    { label: "Sick leave", q: "How many sick-leave days does a full-time employee get?" },
    { label: "حادث أمني", q: "خلال كم ساعة يجب الإبلاغ عن حادث أمني؟" },
  ],
  summarize: [
    { label: "Procurement", q: "Summarize the procurement approval thresholds." },
    { label: "ملخص الإجازات", q: "لخص سياسة الإجازات" },
  ],
  analyze: [
    { label: "Contract risks", q: "Analyze the risks in the cloud hosting master agreement." },
    { label: "تحليل العقد", q: "حلل مخاطر اتفاقية الاستضافة السحابية" },
  ],
  extract: [
    { label: "SLA fields", q: "Extract SLA, liability cap, and termination terms from the vendor contract." },
    { label: "استخراج", q: "استخرج حدود الاعتماد والمبالغ من سياسة المشتريات" },
  ],
  decide: [
    { label: "Cloud choice", q: "Should we migrate production to Azure, AWS, or stay on-premises?" },
    { label: "قرار سحابي", q: "هل يجب الانتقال إلى أزور أم البقاء في الموقع؟" },
  ],
};

const ARABIC = /[\u0600-\u06FF]/;
let capability = "auto";

const form = document.getElementById("runForm");
const queryEl = document.getElementById("query");
const extraEl = document.getElementById("extra");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
const stepsEl = document.getElementById("steps");
const resultEl = document.getElementById("result");
const resultTitle = document.getElementById("resultTitle");
const chipsEl = document.getElementById("chips");
const modeBanner = document.getElementById("modeBanner");
const docCount = document.getElementById("docCount");

document.querySelectorAll(".cap").forEach((button) => {
  button.addEventListener("click", () => {
    capability = button.dataset.cap;
    document.querySelectorAll(".cap").forEach((item) => {
      item.classList.toggle("is-active", item === button);
      item.setAttribute("aria-selected", item === button ? "true" : "false");
    });
    renderChips();
  });
});

queryEl.addEventListener("input", () => {
  queryEl.classList.toggle("rtl", ARABIC.test(queryEl.value));
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryEl.value.trim();
  if (!query) {
    statusEl.textContent = "Enter a question or search query.";
    return;
  }
  setBusy(true);
  statusEl.textContent = "Running the graph…";
  resultEl.innerHTML = "";
  try {
    const response = await fetch("/v1/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        capability,
        extra_text: extraEl.value.trim() || null,
        evaluate: true,
        include_sources: true,
      }),
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "Request failed.");
    }
    renderResult(body);
    statusEl.textContent = `${body.capability} · ${Math.round(body.metrics.latency_ms)} ms · ${body.metrics.demo_mode ? "demo" : "live"}`;
  } catch (error) {
    statusEl.textContent = error.message;
    resultEl.innerHTML = `<p class="muted">${escapeHtml(error.message)}</p>`;
  } finally {
    setBusy(false);
  }
});

function renderChips() {
  const items = SAMPLES[capability] || SAMPLES.auto;
  chipsEl.innerHTML = "";
  items.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = item.label;
    button.addEventListener("click", () => {
      queryEl.value = item.q;
      queryEl.classList.toggle("rtl", ARABIC.test(item.q));
      queryEl.focus();
    });
    chipsEl.appendChild(button);
  });
}

function renderResult(body) {
  stepsEl.innerHTML = "";
  (body.steps || []).forEach((step) => {
    const li = document.createElement("li");
    li.className = step.status || "ok";
    li.textContent = `${step.name}: ${step.detail}`;
    stepsEl.appendChild(li);
  });

  const rtl = body.language === "ar" || ARABIC.test(body.answer || "");
  const parts = [];
  parts.push(`<p class="answer${rtl ? " rtl" : ""}">${escapeHtml(body.answer || "")}</p>`);
  if (body.evaluation) {
    parts.push(
      `<p class="meta">Groundedness ${body.evaluation.groundedness} · relevance ${body.evaluation.relevance} · language ${body.evaluation.language_quality}</p>`
    );
  }
  if (body.summary) {
    parts.push(listBlock("Executive bullets", body.summary.bullets));
    parts.push(listBlock("Key facts", body.summary.key_facts));
  }
  if (body.analysis) {
    parts.push(listBlock("Themes", body.analysis.themes));
    parts.push(risksBlock(body.analysis.risks || []));
    parts.push(listBlock("Next steps", body.analysis.recommended_next_steps));
  }
  if (body.extraction) {
    parts.push(listBlock("Entities", body.extraction.entities));
    parts.push(listBlock("Amounts", body.extraction.amounts));
    parts.push(listBlock("Dates", body.extraction.dates));
    parts.push(listBlock("Clauses", body.extraction.clauses));
  }
  if (body.decision) {
    parts.push(`<div class="block"><strong>Recommendation</strong><p>${escapeHtml(body.decision.recommendation)} (confidence ${body.decision.confidence})</p><p>${escapeHtml(body.decision.rationale || "")}</p></div>`);
    (body.decision.options || []).forEach((option) => {
      parts.push(
        `<div class="block"><strong>${escapeHtml(option.name)}</strong><p>${escapeHtml(option.summary)}</p><p class="meta">${escapeHtml((option.pros || []).join(" · "))}</p></div>`
      );
    });
  }
  (body.hits || body.sources || []).slice(0, 6).forEach((hit) => {
    const title = hit.title || hit.source || "Source";
    const snippet = hit.snippet || hit.content || "";
    const meta = [hit.department, hit.doc_type, hit.score != null ? Number(hit.score).toFixed(3) : ""]
      .filter(Boolean)
      .join(" · ");
    parts.push(`<article class="hit"><strong>${escapeHtml(title)}</strong><p class="meta">${escapeHtml(meta)}</p><p>${escapeHtml(snippet)}</p></article>`);
  });
  resultTitle.textContent = body.capability;
  resultEl.innerHTML = parts.join("");
}

function listBlock(title, items) {
  if (!items || !items.length) return "";
  return `<div class="block"><strong>${escapeHtml(title)}</strong><ul>${items
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("")}</ul></div>`;
}

function risksBlock(risks) {
  if (!risks.length) return "";
  return `<div class="block"><strong>Risks</strong><ul>${risks
    .map((risk) => `<li><strong>${escapeHtml(risk.severity)}</strong> — ${escapeHtml(risk.title)}: ${escapeHtml(risk.detail)}</li>`)
    .join("")}</ul></div>`;
}

function setBusy(busy) {
  runBtn.disabled = busy;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

fetch("/v1/health")
  .then((response) => response.json())
  .then((data) => {
    if (data.mode === "demo") {
      modeBanner.hidden = false;
      modeBanner.textContent = "Demo mode — deterministic stand-in, no API key required.";
    }
    docCount.textContent = `${data.document_count} documents indexed`;
  })
  .catch(() => {});

renderChips();
