"use strict";
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="annotation-token"]').content;
const dimensionNames = {subject_features: "主体关键特征", structural_integrity: "结构完整性", prompt_adherence: "选定提示词要求"};
let overview, sample, currentIndex = 0, dirty = false, busy = false;
let selectedScope = "smoke";
let loadSequence = 0;

function notice(text = "", error = false) {
  $("notice").textContent = text; $("notice").className = error ? "error" : "";
}
async function api(url, data) {
  const options = data === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json", "X-Annotation-Token": token},
    body: JSON.stringify(data)
  };
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "操作未成功，请重试。");
  return payload;
}
function complete(label) {
  return ["pass", "fail", "unjudgeable"].includes(label.status)
    && ["candidate_region", "candidate_observation", "reason"].every(key => label[key].trim());
}
function markDirty(value) { dirty = value; $("dirty").textContent = value ? "有未保存的修改" : "当前显示已保存记录"; }
function mayLeave() { return !busy && (!dirty || window.confirm("本项有未保存修改，确定离开吗？")); }
function visibleSamples() {
  return overview.samples.filter(row => selectedScope === "smoke" ? overview.smoke_ids.includes(row.sample_id)
    : selectedScope === "development" ? row.split === "development" : true);
}
function renderSamples() {
  const rows = visibleSamples(); $("samples").replaceChildren();
  for (const row of rows) {
    const button = document.createElement("button"); button.type = "button";
    button.className = "sample-button" + (sample?.sample_id === row.sample_id ? " active" : "");
    button.dataset.sampleId = row.sample_id;
    const title = document.createElement("strong"); title.textContent = row.sample_id;
    const detail = document.createElement("span");
    detail.textContent = `${row.complete} / ${row.total} 项已填${row.confirmed ? " · 已确认" : ""}`;
    button.append(title, detail);
    button.onclick = () => { if (mayLeave()) loadSample(row.sample_id).catch(showError); };
    $("samples").append(button);
  }
  const count = rows.reduce((sum, row) => sum + row.complete, 0);
  const total = rows.reduce((sum, row) => sum + row.total, 0);
  $("overall-progress").textContent = `${count} / ${total} 项已填`;
  $("overall-bar").max = total || 1; $("overall-bar").value = count;
}
function renderCriteria() {
  $("criteria").replaceChildren(); let dimension;
  sample.criteria.forEach((criterion, index) => {
    if (dimension !== criterion.dimension) {
      dimension = criterion.dimension;
      const heading = document.createElement("p"); heading.className = "nav-heading";
      heading.textContent = dimensionNames[dimension]; $("criteria").append(heading);
    }
    const record = sample.records[criterion.id];
    const done = record && complete(record.label);
    const button = document.createElement("button"); button.type = "button";
    button.className = "criterion-button" + (index === currentIndex ? " active" : "");
    button.dataset.criterionId = criterion.id;
    const dot = document.createElement("span"); dot.className = "dot" + (done ? " done" : "");
    dot.textContent = done ? "●" : "○";
    const title = document.createElement("span"); title.textContent = criterion.name;
    button.append(dot, title);
    button.onclick = () => { if (mayLeave()) { currentIndex = index; renderForm(); } };
    $("criteria").append(button);
  });
  requestAnimationFrame(revealActiveCriterion);
}
function revealActiveCriterion() {
  const nav = $("criteria").parentElement;
  const active = nav.querySelector(".criterion-button.active");
  if (!active) return;
  const container = nav.getBoundingClientRect(), item = active.getBoundingClientRect();
  if (item.top < container.top || item.bottom > container.bottom) {
    nav.scrollTop += item.top - container.top - 30;
  }
}
function renderForm() {
  const criterion = sample.criteria[currentIndex], record = sample.records[criterion.id];
  const label = record?.label || {status: "", candidate_region: criterion.region, candidate_observation: "", reason: ""};
  $("dimension").textContent = dimensionNames[criterion.dimension];
  $("criterion-name").textContent = criterion.name;
  $("criterion-id").textContent = criterion.id;
  $("required").textContent = criterion.required ? "必需项" : "辅助项";
  $("pass-rule").textContent = criterion.pass_when;
  $("fail-rule").textContent = criterion.fail_when;
  $("unknown-rule").textContent = criterion.unjudgeable_when;
  $("extra-rule").textContent = [
    criterion.allowed_variations.length ? `允许：${criterion.allowed_variations.join("；")}` : "",
    criterion.not_required.length ? `不额外要求：${criterion.not_required.join("；")}` : "",
    criterion.boundary,
    criterion.source_quote ? `原文要求：${criterion.source_quote}` : ""
  ].filter(Boolean).join("　");
  $("status").value = label.status; $("region").value = label.candidate_region;
  $("observation").value = label.candidate_observation; $("reason").value = label.reason;
  $("change-reason").value = "";
  $("change-reason").required = false; // Only server enforces it for actual changes after confirmation.
  const origins = {author_markdown_original: "人工原始填写",
    author_status_codex_evidence: "水水的原始判断；证据文字经 Codex 辅助补写",
    author_page_edit: "标注页保存；原始版本保留在历史记录"};
  $("provenance").textContent = record
    ? `${origins[record.origin.kind] || "导入记录"} · 保存版本 ${record.revision} · ${record.annotator}`
    : "尚未保存。状态保持空白，可先保存未完成草稿。";
  $("previous").disabled = currentIndex === 0; $("next").disabled = currentIndex === sample.criteria.length - 1;
  $("sample-title").textContent = sample.sample_id;
  $("sample-progress").textContent = `${sample.progress.complete} / ${sample.progress.total} 项已填`;
  $("confirmation-state").textContent = sample.initial
    ? `已保存初始快照 · ${sample.initial.confirmed_by}` : "初始记录尚未确认";
  $("confirm").disabled = Boolean(sample.initial) || sample.progress.complete !== sample.progress.total;
  $("confirm").textContent = sample.initial ? "初始快照已保留" : "确认本图初始记录";
  markDirty(false); renderCriteria(); renderSamples();
}
async function loadSample(sid) {
  const sequence = ++loadSequence;
  const loaded = await api(`/api/sample?sample_id=${encodeURIComponent(sid)}`);
  if (sequence !== loadSequence) return;
  sample = loaded;
  currentIndex = 0;
  $("prompt").textContent = sample.prompt;
  $("candidate-caption").textContent = sample.sample_id;
  $("candidate-image").src = `/api/image/${sample.sample_id}`;
  $("candidate-button").dataset.image = sample.sample_id;
  notice(); renderForm();
}
function showError(error) { notice(error.message || "操作未完成，请重试。", true); }
async function refreshOverview() {
  overview = await api("/api/overview"); $("version").textContent = overview.rule_version; renderSamples();
}
$("scope").onchange = async event => {
  if (!mayLeave()) { event.target.value = selectedScope; return; }
  selectedScope = event.target.value;
  try { renderSamples(); if (!visibleSamples().some(row => row.sample_id === sample.sample_id))
    await loadSample(visibleSamples()[0].sample_id); } catch (error) { showError(error); }
};
$("label-form").addEventListener("input", () => markDirty(true));
$("annotator").addEventListener("input", () => markDirty(true));
$("label-form").onsubmit = async event => {
  event.preventDefault(); if (busy) return;
  const criterion = sample.criteria[currentIndex];
  const payload = {sample_id: sample.sample_id, annotator: $("annotator").value,
    expected_revision: sample.records[criterion.id]?.revision || 0,
    change_reason: $("change-reason").value,
    label: {criterion_id: criterion.id, status: $("status").value,
      candidate_region: $("region").value, candidate_observation: $("observation").value,
      reason: $("reason").value}};
  busy = true; $("save").disabled = true;
  try {
    await api("/api/save", payload);
    sample = await api(`/api/sample?sample_id=${sample.sample_id}`);
    await refreshOverview(); renderForm();
    notice(complete(payload.label) ? "本项已保存，可以继续下一项。" : "草稿已保存；缺少状态或证据的条目暂不计入已填数量。");
  } catch (error) { showError(error); }
  finally { busy = false; $("save").disabled = false; }
};
for (const [id, step] of [["previous", -1], ["next", 1]]) $(id).onclick = () => {
  if (mayLeave()) { currentIndex += step; notice(); renderForm(); }
};
$("confirm").onclick = async () => {
  if (!mayLeave() || !window.confirm("确认保存本图当前全部条目为初始快照？之后仍可修改，但须写明原因并保留原值。")) return;
  busy = true;
  try {
    await api("/api/confirm", {sample_id: sample.sample_id, annotator: $("annotator").value});
    sample = await api(`/api/sample?sample_id=${sample.sample_id}`);
    await refreshOverview(); renderForm(); notice("本图初始快照已保存，后续修改会单独记录。");
  } catch (error) { showError(error); } finally { busy = false; }
};
$("export").onclick = async () => {
  try {
    const data = await api(`/api/export?sample_id=${sample.sample_id}`);
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: "application/json"}));
    const link = document.createElement("a"); link.href = url;
    link.download = `${sample.sample_id}_人工记录.json`; link.click(); URL.revokeObjectURL(url);
  } catch (error) { showError(error); }
};
document.querySelectorAll(".image-button").forEach(button => button.onclick = () => {
  $("zoom-image").src = `/api/image/${button.dataset.image}`;
  $("image-dialog").showModal();
});
$("close-image").onclick = () => $("image-dialog").close();
window.addEventListener("beforeunload", event => { if (dirty) { event.preventDefault(); event.returnValue = ""; } });
window.addEventListener("resize", () => requestAnimationFrame(revealActiveCriterion));
(async () => {
  try { await refreshOverview(); await loadSample(overview.smoke_ids[0]); }
  catch (error) { showError(error); }
})();
