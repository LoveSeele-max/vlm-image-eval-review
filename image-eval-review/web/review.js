"use strict";
const $ = id => document.getElementById(id);
const statuses = {pass:"符合", fail:"不符", unjudgeable:"不可判断"};
const dimensions = {subject_features:"主体特征", structural_integrity:"结构完整性", prompt_adherence:"选定提示词要求"};
const taskNames = {succeeded:"工程通过", failed:"工程失败", pending:"待运行", running:"运行中"};
let activeFilter = "all", requestNumber = 0, queryTimer;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "结果读取失败");
  return data;
}
function imageFigure(key, title) {
  const figure = element("figure");
  const button = element("button", undefined, "image-button");
  button.type = "button";
  button.setAttribute("aria-label", "放大" + title);
  const image = element("img");
  image.src = "/api/image/" + encodeURIComponent(key);
  image.alt = title;
  image.loading = "lazy";
  button.append(image);
  button.addEventListener("click", () => {
    $("large-image").src = image.src;
    $("large-image").alt = title;
    $("image-dialog").showModal();
  });
  figure.append(element("figcaption", title), button);
  return figure;
}
function evidence(label, isHuman, provenance) {
  const cell = element("div", undefined, "evidence");
  if (!label) {
    cell.append(element("strong", "未返回有效 AI 判断", "missing"),
      element("p", "此条为工程缺失，不转换为视觉“不符”或“不可判断”。", "muted"));
    return cell;
  }
  cell.append(element("span", statuses[label.status], "status " + label.status));
  for (const [key, title] of [["candidate_region","检查部位"],["candidate_observation","实际观察"],["reason","判断原因"]]) {
    const block = element("p");
    block.append(element("b", title + "："), document.createTextNode(label[key]));
    cell.append(block);
  }
  if (isHuman) cell.append(element("p", provenance, "source small"));
  return cell;
}
function criterion(row) {
  const article = element("article", undefined, "criterion" + (row.disagreement ? " different" : "") + (row.engineering_failure ? " missing-result" : ""));
  article.dataset.criterion = row.criterion_id;
  const top = element("div", undefined, "criterion-heading");
  const title = element("div");
  title.append(element("h4", row.name), element("p", row.criterion_id + " · " + dimensions[row.dimension] + " · " + (row.required ? "必需" : "辅助"), "muted small"));
  top.append(title);
  if (row.disagreement) top.append(element("span", "状态不同", "flag"));
  else if (row.engineering_failure) top.append(element("span", "工程缺失", "flag error"));
  else if (row.unjudgeable) top.append(element("span", "包含不可判断", "flag neutral"));
  article.append(top);
  const sides = element("div", undefined, "comparison");
  const author = element("div");
  author.append(element("h5", "人工初始判断"), evidence(row.human, true, row.human_provenance));
  const ai = element("div");
  ai.append(element("h5", "AI V1 判断"), evidence(row.ai, false));
  sides.append(author, ai);
  article.append(sides);
  const details = element("details", undefined, "rules");
  details.append(element("summary", "查看本项判定边界"));
  for (const [key, label] of [["pass_when","符合"],["fail_when","不符"],["unjudgeable_when","不可判断"],["boundary","补充边界"]]) {
    if (row[key]) details.append(element("p", label + "：" + row[key]));
  }
  article.append(details);
  return article;
}
function render(data) {
  const summary = data.summary;
  $("summary").replaceChildren(...[
    ["开发集样本", summary.samples], ["工程通过", summary.successful_samples],
    ["工程失败", summary.failed_samples], ["状态分歧条目", summary.disagreement],
    ["有效模型条目", summary.valid_model_items + " / " + summary.all]
  ].map(([label,value]) => {const box=element("div");box.append(element("span",label),element("strong",String(value)));return box;}));
  $("version").textContent = data.meta.reference_version + " · " + data.meta.prompt_version + " · 本次核验时间 " + data.meta.loaded_at;
  const selected = $("sample").value;
  const options = [new Option("全部开发集", "all"), ...data.samples.map(s => new Option(s.sample_id + " · " + taskNames[s.task_status], s.sample_id))];
  $("sample").replaceChildren(...options);
  $("sample").value = selected;
  for (const button of document.querySelectorAll("[data-filter]")) {
    button.querySelector("span").textContent = String(data.filter_counts[button.dataset.filter]);
    button.classList.toggle("active", button.dataset.filter === activeFilter);
    button.setAttribute("aria-pressed", String(button.dataset.filter === activeFilter));
  }
  $("result-title").textContent = activeFilter === "all" ? "逐项对照" : ({disagreement:"人机状态分歧",unjudgeable:"包含不可判断的条目",engineering_failure:"工程失败与未返回条目"})[activeFilter];
  $("result-count").textContent = data.matched_samples + " 张图片 · " + data.matched_items + " 项";
  const results = $("results");
  results.replaceChildren();
  if (!data.rows.length) {
    results.append(element("p", "当前筛选没有匹配条目。可以切换样本、维度或清空搜索。", "empty"));
    return;
  }
  for (const sample of data.samples) {
    const rows = data.rows.filter(row => row.sample_id === sample.sample_id);
    if (!rows.length) continue;
    const section = element("section", undefined, "sample-result");
    section.dataset.sample = sample.sample_id;
    const heading = element("div", undefined, "sample-heading");
    heading.append(element("h3", sample.sample_id), element("span", taskNames[sample.task_status] + (sample.cached ? " · 缓存复用" : ""), "badge"));
    section.append(heading);
    if (sample.error_code) section.append(element("p", sample.error_code + "；没有有效模型结果。已有人工初始判断保留。", "failure-notice"));
    if (sample.candidate_status) section.append(element("p", "固定规则聚合：" + ({candidate_pass:"达标",candidate_fail:"未达标",needs_review:"待复核"})[sample.candidate_status] + "。", "muted small"));
    const images = element("div", undefined, "images");
    images.append(imageFigure(sample.sample_id, "候选图"), imageFigure("REF-MAIN", "主参考图"), imageFigure("REF-SUPPLEMENT", "补充参考图"));
    section.append(images);
    const prompt = element("details", undefined, "prompt");
    prompt.append(element("summary", "原始提示词及尝试记录"), element("p", sample.prompt));
    prompt.append(element("p", sample.attempts.length ? sample.attempts.map(a=>"第 " + a.number + " 次：HTTP " + (a.http_status ?? "未知") + "，" + (a.error_code || "校验完成") + "，" + (a.elapsed_seconds ?? "未知") + " 秒").join("；") : "本批没有新增尝试，复用原成功证据。", "small"));
    section.append(prompt, ...rows.map(criterion));
    results.append(section);
  }
}
async function load() {
  const number = ++requestNumber;
  $("notice").className = "muted";
  $("notice").textContent = "正在读取已核验结果…";
  const params = new URLSearchParams({filter:activeFilter, sample:$("sample").value, dimension:$("dimension").value, q:$("query").value.trim()});
  try {
    const data = await api("/api/review?" + params);
    if (number !== requestNumber) return;
    render(data);
    $("notice").textContent = "";
  } catch (error) {
    if (number !== requestNumber) return;
    $("results").replaceChildren();
    $("result-count").textContent = "";
    $("notice").className = "failure-notice";
    $("notice").textContent = error.message;
  }
}
for (const id of ["sample","dimension"]) $(id).addEventListener("change", load);
$("query").addEventListener("input", () => {clearTimeout(queryTimer); queryTimer=setTimeout(load,180);});
for (const button of document.querySelectorAll("[data-filter]")) button.addEventListener("click", () => {activeFilter=button.dataset.filter;load();});
$("reload").addEventListener("click", async () => {
  $("reload").disabled = true;
  $("notice").textContent = "正在重新核验本机证据，请稍候…";
  try {await api("/api/reload", {method:"POST"});await load();}
  catch(error) {$("results").replaceChildren();$("notice").className="failure-notice";$("notice").textContent=error.message;}
  finally {$("reload").disabled=false;}
});
$("close-image").addEventListener("click",()=>$("image-dialog").close());
load();
