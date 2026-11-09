"use strict";
const $=id=>document.getElementById(id);
const names={pass:"符合",fail:"不符",unjudgeable:"不可判断"};
const decisions={keep:"维持初标",revise:"修订标签",unresolved:"暂不能裁定"};
const dimensions={subject_features:"主体特征",structural_integrity:"结构完整性",prompt_adherence:"选定提示词要求"};
const filters={binary:"人机二元冲突",version_change:"两版状态不同",ai_uncertain:"AI 不可判断",human_uncertain:"人工初标不可判断",engineering:"至少一版工程缺失",all:"全部条目"};
let active="binary",page=0,token="",request=0,queryTimer,dirty=new Set(),zoom=1,loaded=false;
function el(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
async function api(url,options={}){const r=await fetch(url,options);const data=await r.json();if(!r.ok)throw new Error(data.error||"操作未完成");return data;}
function discard(){return dirty.size===0||window.confirm("有未保存的复核内容。确认切换并放弃这些内容？");}
function setZoom(value){zoom=Math.max(.5,Math.min(6,value));$("large-image").style.width=(zoom*100)+"%";$("zoom-label").textContent=Math.round(zoom*100)+"%";}
function image(key,title){const f=el("figure"),b=el("button",undefined,"image-button"),img=el("img");b.type="button";b.setAttribute("aria-label","放大"+title);img.src="/api/image/"+encodeURIComponent(key);img.alt=title;img.loading="lazy";b.append(img);b.addEventListener("click",()=>{$("large-image").src=img.src;$("large-image").alt=title;setZoom(1);$("image-dialog").showModal();});f.append(el("figcaption",title),b);return f;}
function evidence(label,source){const b=el("div",undefined,"evidence");if(!label){b.append(el("strong","未返回有效判断","missing"),el("p","工程缺失，保留失败记录。","muted"));return b;}b.append(el("span",names[label.status],"status "+label.status));for(const [k,t] of [["candidate_region","检查部位"],["candidate_observation","实际观察"],["reason","判断原因"]]){const p=el("p");p.append(el("b",t+"："),document.createTextNode(label[k]));b.append(p);}if(source)b.append(el("p",source,"source small"));return b;}
function labelled(title,control){const b=el("div"),l=el("label",title);l.htmlFor=control.id;b.append(l,control);return b;}
function reviewForm(row){
 const key=row.sample_id+"/"+row.criterion_id,f=el("form",undefined,"review-form"),saved=row.review;
 f.dataset.key=key;f.dataset.resolved=String(Boolean(saved&&saved.decision!=="unresolved"));f.append(el("h5","本项复核结论"));
 if(saved){const box=el("div",undefined,"review-result "+saved.decision);box.append(el("strong",decisions[saved.decision]+(saved.final_label?" · "+names[saved.final_label.status]:"")),el("p",saved.review_reason),el("p",saved.annotator+" · "+saved.confirmed_at+" · 版本 "+saved.revision,"small"));f.append(box);}
 const fieldset=el("fieldset"),legend=el("legend","请选择处理方式");fieldset.append(legend);
 for(const value of ["keep","revise","unresolved"]){const l=el("label",undefined,"choice"),radio=el("input");radio.type="radio";radio.name="decision";radio.value=value;radio.required=true;if(saved&&saved.decision===value)radio.checked=true;l.append(radio,document.createTextNode(decisions[value]));fieldset.append(l);}
 const revised=el("div",undefined,"revision-fields");
 const status=el("select");status.id=key+"-status";status.append(new Option("请选择修订状态",""),...Object.entries(names).map(([k,v])=>new Option(v,k)));
 const region=el("input");region.id=key+"-region";region.type="text";region.maxLength=2000;
 const observation=el("textarea");observation.id=key+"-observation";observation.maxLength=2000;
 const reason=el("textarea");reason.id=key+"-reason";reason.maxLength=2000;
 const prior=saved?.decision==="revise"?saved.final_label:null;
 status.value=prior?.status||"";region.value=prior?.candidate_region||"";observation.value=prior?.candidate_observation||"";reason.value=prior?.reason||"";
 revised.append(labelled("修订后状态",status),labelled("检查部位",region),labelled("重新核对到的可见事实",observation),labelled("修订后判断原因",reason));
 const note=el("textarea");note.id=key+"-note";note.maxLength=2000;note.required=true;note.value=saved?.review_reason||"";note.placeholder="指出具体部位、可见事实和适用边界；修订时说明初标哪里需要更正。";
 const confirm=el("input");confirm.type="checkbox";confirm.required=true;
 const confirmLabel=el("label",undefined,"confirm-label");confirmLabel.append(confirm,document.createTextNode("我已核对图片与固定规则，确认保存以上结论。"));
 const button=el("button","确认保存本项","save-button");button.type="submit";button.disabled=true;
 const feedback=el("span","选择结论、填写理由并勾选确认后保存。","save-note");feedback.setAttribute("role","status");
 f.append(fieldset,revised,labelled("复核依据与处理理由（必填）",note),confirmLabel,button,feedback);
 function choice(){return f.querySelector('input[name="decision"]:checked')?.value||"";}
 function controls(){const edit=choice()==="revise";revised.hidden=!edit;for(const c of [status,region,observation,reason])c.required=edit;button.disabled=!confirm.checked||!choice()||!note.value.trim();}
 f.addEventListener("input",e=>{dirty.add(key);if(e.target!==confirm)confirm.checked=false;controls();});
 f.addEventListener("change",e=>{dirty.add(key);if(e.target!==confirm)confirm.checked=false;controls();});
 f.addEventListener("submit",async event=>{event.preventDefault();if(!f.reportValidity()||!confirm.checked)return;
   button.disabled=true;feedback.textContent="正在保存复核历史…";
   const payload={sample_id:row.sample_id,criterion_id:row.criterion_id,decision:choice(),final_label:choice()==="revise"?{criterion_id:row.criterion_id,status:status.value,candidate_region:region.value.trim(),candidate_observation:observation.value.trim(),reason:reason.value.trim()}:null,review_reason:note.value.trim(),annotator:$("annotator").value.trim(),expected_revision:saved?.revision||0,confirmed:true};
   try{const result=await api("/api/save",{method:"POST",headers:{"Content-Type":"application/json","X-Review-Token":token},body:JSON.stringify(payload)});dirty.delete(key);row.review=result.review;if(!row.history.some(h=>h.revision===result.review.revision))row.history.push(result.review);f.replaceWith(reviewForm(row));$("notice").textContent="本项已保存；已保存条目保留在本批，点击下一批更新队列。";await updateSummary();}
   catch(error){feedback.textContent=error.message;confirm.checked=false;controls();}
 });controls();
 if(row.history.length){const details=el("details",undefined,"review-history"),list=el("ol");details.append(el("summary","查看历史记录（"+row.history.length+" 次）"));for(const h of row.history){const li=el("li",h.confirmed_at+" · "+h.annotator+" · "+decisions[h.decision]+(h.final_label?" → "+names[h.final_label.status]:"")+"："+h.review_reason);list.append(li);}details.append(list);f.append(details);}
 return f;
}
function criterion(row){const card=el("article",undefined,"criterion"+(row.flags.binary?" different":"")+(row.flags.engineering?" missing-result":""));card.dataset.criterion=row.criterion_id;card.dataset.sample=row.sample_id;
 const heading=el("div",undefined,"criterion-heading"),title=el("div");title.append(el("h4",row.name),el("p",row.criterion_id+" · "+dimensions[row.dimension]+" · "+(row.required?"必需":"辅助"),"muted small"));heading.append(title);card.append(heading);
 const tags=el("div",undefined,"flag-line");for(const k of ["binary","version_change","ai_uncertain","human_uncertain","engineering"])if(row.flags[k])tags.append(el("span",filters[k],"flag"+(k==="engineering"?" error":"")));card.append(tags);
 const rules=el("details",undefined,"rules");rules.open=true;rules.append(el("summary","固定判定边界"));for(const [k,t]of [["pass_when","符合"],["fail_when","不符"],["unjudgeable_when","不可判断"],["boundary","补充边界"]])if(row[k])rules.append(el("p",t+"："+row[k]));card.append(rules);
 const sides=el("div",undefined,"comparison");for(const [key,name]of [["human","冻结人工初标"],["v1","AI V1"],["v2","AI V2"]]){const side=el("div");side.append(el("h5",name),evidence(row[key],key==="human"?row.human_provenance:null));sides.append(side);}card.append(sides,reviewForm(row));return card;}
function summary(data){const s=data.summary;$("summary").replaceChildren(...[["二元冲突待复核",s.binary_pending+" / "+s.binary],["已明确处理",s.resolved+" 项"],["暂不能裁定",s.unresolved+" 项"],["保留集图片",s.samples+" 张"]].map(([label,value])=>{const b=el("div");b.append(el("span",label),el("strong",value));return b;}));}
function params(){return new URLSearchParams({filter:active,state:$("review-state").value,sample:$("sample").value,dimension:$("dimension").value,q:$("query").value.trim(),page:String(page),page_size:"5"});}
async function updateSummary(){try{const data=await api("/api/review?"+params());summary(data);}catch(error){$("notice").textContent="本项已保存，但进度读取失败："+error.message;}}
function render(data){token=data.csrf_token;page=data.page;summary(data);$("version").textContent=data.meta.reference_version+" · V1 / V2 已核验 · "+data.meta.loaded_at;
 const selected=$("sample").value;$("sample").replaceChildren(new Option("全部保留集","all"),...data.samples.map(s=>new Option(s.sample_id,s.sample_id)));$("sample").value=selected;
 for(const b of document.querySelectorAll("[data-filter]")){b.querySelector("span").textContent=data.filter_counts[b.dataset.filter];b.classList.toggle("active",b.dataset.filter===active);b.setAttribute("aria-pressed",String(b.dataset.filter===active));}
 $("result-title").textContent=filters[active];$("result-count").textContent=data.matched_items+" 项匹配";$("page-label").textContent="第 "+(page+1)+" / "+data.pages+" 批";$("previous").disabled=page===0;$("next").disabled=page+1>=data.pages;
 const out=$("results");out.replaceChildren();if(!data.rows.length){out.append(el("p","当前筛选没有待处理条目。可查看已明确处理记录或切换其他队列。","empty"));return;}
 for(const sample of data.samples){const rows=data.rows.filter(r=>r.sample_id===sample.sample_id);if(!rows.length)continue;const s=el("section",undefined,"sample-result");s.dataset.sample=sample.sample_id;s.append(el("h3",sample.sample_id));
 const statuses=Object.entries(sample.engineering).map(([v,r])=>v+"："+(r.status==="succeeded"?"工程成功":r.error_code||"工程缺失")).join(" · ");s.append(el("p",statuses,"status-label"));const images=el("div",undefined,"images");images.append(image(sample.sample_id,"候选图"),image("REF-MAIN","主参考图"),image("REF-SUPPLEMENT","补充参考图"));s.append(images);const prompt=el("details",undefined,"prompt");prompt.append(el("summary","原始生成提示词"),el("p",sample.prompt));s.append(prompt,...rows.map(criterion));out.append(s);}
}
async function load(){const n=++request;$("notice").textContent="正在读取复核队列…";try{const data=await api("/api/review?"+params());if(n!==request)return;dirty.clear();render(data);$("notice").textContent="";loaded=true;}catch(error){if(n!==request)return;$("notice").textContent=error.message;}}
for(const id of ["sample","dimension","review-state"]){const input=$(id);input.addEventListener("focus",()=>input.dataset.previous=input.value);input.addEventListener("change",()=>{if(!discard()){input.value=input.dataset.previous||"all";return;}page=0;load();});}
$("query").addEventListener("input",()=>{clearTimeout(queryTimer);queryTimer=setTimeout(()=>{if(discard()){page=0;load();}},350);});
for(const b of document.querySelectorAll("[data-filter]"))b.addEventListener("click",()=>{if(discard()){active=b.dataset.filter;page=0;load();}});
for(const [id,delta]of [["previous",-1],["next",1]])$(id).addEventListener("click",()=>{if(discard()){const shrunk=delta===1&&$("review-state").value==="pending"&&document.querySelector('.review-form[data-resolved="true"]');if(!shrunk)page+=delta;load();window.scrollTo({top:0,behavior:"smooth"});}});
$("reload").addEventListener("click",async()=>{if(!discard())return;$("reload").disabled=true;$("notice").textContent="正在完整复验来源，可能需要约两分钟…";try{await api("/api/reload",{method:"POST",headers:{"X-Review-Token":token}});await load();}catch(error){$("notice").textContent=error.message;}finally{$("reload").disabled=false;}});
$("annotator").addEventListener("input",()=>{for(const f of document.querySelectorAll(".review-form")){const c=f.querySelector('input[type="checkbox"]');c.checked=false;f.querySelector(".save-button").disabled=true;}});
$("refresh").addEventListener("click",()=>{if(discard())load();});
window.addEventListener("beforeunload",event=>{if(dirty.size){event.preventDefault();event.returnValue="";}});
$("close-image").addEventListener("click",()=>$("image-dialog").close());$("zoom-in").addEventListener("click",()=>setZoom(zoom+.5));$("zoom-out").addEventListener("click",()=>setZoom(zoom-.5));$("zoom-fit").addEventListener("click",()=>setZoom(1));
load();
