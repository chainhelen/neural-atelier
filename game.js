import {cardFor,exampleFor,scoreComparison,aggregationExample} from './module-cards.js?v=ffn-progression-1';
import {createProgress} from './level-progress.js?v=url-navigation-1';
const M=window.NeuralPrologue,$=s=>document.querySelector(s),clone=x=>JSON.parse(JSON.stringify(x));
const fmt=x=>x===-Infinity?'−∞':Number.isFinite(x)?Number(x.toFixed(2)).toString():'?';
const vector=v=>v?'['+v.map(fmt).join(', ')+']':I18n.t("等待输入");
function comparisonValues(actual,expected,tolerance=1e-9){
 const a=actual?.map(fmt),e=expected?.map(fmt);let expanded=false;
 if(a&&e)actual.forEach((value,i)=>{
  const target=expected[i];
  if(!Number.isFinite(value)||!Number.isFinite(target)||Math.abs(value-target)<tolerance||a[i]!==e[i])return;
  let digits=6;
  while(digits<12&&value.toFixed(digits)===target.toFixed(digits))digits++;
  a[i]=value.toFixed(digits);e[i]=target.toFixed(digits);expanded=true;
 });
 const show=v=>v?'['+v.join(', ')+']':I18n.t('未得到输出');
 return {actual:show(a),expected:show(e),expanded};
}
function animationSchedule(nodes,depths,durationFor,travel,independent){
 const timings=new Map(),depthEnds=[0];
 for(const n of [...nodes].sort((a,b)=>depths[a.id]-depths[b.id])){
  const depth=depths[n.id],duration=Math.max(1,durationFor(n));
  const ready=independent?Math.max(0,...n.links.map(from=>timings.get(from)?.end||0)):depthEnds[depth-1];
  const start=ready+travel,end=start+duration;
  timings.set(n.id,{start,end});depthEnds[depth]=Math.max(depthEnds[depth]||0,end);
 }
 // Small teaching boards retain their step-by-step waves. The full model
 // follows data dependencies, without a barrier between unrelated branches.
 if(!independent)for(const n of nodes)timings.get(n.id).end=depthEnds[depths[n.id]];
 return {nodes:timings,depthEnds,end:Math.max(...depthEnds)+travel};
}
// Keep complete numeric matrices readable before the next batch clears them.
// Scalar trials retain their individual results; token outputs have their own
// word-selection and append animations.
for(const layout of Object.values(M.levels))if(layout.sequence&&!layout.tokenOutput)layout.outputArrivalHold??=450;
I18n.prepare({definitions:M.definitions,levels:M.levels,routes:M.routes,controls:M.engineControls,metrics:M.metricLabels,vocab:M.math.vocab});
for(const id of Object.keys(M.levels))I18n.prepare(M.samplesFor(id));
I18n.activate();
const progress=createProgress(M,localStorage);
function normalizeNavigation(target){
 const levelId=M.resolveLevelId(target.levelId)||progress.current(),screen=['game','levels'].includes(target.screen)?target.screen:'home';
 if(screen!=='levels')return {screen,levelId};
 const trackId=Object.hasOwn(M.routes,target.trackId)?target.trackId:M.levels[levelId].track,groups=M.routes[trackId].groups;
 const groupId=(groups?.find(g=>g.id===target.groupId)||groups?.find(g=>g.ids.includes(levelId))||groups?.[0])?.id;
 return {screen,levelId,trackId,groupId,returnScreen:target.returnScreen==='game'?'game':'home'};
}
function readNavigation(hash=window.location?.hash||''){
 try{
  const fragment=hash.replace(/^#/,'');decodeURIComponent(fragment);
  const [page,query='']=fragment.split('?'),id=M.resolveLevelId(decodeURIComponent(page));
  if(id)return normalizeNavigation({screen:'game',levelId:id});
  if(page==='home'||page==='menu'){
   const fields=Object.fromEntries(query.split('&').filter(Boolean).map(pair=>{const [key,...value]=pair.split('=');return [decodeURIComponent(key),decodeURIComponent(value.join('='))];}));
   return normalizeNavigation({screen:page==='menu'?'levels':'home',levelId:fields.lesson,trackId:fields.track,groupId:fields.group,returnScreen:fields.from});
  }
 }catch{}
 return normalizeNavigation({screen:'home'});
}
function navigationHash(target){
 if(target.screen==='game')return '#'+target.levelId;
 if(target.screen==='home')return '#home?lesson='+target.levelId;
 return '#menu?lesson='+target.levelId+'&track='+target.trackId+(target.groupId?'&group='+target.groupId:'')+'&from='+target.returnScreen;
}
function writeNavigation(target,replace=false){
 const loc=window.location,nav=window.history,hash=navigationHash(target);if(!loc||!nav||loc.hash===hash)return;
 nav[replace?'replaceState':'pushState'](null,'',loc.pathname+loc.search+hash);
}
const initialNavigation=readNavigation();let navigationState=null;
let stageId=initialNavigation.levelId,state=progress.load(stageId,{resume:true}),history=[],selected=null,pending=null,current=null,results=[],sample=0,busy=false,paused=false,pauseStarted=0,pausedDuration=0,runId=0,level=99,world=null,fault=null;
let selectedWire=null,actionNode=null,placing=null,lastFusionKind=null,menuTrack=M.levels[stageId].track,menuGroup=null;
let demoSamples=null;
const stage=()=>M.levels[stageId],samples=()=>demoSamples||(M.samplesFor?.(stageId)||M.samples),save=()=>progress.save(state);
const kindOf=id=>M.kindOf(state,id),definition=id=>M.definitionFor(state,id),weightsFor=id=>M.weights[kindOf(id)],nodeName=id=>M.nodeName(state,id);
let suppressPaletteClick=false,suppressNodeClick=false;
const shelfSizeKey='neural-atelier-shelf-size-v1';
let shelfSize=0;
function setShelfSize(value,persist=true){
 const next=Number(value);shelfSize=Number.isInteger(next)&&next>=0&&next<=2?next:0;
 $('#module-shelf').dataset.size=['small','medium','large'][shelfSize];
 $('#shelf-smaller').disabled=shelfSize===0;$('#shelf-larger').disabled=shelfSize===2;
 if(persist)try{localStorage.setItem(shelfSizeKey,String(shelfSize));}catch{}
 world?.syncShelf?.();
}
try{const saved=localStorage.getItem(shelfSizeKey);setShelfSize(saved===null?0:saved,false);}catch{setShelfSize(0,false);}
$('#shelf-smaller').onclick=()=>setShelfSize(shelfSize-1);
$('#shelf-larger').onclick=()=>setShelfSize(shelfSize+1);
const shape=dim=>stage().shapes?.[dim]||(stage().sequence?({3:I18n.t("3 个 ID"),6:'3 × 2',9:'3 × 3',12:'3 × 4',18:'3 × 6'}[dim]||dim):dim);
const moduleShape=(kind,dim)=>M.moduleShape?.(kind,dim)??shape(dim);
function attentionPackageDiagram(kind,p){
 const d=p.attentionDiagram,c=M.headDiagramCopy,sub=['','₁','₂'];
 const step=(type,title,formula)=>`<div class="head-step" data-head-step="${type}"><small>${title}</small><code>${formula}</code></div>`;
 const heads=d.numbers.map(number=>{
  const s=sub[number],score=d.masked?'S′'+s:'S'+s;
  return `<section class="attention-head" data-head="${number}"><header><b>${M.definitions[number===1?'head-a':'head-b'].name}</b><small>${c.projection} ${d.inputWidth} → ${d.headWidth}</small></header><div class="head-paths">${[['Q',c.query],['K',c.key],['V',c.value]].map(([symbol,label])=>`<div class="head-projection" data-projection="${symbol}"><small>${label}${d.cross?' ← '+(symbol==='Q'?c.target:c.memory):''}</small><code>${symbol+s}</code></div>`).join('')}<div class="head-matching"><span class="head-qk-join" aria-hidden="true">↘　↙</span>${step('score',c.score,'S'+s+' = Q'+s+'K'+s+'ᵀ / √'+d.headWidth)}${d.masked?'<span class="package-arrow" aria-hidden="true">↓</span>'+step('mask',c.mask,score+' = mask(S'+s+')'):''}<span class="package-arrow" aria-hidden="true">↓</span>${step('softmax',c.weights,'A'+s+' = softmax('+score+')')}<span class="package-arrow" aria-hidden="true">↓</span></div><div class="head-value-path" data-value-route="V${s}"><span aria-hidden="true"></span><code>V${s}</code><small>${c.bypass}</small><span aria-hidden="true">↓</span></div>${step('mix',c.mix,'H'+s+' = A'+s+' × V'+s)}</div></section>`;
 }).join('');
 const merge=d.numbers.length===2?`<div class="head-rejoin" aria-hidden="true">↘　↙</div>${step('concat',c.concat,'C = Concat(H₁, H₂) · 3 × '+d.inputWidth)}<span class="package-arrow" aria-hidden="true">↓</span>${step('output',c.output,'C × Wo')}`:'';
 return `<div class="package-diagram attention-package" data-package="${kind}" data-head-count="${d.numbers.length}"><p class="heads-input">${d.cross?c.cross:d.numbers.length===2?c.shared:c.single}</p><div class="attention-heads">${heads}</div>${merge}<p class="heads-roles">${c.roles}</p></div><p class="package-note">${p.note}</p>`;
}
function packageDiagram(kind){
 const p=M.packages?.[kind];if(!p)return '';
 if(p.attentionDiagram)return attentionPackageDiagram(kind,p);
 return `<div class="package-diagram" data-package="${kind}">${p.stages.map((names,i)=>`${i?'<span class="package-arrow" aria-hidden="true">↓</span>':''}<div class="package-stage">${names.map(name=>`<span>${name}</span>`).join('')}</div>`).join('')}</div><p class="package-note">${p.note}</p>`;
}
function encoderPrimer(){
 const g=M.encoderGuide,x=[30,92,154];
 const jobs=['exchange','process'].map((key,index)=>{
  const job=g[key],paths=x.flatMap((from,i)=>x.filter((_,j)=>key==='exchange'||i===j).map(to=>`<path d="M${from} 20 L${to} 44" class="${to===92?'emphasized':''}"/>`)).join('');
  const features=x.map((center,i)=>[0,1,2,3].map(j=>`<rect x="${center-15+j*8}" y="47" width="6" height="8" rx="1" opacity="${.3+((i+j)%4)*.2}"/>`).join('')).join('');
  return `<section class="encoder-job" data-encoder-job="${key}"><h4><span>${index+1}</span>${job.title}</h4><small>${job.tag}</small><svg viewBox="0 0 184 58" aria-hidden="true">${paths}${x.map((center,i)=>`<text x="${center}" y="12" text-anchor="middle">${g.tokens[i]}</text>`).join('')}${features}</svg><p>${job.text}</p><div class="encoder-stage-path">${g.stageNotes[index].path}</div><p class="encoder-stage-source">${g.stageNotes[index].source}</p><p class="encoder-stage-finish">${g.stageNotes[index].finish}</p></section>`;
 }).join('');
 return `<div class="encoder-primer">${jobs}<div class="encoder-pattern"><b>${g.pattern}</b><span>${g.operations.join(' → ')}</span></div><details class="encoder-formulas"><summary>${g.formulaTitle}</summary><p>${g.first}</p><code>X1 = LN(X + MHA(X))</code><p>${g.second}</p><code>Y = LN(X1 + FFN(X1))</code><p>${g.formulaNote}</p></details></div>`;
}
const playbackSpeedKey='neural-atelier-playback-speed-v1';
const defaultPlaybackSpeed=.25;
let playbackSpeed=defaultPlaybackSpeed;
function setPlaybackSpeed(value,persist=true){
 const speed=Number(value);playbackSpeed=Number.isFinite(speed)&&speed>=.25&&speed<=3?Math.round(speed*4)/4:defaultPlaybackSpeed;
 $('#playback-speed').value=String(playbackSpeed);$('#playback-speed').setAttribute('aria-valuetext',`${playbackSpeed}${I18n.t(" 倍速")}`);$('#playback-rate').textContent=`${playbackSpeed}×`;
 if(persist)try{localStorage.setItem(playbackSpeedKey,String(playbackSpeed));}catch{}
}
try{setPlaybackSpeed(localStorage.getItem(playbackSpeedKey),false);}catch{setPlaybackSpeed(defaultPlaybackSpeed,false);}
$('#playback-speed').addEventListener('input',e=>setPlaybackSpeed(e.target.value));
function animationTime(now=performance.now()){return now-pausedDuration-(paused?now-pauseStarted:0);}
function refreshPauseControl(){
 const button=$('#pause');button.disabled=!busy;button.setAttribute('aria-pressed',String(paused));
 button.innerHTML=paused?I18n.t('<span>▶</span> 继续'):I18n.t('<span>Ⅱ</span> 暂停');
}
function togglePause(){
 if(!busy)return;
 if(paused){pausedDuration+=performance.now()-pauseStarted;paused=false;}else{pauseStarted=performance.now();paused=true;}
 refreshPauseControl();
}
$('#pause').onclick=togglePause;
let fullscreenPending=false,fullscreenFallback=false;
function syncFullscreen(){
 const active=fullscreenFallback||document.fullscreenElement===$('#game'),button=$('#demo-fullscreen');
 button.dataset.fullscreenMode=active?(fullscreenFallback?'page':'native'):'off';
 $('#game').classList.toggle('demo-expanded',active);document.body.classList.toggle('demo-fullscreen',active);
 button.textContent=I18n.t(active?'退出全屏':'全屏');button.setAttribute('aria-pressed',String(active));
 button.title=I18n.t(active?'退出全屏（Esc）':'全屏查看，保留运行与调速');
 button.disabled=fullscreenPending;
}
$('#demo-fullscreen').onclick=async()=>{
 if(fullscreenPending)return;
 fullscreenPending=true;syncFullscreen();
 try{
  if(fullscreenFallback)fullscreenFallback=false;
  else if(document.fullscreenElement===$('#game'))await document.exitFullscreen();
  else if($('#game').requestFullscreen)await $('#game').requestFullscreen({navigationUI:'hide'});
  else fullscreenFallback=true;
 }catch(error){console.warn('Fullscreen request failed; using page view.',error);fullscreenFallback=document.fullscreenElement!==$('#game');}
 finally{fullscreenPending=false;syncFullscreen();}
};
document.addEventListener('fullscreenchange',syncFullscreen);
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&fullscreenFallback){fullscreenFallback=false;syncFullscreen();}});
let engineReport=null;
const levelPassed=()=>results.length===stage().sampleCount&&results.every(r=>r.pass)&&(!stage().engine||engineReport?.pass);
function closeCompletion(){if($('#level-complete').open)$('#level-complete').close();}
function renderCompletion(){
 $('#complete-level').textContent=stage().chapter+' · '+stage().title;
 $('#complete-detail').textContent=stage().generationRounds?I18n.t('完整模型已连续运行三轮。')+' '+I18n.t('新生成：')+results.map(r=>M.math.vocab[r.actual.at(-1)]).join(' '):stage().demonstration?I18n.t("完整模型已运行完毕。"):stage().engine?I18n.t("数值与性能约束全部通过。"):I18n.t("全部输入验证通过。");
 $('#complete-next').textContent=stage().next?I18n.t("下一关 →"):I18n.t("选择关卡");
 const reward=M.fusions[stageId];$('#complete-reward').hidden=!reward;
 if(reward){$('#complete-reward>small').textContent=stage().buildable?M.fusionCopy.collected:M.fusionCopy.storedLater;$('#complete-reward>strong').textContent=M.definitions[reward.kind].name;}
}
function showCompletion(){if(busy||!levelPassed())return;renderCompletion();if(!$('#level-complete').open)$('#level-complete').showModal();}
$('#complete-close').onclick=closeCompletion;
$('#complete-replay').onclick=()=>{changed();say('');$('#run').focus({preventScroll:true});};
$('#complete-next').onclick=()=>{closeCompletion();if(stage().next)switchLevel(stage().next);else levelMenu();};
$('#fusion-skip').onclick=()=>world?.skipFusion?.();
function engineControls(){
 const c=stage(),area=$('#engine-controls');if(!c.engine){area.innerHTML='';return;}
 area.innerHTML=I18n.t("<div class=\"eyebrow\">引擎参数</div><p class=\"engine-model-note\">教学成本模型 · 非设备实测</p>")+c.controls.map(k=>{const d=M.engineControls[k];return `<label>${d.label}<select data-setting="${k}">${d.values.map(([v,name])=>`<option value="${v}" ${String(state.settings[k])===String(v)?'selected':''}>${name}</option>`).join('')}</select></label>`;}).join('')+I18n.t("<p class=\"engine-model-note\">选中模块，再点芯片区域分配；也可拖动模块的芯片标记。</p>");
}
function assignDevice(id,device){if(busy||!stage().engine)return;const n=state.nodes.find(n=>n.id===id);if(!n)return;checkpoint();n.device=device;changed();say(nodeName(id)+' → '+device.toUpperCase());}
function closeReport(restoreFocus=false){$('#run-report').hidden=true;if(restoreFocus)$('#run').focus({preventScroll:true});}
function resetReport(){closeReport();$('#run-report').replaceChildren();}
function renderReport(){
 const panel=$('#run-report'),attentionValues=M.valueForKind(state,current,'softmax')||M.valueForKind(state,current,'attentionweights')||M.valueForKind(state,current,'original-weights');let html=I18n.t("<div class=\"run-report-content\" tabindex=\"0\" aria-label=\"本次运行详情\">");
 if(stage().tokenOutput){html+='<div class="word-trials">'+results.map((r,i)=>`${I18n.t("<div><small>输入：")}${samples()[i].word}</small><strong class="${r.pass?'good':'bad'}">${r.actual?M.readTokens(r.actual):I18n.t("未得到输出")}${I18n.t("</strong><span>目标：")}${M.readTokens(r.expected)}</span></div>`).join('')+I18n.t("</div><p>文字来自词表分数最高的候选。计算出错可能改词，也可能碰巧仍是同一个词。</p>");}
 if(stage().attentionReadout&&attentionValues){const a=M.math.rows(attentionValues,3),words=samples()[sample].word.split(' / ');html+=I18n.t("<h3>各位置从哪里读取内容</h3><div class=\"attention-map\">")+a.map((r,i)=>`<div><b>${words[i]}${I18n.t(" · 位置 ")}${i}</b>${r.map((v,j)=>`<span style="--weight:${Math.max(0,Math.min(1,v))}">${words[j]} ${Math.round(v*100)}%</span>`).join('')}</div>`).join('')+I18n.t("</div><p>比例是这次计算的注意力权重，不是对真实语言理解的完整解释。</p>");}
 if(current?.componentIssue)html+=`<p class="bad">${current.componentIssue.message}</p>`;
 if(!current?.componentIssue&&stage().resultNote){const r=results[results.length-1];html+=`<p>${r?.pass?I18n.t("数值匹配。"):I18n.t("数值不匹配。")}${stage().resultNote}</p>`;}
 if(current?.cloudMemory){const c=current.cloudMemory,copy=M.cloudCopyLabels;html+=`<div class="cloud-copy-report"><h3>${copy.title}</h3><strong>${c.bytes} B${c.limit!=null?' · '+copy.limit+' '+c.limit+' B':''}</strong><p>${c.copies.map(t=>t.from.toUpperCase()+' → '+t.to.toUpperCase()+': '+t.bytes+' B').join(' · ')}</p><p>${copy.note}</p>${current.error?`<p class="bad">${current.error}</p>`:''}</div>`;}
 if(stageId==='vocabulary'&&current?.values){const logits=M.valueForKind(state,current,'original-head')||M.valueForKind(state,current,'lmhead');if(logits){const scores=M.math.rows(logits,6);html+=I18n.t("<h3>词表分数 · 各位置取最高分</h3><div class=\"vocab-scores\"><div><b>位置</b>")+M.math.vocab.map(v=>'<b>'+v+'</b>').join('')+'</div>'+scores.map((r,i)=>'<div><b>'+i+'</b>'+r.map(v=>`<span class="${v===Math.max(...r)?'winner':''}">${fmt(v)}</span>`).join('')+'</div>').join('')+'</div>';}}
 if(stageId==='autoregressive'&&results.every(r=>r.pass)){const rollout=M.rollout(state,samples()[0].v);html+=I18n.t("<h3>把新词追加回去，连续解码六次</h3><div class=\"word-trials\"><div><strong>")+rollout.text+I18n.t("</strong><small>每次只采用最后一个位置的预测，再更新上下文。</small></div></div>");}
 if(stage().engine&&results.length){html+='<div class="word-trials">'+results.map((r,i)=>`${I18n.t("<div><small>提示：")}${samples()[i].word}</small><strong class="${r.pass?'good':'bad'}">${r.actual?M.decodeFeatures(r.actual):I18n.t("未输出")}${I18n.t("</strong><span>目标读出：")}${M.decodeFeatures(r.expected)}</span></div>`).join('')+'</div>';}
 if(engineReport){if(['kv-cache','capstone'].includes(stageId)){const t=engineReport.cacheTrace;html+=`${I18n.t("<h3>逐词计算检查 · 3 个位置</h3><p>")}${state.settings.cache?I18n.t("保留旧 K/V"):I18n.t("每步重算历史")}${I18n.t("：共计算 ")}${t.processed}${I18n.t(" 个位置的 K/V。两种方法的 Attention 数值一致；缓存减少重复投影，但仍然读取历史 K/V。</p>")}`;}html+=I18n.t("<p>教学模拟 · 时间包含派发、同步与数据读写；ANE 工作占比表示分配的计算量，不是硬件计数器。</p>");
 engineReport.runs.forEach((r,i)=>{html+=`${I18n.t("<h3>负载 ")}${i+1} · ${r.work.prompt}${I18n.t(" 个提示位置 / ")}${r.work.context}${I18n.t(" 个上下文位置</h3><div class=\"metric-grid\">")}`+r.checks.map(c=>{const [name,unit]=M.metricLabels[c.key];return `<div class="${c.pass?'good':'bad'}"><small>${name}</small><strong>${fmt(c.value)} <em>${unit}${I18n.t("</em></strong><span>目标 ")}${c.key==='tps'||c.key==='aneMin'?'≥':'≤'} ${c.target} ${unit}</span></div>`;}).join('')+'</div>';
 if(r.errors.length)html+='<p class="bad">'+r.errors.join('；')+'</p>';
 html+='<div class="runtime-timeline">'+r.timeline.map(t=>`<div class="lane-${t.device}" style="flex:${Math.max(1,t.duration)}" title="${t.name} · ${fmt(t.duration)} ms"><b>${t.device.toUpperCase()}</b><span>${t.name}</span><small>${fmt(t.duration)} ms</small></div>`).join('')+'</div>';});}
 if(stage().demonstration&&results.every(r=>r.pass))html+=I18n.t('<div class="campaign-complete"><span>主线完成</span><h3>从向量，到逐词生成</h3><p>所有主线关卡都可以重玩。端侧推理的独立挑战在关卡菜单的番外中。</p></div>');
 if(stageId==='capstone'&&engineReport?.pass&&results.every(r=>r.pass))html+=I18n.t("<div class=\"campaign-complete\"><span>番外完成</span><h3>从一个向量，到可运行的端侧引擎</h3><p>你已经完成数值计算、注意力、逐词生成，以及正确性、内存和速度的联合验收。所有关卡仍可重玩，继续比较不同调度方案。</p></div>");
 if(!stage().sequence)html+=I18n.t("<p>输入与输出都是特征向量，分别对应“我 / 爱 / 猫”，不是生成的新词。</p>");
 panel.innerHTML=html+I18n.t("</div><button type=\"button\" class=\"report-close\" data-close-report aria-label=\"关闭运行说明\" title=\"关闭运行说明\">×</button>");panel.hidden=false;
}
$('#engine-controls').addEventListener('change',e=>{const key=e.target.dataset.setting;if(!key||busy)return;checkpoint();state.settings[key]=M.engineControls[key].values.find(([v])=>String(v)===e.target.value)[0];changed();});
$('#device-docks').addEventListener('click',e=>{const b=e.target.closest('[data-device]');if(!b)return;if(!selected)return say(I18n.t("先选中需要分配的模块。"));assignDevice(selected,b.dataset.device);});
$('#run-report').addEventListener('click',e=>{if(e.target.closest('[data-close-report]'))closeReport(true);});
function causalPrimer(){
 const g=M.causalGuide,words=M.samplesFor('causal')[0].word.split(' / '),x=M.samplesFor('causal')[0].v;
 const scores=M.operations.score([M.operations.q([x]),M.operations.k([x])]),masked=M.operations.mask([scores]),weights=M.operations.softmax([masked]);
 const values=v=>vector(v.slice(3,6));
 return `<div class="causal-primer"><table><caption>${g.caption}</caption><thead><tr><th>${g.reference}</th>${words.map(word=>`<th scope="col">${word}</th>`).join('')}</tr></thead><tbody>${words.map((word,i)=>`<tr><th scope="row">${word}</th>${words.map((_,j)=>`<td class="${j<=i?'allowed':'blocked'}" aria-label="${j<=i?g.allowed:g.blocked}">${j<=i?'✓':'0'}</td>`).join('')}</tr>`).join('')}</tbody></table><small>${g.legend}</small><p>${g.read}</p><details><summary>${g.detail}</summary>${[[g.before,scores],[g.masked,masked],[g.after,weights]].map(([label,value])=>`<div class="causal-step"><span>${label}</span><code>${values(value)}</code></div>`).join('')}<p>${g.why}</p></details></div>`;
}
function renderStructureMap(){
 $('#structure-title').textContent=M.structureMap.copy.title;
 $('#structure-close').setAttribute('aria-label',M.structureMap.copy.close);
 if($('#structure-dialog').open)$('#structure-dialog-content').innerHTML=M.structureMap.render(stageId,true);
}
function closeHint(restoreFocus=false){if(!$('#structure-dialog').open)return;$('#structure-dialog').close();if(restoreFocus)$('#hint').focus?.({preventScroll:true});}
function showHint(){renderStructureMap();$('#structure-dialog-content').innerHTML=M.structureMap.render(stageId,true);if(!$('#structure-dialog').open)$('#structure-dialog').showModal();}
$('#hint').onclick=showHint;
$('#structure-close').onclick=()=>closeHint(true);
$('#structure-dialog').addEventListener('click',e=>{if(e.target!==$('#structure-dialog'))return;const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)closeHint(true);});
$('#structure-dialog').addEventListener('cancel',e=>{e.preventDefault();closeHint(true);});
function stageColor(c){return M.routes[c.track]?.groups?.find(group=>group.ids.includes(c.id))?.id||c.track;}
function maskedHeadsPrimer(){
 const g=M.maskedHeadsGuide,sample=M.samplesFor('masked-multihead')[0],ev=M.evaluate(M.solutions('masked-multihead'),sample.v),words=sample.word.split(' / ');
 return `<div class="heads-primer"><div class="heads-input">${g.input}</div><div class="heads-pair">${g.heads.map((name,i)=>`<div class="heads-branch" data-head="${i}"><b>${name}</b><code>Q<sub>${i?'B':'A'}</sub> · K<sub>${i?'B':'A'}</sub> · V<sub>${i?'B':'A'}</sub></code><small>${g.result}</small></div>`).join('')}</div><div class="heads-join">${g.join}<span>↓ ${g.output}</span></div><details><summary>${g.exampleTitle}</summary><table><thead><tr><th></th>${words.map(w=>`<th>${w}</th>`).join('')}</tr></thead><tbody>${g.heads.map((name,i)=>`<tr><th>${name}</th>${ev.values[i?'softmax-2':'softmax'].slice(3,6).map(value=>`<td>${Math.round(value*100)}%</td>`).join('')}</tr>`).join('')}</tbody></table><p>${g.exampleNote}</p></details><details><summary>${g.widthTitle}</summary><p>${g.width}</p></details><details><summary>${g.oldTitle}</summary><p>${g.old}</p></details></div>`;
}
function stageChrome(){
 const c=stage();document.body.classList.toggle('demo-level',!!c.demonstration);
 if(!c.floating)fullscreenFallback=false;
 $('#demo-fullscreen').hidden=!c.floating;syncFullscreen();
 if(!c.floating&&document.fullscreenElement===$('#game'))document.exitFullscreen().catch(()=>{});
 $('#scene').setAttribute('aria-label',I18n.t(c.floating?'拖动旋转 · Shift / 右键拖动平移 · 滚轮缩放':c.demonstration?'拖动转动视角 · 双击查看基础组件':'拖动空白转动视角，拖动模块重新摆放'));
 $('#demo-follow').hidden=!c.floating;$('#demo-follow-next').hidden=!c.floating;$('#demo-follow-view').hidden=!c.floating;$('#demo-pan').hidden=false;$('#demo-views').hidden=!c.demonstration;$('#demo-views').innerHTML=(c.demoViews||[]).map(view=>`<button type="button" data-demo-view="${view.key}">${view.label}</button>`).join('');
 document.body.dataset.stageColor=stageColor(c);renderStructureMap();document.body.classList.toggle('engine-level',!!c.engine);$('#device-docks').hidden=!c.engine;resetReport();$('#engine-controls').hidden=!c.engine;engineControls();$('.scene-legend').hidden=c.nodes.length===0;$('.scene-identity .chip').textContent=c.demonstration?I18n.t('完整演示'):I18n.t('自由接线');$('#scene-help').textContent=c.floating?I18n.t('拖动旋转 · Shift / 右键拖动平移 · 滚轮缩放'):c.demonstration?I18n.t('拖动转动视角 · 双击查看基础组件'):c.nodes.length?I18n.t("拖模块摆放 · 点连线删除 · 双击查看模块卡"):I18n.t("圆口出，方口进 · 拖动空白转动视角");$('#pass-count').textContent='0 / '+c.sampleCount;document.title=I18n.t("神经工坊 · ")+c.title;$('.chapter span').textContent=c.chapter;$('.chapter strong').textContent=c.title;
 $('.mission h2').textContent=c.mission;$('.mission p').textContent=c.description;$('#mission-limits').innerHTML=c.engine?Object.entries(c.limits).map(([key,value])=>{const [name,unit]=M.metricLabels[key];return `<span>${name} ${key==='tps'||key==='aneMin'?'≥':'≤'} ${value} ${unit}</span>`;}).join(''):'';
 const reason=c.rationale,details=$('#mission-details');$('#mission-reason').hidden=!reason;
 if(details.dataset.lesson!==c.id){details.toggleAttribute('open',!!c.practice||c.id==='pack-scores');details.dataset.lesson=c.id;}
 $('#mission-reason-title').textContent=reason?.title||I18n.t("为什么要这样算");
 for(const field of ['text','takeaway','scope']){
  const el=$('#mission-reason-'+field);el.textContent=reason?.[field]||'';el.hidden=!reason?.[field];
 }
 if(reason?.kind==='swiglu'){
  $('#mission-reason-example').innerHTML=`<div class="swiglu-primer"><h4>${I18n.t("SwiGLU 是哪一段？")}</h4><p>${I18n.t("前面的进阶关已经把双路门控与降维组成 SwiGLU FFN；本关再接入归一化输入和残差。")}</p><div class="swiglu-source">${I18n.t("同一份输入 Z")}</div><div class="swiglu-branches"><div><b>${I18n.t("线性投影")}</b><span>${I18n.t("内容特征")}</span></div><div class="swiglu-gate"><b>${I18n.t("门投影 → SiLU")}</b><span>${I18n.t("调节系数")}</span></div></div><div class="swiglu-merge"><span aria-hidden="true">↓</span><b>${I18n.t("逐元素相乘")}</b><span aria-hidden="true">↓</span><b>${I18n.t("降维投影")}</b></div><p>${I18n.t("同样的内容值 2，乘系数 0.3 得到 0.6，乘 1.5 得到 3。系数随输入变化，决定这个特征贡献多少。")}</p></div>`;
 }else if(reason?.kind==='encoder-layer'){
  $('#mission-reason-example').innerHTML=encoderPrimer();
 }else if(reason?.kind==='masked-heads'){
  $('#mission-reason-example').innerHTML=maskedHeadsPrimer();
 }else if(reason?.kind==='causal'){
  $('#mission-reason-example').innerHTML=causalPrimer();
 }else if(reason?.kind==='embedding-width'){
  const ex=M.embeddingWidthExample;
  $('#mission-reason-example').innerHTML=`<div class="embedding-width-example">${[[ex.beforeLabel,ex.before],[ex.afterLabel,ex.after]].map(([label,values])=>`<div data-width="${values.length}"><small>${label}</small><div class="embedding-lookup"><b>${ex.token}</b><span aria-hidden="true">→</span><div class="embedding-coordinates">${values.map((value,i)=>`<span style="--feature-color:${['#397aa2','#8862aa','#b68036','#288778'][i]}">${fmt(value)}</span>`).join('')}</div></div></div>`).join('')}</div>`;
 }else if(reason?.kind==='residual'){
  const x=reason.input,f=M.operations['relu-ffn']([x]),y=x.map((v,i)=>v+f[i]);
  $('#mission-reason-example').innerHTML=`<div class="residual-example">${[['baseline',I18n.t("原有特征 X"),x],['update',I18n.t("修改量 F"),f],['result',I18n.t("更新结果 Y"),y]].map(([role,label,values])=>`<div data-role="${role}"><span>${label}</span><code>${vector(values)}</code></div>`).join('')}<small>Y = X + F</small></div>`;
 }else $('#mission-reason-example').innerHTML=reason?.inputs?`<table><caption>${I18n.t("比较两种输入的第 3 个特征")}</caption><thead><tr><th>${I18n.t("输入")}</th><th>${I18n.t("内容 × 系数 ≈ 结果")}</th></tr></thead><tbody>${reason.inputs.map(input=>{
  const i=reason.feature,u=M.linear(input,M.weights.up)[i],g=M.linear(input,M.weights.gate)[i],s=M.silu(g);
  return `<tr><td>${vector(input)}</td><td><span data-factor="content">${fmt(u)}</span> × <span data-factor="scale">${fmt(s)}</span> ≈ <strong data-factor="result">${fmt(u*s)}</strong></td></tr>`;
 }).join('')}</tbody></table>`:'';
 if(c.packages?.length){
  const introduction=c.id==='pack-scores';
  $('#mission-reason-example').innerHTML+=c.packages.map(kind=>`<details class="package-primer" ${introduction?'open':''}><summary>${c.packageLabel||(c.id==='pack-scores'?I18n.t('已收录组件'):I18n.t('新组合组件'))} · ${M.definitions[kind].name}</summary>${packageDiagram(kind)}</details>`).join('');
 }
 $('#mission-reason-example').hidden=!$('#mission-reason-example').hasChildNodes();
 $('.route').innerHTML=`${c.routeInput?'<span>'+c.routeInput+'</span><i>→</i>':I18n.t("<span>输入 token</span><i>→</i>")}${c.id==='ffn'?I18n.t("<span>归一化输入 · 已就绪</span><i>→</i>"):''}${c.concept===c.title?'':'<span>'+c.concept+'</span><i>→</i>'}<strong>${c.outputTitle}</strong>${I18n.t('<span class="route-tail">教学缩尺 ')}${c.shape}</span>`;
}
function renderLevelMenu(){
 const route=M.routes[menuTrack],list=$('#level-list'),navigation=$('#level-stage-nav');
 $('#level-menu').dataset.stageColor=menuTrack;
 $('#level-tracks').innerHTML=Object.entries(M.routes).map(([id,r])=>`<button type="button" role="tab" id="level-track-${id}" data-track="${id}" data-stage-color="${id}" aria-selected="${id===menuTrack}" aria-controls="level-list">${r.name}<span>${I18n.t('共 {count} 关').replace('{count}',r.ids.length)}</span></button>`).join('');
 $('.level-menu-intro').textContent=route.description;
 navigation.hidden=!route.groups;navigation.setAttribute('aria-label',M.levelMenuCopy.stages);navigation.replaceChildren();
 list.setAttribute('aria-labelledby','level-track-'+menuTrack);list.classList.toggle('grouped-levels',!!route.groups);
 const card=id=>{
  const c=M.levels[id],menu=c.menu;
  return `<button type="button" class="level-choice ${id===stageId?'current':''} ${id===route.ids.at(-1)?'track-finale':''}" data-level="${id}" ${id===stageId?'aria-current="step"':''}><span class="level-choice-top"><span>${c.chapter}</span><span>${progress.completed(id)?I18n.t("✓ 已通过"):id===stageId?I18n.t("当前关卡"):''}</span></span>${c.demonstration?`<span class="level-purpose">${menu.label}</span>`:''}<strong>${c.title}</strong><span>${menu?.summary||c.summary}</span></button>`;
 };
 if(!route.groups){list.innerHTML=route.ids.map(card).join('');return;}
 const groupId=group=>'level-group-'+menuTrack+'-'+group.id;
 const lessonNotes=ids=>ids.filter(id=>M.levels[id].menu).map(id=>{const c=M.levels[id],menu=c.menu;return `<div class="lesson-note" data-lesson-note="${id}"><b>${c.chapter} · ${c.title}</b><span class="level-purpose">${menu.label}</span>${menu.connection?`<span class="lesson-connection">${menu.connection}</span>`:''}</div>`;}).join('');
 const connections=flow=>`<div class="level-dependencies" aria-label="${M.levelMenuCopy.connections}"><div class="level-dependency-sources">${flow.sources.map(source=>`<div><small>${source.label}</small><b>${source.text}</b></div>`).join('')}</div><svg viewBox="0 0 32 80" aria-hidden="true"><path d="M2 18 H14 V40 H28 M2 62 H14 V40 M23 35 L28 40 L23 45"/></svg><div class="level-dependency-target"><small>${flow.target.label}</small><b>${flow.target.text}</b></div></div>`;
 navigation.innerHTML=route.groups.map(group=>`<button type="button" data-level-group-target="${groupId(group)}" data-stage-color="${group.id}" ${group.id===menuGroup?'aria-current="step"':''}>${group.label}</button>`).join('');
 list.innerHTML=route.groups.map(group=>`<section class="level-group" id="${groupId(group)}" data-level-group="${group.id}" aria-labelledby="${groupId(group)}-title"><header class="level-group-heading"><div><h3 id="${groupId(group)}-title">${group.label}</h3><small>${group.ids.filter(id=>progress.completed(id)).length} / ${group.ids.length} ${M.levelMenuCopy.progress}</small></div></header><div class="level-group-cards">${group.ids.map(card).join('')}</div><details class="level-group-details"><summary>${M.levelMenuCopy.connections}</summary><p>${group.description}</p>${group.connections?connections(group.connections):''}${group.id==='decoder'?`<h4>${M.levelMenuCopy.decoderSteps}</h4>`:''}${lessonNotes(group.ids)}</details></section>`).join('');
}
function positionLevelMenu(focus=true){
 const group=menuGroup?document.getElementById('level-group-'+menuTrack+'-'+menuGroup):null;
 const current=group?.querySelector('.current')||(!menuGroup?$('#level-list .current'):null);
 if(focus)(current||$('#level-stage-nav [aria-current]')||$('#level-tracks [aria-selected="true"]'))?.focus?.({preventScroll:true});
 $('.level-menu-content').scrollTop=0;group?.scrollIntoView({block:'start'});
}
function navigate(target,{source='user'}={}){
 const next=normalizeNavigation(target),same=navigationState&&navigationHash(next)===navigationHash(navigationState);
 // Both popstate and hashchange can describe the same history entry.
 shell?.cancelEnter();
 if(same){
  if(source==='user'){if(next.screen==='levels')positionLevelMenu(false);else if(next.screen==='home')shell?.showScreen('home');else closeCompletion();}
  writeNavigation(next,true);return;
 }
 const changedLevel=next.levelId!==stageId,reuseMenu=navigationState?.screen==='levels'&&navigationState.trackId===next.trackId&&!changedLevel;
 if(changedLevel||next.screen!=='game')stopRun();
 if(changedLevel||navigationState?.screen==='game')save();
 closeCompletion();$('#settings').close();$('#about').close();
 if(next.screen!=='game'){
  closeHint();resetReport();if($('#module-card').open)$('#card-close').onclick?.();
  pending=selectedWire=placing=actionNode=null;world?.clearPreview?.();world?.clearPlacement?.();world?.labels();
 }
 if(changedLevel){
  world?.beforeStage();stageId=next.levelId;state=progress.load(stageId,{resume:source!=='user'});progress.select(stageId);
  demoSamples=null;history=[];selected=pending=current=fault=selectedWire=placing=actionNode=lastFusionKind=null;engineReport=null;results=[];sample=0;level=99;
  stageChrome();world?.setStage();save();refresh();say('');
 }
 navigationState=next;document.body.dataset.screen=next.screen;
 if(next.screen!=='levels')$('#level-menu').close();
 shell?.showScreen(next.screen);
 if(next.screen==='levels'){
  menuTrack=next.trackId;menuGroup=next.groupId;
  if(reuseMenu)for(const button of $('#level-stage-nav').children){if(button.dataset.levelGroupTarget==='level-group-'+menuTrack+'-'+menuGroup)button.setAttribute('aria-current','step');else button.removeAttribute('aria-current');}
  else renderLevelMenu();
  if(!$('#level-menu').open)$('#level-menu').showModal();positionLevelMenu();
 }else if(next.screen==='game')$('#run').focus?.({preventScroll:true});
 writeNavigation(next,source!=='user');
}
function levelMenu(fromHome=false){navigate({screen:'levels',levelId:stageId,returnScreen:fromHome===true?'home':'game'});}
function switchLevel(id,fromHash=false){
 id=M.resolveLevelId(id);if((busy&&!fromHash)||!id)return;
 navigate({screen:'game',levelId:id},{source:fromHash?'history':'user'});
}
stageChrome();progress.select(stageId);save();
const shell=window.AtelierShell?.init({beforeLanguageChange:save,onEnter:()=>levelMenu(true),onHome:()=>navigate({screen:'home',levelId:stageId})});
navigate(initialNavigation,{source:'boot'});
I18n.onChange(()=>{
 const reportWasOpen=!$('#run-report').hidden;
 if(current&&(stage().cloudSingle||current.componentIssue))current=M.evaluate(state,samples()[sample].v);
 stageChrome();setPlaybackSpeed(playbackSpeed,false);world?.relocalize();refresh();
 if(engineReport)engineReport=M.engineReport(state);
 if(reportWasOpen)renderReport();
 if(!busy)say('');
 if($('#level-menu').open){renderLevelMenu();positionLevelMenu(false);}
 if($('#level-complete').open)renderCompletion();
});
function restoreNavigation(){navigate(readNavigation(),{source:'history'});}
window.addEventListener?.('popstate',restoreNavigation);
window.addEventListener?.('hashchange',restoreNavigation);
const levelDialog=$('#level-menu');let levelBackdropStart=null;
function closeLevelMenu(){levelBackdropStart=null;navigate({screen:navigationState.returnScreen,levelId:stageId});if(navigationState.screen==='game')$('#levels-open').focus?.({preventScroll:true});}
function outsideLevelDialog(e){const r=levelDialog.getBoundingClientRect();return e.target===levelDialog&&(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom);}
$('#levels-open').onclick=levelMenu;$('#levels-close').onclick=closeLevelMenu;
levelDialog.addEventListener('pointerdown',e=>{levelBackdropStart=e.button===0&&e.isPrimary!==false&&outsideLevelDialog(e)?{id:e.pointerId,x:e.clientX,y:e.clientY}:null;});
levelDialog.addEventListener('pointercancel',()=>levelBackdropStart=null);
levelDialog.addEventListener('click',e=>{
 const start=levelBackdropStart;levelBackdropStart=null;
 // Only a complete click on the backdrop dismisses it, never a drag that
 // began in the list or a click on the dialog's padding or scrollbar.
 if(start&&outsideLevelDialog(e)&&(e.pointerId==null||e.pointerId===start.id)&&Math.hypot(e.clientX-start.x,e.clientY-start.y)<8){e.preventDefault();e.stopPropagation();closeLevelMenu();}
});
levelDialog.addEventListener('cancel',e=>{e.preventDefault();closeLevelMenu();});
levelDialog.addEventListener('close',()=>levelBackdropStart=null);
$('#level-stage-nav').onclick=e=>{const jump=e.target.closest('[data-level-group-target]'),group=jump&&document.getElementById(jump.dataset.levelGroupTarget);if(group)navigate({...navigationState,groupId:group.dataset.levelGroup});};
$('#level-list').onclick=e=>{const b=e.target.closest('[data-level]');if(b)switchLevel(b.dataset.level);};
$('#level-tracks').onclick=e=>{const b=e.target.closest('[data-track]');if(b&&b.dataset.track!==menuTrack)navigate({...navigationState,trackId:b.dataset.track,groupId:null});};
$('#next-level').onclick=()=>switchLevel(stage().next);
function say(text,tone=''){$('#feedback').textContent=text;$('#feedback').className='feedback '+tone;}
function checkpoint(){history.push(clone(state));if(history.length>40)history.shift();}
function changed(){demoSamples=null;closeCompletion();resetReport();engineReport=null;fault=null;current=null;selectedWire=actionNode=lastFusionKind=null;results=[];sample=0;level=99;world?.clearConnectionError?.();world?.clearPreview?.();save();world?.rebuild();refresh();}
function select(id){if(world?.focused())return;selected=id;selectedWire=null;actionNode=stage().buildable?id:null;world?.labels();renderShelf();}
function outputName(id){if(stage().sourceNames?.[id])return stage().sourceNames[id];return id==='R'?(stageId==='ffn'?I18n.t("原始输入 R"):I18n.t("输入 X")):id==='Z'?I18n.t("归一化输入 Z"):nodeName(id);}
function chooseOutput(id){if(busy||stage().demonstration)return;placing=selectedWire=actionNode=null;world?.clearConnectionError?.();world?.clearPlacement?.();world?.clearPreview?.();pending=pending===id?null:id;world?.labels();say(pending?`${I18n.t("已拿起「")}${outputName(id)}${I18n.t("」的线，点一个方形输入口。")}`:I18n.t("已放下连线。"));}
function link(from,to,port){
 if(busy||stage().demonstration)return;const next=clone(state),err=M.connect(next,from,to,port);
 if(err){
  const source=stage().sources.includes(from),out=source?(stage().sourceDim||2):definition(from)?.out,dim=to==='sink'?stage().sinkDim:definition(to)?.ins[port];
  const message=from!==to&&out&&dim&&out!==dim?`${I18n.t("尺寸不匹配：输出 ")}${source?shape(out):moduleShape(kindOf(from),out)}${I18n.t("，入口需要 ")}${to==='sink'?shape(dim):moduleShape(kindOf(to),dim)}`:err;
  say(message,'error');world?.showConnectionError?.(message,to,port);return;
 }
 checkpoint();state=next;pending=null;changed();say(I18n.t("已连接。"));
}
function chooseWire(to,index){if(busy||stage().demonstration)return;const from=to==='sink'?state.sink:state.nodes.find(n=>n.id===to)?.links[index];if(!from)return;pending=placing=actionNode=null;selectedWire={from,to,index};world?.clearConnectionError?.();world?.clearPreview?.();world?.clearPlacement?.();world?.labels();}
function deleteWire(){if(busy||!selectedWire)return;const {to,index}=selectedWire;checkpoint();M.disconnect(state,to,index);changed();say(I18n.t("已删除连线。"));}
function chooseInput(to,port){if(busy||stage().demonstration)return;if(pending)return link(pending,to,port);const old=to==='sink'?state.sink:state.nodes.find(n=>n.id===to).links[port];if(!old)return say(I18n.t("先点一个圆形输出口，再接到这里。"));chooseWire(to,port);}
const shelfUnlocked=kind=>M.fusions[stageId]?.kind!==kind||stage().nodes.includes(kind)||progress.completed(stageId);
const shelfInventory=()=>stage().buildable?M.inventory(stage()):M.fusions[stageId]&&progress.completed(stageId)?{[M.fusions[stageId].kind]:1}:{};
function renderShelf(){
 hideShelfTooltip();
 const shelf=$('#module-shelf'),available=shelfInventory(),collection=!stage().buildable;shelf.hidden=!stage().buildable&&!Object.keys(available).length;shelf.dataset.mode=collection?'collection':'inventory';
 if(shelf.hidden)return;
 const cap=M.moduleLimit(stage());$('#module-count').textContent=collection?M.fusionCopy.collectedShort:state.nodes.length+' / '+cap;$('#module-count').setAttribute('aria-label',collection?M.fusionCopy.collected:I18n.t('台面数量 / 上限'));
 shelf.querySelector('summary').title=shelf.hasAttribute('open')?I18n.t("收起组件架"):I18n.t("展开组件架");
 const palette=$('#module-palette'),earlier=palette.querySelector('[data-earlier-parts]'),earlierOpen=earlier?.dataset.shelfLevel===stageId&&earlier.hasAttribute('open'),scrollTop=palette.scrollTop;
 const kinds=[...new Set((collection?Object.keys(available):state.palette||Object.keys(available)).filter(shelfUnlocked).map(kind=>M.shelfKind(stage(),kind)))];
 const renderStock=kind=>{
  const d=M.definitions[kind],name=M.moduleName(stage(),kind),remaining=collection?1:M.stock(state,kind),disabled=busy||collection||!M.canAdd(state,kind),w=M.weights[kind];
  const symbols={layernorm:'LN',position:'sin',concat:'∥',encoder:'Enc',decoder:'Dec',activation:'∿',add:'+',residual:'+',multiply:'×',norm:'RMS',rope:'↻',score:'QKᵀ',mask:'◩',softmax:'σ',mix:'AV',attention:'Attn',ffn:'FFN'};
  const visual=w?`<span class="stock-matrix" aria-hidden="true" style="grid-template-columns:repeat(${w[0].length},1fr)">`+w.flat().slice(0,8).map(v=>`<i class="${v<0?'negative':''}"></i>`).join('')+'</span>':`<span class="stock-symbol" aria-hidden="true" style="--symbol-chars:${[...(symbols[d.type]||'◇')].length}">${symbols[d.type]||'◇'}</span>`;
  const color=M.extraColors?.[kind]?'#'+M.extraColors[kind].toString(16).padStart(6,'0'):({project:'#5eaa9c',down:'#90a79a',add:'#c3a26f',gate:'#a894bf'})[kind]||'#8aa59a';
  const code=({'silu-ffn':'SiLU','swiglu-ffn':'SwiGLU','swiglu-residual':'+ FFN','original-transformer':'Enc/Dec','modern-block':'Block','single-attention':'1H','two-head-attention':'2H','single-cross-attention':'Cross',relu:'ReLU',layernorm2:'LN','ln-attn':'LN','ln-ffn':'LN','ln-cross':'LN',concat:'A ∥ B',position:'PE',project:'W',gate:'Wg',down:'Wd',output:'Wo',q:'Q',k:'K',v:'V',ropeq:'Q ↻',ropek:'K ↻',residual1:'+ A',residual2:'+ F'})[kind]||({activation:'SiLU',add:'+',residual:'+',multiply:'×',norm:'RMS',score:'QKᵀ',mask:'◩',softmax:'σ',mix:'AV',attention:'Attn',ffn:'FFN'})[d.type]||kind;
  const origin=M.packageOrigin(kind),source=origin?I18n.t('来源：')+origin.chapter:'';
  const guidance=`${collection?M.fusionCopy.useLater:I18n.t("可用数量")+' '+remaining+' / '+available[kind]} · ${I18n.t("双击查看说明")}`;
  const tooltip=source?source+'\n'+guidance:guidance;
  const description=`${name} · ${d.ins.map(dim=>moduleShape(kind,dim)).join(' + ')} → ${moduleShape(kind,d.out)} · ${source?source+' · ':''}${guidance}`;
  return `<div class="module-stock ${lastFusionKind===kind?'recently-crafted':''}" role="button" tabindex="${busy?'-1':'0'}" aria-disabled="${busy}" data-add-disabled="${disabled}" aria-label="${description}" data-tooltip="${tooltip}" data-add-module="${kind}" style="--stock-color:${color}">${visual}<span class="stock-description"><b class="stock-name">${name}</b><small>${d.ins.map(dim=>moduleShape(kind,dim)).join(' + ')} → ${moduleShape(kind,d.out)}</small></span><b class="stock-code" aria-hidden="true">${code}</b><span class="stock-count" aria-hidden="true">${remaining}</span></div>`;
 };
 const grouping=stage().shelf;
 if(grouping){
  const current=kinds.filter(kind=>{if(grouping.currentKinds)return grouping.currentKinds.includes(kind);const d=M.definitions[kind];return d.out===grouping.featureWidth&&d.ins.every(dim=>dim===grouping.featureWidth);});
  const ordered=[...grouping.order.filter(kind=>current.includes(kind)),...current.filter(kind=>!grouping.order.includes(kind))],older=kinds.filter(kind=>!current.includes(kind));
  const group=(items,title,type,className='shelf-current')=>items.length?`<section class="${className}" data-shelf-group="${type}" aria-label="${title}"><h4>${title}</h4>${items.map(renderStock).join('')}</section>`:'';
  const isPackage=kind=>grouping.packageKinds?grouping.packageKinds.includes(kind):!!M.packages[kind];
  const currentHTML=grouping.packages?group(ordered.filter(isPackage),grouping.packages,'packages')+group(ordered.filter(kind=>!isPackage(kind)),grouping.basics,'basics'):group(ordered,grouping.current,'current');
  const olderHTML=grouping.packages?group(older.filter(isPackage),grouping.packages,'packages','shelf-earlier-group')+group(older.filter(kind=>!isPackage(kind)),grouping.basics,'basics','shelf-earlier-group'):older.map(renderStock).join('');
  palette.innerHTML=`${currentHTML}${older.length?`<details class="shelf-earlier" data-earlier-parts data-shelf-level="${stageId}" ${earlierOpen?'open':''}><summary>${grouping.earlier}<span>${older.length}</span></summary><p>${grouping.note}</p><div class="shelf-earlier-list">${olderHTML}</div></details>`:''}`;
 }else palette.innerHTML=kinds.map(renderStock).join('');
 palette.scrollTop=scrollTop;
 world?.syncShelf?.();
}
function chooseModule(kind){if(busy||!stage().buildable||!shelfUnlocked(kind)||!M.canAdd(state,kind)||world?.focused())return;pending=selectedWire=actionNode=null;placing=kind;world?.clearConnectionError?.();world?.clearPreview?.();world?.labels();say(I18n.t("拖到台面松开；也可以点空位放下。"));}
function placeModule(x,z){if(!placing||busy)return false;const next=clone(state);if(!M.addNode(next,placing,x,z)){const message=I18n.t("这里放不下，换一处空位。");say(message,'error');world?.showPlacementError?.(message,x,z);return false;}checkpoint();selected=next.nodes.at(-1).id;state=next;placing=null;world?.clearPlacement?.();changed();say(I18n.t("组件已放置。"));return true;}
function hideShelfTooltip(){
 const tip=$('#shelf-tooltip');tip.hidden=true;
 document.querySelector('[aria-describedby="shelf-tooltip"]')?.removeAttribute('aria-describedby');
}
function showShelfTooltip(button){
 if(busy||world?.focused()||!button?.dataset.tooltip)return;
 const tip=$('#shelf-tooltip');if(!tip.hidden&&button.getAttribute('aria-describedby')==='shelf-tooltip')return;
 hideShelfTooltip();tip.textContent=button.dataset.tooltip;button.setAttribute('aria-describedby','shelf-tooltip');tip.hidden=false;
 const r=button.getBoundingClientRect(),t=tip.getBoundingClientRect(),vw=window.innerWidth||document.documentElement.clientWidth||1280,vh=window.innerHeight||document.documentElement.clientHeight||800;
 tip.style.left=Math.max(10,Math.min(r.right+9,vw-t.width-10))+'px';tip.style.top=Math.max(10,Math.min(r.top+(r.height-t.height)/2,vh-t.height-10))+'px';
}
$('#module-palette').addEventListener('pointerover',e=>{if(e.pointerType==='touch'||e.buttons)return;showShelfTooltip(e.target.closest('[data-add-module]'));});
$('#module-palette').addEventListener('pointerout',e=>{const button=e.target.closest('[data-add-module]');if(button&&!button.contains(e.relatedTarget))hideShelfTooltip();});
$('#module-palette').addEventListener('focusin',e=>showShelfTooltip(e.target.closest('[data-add-module]')));
$('#module-palette').addEventListener('focusout',hideShelfTooltip);
document.addEventListener('pointerdown',hideShelfTooltip,true);
document.addEventListener('scroll',hideShelfTooltip,true);
document.addEventListener('keydown',e=>{if(e.key==='Escape')hideShelfTooltip();});
window.addEventListener?.('resize',hideShelfTooltip);
$('#module-palette').onclick=e=>{if(suppressPaletteClick||world?.focused())return;const b=e.target.closest('[data-add-module]');if(b&&b.dataset.addDisabled!=='true')chooseModule(b.dataset.addModule);};
$('#module-palette').ondblclick=e=>{const b=e.target.closest('[data-add-module]');if(!b||busy||suppressPaletteClick)return;e.preventDefault();world?.focusShelf?.(b.dataset.addModule);};
$('#module-palette').addEventListener('keydown',e=>{if(e.key!=='Enter'&&e.key!==' ')return;const b=e.target.closest('[data-add-module]');if(!b||busy)return;e.preventDefault();if(e.key==='Enter')world?.focusShelf?.(b.dataset.addModule);else if(b.dataset.addDisabled!=='true')chooseModule(b.dataset.addModule);});
function removeModule(id=selected){if(busy)return;const next=clone(state);if(!M.removeNode(next,id))return;checkpoint();state=next;if(selected===id)selected=null;pending=placing=actionNode=null;world?.clearPlacement?.();changed();say(I18n.t("组件已收回，相关连线已移除。"));}
function renderGeneration(){
 const area=$('#demo-generation');area.hidden=!stage().generationRounds;if(area.hidden)return;
 const seed=M.samplesFor(stageId)[0].v,generated=results.filter(r=>r.pass).map(r=>r.actual.at(-1));
 $('#demo-generation-count').textContent=I18n.t(busy&&generated.length<=sample?'正在生成':'已生成')+' '+(busy&&generated.length<=sample?sample+1:generated.length)+' / '+stage().generationRounds;
 $('#demo-transcript').replaceChildren(...[...seed,...generated].map((id,i)=>{const el=document.createElement('span');el.className=i<seed.length?'seed-token':'generated-token';el.textContent=M.math.vocab[id];return el;}));
 $('#demo-transcript').setAttribute('aria-label',I18n.t('完整文字'));
}
function refresh(){renderGeneration();refreshPauseControl();$('#progress-reset').disabled=busy;$('#progress-reset-apply').disabled=busy;$('#progress-restart').disabled=busy;$('#progress-restart-undo').disabled=busy;$('#progress-restart-undo').hidden=!progress.canUndoRestart();document.body.classList.toggle('network-running',busy);renderShelf();const t=samples()[sample];const won=levelPassed();$('.route strong').textContent=stage().outputTitle+(won?I18n.t(" · 已通过"):'');$('#next-level').hidden=!(won||progress.completed(stageId))||!stage().next;$('#engine-controls').querySelectorAll('input,select,button').forEach(e=>e.disabled=busy);$('#device-docks').querySelectorAll('button').forEach(e=>e.disabled=busy);$('#sample-name').textContent=t.word;$('#sample-value').textContent=stage().generationRounds?I18n.t('目标输入 · 最近三个词'):stage().sequence?(stage().sourceDim===3?'Token ID '+vector(t.v):I18n.t(stage().sourceDim===12?"3 个位置 × 4 个特征":"3 个位置 × 2 个特征")):vector(t.v);$('#pass-count').textContent=results.filter(r=>r.pass).length+' / '+stage().sampleCount;$('#test-results').innerHTML=results.map((r,i)=>`<div class="test-row ${r.pass?'pass':'fail'}"><span>${r.pass?'✓':'×'} ${samples()[i].word}</span><span>${stage().tokenOutput&&r.actual?M.readTokens(r.actual):stage().sequence?(r.actual?I18n.t("矩阵 ")+shape(r.actual.length):I18n.t("等待输入")):(r.pass?vector(r.actual):comparisonValues(r.actual,r.expected).actual)}</span></div>`).join('');$('#run').disabled=!world;$('#run').innerHTML=busy?I18n.t("<span>■</span> 停止"):I18n.t("<span>▶</span> 试跑我的网络");$('#undo').disabled=busy||!history.length;$('#reset').disabled=busy;$('#reset').hidden=!!stage().demonstration;$('#undo').hidden=!!stage().demonstration;$('#restore-layout').hidden=!progress.canRestore(stageId,state);$('#restore-layout').disabled=busy;}
function stopRun(){
 if(!busy&&!paused)return;
 busy=paused=false;runId++;world?.clearPackets();world?.labels();refresh();
}
async function run(){
 closeHint();closeCompletion();world?.clearConnectionError?.();
 if(busy){stopRun();say(I18n.t("已停止。"));return;}
 busy=true;paused=false;pausedDuration=0;pending=selectedWire=placing=actionNode=lastFusionKind=null;world.clearPreview?.();world.clearPlacement?.();fault=null;engineReport=null;results=[];demoSamples=stage().generationRounds?M.samplesFor(stageId).map(t=>M.demo.sampleFor(t.v)):null;sample=0;current=null;resetReport();if(demoSamples)world.relocalize();const id=++runId;
 for(let i=0;i<stage().sampleCount;i++){
  sample=i;current=M.evaluate(state,samples()[i].v);level=0;
  const expected=M.reference(samples()[i].v,stageId),actual=current.output;
  refresh();world.labels();say(`${I18n.t("试跑：「")}${samples()[i].word}」`);
  await world.animate(current,650,id);
  if(id!==runId||!busy)return;
  const pass=!current.error&&!!actual&&actual.length===expected.length&&actual.every((x,j)=>Number.isFinite(x)&&Math.abs(x-expected[j])<(stage().engine?(stage().limits?.error||.06):1e-9));results.push({pass,actual,expected});
  if(!pass){fault={id:'sink',actual,expected};level=99;busy=paused=false;if(stage().engine)engineReport=M.engineReport(state);world.labels();refresh();renderReport();say(`「${samples()[i].word}${I18n.t("」未通过。")}`,'error');return;}
  level=99;world.labels();refresh();
  if(stage().generationRounds&&i+1<stage().sampleCount){
   demoSamples[i+1]=M.demo.sampleFor(M.demo.nextInput(samples()[i].v,actual));
   await world.feedback?.(actual.at(-1),id);
   if(id!==runId||!busy)return;
  }
 }
 if(stage().floating){await world.finishDemo?.(id);if(id!==runId||!busy)return;}
 if(stage().engine)engineReport=M.engineReport(state);
 if(levelPassed()){
  progress.complete(stageId,state);const reward=M.fusions[stageId];
  if(reward){
   lastFusionKind=reward.kind;refresh();
   try{await world.fuse?.(reward,id);}catch(error){console.error('Component celebration failed',error);world.clearPackets();}
   if(id!==runId||!busy)return;
  }
 }
 busy=paused=false;world.labels();refresh();renderReport();say(stage().engine?(engineReport.pass?I18n.t("数值与性能约束全部通过。"):I18n.t("数值通过；性能验收未通过，查看运行报告。")):`${I18n.t("通过：")}${stage().sampleCount} / ${stage().sampleCount}。`,stage().engine&&!engineReport.pass?'error':'success');showCompletion();
}
$('#run').onclick=run;$('#undo').onclick=()=>{state=history.pop();pending=null;changed();say(I18n.t("已撤销。"));};$('#reset').onclick=()=>{checkpoint();state=M.initial(stageId);selected=pending=placing=null;world?.clearPlacement?.();changed();say(I18n.t("已恢复本关起点。"));};
let progressResetMessage='',progressResetMode='badges';
function progressResetStatus(message){progressResetMessage=message;$('#progress-reset-status').textContent=I18n.t(message);$('#progress-reset-status').hidden=false;}
function renderProgressReset(){
 const all=progressResetMode==='all';
 $('#progress-reset-description').textContent=I18n.t(all?'所有关卡回到初始状态，并清除通关标记？教学预设组件会保留，自己添加的组件和连线会重置。语言、音乐和速度不变；之后可撤销这次重置。':'把所有关卡标为未通过？已搭好的组件和连线会保留，通关标记与完成计数会清零。此操作无法撤销。');
 $('#progress-reset-apply').textContent=I18n.t(all?'确认重新开始':'确认清除');
}
I18n.onChange(()=>{renderProgressReset();if(progressResetMessage)$('#progress-reset-status').textContent=I18n.t(progressResetMessage);});
function clearProgressResetUI(){
 $('#progress-reset-confirm').hidden=true;for(const id of ['progress-reset','progress-restart'])$('#'+id).setAttribute('aria-expanded','false');$('#progress-reset-status').hidden=true;
}
globalThis.AtelierProgressResetUI={clear:clearProgressResetUI};
const resetTrigger=()=>$(progressResetMode==='all'?'#progress-restart':'#progress-reset');
function confirmProgressReset(mode){if(busy)return;clearProgressResetUI();progressResetMode=mode;renderProgressReset();$('#progress-reset-confirm').hidden=false;resetTrigger().setAttribute('aria-expanded','true');$('#progress-reset-cancel').focus();}
$('#progress-reset').onclick=()=>confirmProgressReset('badges');$('#progress-restart').onclick=()=>confirmProgressReset('all');
$('#progress-reset-cancel').onclick=()=>{clearProgressResetUI();resetTrigger().focus();};
function reloadResetBoard(resume=false){
 world?.beforeStage();state=progress.load(stageId,{resume});history=[];selected=pending=placing=null;
 changed();world?.setStage();
}
$('#progress-reset-apply').onclick=()=>{
 if(busy)return;
 if(progressResetMode==='all'){
  if(!progress.save(state)||!progress.restartAll()){progressResetStatus('保存失败，游戏进度未更改。请检查浏览器存储后重试。');return;}
  reloadResetBoard();renderLevelMenu();clearProgressResetUI();say('');progressResetStatus('全部关卡已恢复初始状态，通关标记已清除。语言、音乐和速度已保留。');resetTrigger().focus();return;
 }
 if(!progress.resetCompletions()){progressResetStatus('保存失败，通关记录未清除。请检查浏览器存储后重试。');return;}
 changed();renderLevelMenu();clearProgressResetUI();say('');
 progressResetStatus('所有关卡已标为未通过，棋盘和设置已保留。');$('#progress-reset').focus();
};
$('#progress-restart-undo').onclick=()=>{
 if(busy)return;
 if(!progress.undoRestart()){progressResetStatus('保存失败，游戏进度未更改。请检查浏览器存储后重试。');return;}
 reloadResetBoard(true);renderLevelMenu();clearProgressResetUI();say('');progressResetStatus('已恢复重置前的棋盘和通关记录。');$('#progress-restart').focus();
};
$('#restore-layout').onclick=()=>{if(busy)return;const restored=progress.restore(stageId);if(!restored)return;checkpoint();state=restored;selected=pending=placing=null;world?.clearPlacement?.();changed();say(I18n.t("已恢复上次方案。"));};
$('#about-open').onclick=()=>$('#about').showModal();$('#about-close').onclick=()=>$('#about').close();$('#about').addEventListener('click',e=>{if(e.target===$('#about')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});
try{
 const T=await import('./vendor/three.module.js');
 const host=$('#scene'),overlay=$('#labels'),scene=new T.Scene(),orthoCamera=new T.OrthographicCamera(-11.5,11.5,7,-7,.1,100),spaceCamera=new T.PerspectiveCamera(32,1,.1,2000),renderer=new T.WebGLRenderer({antialias:true,alpha:true});
 let camera=orthoCamera,cameraPan=false,followData=true,followView='medium',followZoom=null,followedPacket=null,followSeed=null;
 const followProfiles={wide:{padding:34,min:3,yaw:.3,pitch:.9},medium:{padding:21,min:1.5,yaw:.15,pitch:.65},close:{padding:12,min:.9,yaw:.3,pitch:.42}};
 renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor(0,0);renderer.shadowMap.enabled=true;renderer.shadowMap.type=T.PCFSoftShadowMap;host.appendChild(renderer.domElement);
 scene.add(new T.HemisphereLight(0xfffdf2,0xa0b4a6,1.9));const sun=new T.DirectionalLight(0xfffcf2,1.9);sun.position.set(-4,14,8);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);Object.assign(sun.shadow.camera,{left:-15,right:15,top:12,bottom:-12});sun.shadow.bias=-.001;scene.add(sun);
 const colors={...M.extraColors,project:0x5eaa9c,gate:0xa894bf,silu:0xc6a574,multiply:0x669b91,down:0x90a79a,add:0xc3a26f,R:0x6e9fb5,Z:0x90b2bb,sink:0x438f7e};
 const nodeColor=id=>colors[kindOf(id)]||0x91afa4;
 const material=(c,extra={})=>new T.MeshStandardMaterial({color:c,roughness:.62,metalness:.07,...extra});
 function box(parent,w,h,d,x,y,z,c,extra={}){const b=new T.Mesh(new T.BoxGeometry(w,h,d),material(c,extra));b.position.set(x,y,z);b.castShadow=true;b.receiveShadow=true;parent.add(b);return b;}
 const {createSceneArt}=await import('./scene-art.js?v=default-detail-1');
 function scenePalette(){
  if(typeof getComputedStyle!=='function')return {};
  const css=getComputedStyle(document.body),read=name=>css.getPropertyValue(name).trim();
  return {surface:read('--group-bg'),edge:read('--group-border'),accent:read('--group-accent')};
 }
 const art=createSceneArt(T,scene,colors,{anisotropy:renderer.capabilities.getMaxAnisotropy()});art.surroundings(samples(),stage(),scenePalette());
 let fusionEffect=null,tokenPreview=null,arrivalPreview=null;
 const assembly=new T.Group(),wires=new T.Group();scene.add(assembly,wires);const neuronGeo=new T.SphereGeometry(.115,16,12);let elements=[],curves=[],packets=[],drag=null,preview=null,yaw=.05,pitch=.8,zoom=stage().zoom,lastDepth=-1,playbackTimeline=null;
 const cameraTarget=new T.Vector3();let focusId=null,shelfFocus=null,shelfModel=null,overview=null,cameraFrame=0,transitioning=false,lastTap=null;
 let followFlight=null,followFlightFrame=0;
 const ray=new T.Raycaster(),cursor=new T.Vector2(),plane=new T.Plane(new T.Vector3(0,1,0),0);
 function dispose(group){for(const o of [...group.children]){group.remove(o);o.traverse(c=>{if(c.geometry&&!c.geometry.userData?.shared)c.geometry.dispose();c.material?.map?.dispose();c.material?.dispose?.();});}}
 const sourceY=id=>stage().sourceHeights?.[id]||0;
 function point(id,kind='out',index=0){if(id==='R'||id==='Z')return new T.Vector3(stage().sourceX+(stage().sourceDim===12?1.65:stage().sequence?1.15:.85),sourceY(id)+.65,stage().dualSource?art.sourceOffset(stage(),id):stageId==='ffn'?(id==='Z'?-.75:.75):0);if(id==='sink')return new T.Vector3(stage().sinkX-(stage().sequence?1.5:stage().sinkDim===4?1.2:.8),(stage().sinkY||0)+.65,0);const n=state.nodes.find(n=>n.id===id),d=definition(id);return new T.Vector3(n.x+(kind==='in'?-1.12:1.12),(n.y||0)+.68,n.z+(kind==='in'&&d.ins.length===2?(index===0?-.55:.55):0));}
 let connectionNotice=null,connectionTimer=null;
 function clearConnectionError(){
  if(connectionTimer!==null)clearTimeout(connectionTimer);connectionTimer=null;
  connectionNotice?.el.remove();connectionNotice=null;
  overlay.querySelectorAll('.port.rejected').forEach(el=>el.classList.remove('rejected'));
 }
 function showSceneError(heading,message,anchor){
  clearConnectionError();
  const el=document.createElement('div');el.className='connection-toast';el.setAttribute('role','status');el.setAttribute('aria-live','polite');el.setAttribute('aria-atomic','true');
  const title=document.createElement('strong'),text=document.createElement('span');title.textContent=heading;text.textContent=message;el.append(title,text);overlay.appendChild(el);
  const notice={el,...anchor};el.dataset.errorKind=anchor.position?'placement':'connection';connectionNotice=notice;render();
  connectionTimer=setTimeout(()=>{if(connectionNotice===notice)clearConnectionError();},3600);
 }
 function showConnectionError(message,to,port){
  if(to!=='sink'&&!state.nodes.some(n=>n.id===to)){clearConnectionError();return;}
  showSceneError(I18n.t('连线未接上'),message,{to,port});
 }
 function showPlacementError(message,x,z){
  if(!Number.isFinite(x)||!Number.isFinite(z))return;
  showSceneError(I18n.t('组件未放置'),message,{position:new T.Vector3(x,0,z)});
 }
 function syncConnectionError(w,h){
  if(!connectionNotice)return;const {el,to,port,position}=connectionNotice;
  const target=position?null:overlay.querySelector(`[data-input="${to}:${port}"]`);target?.classList.add('rejected');
  const [x,y]=project(position||point(to,'in',port)),width=el.offsetWidth,height=el.offsetHeight,gap=(target?.offsetHeight||30)/2+12;
  const left=Math.max(12,Math.min(w-width-12,x-width/2)),above=y-height-gap>=12;
  el.style.left=left+'px';el.style.top=Math.max(12,Math.min(h-height-12,above?y-height-gap:y+gap))+'px';
  el.dataset.side=above?'above':'below';el.style.setProperty('--tip-x',Math.max(16,Math.min(width-16,x-left))+'px');
 }
 const curve=(a,b)=>new T.CubicBezierCurve3(a,new T.Vector3(a.x+.8,a.y+.4,a.z),new T.Vector3(b.x-.8,b.y+.4,b.z),b);
 const guide=document.createElementNS('http://www.w3.org/2000/svg','svg'),guidePath=document.createElementNS('http://www.w3.org/2000/svg','path');
 guide.setAttribute('class','connection-guide');guide.setAttribute('aria-hidden','true');guide.setAttribute('hidden','');guidePath.setAttribute('vector-effect','non-scaling-stroke');guide.appendChild(guidePath);
 const placementGuide=document.createElementNS('http://www.w3.org/2000/svg','svg'),placementPath=document.createElementNS('http://www.w3.org/2000/svg','path'),placementOutline=document.createElementNS('http://www.w3.org/2000/svg','path');
 placementGuide.setAttribute('class','placement-guide');placementGuide.setAttribute('aria-hidden','true');placementGuide.setAttribute('hidden','');placementPath.setAttribute('class','placement-guide-path');placementOutline.setAttribute('class','placement-guide-outline');placementGuide.append(placementPath,placementOutline);
 const wireTargets=document.createElementNS('http://www.w3.org/2000/svg','svg');wireTargets.setAttribute('class','wire-targets');let wireTargetKey='',wirePaths=[],placementMesh=null;
 const colorHex=color=>'#'+color.toString(16).padStart(6,'0');
 const flowIdentity=()=>stage().sequence?I18n.t("整段输入"):samples()[sample].word;
 const packetIdentity=from=>stage().demonstration?(M.demo.sourceNodes.includes(from)?I18n.t('源序列'):I18n.t('目标前缀')):stage().dualSource?(['Z','source-embed','original-encoder','original-k','original-v'].includes(kindOf(from))?I18n.t('源序列'):(stage().targetFlowLabel||I18n.t('目标前缀'))):flowIdentity();
 function rowEndpoint(kind){return new T.Vector3(kind==='source'?stage().sourceX+.34:stage().sinkX+.18,kind==='source'?.95:.98,art.rowZ(sample));}
 function wire(from,to,index){
  const a=point(from),b=point(to,'in',index);
  // Keep the two long cross-attention inputs in the gutters between stages.
  const crossInput=stage().demonstration&&M.demo.sections[to]==='cross'&&[M.demo.encoder,M.demo.masked].includes(from);
  const lane=from===M.demo?.encoder?6:8;
  const cv=crossInput?new T.CatmullRomCurve3([a,new T.Vector3(a.x+2,a.y+.27,a.z),new T.Vector3(a.x+2,a.y+.27,lane),new T.Vector3(b.x-3,b.y+.27,lane),new T.Vector3(b.x-3,b.y+.12,b.z),b],false,'centripetal',.25):curve(a,b),color=nodeColor(from),dim=Math.min(4,(from==='R'||from==='Z')?(stage().sourceDim||2):definition(from).out),visuals=[];
  for(let j=0;j<dim;j++){const offset=(j-(dim-1)/2)*.035,points=cv.getPoints(32).map(p=>p.add(new T.Vector3(0,0,offset))),lane=new T.CatmullRomCurve3(points),mesh=new T.Mesh(new T.TubeGeometry(lane,32,.012,6,false),material(color));wires.add(mesh);visuals.push(mesh);}
  const arrow=new T.Mesh(new T.ConeGeometry(.08,.19,8),material(color));arrow.position.copy(cv.getPoint(.6));arrow.quaternion.setFromUnitVectors(new T.Vector3(0,1,0),cv.getTangent(.6).normalize());wires.add(arrow);visuals.push(arrow);visuals.forEach(mesh=>mesh.userData.wire={from,to});curves.push({from,to,index,curve:cv,visuals});
 }
 function neuralModule(n){art.module(assembly,n,definition(n.id),M.weights);}
 function rebuild(){if(focusId)leaveFocus();clearPreview();dispose(assembly);dispose(wires);art.resetUnits();curves=[];state.nodes.forEach(neuralModule);state.nodes.forEach(n=>n.links.forEach((id,i)=>{if(id)wire(id,n.id,i);}));if(state.sink)wire(state.sink,'sink',0);labels();}
 function addLabel(tag,klass,html,attrs,pos){const el=document.createElement(tag);el.className=klass;const owner=attrs['data-owner']||attrs['data-node']||attrs['data-output']||attrs['data-input']?.split(':')[0];el.innerHTML=String(html);for(const [k,v]of Object.entries(attrs))el.setAttribute(k,v);if(owner)el.dataset.owner=owner;overlay.appendChild(el);elements.push({el,pos});return el;}
 function labels(){art.sync(current?.values,current?.depths||{},level,selected,sample,results,busy,stage(),current?.output,tokenPreview,playbackTimeline,arrivalPreview);overlay.querySelectorAll('.node-label,.port,.source-label,.node-value,.local-fault,.fault-pin,.device-badge,.row-marker,.wire-actions,.demo-section-label').forEach(el=>el.remove());elements=[];
  for(const section of stage().demoSections||[])addLabel('span','demo-section-label',section.label,{'style':'--section-color:'+section.color},()=>new T.Vector3(section.x,(section.y||0)+.2,section.z));
  const edge=selectedWire&&curves.find(c=>c.to===selectedWire.to&&c.index===selectedWire.index&&c.from===selectedWire.from);
  if(edge&&!busy&&!focusId)addLabel('div','wire-actions',`<span>${outputName(edge.from)} → ${edge.to==='sink'?stage().outputTitle:nodeName(edge.to)}</span><button type="button" data-delete-wire>${I18n.t("删除连线")}</button><button type="button" data-close-wire aria-label="${I18n.t("关闭")}">×</button>`,{},()=>edge.curve.getPoint(.5));
  const removable=stage().buildable&&state.nodes.find(n=>n.id===actionNode);
  if(removable&&!busy&&!focusId&&!pending&&!placing&&!selectedWire)addLabel('div','wire-actions node-actions',`<button type="button" data-delete-node="${removable.id}">${I18n.t("删除组件")}</button><button type="button" data-close-node aria-label="${I18n.t("关闭")}">×</button>`,{'data-owner':removable.id},()=>new T.Vector3(removable.x,.7,removable.z+1.1));
  addLabel('span','source-label',`<b>${stage().sourceTitle}</b>${stage().sourceNote?'<br>'+stage().sourceNote:''}`,{'data-source-label':'R'},()=>new T.Vector3(stage().sourceX,sourceY('R')+(stage().dualSource?.95:1.6),art.sourceOffset(stage(),'R')-2.05));
  if(stage().dualSource)addLabel('span','source-label',`<b>${stage().sourceNames.Z}</b><br>${stage().secondaryTitle}`,{'data-source-label':'Z'},()=>new T.Vector3(stage().sourceX,sourceY('Z')+.95,art.sourceOffset(stage(),'Z')-2.05));
  addLabel('span','source-label',`<b>${stage().outputTitle}</b><br>${stage().tokenOutput?(stage().outputNote||I18n.t("词表选择后的文字")):shape(stage().sinkDim)+(stage().outputNote?' · '+stage().outputNote:stage().attentionReadout&&stage().sinkDim===9?' · '+M.causalGuide.weightsNote:I18n.t(" · 各位置的特征"))}`,{},()=>new T.Vector3(stage().sinkX,(stage().sinkY||0)+1.6,-2.05));
  if(busy&&!stage().sequence){const marker=addLabel('span','row-marker',flowIdentity(),{'data-owner':'R','data-row':sample<3?String(sample):'validation'},()=>sample<3?new T.Vector3(stage().sourceX-1,.99,art.rowZ(sample)):new T.Vector3(stage().sourceX,1.4,2));marker.style.setProperty('--row-color',colorHex(art.rowColor(sample)));}
  stage().sources.forEach(id=>addLabel('button','port '+(pending===id?'pending':''),(stage().sourcePorts?.[id]||(id==='R'&&stageId!=='ffn'?'X':id))+' · '+shape(stage().sourceDim||2),{'type':'button','data-output':id,'aria-label':`${outputName(id)}${I18n.t(" 输出口，")}${shape(stage().sourceDim||2)}`},()=>point(id)));
  if(stageId==='ffn')addLabel('span','source-label','Z = RMSNorm(R)',{},()=>new T.Vector3(stage().sourceX,.65,2.55));
  state.nodes.forEach(n=>{const d=definition(n.id);addLabel('button','node-label '+(selected===n.id?'selected ':''),nodeName(n.id),{'type':'button','data-node':n.id,'aria-label':`${nodeName(n.id)}${I18n.t(stage().demonstration?" 模块，双击查看":" 模块，拖动移动")}`},()=>new T.Vector3(n.x,(n.y||0)+1.05,n.z-1.15));
   if(stage().engine)addLabel('button','device-badge device-'+n.device+(busy&&current?.depths[n.id]===level?' processing':''),n.device.toUpperCase(),{'type':'button','data-assign':n.id,'data-owner':n.id,'aria-label':definition(n.id).name+I18n.t(" 当前在 ")+n.device+I18n.t("，选择后点击芯片区域")},()=>new T.Vector3(n.x,1.9,n.z-.45));
   d.ins.forEach((dim,i)=>addLabel('button','port input '+(n.links[i]?'connected ':''),n.id==='enattn'?'Q/K/V':moduleShape(M.nodeKind(n),dim),{'type':'button','data-input':`${n.id}:${i}`,'aria-label':`${nodeName(n.id)}${I18n.t(" 输入 ")}${i+1}，${moduleShape(M.nodeKind(n),dim)}${n.links[i]?I18n.t("，已连接"):''}`},()=>point(n.id,'in',i)));
   addLabel('button','port '+(pending===n.id?'pending':''),n.id==='enqkv'?'Q/K/V':moduleShape(M.nodeKind(n),d.out),{'type':'button','data-output':n.id,'aria-label':`${nodeName(n.id)}${I18n.t(" 输出口，")}${moduleShape(M.nodeKind(n),d.out)}`},()=>point(n.id));
   if(busy&&current?.values[n.id]&&!stage().sequence&&(current.depths[n.id]||0)<=level&&fault?.id!==n.id)addLabel('span','node-value',`${flowIdentity()} · ${vector(current.values[n.id])}`,{'data-owner':n.id},()=>new T.Vector3(n.x,.5,n.z+1.45));
  });
  addLabel('button','port input '+(state.sink?'connected ':''),String(shape(stage().sinkDim)),{'type':'button','data-input':'sink:0','aria-label':`${I18n.t("本层输出入口，")}${shape(stage().sinkDim)}${state.sink?I18n.t("，已连接"):''}`},()=>point('sink'));
  if(busy&&current?.output&&level===99&&!fault&&!stage().sequence)addLabel('span','node-value',vector(current.output),{},()=>new T.Vector3(stage().sinkX,.6,1.6));
  if(fault){
   const issue=current?.componentIssue,owner=issue?.id||'sink',pos=()=>point(owner),comparison=comparisonValues(fault.actual,fault.expected,stage().engine?(stage().limits?.error||.06):1e-9);
   const tokenText=stage().tokenOutput?`${I18n.t("<p class=\"token-verdict\">实际：")}${fault.actual?M.readTokens(fault.actual):I18n.t("未输出")}${I18n.t("<br>目标：")}${M.readTokens(fault.expected)}</p>`:'';
   const content=issue?`<button type="button" class="fault-close" data-dismiss-result aria-label="${I18n.t('关闭结果提示')}">×</button><strong>${I18n.t('组件用途不匹配')}</strong><p>${issue.message}</p>`:`${I18n.t("<button type=\"button\" class=\"fault-close\" data-dismiss-result aria-label=\"关闭结果提示\" title=\"关闭结果提示\">×</button><strong>最终结果 · ")}${sample<3&&!stage().sequence?I18n.t("位置 ")+sample+I18n.t("（输入“")+samples()[sample].word+'”）':samples()[sample].word}</strong>${tokenText}${I18n.t("<div><span>你的结果</span><b>")}${comparison.actual}${I18n.t("</b></div><div class=\"expected\"><span>预期结果</span><b>")}${comparison.expected}</b></div>${comparison.expanded?'<p class="fault-precision-note">'+I18n.t('此处显示更多小数；棋盘上的数字已四舍五入。')+'</p>':''}`;
   addLabel('span','fault-pin','!',{'data-owner':owner,'aria-hidden':'true'},pos);
   addLabel('div','local-fault',content,{'data-owner':owner,'role':'status'},pos);
  }
  render();
 }
 function project(p){const v=p.clone().project(camera);return [(v.x+1)*host.clientWidth/2,(1-v.y)*host.clientHeight/2];}
 function syncWireTargets(w,h){
  if(wireTargets.parentNode!==overlay)overlay.prepend(wireTargets);wireTargets.style.display=stage().demonstration||busy||focusId||placing||pending?'none':'';wireTargets.setAttribute('viewBox',`0 0 ${w} ${h}`);if(stage().demonstration)return;
  const key=I18n.language()+curves.map(c=>`${c.from}:${c.to}:${c.index}`).join('|');
  if(key!==wireTargetKey){wireTargetKey=key;wireTargets.replaceChildren();wirePaths=curves.map(edge=>{const path=document.createElementNS('http://www.w3.org/2000/svg','path');path.setAttribute('tabindex','0');path.setAttribute('role','button');path.setAttribute('aria-label',outputName(edge.from)+' → '+(edge.to==='sink'?stage().outputTitle:nodeName(edge.to))+I18n.t("，选择连线"));path.dataset.wireTo=edge.to;path.dataset.wirePort=String(edge.index);path.addEventListener('click',()=>chooseWire(edge.to,edge.index));path.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();chooseWire(edge.to,edge.index);}});wireTargets.appendChild(path);return path;});}
  curves.forEach((edge,i)=>{wirePaths[i].setAttribute('d',edge.curve.getPoints(32).map((p,j)=>{const [x,y]=project(p);return `${j?'L':'M'}${x.toFixed(1)} ${y.toFixed(1)}`;}).join(' '));wirePaths[i].classList.toggle('selected',!!selectedWire&&edge.to===selectedWire.to&&edge.index===selectedWire.index);});
 }
 function syncConnectionGuide(w,h){
  const visible=stageId==='wire'&&!state.sink&&!busy&&!pending&&!focusId;
  if(!visible){guide.setAttribute('hidden','');return;}
  if(guide.parentNode!==overlay)overlay.prepend(guide);
  guide.removeAttribute('hidden');guide.setAttribute('viewBox',`0 0 ${w} ${h}`);
  guidePath.setAttribute('d',curve(point('R'),point('sink','in')).getPoints(32).map((p,i)=>{const [x,y]=project(p);return `${i?'L':'M'}${x.toFixed(1)} ${y.toFixed(1)}`;}).join(' '));
 }
 function syncPlacementGuide(w,h){
  const flag=$('#module-palette [data-add-module]');
  if(!stage().placementTutorial||state.nodes.length||busy||focusId||!$('#module-shelf').hasAttribute('open')||!flag){placementGuide.setAttribute('hidden','');return;}
  if(placementGuide.parentNode!==overlay)overlay.prepend(placementGuide);
  placementGuide.removeAttribute('hidden');placementGuide.setAttribute('viewBox',`0 0 ${w} ${h}`);
  const [x,y]=project(new T.Vector3(0,.13,0));
  const corners=[[-1.04,-1.22],[1.04,-1.22],[1.04,1.22],[-1.04,1.22]].map(([a,b])=>project(new T.Vector3(a,.13,b)));
  placementOutline.setAttribute('d',corners.map(([a,b],i)=>`${i?'L':'M'}${a} ${b}`).join(' ')+' Z');
  const flagRect=flag.getBoundingClientRect(),hostRect=host.getBoundingClientRect(),sx=flagRect.right-hostRect.left+5,sy=flagRect.top+flagRect.height/2-hostRect.top;
  placementPath.setAttribute('d',`M${sx} ${sy} Q${(sx+x)/2} ${sy} ${x} ${y}`);placementPath.style.opacity=placing?'0':'.65';
 }
 // Check the visible feature blocks as well as the tray: projection fans the
 // features away from the packet center. Screen space also follows camera moves.
 function flowFootprints(pixelsPerUnit){
  const footprints=[];if(!busy)return footprints;
  for(const p of packets)for(const mesh of [p.mesh,p.otherMesh,p.outputMesh]){
   if(!mesh?.visible)continue;
   mesh.traverseVisible(o=>{
    if(!o.userData.packetTray&&o.userData.feature==null&&o.userData.featureValue==null)return;
    const alpha=o.material?.opacity??1;if(alpha<=0)return;
    if(!o.geometry.boundingSphere)o.geometry.computeBoundingSphere();
    const sphere=o.geometry.boundingSphere,center=sphere.center.clone().applyMatrix4(o.matrixWorld),[x,y]=project(center);
    footprints.push({x,y,r:sphere.radius*o.matrixWorld.getMaxScaleOnAxis()*pixelsPerUnit,alpha:Math.min(1,alpha/.3)});
   });
  }
  return footprints;
 }
 function fadePort(el,x,y,footprints,width,height,calculating){
  let fade=calculating.get(el.dataset.owner)||0;const halfWidth=(width||27)/2,halfHeight=(height||27)/2;
  for(const p of packets)if((p.worked||p.tokenOperation||p.centering)&&(p.node.id===el.dataset.output||(p.paired||p.tokenOperation||p.centering||p.sequenceProjection)&&el.dataset.input?.startsWith(p.node.id+':')))fade=Math.max(fade,p.portFade||0);
  for(const p of footprints){
   const gap=Math.hypot(Math.max(0,Math.abs(x-p.x)-halfWidth),Math.max(0,Math.abs(y-p.y)-halfHeight))-p.r;
   const t=Math.max(0,Math.min(1,gap/24)),ease=t*t*(3-2*t);
   fade=Math.max(fade,(1-ease)*p.alpha);
  }
  el.style.opacity=fade?String(1-.92*fade):'';
 }
 const minimumZoom=()=>stage().floating?.18:.45;
 function overviewZoomLimit(){return Math.max(1.35,stage().zoom);}
 function syncZoomControl(){
  const input=$('#camera-zoom');input.disabled=!!focusId||transitioning;
  if(focusId)return;
  const minScale=1/(overviewZoomLimit()*2),maxScale=1/minimumZoom();
  input.value=String(Math.max(0,Math.min(100,(1/zoom-minScale)/(maxScale-minScale)*100)));
  const percent=Math.round(stage().zoom/zoom*100)+'%';
  $('#camera-zoom-rate').textContent=percent;input.setAttribute('aria-valuetext',percent);
 }
 function render(){
  const w=host.clientWidth,h=host.clientHeight;if(!w||!h)return;
  syncZoomControl();
  const calculating=art.focusCalculations();
  for(const edge of curves){const fade=Math.max(calculating.get(edge.from)||0,calculating.get(edge.to)||0);for(const mesh of edge.visuals){const m=mesh.material,transparent=fade>0;if(m.transparent!==transparent){m.transparent=transparent;m.needsUpdate=true;}m.opacity=1-.86*fade;m.depthWrite=!transparent;}}
  renderer.setSize(w,h,false);const width=23*zoom,height=width*h/w;
  camera=stage().floating?spaceCamera:orthoCamera;
  if(stage().floating){
   camera.aspect=w/h;const distance=width/(2*Math.tan(camera.fov*Math.PI/360)*camera.aspect);
   camera.position.set(cameraTarget.x+Math.sin(yaw)*Math.cos(pitch)*distance,cameraTarget.y+Math.sin(pitch)*distance,cameraTarget.z+Math.cos(yaw)*Math.cos(pitch)*distance);
  }else{
   Object.assign(camera,{left:-width/2,right:width/2,top:height/2,bottom:-height/2});
   camera.position.set(cameraTarget.x+Math.sin(yaw)*24,cameraTarget.y+Math.sin(pitch)*24,cameraTarget.z+Math.cos(yaw)*Math.cos(pitch)*24);
  }
  camera.lookAt(cameraTarget);camera.updateProjectionMatrix();camera.updateMatrixWorld();art.updateDetail(camera,w,h,stage().floating||focusId?null:stage().zoom/zoom);syncFusion();
  for(const p of packets)if(p.tokenOperation||p.centering||p.sequenceProjection)for(const mesh of [p.mesh,p.mixing?p.otherMesh:null].filter(Boolean))for(const {label}of mesh.userData.featureRoutes)label.visible=label.material.opacity>0&&art.detailsVisible();
  renderer.render(scene,camera);
  syncConnectionGuide(w,h);syncPlacementGuide(w,h);syncWireTargets(w,h);syncConnectionError(w,h);
  const flow=flowFootprints(w/width);
  const titleFade=focusId?0:Math.max(0,Math.min(1,(zoom/overviewZoomLimit()-1)/.6));
  const titleOpacity=1-titleFade*titleFade*(3-2*titleFade);
  const labelScale=focusId?1:(stage().labelZoom||stage().zoom)/zoom;
  // Measure overlays together before moving them, avoiding layout per port.
  const layouts=elements.map(({el,pos})=>({el,xyz:pos(),xy:project(pos()),width:el.offsetWidth,height:el.offsetHeight}));
  layouts.forEach(({el,xyz,xy:[x,y],width,height})=>{
   const depth=stage().floating?-xyz.clone().applyMatrix4(camera.matrixWorldInverse).z:1,localScale=stage().floating?Math.min(1.25,labelScale*Math.min(3,camera.position.distanceTo(cameraTarget)/Math.max(.1,depth))):labelScale;
   el.hidden=!!focusId&&(el.dataset.owner!==focusId||el.classList.contains('node-label')||el.classList.contains('local-fault')||el.classList.contains('fault-pin'))||el.classList.contains('node-label')&&calculating.has(el.dataset.owner);
   if(el.classList.contains('node-label')||el.classList.contains('source-label')||el.classList.contains('port'))el.style.transform=`translate(-50%,-50%) scale(${localScale})`;
   if(el.classList.contains('node-label')||el.classList.contains('source-label')){el.style.opacity=String(titleOpacity);el.hidden=el.hidden||titleOpacity===0;}
   if(el.classList.contains('port')){el.disabled=!!stage().demonstration||!!focusId||titleOpacity===0;fadePort(el,x,y,flow,width*localScale,height*localScale,calculating);if(titleOpacity<1)el.style.opacity=String(Number(el.style.opacity||1)*titleOpacity);el.hidden=el.hidden||titleOpacity===0;}
   if(stage().floating&&(depth<=camera.near||xyz.clone().project(camera).z>1))el.hidden=true;
   if(el.classList.contains('local-fault')||el.classList.contains('wire-actions')){
    el.style.left=Math.max(12,Math.min(w-width-12,x+18))+'px';el.style.top=Math.max(12,Math.min(h-height-46,y+15))+'px';
   }else if(el.classList.contains('source-label')&&!stage().floating&&x>=0&&x<=w){
    const half=width*localScale/2+8;el.style.left=Math.max(half,Math.min(w-half,x))+'px';el.style.top=y+'px';
   }else{el.style.left=x+'px';el.style.top=y+'px';}
  });
  packets.forEach(p=>{
   const [x,y]=project(p.p.clone().add(new T.Vector3(0,.7,0)));p.el.style.left=x+'px';p.el.style.top=y+'px';
   if(p.calculation){
    const resultScale=w/width/60;
    const scale=stage().floating?Math.min(resultScale,w*.68/(p.calculation.offsetWidth||800)):focusId?resultScale:stage().zoom/zoom;
    p.equationVisible=art.detailsVisible();
    p.calculation.dataset.detail=p.equationVisible?'full':'abstract';p.calculation.style.setProperty('--calculation-scale',String(scale));
    const anchor=p.selection?art.selectionAnchor(p.node.id,p.selectionRow,p.progress??0):p.tokenOperation?art.tokenAnchor(p.node.id,p.tokenRow,p.input,p.progress??0):p.centering?art.layerNormAnchor(p.node.id,p.rmsRow,p.progress??0):p.normalizing?art.rmsAnchor(p.node.id,p.rmsRow):art.projectionOutlet(p.node.id,p.column);
    const [cx,cy]=project(anchor),cw=(p.equationVisible?(p.calculation.offsetWidth||p.equation.textContent.length*7+14):24)*scale;
    const radius=(p.normalizing||p.tokenOperation||p.selection?.2:art.projectionRadius(p.node.id,p.column))*w/width,offset=radius+9*scale,side=cx+offset+cw<w-12?'right':'left';
    p.calculation.dataset.side=side;
    p.calculation.style.left=(side==='right'?cx+offset:cx-offset)+'px';p.calculation.style.top=cy+'px';
    const accent=colorHex(p.selection?art.rowColor(p.selectionRow):p.tokenOperation?art.rowColor(p.tokenRow):p.centering&&!stage().sequence?p.color:p.normalizing?art.rowColor(p.rmsRow):p.paired||p.sequenceProjection?nodeColor(p.node.id):art.projectionColor);
    p.calculation.style.setProperty('--calculation-color',accent);p.calculation.style.setProperty('--calculation-halo',accent+'33');
    p.calculation.style.setProperty('--calculation-focus',String(p.focus||0));
    for(const {el,bar}of p.outputLabels){
     const [rx,ry]=project(bar.getWorldPosition(new T.Vector3()));
     if(p.sequenceProjection){el.style.left=rx+'px';el.style.top=(ry-12*resultScale)+'px';el.style.setProperty('--calculation-scale',String(resultScale));el.style.visibility=p.equationVisible?'':'hidden';continue;}
     const [sx,sy]=project(art.projectionOutlet(p.node.id,bar.userData.feature)),g=p.outputMesh.userData.gather;
     const aligned=p.tokenOperation||p.centering||p.selection;
     el.style.left=((sx+(aligned?0:28)*resultScale)*(1-g)+(rx-(aligned?0:7)*resultScale)*g)+'px';el.style.top=((sy-(aligned?16:0)*resultScale)*(1-g)+(ry-12*resultScale)*g)+'px';
     el.style.setProperty('--calculation-scale',String(resultScale));el.style.visibility=p.equationVisible?'':'hidden';
    }
   }
  });
 }

 function transition(target,targetZoom,targetYaw=yaw,targetPitch=pitch){
  cancelAnimationFrame(cameraFrame);const start=performance.now(),from=cameraTarget.clone(),z0=zoom,y0=yaw,p0=pitch,reduced=matchMedia('(prefers-reduced-motion:reduce)').matches;transitioning=true;host.setAttribute('aria-busy','true');
  function frame(now){const t=reduced?1:Math.min(1,(now-start)/480),ease=t*t*(3-2*t);cameraTarget.copy(from).lerp(target,ease);zoom=z0+(targetZoom-z0)*ease;yaw=y0+(targetYaw-y0)*ease;pitch=p0+(targetPitch-p0)*ease;render();if(t<1)cameraFrame=requestAnimationFrame(frame);else{transitioning=false;host.setAttribute('aria-busy','false');}}cameraFrame=requestAnimationFrame(frame);
 }
 function focus(id){
  const n=state.nodes.find(n=>n.id===id);if(!n)return;if(shelfFocus)leaveFocus();if(!focusId)overview={target:cameraTarget.clone(),zoom,yaw,pitch};focusId=id;selected=id;pending=placing=actionNode=null;clearPreview();clearPlacement();
  document.body.classList.add('module-focused');openModuleCard(id);
  $('#scene-help').textContent=I18n.t("近看模式 · 拖动旋转 · 滚轮缩放 · Esc 返回");
  assembly.children.forEach(g=>g.visible=g.userData.node===id);wires.visible=false;art.focus(true);labels();renderShelf();
  const width=Math.max(4.1,3.5*host.clientWidth/host.clientHeight);transition(new T.Vector3(n.x,(n.y||0)+.7,n.z),width/23,.32,.92);
 }
 function focusShelf(kind){
  if(busy||!shelfUnlocked(kind)||!Object.prototype.hasOwnProperty.call(shelfInventory(),kind))return;
  if(focusId)leaveFocus();
  overview={target:cameraTarget.clone(),zoom,yaw,pitch};shelfFocus={kind,selected};focusId='shelf-card';
  selected=actionNode=pending=placing=selectedWire=null;drag=lastTap=null;clearPreview();clearPlacement();
  const d=M.definitions[kind];shelfModel=art.module(scene,{id:'shelf-card',kind,x:0,z:0,links:d.ins.map(()=>null)},d,M.weights,{register:false});shelfModel.userData.shelfPreview=true;
  document.body.classList.add('module-focused');openModuleCard(null,kind,M.moduleName(stage(),kind));
  assembly.visible=false;wires.visible=false;art.focus(true);labels();renderShelf();
  const width=Math.max(4.1,3.5*host.clientWidth/host.clientHeight);transition(new T.Vector3(0,.7,0),width/23,.32,.92);
 }
 function openModuleCard(id,kind=kindOf(id),name=nodeName(id)){
  clearConnectionError();
  const cardStageId=M.exampleLevel(kind,stageId),cardStage=M.levels[cardStageId];
  const c=cardFor(kind,cardStageId)||M.extraCard(kind,cardStageId),d=M.definitions[kind],card=$('#module-card'),outputProjection=['output','original-output'].includes(kind);
  $('#card-attention-flow').hidden=d.type!=='mix'&&!outputProjection;$('#card-attention-flow').replaceChildren();
  $('.card-example>small').textContent=d.type==='mix'?M.aggregateGuide.rounding:I18n.t('按正确计算路径演算 · 显示数值已四舍五入');
  card.style.setProperty('--card-accent',c.color);$('.card-example-title').firstChild.textContent=cardStage.sequence?I18n.t("用三个输入位置看一次"):I18n.t("用“猫”的向量看一次");
  $('#card-purpose-title').textContent=I18n.t(cardStageId==='autoregressive'?'它在生成循环中的作用':'它在这一层的作用');
  const origin=M.packageOrigin(kind);
  for(const [field,value]of Object.entries({name,seal:c.seal,english:c.english,family:c.family,number:origin?I18n.t('来源：')+origin.chapter:stage().chapter,ability:c.ability,meaning:c.meaning,formula:c.formula,purpose:M.componentUse(kind)||c.purpose}))$('#card-'+field).textContent=value;
  if(origin)$('#card-number').dataset.originLevel=origin.id;else delete $('#card-number').dataset.originLevel;
  $('#card-stats').innerHTML=`${I18n.t("<div><span>向量维度</span><b>")}${d.ins.map(dim=>moduleShape(kind,dim)).join(' / ')} → ${moduleShape(kind,d.out)}${I18n.t("</b></div><div><span>权重矩阵</span><b>")}${c.weight}${I18n.t("</b></div><div><span>计算方式</span><b>")}${c.trait}</b></div>`;
  $('#card-package').hidden=!M.packages?.[kind];$('#card-package').innerHTML=M.packages?.[kind]?`<h4>${I18n.t('内部组成')}</h4>${packageDiagram(kind)}`:'';
  $('#card-example-input').textContent=cardStageId==='ffn'?'R = [1, 1]':cardStage.sequence?I18n.t("三个位置的特征 · 教学缩尺"):'X = [1, 1]';
  const names={Q:I18n.t("查询"),K:I18n.t("键"),V:I18n.t("内容"),A:I18n.t("注意力"),L:I18n.t("词表分数"),T:I18n.t("选出的词"),B:I18n.t("计算结果"),X:I18n.t("输入向量"),P:I18n.t("投影结果"),R:I18n.t("原始输入"),Z:I18n.t("归一化输入"),G:I18n.t("门分支"),U:I18n.t("内容分支"),S:cardStage.original?I18n.t('相关分数 S'):['activation','plain-ffn'].includes(cardStageId)?I18n.t("SiLU 的输出"):I18n.t("激活门值"),H:I18n.t("门控结果"),F:I18n.t("本次变换"),Y:I18n.t("本层输出")};
  Object.assign(names,M.original?.exampleLabels(cardStageId));
  if(cardStageId==='add-norm')names.R=I18n.t('残差相加结果');
  const ex=M.extraExample?.(kind,cardStageId,id)||exampleFor(M,kind,cardStageId),vectorRow=([name,values],output=false,focus=-1,label=names[name]||name)=>`<div class="card-vector ${output?'result':''}" data-symbol="${name}"><b>${name}${label===name?'':' · '+label}</b><span ${(cardStage.original||ex.rms)&&cardStage.sequence?'style="display:grid;grid-template-columns:repeat('+Math.ceil(values.length/3)+',minmax(0,1fr))"':''}>${values.map((v,i)=>`<i class="${v<0?'negative':''} ${i===focus?'example-focus':''}">${fmt(v)}</i>`).join('')}</span></div>`;
  const formulaNumber=v=>Number(v.toFixed(4)).toString();
  if(ex.sampleIndex===0&&!cardStage.sequence){$('.card-example-title').firstChild.textContent=I18n.t("用“我”的向量看一次");$('#card-example-input').textContent=ex.exampleSource[0]+' = '+vector(ex.exampleSource[1]);}
  if(['step','feedback'].includes(kind)||kind==='pick'&&cardStageId==='autoregressive'){
   $('.card-example-title').firstChild.textContent=I18n.t('用“我 / 爱 / 猫”看一次更新');
   $('#card-example-input').textContent=I18n.t('词与预测的对应关系');
   $('.card-example>small').textContent=I18n.t('固定查表的教学例子，不代表语言模型的真实预测。');
   $('#card-example-values').innerHTML=M.structureMap.generationExample(kind);
  }else if(outputProjection){
   const copy=M.outputProjectionGuide,w=M.weights[kind],width=w.length,columns=w[0].length,source=c.inputSymbol,words=ex.queryWords||M.samplesFor(cardStageId)[ex.sampleIndex||0].word.split(' / ');
   const input=ex.inputs[0][1],output=ex.output[1],values=input.slice(0,width),result=output.slice(0,columns),inputName=source==='C'?copy.concatInput:copy.input;
   const factor=v=>v<0?'('+fmt(v)+')':fmt(v);
   $('.card-example-title').firstChild.textContent=copy.title;$('#card-example-input').textContent=words[0];
   $('#card-attention-flow').innerHTML=`<h4>${copy.flowTitle}</h4><div class="wo-flow"><div><code>${source==='C'?'head A ∥ head B → C':'A × V → O'}</code><p>${source==='C'?copy.concat:copy.aggregate}</p></div><div><code>${source} × Wo → Y</code><p>${copy.projection}</p></div></div>`;
   $('#card-example-values').innerHTML=`<div class="wo-example" data-width="${width}"><p class="wo-vector"><b>${source} · ${words[0]}</b><code>${vector(values)}</code></p><div class="wo-columns">${w[0].map((_,j)=>`<div class="wo-column" data-column="${j}" data-value="${result[j]}"><div><b>${copy.feature} ${j+1}</b><small>${copy.column} ${j+1}</small><code>${vector(w.map(row=>row[j]))}</code></div><code class="wo-equation">${values.map((x,i)=>`<span>${factor(x)} × ${factor(w[i][j])}</span>`).join(' + ')} ≈ <strong>${fmt(result[j])}</strong></code></div>`).join('')}</div><p class="wo-vector result"><b>Y · ${words[0]}</b><code>${vector(result)}</code></p><p>${copy.recipe} ${copy.sameWeights}</p><details class="wo-full"><summary>${copy.full}</summary><p>${cardStage.chapter} · ${cardStage.title}</p><table><thead><tr><th>${copy.word}</th><th>${source} · ${inputName}</th><th>Y · ${copy.output}</th></tr></thead><tbody>${words.map((word,i)=>`<tr><th>${word}</th><td><code>${vector(input.slice(i*width,(i+1)*width))}</code></td><td><code>${vector(output.slice(i*columns,(i+1)*columns))}</code></td></tr>`).join('')}</tbody></table></details><details class="wo-heads"><summary>${copy.multiheadTitle}</summary><p>${copy.multihead}</p></details></div>`;
  }else if(d.type==='score'){
   const data=scoreComparison(M,ex),copy=M.scoreGuide,words=ex.queryWords||M.samplesFor(cardStageId)[ex.sampleIndex||0].word.split(' / ');
   const keys=ex.keyWords||(cardStage.dualSource?cardStage.secondaryTokenIds.map(id=>M.math.vocab[id]):words);
   const percent=value=>(value*100).toFixed(1)+'%',percentVector=values=>'['+values.map(percent).join(', ')+']';
   const signed=x=>x<0?'('+formulaNumber(x)+')':formulaNumber(x);
   const terms=data.q.map((x,i)=>signed(x)+' × '+signed(data.k[i])).join(' + ');
   $('.card-example-title').firstChild.textContent=copy.title;$('#card-example-input').textContent='dₖ = '+data.width;
   $('#card-example-values').innerHTML=`<div class="score-example" data-width="${data.width}"><p class="score-shapes"><code>Q (3 × ${data.width}) × Kᵀ (${data.width} × 3) → 3 × 3</code>${copy.shape}</p><p class="score-dimension">${copy.dimension} <b>${data.width}</b>：<code>√dₖ = √${data.width} ≈ ${formulaNumber(data.divisor)}</code></p><div class="score-pair"><div class="query"><b>${copy.query} · ${words[data.query]}</b><code>${vector(data.q)}</code></div><div class="key"><b>${copy.key} · ${keys[data.key]}</b><code>${vector(data.k)}</code></div></div><div class="score-worked"><code>${terms} ≈ ${formulaNumber(data.raw[data.key])}<br>${formulaNumber(data.raw[data.key])} / √${data.width} ≈ <b>${formulaNumber(data.scaled[data.key])}</b></code></div><table class="score-table"><thead><tr><th>${copy.key}</th><th>${copy.dot}</th><th>${copy.scaled}</th></tr></thead><tbody>${keys.map((word,i)=>`<tr class="${i===data.key?'worked':''}"><th>${word}</th><td>${formulaNumber(data.raw[i])}</td><td>${formulaNumber(data.scaled[i])}</td></tr>`).join('')}</tbody></table><section class="score-softmax"><h4>${copy.compare}</h4><p>${copy.boundary}</p><div class="score-distributions"><span></span><b>${copy.before}</b><b>${copy.after}</b>${keys.map((word,i)=>`<span>${word}</span>${['before','after'].map(mode=>`<div class="score-bar ${mode}" data-mode="${mode}" data-weight="${data[mode][i]}"><i style="width:${data[mode][i]*100}%" aria-hidden="true"></i><span>${percent(data[mode][i])}</span></div>`).join('')}`).join('')}</div><p>${copy.explanation}</p><div class="score-shortcut"><b>${copy.shortcut}</b><code>Softmax(${vector(data.simple.raw)})<br>≈ ${percentVector(data.simple.before)}<br>Softmax(${vector(data.simple.raw)} / √${data.width})<br>≈ ${percentVector(data.simple.after)}</code></div></section><details class="score-theory"><summary>${copy.reason}</summary><p>${copy.theory}</p><p>${copy.training}</p></details><details class="score-full"><summary>${copy.full}</summary>${ex.inputs.map(v=>vectorRow(v)).join('')+vectorRow(ex.output,true)}</details></div>`;
  }else if(d.type==='mix'){
   const data=aggregationExample(M,ex),copy=M.aggregateGuide,queries=ex.queryWords||M.samplesFor(cardStageId)[ex.sampleIndex||0].word.split(' / ');
   const cross=ex.cross??cardStage.dualSource,sources=ex.keyWords||(cross?cardStage.secondaryTokenIds.map(id=>M.math.vocab[id]):queries);
   const scorePath=cross?copy.crossScorePath:(ex.masked??(cardStage.nodes.includes('mask')||cardStageId==='aggregation'))?copy.maskedScorePath:copy.scorePath;
   $('#card-attention-flow').innerHTML=`<h4>${copy.flowTitle}</h4><div class="av-branches"><div class="av-a"><small>${scorePath}</small><b>${copy.a}</b></div><div class="av-v"><small>${cross?copy.crossValuePath:copy.valuePath}</small><b>${copy.v}</b></div></div><div class="av-arrows" aria-hidden="true"><span>↘</span><span>↙</span></div><div class="av-merge"><b><span>A</span> × <span>V</span> → O</b><small>${copy.o}</small></div><p>${copy.dynamic}</p>`;
   const percent=v=>Number((v*100).toFixed(1))+'%';
   const breakdown=(part,words=null)=>`<table class="av-table"><thead><tr><th>${copy.proportion}</th><th></th><th>${copy.content}</th><th></th><th>${copy.contribution}</th></tr></thead><tbody>${part.weights.map((weight,i)=>`<tr data-weight="${weight}"><th scope="row">${percent(weight)}</th><td class="av-operator">×</td><td class="av-value">${words?'<b class="av-word">'+words[i]+'</b>':''}<code>${vector(part.values[i])}</code></td><td class="av-operator">${words?'≈':'='}</td><td class="av-contribution"><code>${vector(part.contributions[i])}</code></td></tr>`).join('')}</tbody></table><div class="av-sum"><b>${copy.sum}</b><code>${words?'O · '+queries[data.query]+' ≈ ':part.contributions.map(vector).join(' + ')+' = '}<strong>${vector(part.output)}</strong></code></div>`;
   $('.card-example-title').firstChild.textContent=copy.title;$('#card-example-input').textContent='A × V';
   $('#card-example-values').innerHTML=`<div class="av-example" data-width="${data.width}"><p>${copy.simple}</p><div class="av-simple">${breakdown(data.simple)}</div><p>${copy.whole}</p><details class="av-actual"><summary>${copy.actual}</summary><p class="av-example-level">${cardStage.chapter} · ${cardStage.title}</p><p class="av-query">${copy.query} <b>${queries[data.query]}</b></p><p class="av-sources">${copy.sources} ${sources.join(' / ')}</p>${breakdown(data,sources)}<p>${copy.rest}</p><div class="av-shape"><code>A (3 × 3) × V (3 × ${data.width}) → O (3 × ${data.width})</code><p>${copy.dimensions}</p></div></details></div>`;
  }else if(ex.rms){
   const {width,meanSquare,divisor}=ex.rms,v=ex.inputs[0][1].slice(0,width);
   $('#card-example-values').innerHTML=ex.inputs.map(v=>vectorRow(v)).join('')+`<div class="card-silu-worked"><small>${I18n.t('这里只展开第一个位置的归一化，其他位置分别计算。')}</small><code>mean(X²) = ${formulaNumber(meanSquare)}<br>r = √(${formulaNumber(meanSquare)} + ε) ≈ ${formulaNumber(divisor)}<br>${vector(v)} / ${formulaNumber(divisor)} ≈ ${vector(ex.output[1].slice(0,width))}</code></div>`+vectorRow(ex.output,true);
  }else if(ex.normalization){
   $('#card-stats').children[1].querySelector('span').textContent=M.layerNormGuide.parameterTitle;
   const {width,mean,variance,epsilon}=ex.normalization,copy=M.layerNormGuide,input=ex.inputs[0][1].slice(0,width),centered=input.map(x=>x-mean),output=ex.output[1].slice(0,width),divisor=Math.sqrt(variance+epsilon),term=x=>x<0?'('+formulaNumber(x)+')':formulaNumber(x);
   $('#card-example-values').innerHTML=ex.inputs.map(v=>vectorRow(v,false,-1,copy.input)).join('')+`<div class="card-silu-worked card-layernorm-steps">${cardStage.sequence?'<small>'+I18n.t('这里只展开第一个位置的归一化，其他位置分别计算。')+'</small>':''}<b>${copy.center}</b><code>μ = (${input.map(term).join(' + ')}) / ${width} ≈ ${formulaNumber(mean)}<br>${vector(input)} − ${term(mean)} = ${vector(centered)}</code><b>${copy.scale}</b><code>σ² = (${centered.map(x=>term(x)+'²').join(' + ')}) / ${width} ≈ ${formulaNumber(variance)}<br>${vector(centered)} / √(${formulaNumber(variance)} + ε)<br>≈ ${vector(centered)} / ${formulaNumber(divisor)} ≈ ${vector(output)}</code><b>${copy.affine}</b><code>1 × ${vector(output)} + 0 ≈ ${vector(output)}</code><small>${copy.parameters}</small></div>`+vectorRow(ex.output,true,-1,copy.output);
  }else if(ex.activation?.kind==='relu'){
   const {feature}=ex.activation,input=ex.inputs[0],output=ex.output,x=input[1][feature];
   $('#card-example-values').innerHTML=`<div class="card-silu-example">${vectorRow(input,false,feature)}<div class="card-silu-link"><b>ReLU</b><span>${input[1].map((_,i)=>`<i class="${i===feature?'example-focus':''}">↓</i>`).join('')}</span></div>${vectorRow(output,true,feature)}<div class="card-silu-worked"><code>max(0, ${formulaNumber(x)}) = ${fmt(output[1][feature])}</code></div></div>`;
  }else if(ex.activation){
   const {feature,content,product}=ex.activation,input=ex.inputs[0],output=ex.output,x=input[1][feature],y=output[1][feature];
   const next=content?`<p class="card-silu-use">${I18n.t("把这个 S 与对应的内容特征相乘：")}</p><div class="card-gate-result"><span data-role="content"><small>${I18n.t("内容 U")}</small><b>${fmt(content[feature])}</b></span><i>×</i><span data-role="scale"><small>${I18n.t("SiLU 输出 S")}</small><b>${fmt(y)}</b></span><i>≈</i><span data-role="result"><small>${I18n.t("相乘结果")}</small><b>${fmt(product[feature])}</b></span></div>`:'';
   $('#card-example-values').innerHTML=`<div class="card-silu-example">${vectorRow(input,false,feature,I18n.t("SiLU 的输入"))}<div class="card-silu-link"><b>SiLU</b><span>${input[1].map((_,i)=>`<i class="${i===feature?'example-focus':''}" aria-hidden="true">↓</i>`).join('')}</span></div>${vectorRow(output,true,feature,I18n.t("SiLU 的输出"))}<div class="card-silu-worked"><small>${I18n.t("把高亮的输入代入公式：")}</small><code>SiLU(${formulaNumber(x)}) = ${formulaNumber(x)} / (1 + e<sup>${formulaNumber(-x)}</sup>) ≈ ${fmt(y)}</code></div>${next}</div>`;
  }else if(ex.addition){
   const {feature}=ex.addition,[base,update]=ex.inputs,result=ex.output,roles={[base[0]]:I18n.t("原有特征"),[update[0]]:I18n.t("修改量"),[result[0]]:I18n.t("更新结果")};
   const a=base[1][feature],b=update[1][feature],y=result[1][feature],signed=v=>v<0?'('+fmt(v)+')':fmt(v),equal=[a,b,y].some(v=>Math.abs(v-Number(v.toFixed(2)))>1e-9)?'≈':'=';
   $('#card-example-values').innerHTML=`<div class="card-add-example">${[base,update,result].map(row=>vectorRow(row,row===result,feature,roles[row[0]])).join('')}<div class="card-add-step"><small>${I18n.t("同一位置，原值加上修改量：")}</small><code><span data-role="baseline">${signed(a)}</span> + <span data-role="update">${signed(b)}</span> ${equal} <b>${fmt(y)}</b></code></div><p class="card-add-zero">${I18n.t("如果修改量为零：")} <code>${update[0]} = ${vector(update[1].map(()=>0))} → ${result[0]} = ${base[0]}</code></p></div>`;
  }else if(ex.comparison){
   const {input,rows,feature}=ex.comparison,values=Object.fromEntries(rows),u=values.U[feature],g=values.G[feature],s=values.S[feature],h=values.H[feature];
   const roles={U:I18n.t("线性投影"),G:I18n.t("门投影"),S:I18n.t("SiLU 系数"),H:I18n.t("相乘结果")};
   $('.card-example-title').firstChild.textContent=I18n.t("用“我”的向量比较两条支路");$('#card-example-input').textContent=input[0]+' = '+vector(input[1]);
   $('#card-example-values').innerHTML=`<div class="card-comparison"><p class="card-example-focus">${I18n.t("只看高亮的这一项：")}</p>${rows.map(row=>vectorRow(row,row[0]==='H',feature,roles[row[0]])).join('')}<div class="card-gate-step">SiLU(${formulaNumber(g)}) ≈ ${fmt(s)}</div><div class="card-gate-result"><span data-role="content"><small>${I18n.t("内容")}</small><b>${fmt(u)}</b></span><i>×</i><span data-role="scale"><small>${I18n.t("调节系数")}</small><b>${fmt(s)}</b></span><i>≈</i><span data-role="result"><small>${I18n.t("相乘结果")}</small><b>${fmt(h)}</b></span></div></div>`;
  }else $('#card-example-values').innerHTML=ex.inputs.map(v=>vectorRow(v)).join('')+vectorRow(ex.output,true);
  const contrast=$('#card-merge-contrast');contrast.hidden=d.communication||!['multiply','add','residual'].includes(d.type);contrast.replaceChildren();
  if(!contrast.hidden){
   const a=[4,2],b=[.25,1.5],product=a.map((v,i)=>v*b[i]),sum=a.map((v,i)=>v+b[i]);
   contrast.innerHTML=`<b>${I18n.t('用同两组数比较：')}</b><code>A = ${vector(a)}　B = ${vector(b)}</code><div class="merge-compare-row ${d.type==='multiply'?'current':''}" data-operation="multiply"><strong>×</strong><span><code>A ⊙ B = ${vector(product)}</code><small>${I18n.t('逐项调节强弱；乘以 1 保持原值。')}</small></span></div><div class="merge-compare-row ${d.type!=='multiply'?'current':''}" data-operation="add"><strong>+</strong><span><code>A + B = ${vector(sum)}</code><small>${I18n.t('在原值上增加或减少；加上 0 保持原值。')}</small></span></div><small>${I18n.t('这里只比较运算：门控中 B 是调节系数，残差中 B 是修改量。')}</small>`;
  }
  $('.card-path').hidden=!!cardStage.demonstration;
  $('#card-return').innerHTML=I18n.t(cardStage.demonstration?'收起卡片，继续查看 ↗':'收起卡片，继续搭建 ↗');
  $('.card-path>span').textContent=cardStage.title;
  $('#card-path').innerHTML=(cardStageId==='ffn'?[I18n.t("双路投影"),'SiLU',I18n.t("门控"),I18n.t("降维"),I18n.t("残差")]:cardStage.nodes.map(k=>M.definitions[k].name)).map((label,i)=>`${i?'<i>›</i>':''}<span class="${(cardStageId==='ffn'?i===c.stage:cardStage.nodes[i]===kind)?'active':''}">${label}</span>`).join('');
  host.setAttribute('aria-label',I18n.t("拖动旋转模块，滚轮缩放"));$('#card-portrait').append(host,overlay);if(!card.open)card.showModal();
 }
 function leaveFocus(){
  if(!focusId)return;
  // Restore synchronously, including when the opening animation is unfinished.
  cancelAnimationFrame(cameraFrame);cameraFrame=0;transitioning=false;
  const previous=overview,returnShelf=shelfFocus;overview=null;
  cameraTarget.copy(previous?.target||new T.Vector3());zoom=previous?.zoom??stage().zoom;yaw=previous?.yaw??.05;pitch=previous?.pitch??.8;
  host.setAttribute('aria-busy','false');
  focusId=null;shelfFocus=null;pending=null;clearPreview();if(shelfModel){scene.remove(shelfModel);dispose(shelfModel);shelfModel=null;}if(returnShelf)selected=returnShelf.selected;
  document.body.classList.remove('module-focused');$('.arena').prepend(host,overlay);host.setAttribute('aria-label',I18n.t(stage().floating?"拖动旋转 · Shift / 右键拖动平移 · 滚轮缩放":stage().demonstration?"拖动转动视角 · 双击查看基础组件":"拖动空白转动视角，拖动模块重新摆放"));$('#module-card').close();
  $('#scene-help').textContent=I18n.t(stage().floating?"拖动旋转 · Shift / 右键拖动平移 · 滚轮缩放":stage().demonstration?"拖动转动视角 · 双击查看基础组件":"拖模块摆放 · 点连线删除 · 双击查看模块卡");assembly.visible=true;assembly.children.forEach(g=>g.visible=true);wires.visible=true;art.focus(false);labels();renderShelf();
  $(returnShelf?'[data-add-module="'+returnShelf.kind+'"]':'[data-node="'+selected+'"]')?.focus({preventScroll:true});
 }
 function nodeTap(id,e){const now=performance.now();if(lastTap?.id===id&&now-lastTap.time<380&&Math.hypot(e.clientX-lastTap.x,e.clientY-lastTap.y)<10){lastTap=null;focus(id);}else{lastTap={id,time:now,x:e.clientX,y:e.clientY};select(id);}}
 function worldPoint(x,y){const r=host.getBoundingClientRect();cursor.set((x-r.left)/r.width*2-1,-((y-r.top)/r.height*2-1));ray.setFromCamera(cursor,camera);const p=new T.Vector3();return ray.ray.intersectPlane(plane,p)?p:null;}
 function finishFusion(completed=false){
  if(!fusionEffect)return;const effect=fusionEffect;fusionEffect=null;cancelAnimationFrame(effect.frame);effect.visual.dispose();
  assembly.visible=wires.visible=true;document.body.classList.remove('synthesizing-component');$('#component-fusion').hidden=true;
  document.querySelectorAll('.fusion-awaiting,.fusion-arrival').forEach(el=>{el.classList.remove('fusion-awaiting','fusion-arrival');el.style.removeProperty('box-shadow');});
  render();effect.resolve(completed);
 }
 function syncFusion(){
  const effect=fusionEffect;if(!effect)return;
  const {kind}=effect.reward,d=M.definitions[kind],target=$('[data-add-module="'+kind+'"]'),phase=effect.phase;
  if(phase>=.67&&!effect.scrolled){target?.scrollIntoView?.({block:'nearest'});effect.scrolled=true;}
  const hostRect=host.getBoundingClientRect?.()||{left:0,top:0,width:host.clientWidth,height:host.clientHeight};
  const rect=target?.getBoundingClientRect?.(),x=rect?.width?rect.left+rect.width/2-hostRect.left:73,y=rect?.height?rect.top+rect.height/2-hostRect.top:145;
  const depth=effect.visual.center.clone().project(camera).z,destination=new T.Vector3(x/host.clientWidth*2-1,1-y/host.clientHeight*2,depth).unproject(camera);
  effect.visual.update(phase,destination,effect.reduced);
  $('#component-fusion').dataset.phase=phase<.38?'gather':phase<.91?'formed':'shelf';
  $('#fusion-stage').textContent=phase<.38?M.fusionCopy.gathering:phase<.91?M.fusionCopy.formed:stage().buildable?M.fusionCopy.stored:M.fusionCopy.storedLater;
  $('#fusion-name').textContent=d.name;$('#fusion-shape').textContent=d.ins.map(dim=>moduleShape(kind,dim)).join(' + ')+' → '+moduleShape(kind,d.out);$('#fusion-skip').textContent=M.fusionCopy.skip;
  if(target){target.classList.toggle('fusion-awaiting',phase<.84);target.classList.toggle('fusion-arrival',phase>=.84);if(phase>=.84)target.style.boxShadow='0 0 0 '+(3+Math.sin((phase-.84)/.16*Math.PI)*5)+'px #76b7a345';}
 }
 function fuse(reward,id){
  clearPackets();const visual=art.fusionVisual(assembly,wires,reward.kind,M.definitions[reward.kind],M.weights),reduced=matchMedia('(prefers-reduced-motion:reduce)').matches;
  return new Promise(resolve=>{
   const effect={reward,visual,reduced,resolve,phase:0,frame:0,scrolled:false};fusionEffect=effect;
   assembly.visible=wires.visible=false;document.body.classList.add('synthesizing-component');$('#component-fusion').hidden=false;
   $('#module-shelf').setAttribute('open','');$('[data-add-module="'+reward.kind+'"]')?.scrollIntoView?.({block:'nearest'});
   let previous=animationTime(),elapsed=0;render();
   function tick(){
    if(fusionEffect!==effect)return;
    if(id!==runId||!busy){finishFusion(false);return;}
    try{
     const now=animationTime();elapsed+=Math.max(0,now-previous);previous=now;effect.phase=Math.min(1,elapsed/(reduced?900:3600));render();
     if(effect.phase>=1){finishFusion(true);return;}effect.frame=requestAnimationFrame(tick);
    }catch(error){console.error('Component celebration failed',error);finishFusion(false);}
   }
   effect.frame=requestAnimationFrame(tick);
  });
 }
 function clearPackets(){stopFollowFlight();finishFusion(false);tokenPreview=arrivalPreview=null;art.outputReadout(sample,results,busy,stage());playbackTimeline=null;followedPacket=null;packets.forEach(p=>{p.el.remove();p.calculation?.remove();p.outputLabels?.forEach(({el})=>el.remove());for(const mesh of [p.mesh,p.otherMesh,p.outputMesh].filter(Boolean)){scene.remove(mesh);mesh.traverse(o=>{o.geometry?.dispose();o.material?.map?.dispose();o.material?.dispose();});}});packets=[];art.clearProcessing();render();}
 function finishDemo(id){
  if(!followData)return Promise.resolve();
  const from=cameraTarget.clone(),startZoom=zoom,startYaw=yaw,startPitch=pitch;let previous=animationTime(),elapsed=0;
  return new Promise(resolve=>{function tick(now){
   if(id!==runId||!busy||!followData){resolve();return;}
   const clock=animationTime(now);if(paused){previous=clock;requestAnimationFrame(tick);return;}
   elapsed+=Math.max(0,clock-previous);previous=clock;const t=Math.min(1,elapsed/800),ease=t*t*(3-2*t);
   cameraTarget.copy(from).multiplyScalar(1-ease);zoom=startZoom+(stage().zoom-startZoom)*ease;yaw=startYaw+(.05-startYaw)*ease;pitch=startPitch+(.8-startPitch)*ease;
   render();if(t===1)resolve();else requestAnimationFrame(tick);
  }requestAnimationFrame(tick);});
 }
 // Token feedback belongs to the generation loop, outside the Transformer.
 function feedback(token,id){
  clearPackets();
  const mesh=new T.Mesh(new T.SphereGeometry(.32,20,16),material(0xe2b64d,{emissive:0x9c6a20,emissiveIntensity:.6}));scene.add(mesh);
  const el=document.createElement('span');el.className='packet feedback-packet';el.dataset.motion='feedback';el.dataset.from='sink';el.innerHTML='<b class="packet-title"></b> <span class="packet-values"></span>';el.querySelector('b').textContent=I18n.t('接回输入');el.querySelector('span').textContent=M.math.vocab[token];overlay.appendChild(el);
  const p={mesh,el,p:point('sink'),motion:'feedback',values:[token],cells:el.querySelector('.packet-values')};packets.push(p);
  const a=point('sink'),b=point('R'),path=new T.CatmullRomCurve3([a,new T.Vector3(15,8,6),new T.Vector3(-18,8,6),b]);
  const duration=700;let elapsed=0,previous=animationTime();
  return new Promise(resolve=>{function tick(now){
   if(id!==runId){resolve();return;}if(!busy){clearPackets();resolve();return;}
   const clock=animationTime(now);if(paused){previous=clock;requestAnimationFrame(tick);return;}
   const delta=Math.max(0,clock-previous);previous=clock;elapsed+=delta*playbackSpeed;
   const t=Math.min(1,elapsed/duration);p.p=path.getPoint(t*t*(3-2*t));mesh.position.copy(p.p);
   if(followData&&!focusId&&!followFlight){const ease=1-Math.exp(-Math.min(delta,100)/350);const profile=followProfiles[followView];cameraTarget.lerp(p.p,ease);zoom+=((followZoom??Math.max(profile.min,followView==='close'?.9:1.6))-zoom)*ease;yaw+=(profile.yaw-yaw)*ease;pitch+=(profile.pitch-pitch)*ease;}
   render();if(t===1){clearPackets();resolve();}else requestAnimationFrame(tick);
  }requestAnimationFrame(tick);});
 }
 const normalizationFrame=(time,centering)=>art[centering?'layerNormFrame':'rmsFrame'](time,stage().sequence?3:1);
 function animate(ev,ms,id){
  clearPackets();lastDepth=-1;const reduced=matchMedia('(prefers-reduced-motion:reduce)').matches;if(reduced&&!stage().demonstration)ms=70;
  let previous=animationTime(),elapsed=0;
  const seq=!!stage().sequence,readyNodes=state.nodes.filter(n=>ev.values[n.id]);
  // Follow the actual outlet producer: selected predictions and an appended
  // context have different reveal clocks. Unused branches cannot fill the outlet.
  const tokenSink=seq&&stage().tokenOutput&&ev.output&&readyNodes.find(n=>n.id===state.sink&&(definition(n.id).type==='pick'||M.nodeKind(n)==='feedback'));
  const carriesTokens=from=>definition(from)?.type==='pick'||kindOf(from)==='feedback';
  const pairedNode=id=>!seq&&['multiply','add'].includes(definition(id)?.type);
  const mixingNode=id=>definition(id)?.type==='mix';
  const projectionNode=id=>definition(id)?.type==='linear'||definition(id)?.type==='vocabulary'&&!!weightsFor(id);
  const handledSecondary=id=>pairedNode(id)||mixingNode(id)||definition(id)?.type==='concat'&&definition(id)?.out===4;
  const wordsFrom=source=>{
   let from=source;const visited=new Set();
   while(from&&!visited.has(from)){visited.add(from);const ids=ev.values[from];if(ids?.length===3&&ids.every(v=>Number.isInteger(v)&&M.math.vocab[v]!=null))return ids.map(i=>M.math.vocab[i]);if(from==='Z'&&stage().secondaryTokenIds)return stage().secondaryTokenIds.map(i=>M.math.vocab[i]);from=state.nodes.find(other=>other.id===from)?.links[0];}
   return samples()[sample].word.split(' / ');
  };
  // Travel is 1.5× faster than before; every operator keeps its original clock.
  const travelDuration=2/3,independent=!!stage().demonstration;
  const schedule=animationSchedule(readyNodes,ev.depths,n=>{
   const d=definition(n.id),tokenOperation=art.tokenOperation(d);
   return d.type==='pick'?art.selectionFrame(0,d.ins[0]/d.out,d.out).duration:tokenOperation?art.tokenFrame(0,tokenOperation).duration:['norm','layernorm'].includes(d.type)?normalizationFrame(0,d.type==='layernorm').duration:seq&&(projectionNode(n.id)||d.type==='mix')?art.sequenceProjectionFrame(0,d.type==='mix'?d.out/3:weightsFor(n.id)[0].length).duration:!seq&&(projectionNode(n.id)||['multiply','add'].includes(d.type))?art.projectionFrame(0,ev.values[n.id].length).duration:1;
  },travelDuration,independent);
  const timings=schedule.nodes,ends=schedule.depthEnds,lastWave=ends.length-1;
  // Reveal the actual matrix only after its packet reaches the outlet, then
  // give it reading time on the same pausable, speed-controlled clock.
  const outlet=(stage().outputArrivalHold&&seq||!seq&&sample>=3)&&ev.output&&curves.find(edge=>edge.to==='sink'&&ev.values[edge.from]);
  const arrived=outlet?(timings.get(outlet.from)?.end||0)+travelDuration:null,hold=outlet?(stage().outputArrivalHold??450)/ms:0;
  const max=outlet?Math.max(schedule.end,arrived+hold):schedule.end;
  if(independent)playbackTimeline={nodes:timings,time:-1};
  // The complete model uses the same operation clock as every teaching level.
  // Never compress it to a total runtime: slow playback must keep each column readable.
  const clamp=x=>Math.max(0,Math.min(1,x)),ramp=(x,a,b)=>{const v=clamp((x-a)/(b-a));return v*v*(3-2*v);};
  function fadeMaterials(mesh){const materials=[];mesh.traverse(o=>{if(o.material){o.material.transparent=true;o.material.depthWrite=false;o.castShadow=false;materials.push(o.material);}});return materials;}
  function movingPacket(values,from,to,motion){
   const color=seq?(nodeColor(from)):art.rowColor(sample);
   const el=document.createElement('span');el.className='packet';el.hidden=true;el.dataset.motion=motion;el.dataset.row=seq?'sequence':sample<3?String(sample):'validation';el.dataset.from=from;el.dataset.to=to;el.style.setProperty('--row-color',colorHex(color));
   const title=document.createElement('b');title.className='packet-title';title.textContent=packetIdentity(from);el.appendChild(title);
   const cells=document.createElement('span');cells.className='packet-values';
   cells.textContent=motion==='wire'&&carriesTokens(from)?M.readTokens(values):seq?shape(values.length):vector(values);
   el.appendChild(document.createTextNode(' '));el.appendChild(cells);overlay.appendChild(el);
   const mesh=art.packet(values,color,{sequence:seq});Object.assign(mesh.userData,{packet:motion==='wire',row:el.dataset.row,from,to});mesh.visible=false;scene.add(mesh);
   return {values,el,cells,mesh,color,materials:fadeMaterials(mesh),p:new T.Vector3(),motion};
  }
  // Each wire departs when its producer finishes. Early inputs wait only at
  // their actual join, until that component has received every input.
  const processPaths=new Map(readyNodes.map(n=>[n.id,new T.CubicBezierCurve3(point(n.id,'in',0),new T.Vector3(n.x-.7,(n.y||0)+1.15,n.z),new T.Vector3(n.x+.7,(n.y||0)+1.15,n.z),point(n.id))]));
  curves.filter(c=>ev.values[c.from]).forEach(edge=>{
   const p=movingPacket(ev.values[edge.from],edge.from,edge.to,'wire');
   const perRow=!seq&&sample<3,departure=perRow&&['R','Z'].includes(edge.from)?new T.LineCurve3(rowEndpoint('source'),edge.curve.getPoint(0)):null,arrival=perRow&&edge.to==='sink'?new T.LineCurve3(edge.curve.getPoint(1),rowEndpoint('sink')):null;
   const start=timings.get(edge.from)?.end||0,processStart=timings.get(edge.to)?.start??null;
   Object.assign(p,{edge,departure,arrival,start,travelEnd:start+travelDuration,processStart,end:processStart==null?start+travelDuration:processStart+(edge.index>0&&!handledSecondary(edge.to)?.25:0)});
   p.p=departure?.getPoint(0)||edge.curve.getPoint(0);p.mesh.position.copy(p.p);packets.push(p);
  });
  readyNodes.forEach(n=>{
   const input=ev.values[n.links[0]],output=ev.values[n.id],p=movingPacket(input,n.id,n.id,'process');
   p.el.classList.add('processing-packet');p.el.dataset.node=n.id;p.mesh.userData.processing=n.id;
   const operation=definition(n.id).type,appending=M.nodeKind(n)==='feedback',mixing=operation==='mix',selection=operation==='pick',vocabulary=operation==='vocabulary'&&projectionNode(n.id),detailed=projectionNode(n.id)||mixing,sequenceProjection=seq&&detailed,paired=pairedNode(n.id),gather=operation==='concat'&&output.length===4,worked=detailed||paired,other=paired||mixing||gather?ev.values[n.links[1]]:null;
   const tokenOperation=art.tokenOperation(definition(n.id)),centering=operation==='layernorm',normalizing=operation==='norm'||centering,pairTones=paired?art.pairColorsFor(n.id):null,operator=operation==='add'?'+':'×';
   const columns=mixing?output.length/3:detailed?weightsFor(n.id)[0].length:output.length,matrix=mixing?M.math.rows(other,columns):detailed?weightsFor(n.id):null;
   Object.assign(p,{appending,selection,vocabulary,detailed,sequenceProjection,columns,matrix,mixing,paired,gather,worked,other,normalizing,centering,tokenOperation});
   if(appending)p.cells.textContent=M.readTokens(input);
   if(mixing)p.valueWords=()=>wordsFrom(n.links[1]);
   if(worked||tokenOperation||centering||selection)art.prepareProjectionPacket(p.mesh,paired?()=>pairTones[0]:mixing?i=>art.rowColor(i%3):undefined);
   if(paired||mixing||gather){
    const otherMesh=art.packet(other,p.color,{sequence:mixing});otherMesh.userData={processingSecondary:n.id,row:p.el.dataset.row};otherMesh.visible=false;scene.add(otherMesh);
    art.prepareProjectionPacket(otherMesh,mixing?i=>art.rowColor(Math.floor(i/columns)):()=>gather?0x9b88c4:pairTones[1]);fadeMaterials(otherMesh);p.otherMesh=otherMesh;
   }
   const outputMesh=art.packet(output,p.color,{sequence:seq});outputMesh.userData={processingOutput:n.id,row:p.el.dataset.row};outputMesh.visible=false;scene.add(outputMesh);
   if(gather){
    p.cells.textContent=vector(input)+' ∥ '+vector(other);
    const tint=(mesh,tone)=>mesh.children.filter(o=>o.userData.feature!=null).forEach(o=>{const color=typeof tone==='function'?tone(o.userData.feature):tone;o.material.color.set(color);o.material.emissive.set(color);});
    tint(p.mesh,0x659bab);tint(p.otherMesh,0x9b88c4);tint(outputMesh,i=>i<2?0x659bab:0x9b88c4);
   }
   const morph=document.createElement('span');morph.className='packet-morph';p.cells.replaceWith(morph);morph.appendChild(p.cells);
   const after=document.createElement('span');after.className='packet-values';after.textContent=selection||appending?M.readTokens(output):seq?shape(output.length):vector(output);after.style.opacity='0';morph.appendChild(after);
   p.el.style.visibility='hidden';p.el.hidden=false;
   const beforeWidth=p.cells.offsetWidth||p.cells.textContent.length*6.6,afterWidth=after.offsetWidth||after.textContent.length*6.6;
   p.el.hidden=true;p.el.style.visibility='';morph.style.width=beforeWidth+'px';
   if(worked||normalizing||tokenOperation||selection){
    art.prepareProjectionOutput(outputMesh);
    p.outputLabels=output.map((value,j)=>{
     const el=document.createElement('span');el.className='projection-result'+(tokenOperation||centering||sequenceProjection||selection?' token-result':'');el.hidden=true;el.textContent=selection?M.math.vocab[value]:fmt(value);el.dataset.node=n.id;el.dataset.column=String(j);el.dataset.row=p.el.dataset.row;overlay.appendChild(el);
     return {el,bar:outputMesh.children.find(o=>o.userData.feature===j)};
    });
   }
   if(selection){
    const calculation=document.createElement('span');calculation.className='projection-calculation selection-calculation';calculation.hidden=true;calculation.dataset.node=n.id;calculation.dataset.operation='pick';
    const equation=document.createElement('span');equation.className='projection-equation';
    const token=document.createElement('b'),comparison=document.createElement('span'),result=document.createElement('b');equation.append(token,comparison,result);calculation.append(equation);
    const abstract=document.createElement('span');abstract.className='projection-sum-symbol';abstract.textContent='max';calculation.append(abstract);overlay.append(calculation);p.words=()=>wordsFrom(n.links[0]);
    const width=input.length/output.length,updateSelection=timing=>{
     const key=I18n.language()+':'+timing.row+':'+timing.column+':'+timing.stage;if(p.selectionKey===key)return;p.selectionKey=key;
     const row=timing.row,column=timing.column,scores=input.slice(row*width,(row+1)*width),choice=art.selectionChoice(scores,column),name=i=>M.math.vocab[i]+' '+fmt(scores[i]);
     token.textContent=p.words()[row]+' · ';result.textContent='';
     if(timing.stage==='emit'){comparison.textContent='max → ';result.textContent=name(output[row])+' · ID '+output[row];}
     else if(choice.previous===null)comparison.textContent=name(column);
     else{comparison.textContent=name(column)+(scores[column]>scores[choice.previous]?' > ':scores[column]===scores[choice.previous]?' = ':' < ')+name(choice.previous);result.textContent=' → '+M.math.vocab[choice.best];}
     calculation.dataset.selectionRow=String(row);calculation.dataset.candidate=String(column);calculation.dataset.best=String(choice.best);p.selectionRow=row;
    };
    Object.assign(p,{calculation,equation,updateSelection,selectionRow:0});updateSelection(art.selectionFrame(0,width,output.length));
   }
   if(tokenOperation){
    // Follow the actual ID-producing input, including repeated/reordered IDs
    // and additional operator copies. Feature vectors retain position identity.
    let from=n.links[0],ids=null;const visited=new Set();
    while(from&&!visited.has(from)){
     visited.add(from);const values=ev.values[from];
     if(values?.length===3&&values.every(v=>Number.isInteger(v)&&M.math.vocab[v]!=null)){ids=[...values];break;}
     from=state.nodes.find(node=>node.id===from)?.links[0];
    }
    const words=()=>ids?ids.map(i=>M.math.vocab[i]):samples()[sample].word.split(' / '),width=output.length/3;
    const calculation=document.createElement('span');calculation.className='projection-calculation token-calculation';calculation.hidden=true;calculation.dataset.node=n.id;calculation.dataset.operation=tokenOperation;
    const equation=document.createElement('span');equation.className='projection-equation';
    const token=document.createElement('b'),left=document.createElement('span'),join=document.createTextNode(''),right=document.createElement('span'),result=document.createElement('b');
    left.style.color=colorHex(0x397aa2);right.style.color=colorHex(0xb68036);equation.append(token,left,join,right,result);calculation.append(equation);
    const abstract=document.createElement('span');abstract.className='projection-sum-symbol';abstract.setAttribute('aria-hidden','true');calculation.append(abstract);overlay.append(calculation);
    const updateToken=timing=>{
     const key=I18n.language()+':'+timing.row+':'+timing.stage;if(p.tokenEquationKey===key)return;p.tokenEquationKey=key;
     const row=timing.row,values=tokenOperation==='lookup'?output.slice(row*width,(row+1)*width):input.slice(row*width,(row+1)*width),out=output.slice(row*width,(row+1)*width);
     token.textContent=words()[row]+' · ';result.textContent='';
     if(tokenOperation==='lookup'){
      left.textContent=timing.stage==='address'?'ID '+input[row]:'E['+input[row]+']';join.textContent=timing.stage==='address'?' → ':' = ';right.textContent=timing.stage==='address'?'E['+input[row]+']':vector(values);abstract.textContent='E[ ]';
     }else if(tokenOperation==='mask'){
      left.textContent=vector(values);join.textContent=' → ';right.textContent=timing.stage==='arrive'?vector(values):vector(out);abstract.textContent='◩';
      calculation.dataset.maskedColumns=Array.from({length:width},(_,j)=>j>row?j:null).filter(j=>j!==null).join(',');
     }else{
      const scaled=values.map(x=>x*Math.sqrt(width)),angles=values.map((_,j)=>row/10000**(2*Math.floor(j/2)/width)),codes=angles.map((angle,j)=>(j%2?Math.cos:Math.sin)(angle));
      if(timing.stage==='scale'){left.textContent=vector(values)+' × '+(width===2?'√2':'2');join.textContent=' ≈ ';right.textContent=vector(scaled);abstract.textContent='×';}
      else if(timing.stage==='position'){left.textContent='p = '+row+' · ['+angles.map((angle,j)=>(j%2?'cos':'sin')+'('+fmt(angle)+')').join(', ')+']';join.textContent=' ≈ ';right.textContent=vector(codes);abstract.textContent='∿';}
      else{left.textContent=vector(scaled);join.textContent=' + ';right.textContent=vector(codes);result.textContent=' ≈ '+vector(out);abstract.textContent='+';}
     }
     calculation.dataset.tokenRow=String(row);calculation.dataset.tokenStage=timing.stage;p.tokenRow=row;
    };
    Object.assign(p,{calculation,equation,words,updateToken,tokenRow:0});updateToken(art.tokenFrame(0,tokenOperation));
   }
   if(normalizing){
    const calculation=document.createElement('span');calculation.className='projection-calculation '+(centering?'layernorm-calculation':'rms-calculation');calculation.hidden=true;calculation.dataset.node=n.id;calculation.dataset.operation=centering?'layernorm':'rmsnorm';
    const equation=document.createElement('span');equation.className='projection-equation';
    const token=document.createElement('b'),formula=document.createElement('span');equation.append(token,formula);calculation.append(equation);
    const abstract=document.createElement('span');abstract.className='projection-sum-symbol';abstract.textContent='÷r';abstract.setAttribute('aria-hidden','true');calculation.append(abstract);overlay.append(calculation);
    const factor=v=>v<0?'('+fmt(v)+')':fmt(v),width=input.length/(seq?3:1);
    const updateRMS=timing=>{
     const key=I18n.language()+':'+timing.row+':'+timing.stage;if(p.rmsEquationKey===key)return;p.rmsEquationKey=key;
     const row=timing.row,values=input.slice(row*width,(row+1)*width),squares=values.map(x=>x*x),r=Math.sqrt(squares.reduce((a,b)=>a+b,0)/width+1e-6);
     const word=seq?samples()[sample].word.split(' / ')[row]:samples()[sample].word;
     token.textContent=word+' · ';
     if(centering){
      const stats=art.layerNormStats(values),divisor=Number(stats.divisor.toFixed(4)),stage=timing.stage;
      formula.textContent=stage==='mean'?'μ = ('+values.map(factor).join(' + ')+') / '+width+' ≈ '+fmt(stats.mean):stage==='center'?vector(values)+' − '+factor(stats.mean)+' ≈ '+vector(stats.centered):stage==='variance'?'s = √(('+stats.centered.map(x=>factor(x)+'²').join(' + ')+') / '+width+' + ε) ≈ '+divisor:vector(stats.centered)+' / '+divisor+' ≈ '+vector(output.slice(row*width,(row+1)*width));
      abstract.textContent=stage==='mean'?'μ':stage==='center'?'−μ':stage==='variance'?'√v':'÷s';
     }else formula.textContent=timing.stage==='squares'?vector(values)+'² → '+vector(squares):timing.stage==='rms'?'r = √(('+values.map(x=>factor(x)+'²').join(' + ')+') / '+width+' + ε) ≈ '+fmt(r):vector(values)+' / '+Number(r.toFixed(4))+' ≈ '+vector(output.slice(row*width,(row+1)*width));
     calculation.dataset.normRow=String(row);calculation.dataset.normStage=timing.stage;p.rmsRow=row;
    };
    Object.assign(p,{calculation,equation,updateRMS,rmsRow:0});updateRMS(normalizationFrame(0,centering));
   }
   if(worked){
    const calculation=document.createElement('span');calculation.className='projection-calculation';calculation.hidden=true;calculation.dataset.node=n.id;calculation.dataset.operation=operation;
    const equation=document.createElement('span');equation.className='projection-equation';calculation.appendChild(equation);
    const abstract=document.createElement('span');abstract.className='projection-sum-symbol';abstract.textContent=paired?operator:'∑';abstract.setAttribute('aria-hidden','true');calculation.appendChild(abstract);
    const factor=v=>v<0?'('+fmt(v)+')':fmt(v),terms=[],width=sequenceProjection?matrix.length:input.length;
    const token=document.createElement('b');if(sequenceProjection){calculation.classList.add('sequence-projection-calculation');equation.append(token);p.words=()=>wordsFrom(n.links[0]);}
    if(mixing)calculation.classList.add('aggregation-calculation');
    (paired?[input,other]:input.slice(0,width)).forEach((_,i)=>{if(i)equation.appendChild(document.createTextNode(paired?' '+operator+' ':' + '));const term=document.createElement('span');term.style.color=colorHex(paired?pairTones[i]:mixing?art.rowColor(i):art.featureColor(i));equation.appendChild(term);terms.push(term);});
    const bias=vocabulary?document.createElement('span'):null;if(bias){bias.className='projection-bias';bias.style.color=colorHex(0xb68036);equation.appendChild(document.createTextNode(' + '));equation.appendChild(bias);}
    const sum=document.createElement('b');sum.style.opacity='0';equation.appendChild(sum);overlay.appendChild(calculation);
    const updateEquation=(column,row=0)=>{
     const values=sequenceProjection?input.slice(row*width,(row+1)*width):input,result=output[row*p.columns+column];
     const operands=paired?[input[column],other[column]]:[...values,...matrix.map(w=>w[column])];
     terms.forEach((term,i)=>term.textContent=paired?factor(operands[i]):factor(values[i])+' × '+factor(matrix[i][column]));
     if(bias){const value=definition(n.id).bias?.[column]||0;bias.textContent=factor(value);bias.dataset.value=String(value);}
     const rounded=[...operands,result].some(v=>Math.abs(v-Number(v.toFixed(2)))>1e-9);sum.textContent=(rounded?' ≈ ':' = ')+fmt(result);
     if(sequenceProjection){token.textContent=(mixing?'O · ':'')+p.words()[row]+(vocabulary?' → '+M.math.vocab[column]:'')+' · ';calculation.dataset.projectionRow=String(row);}
    };
    updateEquation(0);Object.assign(p,{calculation,equation,sum,updateEquation,column:0,projectionRow:0});
   }
   Object.assign(p,{node:n,input,output,outputMesh,outputMaterials:fadeMaterials(outputMesh),morph,after,beforeWidth,afterWidth,...timings.get(n.id),path:processPaths.get(n.id)});
   p.p=p.path.getPoint(0);p.mesh.position.copy(p.p);p.outputMesh.position.copy(p.p);packets.push(p);
  });
  art.pulseSource(reduced?1:0);
  return new Promise(resolve=>{function tick(now){
   if(id!==runId){resolve();return;}
   if(!busy){clearPackets();resolve();return;}
   // Accumulate time so changing speed mid-flight never jumps or rewinds a packet.
   const clock=animationTime(now);
   if(paused){previous=clock;requestAnimationFrame(tick);return;}
   const delta=Math.max(0,clock-previous);elapsed+=delta*playbackSpeed;previous=clock;
   const t=Math.min(max,elapsed/ms);level=0;while(level<lastWave&&t>=ends[level+1])level++;
   if(t>=max){clearPackets();labels();renderShelf();resolve();return;}
   if(playbackTimeline)playbackTimeline.time=t;
   const phase=independent?[...timings.values()].reduce((count,n)=>count+Number(t>=n.start)+Number(t>=n.end),0):level;
   if(phase!==lastDepth){lastDepth=phase;labels();renderShelf();}
   if(outlet){
    const progress=clamp((t-arrived)/hold);
    arrivalPreview=t>=arrived?{values:ev.output,pulse:reduced||progress>=.45?0:Math.sin(Math.PI*progress/.45)**2}:null;
    art.outputReadout(sample,results,busy,stage(),tokenPreview,arrivalPreview);
   }
   if(tokenSink){
    const values=ev.values[tokenSink.id],clock=t-timings.get(tokenSink.id).start;
    let ready=0;
    if(M.nodeKind(tokenSink)==='feedback'){
     // The retained words come from the stored context. Reveal the new final
     // token when the append operation's outgoing packet is fully formed.
     ready=clock<0?0:clock<.9?values.length-1:values.length;
    }else{
     const timing=art.selectionFrame(clock,definition(tokenSink.id).ins[0]/values.length,values.length);
     ready=Math.max(0,Math.min(values.length,Math.floor((clock-timing.ready)/timing.rowDuration)+1));
    }
    if(ready!==(tokenPreview?.length||0)){tokenPreview=values.slice(0,ready);labels();}
   }
   art.pulseSource(reduced?1:t);
   packets.forEach(p=>{
    const active=t>=p.start&&t<p.end;
    if(stage().demonstration&&!active&&!p.wasActive)return;p.wasActive=active;
    const tt=t-p.start,timing=p.selection?art.selectionFrame(tt,p.input.length/p.output.length,p.output.length):p.sequenceProjection?art.sequenceProjectionFrame(tt,p.columns):p.tokenOperation?art.tokenFrame(tt,p.tokenOperation):p.normalizing?normalizationFrame(tt,p.centering):p.worked?art.projectionFrame(tt,p.output.length):null,progress=timing?timing.phase:clamp(tt/(p.motion==='wire'?travelDuration:1));
    if(p.motion==='process'){
     p.progress=tt;
     const visible=tt>=0&&t<p.end&&(!focusId||focusId===p.node.id),blend=ramp(progress,p.worked?.76:.55,p.worked?.9:.83);
     p.el.hidden=!visible;p.el.dataset.phase=progress<.23?'enter':progress<.65?'compute':'gather';p.p=p.path.getPoint(progress);
     p.cells.style.opacity=String(1-blend);p.after.style.opacity=String(blend);p.morph.style.width=(p.beforeWidth+(p.afterWidth-p.beforeWidth)*blend)+'px';
     const inputAlpha=1-ramp(progress,p.worked?.72:.2,p.worked?.86:.42),outputAlpha=ramp(progress,p.worked?.8:.65,p.worked?.96:.9);
     p.mesh.visible=visible&&inputAlpha>0;p.outputMesh.visible=visible&&outputAlpha>0;
     p.materials.forEach(m=>m.opacity=inputAlpha);p.outputMaterials.forEach(m=>m.opacity=outputAlpha);p.outputMesh.position.copy(p.p);
     if(p.gather){
      const center=new T.Vector3(p.node.x,(p.node.y||0)+1.7,p.node.z),join=ramp(progress,.12,.68),depart=ramp(progress,.8,1);
      p.mesh.position.copy(point(p.node.id,'in',0)).lerp(center.clone().add(new T.Vector3(-.24,0,0)),join);
      p.otherMesh.position.copy(point(p.node.id,'in',1)).lerp(center.clone().add(new T.Vector3(.24,0,0)),join);
      const alpha=1-ramp(progress,.68,.8);p.mesh.visible=p.otherMesh.visible=visible&&alpha>0;
      p.mesh.traverse(o=>{if(o.material)o.material.opacity=alpha;});p.otherMesh.traverse(o=>{if(o.material)o.material.opacity=alpha;});
      const outputAlpha=ramp(progress,.68,.8);p.outputMesh.visible=visible&&outputAlpha>0;p.outputMaterials.forEach(m=>m.opacity=outputAlpha);
      p.outputMesh.position.copy(center).lerp(point(p.node.id),depart);p.p.copy(p.outputMesh.position);
     }
     if(p.worked){
      if(p.paired){
       p.otherMesh.visible=visible&&inputAlpha>0;
       art.routePairedPacket(p.node.id,p.mesh,tt,0);art.routePairedPacket(p.node.id,p.otherMesh,tt,1);
       // Once the values reach their sockets, leave only the token name above
       // the operator. The local paired equation carries the arithmetic.
       const entering=1-ramp(progress,.18,.35),leaving=ramp(progress,.8,.94);
       p.cells.style.opacity=String(entering);p.after.style.opacity=String(leaving);p.morph.style.width=(p.beforeWidth*entering+p.afterWidth*leaving)+'px';
      }else if(p.sequenceProjection){
       p.mesh.visible=visible&&tt<timing.end;p.el.hidden=!visible||(progress>.08&&progress<.9);
       art.routeSequenceProjectionInput(p.node.id,p.mesh,tt,p.p,p.path.getPoint(.04));
       if(p.mixing){p.otherMesh.visible=visible&&tt<.33;art.routeAggregationValues(p.node.id,p.otherMesh,tt);}
      }else art.routeProjectionPacket(p.node.id,p.mesh,progress,p.p,p.path.getPoint(.12));
      if(p.sequenceProjection)art.routeSequenceProjectionOutput(p.node.id,p.outputMesh,tt,p.p);else art.routeProjectionOutput(p.node.id,p.outputMesh,tt,p.p);
      p.outputMesh.visible=visible&&tt>=.67;
      p.portFade=visible?ramp(progress,p.sequenceProjection?.04:.2,p.sequenceProjection?.12:.38):0;
      const local=(p.sequenceProjection?timing.rowTime:tt)-timing.offset;
      p.focus=visible?timing.focus:0;
      p.calculation.hidden=!visible||local<.36||local>=.76;p.calculation.style.opacity=String(ramp(local,.36,.4)*(1-ramp(local,.71,.76)));p.sum.style.opacity=String(ramp(local,.61,.67));
      p.outputLabels.forEach(({el},j)=>{
       const ready=p.sequenceProjection?Math.floor(j/p.columns)*timing.rowDuration+.97+j%p.columns*timing.step:.76+j*timing.step;
       const alpha=ramp(tt,ready,ready+.04)*(1-ramp(progress,.89,.97));
       el.hidden=!visible||alpha<=0;el.style.opacity=String(alpha);
      });
      p.calculation.dataset.column=String(timing.column);
      if(p.column!==timing.column||p.sequenceProjection&&p.projectionRow!==timing.row){p.column=timing.column;p.projectionRow=timing.row||0;p.updateEquation(p.column,p.projectionRow);}
     }
     if(p.selection){
      p.el.hidden=!visible||(progress>.04&&progress<.9);p.mesh.visible=visible&&tt<timing.end;
      art.routeSelectionInput(p.node.id,p.mesh,tt,p.p,p.path.getPoint(.04));
      art.routeSelectionOutput(p.node.id,p.outputMesh,tt,p.p);p.outputMesh.visible=visible&&tt>=timing.ready;
      p.portFade=visible?ramp(progress,.04,.12):0;p.focus=visible?timing.focus:0;
      p.updateSelection(timing);p.calculation.hidden=!visible||timing.rowTime<.36||timing.rowTime>=timing.rowDuration-.02;
      p.calculation.style.opacity=String(ramp(timing.rowTime,.36,.4)*(1-ramp(timing.rowTime,timing.rowDuration-.08,timing.rowDuration)));
      p.outputLabels.forEach(({el},j)=>{const ready=j*timing.rowDuration+timing.ready,alpha=ramp(tt,ready,ready+.03)*(1-ramp(progress,.9,.98));el.hidden=!visible||alpha<=0;el.style.opacity=String(alpha);});
     }
     if(p.tokenOperation){
      p.el.hidden=!visible||(progress>.12&&progress<.88);p.mesh.visible=visible&&tt<timing.end;
      art.routeTokenInput(p.node.id,p.mesh,tt,p.p,p.path.getPoint(.04));
      art.routeTokenOutput(p.node.id,p.outputMesh,tt,p.p);p.outputMesh.visible=visible&&tt>=timing.start+.82*timing.step;
      p.portFade=visible?ramp(progress,.1,.2):0;p.focus=visible&&timing.local>=.04&&timing.local<.94?1:0;
      p.updateToken(timing);p.calculation.hidden=!visible||tt<timing.start||tt>=timing.end;p.calculation.style.opacity=String(ramp(timing.local,0,.06)*(1-ramp(timing.local,.94,1)));
      const width=p.output.length/3;
      p.outputLabels.forEach(({el},j)=>{const ready=timing.start+(Math.floor(j/width)+.84)*timing.step,alpha=ramp(tt,ready,ready+.04)*(1-ramp(progress,.9,.98));el.hidden=!visible||alpha<=0;el.style.opacity=String(alpha);});
     }
     if(p.normalizing){
      p.el.hidden=!visible||(progress>.18&&progress<.88);
      if(p.centering){p.mesh.visible=visible&&tt<timing.end;art.routeLayerNormInput(p.node.id,p.mesh,tt,p.p,p.path.getPoint(.04),p.input,p.output);art.routeLayerNormOutput(p.node.id,p.outputMesh,tt,p.p);}
      else{p.mesh.visible=visible&&progress<.3;p.materials.forEach(m=>m.opacity=1-ramp(progress,.15,.3));art.routeRMSOutput(p.node.id,p.outputMesh,tt,p.p);}
      p.outputMesh.visible=visible&&tt>=timing.start+(p.centering?.89:.82)*timing.step;
      p.portFade=visible?ramp(progress,.18,.3):0;p.focus=visible&&timing.local>=.05&&timing.local<.94?1:0;
      p.updateRMS(timing);p.calculation.hidden=!visible||tt<timing.start||tt>=timing.end;p.calculation.style.opacity=String(ramp(timing.local,0,.06)*(1-ramp(timing.local,.94,1)));
      const width=p.output.length/(seq?3:1);
      p.outputLabels.forEach(({el},j)=>{
       const ready=timing.start+(Math.floor(j/width)+(p.centering?.94:.84))*timing.step,alpha=ramp(tt,ready,ready+.05)*(1-ramp(progress,.9,.98));
       el.hidden=!visible||alpha<=0;el.style.opacity=String(alpha);
      });
     }
     art.processing(p.node.id,tt,p.input,p.output,p.color,{sequence:seq,detailed:p.detailed,paired:p.paired,other:p.other,words:p.words?.(),valueWords:p.valueWords?.()});
    }else{
     const inView=!focusId||p.edge.from===focusId||p.edge.to===focusId,visible=t>=p.start&&t<p.end&&inView;
     p.el.hidden=!visible;p.mesh.visible=visible;
     const a=p.departure?.18:0,b=p.arrival?.18:0;
     p.p=a&&progress<a?p.departure.getPoint(progress/a):b&&progress>1-b?p.arrival.getPoint((progress-1+b)/b):p.edge.curve.getPoint((progress-a)/(1-a-b));
     // A secondary input joins the same processing event after both are ready.
     const joining=p.processStart!=null&&p.edge.index>0&&!handledSecondary(p.edge.to)&&t>=p.processStart;
     const alpha=joining?1-ramp(t-p.processStart,0,.25):1;
     if(joining)p.p.lerp(processPaths.get(p.edge.to).getPoint(.3),ramp(t-p.processStart,0,.25));
     p.el.style.opacity=String(alpha);p.materials.forEach(m=>m.opacity=alpha);
    }
    if(!p.paired)p.mesh.position.copy(p.p);
   });
   if(stage().floating&&followData&&!focusId)frameFollowing(followTracking(),delta);
   render();requestAnimationFrame(tick);
  }requestAnimationFrame(tick);});
 }

 function clearPreview(){if(preview){scene.remove(preview);preview.geometry.dispose();preview.material.dispose();preview=null;}}
 function clearPlacement(){if(placementMesh){scene.remove(placementMesh);dispose(placementMesh);placementMesh=null;}}
 function previewPlacement(x,y){
  if(!placing)return;const rect=host.getBoundingClientRect();
  if(x<rect.left||x>rect.right||y<rect.top||y>rect.bottom){if(placementMesh)placementMesh.visible=false;return;}
  const p=worldPoint(x,y);if(!p)return;
  if(!placementMesh){placementMesh=art.module(scene,{id:'placement-preview',kind:placing,x:0,z:0,links:[]},M.definitions[placing],M.weights,{register:false});placementMesh.userData.placementGhost=true;
   placementMesh.traverse(o=>{if(o.material){o.material.transparent=true;o.material.opacity=.48;o.material.depthWrite=false;o.castShadow=false;}});
  }
  placementMesh.visible=true;placementMesh.position.set(p.x,.1,p.z);placementMesh.userData.validPlacement=!!M.placement(state,placing,p.x,p.z);
  placementMesh.traverse(o=>{if(o.material?.emissive){o.material.emissive.set(placementMesh.userData.validPlacement?0x234d3e:0xa03524);o.material.emissiveIntensity=.3;}});render();
 }
 function previewAt(x,y){clearPreview();if(!pending)return;const p=worldPoint(x,y);if(!p)return;p.y=.6;preview=new T.Line(new T.BufferGeometry().setFromPoints(curve(point(pending),p).getPoints(30)),new T.LineDashedMaterial({color:0xbd8e57,dashSize:.2,gapSize:.1}));preview.computeLineDistances();scene.add(preview);render();}
 overlay.addEventListener('click',e=>{const b=e.target.closest('button');if(!b||busy)return;if(b.hasAttribute('data-delete-node')){removeModule(b.dataset.deleteNode);return;}if(b.hasAttribute('data-close-node')){actionNode=null;labels();return;}if(b.hasAttribute('data-delete-wire')){deleteWire();return;}if(b.hasAttribute('data-close-wire')){selectedWire=null;labels();return;}if(b.hasAttribute('data-dismiss-result')){e.stopPropagation();fault=null;$('#run-report').hidden=true;labels();$('#run').focus({preventScroll:true});return;}if(wantsPan(e))return;if(b.dataset.assign){select(b.dataset.assign);say(I18n.t("把芯片标记拖到目标区域，或点击下方 CPU / GPU / ANE。"));return;}if(b.dataset.suppress)return;if(b.dataset.output)chooseOutput(b.dataset.output);if(b.dataset.input){const [to,i]=b.dataset.input.split(':');chooseInput(to,Number(i));}if(b.dataset.node&&!suppressNodeClick)select(b.dataset.node);});
 overlay.addEventListener('keydown',e=>{const b=e.target.closest('[data-node]');if(b&&e.key==='Enter'){e.preventDefault();focus(b.dataset.node);}});
 function activeFollowPackets(){return packets.filter(p=>p.wasActive||p.motion==='feedback');}
 function followTracking(active=activeFollowPackets()){
  if(!active.length)return [];
  if(!followedPacket&&followSeed)followedPacket=active.find(p=>p.edge?.from===followSeed)||null;
  if(followedPacket&&!active.includes(followedPacket)){
   // Stay on the chosen path through wire → operator → output wire. If a
   // frame skips a short transfer, search its downstream dependencies too.
   const boundary=followedPacket.node?.id||followedPacket.edge?.to,queue=[boundary],seen=new Set();let next=null;
   while(queue.length&&!next){
    const id=queue.shift();if(!id||seen.has(id))continue;seen.add(id);
    next=active.find(p=>p.node?.id===id)||active.find(p=>p.edge?.from===id);
    if(!next)for(const n of state.nodes)if(n.links.includes(id))queue.push(n.id);
   }
   followedPacket=next||active.find(p=>p.motion==='process')||active[0];
  }
  if(followedPacket)return [followedPacket];
  // Keep transfers in the same model section as its operators. Choosing work
  // globally first briefly jumped to decoder work between encoder operators.
  // Inputs already parked at a join (especially residuals) are not moving work.
  const candidates=active.filter(p=>p.motion!=='wire'||playbackTimeline?.time<p.travelEnd);
  const sectionOf=p=>M.demo.sections[p.node?.id||p.edge?.to],section=['encoder','masked','cross','readout'].find(section=>candidates.some(p=>sectionOf(p)===section));
  const local=candidates.filter(p=>!section||sectionOf(p)===section),work=local.filter(p=>p.motion==='process');
  return work.length?work:local;
 }
 function followFraming(tracking){
  if(!tracking.length)return null;
  const bounds=new T.Box3().setFromPoints(tracking.map(p=>p.motion==='process'?new T.Vector3(p.node.x,(p.node.y||0)+.7,p.node.z):p.p)),target=bounds.getCenter(new T.Vector3()),right=new T.Vector3().setFromMatrixColumn(camera.matrixWorld,0),up=new T.Vector3().setFromMatrixColumn(camera.matrixWorld,1);
  const offsets=tracking.map(p=>p.p.clone().sub(target)),xs=offsets.map(p=>p.dot(right)),ys=offsets.map(p=>p.dot(up));
  const profile=followProfiles[followView],span=Math.max(Math.max(...xs)-Math.min(...xs)+profile.padding,(Math.max(...ys)-Math.min(...ys)+profile.padding*.67)*host.clientWidth/host.clientHeight),framing=Math.max(profile.min,Math.min(stage().zoom,span*1.1/23));
  return {target,zoom:followZoom??framing,yaw:profile.yaw,pitch:profile.pitch};
 }
 function frameFollowing(tracking,delta=100){
  if(followFlight)return;const frame=followFraming(tracking);if(!frame)return;
  const ease=1-Math.exp(-Math.min(delta,100)/300);
  cameraTarget.lerp(frame.target,ease);zoom+=(frame.zoom-zoom)*ease;yaw+=(frame.yaw-yaw)*ease;pitch+=(frame.pitch-pitch)*ease;
 }
 function stopFollowFlight(){cancelAnimationFrame(followFlightFrame);followFlightFrame=0;followFlight=null;}
 function glideToFollow(){
  stopFollowFlight();
  const flight=followFlight={from:cameraTarget.clone(),zoom,yaw,pitch,start:performance.now()},duration=matchMedia('(prefers-reduced-motion:reduce)').matches?100:650;
  function tick(now){
   if(followFlight!==flight)return;
   if(!followData||focusId||!stage().floating){stopFollowFlight();return;}
   const active=followTracking(),frame=followFraming(active.length?active:followSeed?[{motion:'wire',p:point(followSeed)}]:[]);
   if(!frame){stopFollowFlight();return;}
   // This is a camera gesture: it keeps moving while computation is paused.
   const t=Math.min(1,(now-flight.start)/duration),ease=t*t*(3-2*t);
   cameraTarget.copy(flight.from).lerp(frame.target,ease);zoom=followZoom??flight.zoom+(frame.zoom-flight.zoom)*ease;yaw=flight.yaw+(frame.yaw-flight.yaw)*ease;pitch=flight.pitch+(frame.pitch-flight.pitch)*ease;
   render();if(t<1)followFlightFrame=requestAnimationFrame(tick);else stopFollowFlight();
  }
  followFlightFrame=requestAnimationFrame(tick);
 }
 function cycleFollow(){
  if(!stage().floating)return;if(focusId)leaveFocus();setFollow(true);
  const active=activeFollowPackets();
  if(active.length){
   const current=followTracking(active)[0];followedPacket=active[(active.indexOf(current)+1)%active.length];followSeed=null;
  }else{
   const sources=stage().sources;followSeed=sources[(sources.indexOf(followSeed)+1)%sources.length];
  }
  glideToFollow();
 }
 function setFollow(enabled){followData=!!enabled;if(!followData)stopFollowFlight();$('#demo-follow').setAttribute('aria-pressed',String(followData));if(followData){cameraPan=false;$('#demo-pan').setAttribute('aria-pressed','false');}}
 const wantsPan=e=>!focusId&&(cameraPan||e.shiftKey||e.button===1||e.button===2);
 function dragCamera(e){
  if(stage().floating&&!focusId)setFollow(false);
  const pan=wantsPan(e);
  drag={kind:'camera',x:e.clientX,y:e.clientY,yaw,pitch,moved:false,pan,target:cameraTarget.clone(),right:new T.Vector3().setFromMatrixColumn(camera.matrixWorld,0),up:new T.Vector3().setFromMatrixColumn(camera.matrixWorld,1),units:23*zoom/host.clientWidth};
 }
 for(const target of [host,overlay])target.addEventListener('contextmenu',e=>{if(!focusId)e.preventDefault();});
 $('#demo-follow').onclick=()=>setFollow(!followData);
 $('#demo-follow-next').onclick=cycleFollow;
 $('#demo-follow-view').onchange=e=>{if(!followProfiles[e.target.value])return;followView=e.target.value;followZoom=null;setFollow(true);};
 $('#demo-pan').onclick=()=>{setFollow(false);cameraPan=!cameraPan;$('#demo-pan').setAttribute('aria-pressed',String(cameraPan));if(cameraPan){pending=selectedWire=placing=actionNode=null;clearPreview();clearPlacement();labels();renderShelf();}};
 overlay.addEventListener('pointerdown',e=>{if(transitioning)return;if(busy||wantsPan(e)){dragCamera(e);return;}const b=e.target.closest('button');if(!b)return;if(stage().demonstration){dragCamera(e);if(!drag.pan)drag.inspect=b.dataset.node;return;}if(b.dataset.assign){selected=b.dataset.assign;drag={kind:'device',id:b.dataset.assign,x:e.clientX,y:e.clientY,moved:false};return;}if(b.dataset.output){clearConnectionError();drag={kind:'wire',id:b.dataset.output,el:b,x:e.clientX,y:e.clientY,moved:false};}if(b.dataset.node){if(focusId){drag={kind:'camera',x:e.clientX,y:e.clientY,yaw,pitch,moved:false};return;}const n=state.nodes.find(n=>n.id===b.dataset.node),p=worldPoint(e.clientX,e.clientY);selected=n.id;actionNode=null;drag={kind:'node',id:n.id,x:e.clientX,y:e.clientY,moved:false,old:clone(state),offset:p?[n.x-p.x,n.z-p.z]:[0,0]};renderShelf();}});
 host.addEventListener('pointerdown',e=>{if(transitioning)return;if(busy||wantsPan(e)){dragCamera(e);return;}if(stage().floating&&!focusId){dragCamera(e);ray.setFromCamera(new T.Vector2((e.clientX-host.getBoundingClientRect().left)/host.clientWidth*2-1,-((e.clientY-host.getBoundingClientRect().top)/host.clientHeight)*2+1),camera);let hit=ray.intersectObjects(assembly.children,true)[0]?.object;while(hit&&!hit.userData.node)hit=hit.parent;if(!drag.pan)drag.inspect=hit?.userData.node;return;}if(placing){const p=worldPoint(e.clientX,e.clientY);if(p)placeModule(p.x,p.z);return;}if(selectedWire||actionNode){selectedWire=actionNode=null;labels();}if(focusId){drag={kind:'camera',x:e.clientX,y:e.clientY,yaw,pitch,moved:false};return;}const p=worldPoint(e.clientX,e.clientY),hit=ray.intersectObjects(assembly.children,true)[0];if(hit){let object=hit.object;while(object&&!object.userData.node)object=object.parent;if(object?.userData.node){if(stage().demonstration){dragCamera(e);drag.inspect=object.userData.node;return;}const n=state.nodes.find(n=>n.id===object.userData.node);selected=n.id;actionNode=null;drag={kind:'node',id:n.id,x:e.clientX,y:e.clientY,moved:false,old:clone(state),offset:p?[n.x-p.x,n.z-p.z]:[0,0]};renderShelf();return;}}drag={kind:'camera',x:e.clientX,y:e.clientY,yaw,pitch,moved:false};if(pending){pending=null;clearConnectionError();labels();clearPreview();say(I18n.t("已取消连线。"));}});
 $('#module-palette').addEventListener('pointerdown',e=>{const b=e.target.closest('[data-add-module]');if(!b||b.dataset.addDisabled==='true'||busy||focusId)return;e.preventDefault();chooseModule(b.dataset.addModule);drag={kind:'palette',id:placing,x:e.clientX,y:e.clientY,moved:false};});
 $('#module-shelf').addEventListener('toggle',()=>{const shelf=$('#module-shelf');shelf.querySelector('summary').title=shelf.hasAttribute('open')?I18n.t('收起组件架'):I18n.t('展开组件架');render();});
 document.addEventListener('pointermove',e=>{if(placing)previewPlacement(e.clientX,e.clientY);if(!drag||(busy&&drag.kind!=='camera'))return;if(Math.hypot(e.clientX-drag.x,e.clientY-drag.y)>5)drag.moved=true;if(!drag.moved)return;if(drag.kind==='wire'){pending=drag.id;previewAt(e.clientX,e.clientY);}if(drag.kind==='node'){const p=worldPoint(e.clientX,e.clientY),n=state.nodes.find(n=>n.id===drag.id);if(p){const b=M.bounds(stage());n.x=Math.max(b.left,Math.min(b.right,p.x+drag.offset[0]));n.z=Math.max(b.back,Math.min(b.front,p.z+drag.offset[1]));rebuild();}}if(drag.kind==='camera'){const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(drag.pan){cameraTarget.copy(drag.target).addScaledVector(drag.right,-dx*drag.units).addScaledVector(drag.up,dy*drag.units);}else{yaw=stage().floating?drag.yaw-dx*.005:Math.max(-.65,Math.min(.65,drag.yaw-dx*.003));pitch=Math.max(stage().floating?-1.4:.4,Math.min(stage().floating?1.4:1.35,drag.pitch+dy*.004));}render();}});
 document.addEventListener('pointerup',e=>{if(!drag)return;const d=drag;drag=null;if(d.inspect&&!d.moved&&!busy){nodeTap(d.inspect,e);return;}if(busy&&d.kind!=='camera')return;if(d.kind==='palette'){if(d.moved){suppressPaletteClick=true;setTimeout(()=>suppressPaletteClick=false,0);const rect=host.getBoundingClientRect();if(e.clientX>=rect.left&&e.clientX<=rect.right&&e.clientY>=rect.top&&e.clientY<=rect.bottom){const p=worldPoint(e.clientX,e.clientY);if(p)placeModule(p.x,p.z);}placing=null;clearPlacement();labels();}return;}if(d.kind==='device'){const target=document.elementFromPoint(e.clientX,e.clientY)?.closest('[data-device]');if(target)assignDevice(d.id,target.dataset.device);return;}if(d.kind==='wire'&&d.moved){d.el.dataset.suppress='true';setTimeout(()=>delete d.el.dataset.suppress,0);const target=document.elementFromPoint(e.clientX,e.clientY)?.closest('[data-input]');clearPreview();if(target){const [to,i]=target.dataset.input.split(':');link(d.id,to,Number(i));}else{pending=d.id;labels();say(I18n.t("继续点一个方形输入口，完成这条线。"));}}if(d.kind==='node'&&d.moved){suppressNodeClick=true;setTimeout(()=>suppressNodeClick=false,0);history.push(d.old);save();refresh();labels();}if(d.kind==='node'&&!d.moved){nodeTap(d.id,e);labels();}});
 document.addEventListener('pointercancel',()=>{drag=null;pending=placing=null;clearConnectionError();clearPreview();clearPlacement();labels();});
 function viewPlanePoint(x,y){const rect=host.getBoundingClientRect();ray.setFromCamera(new T.Vector2((x-rect.left)/rect.width*2-1,-((y-rect.top)/rect.height)*2+1),camera);return ray.ray.intersectPlane(new T.Plane().setFromNormalAndCoplanarPoint(camera.getWorldDirection(new T.Vector3()),cameraTarget),new T.Vector3());}
 function zoomCamera(e){e.preventDefault();if(transitioning)return;if(stage().floating&&!focusId)setFollow(false);const anchor=stage().floating&&!focusId?viewPlanePoint(e.clientX,e.clientY):null;zoom=Math.max(focusId?.13:stage().floating?.18:.45,Math.min(focusId?.7:overviewZoomLimit()*2,zoom+e.deltaY*(focusId?.00028:.0008)));render();if(anchor){const after=viewPlanePoint(e.clientX,e.clientY);if(after){cameraTarget.add(anchor.sub(after));render();}}}
 $('#camera-zoom').addEventListener('input',e=>{if(focusId||transitioning)return;const amount=Math.max(0,Math.min(100,Number(e.target.value)))/100,minScale=1/(overviewZoomLimit()*2);zoom=1/(minScale+(1/minimumZoom()-minScale)*amount);if(stage().floating&&followData)followZoom=zoom;render();});
 host.addEventListener('wheel',zoomCamera,{passive:false});overlay.addEventListener('wheel',zoomCamera,{passive:false});
 document.addEventListener('keydown',e=>{if($('#settings').open||$('#welcome').open||$('#level-menu').open||$('#level-complete').open||$('#about').open||/^(INPUT|TEXTAREA|SELECT)$/.test(e.target?.tagName))return;if(e.key==='Escape'&&fusionEffect){e.preventDefault();finishFusion(true);return;}if((e.key==='Delete'||e.key==='Backspace')&&selectedWire&&!busy){e.preventDefault();deleteWire();return;}if((e.key==='Delete'||e.key==='Backspace')&&actionNode&&!busy&&!focusId){e.preventDefault();removeModule(actionNode);return;}if(e.key==='Escape'){clearConnectionError();if(focusId){e.preventDefault();leaveFocus();}else if(!$('#run-report').hidden){e.preventDefault();closeReport(true);}else{pending=placing=selectedWire=actionNode=null;fault=null;clearPreview();clearPlacement();labels();}}});
 document.addEventListener('pointercancel',()=>{if(drag?.kind==='palette'){placing=null;clearPlacement();}drag=null;render();});
 $('#card-close').onclick=leaveFocus;$('#card-return').onclick=leaveFocus;
 $('#module-card').addEventListener('cancel',e=>{e.preventDefault();leaveFocus();});
 $('#module-card').addEventListener('click',e=>{if(e.target!==$('#module-card'))return;const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)leaveFocus();});
 $('#card-camera-reset').onclick=()=>$('#camera-reset').click();
 $('#demo-views').onclick=e=>{const view=stage().demoViews?.find(v=>v.key===e.target.closest('[data-demo-view]')?.dataset.demoView);if(!view)return;setFollow(false);if(focusId)leaveFocus();cameraTarget.set(view.x,view.y||0,view.z);zoom=view.zoom;yaw=.05;pitch=.8;render();};
 $('#camera-reset').onclick=()=>{if(stage().floating&&!focusId)setFollow(false);if(focusId){const n=shelfFocus?{x:0,z:0}:state.nodes.find(n=>n.id===focusId);transition(new T.Vector3(n.x,(n.y||0)+.72,n.z),Math.max(4.1,3.5*host.clientWidth/host.clientHeight)/23,.32,.92);}else{yaw=.05;pitch=.8;zoom=stage().zoom;cameraTarget.set(0,0,0);render();}};
 function beforeStage(){clearConnectionError();if(focusId)leaveFocus();cancelAnimationFrame(cameraFrame);transitioning=false;host.setAttribute('aria-busy','false');drag=lastTap=null;clearPreview();clearPlacement();clearPackets();elements=[];overlay.replaceChildren();}
 function setStage(){followSeed=null;followZoom=null;setFollow(!!stage().floating);cameraPan=false;$('#demo-pan').setAttribute('aria-pressed','false');art.surroundings(samples(),stage(),scenePalette());yaw=.05;pitch=.8;zoom=stage().zoom;cameraTarget.set(0,0,0);rebuild();}
 function relocalize(){clearConnectionError();art.surroundings(samples(),stage(),scenePalette());rebuild();packets.forEach(p=>{p.el.querySelector('.packet-title').textContent=p.motion==='feedback'?I18n.t('接回输入'):packetIdentity(p.el.dataset.from);if(p.sequenceProjection)p.updateEquation(p.column,p.projectionRow);if(p.motion==='wire'&&(definition(p.el.dataset.from)?.type==='pick'||kindOf(p.el.dataset.from)==='feedback'))p.cells.textContent=M.readTokens(p.values);if(p.appending){p.cells.textContent=M.readTokens(p.input);p.after.textContent=M.readTokens(p.output);}if(p.selection&&p.progress!=null){p.after.textContent=M.readTokens(p.output);p.updateSelection(art.selectionFrame(p.progress,p.input.length/p.output.length,p.output.length));p.outputLabels.forEach(({el},j)=>el.textContent=M.math.vocab[p.output[j]]);}if(p.normalizing&&p.progress!=null)p.updateRMS(normalizationFrame(p.progress,p.centering));if(p.tokenOperation&&p.progress!=null)p.updateToken(art.tokenFrame(p.progress,p.tokenOperation));if(p.motion==='process'&&p.progress!=null)art.processing(p.node.id,p.progress,p.input,p.output,p.color,{sequence:!!stage().sequence,detailed:p.detailed,paired:p.paired,other:p.other,words:p.words?.(),valueWords:p.valueWords?.()});});render();}
 new ResizeObserver(render).observe(host);world={relocalize,labels,rebuild,animate,feedback,finishDemo,fuse,skipFusion:()=>finishFusion(true),clearPackets,showConnectionError,showPlacementError,clearConnectionError,clearPreview,clearPlacement,previewAt,previewPlacement,focus,focusShelf,focused:()=>focusId,beforeStage,setStage,syncShelf:()=>syncPlacementGuide(host.clientWidth,host.clientHeight)};$('#loading').hidden=true;rebuild();refresh();
}catch(e){$('#loading').innerHTML=I18n.t("<strong>3D 台面没有加载成功，请刷新重试。</strong>");say(I18n.t("页面尚未开始运行。"),'error');console.error(e);}
