// Curated glossary over the page copy. Specialised terms get a dashed
// underline, and hovering or focusing one opens a short popover definition.
export const glossaryEntries = {
 'FFN': {
  zh: '前馈网络（Feed-Forward Network）：同一层里对每个 token 单独做的一组线性变换与激活，不同位置之间不混合。',
  en: 'Feed-Forward Network — the part of a layer that transforms each token on its own, without mixing positions.',
  name: {zh: '前馈网络', en: 'Feed-Forward Network'},
  alias: ['前馈网络', 'feed-forward network', 'feedforward network', 'FFNs'],
  seeAlso: 'relu-ffn'
 },
 'RMSNorm': {
  zh: 'RMSNorm（均方根归一化）：把一个 token 的特征按自己的均方根缩放，不减均值，也不把各特征拉成同一个数。',
  en: 'RMSNorm — scales a token’s features by its own root mean square, without subtracting the mean.',
  name: {zh: '均方根归一化', en: 'Root Mean Square Normalization'},
  alias: ['均方根归一化'],
  seeAlso: 'norm'
 },
 'LayerNorm': {
  zh: 'LayerNorm（层归一化）：把同一位置的特征减均值、除标准差，再按学到的比例与偏移还原。',
  en: 'LayerNorm — subtracts the mean and divides by the standard deviation across one position’s features.',
  name: {zh: '层归一化', en: 'Layer Normalization'},
  alias: ['层归一化'],
  seeAlso: 'layernorm2'
 },
 'Pre-LN': {
  zh: 'Pre-LN（前置归一化）：先归一化再进子层；和原版先加残差、后归一化的 Post-LN 顺序正好相反。',
  en: 'Pre-LN — normalizes before the sublayer, the opposite of the original Post-LN order.',
  name: {zh: '前置归一化', en: 'Pre-normalization'},
  alias: ['前置归一化', 'post-ln', 'postln']
 },
 'SwiGLU': {
  zh: 'SwiGLU：FFN 的一种写法。同一份输入分成内容与门两路，门路过 SiLU 后与内容逐元素相乘，再降维回到原宽度。',
  en: 'SwiGLU — an FFN variant that splits the input into a content branch and a gated branch, then multiplies them.',
  name: {zh: '门控 FFN', en: 'Gated FFN'},
  alias: [],
  seeAlso: 'swiglu-ffn'
 },
 'SiLU': {
  zh: 'SiLU：一种激活函数，对每个数单独算 x / (1 + e⁻ˣ)；负数被压近 0，正数接近原值，维度不变。',
  en: 'SiLU — an activation applied to each number as x / (1 + e⁻ˣ); it keeps the dimension.',
  name: {zh: '激活函数', en: 'Activation function'},
  alias: [],
  seeAlso: 'silu'
 },
 'ReLU': {
  zh: 'ReLU：最常见的激活函数，逐个取 max(0, x)：负数变 0，正数保持原值，维度不变。',
  en: 'ReLU — the common activation max(0, x): negatives become zero and positives stay unchanged.',
  name: {zh: '激活函数', en: 'Activation function'},
  alias: [],
  seeAlso: 'relu'
 },
 'softmax': {
  zh: 'softmax：把一组分数变成加起来为 1 的比例；分数最高的项拿到最大的权重。',
  en: 'softmax — turns a list of scores into weights that sum to one; the highest score gets the largest share.',
  name: {zh: '归一化指数', en: 'Softmax'},
  alias: [],
  seeAlso: 'softmax'
 },
 'token': {
  zh: 'token（词元）：模型一次处理的一个单位，通常是一个词或它的片段；每个 token 带着一组特征走完全部计算。',
  en: 'Token — one unit the model processes at a time, usually a word or a piece of one, carrying its own features through the whole layer.',
  name: {zh: '词元', en: 'Token'},
  alias: ['tokens', '词元'],
  seeAlso: null
 },
 'embedding': {
  zh: 'embedding（词嵌入）：把词或词元编号查表换成一组数，这组数就是它进入模型时带的特征。',
  en: 'Embedding — the feature vector a token starts with, usually looked up from a table.',
  name: {zh: '词嵌入', en: 'Embedding'},
  alias: ['词嵌入', 'embeddings'],
  seeAlso: 'embedding'
 },
 'Q / K / V': {
  zh: 'Q / K / V：查询、键与内容三种投影。Q 拿去和 K 匹配算出权重，再用这组权重去加权 V 的内容。',
  en: 'Q / K / V — the query, key and value projections. Q matches against K to score, and those scores weight V.',
  name: {zh: '查询 / 键 / 值', en: 'Query / Key / Value'},
  alias: ['Q/K/V', 'Q、K、V', 'K / V', 'K/V'],
  seeAlso: 'q'
 },
 'attention head': {
  zh: 'attention head（注意力头）：注意力里独立算一遍的一路。多个头各读各的特征，最后把结果拼起来。',
  en: 'Attention head — one independent pass through the attention calculation; several heads read different features and are concatenated.',
  name: {zh: '注意力头', en: 'Attention head'},
  alias: ['attention heads', 'multi-head attention', 'single-head attention', '注意力头', '多头注意力', '单头注意力'],
  seeAlso: 'head-a'
 },
 'residual': {
  zh: 'residual（残差）：把支路算出的修改量加回原来的输入，让原信息有一条不被改写的直路。',
  en: 'Residual — adds a branch’s update back to the input, so the original signal keeps a direct path.',
  name: {zh: '残差', en: 'Residual'},
  alias: ['残差', 'residuals'],
  seeAlso: 'add'
 },
 'projection': {
  zh: 'projection（投影）：用一套权重把输入向量重新组合成另一个维度；换个角度看，就是换一组坐标来描述同一份信息。',
  en: 'Projection — reweights the input into another set of features, a new coordinate view of the same information.',
  name: {zh: '投影', en: 'Projection'},
  alias: ['投影', 'projections', 'project', 'projects', 'projected', 'projecting'],
  seeAlso: 'project'
 },
 'activation': {
  zh: 'activation（激活）：跟在投影后面的非线性函数，让同样的输入不再总是得到同样的输出。',
  en: 'Activation — the non-linear step after a projection, so the layer is not a single linear change.',
  name: {zh: '激活', en: 'Activation'},
  alias: ['激活', 'activations'],
  seeAlso: 'silu'
 },
 'logits': {
  zh: 'logits（词表分数）：模型给词表里每个候选词打的分，取最高分的词就是这一步的预测。',
  en: 'Logits — the score given to every candidate word; the highest one becomes the prediction.',
  name: {zh: '词表分数', en: 'Logits'},
  alias: ['词表分数', '词表打分', 'vocabulary scoring', 'vocabulary scores'],
  seeAlso: 'lmhead'
 },
 'RoPE': {
  zh: 'RoPE：旋转位置编码。按位置把 Q 和 K 的两维旋转一个角度，让打分的点积自带相对位置信息。',
  en: 'RoPE — rotary position embedding: Q and K are rotated by a position-dependent angle so their dot product carries relative position.',
  name: {zh: '旋转位置编码', en: 'Rotary Position Embedding'},
  alias: ['rope'],
  seeAlso: 'ropeq'
 },
 'KV cache': {
  zh: 'KV cache：把已经算过的 K、V 存起来。逐词生成时复用它们，就不用每一步都重算整段历史。',
  en: 'KV cache — stores the keys and values already computed so each new token reuses them instead of recomputing the history.',
  name: {zh: 'KV 缓存', en: 'KV Cache'},
  alias: ['KV 缓存', 'kv cache', 'KV'],
  seeAlso: null
 },
 'autoregressive': {
  zh: 'autoregressive（自回归）：一次只预测一个词，再把它接到自己生成的序列后面，接着预测下一个。',
  en: 'Autoregressive — predicts one token at a time and appends it to its own output before predicting the next.',
  name: {zh: '自回归', en: 'Autoregressive'},
  alias: ['自回归'],
  seeAlso: null
 },
 'ANE': {
  zh: 'ANE：Apple Neural Engine，苹果芯片上的神经网络加速器；本游戏把它当作一种可分配的算力来调度。',
  en: 'ANE — the Apple Neural Engine, the neural accelerator on Apple silicon; the game treats it as a schedulable compute target.',
  name: {zh: '神经引擎', en: 'Neural Engine'},
  alias: [],
  seeAlso: null
 },
 '归一化': {
  zh: '归一化：把一组数调整到统一的尺度，好让后面的计算不必被数值大小牵着走。',
  en: 'Normalization — rescales a group of numbers to a common scale so later steps are not driven by raw magnitude.',
  name: {zh: '归一化', en: 'Normalization'},
  alias: ['normalization', 'normalize', 'normalized', 'normalizes', 'normalizing', 'unnormalized'],
  seeAlso: 'layernorm2'
 }
};

const escapeRegExp=value=>value.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
const SKIP_TAGS=new Set(['CODE','PRE','KBD','SAMP','SCRIPT','STYLE','TEXTAREA','OPTION','SELECT','SVG','MATH']);
const SKIP_CLASS=/(?:^|[\s-])(matrix|vector|cell|coordinates|scores|popover|term)(?:$|[\s-])/;
const STATIC_SOURCES='#device-docks button>span,#about p,#about h2,#about .eyebrow';
const POPOVER_ID='glossary-popover';

export const glossary=(()=>{
 const entries=Object.entries(glossaryEntries).map(([term,entry])=>({term,...entry}));
 const patterns=entries.map(entry=>{
  const aliases=[...new Set([entry.term,...(entry.alias||[])])].sort((a,b)=>b.length-a.length).map(escapeRegExp);
  return new RegExp('(?<![A-Za-z0-9_])(?:'+aliases.join('|')+')(?![A-Za-z0-9_])','gi');
 });
 const byTerm=new Map(entries.map(entry=>[entry.term,entry]));
 const staticTexts=new Map();
 let activeTerm=null;

 const language=()=>globalThis.I18n?.language?.()==='en'?'en':'zh';
 const moduleLabel=kind=>{
  const name=globalThis.NeuralPrologue?.definitions?.[kind]?.name;
  return typeof name==='string'&&name?name:null;
 };
 const popover=()=>document.getElementById(POPOVER_ID);
 // A modal dialog paints in the top layer, above every page element, so the
 // popover moves into the open dialog to stay readable while it is shown.
 const popoverHost=term=>{
  const dialog=term?.closest?.('dialog');
  return dialog&&dialog.open?dialog:document.body;
 };
 const placePopover=(box,host)=>{if(box.parentNode!==host)host.appendChild(box);};

 function findMatches(text){
  const found=[];
  entries.forEach((entry,index)=>{
   const pattern=patterns[index];pattern.lastIndex=0;
   let match;
   while((match=pattern.exec(text))){
    if(match[0])found.push({start:match.index,end:match.index+match[0].length,entry});
    else pattern.lastIndex++;
   }
  });
  found.sort((a,b)=>a.start-b.start||(b.end-b.start)-(a.end-a.start));
  const picked=[];let end=-1;
  for(const match of found){if(match.start>=end){picked.push(match);end=match.end;}}
  return picked;
 }
 function termSpan(entry,label){
  const span=document.createElement('span');
  span.className='glossary-term';
  span.dataset.term=entry.term;
  span.setAttribute('tabindex','0');
  span.setAttribute('aria-describedby',POPOVER_ID);
  span.textContent=label;
  return span;
 }
 function appendMarked(parent,text){
  const matches=findMatches(text);
  if(!matches.length){parent.appendChild(document.createTextNode(text));return;}
  let cursor=0;
  for(const match of matches){
   if(match.start>cursor)parent.appendChild(document.createTextNode(text.slice(cursor,match.start)));
   parent.appendChild(termSpan(match.entry,text.slice(match.start,match.end)));
   cursor=match.end;
  }
  if(cursor<text.length)parent.appendChild(document.createTextNode(text.slice(cursor)));
 }
 function skipElement(element){
  if(SKIP_TAGS.has(element.tagName))return true;
  if(element.id===POPOVER_ID)return true;
  const className=element.getAttribute?.('class')||'';
  return typeof className==='string'&&SKIP_CLASS.test(className);
 }
 function markTextNode(node){
  const text=node.nodeValue||'';
  if(!text.trim()||!/[\p{L}]/u.test(text))return;
  if(!findMatches(text).length)return;
  const fragment=document.createDocumentFragment();
  appendMarked(fragment,text);
  node.parentNode?.replaceChild(fragment,node);
 }
 function walk(node){
  for(const child of [...(node.childNodes||[])]){
   if(child.nodeType===3)markTextNode(child);
   else if(child.nodeType===1){
    if(staticTexts.has(child)){renderStatic(child);continue;}
    if(!skipElement(child))walk(child);
   }
  }
 }
 function scan(root){
  const target=root||document.body;
  if(!target)return;
  if(target.nodeType===3)markTextNode(target);
  else if(!skipElement(target))walk(target);
 }
 // Static prose keeps one Chinese source; the glossary re-renders it in the
 // active language so the marked spans survive a language switch.
 function captureStatic(){
  for(const element of document.querySelectorAll(STATIC_SOURCES)){
   if(element.querySelector('*'))continue;
   const raw=element.textContent||'',source=raw.trim();
   if(!source)continue;
   const english=globalThis.AtelierEnglish?.[source]||'';
   if(!findMatches(source).length&&!findMatches(english).length)continue;
   staticTexts.set(element,{raw,source});
  }
 }
 function renderStatic(element){
  const text=staticTexts.get(element);
  if(!text)return;
  const translated=text.raw.replace(text.source,globalThis.I18n.t(text.source));
  const fragment=document.createDocumentFragment();
  appendMarked(fragment,translated);
  element.textContent='';
  element.appendChild(fragment);
 }
 function renderAllStatic(){for(const element of [...staticTexts.keys()])renderStatic(element);}
 function hide(){
  const box=popover();
  if(box)box.hidden=true;
  if(box)placePopover(box,document.body);
  activeTerm?.removeAttribute('aria-describedby');
  activeTerm=null;
 }
 function show(term){
  const entry=byTerm.get(term.dataset.term);
  const box=popover();
  if(!entry||!box)return;
  hide();
  const title=document.createElement('b');title.className='glossary-popover-term';title.textContent=entry.term;
  const name=document.createElement('span');name.className='glossary-popover-name';
  name.textContent=language()==='en'?entry.name.zh:entry.name.en;
  const definition=document.createElement('p');definition.className='glossary-popover-definition';
  definition.textContent=language()==='en'?entry.en:entry.zh;
  const fragment=document.createDocumentFragment();
  fragment.append(title,name,definition);
  const label=entry.seeAlso?moduleLabel(entry.seeAlso):null;
  if(label){
   const hint=document.createElement('small');hint.className='glossary-popover-hint';
   hint.textContent=language()==='en'?`More detail in the “${label}” module card.`:`更多细节见「${label}」模块卡。`;
   fragment.appendChild(hint);
  }
  box.textContent='';
  box.appendChild(fragment);
  box.hidden=false;
  placePopover(box,popoverHost(term));
  term.setAttribute('aria-describedby',POPOVER_ID);
  activeTerm=term;
  position(box,term);
 }
 function position(box,term){
  const rect=term.getBoundingClientRect(),size=box.getBoundingClientRect();
  const width=size.width||box.offsetWidth||260,height=size.height||box.offsetHeight||90;
  const viewWidth=window.innerWidth||document.documentElement.clientWidth||1280;
  const viewHeight=window.innerHeight||document.documentElement.clientHeight||800;
  const gap=8,above=rect.top-height-gap,below=rect.bottom+gap;
  const top=above>=gap?above:(below+height<=viewHeight-gap?below:Math.max(gap,above));
  const left=Math.max(gap,Math.min(rect.left+rect.width/2-width/2,viewWidth-width-gap));
  box.style.left=left+'px';
  box.style.top=top+'px';
 }
 function termOf(target){
  return target?.closest?.('.glossary-term')||null;
 }
 document.addEventListener('pointerover',event=>{const term=termOf(event.target);if(term)show(term);});
 document.addEventListener('pointerout',event=>{const term=termOf(event.target);if(term&&!term.contains(event.relatedTarget))hide();});
 document.addEventListener('focusin',event=>{const term=termOf(event.target);if(term)show(term);});
 document.addEventListener('focusout',event=>{if(termOf(event.target))hide();});
 document.addEventListener('pointerdown',hide,true);
 document.addEventListener('scroll',hide,true);
 document.addEventListener('keydown',event=>{if(event.key==='Escape')hide();});
 document.addEventListener('close',hide,true);
 window.addEventListener?.('resize',hide);
 globalThis.I18n?.onChange?.(()=>renderAllStatic());
 captureStatic();
 return {entries,scan,refresh(){scan(document.body);},show,hide,byTerm};
})();
