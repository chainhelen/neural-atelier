export function createProgress(M,storage){
 const legacy='neural-atelier-prologue-v1',metaKey='neural-atelier-chapters-v1';
 const read=key=>{try{return JSON.parse(storage.getItem(key));}catch{return null;}};
 const write=(key,value)=>{try{storage.setItem(key,JSON.stringify(value));return true;}catch{return false;}};
 const known=id=>Object.hasOwn(M.levels,id)||Object.hasOwn(M.levelRedirects||{},id);
 const saved=read(metaKey),savedCurrent=M.resolveLevelId?.(saved?.current)||(Object.hasOwn(M.levels,saved?.current)?saved.current:null);
 // Retired completion badges stay recoverable but do not count toward routes.
 // Only navigation redirects; never turn an old board into the next lesson's answer.
 const meta={current:savedCurrent||(read(legacy)?'ffn':'wire'),completed:(Array.isArray(saved?.completed)?saved.completed:[]).filter(known),practiceRuns:{}};
 for(const [id,run]of Object.entries(saved?.practiceRuns||{}))if(M.levels[id]?.practice&&run&&typeof run.finished==='boolean'&&(!run.finished||typeof run.circuit==='string'))meta.practiceRuns[id]={finished:run.finished,...(run.finished?{circuit:run.circuit}:{})};
 const generation=value=>Number.isSafeInteger(value)&&value>=0?value:0;
 meta.generation=generation(saved?.generation);meta.nextGeneration=Math.max(meta.generation,generation(saved?.nextGeneration));
 const backup=saved?.restartBackup;
 if(backup&&generation(backup.generation)===backup.generation&&Array.isArray(backup.completed)&&backup.practiceRuns&&typeof backup.practiceRuns==='object')meta.restartBackup={generation:backup.generation,completed:backup.completed.filter(known),practiceRuns:backup.practiceRuns};
 const key=id=>{const base=id==='ffn'?legacy:`neural-atelier-level-${id}-v1`;return meta.generation?base+'-run-'+meta.generation:base;};
 const completedKey=id=>key(id)+'-completed';
 const placementKey=id=>key(id)+'-before-placement';
 const circuitKey=(s,positions=false)=>JSON.stringify([s.sink,[...s.nodes].sort((a,b)=>a.id.localeCompare(b.id)).map(n=>[n.id,M.nodeKind(n),n.links,...(positions?[n.x,n.z]:[])])]);
 function mergeProjections(s){
  if(s?.version!==1||!Array.isArray(s.nodes))return s;
  const old=s.nodes.filter(n=>M.nodeKind(n)==='up'&&M.validNodeId(n.id,'up'));
  if(!old.length)return s;
  // Reserve every existing ID before assigning aliases, including spare or disconnected copies.
  const used=new Set(s.nodes.map(n=>n?.id)),renames=new Map();
  for(const n of old){let id='project',i=2;while(used.has(id))id='project-'+i++;used.add(id);renames.set(n.id,id);}
  const rename=id=>renames.get(id)||id;
  return {...s,nodes:s.nodes.map(n=>n?{...n,...(renames.has(n.id)?{id:rename(n.id),kind:'project'}:{}),links:Array.isArray(n.links)?n.links.map(rename):n.links}:n),sink:rename(s.sink),palette:Array.isArray(s.palette)?s.palette.map(k=>k==='up'?'project':k):s.palette};
 }
 function upgradeReadout(s,id){
  const aliases=M.levels[id]?.saveAliases;
  if(!aliases||s?.version!==1||!Array.isArray(s.nodes))return s;
  const old=s.nodes.filter(n=>aliases[M.nodeKind(n)]&&M.validNodeId(n.id,M.nodeKind(n)));if(!old.length)return s;
  const used=new Set(s.nodes.filter(n=>!old.includes(n)).map(n=>n?.id)),renames=new Map();
  for(const n of old){const kind=aliases[M.nodeKind(n)];let name=kind,i=2;while(used.has(name))name=kind+'-'+i++;used.add(name);renames.set(n.id,name);}
  const rename=name=>renames.get(name)||name;
  return {...s,placementRevision:1,nodes:s.nodes.map(n=>n?{...n,...(renames.has(n.id)?{id:rename(n.id),kind:aliases[M.nodeKind(n)]}:{}),links:Array.isArray(n.links)?n.links.map(rename):n.links}:n),sink:rename(s.sink),palette:Array.isArray(s.palette)?s.palette.map(k=>aliases[k]||k):s.palette};
 }
 function validated(raw,id){
  const s=mergeProjections(upgradeReadout(raw,id)),l=M.levels[id];if(!l||s?.version!==1||!Array.isArray(s.nodes))return null;
  const stock=M.inventory(l),allowed=[...new Set([...Object.keys(stock),...Object.keys(l.legacyInventory||{})])];
  const valid=new Set([...l.sources,...s.nodes.map(n=>n?.id)]),counts={};
  if((l.buildable?s.nodes.length>M.moduleLimit(l):s.nodes.length!==l.nodes.length)||new Set(s.nodes.map(n=>n?.id)).size!==s.nodes.length||s.nodes.some(n=>{
   const kind=M.nodeKind(n);counts[kind]=(counts[kind]||0)+1;
   return !n||!allowed.includes(kind)||!M.validNodeId(n.id,kind)||(!l.buildable&&n.id!==kind)||counts[kind]>Math.max(stock[kind]||0,l.legacyInventory?.[kind]||0)||!Number.isFinite(n.x)||!Number.isFinite(n.z)||!Array.isArray(n.links)||n.links.length!==M.definitions[kind].ins.length||n.links.some(k=>k!==null&&!valid.has(k));
  })||(s.sink!==null&&!valid.has(s.sink)))return null;
  // Old saves keep their IDs and positions; only append newly available stock.
  s.palette=[...new Set((Array.isArray(s.palette)?s.palette:[]).filter(k=>k in stock))];
  s.palette.push(...Object.keys(stock).filter(k=>!s.palette.includes(k)));
  return {...s,levelId:id,...(l.engine?{settings:{...M.engineDefaults,...l.settings,...s.settings},nodes:s.nodes.map(n=>({...n,device:['cpu','gpu','ane'].includes(n.device)?n.device:'gpu'}))}:{})};
 }
 function solved(s,id){
  // Old saves have a lifetime completion badge but no per-attempt status.
  // Only restart a completed answer, never a partly edited circuit.
  return M.samplesFor(id).slice(0,M.levels[id].sampleCount).every(({v})=>{
   const ev=M.evaluate(s,v),expected=M.reference(v,id);
   return !ev.error&&ev.output?.length===expected.length&&ev.output.every((x,i)=>Number.isFinite(x)&&Math.abs(x-expected[i])<1e-9);
  });
 }
 function load(id,{resume=false}={}){
  if(M.levels[id]?.demonstration)return M.initial(id);
  const l=M.levels[id],raw=read(key(id)),upgraded=upgradeReadout(raw,id);
  // These lessons changed from bare projections to the previously assembled
  // ReLU FFN. Keep the exact old circuit before opening the revised exercise.
  if(raw?.version===1&&Array.isArray(raw.nodes)&&raw.nodes.some(n=>l.retiredCircuitKinds?.includes(M.nodeKind(n)))){
   const archive=key(id)+'-before-ffn-reuse',fresh=M.initial(id);
   if(!read(archive)&&!write(archive,raw))return raw;
   if(!write(key(id),fresh))return raw;
   meta.practiceRuns[id]={finished:false};write(metaKey,meta);return fresh;
  }
  // Keep an exact recovery copy before upgrading old two-feature modules.
  // Existing positions, wiring and completion badges are not reset.
  if(upgraded!==raw&&!read(key(id)+'-before-four-feature-readout')&&!write(key(id)+'-before-four-feature-readout',raw))return raw;
  let s=validated(upgraded,id)||M.initial(id);
  if(l.requirePlacement&&s.placementRevision!==1){
   // Only the former auto-filled, still-unwired starting set is retired.
   // Keep a recoverable copy, and never clear a connected or completed board.
   const prepared=l.migratePreparedStart&&!meta.completed.includes(id)&&s.sink===null&&s.nodes.length===l.nodes.length&&s.nodes.every(n=>l.nodes.includes(n.id)&&M.nodeKind(n)===n.id&&n.links.every(x=>x===null));
   if(prepared){if(!write(placementKey(id),s))return s;s=M.initial(id);}
   else s={...s,placementRevision:1};
   write(key(id),s);
  }
  if(!l.practice)return s;
  const run=meta.practiceRuns[id],finished=run?run.finished&&run.circuit===circuitKey(s):meta.completed.includes(id)&&solved(s,id);
  if(finished&&!resume){
   // Keep the only copy if browser storage cannot archive it.
   if(!write(completedKey(id),s))return s;const fresh=M.initial(id);
   meta.practiceRuns[id]={finished:false};write(key(id),fresh);write(metaKey,meta);return fresh;
  }
  if(resume&&finished){
   if(!run){meta.practiceRuns[id]={finished:true,circuit:circuitKey(s)};write(metaKey,meta);}
   return s;
  }
  if(!run||run.finished){meta.practiceRuns[id]={finished:false};write(metaKey,meta);}
  return s;
 }
 function save(state){
  const id=state.levelId||'ffn',run=meta.practiceRuns[id];
  if(M.levels[id]?.practice&&(!run||(run.finished&&run.circuit!==circuitKey(state)))){
   meta.practiceRuns[id]={finished:false};write(metaKey,meta);
  }
  return write(key(id),state);
 }
 const previous=id=>M.levels[id]?.practice?validated(read(completedKey(id)),id):M.levels[id]?.requirePlacement?validated(read(placementKey(id)),id):null;
 return {current:()=>meta.current,completed:id=>meta.completed.includes(id),load,
  save,
  resetCompletions(){
   const next={...meta,completed:[],practiceRuns:Object.fromEntries(Object.keys(meta.practiceRuns).map(id=>[id,{finished:false}]))};
   if(!write(metaKey,next))return false;
   meta.completed=next.completed;meta.practiceRuns=next.practiceRuns;return true;
  },
  restartAll(){
   // Switch all boards with one storage write. Older saves remain intact for
   // undo; the generation counter never rewinds or reuses a played board.
   const next={...meta,generation:meta.nextGeneration+1,nextGeneration:meta.nextGeneration+1,completed:[],practiceRuns:{},restartBackup:JSON.parse(JSON.stringify({generation:meta.generation,completed:meta.completed,practiceRuns:meta.practiceRuns}))};
   if(!write(metaKey,next))return false;
   Object.assign(meta,next);return true;
  },
  canUndoRestart:()=>!!meta.restartBackup,
  undoRestart(){
   if(!meta.restartBackup)return false;
   const next={...meta,...meta.restartBackup,restartBackup:null};
   if(!write(metaKey,next))return false;
   Object.assign(meta,next);return true;
  },
  canRestore(id,state){const s=previous(id);return !!s&&(!state||circuitKey(s,true)!==circuitKey(state,true));},
  restore(id){let s=previous(id);if(!s)return null;if(M.levels[id].practice)meta.practiceRuns[id]={finished:false};if(M.levels[id].requirePlacement)s={...s,placementRevision:1};write(metaKey,meta);write(key(id),s);return s;},
  select(id){meta.current=id;write(metaKey,meta);},
  complete(id,state){
   if(!meta.completed.includes(id))meta.completed.push(id);
   if(M.levels[id]?.practice){const s=validated(state||read(key(id)),id);if(s){save(s);write(completedKey(id),s);meta.practiceRuns[id]={finished:true,circuit:circuitKey(s)};}}
   write(metaKey,meta);
  }
 };
}
