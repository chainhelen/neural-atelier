const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
const ctx={console,navigator:{languages:['zh-CN']}};ctx.window=ctx;vm.createContext(ctx);for(const f of ['locales/en.js','i18n.js','music.js','shell.js','model.js','campaign.js','engine-model.js','original-curriculum.js','transformer-demo.js','cloud-curriculum.js','lesson-io.js'])vm.runInContext(fs.readFileSync(__dirname+'/'+f,'utf8'),ctx,{filename:f});
const M=ctx.NeuralPrologue;
const close=(a,b,tol=1e-9)=>a.length===b.length&&a.every((x,i)=>Number.isFinite(x)&&Math.abs(x-b[i])<=tol);
test('random arrangements fit the table, keep modules apart, and do not change the circuit',()=>{
 const rng=seed=>()=>((seed=(seed*1664525+1013904223)>>>0)/4294967296);
 for(const id of Object.keys(M.levels)){
  const s=M.solutions(id),links=JSON.stringify(s.nodes.map(n=>n.links)),a=JSON.stringify(M.arrange(s,rng(2)).nodes.map(n=>[n.x,n.z]));
  M.arrange(s,rng(71));if(s.nodes.length)assert.notEqual(JSON.stringify(s.nodes.map(n=>[n.x,n.z])),a);
  assert.equal(JSON.stringify(s.nodes.map(n=>n.links)),links);const b=M.bounds(M.levels[id]);
  for(const n of s.nodes){assert.ok(n.x>=b.left&&n.x<=b.right&&n.z>=b.back&&n.z<=b.front);for(const m of s.nodes)if(n!==m)assert.ok(Math.abs(n.x-m.x)>=2.65||Math.abs(n.z-m.z)>=2.9,id+' overlaps');}
 }
});
test('disconnect removes one selected branch and reconnect restores the computation',()=>{
 const s=M.solutions('gating');assert.ok(M.disconnect(s,'multiply',0));assert.deepEqual(Array.from(s.nodes.find(n=>n.id==='multiply').links),[null,'project']);assert.equal(s.sink,'multiply');assert.equal(M.disconnect(s,'multiply',0),false);
 assert.equal(M.connect(s,'silu','multiply',0),null);assert.ok(close(M.evaluate(s,[1,1]).output,M.reference([1,1],'gating')));
 assert.ok(M.disconnect(s,'sink',0));assert.equal(s.sink,null);assert.equal(s.nodes.find(n=>n.id==='multiply').links[0],'silu');
});
test('construction levels can be built from the shelf, reject invalid placement, and remove dependent wires',()=>{
 for(const id of ['workshop','ffn','attention','block']){
  const s=M.initial(id),solution=M.solutions(id);assert.equal(s.nodes.length,0);assert.equal(s.sink,null);
  const first=solution.nodes[0];assert.equal(M.addNode(s,first.id,999,0),false);assert.equal(M.addNode(s,'unknown',0,0),false);
  for(const n of solution.nodes)assert.equal(M.addNode(s,n.id,n.x,n.z),true);assert.equal(M.addNode(s,first.id,first.x,first.z),false);
  for(const n of solution.nodes)for(const [port,from]of n.links.entries())assert.equal(M.connect(s,from,n.id,port),null);
  assert.equal(M.connect(s,solution.sink,'sink',0),null);const input=M.samplesFor(id)[0].v;assert.ok(close(M.evaluate(s,input).output,M.reference(input,id)));
  assert.ok(M.removeNode(s,solution.sink));assert.equal(s.sink,null);assert.ok(s.nodes.every(n=>!n.links.includes(solution.sink)));
  const removed=solution.nodes.find(n=>n.id===solution.sink);if(s.nodes.length)assert.equal(M.addNode(s,removed.id,s.nodes[0].x,s.nodes[0].z),false,'overlap must be rejected');assert.equal(M.addNode(s,removed.id,removed.x,removed.z),true);
 }
});
for(const id of Object.keys(M.levels))test(id+' reference circuit matches three / five inputs',()=>{
 const s=M.solutions(id);if(M.levels[id].engine)s.settings.precision=16;
 for(const t of M.samplesFor(id).slice(0,M.levels[id].sampleCount)){const e=M.evaluate(s,t.v);assert.equal(e.error,null);assert.ok(close(e.output,M.reference(t.v,id)),id+': '+e.output+' ≠ '+M.reference(t.v,id));}
});
test('main and bonus tracks are separate, complete and have no dangling links',()=>{const visited=[];for(const [track,route]of Object.entries(M.routes)){let id=route.ids[0],ids=[];while(id){assert.ok(!visited.includes(id));assert.equal(M.levels[id].track,track);visited.push(id);ids.push(id);id=M.levels[id].next;}assert.deepEqual(ids,Array.from(route.ids));}assert.equal(visited.length,Object.keys(M.levels).length);assert.equal(M.levels.autoregressive.next,'transformer-demo');assert.equal(M.routes.main.ids.length,30);assert.equal(M.routes.engine.ids.length,9);for(const id of visited){const s=M.initial(id),allowed=new Set([...s.nodes.map(n=>n.id),...M.levels[id].sources]);for(const n of s.nodes)for(const link of n.links)assert.ok(link===null||allowed.has(link));}});
test('causal attention cannot read future V',()=>{const s=M.solutions('attention'),a=M.evaluate(s,[1,0,0,1,1,1]),b=M.evaluate(s,[1,0,0,1,-8,3]);assert.ok(close(a.output.slice(0,4),b.output.slice(0,4)));const weights=a.values.softmax;for(let i=0;i<3;i++){assert.ok(Math.abs(weights.slice(i*3,i*3+3).reduce((a,b)=>a+b,0)-1)<1e-12);for(let j=i+1;j<3;j++)assert.equal(weights[i*3+j],0);}});
test('RoPE preserves each vector norm and position zero',()=>{const x=[2,3,4,5,-1,2],r=M.math.rotate(x);assert.deepEqual(Array.from(r.slice(0,2)),[2,3]);for(let i=0;i<3;i++)assert.ok(Math.abs(r[2*i]**2+r[2*i+1]**2-x[2*i]**2-x[2*i+1]**2)<1e-12);});
test('same dimensions but wrong activation branch fails',()=>{const s=M.solutions('gating');s.nodes.find(n=>n.id==='silu').links=['project'];s.nodes.find(n=>n.id==='multiply').links=['silu','gate'];assert.ok(M.samples.some(t=>!close(M.evaluate(s,t.v).output,M.reference(t.v,'gating'))));});
test('wrong features produce computed words, not canned failure text',()=>{const wrong=M.solutions('vocabulary');wrong.nodes.find(n=>n.id==='original-head').links=['wide-invert'];const initial=M.evaluate(wrong,M.samplesFor('vocabulary')[0].v);assert.equal(M.readTokens(initial.output),'你恨你');const correct=M.evaluate(M.solutions('vocabulary'),M.samplesFor('vocabulary')[0].v);assert.equal(M.readTokens(correct.output),'我爱我');});
test('autoregression appends only the last position prediction',()=>{const s=M.solutions('autoregressive');let input=[0,1,2];input=M.evaluate(s,input).output;assert.deepEqual(Array.from(input),[1,2,0]);input=M.evaluate(s,input).output;assert.deepEqual(Array.from(input),[2,0,1]);});
test('generation examples preserve the old context and match playable updates, including repeated tokens',()=>{
 const s=M.solutions('autoregressive');
 for(const [input,expected]of [[[0,1,2],[1,2,0]],[[3,4,5],[4,5,3]],[[0,0,3],[0,3,4]]]){
  const before=[...input],example=M.autoregressiveExample(input),ev=M.evaluate(s,input);
  assert.deepEqual(input,before);assert.deepEqual(Array.from(example.next),expected);
  assert.deepEqual(Array.from(example.next),Array.from(ev.output));assert.deepEqual(Array.from(example.predictions),Array.from(ev.values.pick));
  assert.deepEqual(Array.from(example.scores),Array.from(ev.values.step));
 }
 const trace=M.rollout(s,[0,1,2]).trace;assert.equal(trace.length,6);
 for(let i=1;i<trace.length;i++)assert.deepEqual(Array.from(trace[i].context),Array.from(trace[i-1].next));
});
test('cycle fails; disconnected branch still runs for visible animation',()=>{const s=M.solutions('attention');s.nodes.find(n=>n.id==='ropeq').links=['ropeq'];assert.ok(M.evaluate(s,M.samplesFor('attention')[0].v).error);const p=M.solutions('projection');p.sink=null;const e=M.evaluate(p,[1,1]);assert.ok(e.error);assert.ok(close(e.values.project,[1,1,1.5,-.5]));});
test('quantization changes actual numbers; cache and batches do not',()=>{const s=M.solutions('quantization'),x=M.samplesFor('quantization')[0].v;s.settings.precision=16;const a=M.evaluate(s,x).output;s.settings.precision=2;const b=M.evaluate(s,x).output;assert.ok(!close(a,b));s.settings.precision=16;s.settings.cache=false;s.settings.batch=1;assert.ok(close(M.evaluate(s,x).output,a));});
// Only vary controls exposed to players, plus dragging tasks to supported devices.
for(const id of Object.keys(M.levels).filter(id=>M.levels[id].engine))test(id+' is solvable using only visible controls',()=>{
 const level=M.levels[id],initial=M.initial(id);assert.equal(M.engineReport(initial).pass,false,id+' unexpectedly starts solved');
 let found=null;
 const proposals=[{}, {batch:16,layout:'channels',cache:true,precision:4,fusion:true,resident:true,split:40,prefix:true,context:4096}, {batch:32,layout:'channels',cache:true,precision:4,fusion:true,resident:true,split:60,prefix:true,context:4096}];
 for(const proposal of proposals)for(const device of ['gpu','ane']){
  const s=M.solutions(id);for(const key of level.controls)if(key in proposal)s.settings[key]=proposal[key];
  for(const n of s.nodes)n.device=['enqkv','enffn'].includes(n.id)?device:['load','enembed','enhead'].includes(n.id)?'cpu':'gpu';
  const report=M.engineReport(s);if(report.pass){for(const t of M.samplesFor(id)){const ev=M.evaluate(s,t.v);assert.equal(ev.error,null);assert.ok(close(ev.output,M.reference(t.v,id),level.limits.error||.06));}found={settings:s.settings,devices:s.nodes.map(n=>[n.id,n.device]),metrics:report.runs.map(r=>r.metrics)};break;}
 }
 assert.ok(found,id+' has no player-accessible solution');
});
test('correct circuit can fail performance without changing generated result',()=>{const s=M.solutions('kv-cache');const x=M.samplesFor(s.levelId)[0].v,a=M.evaluate(s,x).output;assert.ok(close(a,M.reference(x,s.levelId),.06));assert.equal(M.engineReport(s).pass,false);s.settings.cache=true;assert.ok(close(a,M.evaluate(s,x).output));assert.equal(M.engineReport(s).pass,true);});
test('ANE incompatible layout and unsupported ops fail compilation',()=>{const s=M.solutions('layout');assert.ok(M.engineReport(s).errors.some(e=>e.includes('布局')));s.settings.layout='channels';s.nodes.find(n=>n.id==='enattn').device='ane';assert.ok(M.engineReport(s).errors.some(e=>e.includes('支持集合')));});
test('KV cache reuses three rows instead of six and preserves Attention output',()=>{const x=M.samplesFor('attention')[0].v,a=M.cachedAttention(x,true),b=M.cachedAttention(x,false);assert.equal(a.processed,3);assert.equal(b.processed,6);assert.ok(close(a.output,b.output));assert.ok(close(a.output,M.math.attention(x)));});
test('more reserved KV capacity consumes memory even when prompt is unchanged',()=>{const s=M.solutions('quantization');s.settings.context=2048;const a=M.engineReport(s).metrics.memory;s.settings.context=8192;assert.ok(M.engineReport(s).metrics.memory>a);});

test('multiple projection and residual instances have independent edges, values and removable stock',()=>{
 const s=M.initial('ffn'),kinds=['project','down','project','down','add','add'],coords=[[-4.7,-2],[-1.4,-2],[1.9,-2],[5.2,-2],[-4.7,2],[-1.4,2]];
 kinds.forEach((kind,i)=>assert.equal(M.addNode(s,kind,...coords[i]),true));
 assert.deepEqual(Array.from(s.nodes,n=>n.id),['project','down','project-2','down-2','add','add-2']);
 const edges=[['R','project',0],['project','down',0],['down','project-2',0],['project-2','down-2',0],['down','add',0],['down-2','add',1],['add','add-2',0],['R','add-2',1],['add-2','sink',0]];
 for(const edge of edges)assert.equal(M.connect(s,...edge),null);
 const ev=M.evaluate(s,[1,0]);assert.equal(ev.error,null);assert.ok(close(ev.values.down,[.55,.35]));assert.ok(close(ev.values['down-2'],[.3725,.2275]));assert.ok(close(ev.output,[1.9225,.5775]));assert.notDeepEqual(ev.values.project,ev.values['project-2']);
 assert.equal(M.stock(s,'project'),2,'merged stock leaves two more projections available');
 assert.equal(M.removeNode(s,'down-2'),true);assert.deepEqual(Array.from(s.nodes.find(n=>n.id==='add').links),['down',null]);assert.deepEqual(Array.from(s.nodes.find(n=>n.id==='down').links),['project']);assert.equal(M.stock(s,'down'),1);assert.equal(M.addNode(s,'down',5.2,-2),true);
});
test('construction limits count current instances and fixed lessons cannot create modules',()=>{
 for(const id of ['wire','projection','relu','down-projection','original-ffn','residual'])assert.equal(M.canAdd(M.initial(id),'project'),false);
 const tutorial=M.initial('workshop');assert.equal(M.addNode(tutorial,'relu-ffn',0,0),true);assert.equal(M.canAdd(tutorial,'relu-ffn'),false);assert.equal(M.removeNode(tutorial,'relu-ffn'),true);assert.equal(M.canAdd(tutorial,'relu-ffn'),true);
 const s=M.initial('ffn'),kinds=['project','project','down','down','add','add','gate','project'];
 kinds.forEach((kind,i)=>assert.equal(M.addNode(s,kind,[-4.7,-1.4,1.9,5.2][i%4],i<4?-2:2),true));
 assert.equal(s.nodes.length,M.moduleLimit(M.levels.ffn));assert.equal(M.stock(s,'silu'),2);assert.equal(M.canAdd(s,'silu'),false,'free stock cannot exceed the table limit');
 M.removeNode(s,'project');assert.equal(M.canAdd(s,'silu'),true);
});
test('one linear projection stock replaces the content operator and keeps the combined copy budget',()=>{
 const s=M.initial('ffn');assert.equal(M.definitions.up,undefined);assert.equal('up' in M.inventory(M.levels.ffn),false);assert.equal(M.inventory(M.levels.ffn).project,4);
 const coordinates=[[-4.7,-2],[-1.4,-2],[1.9,-2],[5.2,-2]];
 for(const point of coordinates)assert.equal(M.addNode(s,'project',...point),true);
 assert.equal(M.canAdd(s,'project'),false);assert.equal(M.addNode(s,'project',0,2),false);
 assert.equal(M.removeNode(s,'project-3'),true);assert.equal(M.stock(s,'project'),1);assert.equal(M.addNode(s,'project',1.9,-2),true);
 for(const id of ['gating','ffn']){
  const circuit=M.solutions(id);assert.equal(circuit.nodes.filter(n=>n.kind==='project').length,1);assert.ok(!circuit.nodes.some(n=>n.kind==='up'));
  for(const sample of M.samples){const ev=M.evaluate(circuit,sample.v),input=id==='ffn'?M.norm(sample.v):sample.v;assert.ok(close(ev.values.project,M.linear(input,M.weights.up)));assert.ok(close(ev.output,M.reference(sample.v,id)));}
 }
});
test('an alternate attention instance uses its own operator and its connected attention map',()=>{
 const s=M.solutions('attention'),original=s.nodes.find(n=>n.id==='softmax');original.id='softmax-2';original.kind='softmax';s.nodes.find(n=>n.id==='aggregate').links[0]='softmax-2';
 const ev=M.evaluate(s,M.samplesFor('attention')[0].v);assert.equal(ev.error,null);assert.ok(close(ev.output,M.reference(M.samplesFor('attention')[0].v,'attention')));assert.equal(M.valueForKind(s,ev,'softmax'),ev.values['softmax-2']);
});

test('bridge lessons precede compositions and each starts with a real unsolved task',()=>{
 const order=[...M.routes.main.ids,...M.routes.modern.ids],before=(a,b)=>assert.ok(order.indexOf(a)<order.indexOf(b),a+' must precede '+b);
 for(const chain of [['relu','down-projection','original-ffn','workshop','residual','activation','plain-ffn','gate-projection','pairwise','gating','swiglu','normalize','ffn'],['scores','softmax','causal','rotation','rope','aggregation','attention'],['vocabulary-scores','vocabulary','autoregressive']])for(let i=1;i<chain.length;i++)before(chain[i-1],chain[i]);
 for(const id of ['down-projection','plain-ffn','gate-projection','pairwise','swiglu','softmax','rotation','aggregation','vocabulary-scores']){
  const s=M.initial(id);assert.deepEqual(Array.from(s.nodes,n=>n.kind),M.levels[id].requirePlacement?[]:Array.from(M.levels[id].nodes),id+' follows its placement policy');
  if(id!=='down-projection')assert.equal(M.levels[id].buildable,true,id+' now permits returning and placing modules');
  assert.ok(M.samplesFor(id).slice(0,M.levels[id].sampleCount).some(t=>{const ev=M.evaluate(s,t.v);return ev.error||!close(ev.output,M.reference(t.v,id));}),id+' should not start solved');
 }
});

test('FFN bridge lessons distinguish activation, pairing, and normalized residual updates',()=>{
 for(const t of M.samples.slice(0,3)){
  const x=t.v,p=M.linear(x,M.weights.project),g=M.linear(x,M.weights.gate);
  assert.ok(close(M.reference(x,'pairwise'),p.map((v,i)=>v*g[i])));
  const swapped=M.solutions('pairwise');swapped.nodes.find(n=>n.id==='multiply').links.reverse();assert.ok(close(M.evaluate(swapped,x).output,M.reference(x,'pairwise')),'pairing is commutative');
  const update=M.reference(M.norm(x),'swiglu');assert.ok(close(update.map((v,i)=>v+x[i]),M.reference(x,'ffn')),'normalization and residual finish the same SwiGLU computation');
 }
 const skipped=M.solutions('plain-ffn');skipped.nodes.find(n=>n.id==='down').links=['project'];
 assert.ok(M.samples.some(t=>!close(M.evaluate(skipped,t.v).output,M.reference(t.v,'plain-ffn'))),'bypassing activation must not solve the single-branch lesson');
 const self=M.solutions('pairwise');self.nodes.find(n=>n.id==='multiply').links=['project','project'];
 assert.ok(M.samples.some(t=>!close(M.evaluate(self,t.v).output,M.reference(t.v,'pairwise'))),'squaring one branch must not solve the pairing lesson');
});

test('attention bridge output matches the corresponding computations inside full Attention',()=>{
 for(const t of M.samplesFor('attention')){
  const full=M.evaluate(M.solutions('attention'),t.v),gather=M.evaluate(M.solutions('aggregation'),t.v),weights=gather.values.attentionweights;
  assert.ok(close(weights,full.values.softmax));assert.ok(close(gather.output,full.values.aggregate));
  assert.ok(close(M.linear(gather.output.slice(0,2),M.weights.output),full.output.slice(0,2)));
  const a=M.reference(t.v,'softmax');
  for(let i=0;i<3;i++){assert.ok(a.slice(i*3,i*3+3).every(v=>v>0),'unmasked softmax keeps all positions');assert.ok(Math.abs(a.slice(i*3,i*3+3).reduce((sum,v)=>sum+v,0)-1)<1e-12);}
 }
});
test('cloud TP lessons preserve full matrix products and reject swapped, missing or nonlinear shortcuts',()=>{
 assert.deepEqual(Object.keys(M.routes),['main','modern','cloud','engine']);
 const expected={
  'tp-columns':x=>M.linear(x,M.cloud.weights),
  'tp-rows':x=>M.linear(M.linear(x,M.cloud.weights),M.cloud.down),
  'tp-ffn':x=>M.linear(M.linear(x,M.cloud.weights).map(v=>Math.max(0,v)),M.cloud.down)
 };
 const inputs=[[0,0],[1,0],[0,1],[1,1],[-2,3],[.3,-.9],[5,-4]];
 for(const [id,reference]of Object.entries(expected)){
  const fresh=M.initial(id);assert.equal(fresh.nodes.length,0);assert.equal(fresh.sink,null);
  const state=M.solutions(id);for(const x of inputs)assert.ok(close(M.evaluate(state,x).output,reference(x)));
  const last=state.nodes.find(n=>n.id===state.sink);last.links[1]=null;assert.ok(M.evaluate(state,[1,1]).error);
 }
 const swapped=M.solutions('tp-columns');swapped.nodes.find(n=>n.id==='tp-gather').links.reverse();assert.ok(!close(M.evaluate(swapped,[1,1]).output,expected['tp-columns']([1,1])));
 const rows=M.solutions('tp-rows');rows.nodes.find(n=>n.id==='tp-row-a').links=['tp-slice-b'];assert.ok(!close(M.evaluate(rows,[1,1]).output,expected['tp-rows']([1,1])));
 const ffn=M.solutions('tp-ffn');ffn.nodes.find(n=>n.id==='tp-row-b').links=['tp-col-b'];assert.ok(!close(M.evaluate(ffn,[1,1]).output,expected['tp-ffn']([1,1])));
});
test('single GPU lessons enforce actual tensor residency and count copies independently of values',()=>{
 const ids=['single-upload','single-fallback','single-resident'];assert.deepEqual(Array.from(M.routes.cloud.ids.slice(0,3)),ids);
 for(const [i,id] of ids.entries()){
  assert.equal(M.initial(id).nodes.length,0);const s=M.solutions(id),ev=M.evaluate(s,[1,1]);assert.equal(ev.error,null);assert.equal(ev.cloudMemory.bytes,[24,48,16][i]);assert.equal(ev.cloudMemory.locations[s.sink],'cpu');
  for(const t of M.samplesFor(id))assert.ok(close(M.evaluate(s,t.v).output,M.reference(t.v,id)));
 }
 const missing=M.solutions('single-upload');missing.nodes.find(n=>n.id==='sg-project').links=['R'];let ev=M.evaluate(missing,[1,1]);assert.ok(ev.error.includes('GPU'));assert.ok(close(ev.output,M.reference([1,1],'single-upload')),'numerically right still needs a valid device path');
 const unread=M.solutions('single-upload');unread.sink='sg-project';assert.match(M.evaluate(unread,[1,1]).error,/出口在 CPU/);
 const wrongCpu=M.solutions('single-fallback');wrongCpu.nodes.find(n=>n.id==='sg-cpu-relu').links=['sg-project'];assert.match(M.evaluate(wrongCpu,[1,1]).error,/CPU/);
 const detour=M.solutions('single-fallback');detour.levelId='single-resident';ev=M.evaluate(detour,[1,1]);assert.ok(close(ev.output,M.reference([1,1],'single-resident')));assert.equal(ev.cloudMemory.bytes,48);assert.match(ev.error,/上限/);
 const valid=M.solutions('single-resident');const x=[-1,2];assert.equal(M.evaluate(valid,x).cloudMemory.bytes,16,'byte count comes from shapes, not input values');
});
