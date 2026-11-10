const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),test=require('node:test');
const {parseHTML}=require('linkedom');
function boot(storage=new Map(),hash='',languages=['zh-CN']){
 const {document,window}=parseHTML(fs.readFileSync(__dirname+'/index.html','utf8'));
 window.HTMLElement.prototype.scrollIntoView=function(){};
 for(const el of document.querySelectorAll('dialog')){el.showModal=function(){this.open=true};el.close=function(){this.open=false};}
 const location=new URL('https://hacking.pub/games/neural-atelier/?version=campaign-1'+hash),listeners=new Map(),entries=[location.href];let cursor=0;
 const hashchange=()=>listeners.get('hashchange')?.(),popstate=()=>listeners.get('popstate')?.();
 const traverse=next=>{const hash=location.hash;location.href=entries[cursor=next];popstate();if(location.hash!==hash)hashchange();};
 const navigation={pushState(_,title,url){location.href=new URL(url,location.href).href;entries.splice(++cursor);entries.push(location.href);},replaceState(_,title,url){location.href=new URL(url,location.href).href;entries[cursor]=location.href;},back(){if(cursor)traverse(cursor-1);},forward(){if(cursor+1<entries.length)traverse(cursor+1);},get length(){return entries.length;}};
 const c={window:{},document,console,navigator:{languages},location,history:navigation,addEventListener:(type,fn)=>listeners.set(type,fn),localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)}};c.window=c;vm.createContext(c);
 for(const file of ['locales/en.js','i18n.js','music.js','shell.js','model.js','campaign.js','engine-model.js','original-curriculum.js','transformer-demo.js','cloud-curriculum.js','lesson-io.js','structure-map.js'])vm.runInContext(fs.readFileSync(__dirname+'/'+file,'utf8'),c,{filename:file});
 for(const file of ['module-cards.js','level-progress.js'])vm.runInContext(fs.readFileSync(__dirname+'/'+file,'utf8').replaceAll('export ',''),c,{filename:file});
 let code=fs.readFileSync(__dirname+'/game.js','utf8').replace(/^import .*\n/gm,'');code=code.slice(0,code.indexOf("try{\n const T="));
 vm.runInContext(code+`\nworld={relocalize(){},labels(){},rebuild(){},beforeStage(){},setStage(){},clearPackets(){},focused(){return null},async animate(){}};refresh();`,c);
 return {c,document,window,storage,location,navigation,hashchange,popstate,eval:code=>vm.runInContext(code,c)};
}
test('failure comparisons reveal rounded-away differences without changing numbers or tolerance',()=>{
 const app=boot();
 const compare=(a,b,tolerance=1e-9)=>JSON.parse(app.eval(`JSON.stringify(comparisonValues(${JSON.stringify(a)},${JSON.stringify(b)},${tolerance}))`));
 assert.deepEqual(compare([1,-1],[0,1]),{actual:'[1, -1]',expected:'[0, 1]',expanded:false});
 const close=compare([.9999500037496875,-.9999500037496875],[.9999986111140047,-.9999986111140046]);
 assert.equal(close.actual,'[0.999950, -0.999950]');assert.equal(close.expected,'[0.999999, -0.999999]');assert.equal(close.expanded,true);
 for(const [actual,expected]of [[1,1+1e-8],[1e-8,-1e-8],[0,1e-9]]){
  const out=compare([actual],[expected]);assert.equal(out.expanded,true);assert.notEqual(out.actual,out.expected);
 }
 assert.equal(compare([1],[1+1e-12]).expanded,false,'differences below grading tolerance need no expansion');
 assert.equal(compare([.001],[.002],.06).expanded,false,'engine tolerances stay unchanged');
 assert.equal(compare(null,[1]).actual,'未得到输出');
 assert.equal(compare([1],[1,2]).expanded,false,'width mismatches retain their widths');
});
test('retired FFN URLs and saved current selection continue without copying answers or losing saves',()=>{
 const fixture=boot(),M=fixture.c.NeuralPrologue,metaKey='neural-atelier-chapters-v1';
 for(const generation of [0,3])for(const hash of ['', '#pack-ffn']){
  const suffix=generation?'-run-'+generation:'',oldKey='neural-atelier-level-pack-ffn-v1'+suffix,nextKey='neural-atelier-level-add-norm-v1'+suffix;
  const retired={...M.solutions('workshop'),levelId:'pack-ffn'},next=M.solutions('add-norm');next.sink=null;next.nodes[0].x=.42;
  const oldRaw=JSON.stringify(retired),nextRaw=JSON.stringify(next),completed=['wire','layernorm','pack-ffn'];
  const backup={generation:0,completed:['wire','pack-ffn'],practiceRuns:{}};
  const protectedEntries=[[oldKey,oldRaw],[oldKey+'-completed',oldRaw],[oldKey+'-before-placement',oldRaw],[nextKey,nextRaw],['neural-atelier-language-v1','en'],['neural-atelier-speed-v1','.25']];
  const storage=new Map([...protectedEntries,[metaKey,JSON.stringify({current:'pack-ffn',completed,generation,nextGeneration:generation,restartBackup:backup})]]),app=boot(storage,hash);
  assert.equal(app.eval('stageId'),'add-norm');assert.equal(app.location.hash,hash?'#add-norm':'#home?lesson=add-norm');assert.equal(app.navigation.length,1);
  assert.equal(app.eval('progress.current()'),'add-norm');assert.equal(app.eval("progress.completed('pack-ffn')"),true);assert.equal(app.eval("progress.completed('add-norm')"),false);
  assert.deepEqual(JSON.parse(app.eval('JSON.stringify(state)')),JSON.parse(nextRaw));
  for(const [key,value]of protectedEntries)assert.equal(storage.get(key),value,key+' preserved');
  assert.deepEqual(JSON.parse(storage.get(metaKey)).restartBackup,backup);
  app.eval('levelMenu()');assert.equal(app.document.querySelectorAll('#level-list [data-level]').length,30);assert.equal(app.document.querySelector('[data-level="pack-ffn"]'),null);
  app.eval("switchLevel('layernorm')");assert.equal(app.eval('stage().next'),'add-norm');app.location.hash='#pack-ffn';app.hashchange();assert.equal(app.eval('stageId'),'add-norm');assert.equal(app.location.hash,'#add-norm');
  assert.equal(app.eval('progress.restartAll()'),true);assert.equal(app.eval('progress.undoRestart()'),true);assert.equal(app.eval("progress.completed('pack-ffn')"),true);
  assert.equal(storage.get(oldKey),oldRaw);assert.equal(storage.get(nextKey),nextRaw);
 }
});
test('reuse lessons accept the learned package without repeating its collection animation',async()=>{
 for(const [id,kind]of [['workshop','relu-ffn'],['pack-scores','original-weights']]){
  const app=boot(new Map(),'#'+id),doc=app.document;let fusions=0;app.c.recordFusion=()=>fusions++;
  app.eval('world.fuse=window.recordFusion');assert.equal(app.eval('state.nodes.length'),0);
  app.eval(`chooseModule('${kind}');placeModule(0,0);chooseOutput('R');chooseInput('${kind}',0);chooseOutput('${kind}');chooseInput('sink',0)`);
  await app.eval('run()');assert.equal(app.eval('levelPassed()'),true);assert.equal(fusions,0);assert.equal(doc.querySelector('#level-complete').open,true);assert.equal(doc.querySelector('#complete-reward').hidden,true);
 }
});
test('revised FFN lessons archive legacy projection circuits in every save generation',()=>{
 for(const generation of [0,3])for(const id of ['workshop','residual']){
  const key=`neural-atelier-level-${id}-v1`+(generation?'-run-'+generation:''),old={version:1,levelId:id,palette:['project','down','add'],nodes:[{id:'project',kind:'project',x:-2,z:1,links:['R']}],sink:'project'};
  if(id==='residual'){old.nodes.push({id:'down',kind:'down',x:1,z:1,links:['project']},{id:'add',kind:'add',x:4,z:1,links:['R','down']});old.sink='add';}
  const raw=JSON.stringify(old),meta={current:id,completed:[id,'wire'],generation,nextGeneration:generation},storage=new Map([[key,raw],['neural-atelier-chapters-v1',JSON.stringify(meta)]]),app=boot(storage,'#'+id);
  assert.equal(storage.get(key+'-before-ffn-reuse'),raw);assert.equal(app.eval('state.nodes.length'),0);assert.equal(app.eval('state.sink'),null);
  assert.equal(app.eval("progress.completed('wire')"),true);assert.equal(app.eval('progress.canRestore(stageId,state)'),false);
  app.eval("M.addNode(state,'relu-ffn',0,0);M.connect(state,'R','relu-ffn',0);save()");
  const edited=app.eval('JSON.stringify(state)'),reloaded=boot(storage,'#'+id);
  assert.equal(reloaded.eval('JSON.stringify(state)'),edited,'an unfinished new circuit survives reload');assert.equal(storage.get(key+'-before-ffn-reuse'),raw);
 }
});
test('independent animation scheduling advances branches separately and waits only for real inputs',()=>{
 const app=boot();
 const result=app.eval(`(()=>{
  const nodes=[{id:'join',links:['av','slow']},{id:'slow',links:['R']},{id:'av',links:['softmax','value']},{id:'softmax',links:['fast']},{id:'value',links:['R']},{id:'fast',links:['R']}];
  const before=JSON.stringify(nodes),depths={R:0,fast:1,slow:1,value:1,softmax:2,av:3,join:4},durations={fast:1,slow:9,value:2,softmax:1,av:3,join:1};
  const independent=animationSchedule(nodes,depths,n=>durations[n.id],2/3,true),waves=animationSchedule(nodes,depths,n=>durations[n.id],2/3,false);
  return {independent:Object.fromEntries(independent.nodes),waves:Object.fromEntries(waves.nodes),end:independent.end,empty:animationSchedule([],{},()=>1,2/3,true).end,preserved:before===JSON.stringify(nodes)};
 })()`);
 const t=result.independent;
 assert.ok(t.softmax.start<t.slow.end,'Softmax does not wait for an unrelated projection');
 assert.ok(t.av.end<t.slow.end,'a downstream operation can finish while another branch still computes');
 for(const [id,inputs]of [['softmax',['fast']],['av',['softmax','value']],['join',['av','slow']]])assert.ok(Math.abs(t[id].start-Math.max(...inputs.map(from=>t[from].end))-2/3)<1e-9,id+' waits exactly for its own inputs plus travel');
 assert.equal(t.softmax.end-t.softmax.start,1,'Softmax keeps its own processing time');
 assert.ok(Math.abs(result.end-t.join.end-2/3)<1e-9);assert.equal(result.empty,2/3);assert.ok(result.preserved);
 assert.equal(result.waves.fast.end,result.waves.slow.end,'teaching waves retain their original timing');
});

test('the full-model demonstration runs immediately, replays without running and keeps puzzle saves untouched',async()=>{
 const storage=new Map(),app=boot(storage,'#transformer-demo'),doc=app.document;
 const graph=app.eval('JSON.stringify(state)');assert.equal(app.eval('state.nodes.length'),67);
 for(const selector of ['#module-shelf','#reset','#undo'])assert.equal(doc.querySelector(selector).hidden,true,selector);
 assert.equal(doc.querySelector('#demo-views').hidden,false);assert.equal(app.eval('busy'),false);
 app.eval("chooseOutput('R');chooseInput('sink',0)");assert.equal(app.eval('pending'),null);assert.equal(app.eval('JSON.stringify(state)'),graph);
 await app.eval('run()');assert.equal(app.eval('levelPassed()'),true);assert.equal(doc.querySelector('#level-complete').open,true);assert.match(doc.querySelector('#complete-detail').textContent,/完整模型/);
 doc.querySelector('#complete-replay').onclick();assert.equal(app.eval('busy'),false);assert.equal(app.eval('JSON.stringify(state)'),graph);
 app.eval("switchLevel('projection');state.nodes[0].links=['R'];save()");const puzzle=storage.get('neural-atelier-level-projection-v1');
 app.eval("switchLevel('transformer-demo')");assert.equal(app.eval('JSON.stringify(state)'),graph);assert.equal(storage.get('neural-atelier-level-projection-v1'),puzzle);
 app.c.I18n.setPreference('en');assert.doesNotMatch(doc.querySelector('.mission').textContent+doc.querySelector('#demo-views').textContent,/[\u3400-\u9fff]/);assert.deepEqual(Array.from(app.c.I18n.missing),[]);
});
test('demo generation uses actual outputs, retains text, and clears it on replay',async()=>{
 const app=boot(new Map(),'#transformer-demo'),doc=app.document,inputs=[],feedback=[];
 app.c.capture={animate(){inputs.push(JSON.parse(app.eval('JSON.stringify(samples()[sample].v)')));},feedback(token){feedback.push(token);}};
 app.eval(`world.animate=capture.animate;world.feedback=capture.feedback;const ev0=M.evaluate,ref0=M.reference;M.evaluate=(s,v)=>{const e=ev0(s,v);if(s.levelId==='transformer-demo'&&v.join(',')==='0,1,2')e.output[2]=3;return e;};M.reference=(v,id)=>{const e=ref0(v,id);if(id==='transformer-demo'&&v.join(',')==='0,1,2')e[2]=3;return e;};`);
 await app.eval('run()');assert.deepEqual(inputs[0],[0,1,2]);assert.deepEqual(inputs[1],[1,2,3],'next input comes from the actual previous output');assert.equal(feedback[0],3);assert.equal(feedback.length,2);
 assert.equal(doc.querySelectorAll('#demo-transcript .seed-token').length,3);assert.equal(doc.querySelectorAll('#demo-transcript .generated-token').length,3);assert.equal(app.eval('levelPassed()'),true);
 doc.querySelector('#complete-replay').onclick();assert.equal(app.eval('busy'),false);assert.equal(doc.querySelectorAll('#demo-transcript .generated-token').length,0);
});
test('demo fullscreen preserves playback and saves, follows Escape, and localizes its live state',async()=>{
 const app=boot(new Map(),'#transformer-demo'),doc=app.document,game=doc.querySelector('#game'),button=doc.querySelector('#demo-fullscreen');
 const before=app.eval('JSON.stringify(state)'),stored=JSON.stringify([...app.storage]);
 const change=()=>doc.dispatchEvent(new app.window.Event('fullscreenchange'));
 game.requestFullscreen=async options=>{assert.equal(options.navigationUI,'hide');doc.fullscreenElement=game;change();};doc.exitFullscreen=async()=>{doc.fullscreenElement=null;change();};
 assert.equal(button.hidden,false);assert.equal(button.previousElementSibling.id,'demo-follow');
 app.eval('busy=true;paused=true;setPlaybackSpeed(.5,false)');
 await button.onclick();assert.equal(doc.fullscreenElement,game);assert.equal(button.textContent,'退出全屏');assert.equal(button.getAttribute('aria-pressed'),'true');assert.equal(button.dataset.fullscreenMode,'native');
 assert.ok(game.contains(doc.querySelector('#run'))&&game.contains(doc.querySelector('#pause'))&&game.contains(doc.querySelector('#playback-speed')));
 app.c.I18n.setPreference('en');assert.equal(button.textContent,'Exit fullscreen');
 await doc.exitFullscreen();assert.equal(button.textContent,'Fullscreen');assert.equal(button.getAttribute('aria-pressed'),'false');
 await button.onclick();await button.onclick();assert.equal(doc.fullscreenElement,null);assert.equal(button.disabled,false);
 assert.equal(app.eval('busy&&paused&&playbackSpeed===.5'),true);assert.equal(app.eval('JSON.stringify(state)'),before);
 assert.equal(JSON.stringify([...app.storage].filter(([k])=>k!=='neural-atelier-language-v1')),stored);
 app.eval("busy=false;paused=false;switchLevel('wire')");assert.equal(button.hidden,true);assert.deepEqual([...app.c.I18n.missing],[]);
});
test('embedded browsers fall back to page fullscreen and Escape preserves the run',async()=>{
 const app=boot(new Map(),'#transformer-demo'),doc=app.document,button=doc.querySelector('#demo-fullscreen');
 doc.querySelector('#game').requestFullscreen=async()=>{throw new Error('denied');};
 app.eval('busy=true');await button.onclick();assert.equal(button.disabled,false);assert.equal(button.getAttribute('aria-pressed'),'true');assert.equal(doc.querySelector('#game').classList.contains('demo-expanded'),true);assert.equal(app.eval('busy'),true);
 const e=new app.window.Event('keydown');e.key='Escape';doc.dispatchEvent(e);assert.equal(button.getAttribute('aria-pressed'),'false');assert.equal(doc.querySelector('#game').classList.contains('demo-expanded'),false);assert.equal(app.eval('busy'),true);
 await button.onclick();app.eval("busy=false;switchLevel('wire')");assert.equal(doc.body.classList.contains('demo-fullscreen'),false);
});
test('score input and output describe X, projected Q/K and pairwise scores in both languages',()=>{
 for(const language of ['zh-CN','en-US']){
  const app=boot(new Map(),'#scores',[language]),M=app.c.NeuralPrologue,doc=app.document,before=app.eval('JSON.stringify(state)');
  assert.equal(M.levels.scores.outputPairAxes,true);assert.match(M.levels.scores.outputTitle,language==='zh-CN'?/匹配分数矩阵 S/:/Matching-score matrix S/);
  assert.match(doc.querySelector('.route').textContent,language==='zh-CN'?/输入向量矩阵 X/:/Input vector matrix X/);
  assert.match(doc.querySelector('.mission').textContent,/Q = XWq/);assert.match(doc.querySelector('.mission').textContent,/K = XWk/);assert.match(doc.querySelector('.mission').textContent,/S = QKᵀ \/ √2/);
  const x=M.samplesFor('scores')[0].v,q=M.operations.q([x]),k=M.operations.k([x]);assert.deepEqual(Array.from(q.slice(0,2)),[1,.25]);assert.deepEqual(Array.from(k.slice(2,4)),[.2,1]);assert.equal(+M.reference(x,'scores')[1].toFixed(2),.32);
  assert.deepEqual(Array.from(x),Array.from(M.samplesFor('qkv')[0].v));assert.notDeepEqual(Array.from(x),Array.from(M.reference(x,'qkv')));
  doc.querySelector('#hint').onclick();assert.match(doc.querySelector('#structure-dialog').textContent,/Wq/);assert.match(doc.querySelector('#structure-dialog').textContent,/Softmax/);
  if(language==='en-US')assert.doesNotMatch(doc.querySelector('.mission').textContent,/[\u3400-\u9fff]/);
  assert.equal(app.eval('JSON.stringify(state)'),before);
 }
});
test('all numeric batch outlets have reading time without changing word-output animations',()=>{
 const app=boot(),M=app.c.NeuralPrologue;
 for(const [id,layout]of Object.entries(M.levels)){
  if(layout.sequence&&!layout.tokenOutput)assert.equal(layout.outputArrivalHold,450,id+' pauses numeric results');
  else assert.equal(layout.outputArrivalHold,undefined,id+' retains its existing individual/word readout');
 }
});
test('QKV distinguishes the computed V matrix from Wv weights in its goal, card and hint',()=>{
 for(const language of ['zh-CN','en-US']){
  const app=boot(new Map(),'#qkv',[language]),M=app.c.NeuralPrologue,doc=app.document,before=app.eval('JSON.stringify(state)');
  assert.equal(M.levels.qkv.outputNote,'V = X × Wᵥ');assert.match(M.levels.qkv.outputTitle,language==='zh-CN'?/值向量矩阵 V/:/Value matrix V/);
  assert.match(doc.querySelector('.mission').textContent,/Wᵥ/);assert.match(doc.querySelector('#mission-reason').textContent,/3 × 2/);assert.match(doc.querySelector('#mission-reason').textContent,/\[0, 2\]/);
  assert.deepEqual(JSON.parse(JSON.stringify(M.weights.v)),[[1,0],[0,2]]);assert.deepEqual(JSON.parse(JSON.stringify(M.linear([0,1],M.weights.v))),[0,2],'the displayed example uses the real weight matrix');
  const card=M.extraCard('v','qkv');assert.equal(card.formula,'V = X × Wᵥ');assert.match(card.meaning,/Wᵥ/);assert.match(card.purpose,/\[0, 2\]/);
  doc.querySelector('#hint').onclick();assert.match(doc.querySelector('#structure-dialog').textContent,/V = X × Wᵥ/);
  if(language==='en-US'){assert.doesNotMatch(doc.querySelector('.mission').textContent,/[\u3400-\u9fff]/);assert.doesNotMatch(card.meaning+card.purpose,/[\u3400-\u9fff]/);}
  assert.equal(app.eval('JSON.stringify(state)'),before);
 }
});
test('QKV passes with only V placed or with Q/K left disconnected, while wrong outlet values fail',async()=>{
 const app=boot();app.eval("switchLevel('qkv');state=M.initial('qkv');M.addNode(state,'v',0,0);M.connect(state,'R','v',0);M.connect(state,'v','sink',0)");
 let played=0;app.c.worldForTest={async animate(){played++;assert.equal(app.eval('fault'),null);assert.ok(app.document.querySelector('#run-report').hidden);}};
 app.eval('world.animate=worldForTest.animate');await app.eval('run()');
 assert.equal(played,3);assert.equal(app.eval('levelPassed()'),true);assert.equal(app.eval('fault'),null);assert.equal(app.document.querySelector('.projection-checks'),null);assert.equal(app.document.querySelector('#level-complete').open,true);
 app.eval("state=M.solutions('qkv');state.nodes.find(n=>n.id==='q').links=[null];state.nodes.find(n=>n.id==='k').links=[null]");await app.eval('run()');assert.equal(app.eval('levelPassed()'),true);
 app.eval("state=M.solutions('qkv');state.sink='q'");await app.eval('run()');assert.equal(app.eval('levelPassed()'),false);assert.equal(app.eval('fault.id'),'sink');assert.ok(!app.document.querySelector('#level-complete').open);
});
test('QKV keeps saved V-only circuits and badges from before or during the temporary stricter rule',async()=>{
 const setup=boot(),M=setup.c.NeuralPrologue,key='neural-atelier-level-qkv-v1',meta='neural-atelier-chapters-v1';
 const saved=M.solutions('qkv');saved.nodes.find(n=>n.id==='q').links=[null];saved.nodes.find(n=>n.id==='k').links=[null];
 for(const completed of [['qkv','scores'],['scores']]){
  const storage=new Map([[key,JSON.stringify(saved)],[meta,JSON.stringify({current:'qkv',completed,goalRevisions:{qkv:1}})]]),app=boot(storage,'#qkv');
  assert.equal(app.eval("progress.completed('qkv')"),completed.includes('qkv'));assert.equal(app.eval("progress.completed('scores')"),true);
  const before=app.eval('JSON.stringify(state)');assert.deepEqual(JSON.parse(before),JSON.parse(JSON.stringify(saved)));await app.eval('run()');assert.equal(app.eval('levelPassed()'),true);assert.equal(app.eval("progress.completed('qkv')"),true);assert.equal(app.eval('JSON.stringify(state)'),before);
 }
});
test('saved borrowed LayerNorm circuits retain their wiring but report a role error instead of a false numeric failure',async()=>{
 const base=boot(),saved=JSON.parse(JSON.stringify(base.c.NeuralPrologue.solutions('decoder-layer')));
 const borrowed=saved.nodes.find(n=>n.id==='ln-attn');borrowed.id='ln-ffn-2';borrowed.kind='ln-ffn';
 for(const n of saved.nodes)n.links=n.links.map(id=>id==='ln-attn'?'ln-ffn-2':id);
 for(const language of ['zh-CN','en-US']){
  const storage=new Map([['neural-atelier-level-decoder-layer-v1',JSON.stringify(saved)],['neural-atelier-chapters-v1',JSON.stringify({current:'decoder-layer',completed:['decoder-self']})]]);
  const app=boot(storage,'#decoder-layer',[language]),doc=app.document,before=app.eval('JSON.stringify(state)');let played=0;
  assert.deepEqual(JSON.parse(before),saved);
  app.c.observeRun=()=>{played++;assert.equal(app.eval('fault'),null);assert.equal(doc.querySelector('#run-report').hidden,true);};app.eval('world.animate=window.observeRun');
  await app.eval('run()');assert.equal(played,1);assert.equal(app.eval('levelPassed()'),false);assert.ok(!doc.querySelector('#level-complete').open);
  assert.equal(app.eval('current.componentIssue.id'),'ln-ffn-2');assert.equal(app.eval('JSON.stringify(state)'),before);assert.equal(app.eval("progress.completed('decoder-self')"),true);
  assert.match(doc.querySelector('#run-report').textContent,language==='en-US'?/Module role mismatch.*FFN/s:/组件用途不匹配.*FFN/s);
  assert.doesNotMatch(doc.querySelector('#run-report').textContent,/数值不匹配|Values do not match/);
  if(language==='en-US')assert.doesNotMatch(doc.querySelector('#run-report').textContent,/[\u3400-\u9fff]/);
  app.c.I18n.setPreference(language==='en-US'?'zh':'en');
  assert.match(doc.querySelector('#run-report').textContent,language==='en-US'?/组件用途不匹配/:/Module role mismatch/);
  assert.deepEqual([...app.c.I18n.missing],[]);
 }
});
test('all levels automatically display their report beside the run controls',async()=>{
 const app=boot(),M=app.c.NeuralPrologue;
 for(const id of Object.keys(M.levels)){
  app.eval(`switchLevel('${id}');state=M.solutions(stageId);if(stage().engine)state.settings.precision=16;`);
  assert.ok(app.document.querySelector('.mission h2').textContent.length);assert.equal(app.document.querySelector('#demo-pan').hidden,false,id+' offers camera panning');
  const reason=app.document.querySelector('#mission-reason'),example=app.document.querySelector('#mission-reason-example'),hasExample=['gating','residual','ffn','wider-features','causal','masked-multihead'].includes(id)||!!M.levels[id].packages?.length;
  assert.equal(reason.hidden,false,id+' explains its purpose');
  for(const field of ['title','text','takeaway'])assert.ok(app.document.querySelector('#mission-reason-'+field).textContent.trim(),id+' '+field);
  assert.equal(example.hidden,!hasExample);assert.equal(example.hasChildNodes(),hasExample,'text-only lessons clear earlier worked examples');
  const scope=app.document.querySelector('#mission-reason-scope');assert.equal(scope.hidden,!M.levels[id].rationale.scope);
  assert.equal(app.document.querySelectorAll('[data-setting]').length,M.levels[id].controls?.length||0);
  await app.eval('run()');assert.equal(app.eval('busy'),false);
  const panel=app.document.querySelector('#run-report');
  assert.ok(panel.querySelector('.run-report-content').textContent.length);
  assert.equal(panel.hidden,false);
  assert.equal(panel.parentElement.className,'run-toolbar');
  assert.equal(app.document.querySelector('#report-toggle'),null);
  for(const n of M.levels[id].nodes){app.eval(`select('${n}')`);const card=app.eval(`cardFor('${n}',stageId)||M.extraCard('${n}',stageId)`);assert.ok(card.meaning);const example=app.eval(`M.extraExample('${n}',stageId)||exampleFor(M,'${n}',stageId)`);assert.ok(example.output[1].every(Number.isFinite)||n==='mask');}
 }
});
test('gating rationale uses actual weights to show equal content receiving different scales',()=>{
 const app=boot(),M=app.c.NeuralPrologue,doc=app.document,round=v=>Number(v.toFixed(2));
 for(const language of ['zh','en']){
  app.c.I18n.setPreference(language);app.eval("switchLevel('gating')");
  const saved=app.eval('JSON.stringify(state)'),reason=doc.querySelector('#mission-reason');assert.equal(reason.hidden,false);
  const rows=[...reason.querySelectorAll('tbody tr')],contents=[],factors=[];assert.equal(rows.length,2);
  for(const row of rows){
   const input=JSON.parse(row.querySelector('td').textContent),ev=M.evaluate(M.solutions('gating'),input);
   for(const [role,key]of [['content','project'],['scale','silu'],['result','multiply']])assert.equal(Number(row.querySelector(`[data-factor="${role}"]`).textContent),round(ev.values[key][2]));
   contents.push(ev.values.project[2]);factors.push(ev.values.silu[2]);
  }
  assert.equal(contents[0],contents[1]);assert.ok(factors[0]>0&&factors[0]<1);assert.ok(factors[1]>1);
  assert.equal(app.eval('JSON.stringify(state)'),saved,'the comparison never prewires the player’s circuit');
  if(language==='en')assert.doesNotMatch(reason.textContent,/[\u3400-\u9fff]/);
  app.eval("switchLevel('activation')");assert.equal(reason.hidden,false);assert.equal(reason.querySelector('table'),null);assert.equal(doc.querySelector('#mission-reason-example').hidden,true);assert.equal(doc.querySelector('#mission-reason-scope').hidden,true);
 }
 assert.deepEqual([...app.c.I18n.missing],[]);
});
test('inline report appears without clicking, stays during scene interaction, dismisses independently and returns next run',async()=>{
 const app=boot(),panel=app.document.querySelector('#run-report');app.document.querySelector('#run').focus=()=>{};
 assert.equal(panel.hidden,true);
 app.eval("switchLevel('vocabulary');state=M.solutions(stageId);state.nodes.find(n=>n.id==='original-head').links=['wide-invert']");await app.eval('run()');
 assert.ok(panel.textContent.includes('你恨狗'));assert.equal(panel.hidden,false);
 const state=app.eval('JSON.stringify(state)'),fault=app.eval('JSON.stringify(fault)'),results=app.eval('JSON.stringify(results)');
 app.document.querySelector('#scene').dispatchEvent(new app.window.Event('pointerdown',{bubbles:true}));assert.equal(panel.hidden,false);
 app.document.querySelector('[data-close-report]').dispatchEvent(new app.window.Event('click',{bubbles:true}));assert.equal(panel.hidden,true);
 assert.equal(app.eval('JSON.stringify(state)'),state);assert.equal(app.eval('JSON.stringify(fault)'),fault);assert.equal(app.eval('JSON.stringify(results)'),results);
 await app.eval('run()');assert.equal(panel.hidden,false);
 app.eval('changed()');assert.equal(panel.hidden,true);assert.equal(panel.textContent,'');
});
test('device assignment saves and undo restores; controls survive reload',()=>{const app=boot();app.eval("switchLevel('capstone');select('enffn');assignDevice('enffn','ane')");assert.equal(app.eval("state.nodes.find(n=>n.id==='enffn').device"),'ane');app.eval("$('#undo').onclick()");assert.equal(app.eval("state.nodes.find(n=>n.id==='enffn').device"),'cpu');app.eval("state.settings.precision=4;save();switchLevel('wire');switchLevel('capstone')");assert.equal(app.eval('state.settings.precision'),4);});

test('wire deletion and undo preserve other branches and survive reload',()=>{
 const app=boot();app.eval("switchLevel('gating');state=M.solutions(stageId);chooseInput('multiply',0)");assert.equal(app.eval('selectedWire.from'),'silu');assert.equal(app.eval("state.nodes.find(n=>n.id==='multiply').links[0]"),'silu');
 app.eval('deleteWire()');assert.equal(app.eval("state.nodes.find(n=>n.id==='multiply').links[0]"),null);assert.equal(app.eval("state.nodes.find(n=>n.id==='multiply').links[1]"),'project');
 const restored=boot(app.storage,'#gating');assert.equal(restored.eval("state.nodes.find(n=>n.id==='multiply').links[0]"),null);app.eval("$('#undo').onclick()");assert.equal(app.eval("state.nodes.find(n=>n.id==='multiply').links[0]"),'silu');
});
test('partially constructed tables and older complete saves retain their positions and connections',()=>{
 const app=boot();app.eval("switchLevel('ffn');chooseModule('gate');placeModule(-3,-2);chooseOutput('Z');chooseInput('gate',0)");assert.equal(app.eval('state.nodes.length'),1);
 const saved=app.eval('JSON.stringify(state)'),reloaded=boot(app.storage,'#ffn');assert.equal(reloaded.eval('JSON.stringify(state)'),saved);assert.equal(reloaded.document.querySelector('[data-add-module="gate"]').getAttribute('data-add-disabled'),'false');
 reloaded.eval("select('gate');removeModule()");assert.equal(reloaded.eval('state.nodes.length'),0);reloaded.eval("$('#undo').onclick()");assert.equal(reloaded.eval('JSON.stringify(state)'),saved);
 const old=app.eval("M.solutions('ffn')");delete old.palette;old.nodes[0].x=1.234;
 const legacy=boot(new Map([['neural-atelier-prologue-v1',JSON.stringify(old)]]));assert.equal(legacy.eval('stageId'),'ffn');assert.equal(legacy.eval('state.nodes.length'),6);assert.equal(legacy.eval('state.nodes[0].x'),1.234);assert.equal(legacy.eval('state.sink'),'add');
});
test('learning stages preserve level navigation and keep the other tracks separate',()=>{
 const app=boot(new Map(),'', ['en-US']),doc=app.document,M=app.c.NeuralPrologue;app.eval('levelMenu()');assert.equal(doc.querySelectorAll('#level-list [data-level]').length,30);assert.equal(doc.querySelector('#level-list [data-level="placement"]'),null);
 assert.deepEqual([...doc.querySelectorAll('#level-list [data-level]')].map(e=>e.dataset.level),Array.from(M.routes.main.ids),'grouping keeps every existing level exactly once and in order');
 assert.equal(doc.querySelectorAll('[data-level-group]').length,6);
 assert.equal(doc.querySelector('#level-tracks').closest('.level-menu-content'),null);
 assert.equal(doc.querySelector('#level-stage-nav').closest('.level-menu-content'),null);
 for(const [id,group]of [['multihead','attention'],['wider-features','encoder'],['encoder-layer','encoder'],['causal','decoder'],['cross-attention','decoder']])assert.equal(doc.querySelector(`#level-list [data-level="${id}"]`).closest('[data-level-group]').dataset.levelGroup,group);
 assert.equal(doc.querySelector('[data-lesson-note="wider-features"] .level-purpose').textContent,'Prepare input');
 assert.deepEqual([...doc.querySelectorAll('[data-level-group="decoder"] .lesson-note')].map(e=>e.dataset.lessonNote),['causal','masked-multihead','decoder-self','cross-attention','decoder-layer']);
 assert.match(doc.querySelector('[data-lesson-note="decoder-self"] .lesson-connection').textContent,/D.*15/);
 assert.match(doc.querySelector('[data-lesson-note="cross-attention"] .lesson-connection').textContent,/14.*12.*16/);
 assert.match(doc.querySelector('[data-lesson-note="decoder-layer"] .lesson-connection').textContent,/FFN.*inside the decoder/);
 assert.equal(doc.querySelectorAll('.level-dependencies').length,2);assert.doesNotMatch(doc.querySelector('#level-menu').textContent,/[\u3400-\u9fff]/);
 const before=app.eval('JSON.stringify(state)'),hash=app.location.hash;let scrolled=false;
 doc.querySelector('#level-group-main-decoder').scrollIntoView=()=>{scrolled=true;};
 doc.querySelector('[data-level-group-target="level-group-main-decoder"]').dispatchEvent(new app.window.Event('click',{bubbles:true}));
 assert.ok(scrolled);assert.equal(app.eval('JSON.stringify(state)'),before);assert.notEqual(app.location.hash,hash);assert.match(app.location.hash,/group=decoder/);assert.equal(doc.querySelector('#level-menu').open,true,'browsing a stage does not enter or reset a level');
 doc.querySelector('#level-list [data-level="encoder-layer"]').dispatchEvent(new app.window.Event('click',{bubbles:true}));
 assert.equal(app.eval('stageId'),'encoder-layer');assert.equal(app.location.hash,'#encoder-layer');assert.equal(doc.querySelector('#level-menu').open,false);
 app.eval("progress.complete('wider-features');levelMenu()");
 assert.equal(doc.querySelector('.level-stage-nav [aria-current="step"]').dataset.levelGroupTarget,'level-group-main-encoder');
 assert.match(doc.querySelector('#level-group-main-encoder .level-group-heading small').textContent,/1 \/ 2/);
 const state=app.eval('JSON.stringify(state)');
 for(const language of ['zh','en']){app.c.I18n.setPreference(language);assert.equal(doc.querySelectorAll('#level-list [data-level]').length,30);assert.equal(app.eval('JSON.stringify(state)'),state);if(language==='en')assert.doesNotMatch(doc.querySelector('#level-menu').textContent,/[\u3400-\u9fff]/);}
 doc.querySelector('.level-menu-content').scrollTop=700;
 app.document.querySelector('[data-track="engine"]').dispatchEvent(new app.window.Event('click',{bubbles:true}));assert.equal(app.document.querySelectorAll('#level-list [data-level]').length,9);assert.ok(app.document.querySelector('#level-list [data-level="placement"]'));assert.doesNotMatch(app.document.querySelector('#level-menu').textContent,/[\u3400-\u9fff]/);
 assert.equal(doc.querySelector('[data-level-group]'),null);assert.equal(doc.querySelector('#level-list').classList.contains('grouped-levels'),false);
 assert.equal(doc.querySelector('#level-stage-nav').hidden,true);assert.equal(doc.querySelector('#level-stage-nav').children.length,0);assert.equal(doc.querySelector('.level-menu-content').scrollTop,0);
 doc.querySelector('[data-track="modern"]').dispatchEvent(new app.window.Event('click',{bubbles:true}));assert.equal(doc.querySelectorAll('#level-list [data-level]').length,13);assert.equal(doc.querySelector('[data-level-group]'),null);
 assert.deepEqual([...app.c.I18n.missing],[]);
 app.eval("switchLevel('transformer-demo');progress.complete(stageId);refresh()");assert.equal(app.document.querySelector('#next-level').hidden,true);app.eval("switchLevel('kv-cache');levelMenu()");assert.equal(app.eval('menuTrack'),'engine');
});

test('lesson details reduce default copy without hiding limits or changing the circuit',()=>{
 const app=boot(new Map(),'#encoder-layer'),doc=app.document,details=doc.querySelector('#mission-details');
 assert.equal(details.hasAttribute('open'),false);
 assert.equal(doc.querySelector('#mission-reason-scope').closest('details'),null);
 assert.equal(doc.querySelector('.mission-progress').closest('details'),null);
 assert.equal(doc.querySelector('#mission-limits').closest('details'),null);
 assert.equal(doc.querySelector('.route').textContent.includes(app.eval('stage().title')),false);
 const state=app.eval('JSON.stringify(state)');details.setAttribute('open','');
 app.c.I18n.setPreference('en');assert.equal(details.hasAttribute('open'),true);assert.equal(app.eval('JSON.stringify(state)'),state);
 assert.doesNotMatch(doc.querySelector('.mission').textContent,/[㐀-鿿]/);
 app.eval("switchLevel('causal')");assert.equal(details.hasAttribute('open'),false);
 for(const id of ['wire','workshop','pack-scores']){app.eval(`switchLevel('${id}')`);assert.equal(details.hasAttribute('open'),true,id+' keeps introductory teaching visible');}
 app.eval('levelMenu()');const before=app.eval('JSON.stringify(state)'),hash=app.location.hash;
 assert.equal(doc.querySelectorAll('.level-choice details,.level-choice .lesson-connection,.level-choice small').length,0);
 const groupDetails=doc.querySelector('[data-level-group="decoder"] .level-group-details');
 assert.equal(groupDetails.hasAttribute('open'),false);groupDetails.setAttribute('open','');
 groupDetails.querySelector('summary').dispatchEvent(new app.window.Event('click',{bubbles:true}));
 assert.equal(app.eval('JSON.stringify(state)'),before);assert.equal(app.location.hash,hash);assert.equal(doc.querySelector('#level-menu').open,true);
 assert.ok(groupDetails.querySelector('[data-lesson-note="cross-attention"] .lesson-connection'));
 assert.deepEqual([...app.c.I18n.missing],[]);
});

test('all input rows use the same base animation timing',async()=>{
 const app=boot();app.eval("state=M.solutions('projection');switchLevel('projection');state=M.solutions(stageId);window.timings=[];world.animate=async(ev,ms)=>{timings.push(ms)}");
 await app.eval('run()');assert.deepEqual(Array.from(app.c.timings),[650,650,650]);
});

function completedCircuit(M,id){
 const level=M.levels[id];if(!level.engine)return M.solutions(id);
 const proposal={batch:32,layout:'channels',cache:true,precision:4,fusion:true,resident:true,split:60,prefix:true,context:4096};
 for(const device of ['gpu','ane']){
  const s=M.solutions(id);for(const key of level.controls)if(key in proposal)s.settings[key]=proposal[key];
  for(const n of s.nodes)n.device=['enqkv','enffn'].includes(n.id)?device:['load','enembed','enhead'].includes(n.id)?'cpu':'gpu';
  if(M.engineReport(s).pass)return s;
 }
 throw new Error('No completed circuit fixture: '+id);
}

test('composite rewards wait for a win and their animation, then remain usable without replacing the saved board',async()=>{
 const app=boot(new Map(),'#self-attention'),doc=app.document,selector='[data-add-module="single-attention"]';
 let count=0,begin,finish;const started=new Promise(resolve=>begin=resolve);
 app.c.fuse=reward=>{count++;assert.equal(reward.kind,'single-attention');return new Promise(resolve=>{finish=resolve;begin();});};app.eval('world.fuse=window.fuse');
 assert.equal(doc.querySelector(selector),null);app.eval("chooseModule('single-attention')");assert.equal(app.eval('placeModule(0,0)'),false);
 await app.eval('run()');assert.equal(count,0);assert.equal(app.eval("progress.completed('self-attention')"),false);assert.equal(doc.querySelector(selector),null);
 app.eval("state=M.solutions(stageId);checkpoint();state.nodes[0].z+=.2;save()");const board=app.eval('JSON.stringify(state)'),undo=app.eval('JSON.stringify(history)');
 const running=app.eval('run()');await started;
 assert.equal(app.eval('levelPassed()'),true);assert.equal(app.eval('busy'),true);assert.ok(!doc.querySelector('#level-complete').open);
 assert.equal(doc.querySelector(selector).getAttribute('aria-disabled'),'true','the reward is visible but cannot be placed while its ceremony runs');
 finish(true);await running;assert.equal(app.eval('busy'),false);assert.equal(doc.querySelector('#level-complete').open,true);
 assert.equal(doc.querySelector('#complete-reward').hidden,false);assert.match(doc.querySelector('#complete-reward').textContent,/单头自注意力/);
 assert.equal(app.eval('JSON.stringify(state)'),board);assert.equal(app.eval('JSON.stringify(history)'),undo);
 const restored=boot(app.storage,'#self-attention');assert.ok(restored.document.querySelector(selector));assert.equal(restored.eval('JSON.stringify(state)'),board);
 restored.eval("for(const n of [...state.nodes])removeModule(n.id);chooseModule('single-attention')");assert.equal(restored.eval('placeModule(0,0)'),true);
 restored.eval("chooseOutput('R');chooseInput('single-attention',0);chooseOutput('single-attention');chooseInput('sink',0)");await restored.eval('run()');
 assert.equal(restored.eval('levelPassed()'),true,'the earned component really computes the complete lesson');
 const packed=restored.eval('JSON.stringify(state)'),reloaded=boot(app.storage,'#self-attention');assert.equal(reloaded.eval('JSON.stringify(state)'),packed);
 reloaded.eval("switchLevel('multihead')");assert.ok(reloaded.document.querySelector(selector),'later compatible lessons retain the component');assert.equal(reloaded.document.querySelector('[data-add-module="two-head-attention"]'),null,'the next lesson still needs to be built');
 for(const lang of ['en','zh']){app.c.I18n.setPreference(lang);assert.equal(doc.querySelector('#complete-reward>strong').textContent,app.c.NeuralPrologue.definitions['single-attention'].name);if(lang==='en')assert.doesNotMatch(doc.querySelector('#complete-reward').textContent,/[\u3400-\u9fff]/);}
 assert.deepEqual([...app.c.I18n.missing],[]);
});
test('a failed celebration cannot strand a valid completion or take away its reward',async()=>{
 const app=boot(new Map(),'#self-attention'),errors=[];app.c.console={...console,error:(...args)=>errors.push(args)};
 app.eval("state=M.solutions(stageId);world.fuse=async()=>{throw new Error('presentation unavailable')}");const saved=app.eval('JSON.stringify(state)');await app.eval('run()');
 assert.equal(errors.length,1);assert.equal(app.eval('busy'),false);assert.equal(app.document.querySelector('#level-complete').open,true);assert.ok(app.document.querySelector('[data-add-module="single-attention"]'));assert.equal(app.eval('JSON.stringify(state)'),saved);
});
test('all reusable-network milestones celebrate, while fixed prologue rewards are collected for later building lessons',async()=>{
 const app=boot(),M=app.c.NeuralPrologue,doc=app.document,seen=[];let completedPrologue;
 for(const id of ['original-ffn','softmax','plain-ffn','swiglu','ffn','encoder-decoder','block'])assert.ok(M.fusions[id],id+' has a collection milestone');
 app.c.collect=reward=>{seen.push(app.eval('stageId'));assert.equal(app.eval('levelPassed()'),true);assert.ok(!doc.querySelector('#level-complete').open);assert.ok(doc.querySelector('[data-add-module="'+reward.kind+'"]'));};app.eval('world.fuse=window.collect');
 for(const [id,reward]of Object.entries(M.fusions)){
  app.eval(`switchLevel('${id}');state=M.solutions(stageId)`);const saved=app.eval('JSON.stringify(state)'),count=seen.length;
  await app.eval('run()');assert.equal(seen.length,count+1,id+' celebrates');assert.equal(app.eval('JSON.stringify(state)'),saved);assert.equal(doc.querySelector('#level-complete').open,true);
  assert.equal(doc.querySelector('#complete-reward>strong').textContent,M.definitions[reward.kind].name);assert.equal(doc.querySelector('#module-shelf').hidden,false);
  if(!M.levels[id].buildable){
   completedPrologue=new Map(app.storage);
   assert.equal(doc.querySelector('#module-shelf').dataset.mode,'collection');assert.equal(doc.querySelectorAll('[data-add-module]').length,1);assert.match(doc.querySelector('#complete-reward>small').textContent,/后续搭建关/);
   app.eval(`chooseModule('${reward.kind}')`);assert.equal(app.eval('placeModule(0,0)'),false);assert.equal(app.eval('JSON.stringify(state)'),saved);
  }
  app.eval('state.sink=null');await app.eval('run()');assert.equal(seen.length,count+1,id+' does not celebrate a failure');
 }
 assert.deepEqual(seen,Object.keys(M.fusions));
 const reload=boot(completedPrologue,'#original-ffn');assert.notEqual(reload.eval('state.sink'),null,'refresh resumes the completed circuit');reload.eval("switchLevel('rope');switchLevel('original-ffn')");assert.equal(reload.document.querySelector('#module-shelf').hidden,false);assert.ok(reload.eval('state.nodes.every(n=>n.links.every(x=>x===null))'),'collected rewards do not prewire a deliberately restarted prologue attempt');
 for(const [id,kind]of [['workshop','relu-ffn'],['pack-scores','original-weights']]){
  const fresh=boot(new Map(),'#'+id);
  assert.ok(fresh.document.querySelector('[data-add-module="'+kind+'"]'),'the reuse lesson can use the already taught block before passing');
  fresh.eval(`chooseModule('${kind}')`);assert.equal(fresh.eval('placeModule(0,0)'),true);
 }
});
test('wider-feature examples compare actual embedding tables for the same token in both languages',()=>{
 const app=boot(new Map(),'#wider-features'),M=app.c.NeuralPrologue,saved=app.eval('JSON.stringify(state)');
 for(const language of ['zh','en']){
  app.c.I18n.setPreference(language);const example=app.document.querySelector('.embedding-width-example');assert.ok(example);
  for(const width of [2,4]){
   const row=example.querySelector('[data-width="'+width+'"]');assert.equal(row.querySelector('.embedding-lookup>b').textContent,M.math.vocab[2]);
   assert.deepEqual([...row.querySelectorAll('.embedding-coordinates>span')].map(el=>Number(el.textContent)),Array.from(M.original.embed([2],width)));
  }
  if(language==='en')assert.doesNotMatch(app.document.querySelector('.mission').textContent,/[\u3400-\u9fff]/);
  assert.equal(app.eval('JSON.stringify(state)'),saved,'explanations never change the player’s wiring');
 }
 assert.deepEqual([...app.c.I18n.missing],[]);
});

test('completion waits for every input, replay preserves the circuit and undo, and next loads the saved destination',async()=>{
 const app=boot(),dialog=app.document.querySelector('#level-complete');app.document.querySelector('#run').focus=()=>{};
 app.eval("switchLevel(M.levels.workshop.next);state=M.solutions(stageId);state.nodes[0].x+=.2;save();window.destination=JSON.stringify(state);switchLevel('workshop');state=M.solutions(stageId);checkpoint();state.nodes[0].z+=.25;save();setPlaybackSpeed(.5)");
 const before=app.eval('JSON.stringify(state)'),undo=app.eval('JSON.stringify(history)');
 let runs=0;
 app.eval('world.animate=async()=>{window.checkAnimation()}');
 app.c.checkAnimation=()=>{runs++;assert.ok(!dialog.open,'no completion before all animation has finished');};
 await app.eval('run()');
 assert.equal(dialog.open,true);assert.equal(runs,app.eval('stage().sampleCount'));
 assert.equal(app.document.querySelector('#complete-title').textContent,'本关完成');
 app.c.I18n.setPreference('en');assert.equal(app.document.querySelector('#complete-title').textContent,'Level complete');assert.doesNotMatch(dialog.textContent,/[\u3400-\u9fff]/);
 app.document.querySelector('#complete-replay').onclick();
 assert.equal(dialog.open,false);assert.equal(app.eval('busy'),false);assert.equal(runs,app.eval('stage().sampleCount'));assert.equal(app.eval('results.length'),0);assert.equal(app.document.querySelector('#run-report').hidden,true);
 assert.equal(app.eval('JSON.stringify(state)'),before);assert.equal(app.eval('JSON.stringify(history)'),undo);assert.equal(app.eval('playbackSpeed'),.5);
 await app.eval('run()');assert.equal(dialog.open,true);assert.equal(runs,2*app.eval('stage().sampleCount'));
 app.document.querySelector('#complete-next').onclick();assert.equal(dialog.open,false);assert.equal(app.location.hash,'#'+app.c.NeuralPrologue.levels.workshop.next);assert.equal(app.eval('JSON.stringify(state)'),app.c.destination);
 app.eval("switchLevel('workshop')");assert.equal(app.eval('state.nodes.length'),0);assert.equal(app.eval('state.sink'),null);
 assert.equal(app.document.querySelector('#restore-layout').hidden,false);app.document.querySelector('#restore-layout').onclick();
 assert.equal(app.eval('JSON.stringify(state)'),before);assert.equal(app.eval('busy'),false);assert.equal(runs,2*app.eval('stage().sampleCount'),'restoring never runs the circuit');
});

test('failure, cancellation and unmet engine constraints never show completion; every solved level does',async()=>{
 const app=boot(),dialog=app.document.querySelector('#level-complete'),M=app.c.NeuralPrologue;
 await app.eval('run()');assert.ok(!dialog.open,'unfinished wiring must fail');
 app.eval('state=M.solutions(stageId);world.animate=()=>new Promise(resolve=>window.finishAnimation=resolve)');
 const cancelled=app.eval('run()');await app.eval('run()');app.c.finishAnimation();await cancelled;assert.ok(!dialog.open,'a stopped run is not a win');
 app.eval('world.animate=async()=>{}');
 for(const id of Object.keys(M.levels)){
  app.eval(`switchLevel('${id}')`);app.c.solved=completedCircuit(M,id);app.eval('state=solved');await app.eval('run()');
  assert.equal(dialog.open,true,id+' completes');
  app.document.querySelector('#complete-close').onclick();assert.equal(dialog.open,false);
  app.eval('refresh()');assert.equal(dialog.open,false,'dismissed completion stays dismissed');
 }
 app.eval("switchLevel('capstone');state=M.solutions(stageId);state.settings.precision=16");await app.eval('run()');
 assert.ok(app.eval('results.every(r=>r.pass)'));assert.equal(app.eval('engineReport.pass'),false);assert.equal(dialog.open,false);
});

test('finale completion stays on its track and stale dialogs close on circuit or level changes',async()=>{
 const app=boot(),dialog=app.document.querySelector('#level-complete'),M=app.c.NeuralPrologue;
 for(const [id,track]of [['transformer-demo','main'],['capstone','engine']]){
  app.eval(`switchLevel('${id}')`);app.c.solved=completedCircuit(M,id);app.eval('state=solved');await app.eval('run()');
  assert.equal(app.document.querySelector('#complete-next').textContent,'选择关卡');
  app.document.querySelector('#complete-next').onclick();assert.equal(dialog.open,false);assert.equal(app.document.querySelector('#level-menu').open,true);assert.equal(app.eval('stageId'),id);assert.equal(app.eval('menuTrack'),track);
 }
 app.eval("switchLevel('projection');state=M.solutions(stageId)");await app.eval('run()');assert.equal(dialog.open,true);
 app.eval('changed()');assert.equal(dialog.open,false);
 await app.eval('run()');assert.equal(dialog.open,true);
 app.location.hash='#wire';app.hashchange();assert.equal(dialog.open,false);
});
test('speed slider updates while running, persists across levels and reloads, and leaves circuit and results intact',async()=>{
 const app=boot(),slider=app.document.querySelector('#playback-speed');
 assert.equal(app.eval('playbackSpeed'),.25);
 app.eval("state=M.solutions(stageId)");await app.eval('run()');
 const state=app.eval('JSON.stringify(state)'),results=app.eval('JSON.stringify(results)');
 app.eval('busy=true');slider.value='2';slider.dispatchEvent(new app.window.Event('input',{bubbles:true}));
 assert.equal(app.eval('playbackSpeed'),2);assert.equal(app.document.querySelector('#playback-rate').textContent,'2×');assert.equal(slider.getAttribute('aria-valuetext'),'2 倍速');
 assert.equal(app.eval('JSON.stringify(state)'),state);assert.equal(app.eval('JSON.stringify(results)'),results);
 app.eval("busy=false;switchLevel('projection')");assert.equal(app.eval('playbackSpeed'),2);
 const reloaded=boot(app.storage);assert.equal(reloaded.eval('playbackSpeed'),2);assert.equal(reloaded.document.querySelector('#playback-speed').value,'2');
 for(const invalid of ['0','NaN','999']){app.storage.set('neural-atelier-playback-speed-v1',invalid);assert.equal(boot(app.storage).eval('playbackSpeed'),.25);}
});

test('level fragments open directly and browser back/forward restore each saved circuit',()=>{
 const app=boot(new Map(),'#wire');assert.equal(app.location.hash,'#wire');assert.equal(app.navigation.length,1);
 app.eval("state.sink='R';switchLevel('projection')");assert.equal(app.location.hash,'#projection');assert.equal(app.location.search,'?version=campaign-1');
 app.eval("state.nodes[0].x=1.25;switchLevel('rope')");assert.equal(app.location.hash,'#rope');
 app.navigation.back();assert.equal(app.eval('stageId'),'projection');assert.equal(app.eval('state.nodes[0].x'),1.25);
 app.navigation.back();assert.equal(app.eval('stageId'),'wire');assert.equal(app.eval('state.sink'),'R');
 app.navigation.forward();assert.equal(app.eval('stageId'),'projection');
 const direct=boot(app.storage,'#wire');assert.equal(direct.eval('stageId'),'wire');assert.equal(direct.eval('state.sink'),'R');
 for(const id of Object.keys(app.c.NeuralPrologue.levels))assert.equal(boot(new Map(),'#'+id).eval('stageId'),id);
});
test('invalid fragments fall back safely and changing a fragment stops an active run',async()=>{
 for(const hash of ['#missing','#%E0%A4%A','#constructor']){const app=boot(new Map(),hash);assert.equal(app.eval('stageId'),'wire');assert.equal(app.location.hash,'#home?lesson=wire');assert.equal(app.document.body.dataset.screen,'home');}
 const app=boot();app.eval("state=M.solutions(stageId);world.animate=()=>new Promise(resolve=>window.finishAnimation=resolve)");const running=app.eval('run()');assert.equal(app.eval('busy'),true);
 app.location.hash='#projection';app.hashchange();assert.equal(app.eval('busy'),false);assert.equal(app.eval('stageId'),'projection');app.c.finishAnimation();await running;
 assert.equal(app.location.hash,'#projection');assert.equal(app.eval('results.length'),0);
});

test('chapter back button opens the current route and stage even from a direct lesson link',()=>{
 const fixture=boot(),M=fixture.c.NeuralPrologue;
 for(const id of ['wire','encoder-layer',...Object.values(M.routes).map(route=>route.ids.at(-1))]){
  const app=boot(new Map(),'#'+id),doc=app.document,button=doc.querySelector('.chapter #levels-open'),level=M.levels[id],group=M.routes[level.track].groups?.find(group=>group.ids.includes(id));
  assert.equal(button.tagName,'BUTTON');assert.equal(button.getAttribute('type'),'button');assert.equal(button.getAttribute('aria-controls'),'level-menu');assert.equal(button.getAttribute('aria-label'),'返回关卡选择');assert.equal(button.querySelector('span').textContent,level.chapter);assert.equal(doc.querySelectorAll('#levels-open').length,1);
  app.eval('checkpoint();window.board=state;window.undoStack=history');const before=app.eval('JSON.stringify(state)');
  button.click();assert.equal(doc.body.dataset.screen,'levels');assert.equal(doc.querySelector('#level-menu').open,true);
  assert.equal(app.location.hash,'#menu?lesson='+id+'&track='+level.track+(group?'&group='+group.id:'')+'&from=game');assert.equal(doc.querySelector('#level-tracks [aria-selected=true]').dataset.track,level.track);assert.equal(doc.querySelector('#level-stage-nav [aria-current]')?.dataset.levelGroupTarget,group?'level-group-'+level.track+'-'+group.id:undefined);assert.equal(doc.querySelector('#level-list [aria-current]').dataset.level,id);
  const menuHash=app.location.hash,reload=boot(new Map(app.storage),menuHash);assert.equal(reload.document.body.dataset.screen,'levels');assert.equal(reload.location.hash,menuHash);
  app.navigation.back();assert.equal(app.location.hash,'#'+id);assert.equal(doc.body.dataset.screen,'game');app.navigation.forward();assert.equal(app.location.hash,menuHash);
  let focused=false;button.focus=()=>focused=true;doc.querySelector('#levels-close').click();assert.equal(app.location.hash,'#'+id);assert.equal(focused,true);assert.equal(app.eval('state===window.board&&history===window.undoStack'),true);assert.equal(app.eval('JSON.stringify(state)'),before);
  app.c.I18n.setPreference('en');assert.equal(button.getAttribute('aria-label'),'Back to level selection');assert.equal(button.title,'Back to level selection');assert.doesNotMatch(button.textContent,/[㐀-鿿]/);assert.deepEqual([...app.c.I18n.missing],[]);
 }
});
for(const paused of [false,true])test(`chapter back button remains available during a ${paused?'paused':'running'} run and cancels it safely`,async()=>{
 const app=boot(new Map(),'#wire'),doc=app.document,button=doc.querySelector('.chapter #levels-open');app.c.performance=performance;
 app.eval("state=M.solutions(stageId);checkpoint();world.animate=()=>new Promise(resolve=>window.finishAnimation=resolve);window.board=state;window.undoStack=history");
 const running=app.eval('run()');if(paused)app.eval('togglePause()');assert.equal(app.eval('busy'),true);assert.equal(app.eval('paused'),paused);assert.equal(button.disabled,false);
 button.click();assert.equal(doc.body.dataset.screen,'levels');assert.equal(app.location.hash,'#menu?lesson=wire&track=main&group=workbench&from=game');assert.equal(app.eval('busy||paused'),false);assert.equal(app.eval('state===window.board&&history===window.undoStack'),true);
 app.c.finishAnimation();await running;assert.equal(app.eval('results.length'),0);assert.equal(!!doc.querySelector('#level-complete').open,false);assert.equal(doc.querySelector('#run-report').hidden,true);
 doc.querySelector('#levels-close').click();assert.equal(app.location.hash,'#wire');assert.equal(app.eval('state.sink'),'R');assert.equal(boot(app.storage,'#wire').eval('state.sink'),'R');
});

test('screen, route and stage URLs survive reload and paired browser history events without changing the board',()=>{
 const app=boot(),doc=app.document;app.c.setTimeout=fn=>fn();
 app.eval("checkpoint();state.sink='R';save();window.board=state;window.undoStack=history");
 const saved=new Map(app.storage),snapshots=[],snapshot=()=>({hash:app.location.hash,screen:doc.body.dataset.screen,track:doc.querySelector('#level-tracks [aria-selected=true]')?.dataset.track,group:doc.querySelector('#level-stage-nav [aria-current]')?.dataset.levelGroupTarget});
 const check=expected=>{
  assert.equal(app.location.hash,expected.hash);assert.equal(doc.body.dataset.screen,expected.screen);
  assert.equal(app.eval('state===window.board&&history===window.undoStack'),true);assert.deepEqual(app.storage,saved);
  assert.equal(doc.querySelector('#welcome').open,expected.screen==='home');assert.equal(doc.querySelector('#level-menu').open,expected.screen==='levels');
  if(expected.screen==='levels'){assert.equal(snapshot().track,expected.track);assert.equal(snapshot().group,expected.group);}
  const reload=boot(new Map(app.storage),app.location.hash);assert.equal(reload.document.body.dataset.screen,expected.screen);assert.equal(reload.location.hash,expected.hash);assert.equal(reload.eval('state.sink'),'R');
  if(expected.screen==='levels'){assert.equal(reload.document.querySelector('#level-tracks [aria-selected=true]').dataset.track,expected.track);assert.equal(reload.document.querySelector('#level-stage-nav [aria-current]')?.dataset.levelGroupTarget,expected.group);}
 };
 snapshots.push(snapshot());doc.querySelector('#enter-workshop').onclick();snapshots.push(snapshot());
 doc.querySelector('[data-track="modern"]').click();snapshots.push(snapshot());
 doc.querySelector('[data-track="main"]').click();snapshots.push(snapshot());
 doc.querySelector('[data-level-group-target="level-group-main-decoder"]').click();snapshots.push(snapshot());
 const count=app.navigation.length;doc.querySelector('[data-level-group-target="level-group-main-decoder"]').click();doc.querySelector('[data-track="main"]').click();app.hashchange();app.popstate();app.c.I18n.setPreference('en');
 assert.equal(app.navigation.length,count);assert.equal(snapshot().group,'level-group-main-decoder');
 // Language is a preference, not a circuit mutation or navigation entry.
 saved.set('neural-atelier-language-v1',app.storage.get('neural-atelier-language-v1'));
 doc.querySelector('[data-level="wire"]').click();snapshots.push(snapshot());assert.equal(app.navigation.length,snapshots.length);
 check(snapshots.at(-1));for(let i=snapshots.length-2;i>=0;i--){app.navigation.back();check(snapshots[i]);}
 for(let i=1;i<snapshots.length;i++){app.navigation.forward();check(snapshots[i]);}
 assert.equal(app.navigation.length,snapshots.length);assert.equal(app.location.pathname,'/games/neural-atelier/');assert.equal(app.location.search,'?version=campaign-1');
});

test('menu URLs validate lesson, route, stage and return target against curriculum metadata',()=>{
 const cases=[
  ['#menu','#menu?lesson=wire&track=main&group=workbench&from=home'],
  ['#menu?lesson=pack-ffn&track=constructor&group=__proto__&from=elsewhere','#menu?lesson=add-norm&track=main&group=foundations&from=home'],
  ['#menu?lesson=encoder-layer&track=main&group=decoder&from=game','#menu?lesson=encoder-layer&track=main&group=decoder&from=game'],
  ['#menu?lesson=wire&track=modern&group=decoder','#menu?lesson=wire&track=modern&from=home'],
  ['#menu?lesson=missing&track=engine&group=decoder&from=game','#menu?lesson=wire&track=engine&from=game'],
  ['#menu?lesson=%E0%A4%A','#home?lesson=wire'],
  ['#home?lesson=constructor','#home?lesson=wire'],
  ['#%77ire','#wire']
 ];
 for(const [input,expected]of cases){
  const app=boot(new Map(),input);assert.equal(app.location.hash,expected,input);assert.equal(app.navigation.length,1);
  const n=JSON.parse(app.eval('JSON.stringify(navigationState)'));assert.equal(app.document.body.dataset.screen,n.screen);
  if(n.screen==='levels'){app.document.querySelector('#levels-close').onclick();assert.equal(app.document.body.dataset.screen,n.returnScreen);assert.equal(app.eval('stageId'),n.levelId);}
 }
 const direct=boot(new Map(),'#encoder-layer');assert.equal(direct.document.body.dataset.screen,'game');assert.equal(direct.document.querySelector('#welcome').open,false);
 direct.eval('levelMenu()');direct.document.querySelector('[data-level-group-target="level-group-main-decoder"]').click();
 direct.navigation.back();assert.equal(direct.document.querySelector('#level-stage-nav [aria-current]').dataset.levelGroupTarget,'level-group-main-encoder');
 direct.navigation.forward();assert.equal(direct.document.querySelector('#level-stage-nav [aria-current]').dataset.levelGroupTarget,'level-group-main-decoder');
 assert.equal(direct.eval('stageId'),'encoder-layer');assert.equal(direct.document.querySelector('#level-list [aria-current]').dataset.level,'encoder-layer');
});

test('welcome autofocus belongs to the static heading while actions keep their native tab order',()=>{
 const app=boot(),doc=app.document,title=doc.querySelector('#welcome-title');
 assert.equal(doc.querySelector('#welcome').open,true);assert.equal(doc.querySelector('#welcome [autofocus]'),title);assert.equal(doc.querySelectorAll('#welcome [autofocus]').length,1);assert.equal(title.getAttribute('tabindex'),'-1');
 for(const id of ['enter-workshop','welcome-settings']){const button=doc.getElementById(id);assert.equal(button.tagName,'BUTTON');assert.equal(button.hasAttribute('tabindex'),false);assert.equal(button.hasAttribute('autofocus'),false);assert.equal(button.disabled,false);}
 let focused=0;const order=[],menu=doc.querySelector('#level-menu'),close=menu.close;menu.close=function(){order.push('menu-close');close.call(this);};title.focus=options=>{focused++;order.push('title-focus');assert.equal(options.preventScroll,true);};app.c.setTimeout=fn=>fn();
 doc.querySelector('#enter-workshop').click();app.navigation.back();assert.equal(doc.querySelector('#welcome').open,true);assert.equal(focused,1);assert.deepEqual(order,['menu-close','title-focus']);
 app.navigation.forward();doc.querySelector('#levels-close').click();assert.equal(doc.querySelector('#welcome').open,true);assert.equal(focused,2);
 app.eval("switchLevel('wire')");doc.querySelector('#home-open').click();assert.equal(doc.querySelector('#welcome').open,true);assert.equal(focused,3);
});

test('history cancels a pending welcome transition without opening the picker afterward',()=>{
 const app=boot(new Map(),'#wire'),doc=app.document,callbacks=[];app.c.setTimeout=fn=>callbacks.push(fn);
 doc.querySelector('#home-open').onclick();doc.querySelector('#enter-workshop').onclick();assert.equal(callbacks.length,1);assert.equal(doc.querySelector('#welcome').classList.contains('is-leaving'),true);
 app.navigation.back();const length=app.navigation.length;callbacks.shift()();
 assert.equal(app.location.hash,'#wire');assert.equal(doc.body.dataset.screen,'game');assert.equal(doc.querySelector('#welcome').open,false);assert.equal(doc.querySelector('#level-menu').open,false);assert.equal(app.navigation.length,length);
 app.navigation.forward();doc.querySelector('#enter-workshop').onclick();callbacks.shift()();assert.equal(doc.body.dataset.screen,'levels');
});

test('browser language is recommended on first visit and a saved choice overrides it',()=>{
 for(const [languages,expected] of [[['zh-TW'],'zh'],[['en-GB'],'en'],[['fr-FR','en-US','zh-CN'],'en'],[['ja-JP'],'en']]){
  const app=boot(new Map(),'',languages);assert.equal(app.c.I18n.language(),expected);assert.equal(app.document.documentElement.lang,expected==='zh'?'zh-CN':'en');
 }
 const storage=new Map([['neural-atelier-language-v1','zh']]);const saved=boot(storage,'',['en-US']);assert.equal(saved.c.I18n.language(),'zh');
 saved.c.I18n.setPreference('auto');assert.equal(saved.c.I18n.language(),'en');assert.equal(boot(storage,'',['en-US']).c.I18n.language(),'en');
});
test('language changes preserve the current circuit, results, level hash and speed',async()=>{
 const app=boot();app.eval("switchLevel('projection');state=M.solutions(stageId);state.nodes[0].x=1.75;setPlaybackSpeed(2)");await app.eval('run()');
 const before=app.eval('JSON.stringify(state)'),results=app.eval('JSON.stringify(results)'),hash=app.location.hash;
 app.c.I18n.setPreference('en');assert.equal(app.document.querySelector('#settings-title').textContent,'Game settings');assert.match(app.document.querySelector('#run').textContent,/Run my network/);assert.equal(app.eval('JSON.stringify(state)'),before);assert.equal(app.eval('JSON.stringify(results)'),results);assert.equal(app.location.hash,hash);assert.equal(app.eval('playbackSpeed'),2);assert.equal(app.eval('M.samples[1].word'),'love');assert.equal(app.eval("M.readTokens([0,1,2])"),'I love cats');
 app.c.I18n.setPreference('zh');assert.equal(app.document.querySelector('#settings-title').textContent,'游戏设置');assert.equal(app.eval('JSON.stringify(state)'),before);assert.equal(app.eval('M.samples[1].word'),'爱');
});
test('English covers every level, module card and generated report',async()=>{
 const app=boot(new Map(),'', ['en-US']),M=app.c.NeuralPrologue;
 for(const id of Object.keys(M.levels)){
  app.eval(`switchLevel('${id}');state=M.solutions(stageId);if(stage().engine)state.settings.precision=16;`);await app.eval('run()');
  assert.doesNotMatch(app.document.querySelector('.mission').textContent,/[\u3400-\u9fff]/,id+' mission');
  assert.doesNotMatch(app.document.querySelector('#run-report').textContent,/[\u3400-\u9fff]/,id+' report');
  for(const n of M.levels[id].nodes){app.eval(`select('${n}')`);const card=app.eval(`cardFor('${n}',stageId)||M.extraCard('${n}',stageId)`);assert.doesNotMatch(card.meaning+card.purpose,/[\u3400-\u9fff]/);}
 }
 assert.deepEqual([...app.c.I18n.missing],[]);
});
test('speed stays in the run toolbar while persisted audio controls remain in settings',()=>{
 const app=boot();assert.ok(app.document.querySelector('.run-toolbar #playback-speed'));assert.equal(app.document.querySelector('#settings #playback-speed'),null);
 app.document.querySelector('#settings-open').onclick();assert.equal(app.document.querySelector('#settings').open,true);
 const volume=app.document.querySelector('#music-volume');volume.value='20';volume.dispatchEvent(new app.window.Event('input',{bubbles:true}));app.document.querySelector('#music-enabled').onclick();
 assert.deepEqual(JSON.parse(app.storage.get('neural-atelier-audio-v1')),{enabled:false,volume:.2});const reloaded=boot(app.storage);assert.equal(reloaded.document.querySelector('#music-volume').value,'20');assert.equal(reloaded.document.querySelector('#music-enabled').getAttribute('aria-checked'),'false');
});

test('opening settings from the intro and skipping it never changes the saved circuit',()=>{
 const app=boot();app.c.setTimeout=fn=>fn();app.document.querySelector('#run').focus=()=>{};app.document.querySelector('#welcome-settings').focus=()=>{};app.document.querySelector('#settings-open').focus=()=>{};
 const welcome=app.document.querySelector('#welcome');assert.equal(welcome.open,true);const state=app.eval('JSON.stringify(state)');
 app.document.querySelector('#welcome-settings').onclick();assert.equal(app.document.querySelector('#settings').open,true);app.document.querySelector('#settings-done').onclick();assert.equal(app.document.querySelector('#settings').open,false);assert.equal(welcome.open,true);
 app.document.querySelector('#enter-workshop').onclick();assert.equal(welcome.open,false);assert.equal(app.eval('JSON.stringify(state)'),state);app.document.querySelector('#intro-replay').onclick();assert.equal(welcome.open,true);assert.equal(app.eval('JSON.stringify(state)'),state);
 app.document.querySelector('#enter-workshop').onclick();let prevented=false;const hash=app.location.hash;app.document.querySelector('#home-open').onclick({preventDefault(){prevented=true;}});assert.ok(prevented);assert.equal(welcome.open,true);assert.equal(app.eval('JSON.stringify(state)'),state);assert.notEqual(app.location.hash,hash);assert.equal(app.location.hash,'#home?lesson=wire');
});

test('entering the workshop opens a neutral level picker at the current saved level until a card is chosen',()=>{
 const first=boot();first.eval("switchLevel('qkv');state=M.solutions(stageId);state.nodes[0].x+=.2;save();progress.complete('wire');setPlaybackSpeed(.5)");
 const app=boot(first.storage),doc=app.document;app.c.setTimeout=fn=>fn();
 const before=new Map(app.storage),circuit=app.eval('JSON.stringify(state)'),hash=app.location.hash;
 assert.equal(doc.body.dataset.screen,'home');assert.equal(app.eval('stageId'),'qkv');
 doc.querySelector('#enter-workshop').onclick();
 assert.equal(doc.querySelector('#welcome').open,false);assert.equal(doc.querySelector('#level-menu').open,true);assert.equal(doc.body.dataset.screen,'levels');
 assert.equal(doc.querySelector('#level-list .current').dataset.level,'qkv');assert.equal(doc.querySelector('#level-list [aria-current="step"]').dataset.level,'qkv');
 assert.equal(doc.querySelector('#level-tracks [aria-selected=true]').dataset.track,'main');
 doc.querySelector('[data-track="modern"]').click();assert.equal(doc.body.dataset.screen,'levels');assert.notEqual(app.location.hash,hash);assert.equal(app.location.hash,'#menu?lesson=qkv&track=modern&from=home');assert.equal(app.eval('JSON.stringify(state)'),circuit);
 doc.querySelector('#levels-close').onclick();assert.equal(doc.body.dataset.screen,'home');assert.equal(doc.querySelector('#welcome').open,true);
 doc.querySelector('#enter-workshop').onclick();assert.equal(doc.querySelector('#level-list .current').dataset.level,'qkv');
 doc.querySelector('[data-level="qkv"]').click();assert.equal(doc.body.dataset.screen,'game');assert.equal(doc.querySelector('#level-menu').open,false);assert.equal(app.eval('JSON.stringify(state)'),circuit);assert.deepEqual(app.storage,before);
 doc.querySelector('#levels-open').onclick();assert.equal(doc.body.dataset.screen,'levels');doc.querySelector('#levels-close').onclick();assert.equal(doc.body.dataset.screen,'game');assert.equal(doc.querySelector('#welcome').open,false);
 app.eval('busy=true;paused=true');doc.querySelector('#home-open').onclick();assert.equal(app.eval('busy||paused'),false,'returning home ends playback so the picker can enter any level');doc.querySelector('#enter-workshop').onclick();
 doc.querySelector('[data-level="scores"]').click();assert.equal(doc.body.dataset.screen,'game');assert.equal(app.eval('stageId'),'scores');assert.equal(app.location.hash,'#scores');assert.equal(app.storage.get('neural-atelier-level-qkv-v1'),before.get('neural-atelier-level-qkv-v1'));
});

test('direct menu links restore their level and dismiss to home by Escape or backdrop',()=>{
 const app=boot(new Map(),'#menu?lesson=kv-cache&track=engine&from=home',['en-US']),doc=app.document,dialog=doc.querySelector('#level-menu');app.c.setTimeout=fn=>fn();
 const enter=()=>doc.querySelector('#enter-workshop').onclick(),saved=app.eval('JSON.stringify(state)');
 assert.equal(doc.querySelector('#level-tracks [aria-selected=true]').dataset.track,'engine');assert.equal(doc.querySelector('#level-list .current').dataset.level,'kv-cache');
 assert.doesNotMatch(dialog.textContent,/[\u3400-\u9fff]/);
 dialog.dispatchEvent(new app.window.Event('cancel',{cancelable:true}));assert.equal(doc.body.dataset.screen,'home');assert.equal(doc.querySelector('#welcome').open,true);
 enter();dialog.getBoundingClientRect=()=>({left:100,right:900,top:60,bottom:700});
 const fire=type=>{const e=new app.window.Event(type,{bubbles:true,cancelable:true});Object.assign(e,{button:0,pointerId:1,clientX:20,clientY:200});dialog.dispatchEvent(e);};
 fire('pointerdown');fire('click');assert.equal(dialog.open,false);assert.equal(doc.body.dataset.screen,'home');assert.equal(doc.querySelector('#welcome').open,true);assert.equal(app.eval('JSON.stringify(state)'),saved);assert.equal(app.location.hash,'#home?lesson=kv-cache');
});

test('pause is bilingual and a hash change cancels it without carrying pause into another level',async()=>{
 const app=boot();app.c.performance={now:()=>100};
 const control=app.document.querySelector('#pause');assert.equal(control.disabled,true);assert.equal(control.parentElement.className,'run-toolbar');
 app.eval('busy=true;refresh();togglePause()');assert.equal(control.getAttribute('aria-pressed'),'true');assert.equal(control.textContent,'▶ 继续');
 app.c.I18n.setPreference('en');assert.equal(control.textContent,'▶ Resume');
 app.location.hash='#projection';app.hashchange();assert.equal(app.eval('busy||paused'),false);assert.equal(control.disabled,true);assert.equal(control.textContent,'Ⅱ Pause');
 app.eval('state=M.solutions(stageId)');await app.eval('run()');assert.equal(app.eval('paused'),false);assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);
});

test('multiple module saves preserve instance identities and undo refunds only the removed copy',()=>{
 const app=boot();assert.equal(app.document.querySelector('#shuffle-layout'),null);
 app.eval("switchLevel('ffn');chooseModule('project');placeModule(-3,-2);chooseModule('project');placeModule(1,-2);chooseOutput('Z');chooseInput('project-2',0)");
 assert.equal(app.eval('selected'),'project-2');assert.equal(app.document.querySelector('[data-add-module="project"]').getAttribute('data-add-disabled'),'false');assert.equal(app.eval("M.stock(state,'project')"),2);
 const saved=app.eval('JSON.stringify(state)'),reloaded=boot(app.storage,'#ffn');assert.equal(reloaded.eval('JSON.stringify(state)'),saved);
 reloaded.eval("select('project');removeModule()");assert.equal(reloaded.eval("state.nodes[0].id"),'project-2');assert.equal(reloaded.eval("state.nodes[0].links[0]"),'Z');assert.equal(reloaded.document.querySelector('[data-add-module="project"]').getAttribute('data-add-disabled'),'false');
 reloaded.eval("$('#undo').onclick()");assert.equal(reloaded.eval('JSON.stringify(state)'),saved);
 const corrupt=JSON.parse(saved);corrupt.nodes[1].kind='down';const storage=new Map([['neural-atelier-prologue-v1',JSON.stringify(corrupt)]]);assert.equal(boot(storage,'#ffn').eval('state.nodes.length'),0,'invalid ID/type pair does not become a different operator');
});
test('legacy content projections merge without losing coexisting copies, wiring, positions or completion',async()=>{
 const seed=boot(),M=seed.c.NeuralPrologue;
 const snapshot=s=>JSON.stringify(s.nodes.map(n=>[n.id,n.x,n.z,n.links]));
 for(const id of ['gating','ffn']){
  const solved=M.solutions(id),old=JSON.parse(JSON.stringify(solved));
  const linear=old.nodes.find(n=>n.id==='project');linear.id='up';linear.kind='up';linear.x=2.345;linear.z=-1.234;
  for(const n of old.nodes)n.links=n.links.map(k=>k==='project'?'up':k);
  old.palette=['up',...solved.palette];
  if(id==='ffn'){
   // Both earlier stocks can coexist. References to the spare copy must also survive.
   old.nodes.push({id:'project',kind:'project',x:6,z:3,links:['R']},{id:'up-2',kind:'up',x:6,z:-3,links:['R']});
   old.nodes.find(n=>n.id==='gate').links=['Z'];
  }else for(const n of old.nodes)delete n.kind;
  const key=id==='ffn'?'neural-atelier-prologue-v1':'neural-atelier-level-gating-v1';
  const store=new Map([[key,JSON.stringify(old)],['neural-atelier-chapters-v1',JSON.stringify({current:id,completed:[id]})],['neural-atelier-language-v1','en'],['neural-atelier-playback-speed-v1','.5']]);
  const app=boot(store,'#'+id),migrated=app.eval('state'),content=id==='ffn'?'project-2':'project';
  assert.equal(migrated.nodes.length,old.nodes.length);assert.equal(app.eval('playbackSpeed'),.5);assert.equal(app.eval('progress.completed(stageId)'),true);
  assert.equal(migrated.nodes.find(n=>n.id===content).x,linear.x);assert.equal(migrated.nodes.find(n=>n.id===content).z,linear.z);
  assert.equal(migrated.nodes.find(n=>n.id==='multiply').links[1],content);
  assert.ok(migrated.nodes.every(n=>n.kind!=='up'&&!/^up(?:-|$)/.test(n.id)));
  assert.equal(migrated.palette.filter(k=>k==='project').length,1);assert.ok(!migrated.palette.includes('up'));
  if(id==='ffn'){assert.equal(migrated.nodes.find(n=>n.id==='project').links[0],'R');assert.equal(migrated.nodes.find(n=>n.id==='project-3').links[0],'R');assert.equal(app.eval("M.stock(state,'project')"),1);}
  const saved=snapshot(migrated);await app.eval('run()');assert.ok(app.eval('results.every(r=>r.pass)'));assert.equal(snapshot(app.eval('state')),saved);
  const reloaded=boot(store,'#'+id);assert.equal(snapshot(reloaded.eval('state')),saved);
  if(id==='ffn'){
   reloaded.eval("select('project-2');removeModule()");assert.equal(reloaded.eval("state.nodes.find(n=>n.id==='multiply').links[1]"),null);assert.equal(reloaded.eval("M.stock(state,'project')"),2);
   reloaded.eval("$('#undo').onclick()");assert.equal(snapshot(reloaded.eval('state')),saved);
  }
 }
 // All four former copies are retained, including a saved outlet into an old content copy.
 const full={version:1,levelId:'ffn',palette:['up','project'],sink:'up-2',nodes:['project','up','project-2','up-2'].map((id,i)=>({id,kind:id.startsWith('up')?'up':'project',x:i*3-4,z:1,links:['R']}))};
 const all=boot(new Map([['neural-atelier-prologue-v1',JSON.stringify(full)]]),'#ffn');
 assert.deepEqual(Array.from(all.eval('state.nodes'),n=>n.id),['project','project-3','project-2','project-4']);assert.equal(all.eval('state.sink'),'project-4');assert.equal(all.eval("M.stock(state,'project')"),0);
 all.eval('save()');assert.equal(boot(all.storage,'#ffn').eval('state.sink'),'project-4');
});

test('SwiGLU objective connects familiar modules to the FFN while keeping residual addition outside it',()=>{
 const app=boot(new Map(),'#ffn'),doc=app.document,saved=app.eval('JSON.stringify(state)');
 for(const language of ['zh','en']){
  app.c.I18n.setPreference(language);
  const primer=doc.querySelector('.swiglu-primer');assert.ok(primer);assert.match(primer.textContent,/SwiGLU/);assert.match(primer.textContent,language==='zh'?/归一化输入和残差/:/normalized input and residual/);assert.match(primer.textContent,/SiLU/);
  assert.equal(primer.querySelectorAll('.swiglu-branches>div').length,2);assert.equal(primer.querySelectorAll('.swiglu-merge b').length,2);
  assert.match(doc.querySelector('#mission-reason-takeaway').textContent,language==='zh'?/FFN 外/:/outside the FFN/);
  assert.match(doc.querySelector('#mission-reason-scope').textContent,/Swish/);
  assert.equal(doc.querySelectorAll('[data-add-module="project"]').length,1);assert.equal(doc.querySelector('[data-add-module="up"]'),null);
  assert.doesNotMatch(doc.querySelector('#module-shelf').textContent,/内容投影|Content projection/);
  if(language==='en')assert.doesNotMatch(doc.querySelector('.mission').textContent,/[\u3400-\u9fff]/);
  assert.equal(app.eval('JSON.stringify(state)'),saved);
 }
 app.eval("switchLevel('gating')");assert.equal(doc.querySelector('.swiglu-primer'),null);assert.ok(doc.querySelector('#mission-reason-example table'));
 app.eval("switchLevel('projection')");assert.equal(doc.querySelector('#mission-reason-example').hidden,true);
 app.eval("switchLevel('ffn')");assert.ok(doc.querySelector('.swiglu-primer'));assert.equal(app.eval('JSON.stringify(state)'),saved);
 assert.deepEqual([...app.c.I18n.missing],[]);
});
test('the simple placement lesson introduces one familiar module before the original network chapter',()=>{
 const app=boot(new Map(),'#workshop',['en-US']);assert.equal(app.eval('state.nodes.length'),0);assert.equal(app.document.querySelectorAll('[data-add-module]').length,1);assert.equal(app.document.querySelector('#module-count').textContent,'0 / 1');assert.equal(app.eval('M.levels.workshop.next'),'residual');assert.equal(app.eval('M.levels.residual.next'),'embedding');assert.doesNotMatch(app.document.querySelector('#module-shelf').textContent,/[\u3400-\u9fff]/);
});

test('the shelf remains usable after the placement tutorial and original lessons begin empty',()=>{
 const app=boot(),M=app.c.NeuralPrologue,shelf=app.document.querySelector('#module-shelf');
 for(const [index,id]of M.routes.main.ids.entries()){
  app.eval(`switchLevel('${id}')`);assert.equal(shelf.hidden,index<5||!!M.levels[id].demonstration,id+' shelf visibility');if(M.levels[id].demonstration)continue;
  if(index>=6){assert.equal(app.eval('state.nodes.length'),0,id+' requires taking modules from the shelf');assert.equal(app.eval('state.sink'),null);}
 }
 for(const id of [...M.routes.modern.ids,...M.routes.engine.ids]){
  app.eval(`switchLevel('${id}')`);assert.equal(shelf.hidden,false,id);
  assert.equal(app.document.querySelectorAll('[data-add-module]').length,Object.keys(M.inventory(M.levels[id])).filter(kind=>M.fusions[id]?.kind!==kind||app.eval(`progress.completed('${id}')`)).length,id+' exposes unlocked stock');
 }
 app.eval("switchLevel('normalize')");assert.equal(app.eval('state.nodes.length'),1,'prepared arithmetic lessons keep their starting modules');
 shelf.removeAttribute('open');app.eval("switchLevel('embedding');switchLevel('normalize')");assert.equal(shelf.hidden,false);assert.equal(shelf.hasAttribute('open'),false,'the player can still collapse the persistent shelf');
 app.eval("switchLevel('workshop')");assert.equal(app.eval('state.nodes.length'),0,'the placement tutorial still requires taking a module');
});

test('encoder groups old parts without hiding useful distractors or prewiring the challenge',()=>{
 const app=boot(new Map(),'#encoder-layer'),doc=app.document;
 const current=()=>[...doc.querySelectorAll('.shelf-current [data-add-module]')].map(e=>e.dataset.addModule);
 assert.deepEqual(current(),['self-heads','original-ffn-block','encoder-add','ln-attn','ffn-add','ln-ffn','wide-position']);
 assert.ok(doc.querySelector('[data-add-module="ffn-add"]'));assert.ok(doc.querySelector('[data-add-module="ln-ffn"]'));
 assert.equal(doc.querySelector('[data-add-module="original-encoder"]'),null,'the answer package still requires a win');
 assert.equal(doc.querySelector('[data-earlier-parts]').hasAttribute('open'),false);
 assert.ok(doc.querySelector('.shelf-earlier [data-add-module="v"]'),'earlier learned choices remain inspectable and usable');
 assert.ok(doc.querySelector('.shelf-current [data-add-module="wide-position"]'),'same-width distractor remains in the compact view');
 assert.equal(app.eval('state.nodes.length'),0);assert.equal(app.eval('state.sink'),null);
 doc.querySelector('[data-earlier-parts]').setAttribute('open','');app.eval("chooseModule('v');placeModule(0,0)");
 assert.equal(app.eval("state.nodes[0].kind"),'v');assert.equal(doc.querySelector('[data-earlier-parts]').hasAttribute('open'),true);
 const saved=app.eval('JSON.stringify(state)');
 for(const language of ['en','zh']){
  app.c.I18n.setPreference(language);
  assert.equal(doc.querySelectorAll('[data-encoder-job]').length,2);assert.equal(doc.querySelectorAll('.encoder-job svg').length,2);assert.equal(doc.querySelectorAll('.encoder-stage-path').length,2);assert.ok([...doc.querySelectorAll('.encoder-stage-path')].every(el=>el.textContent.endsWith('LayerNorm')));
  assert.equal(doc.querySelector('.encoder-formulas').hasAttribute('open'),false);
  assert.equal(doc.querySelector('[data-earlier-parts]').hasAttribute('open'),true);
  assert.equal(app.eval('JSON.stringify(state)'),saved);
  if(language==='en')for(const selector of ['.mission','#module-shelf'])assert.doesNotMatch(doc.querySelector(selector).textContent,/[\u3400-\u9fff]/);
 }
 assert.deepEqual([...app.c.I18n.missing],[]);
 app.eval("switchLevel('causal')");assert.equal(doc.querySelector('.encoder-primer'),null);assert.ok(doc.querySelector('[data-earlier-parts]'));assert.equal(doc.querySelector('[data-earlier-parts]').hasAttribute('open'),false);
});

test('encoder retains saved named copies, wiring and badges with independent role stock',async()=>{
 const base=boot(),M=base.c.NeuralPrologue,old=M.solutions('encoder-layer');old.nodes[0].x=-3.19;
 const key='neural-atelier-level-encoder-layer-v1',meta='neural-atelier-chapters-v1';
 const storage=new Map([[key,JSON.stringify(old)],[meta,JSON.stringify({current:'encoder-layer',completed:['multihead','encoder-layer']})]]),app=boot(storage,'#encoder-layer');
 assert.equal(app.eval('JSON.stringify(state)'),JSON.stringify(old));assert.equal(app.eval("progress.completed('encoder-layer')"),true);
 assert.equal(app.document.querySelector('[data-add-module="encoder-add"] .stock-count').textContent,'1');
 assert.ok(app.document.querySelector('.shelf-current [data-add-module="original-encoder"]'),'earned composite stays visible for collection');
 await app.eval('run()');assert.equal(app.eval('levelPassed()'),true);assert.equal(app.eval('JSON.stringify(state)'),JSON.stringify(old));
 app.eval("removeModule('ffn-add')");assert.equal(app.document.querySelector('[data-add-module="ffn-add"] .stock-count').textContent,'2');assert.equal(app.document.querySelector('[data-add-module="encoder-add"] .stock-count').textContent,'1');
 app.document.querySelector('#undo').onclick();assert.equal(app.eval('JSON.stringify(state)'),JSON.stringify(old));
 const extra=JSON.parse(JSON.stringify(old));extra.nodes.push({id:'encoder-add-2',kind:'encoder-add',x:6,z:3,links:[null,null]});
 const legacy=boot(new Map([[key,JSON.stringify(extra)]]),'#encoder-layer');
 assert.equal(legacy.eval('JSON.stringify(state)'),JSON.stringify(extra),'preexisting extra copies and their placement are kept');
 assert.equal(legacy.eval("M.stock(state,'encoder-add')"),0);
});

test('a lesson with a single reference module retains choices and saves manual placement',()=>{
 const app=boot(new Map(),'#layernorm'),M=app.c.NeuralPrologue;
 assert.equal(app.eval('state.nodes.length'),0);assert.ok(app.document.querySelectorAll('[data-add-module]').length>1);
 assert.equal(app.document.querySelector('#module-count').textContent,'0 / 3');
 app.eval("chooseModule('layernorm2');placeModule(0,0)");assert.equal(app.eval('state.nodes.length'),1);assert.equal(app.eval("M.canAdd(state,'layernorm2')"),true);
 const saved=app.eval('JSON.stringify(state)'),reopened=boot(app.storage,'#layernorm');assert.equal(reopened.eval('JSON.stringify(state)'),saved);
 reopened.document.querySelector('#reset').onclick();assert.equal(reopened.eval('state.nodes.length'),0);assert.equal(reopened.eval("M.stock(state,'layernorm2')"),2);
});

test('expanded shelves preserve an old circuit, palette order and completion badges',()=>{
 const base=boot(),M=base.c.NeuralPrologue,old=M.solutions('scores');
 old.palette=['k','score','q'];old.nodes[0].x=-3.25;old.nodes[0].z=1.75;
 const key='neural-atelier-level-scores-v1',storage=new Map([[key,JSON.stringify(old)],['neural-atelier-chapters-v1',JSON.stringify({current:'scores',completed:['qkv','scores','softmax']})]]);
 const app=boot(storage,'#scores'),state=JSON.parse(app.eval('JSON.stringify(state)'));
 assert.deepEqual(state.nodes,JSON.parse(JSON.stringify(old.nodes)));assert.equal(state.sink,old.sink);
 assert.deepEqual(state.palette.slice(0,3),old.palette);assert.ok(state.palette.includes('v'));
 assert.equal(app.eval("progress.completed('qkv')&&progress.completed('scores')&&progress.completed('softmax')"),true);
 assert.ok(app.document.querySelector('[data-add-module="v"]'));assert.equal(app.document.querySelector('[data-add-module="softmax"]'),null);
});

test('scoring reuse explains the existing module in both languages',()=>{
 for(const language of ['zh-CN','en-US'])for(const [id,kind]of [['pack-scores','original-weights']]){
  const app=boot(new Map(),'#'+id,[language]),doc=app.document;
  assert.equal(app.eval('state.nodes.length'),0);assert.equal(app.eval('state.sink'),null);
  assert.ok(doc.querySelector('.package-primer[open]'));assert.ok(doc.querySelector('[data-package="'+kind+'"]'));
  assert.match(doc.querySelector('#mission-reason-example').textContent,/QKᵀ/);
  if(language==='en-US')assert.doesNotMatch(doc.querySelector('.mission').textContent,/[\u3400-\u9fff]/);
 }
});

test('old auto-filled original starts are archived once and can be restored without running',()=>{
 const fixture=boot(),M=fixture.c.NeuralPrologue;
 for(const id of ['embedding','positions','layernorm','softmax','vocabulary-scores']){
  const legacy=M.initial(id,{complete:true,random:()=>.5});delete legacy.placementRevision;
  const key='neural-atelier-level-'+id+'-v1',raw=JSON.stringify(legacy),storage=new Map([[key,raw]]),app=boot(storage,'#'+id);
  assert.equal(app.eval('state.nodes.length'),0,id);assert.equal(storage.get(key+'-before-placement'),raw);
  const restore=app.document.querySelector('#restore-layout');assert.equal(restore.hidden,false);restore.onclick();
  assert.equal(app.eval('busy'),false);assert.equal(app.eval('JSON.stringify(state.nodes)'),JSON.stringify(legacy.nodes));
  const reopened=boot(storage,'#'+id);assert.equal(reopened.eval('state.nodes.length'),legacy.nodes.length,'explicitly restored layouts are not retired again');
  app.document.querySelector('#undo').onclick();assert.equal(app.eval('state.nodes.length'),0,'restoring remains undoable');
 }
});
test('readout lesson order, completion navigation and bilingual copy agree on one four-feature pipeline',async()=>{
 const app=boot(new Map(),'#vocabulary-scores'),doc=app.document;
 for(const language of ['zh','en']){
  app.c.I18n.setPreference(language);app.eval('levelMenu()');
  assert.deepEqual([...doc.querySelectorAll('[data-level-group="generation"] [data-level]')].map(el=>el.dataset.level),['vocabulary-scores','encoder-decoder','vocabulary','autoregressive','transformer-demo']);
  for(const [id,number]of [['vocabulary-scores',17],['encoder-decoder',18],['vocabulary',19]]){
   app.eval(`switchLevel('${id}')`);assert.equal(doc.querySelector('.route strong').textContent,app.eval('stage().outputTitle'));assert.match(doc.querySelector('.chapter strong').textContent,id==='vocabulary-scores'?/打分|score/i:id==='encoder-decoder'?/两半|halves/i:/选|Select/i);
   assert.equal(app.eval('stage().chapter'),(language==='zh'?'原版基础 / ':'Original / ')+number);
   if(id!=='encoder-decoder')assert.match(doc.querySelector('#sample-value').textContent,/4/);
   if(language==='en')assert.doesNotMatch(doc.querySelector('.mission').textContent,/[\u3400-\u9fff]/);
  }
 }
 app.eval("switchLevel('vocabulary-scores');state=M.solutions(stageId)");await app.eval('run()');assert.ok(doc.querySelector('#level-complete').open);
 doc.querySelector('#complete-next').onclick();assert.equal(app.location.hash,'#encoder-decoder');
 app.eval('state=M.solutions(stageId)');await app.eval('run()');doc.querySelector('#complete-next').onclick();assert.equal(app.location.hash,'#vocabulary');
 app.eval('state=M.solutions(stageId)');await app.eval('run()');assert.equal(doc.querySelectorAll('.vocab-scores>div').length,4);assert.equal(app.eval('levelPassed()'),true);
 assert.deepEqual([...app.c.I18n.missing],[]);
});
test('two-feature readout saves upgrade once without losing wires, positions, duplicate modules or badges',()=>{
 const setup=boot(),M=setup.c.NeuralPrologue;
 for(const id of ['vocabulary-scores','vocabulary']){
  const saved=JSON.parse(JSON.stringify(M.solutions(id))),aliases={'original-head':'lmhead','wide-invert':'invert'},rename=name=>aliases[name]||name;
  for(const n of saved.nodes){n.kind=rename(n.kind);n.id=rename(n.id);n.links=n.links.map(rename);}saved.sink=rename(saved.sink);saved.palette=saved.palette.map(rename);
  saved.nodes[0].x=-2.71;saved.nodes.push({id:'original-head',kind:'original-head',x:3.2,z:1.1,links:[null]});
  if(id==='vocabulary-scores')saved.nodes.push({id:'original-transformer',kind:'original-transformer',x:4,z:-1,links:[null,null]});
  const key='neural-atelier-level-'+id+'-v1',raw=JSON.stringify(saved),storage=new Map([[key,raw],['neural-atelier-chapters-v1',JSON.stringify({current:id,completed:['decoder-layer','encoder-decoder',id]})]]),app=boot(storage,'#'+id);
  const state=JSON.parse(app.eval('JSON.stringify(state)'));assert.equal(storage.get(key+'-before-four-feature-readout'),raw);
  assert.equal(state.nodes.length,saved.nodes.length);assert.deepEqual(state.nodes.map(n=>[n.x,n.z]),saved.nodes.map(n=>[n.x,n.z]));
  assert.equal(state.nodes[0].id,'original-head-2');assert.equal(state.nodes[0].kind,'original-head');assert.equal(state.nodes[0].links[0],'R');
  if(id==='vocabulary'){assert.equal(state.nodes.find(n=>n.id==='pick').links[0],'original-head-2');assert.ok(state.nodes.some(n=>n.kind==='wide-invert'));}
  else{assert.equal(state.sink,'original-head-2');assert.ok(state.nodes.some(n=>n.kind==='original-transformer'));assert.equal(app.document.querySelector('[data-add-module="original-transformer"]'),null);}
  assert.equal(app.eval('progress.completed(stageId)'),true);assert.equal(app.eval("progress.completed('encoder-decoder')"),true);
  const reopened=boot(storage,'#'+id);assert.equal(reopened.eval('JSON.stringify(state)'),app.eval('JSON.stringify(state)'));assert.equal(storage.get(key+'-before-four-feature-readout'),raw);
  assert.equal(app.eval('M.evaluate(state,M.samplesFor(stageId)[0].v).error'),null);
 }
});

test('original placement migration preserves wired, completed and partly built saves',()=>{
 const fixture=boot(),M=fixture.c.NeuralPrologue;
 for(const mode of ['wired','completed','partial']){
  const state=mode==='wired'?M.solutions('positions'):M.initial('positions',{complete:true});delete state.placementRevision;
  if(mode==='partial')state.nodes.pop();
  const key='neural-atelier-level-positions-v1',storage=new Map([[key,JSON.stringify(state)]]);
  if(mode==='completed')storage.set('neural-atelier-chapters-v1',JSON.stringify({current:'positions',completed:['positions']}));
  const app=boot(storage,'#positions');assert.equal(app.eval('JSON.stringify(state.nodes)'),JSON.stringify(state.nodes),mode);assert.equal(app.eval('state.sink'),state.sink);assert.equal(storage.has(key+'-before-placement'),false);
  if(mode==='completed')assert.equal(app.eval("progress.completed('positions')"),true);
 }
});

test('a failed archive write leaves the former original board intact',()=>{
 const app=boot(),legacy=app.eval("M.initial('embedding',{complete:true})");delete legacy.placementRevision;
 const key='neural-atelier-level-embedding-v1',raw=JSON.stringify(legacy),storage=new Map([[key,raw]]);
 app.c.limitedStorage={getItem:k=>storage.get(k)||null,setItem:(k,v)=>{if(k.endsWith('-before-placement'))throw new Error('full');storage.set(k,v);}};
 assert.equal(app.eval("JSON.stringify(createProgress(M,limitedStorage).load('embedding').nodes)"),JSON.stringify(legacy.nodes));assert.equal(storage.get(key),raw);
});

test('formerly fixed saved modules can be returned, placed and reconnected with undo and engine defaults',()=>{
 const first=boot(),storage=new Map(),saved=first.eval("M.solutions('normalize')");saved.nodes[0].x=-.75;saved.nodes[0].z=.8;
 storage.set('neural-atelier-level-normalize-v1',JSON.stringify(saved));const app=boot(storage,'#normalize');
 assert.equal(app.eval('state.nodes[0].x'),-.75);assert.equal(app.eval('state.sink'),'norm');
 app.eval("select('norm');removeModule()");assert.equal(app.eval('state.nodes.length'),0);assert.equal(app.eval('state.sink'),null);assert.equal(app.eval("M.stock(state,'norm')"),2);
 app.eval("chooseModule('norm');placeModule(0,0);link('R','norm',0);link('norm','sink',0)");assert.equal(app.eval("M.stock(state,'norm')"),1);
 const again=boot(storage,'#normalize');assert.equal(again.eval('state.nodes.length'),1);assert.equal(again.eval('state.nodes[0].x'),0);assert.equal(again.eval('state.sink'),'norm');
 app.eval("$('#undo').onclick()");assert.equal(app.eval('state.sink'),null);assert.equal(app.eval('state.nodes[0].links[0]'),'R');
 app.eval("switchLevel('placement');const removed=state.nodes.find(n=>n.id==='enqkv');const position=[removed.x,removed.z];removeModule('enqkv');chooseModule('enqkv');placeModule(...position)");
 assert.equal(app.eval("state.nodes.find(n=>n.id==='enqkv').device"),app.c.NeuralPrologue.levels.placement.devices?.enqkv||'gpu');
 assert.equal(app.eval("M.canAdd(state,'enqkv')"),false,'returning a task does not enlarge its allowed stock');
});

test('reordered chapters retain completion and advanced saves while finished prologues restart',()=>{
 const storage=new Map(),app=boot(storage),kept=new Map();
 for(const id of ['activation','gating','residual','workshop','ffn','normalize','attention']){
  app.eval(`switchLevel('${id}');state=M.solutions(stageId);state.nodes[0].x+=.17;save();progress.complete(stageId)`);
  kept.set(id,app.eval('JSON.stringify(state)'));
 }
 const reopened=boot(storage,'#gating');assert.equal(reopened.location.hash,'#gating');
 for(const [id,circuit]of kept){
  reopened.eval(`switchLevel('${id}')`);assert.equal(reopened.eval('progress.completed(stageId)'),true);
  if(reopened.c.NeuralPrologue.levels[id].practice){
   assert.equal(reopened.eval('state.sink'),null);assert.ok(reopened.eval('state.nodes.every(n=>n.links.every(x=>x===null))'));
   assert.equal(reopened.document.querySelector('#restore-layout').hidden,false);reopened.document.querySelector('#restore-layout').onclick();
  }else assert.equal(reopened.document.querySelector('#restore-layout').hidden,true);
  assert.equal(reopened.eval('JSON.stringify(state)'),circuit);
 }
 reopened.eval("switchLevel('down-projection')");assert.equal(reopened.eval('progress.completed(stageId)'),false);assert.equal(reopened.eval('state.sink'),null);
 reopened.eval("switchLevel('gating');levelMenu()");assert.match(reopened.document.querySelector('[data-level="gating"]').textContent,/现代扩展/);
 reopened.c.I18n.setPreference('en');assert.doesNotMatch(reopened.document.querySelector('#level-list').textContent,/[\u3400-\u9fff]/);
 assert.deepEqual([...reopened.c.I18n.missing],[]);
});

test('every fresh prologue requires wiring and the placement lesson requires taking a module',async()=>{
 const app=boot(),M=app.c.NeuralPrologue,ids=['wire','projection','relu','down-projection','original-ffn','workshop','residual'];
 assert.deepEqual(Array.from(M.routes.main.ids).filter(id=>M.levels[id].practice),ids);
 for(const id of ids){
  app.eval(`switchLevel('${id}')`);
  assert.equal(app.eval('state.sink'),null,id+' starts with no output wire');
  assert.ok(app.eval('state.nodes.every(n=>n.links.every(x=>x===null))'),id+' has no preset connections');
  assert.equal(app.eval('state.nodes.length'),M.levels[id].buildable?0:M.levels[id].nodes.length);
  assert.equal(app.document.querySelector('#restore-layout').hidden,true,'no previous solution is supplied');
  await app.eval('run()');assert.equal(app.eval('levelPassed()'),false,id+' cannot pass without playing');
  assert.ok(!app.document.querySelector('#level-complete').open);assert.equal(app.eval('progress.completed(stageId)'),false);
 }
});

test('a legacy completed workshop resumes on refresh and deliberate replay offers restore with undo',()=>{
 const fixture=boot(),solution=fixture.c.NeuralPrologue.solutions('workshop');solution.nodes[0].x=.42;solution.nodes[0].z=-.31;
 const saved=JSON.stringify(solution),storage=new Map([
  ['neural-atelier-chapters-v1',JSON.stringify({current:'workshop',completed:['workshop']})],
  ['neural-atelier-level-workshop-v1',saved]
 ]),app=boot(storage,'#workshop',['en-US']);
 assert.equal(app.eval('JSON.stringify(state)'),saved);assert.equal(JSON.parse(storage.get('neural-atelier-chapters-v1')).practiceRuns.workshop.finished,true);
 app.eval("switchLevel('rope');switchLevel('workshop')");const button=app.document.querySelector('#restore-layout'),fresh=app.eval('JSON.stringify(state)');
 assert.equal(app.eval('state.nodes.length'),0);assert.equal(app.eval('state.sink'),null);assert.equal(app.eval('progress.completed(stageId)'),true);
 assert.equal(storage.get('neural-atelier-level-workshop-v1-completed'),saved);assert.equal(app.document.querySelector('#module-count').textContent,'0 / 1');
 assert.equal(button.hidden,false);assert.equal(button.textContent,'Restore previous solution');
 app.eval('busy=true;refresh()');assert.equal(button.disabled,true);button.onclick();assert.equal(app.eval('JSON.stringify(state)'),fresh);
 app.eval('busy=false;refresh();window.runs=0;world.animate=async()=>runs++');button.onclick();
 assert.equal(app.eval('JSON.stringify(state)'),saved);assert.equal(button.hidden,true);assert.equal(app.eval('busy'),false);assert.equal(app.c.runs,0);
 app.document.querySelector('#undo').onclick();assert.equal(app.eval('JSON.stringify(state)'),fresh);assert.equal(button.hidden,false);
 app.c.I18n.setPreference('zh');assert.equal(button.textContent,'恢复上次方案');button.onclick();assert.equal(app.eval('JSON.stringify(state)'),saved);
 app.document.querySelector('#reset').onclick();assert.equal(app.eval('state.nodes.length'),0);assert.equal(app.eval('state.sink'),null);
 app.document.querySelector('#undo').onclick();assert.equal(app.eval('JSON.stringify(state)'),saved);
 const reloaded=boot(storage,'#workshop');assert.equal(reloaded.eval('JSON.stringify(state)'),saved,'an explicitly restored attempt continues until it is finished');
 assert.equal(reloaded.eval('busy'),false);assert.ok(!reloaded.document.querySelector('#level-complete').open);
 assert.deepEqual([...app.c.I18n.missing],[]);
});

test('legacy prologue migration preserves unfinished work and archives only completed answers',()=>{
 const fixture=boot(),M=fixture.c.NeuralPrologue;
 for(const id of Object.keys(M.levels).filter(id=>M.levels[id].practice))for(const completed of [false,true])for(const wired of [false,true]){
  const state=M.solutions(id);if(state.nodes.length)state.nodes[0].z+=.12;if(!wired)state.sink=null;
  const saved=JSON.stringify(state),storage=new Map([
   ['neural-atelier-chapters-v1',JSON.stringify({current:id,completed:completed?[id]:[]})],
   [`neural-atelier-level-${id}-v1`,saved]
  ]),app=boot(storage,'#'+id);
  assert.equal(app.eval('progress.completed(stageId)'),completed);assert.equal(app.eval('JSON.stringify(state)'),saved,'refresh resumes even a legacy finished circuit');
  app.eval(`switchLevel('rope');switchLevel('${id}')`);
  if(completed&&wired){
   assert.equal(app.eval('state.sink'),null);assert.ok(app.eval('state.nodes.every(n=>n.links.every(x=>x===null))'));
   assert.equal(storage.get(`neural-atelier-level-${id}-v1-completed`),saved);
  }else{
   assert.equal(app.eval('JSON.stringify(state)'),saved,id+' keeps an unfinished attempt');
   assert.equal(storage.has(`neural-atelier-level-${id}-v1-completed`),false);
  }
 }
 const wrong=M.solutions('activation');wrong.sink='project';assert.equal(M.evaluate(wrong,M.samples[0].v).error,null);
 const storage=new Map([
  ['neural-atelier-chapters-v1',JSON.stringify({current:'activation',completed:['activation']})],
  ['neural-atelier-level-activation-v1',JSON.stringify(wrong)]
 ]),app=boot(storage,'#activation');
 assert.equal(app.eval('JSON.stringify(state)'),JSON.stringify(wrong),'a fully connected but incorrect attempt is still unfinished');
 assert.equal(storage.has('neural-atelier-level-activation-v1-completed'),false);
});

test('migration retains the only saved solution when browser storage cannot archive it',()=>{
 const app=boot(),solution=app.c.NeuralPrologue.solutions('workshop'),saved=JSON.stringify(solution),storage=new Map([
  ['neural-atelier-chapters-v1',JSON.stringify({current:'workshop',completed:['workshop']})],
  ['neural-atelier-level-workshop-v1',saved]
 ]);
 app.c.limitedStorage={getItem:key=>storage.get(key)||null,setItem(key,value){if(key.endsWith('-completed'))throw new Error('Storage quota exceeded');storage.set(key,value);}};
 assert.equal(app.eval("JSON.stringify(createProgress(M,limitedStorage).load('workshop'))"),saved);
 assert.equal(storage.get('neural-atelier-level-workshop-v1'),saved,'do not overwrite a solution before it is archived');
});

test('finished prologues resume on refresh and history but deliberate replay restarts only once',async()=>{
 const app=boot(),M=app.c.NeuralPrologue;
 for(const id of Object.keys(M.levels).filter(id=>M.levels[id].practice)){
  app.eval(`switchLevel('${id}');state=M.solutions(stageId)`);await app.eval('run()');
  const solved=app.eval('JSON.stringify(state)'),reopened=boot(app.storage,'#'+id),meta=()=>JSON.parse(app.storage.get('neural-atelier-chapters-v1'));
  assert.equal(reopened.eval('JSON.stringify(state)'),solved);assert.equal(meta().practiceRuns[id].finished,true);
  reopened.eval("switchLevel('rope')");reopened.navigation.back();assert.equal(reopened.eval('JSON.stringify(state)'),solved);assert.equal(meta().practiceRuns[id].finished,true);
  reopened.navigation.forward();reopened.eval(`switchLevel('${id}')`);const fresh=reopened.eval('JSON.stringify(state)');
  assert.equal(reopened.eval('state.sink'),null);assert.ok(reopened.eval('state.nodes.every(n=>n.links.every(x=>x===null))'));
  assert.equal(app.storage.get(`neural-atelier-level-${id}-v1-completed`),solved);
  assert.equal(boot(app.storage,'#'+id).eval('JSON.stringify(state)'),fresh,'reload preserves the newly started attempt and its positions');
  reopened.c.previous=JSON.parse(solved);reopened.eval('state=previous;changed()');
  const pending=boot(app.storage,'#'+id);assert.equal(pending.eval('JSON.stringify(state)'),solved,'a rebuilt answer has not finished until the player runs it');
  assert.equal(pending.eval('results.length'),0);assert.equal(pending.eval('busy'),false);
 }
});

test('editing a completed prologue keeps the unfinished attempt across navigation and reload',async()=>{
 const app=boot(new Map(),'#workshop');app.eval('state=M.solutions(stageId)');await app.eval('run()');
 app.document.querySelector('#run').focus=()=>{};app.document.querySelector('#complete-replay').onclick();
 app.eval("chooseInput('sink',0);deleteWire()");
 const unfinished=app.eval('JSON.stringify(state)');assert.equal(app.eval('state.sink'),null);
 app.eval("switchLevel('projection');switchLevel('workshop')");assert.equal(app.eval('JSON.stringify(state)'),unfinished);
 const reloaded=boot(app.storage,'#workshop');assert.equal(reloaded.eval('JSON.stringify(state)'),unfinished);assert.equal(reloaded.eval('state.nodes.length'),1);
 assert.equal(reloaded.document.querySelector('#restore-layout').hidden,false);
 reloaded.document.querySelector('#restore-layout').onclick();assert.equal(reloaded.eval('state.sink'),'relu-ffn');assert.equal(reloaded.eval('busy'),false);
});

test('bridge cards explain only their actual inputs and calculations in both languages',()=>{
 const app=boot(),M=app.c.NeuralPrologue;
 for(const language of ['zh','en']){
  app.c.I18n.setPreference(language);
  for(const id of ['down-projection','plain-ffn','gate-projection','pairwise','swiglu']){
   app.eval(`switchLevel('${id}')`);const s=M.solutions(id);
   for(const n of s.nodes){
    const c=app.eval(`cardFor('${n.kind}','${id}')`),e=app.eval(`exampleFor(M,'${n.kind}','${id}')`),ev=M.evaluate(s,e.exampleSource[1]);
    assert.deepEqual(Array.from(e.output[1]),Array.from(ev.values[n.id]));
    e.inputs.forEach((input,i)=>assert.deepEqual(Array.from(input[1]),Array.from(ev.values[n.links[i]])));
    if(language==='en')assert.doesNotMatch(c.meaning+c.purpose,/[\u3400-\u9fff]/);
    if(id==='pairwise'||id==='gate-projection'){assert.equal(e.comparison,undefined);assert.equal(e.activation,undefined);}
    if(id==='plain-ffn'&&n.kind==='down')assert.equal(c.formula,'F = S × Wd');
    if(id==='down-projection'&&n.kind==='down')assert.equal(c.formula,'F = P × Wd');
   }
  }
 }
 assert.deepEqual([...app.c.I18n.missing],[]);
});
test('shelf sizing persists independently of circuits, language and the current level',()=>{
 const app=boot(new Map(),'#ffn',['en-US']),shelf=app.document.querySelector('#module-shelf');
 assert.equal(shelf.dataset.size,'small');assert.equal(app.document.querySelector('.shelf-open-label').textContent,'Hide shelf');const state=app.eval('JSON.stringify(state)');
 app.document.querySelector('#shelf-smaller').onclick();assert.equal(shelf.dataset.size,'small');assert.equal(app.document.querySelector('#shelf-smaller').disabled,true);
 app.c.I18n.setPreference('zh');assert.equal(shelf.dataset.size,'small');assert.equal(app.eval('JSON.stringify(state)'),state);
 app.eval("switchLevel('workshop')");assert.equal(shelf.dataset.size,'small');
 const reloaded=boot(app.storage,'#ffn');assert.equal(reloaded.document.querySelector('#module-shelf').dataset.size,'small');assert.equal(reloaded.eval('JSON.stringify(state)'),state);
 reloaded.document.querySelector('#shelf-larger').onclick();reloaded.document.querySelector('#shelf-larger').onclick();assert.equal(reloaded.document.querySelector('#module-shelf').dataset.size,'large');assert.equal(reloaded.document.querySelector('#shelf-larger').disabled,true);
});
test('completion reset confirms, preserves circuits and preferences, and survives reload across tracks',async()=>{
 const app=boot(),doc=app.document,storage=app.storage;
 for(const id of ['wire','workshop','encoder-layer','ffn','kv-cache'])app.eval(`switchLevel('${id}');state=M.solutions(stageId);state.nodes.forEach(n=>n.x+=.13);save();progress.complete(stageId,state)`);
 app.eval("switchLevel('encoder-layer');state=M.solutions(stageId);save()");await app.eval('run()');
 storage.set('neural-atelier-audio-v1',JSON.stringify({enabled:false,volume:.2}));
 const board=app.eval('JSON.stringify(state)'),before=new Map(storage);
 doc.querySelector('#progress-reset').onclick();assert.equal(doc.querySelector('#progress-reset-confirm').hidden,false);assert.equal(app.eval('progress.completed(stageId)'),true);
 doc.querySelector('#progress-reset-cancel').onclick();assert.equal(doc.querySelector('#progress-reset-confirm').hidden,true);assert.deepEqual(storage,before);
 app.eval('busy=true;refresh()');assert.equal(doc.querySelector('#progress-reset').disabled,true);doc.querySelector('#progress-reset-apply').onclick();assert.deepEqual(storage,before);app.eval('busy=false;refresh()');
 doc.querySelector('#progress-reset').onclick();doc.querySelector('#progress-reset-apply').onclick();
 assert.equal(app.eval('Object.keys(M.levels).some(id=>progress.completed(id))'),false);assert.equal(app.eval('levelPassed()'),false);assert.equal(doc.querySelector('#level-complete').open,false);assert.equal(doc.querySelector('#next-level').hidden,true);assert.equal(app.eval('JSON.stringify(state)'),board);
 for(const [key,value]of before)if(key!=='neural-atelier-chapters-v1')assert.equal(storage.get(key),value,key);
 assert.equal(doc.querySelector('#progress-reset-status').hidden,false);
 const reopened=boot(storage,'#encoder-layer');assert.equal(reopened.eval('JSON.stringify(state)'),board);
 for(const id of ['wire','workshop','encoder-layer','ffn','kv-cache']){reopened.eval(`switchLevel('${id}')`);assert.equal(reopened.eval('progress.completed(stageId)'),false);}
 // Practice lessons must not treat a retained winning circuit as finished and auto-clear it.
 assert.equal(storage.get('neural-atelier-level-workshop-v1'),before.get('neural-atelier-level-workshop-v1'));
 app.c.I18n.setPreference('en');for(const selector of ['.progress-setting','#progress-reset-confirm','#progress-reset-status'])assert.doesNotMatch(doc.querySelector(selector).textContent,/[\u3400-\u9fff]/);
});
test('completion reset leaves in-memory and saved badges intact when storage fails',()=>{
 const app=boot();app.eval("progress.complete('wire');progress.complete('workshop',M.solutions('workshop'))");
 const before=new Map(app.storage);app.c.localStorage.setItem=()=>{throw new Error('Storage unavailable');};
 app.document.querySelector('#progress-reset').onclick();app.document.querySelector('#progress-reset-apply').onclick();
 assert.equal(app.eval("progress.completed('wire')&&progress.completed('workshop')"),true);assert.deepEqual(app.storage,before);
 assert.match(app.document.querySelector('#progress-reset-status').textContent,/保存失败/);
});
test('starting over resets every board and completion while preserving settings and an undoable prior run',()=>{
 const app=boot(),doc=app.document,storage=app.storage,M=app.c.NeuralPrologue;
 for(const id of Object.keys(M.levels))app.eval(`switchLevel('${id}');state=M.solutions(stageId);save();progress.complete(stageId,state)`);
 app.eval("switchLevel('residual');state=M.solutions(stageId);state.nodes[0].x+=.4;save();setPlaybackSpeed(.5)");
 doc.querySelector('#progress-reset').onclick();doc.querySelector('#progress-reset-apply').onclick();
 assert.ok(app.eval('state.nodes.some(n=>n.links.some(Boolean))'),'badge-only clear deliberately retains wiring');
 storage.set('neural-atelier-audio-v1',JSON.stringify({enabled:false,volume:.2}));
 const board=app.eval('JSON.stringify(state)'),before=new Map(storage);
 doc.querySelector('#progress-restart').onclick();assert.match(doc.querySelector('#progress-reset-description').textContent,/连线会重置/);
 doc.querySelector('#progress-reset-cancel').onclick();assert.deepEqual(storage,before);
 app.eval('busy=true;refresh()');assert.equal(doc.querySelector('#progress-restart').disabled,true);doc.querySelector('#progress-reset-apply').onclick();assert.deepEqual(storage,before);app.eval('busy=false;refresh()');
 doc.querySelector('#progress-restart').onclick();doc.querySelector('#progress-reset-apply').onclick();
 assert.equal(app.eval('state.sink'),null);assert.equal(app.eval('state.nodes.length'),0,'residual modules must be placed from the shelf');assert.ok(app.eval('state.nodes.every(n=>n.links.every(x=>x===null))'));assert.equal(app.eval('history.length'),0);
 assert.equal(doc.querySelector('#restore-layout').hidden,true,'a reset board does not offer the old completed answer');
 assert.equal(doc.querySelector('#progress-restart-undo').hidden,false);assert.match(doc.querySelector('#progress-reset-status').textContent,/全部关卡已恢复/);
 for(const [key,value]of before)if(key!=='neural-atelier-chapters-v1')assert.equal(storage.get(key),value,'old save and preference retained: '+key);
 const reopened=boot(storage,'#residual');
 for(const id of Object.keys(M.levels)){
  reopened.eval(`switchLevel('${id}')`);assert.equal(reopened.eval('progress.completed(stageId)'),false,id);
  assert.equal(reopened.eval('JSON.stringify([state.sink,state.nodes.map(n=>[n.id,n.links]),state.settings])'),reopened.eval('(()=>{const s=M.initial(stageId);return JSON.stringify([s.sink,s.nodes.map(n=>[n.id,n.links]),s.settings])})()'),id+' starts with its authored components and connections');
 }
 reopened.eval("switchLevel('residual')");reopened.document.querySelector('#progress-restart-undo').onclick();assert.equal(reopened.eval('JSON.stringify(state)'),board);assert.equal(reopened.document.querySelector('#progress-restart-undo').hidden,true);
 assert.equal(reopened.eval('playbackSpeed'),.5);assert.equal(storage.get('neural-atelier-audio-v1'),before.get('neural-atelier-audio-v1'));
 app.c.I18n.setPreference('en');doc.querySelector('#progress-restart').onclick();for(const selector of ['.progress-setting','#progress-reset-confirm','#progress-restart-undo'])assert.doesNotMatch(doc.querySelector(selector).textContent,/[\u3400-\u9fff]/);
});
test('full restart undo restores badges and the live board, and subsequent resets never reuse discarded wiring',()=>{
 const app=boot(),doc=app.document;
 app.eval("switchLevel('residual');state=M.solutions(stageId);save();progress.complete(stageId,state)");const board=app.eval('JSON.stringify(state)');
 const restart=()=>{doc.querySelector('#progress-restart').onclick();doc.querySelector('#progress-reset-apply').onclick();};
 restart();app.eval('state.sink="R";save()');doc.querySelector('#progress-restart-undo').onclick();assert.equal(app.eval('JSON.stringify(state)'),board);assert.equal(app.eval('progress.completed(stageId)'),true);
 restart();assert.equal(app.eval('state.sink'),null);assert.equal(app.eval('progress.completed(stageId)'),false);
 const reloaded=boot(app.storage,'#residual');assert.equal(reloaded.eval('state.sink'),null);assert.equal(reloaded.document.querySelector('#progress-restart-undo').hidden,false);
});
test('failed full reset or undo does not switch the active saved run',()=>{
 const app=boot(),doc=app.document;app.eval("state.sink='R';save();progress.complete(stageId,state)");
 let before=new Map(app.storage),board=app.eval('JSON.stringify(state)'),write=app.c.localStorage.setItem;
 app.c.localStorage.setItem=()=>{throw new Error('Storage unavailable');};doc.querySelector('#progress-restart').onclick();doc.querySelector('#progress-reset-apply').onclick();
 assert.deepEqual(app.storage,before);assert.equal(app.eval('JSON.stringify(state)'),board);assert.equal(app.eval('progress.completed(stageId)'),true);assert.match(doc.querySelector('#progress-reset-status').textContent,/保存失败/);
 app.c.localStorage.setItem=write;doc.querySelector('#progress-reset-apply').onclick();before=new Map(app.storage);board=app.eval('JSON.stringify(state)');
 app.c.localStorage.setItem=()=>{throw new Error('Storage unavailable');};doc.querySelector('#progress-restart-undo').onclick();assert.deepEqual(app.storage,before);assert.equal(app.eval('JSON.stringify(state)'),board);assert.equal(app.eval('progress.completed(stageId)'),false);assert.match(doc.querySelector('#progress-reset-status').textContent,/保存失败/);
});
test('every learning shelf groups packages without losing old parts or saved circuits',()=>{
 const app=boot(),M=app.c.NeuralPrologue,doc=app.document;
 const ids=[...M.routes.main.ids,...M.routes.modern.ids];
 for(const id of ids.filter(id=>M.levels[id].buildable&&!M.levels[id].demonstration)){
  app.eval(`switchLevel('${id}');state=M.solutions(stageId);save();refresh()`);const before=app.eval('JSON.stringify(state)');
  const expected=Object.keys(M.inventory(M.levels[id])).filter(kind=>M.fusions[id]?.kind!==kind||M.levels[id].nodes.includes(kind));
  const actual=[...doc.querySelectorAll('[data-add-module]')].map(el=>el.dataset.addModule);
  assert.deepEqual([...actual].sort(),[...expected].sort(),id+' retains exactly the unlocked inventory');assert.equal(new Set(actual).size,actual.length,id+' renders each kind once');
  const earlier=doc.querySelector('[data-earlier-parts]');if(earlier)assert.equal(earlier.hasAttribute('open'),false);
  assert.ok(doc.querySelector('[data-shelf-group]'),id+' has explicit grouping');
  for(const el of doc.querySelectorAll('[data-shelf-group="packages"] [data-add-module]'))assert.ok(M.packages[el.dataset.addModule]);
  for(const el of doc.querySelectorAll('[data-shelf-group="basics"] [data-add-module]'))assert.ok(!M.packages[el.dataset.addModule]);
  assert.equal(actual.length,doc.querySelectorAll('[data-shelf-group="packages"] [data-add-module], [data-shelf-group="basics"] [data-add-module]').length,id+' classifies every module, including earlier parts');
  for(const kind of M.levels[id].nodes)assert.ok(doc.querySelector(`.shelf-current [data-add-module="${kind}"]`),id+' keeps taught modules visible');
  if(earlier){earlier.setAttribute('open','');app.eval('refresh()');assert.equal(doc.querySelector('[data-earlier-parts]').hasAttribute('open'),true);}
  app.c.I18n.setPreference('en');assert.doesNotMatch(doc.querySelector('#module-palette').textContent,/[\u3400-\u9fff]/);assert.equal(app.eval('JSON.stringify(state)'),before);app.c.I18n.setPreference('zh');
 }
 for(const id of ['pack-scores','weighted-sum','self-attention','multihead']){
  app.eval(`switchLevel('${id}')`);assert.ok(doc.querySelector('[data-shelf-group="packages"] [data-add-module="original-weights"]'),id+' groups the learned score package');
  assert.ok(doc.querySelector('[data-shelf-group="basics"] [data-add-module="v"]'),id+' groups V as a basic module');
 }
 app.eval("switchLevel('self-attention');state=M.solutions(stageId);save();progress.complete(stageId,state);refresh()");assert.ok(doc.querySelector('[data-shelf-group="packages"] [data-add-module="single-attention"]'),'the newly crafted single head joins the package group');
 app.eval("switchLevel('causal')");assert.ok(doc.querySelector('[data-shelf-group="packages"] [data-add-module="single-attention"]'));assert.ok(doc.querySelector('[data-shelf-group="basics"] [data-add-module="v"]'),'V remains a compatible distractor');
 assert.ok(doc.querySelector('.shelf-earlier [data-add-module="original-encoder"]'),'four-feature encoder is still available among earlier parts');
 app.eval("switchLevel('decoder-layer');state=M.solutions(stageId);save();progress.complete(stageId,state);refresh()");assert.ok(doc.querySelector('[data-shelf-group="packages"] [data-add-module="original-decoder"]'),'newly earned reward appears in its package group');
 for(const id of M.routes.engine.ids){app.eval(`switchLevel('${id}')`);assert.equal(doc.querySelector('[data-earlier-parts]'),null);assert.equal(doc.querySelectorAll('[data-add-module]').length,6);}
});
test('cloud track sits before edge inference and preserves independent saves, language and completion',async()=>{
 const app=boot(new Map(),'#tp-columns'),doc=app.document;
 app.eval('levelMenu()');assert.deepEqual([...doc.querySelectorAll('[data-track]')].map(el=>el.dataset.track),['main','modern','cloud','engine']);
 assert.equal(doc.querySelectorAll('#level-list [data-level]').length,6);assert.deepEqual([...doc.querySelectorAll('[data-level-group]')].map(el=>el.dataset.levelGroup),['single','tensor-parallel']);assert.match(doc.querySelector('.level-menu-intro').textContent,/CPU 回退与显存驻留/);assert.match(doc.querySelector('.level-menu-intro').textContent,/两张 GPU/);assert.doesNotMatch(doc.querySelector('.level-menu-intro').textContent,/CUDA Graph|尚未开放/);
 app.eval('state=M.solutions(stageId);save()');await app.eval('run()');assert.equal(app.eval('levelPassed()'),true);const before=app.eval('JSON.stringify(state)');
 const reopened=boot(app.storage,'#tp-columns');assert.equal(reopened.eval('JSON.stringify(state)'),before);assert.equal(reopened.eval('progress.completed(stageId)'),true);
 app.c.I18n.setPreference('en');app.eval('levelMenu()');assert.doesNotMatch(doc.querySelector('#level-menu').textContent,/[\u3400-\u9fff]/);
 doc.querySelector('#complete-next').onclick();assert.equal(app.eval('stageId'),'tp-rows');assert.equal(app.eval('state.nodes.length'),0);
 app.eval("switchLevel('tp-ffn');state=M.solutions(stageId)");await app.eval('run()');assert.equal(doc.querySelector('#complete-next').textContent,'Choose a level');
 app.eval('progress.resetCompletions()');assert.equal(app.eval("progress.completed('tp-columns')"),false);
});
test('single GPU report explains transfers and rejects correct numbers with excessive copies',async()=>{
 const app=boot(new Map(),'#single-upload'),doc=app.document;
 app.eval('state=M.solutions(stageId)');await app.eval('run()');assert.match(doc.querySelector('.cloud-copy-report').textContent,/24 B/);
 doc.querySelector('#complete-next').onclick();assert.equal(app.eval('stageId'),'single-fallback');app.eval('state=M.solutions(stageId)');await app.eval('run()');assert.match(doc.querySelector('.cloud-copy-report').textContent,/48 B/);
 doc.querySelector('#complete-next').onclick();assert.equal(app.eval('stageId'),'single-resident');app.eval("state=M.solutions('single-fallback');state.levelId='single-resident'");await app.eval('run()');assert.equal(app.eval('levelPassed()'),false);assert.match(doc.querySelector('.cloud-copy-report').textContent,/48 \/ 16 B/);
 app.c.I18n.setPreference('en');assert.doesNotMatch(doc.querySelector('.cloud-copy-report').textContent,/[\u3400-\u9fff]/);
 app.eval('state=M.solutions(stageId)');await app.eval('run()');assert.equal(app.eval('levelPassed()'),true);assert.match(doc.querySelector('.cloud-copy-report').textContent,/16 B/);
 doc.querySelector('#complete-next').onclick();assert.equal(app.eval('stageId'),'tp-columns');
});

test('stage colors follow direct links, navigation and language while menu browsing keeps the active scene theme',()=>{
 const app=boot(new Map(),'#scores'),doc=app.document;
 assert.equal(doc.body.dataset.stageColor,'attention');
 for(const [id,color] of [['wire','workbench'],['embedding','foundations'],['encoder-layer','encoder'],['causal','decoder'],['vocabulary','generation'],['ffn','modern'],['placement','engine'],['single-upload','single'],['tp-columns','tensor-parallel']]){
  app.eval(`switchLevel('${id}')`);assert.equal(doc.body.dataset.stageColor,color,id);
 }
 app.navigation.back();assert.equal(doc.body.dataset.stageColor,'single');
 app.navigation.forward();assert.equal(doc.body.dataset.stageColor,'tensor-parallel');
 const state=app.eval('JSON.stringify(state)');app.eval('levelMenu()');
 for(const route of ['modern','engine','main']){
  doc.querySelector(`[data-track="${route}"]`).dispatchEvent(new app.window.Event('click',{bubbles:true}));
  assert.equal(doc.querySelector('#level-menu').dataset.stageColor,route);
  assert.equal(doc.body.dataset.stageColor,'tensor-parallel');
 }
 app.c.I18n.setPreference('en');assert.equal(doc.body.dataset.stageColor,'tensor-parallel');assert.equal(app.eval('JSON.stringify(state)'),state);
});

test('causal guidance explains visibility and score provenance; the screenshot bypass is not a correct Q/K result',async()=>{
 const app=boot(new Map(),'#causal'),doc=app.document,M=app.c.NeuralPrologue;
 assert.equal(doc.querySelectorAll('.causal-primer td.allowed').length,6);assert.equal(doc.querySelectorAll('.causal-primer td.blocked').length,3);
 assert.equal(doc.querySelector('.causal-primer details').hasAttribute('open'),false);
 assert.deepEqual([...doc.querySelectorAll('.causal-step code')].map(el=>el.textContent),['[-0.07, 0.78, 0.71]','[-0.07, 0.78, −∞]','[0.3, 0.7, 0]']);
 app.eval("state=M.solutions('causal');state.nodes.find(n=>n.id==='score').links=['R','R']");
 const wrong=M.evaluate(app.eval('state'),M.samplesFor('causal')[0].v).output;
 assert.equal(wrong[1],0);assert.equal(wrong[2],0);assert.equal(wrong[5],0);assert.ok(Math.abs(wrong[3]-.33)<.01);
 await app.eval('run()');assert.equal(app.eval('levelPassed()'),false);
 app.eval("state=M.solutions('causal')");await app.eval('run()');assert.equal(app.eval('levelPassed()'),true);
 app.c.I18n.setPreference('en');assert.doesNotMatch(doc.querySelector('.mission').textContent,/[\u3400-\u9fff]/);
 const card=app.eval("cardFor('mask','causal')");assert.match(card.purpose,/\[2, 1, 3\]/);assert.doesNotMatch(card.meaning+card.purpose,/[\u3400-\u9fff]/);
 app.eval("switchLevel('scores')");assert.equal(doc.querySelector('.causal-primer'),null);
});

test('every playable lesson has a bilingual structure location, valid focus and a distinct outlet contract',()=>{
 const app=boot(),M=app.c.NeuralPrologue,map=M.structureMap;
 const ids=Object.values(M.routes).flatMap(r=>Array.from(r.ids));assert.equal(ids.length,58);assert.deepEqual(new Set(Object.keys(map.lessons)),new Set(ids));
 for(const lang of ['zh','en']){
  app.c.I18n.setPreference(lang);
  for(const id of ids){
   const entry=map.get(id),keys=entry.steps.map(n=>n.id);
   assert.ok(keys.includes(entry.exit),id+' outlet belongs to its reference structure');for(const focus of entry.focus)assert.ok(keys.includes(focus),id+' focus');
   app.eval(`switchLevel('${id}')`);const saved=app.eval('JSON.stringify(state)'),progress=app.storage.get('neural-atelier-chapters-v1');
   app.document.querySelector('#hint').onclick();
   assert.ok(app.document.querySelector('#structure-dialog').open);
   assert.equal(app.document.querySelectorAll('.map-step-exit').length,1,id+' has one lesson outlet');
   assert.ok(app.document.querySelector('#structure-dialog-content').textContent.includes(entry.output));
   assert.ok(app.document.querySelectorAll('#structure-dialog-content .architecture-drilldown').length>0,id+' has expandable internals');
   if(lang==='en')assert.doesNotMatch(app.document.querySelector('#structure-dialog-content').textContent,/[\u3400-\u9fff]/,id);
   app.document.querySelector('#structure-close').onclick();assert.equal(app.document.querySelector('#structure-dialog').open,false);
   assert.equal(app.eval('JSON.stringify(state)'),saved);assert.equal(app.storage.get('neural-atelier-chapters-v1'),progress);
  }
 }
 app.c.I18n.setPreference('en');
 assert.equal(map.get('causal').flow,'causal');assert.equal(map.get('causal').exit,'softmax');assert.match(map.get('causal').output,/weights A/);
 assert.equal(map.get('qkv').exit,'qkv');assert.match(map.get('qkv').output,/V = X × Wᵥ/);
 assert.equal(map.get('cross-attention').exit,'wo');assert.match(map.get('cross-attention').output,/before residual/);
 app.eval("switchLevel('cross-attention')");
 app.document.querySelector('#hint').onclick();
 const branches=app.document.querySelectorAll('[data-map-flow="cross"] .map-branch-pair>ol');
 assert.equal(branches.length,2);assert.match(branches[0].textContent,/LayerNorm → D/);assert.match(branches[0].textContent,/D → Q projection → Q/);assert.match(branches[1].textContent,/E → K \/ V projections/);
 assert.equal(map.get('cross-attention').focus.includes('prepared'),false,'the preceding lesson is context, not required again on this board');
 const architecture=app.document.querySelector('.architecture-detail');
 for(const lane of ['encoder','decoder']){
  const region=architecture.querySelector(`[data-architecture-lane="${lane}"]`);
  assert.match(region.textContent,/Attention/);assert.match(region.textContent,/FFN/);
  assert.ok(region.querySelector('.architecture-layer[open]'),'layer boundary starts expanded');
 }
 assert.deepEqual([...architecture.querySelectorAll('.architecture-stage.architecture-focus')].map(e=>e.dataset.architectureNode),['decoder-cross']);
 assert.match(architecture.querySelector('[data-text-input="encoder"]').textContent,/user.*text to translate/);
 assert.match(architecture.querySelector('[data-text-input="decoder"]').textContent,/generated so far/);
 assert.match(architecture.querySelector('[data-model-weights]').textContent,/weights.*inside.*modules/);
 assert.match(architecture.querySelector('[data-architecture-edge="E-to-cross"]').textContent,/E supplies K \/ V; D supplies Q/);
 assert.match(architecture.querySelector('[data-architecture-node="decoder-cross"] .architecture-drilldown').textContent,/Q = DWq; K = EWk; V = EWv/);
 assert.match(architecture.querySelector('.architecture-boundary').textContent,/ends after Wo/);
 app.document.querySelector('#structure-close').onclick();
 assert.equal(map.get('decoder-self').exit,'norm1');assert.equal(map.get('encoder-layer').exit,'norm2');assert.equal(map.get('decoder-layer').exit,'norm3');
 assert.match(map.get('ffn').where,/half/);assert.match(map.get('ffn').next,/attention half is absent/);
 assert.match(map.get('encoder-decoder').output,/Candidate-token scores L/);assert.match(map.get('autoregressive').note,/fixed transition table/);
 assert.match(map.get('placement').output,/features/);assert.match(map.get('placement').note,/not a real model/);
 assert.deepEqual(Array.from(map.flows.decoder,n=>n.id),['attention','add1','norm1','cross','add2','norm2','ffn','add3','norm3']);
 assert.deepEqual(Array.from(map.flows.modernLayer,n=>n.id),['norm1','attention','add1','norm2','ffn','add2']);
});

test('wiring map identifies a general operation instead of locating it inside encoder and decoder',()=>{
 const app=boot(),M=app.c.NeuralPrologue;
 for(const lang of ['zh','en']){
  app.c.I18n.setPreference(lang);app.eval("switchLevel('wire')");
  const saved=app.eval('JSON.stringify(state)'),progress=app.storage.get('neural-atelier-chapters-v1');
  app.document.querySelector('#hint').onclick();
  const full=app.document.querySelector('#structure-dialog-content');
  assert.equal(app.document.querySelector('#structure-map'),null);assert.ok(full.querySelector('[data-map-region=wire].map-region-active'));
  assert.equal(full.querySelectorAll('.architecture-focus').length,0,'generic wiring has no dedicated architecture layer');
  assert.ok(full.querySelector('.wiring-reference .architecture-detail'));
  assert.equal(full.querySelector('.wiring-reference').hasAttribute('open'),false,'complete model is optional context');
  assert.match(full.textContent,/\[1, 0\].*\[1, 0\]/);
  assert.doesNotMatch(full.textContent,/着色部分是本关所在位置|Color marks this lesson’s location/);
  assert.equal(full.querySelector('.map-step-focus').dataset.mapStep,'wire');
  assert.equal(full.querySelector('.map-step-exit').dataset.mapStep,'result');
  assert.equal(app.eval('JSON.stringify(state)'),saved);assert.equal(app.storage.get('neural-atelier-chapters-v1'),progress);
  app.document.querySelector('#structure-close').onclick();
  for(const id of ['encoder-decoder','transformer-demo','autoregressive']){
   app.eval(`switchLevel('${id}')`);app.document.querySelector('#hint').onclick();
   const architecture=app.document.querySelector('.architecture-detail');
   for(const key of ['encoder-input','decoder-input','readout'])assert.ok(architecture.querySelector(`[data-architecture-node="${key}"].architecture-focus`),id+' includes '+key);
   app.document.querySelector('#structure-close').onclick();
  }
 }
});

test('generation map names the actual modules, distinguishes predictions from context and preserves the circuit',()=>{
 const app=boot(new Map(),'#autoregressive'),M=app.c.NeuralPrologue;
 app.eval("state=M.solutions('autoregressive')");const saved=app.eval('JSON.stringify(state)');
 for(const lang of ['zh','en']){
  app.c.I18n.setPreference(lang);
  app.document.querySelector('#hint').onclick();
  const map=app.document.querySelector('#structure-dialog-content');
  for(const kind of ['step','pick','feedback'])assert.equal(map.querySelector(`[data-generation-module="${kind}"] strong`).textContent,M.definitions[kind].name);
  assert.equal(map.querySelector('.generation-reference').hasAttribute('open'),false,'full network is secondary to the generation loop');
  assert.equal(map.querySelectorAll('.generation-predictions .generation-chosen').length,1);
  assert.equal(map.querySelector('.generation-predictions .generation-chosen').dataset.predictionIndex,'2');
  const rendered=[...map.querySelectorAll('[data-generation-window]>div:last-child [data-token-id]')].map(el=>Number(el.dataset.tokenId));
  assert.deepEqual(rendered,Array.from(M.evaluate(M.solutions('autoregressive'),[0,1,2]).output));
  assert.equal(app.eval('JSON.stringify(state)'),saved);
  if(lang==='en')assert.doesNotMatch(map.textContent,/[\u3400-\u9fff]/);
  app.document.querySelector('#structure-close').onclick();
 }
 assert.match(M.extraCard('feedback','autoregressive').meaning,/remembers.*original context/i);
 assert.match(M.extraCard('step','autoregressive').purpose,/replaces that scoring stage/);
 assert.deepEqual(Array.from(M.definitions.feedback.glyphs),['…','→','+1']);
});

test('expanded modern and execution maps distinguish architecture, device count and cached data',()=>{
 const app=boot(new Map(),'#block',['en-US']),map=app.c.NeuralPrologue.structureMap;
 const tree=id=>parseHTML('<main>'+map.render(id,true)+'</main>').document;
 const modern=tree('block');
 assert.equal(modern.querySelector('[data-architecture-node="decoder-cross"]'),null);
 assert.match(modern.querySelector('[data-architecture-node="modern-attention"]').textContent,/RoPE/);
 assert.match(modern.querySelector('[data-architecture-node="modern-ffn"]').textContent,/SwiGLU.*parallel branches/s);
 assert.deepEqual([...modern.querySelectorAll('.modern-layer>.architecture-stage')].map(e=>e.dataset.architectureNode),['modern-norm1','modern-attention','modern-add1','modern-norm2','modern-ffn','modern-add2']);
 assert.match(tree('plain-ffn').querySelector('[data-architecture-node="modern-ffn"]').textContent,/SiLU FFN/);
 assert.doesNotMatch(tree('pairwise').querySelector('[data-architecture-node="modern-ffn"] .architecture-drilldown').textContent,/SiLU\(G\)/);
 for(const id of ['single-upload','single-fallback','single-resident']){
  assert.equal(tree(id).querySelector('[data-map-region="gpu2"]'),null,id+' has only one GPU');
  assert.ok(tree(id).querySelector('[data-architecture-node="single-gpu"] .architecture-drilldown'));
 }
 assert.match(tree('single-fallback').querySelector('.architecture-detail').textContent,/back to CPU → ReLU → upload/);
 assert.match(tree('single-resident').querySelector('.architecture-detail').textContent,/no intermediate CPU round trip/);
 for(const id of ['tp-columns','tp-rows','tp-ffn'])assert.equal(tree(id).querySelectorAll('.execution-parallel>.architecture-stage').length,2);
 assert.match(tree('tp-columns').querySelector('[data-architecture-node="merge"]').textContent,/Gather.*concat/);
 assert.match(tree('tp-rows').querySelector('[data-architecture-node="merge"]').textContent,/Reduce.*P₀ \+ P₁/);
 const engine=tree('kv-cache');assert.ok(engine.querySelector('[data-architecture-node="engine-decode"].architecture-focus'));
 assert.match(engine.querySelector('[data-architecture-node="engine-memory"]').textContent,/Model weights ≠ KV cache/);
 assert.match(engine.querySelector('[data-architecture-node="engine-decode"]').textContent,/same layers and weights/);
});

test('hint reference updates on hashes and language and stays inspectable during a run',async()=>{
 const app=boot(new Map(),'#causal'),doc=app.document;
 const open=()=>doc.querySelector('#hint').onclick();
 open();app.c.I18n.setPreference('en');assert.match(doc.querySelector('#structure-dialog-content').textContent,/masked self-attention/);
 app.eval("switchLevel('original-ffn')");assert.match(doc.querySelector('#structure-dialog-content').textContent,/ReLU FFN output/);
 doc.querySelector('#structure-close').onclick();
 app.eval("switchLevel('causal')");assert.equal(doc.querySelector('.structure-compact'),null);open();
 app.location.hash='#encoder-layer';app.hashchange();assert.match(doc.querySelector('#structure-dialog-content').textContent,/Encoder layer output/);
 doc.querySelector('#structure-close').onclick();app.eval("switchLevel('wire');state=M.solutions('wire')");
 let resolve;app.c.pauseMapTest=()=>new Promise(r=>{resolve=r;});app.eval('world.animate=pauseMapTest');const running=app.eval('run()');
 const saved=app.eval('JSON.stringify(state)');open();assert.equal(app.eval('busy'),true);assert.equal(app.eval('JSON.stringify(state)'),saved);
 doc.querySelector('#structure-close').onclick();app.eval('world.animate=async()=>{}');resolve();await running;assert.equal(app.eval('levelPassed()'),true);
 // Native dialog Escape is left to the browser. Backdrop and close button use the same non-mutating close.
 open();doc.querySelector('#structure-dialog').getBoundingClientRect=()=>({left:100,top:100,right:600,bottom:600});
 const click=new app.window.Event('click',{bubbles:true});Object.assign(click,{clientX:20,clientY:20});doc.querySelector('#structure-dialog').dispatchEvent(click);assert.equal(doc.querySelector('#structure-dialog').open,false);
});

test('structure map short labels never overwrite shared animation translations',()=>{
 const app=boot(new Map(),'#cross-attention',['en-US']),M=app.c.NeuralPrologue;
 assert.equal(app.c.I18n.t('目标前缀'),'Target prefix');assert.equal(app.c.I18n.t('源序列'),'Source sequence');
 const catalog=JSON.stringify(app.c.AtelierEnglish);
 for(const language of ['zh','en']){
  app.c.I18n.setPreference(language);
  for(const id of Object.keys(M.structureMap.lessons)){M.structureMap.render(id);M.structureMap.render(id,true);}
 }
 assert.equal(JSON.stringify(app.c.AtelierEnglish),catalog);
 assert.equal(app.c.I18n.t('目标前缀'),'Target prefix');
});

test('causal offers zero new packaged-score copies while preserving existing saved circuits',()=>{
 const app=boot(new Map(),'#causal'),M=app.c.NeuralPrologue;
 assert.equal(M.inventory(M.levels.causal)['original-weights'],0);
 assert.equal(app.document.querySelector('[data-add-module="original-weights"] .stock-count').textContent,'0');
 assert.equal(app.document.querySelector('[data-add-module="original-weights"]').dataset.addDisabled,'true');
 assert.equal(M.canAdd(app.eval('state'),'original-weights'),false);
 assert.equal(M.inventory(M.levels['weighted-sum'])['original-weights'],2);
 const saved=M.solutions('causal');saved.nodes.push({id:'original-weights',x:-4,z:3,links:['R']});
 const storage=new Map([['neural-atelier-level-causal-v1',JSON.stringify(saved)]]),restored=boot(storage,'#causal');
 assert.equal(restored.eval('state.nodes.length'),saved.nodes.length);assert.equal(restored.eval("state.nodes.find(n=>n.id==='original-weights').links[0]"),'R');
 restored.eval("removeModule('original-weights')");assert.equal(restored.eval("M.stock(state,'original-weights')"),0);assert.equal(restored.eval("M.canAdd(state,'original-weights')"),false);
});

test('masked bridge keeps old decoder progress and saves repeated module instances without unlocking its reward early',async()=>{
 const app=boot(),M=app.c.NeuralPrologue;
 app.eval("switchLevel('decoder-self');state=M.solutions('decoder-self');refresh();save()");const old=app.eval('JSON.stringify(state)');
 app.eval("switchLevel('masked-multihead')");assert.equal(app.location.hash,'#masked-multihead');
 assert.equal(app.document.querySelector('[data-add-module="masked-heads"]'),null);
 app.eval("state=M.solutions('masked-multihead');refresh();save()");const bridge=app.eval('JSON.stringify(state)');
 const reload=boot(app.storage,'#masked-multihead');assert.equal(reload.eval('JSON.stringify(state)'),bridge);
 reload.eval("switchLevel('decoder-self')");assert.equal(reload.eval('JSON.stringify(state)'),old);
 reload.eval("switchLevel('masked-multihead')");await reload.eval('run()');
 assert.ok(reload.document.querySelector('#level-complete').open);assert.ok(reload.document.querySelector('[data-add-module="masked-heads"]'));
 assert.equal(reload.eval('stage().next'),'decoder-self');assert.equal(reload.eval('JSON.stringify(state)'),bridge);
});

test('the hint button opens the full optional reference and closes without changing the circuit',async()=>{
 const app=boot(new Map(),'#layernorm'),doc=app.document,button=doc.querySelector('#hint'),dialog=doc.querySelector('#structure-dialog');
 assert.equal(button.closest('.topbar'),doc.querySelector('.topbar'));assert.equal(button.getAttribute('aria-haspopup'),'dialog');assert.equal(button.getAttribute('aria-controls'),dialog.id);
 assert.equal(doc.querySelector('#structure-map'),null);assert.equal(doc.querySelector('#hint-panel'),null);
 assert.match(doc.querySelector('.mission>p').textContent,/不承接前一关/);assert.match(doc.querySelector('.route').textContent,/练习向量/);
 const board=app.eval('JSON.stringify(state)');button.onclick();
 assert.equal(dialog.open,true);assert.equal(doc.querySelector('#structure-title').textContent,'本关提示');assert.ok(doc.querySelector('.architecture-detail'));assert.equal(doc.querySelector('#feedback').textContent,'');assert.equal(app.eval('JSON.stringify(state)'),board);
 const cancel=new app.window.Event('cancel',{cancelable:true});dialog.dispatchEvent(cancel);assert.equal(dialog.open,false);
 button.onclick();doc.querySelector('#structure-close').onclick();assert.equal(dialog.open,false);
 button.onclick();app.eval('levelMenu()');assert.equal(dialog.open,false);
 app.eval("switchLevel('transformer-demo')");assert.equal(button.hidden,false);assert.equal(button.disabled,false);button.onclick();assert.ok(dialog.open);
 app.c.I18n.setPreference('en');assert.equal(doc.querySelector('#structure-title').textContent,'Level hint');assert.doesNotMatch(doc.querySelector('#structure-dialog-content').textContent,/[\u3400-\u9fff]/);
 doc.querySelector('#structure-close').onclick();app.eval("switchLevel('wire');state=M.solutions('wire')");button.onclick();await app.eval('run()');assert.equal(dialog.open,false);assert.equal(app.eval('levelPassed()'),true);
 app.eval('renderLevelMenu()');assert.match(doc.querySelector('#level-track-main').textContent,/30 levels/);
 app.c.I18n.setPreference('zh');app.eval('renderLevelMenu()');assert.match(doc.querySelector('#level-track-main').textContent,/共 30 关/);
});

test('masked head explanation distinguishes whole heads from Q and K using real independent reading weights',()=>{
 const app=boot(new Map(),'#masked-multihead'),doc=app.document,M=app.c.NeuralPrologue,board=app.eval('JSON.stringify(state)');
 const values=M.evaluate(M.solutions('masked-multihead'),M.samplesFor('masked-multihead')[0].v).values;
 for(const language of ['zh','en']){
  app.c.I18n.setPreference(language);const primer=doc.querySelector('.heads-primer');assert.ok(primer);assert.equal(primer.querySelectorAll('.heads-branch').length,2);
  for(const [i,row]of [...primer.querySelectorAll('tbody tr')].entries()){
   assert.deepEqual([...row.querySelectorAll('td')].map(e=>e.textContent),Array.from(values[i?'softmax-2':'softmax'].slice(3,6),v=>Math.round(v*100)+'%'));
   assert.equal(row.lastElementChild.textContent,'0%');
  }
  assert.notDeepEqual(Array.from(values.softmax.slice(3,6)),Array.from(values['softmax-2'].slice(3,6)));
  assert.equal(primer.querySelectorAll('details[open]').length,0);assert.equal(app.eval('JSON.stringify(state)'),board);
  if(language==='en')assert.doesNotMatch(doc.querySelector('.mission').textContent,/[\u3400-\u9fff]/);
 }
 app.eval("switchLevel('causal')");assert.equal(doc.querySelector('.heads-primer'),null);assert.deepEqual([...app.c.I18n.missing],[]);
});

test('combined shelf help shows its original chapter with stock guidance and clears on interaction or navigation',()=>{
 const app=boot(new Map(),'#causal'),doc=app.document,M=app.c.NeuralPrologue,tip=doc.querySelector('#shelf-tooltip'),saved=app.eval('JSON.stringify(state)');
 for(const kind of Object.keys(M.packages))assert.ok(M.levels[M.packageOrigin(kind).id],kind+' has a known source lesson');
 for(const [kind,id]of Object.entries({'single-cross-attention':'cross-attention','self-heads':'multihead','relu-ffn':'original-ffn','original-ffn-block':'original-ffn','original-weights':'softmax','original-encoder':'encoder-layer','masked-heads':'masked-multihead'}))assert.equal(M.packageOrigin(kind).id,id);
 assert.equal(M.packageOrigin('q'),null);
 const hover=button=>button.dispatchEvent(new app.window.Event('pointerover',{bubbles:true}));
 let button=doc.querySelector('[data-add-module="original-weights"]');assert.ok(button);assert.equal(button.hasAttribute('title'),false);
 hover(button);assert.equal(tip.hidden,false);assert.equal(tip.textContent,'来源：原版基础 / 07\n可用数量 0 / 0 · 双击查看说明');assert.equal(button.getAttribute('aria-describedby'),'shelf-tooltip');assert.doesNotMatch(tip.textContent,/3 ×|Softmax/);
 assert.ok(!tip.textContent.includes(M.levels.softmax.title));assert.match(button.getAttribute('aria-label'),/来源：原版基础 \/ 07/);
 assert.match(button.getAttribute('aria-label'),/Softmax/);assert.equal(app.eval('JSON.stringify(state)'),saved);
 button.dispatchEvent(new app.window.Event('pointerout',{bubbles:true}));assert.equal(tip.hidden,true);
 button=doc.querySelector('[data-add-module="q"]');hover(button);assert.equal(tip.hidden,false);
 doc.dispatchEvent(new app.window.Event('pointerdown',{bubbles:true}));assert.equal(tip.hidden,true);
 hover(button);doc.querySelector('#module-palette').dispatchEvent(new app.window.Event('scroll',{bubbles:true}));assert.equal(tip.hidden,true);
 hover(button);app.c.I18n.setPreference('en');assert.equal(tip.hidden,true);
 button=doc.querySelector('[data-add-module="original-weights"]');hover(button);assert.equal(tip.textContent,'From: Original / 07\nCopies available 0 / 0 · Double-click to inspect');
 button=doc.querySelector('[data-add-module="q"]');hover(button);assert.match(tip.textContent,/2 \/ 2/);assert.doesNotMatch(tip.textContent,/[\u3400-\u9fff]|3 ×|From:/);
 app.eval("switchLevel('masked-multihead')");assert.equal(tip.hidden,true);assert.equal(doc.querySelector('[aria-describedby="shelf-tooltip"]'),null);
});

test('attention package diagrams unpack each head with separate QK and V paths in both languages',()=>{
 const app=boot(new Map(),'#encoder-layer'),M=app.c.NeuralPrologue,before=app.eval('JSON.stringify(state)');
 for(const language of ['zh','en']){
  app.c.I18n.setPreference(language);
  for(const [kind,headCount,width]of [['self-heads',2,2],['two-head-attention',2,1],['masked-heads',2,2],['cross-heads',2,2],['head-a',1,1],['head-b',1,1]]){
   const {document:doc}=parseHTML(app.eval(`packageDiagram('${kind}')`)),diagram=doc.querySelector('[data-package]'),heads=[...doc.querySelectorAll('.attention-head')];
   assert.equal(heads.length,headCount,kind);assert.equal(doc.querySelectorAll('details').length,0,'the internal steps are visible immediately');
   for(const head of heads){
    const suffix=head.dataset.head==='1'?'₁':'₂';
    assert.deepEqual([...head.querySelectorAll('[data-projection]')].map(el=>el.dataset.projection),['Q','K','V']);
    assert.deepEqual([...head.querySelectorAll('.head-matching [data-head-step]')].map(el=>el.dataset.headStep),kind==='masked-heads'?['score','mask','softmax']:['score','softmax']);
    const score=head.querySelector('[data-head-step="score"] code').textContent;assert.ok(score.includes('√'+width));assert.ok(score.includes('Q'+suffix+'K'+suffix+'ᵀ'));assert.ok(!score.includes('V'));
    assert.equal(head.querySelector('.head-value-path').dataset.valueRoute,'V'+suffix,'V bypasses matching and Softmax');
    assert.equal(head.querySelector('[data-head-step="mix"] code').textContent,'H'+suffix+' = A'+suffix+' × V'+suffix);
   }
   assert.equal(doc.querySelectorAll('[data-head-step="concat"]').length,headCount===2?1:0);
   assert.equal(doc.querySelectorAll('[data-head-step="output"]').length,headCount===2?1:0,'Wo follows concatenation, never inside a single head');
   if(kind==='cross-heads'){
    assert.match(heads[0].querySelector('[data-projection="Q"]').textContent,language==='zh'?/目标侧/:/Target/);
    for(const symbol of ['K','V'])assert.match(heads[0].querySelector('[data-projection="'+symbol+'"]').textContent,language==='zh'?/编码器/:/Encoder/);
   }
   if(language==='en')assert.doesNotMatch(diagram.textContent,/[\u3400-\u9fff]/,kind);
  }
  const card=M.extraCard('self-heads','encoder-layer');assert.match(card.meaning,language==='zh'?/全部 4 个输入特征/:/all four input features/);
 }
 assert.equal(app.eval('JSON.stringify(state)'),before);assert.deepEqual([...app.c.I18n.missing],[]);
});
