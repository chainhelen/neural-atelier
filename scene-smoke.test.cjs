const fs=require('node:fs'),vm=require('node:vm'),test=require('node:test'),assert=require('node:assert/strict');
const {parseHTML}=require('linkedom');
async function boot({reduced=true,frameMs=1000,onFrame=()=>{},languages=['zh-CN'],timers={setTimeout,clearTimeout},viewport={width:1400,height:760},url}={}){
 const Three=await import('./vendor/three.module.js');
 const {document,window}=parseHTML(fs.readFileSync(__dirname+'/index.html','utf8'));const native=document.createElement.bind(document);
 window.HTMLElement.prototype.scrollIntoView=function(){};
 document.createElement=function(name){const el=native(name);if(name==='canvas')el.getContext=()=>({fillRect(){},fillText(t){el.drawnText=String(t);},measureText(t){return {width:String(t).length*60}}});return el;};
 for(const el of document.querySelectorAll('dialog')){el.showModal=function(){this.open=true};el.close=function(){this.open=false};}
 for(const el of document.querySelectorAll('*')){Object.defineProperty(el,'clientWidth',{get:()=>el.id==='scene'?viewport.width:100});Object.defineProperty(el,'clientHeight',{get:()=>el.id==='scene'?viewport.height:60});el.focus=()=>{};el.scrollIntoView=()=>{};}
 let view,renderedCamera;const scenes=[];class Renderer{constructor(){this.domElement=native('canvas');this.shadowMap={};this.capabilities={getMaxAnisotropy:()=>8};}setPixelRatio(){}setClearColor(){}setSize(){}render(scene,camera){renderedCamera=camera;view={position:camera.position.toArray(),width:camera.right-camera.left};scenes.push(scene);scene.updateMatrixWorld();camera.updateMatrixWorld();scene.traverse(o=>{if(o.isMesh)assert.ok(o.matrixWorld.elements.every(Number.isFinite),'nonfinite transform '+o.type);});}}
 const storage=new Map();let now=0;const c={document,console,navigator:{languages},T:{...Three,WebGLRenderer:Renderer},devicePixelRatio:2,ResizeObserver:class{observe(){}},requestAnimationFrame:cb=>setImmediate(()=>{now+=frameMs;onFrame({now,document,window});cb(now);}),cancelAnimationFrame:clearImmediate,performance:{now:()=>now},matchMedia:()=>({matches:reduced}),setTimeout:timers.setTimeout,clearTimeout:timers.clearTimeout,localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)}};
 // Existing scene tests intentionally omit browser navigation globals.
 if(url){
  const listeners=new Map(),entries=[new URL(url).href];let index=0;
  c.URL=URL;c.URLSearchParams=URLSearchParams;c.location=new URL(url);
  c.addEventListener=(type,fn)=>{if(!listeners.has(type))listeners.set(type,[]);listeners.get(type).push(fn);};
  c.dispatchEvent=event=>{for(const fn of listeners.get(event.type)||[])fn(event);};
  const visit=next=>{if(next<0||next>=entries.length)return;const oldURL=c.location.href;index=next;c.location.href=entries[index];c.dispatchEvent(new window.Event('popstate'));if(new URL(oldURL).hash!==c.location.hash)c.dispatchEvent(new window.Event('hashchange'));};
  c.history={get length(){return entries.length;},pushState(_state,_title,target){c.location.href=new URL(target,c.location.href).href;entries.splice(++index);entries.push(c.location.href);},replaceState(_state,_title,target){c.location.href=new URL(target,c.location.href).href;entries[index]=c.location.href;},back:()=>visit(index-1),forward:()=>visit(index+1)};
 }
 c.window=c;vm.createContext(c);
 for(const f of ['locales/en.js','i18n.js','music.js','shell.js','model.js','campaign.js','engine-model.js','original-curriculum.js','transformer-demo.js','cloud-curriculum.js','lesson-io.js','structure-map.js'])vm.runInContext(fs.readFileSync(__dirname+'/'+f,'utf8'),c,{filename:f});
 for(const f of ['module-cards.js','level-progress.js','glossary.js','scene-art.js'])vm.runInContext(fs.readFileSync(__dirname+'/'+f,'utf8').replaceAll('export ',''),c,{filename:f});
 let src=fs.readFileSync(__dirname+'/game.js','utf8').replace(/^import .*\n/gm,'').replace("const T=await import('./vendor/three.module.js');",'const T=window.T;').replace(/const \{createSceneArt\}=await import\('\.\/scene-art\.js\?v=[^']+'\);/,'');
 src=src.replace('world={relocalize','world={inspectTracking:()=>followTracking(),inspectFollow:()=>({packet:followedPacket,target:cameraTarget.clone(),active:activeFollowPackets(),seed:followSeed,zoom,followZoom,gliding:!!followFlight}),relocalize');
 src+='\nwindow.testAPI={switchLevel,navigate,readNavigation,run,togglePause,refresh,select,assignDevice,chooseOutput,chooseInput,chooseModule,placeModule,getState:()=>state,setState:s=>state=s,getWorld:()=>world,getFault:()=>fault,isCompleted:id=>progress.completed(id)};';
 await vm.runInContext('(async()=>{'+src+'})()',c);
 assert.ok(c.testAPI.getWorld(),'3D boot failed');return {c,document,window,api:c.testAPI,scenes,time:()=>now,setFrameMs:value=>{frameMs=value;},view:()=>view,camera:()=>renderedCamera};
}
test('add-norm reports a visible residual difference and preserves precision for close numeric failures',async()=>{
 const app=await boot(),M=app.c.NeuralPrologue,doc=app.document;app.api.switchLevel('add-norm');
 const state=M.solutions('add-norm');state.nodes=state.nodes.filter(n=>n.id!=='add');state.nodes.find(n=>n.id==='layernorm2').links=['relu-ffn'];app.api.setState(state);app.api.getWorld().rebuild();
 const saved=JSON.stringify(state);await app.api.run();
 let values=[...doc.querySelectorAll('.local-fault div>b')].map(el=>el.textContent);
 assert.deepEqual(values,['[-1, 1]','[1, -1]']);assert.equal(doc.querySelector('.fault-precision-note'),null);assert.equal(JSON.stringify(state),saved);
 const originalSamples=M.samplesFor;M.samplesFor=id=>originalSamples(id).map((s,i)=>id==='add-norm'&&i===0?{...s,v:[1,0]}:s);
 app.api.getWorld().rebuild();await app.api.run();
 values=[...doc.querySelectorAll('.local-fault div>b')].map(el=>el.textContent);
 assert.deepEqual(values,['[0.999950, -0.999950]','[0.999999, -0.999999]']);assert.ok(doc.querySelector('.fault-precision-note'));assert.match(doc.querySelector('#test-results .fail').textContent,/0\.999950/);
 app.c.I18n.setPreference('en');assert.match(doc.querySelector('.fault-precision-note').textContent,/More decimals/);assert.doesNotMatch(doc.querySelector('.local-fault').textContent,/[\u3400-\u9fff]/);
 doc.querySelector('.fault-close').dispatchEvent(new app.window.Event('click',{bubbles:true}));assert.equal(doc.querySelector('.local-fault'),null);assert.equal(JSON.stringify(state),saved);
 M.samplesFor=originalSamples;app.api.setState(M.solutions('add-norm'));app.api.getWorld().rebuild();await app.api.run();assert.equal(doc.querySelectorAll('#test-results .pass').length,5);assert.equal(app.api.getFault(),null);
});
test('every board can pan with the button or a modifier without moving modules or wiring',async()=>{
 const app=await boot(),doc=app.document,M=app.c.NeuralPrologue,T=app.c.T;
 const fire=(target,type,props={})=>{const e=new app.window.Event(type,{bubbles:true,cancelable:true});Object.assign(e,props);target.dispatchEvent(e);};
 const drag=(target,props={})=>{fire(target,'pointerdown',{clientX:40,clientY:40,...props});fire(doc,'pointermove',{clientX:180,clientY:110});fire(doc,'pointerup',{clientX:180,clientY:110});};
 for(const id of ['wire','embedding','ffn','kv-cache']){
  app.api.switchLevel(id);app.api.setState(M.solutions(id));app.api.getWorld().rebuild();app.api.refresh();
  const saved=JSON.stringify(app.api.getState()),button=doc.querySelector('#demo-pan'),host=doc.querySelector('#scene'),direction=()=>app.camera().getWorldDirection(new T.Vector3()),initial=app.camera().position.clone();
  host.getBoundingClientRect=()=>({left:0,top:0,width:1400,height:760});
  assert.equal(button.hidden,false);assert.equal(button.getAttribute('aria-pressed'),'false');
  button.onclick();assert.equal(button.getAttribute('aria-pressed'),'true');const heading=direction();
  drag(host);assert.ok(app.camera().position.distanceTo(initial)>1);assert.ok(direction().distanceTo(heading)<1e-9);
  const label=doc.querySelector('.node-label')||doc.querySelector('[data-output="R"]'),before=app.camera().position.clone();
  drag(label);fire(label,'click');assert.ok(app.camera().position.distanceTo(before)>1);assert.equal(JSON.stringify(app.api.getState()),saved);
  fire(doc.querySelector('[data-output="R"]'),'click');fire(doc.querySelector('[data-input="sink:0"]'),'click');assert.equal(JSON.stringify(app.api.getState()),saved,'pan mode does not start a wire');
  button.onclick();doc.querySelector('#camera-reset').onclick();assert.ok(app.camera().position.distanceTo(initial)<1e-9);
  for(const props of [{shiftKey:true},{button:2}]){const before=app.camera().position.clone();drag(host,props);assert.ok(app.camera().position.distanceTo(before)>1);assert.ok(direction().distanceTo(heading)<1e-9);}
  drag(host);assert.ok(direction().distanceTo(heading)>.01,id+' normal dragging orbits again after leaving pan mode');
  assert.equal(JSON.stringify(app.api.getState()),saved);
 }
 app.c.I18n.setPreference('en');assert.equal(doc.querySelector('#demo-pan').textContent,'Pan view');
});

test('numeric matrix outlets hold arrived values, pulse, then clear before the next batch',async()=>{
 for(const levelId of ['embedding','positions','qkv','scores','softmax','weighted-sum','normalize']){
 let app,mode='complete',paused=false,pauseFrames=0,pausedView,pausedResult,stopped=false;const seen=new Set(),holds=new Map(),pulses=[];
 const outlet=()=>{const out=[];app.scenes.at(-1).traverse(o=>{if(o.userData.sinkRow!=null){const text=o.children.filter(c=>c.material?.map).map(c=>c.material.map.image.drawnText);out[o.userData.sinkRow]={values:text.slice(2),status:text[1],flash:o.children[0].material.emissiveIntensity};}});return out;};
 const blank=()=>outlet().every(r=>r.values.every(v=>v==='·'));
 const fire=(target,type,props={})=>{const e=new app.window.Event(type,{bubbles:true,cancelable:true});Object.assign(e,props);target.dispatchEvent(e);};
 app=await boot({reduced:false,frameMs:80,onFrame:({document,now})=>{
  if(!app||mode==='stopped')return;const M=app.c.NeuralPrologue,name=document.querySelector('#sample-name').textContent,sample=M.samplesFor(levelId).find(s=>s.word===name);if(!sample)return;
  if(levelId==='scores'){
   const query=[],key=[];app.scenes.at(-1).traverse(o=>{if(o.userData.scoreQuery!=null)query[o.userData.scoreQuery]=o.material.map.image.drawnText;if(o.userData.scoreKey!=null)key[o.userData.scoreKey]=o.material.map.image.drawnText;});
   assert.deepEqual(query,sample.word.split(' / '));assert.deepEqual(key,query,'both axes follow the current token order');
  }
  const source=document.querySelector('.packet[data-from="R"][data-motion="wire"]:not([hidden])'),outgoing=document.querySelector('.packet[data-to="sink"][data-motion="wire"]:not([hidden])');
  if(source||outgoing){assert.ok(blank(),'the previous matrix is gone and this one has not arrived yet');seen.add((source?'cleared:':'travelling:')+name);}
  const displayed=outlet(),values=displayed.flatMap(r=>r.values);
  if(mode==='disconnected'){assert.ok(values.every(v=>v==='·'||v==='—'));return;}
  if(!values.some(v=>v!=='·')||displayed.some(r=>r.status!=='·'))return;
  assert.deepEqual(values,Array.from(M.evaluate(app.api.getState(),sample.v).output,v=>String(Number(v.toFixed(2)))));
  if(mode==='stop'){mode='stopped';stopped=true;app.api.run();assert.ok(blank(),'stop clears the ungraded matrix and its pulse');return;}
  if(!holds.has(name))holds.set(name,[]);holds.get(name).push(now);pulses.push(displayed[0].flash);
  if(!paused){paused=true;app.api.togglePause();pausedResult=JSON.stringify(displayed);pausedView=app.camera().position.clone();document.querySelector('#demo-pan').onclick();}
  if(pauseFrames<3){
   assert.equal(JSON.stringify(outlet()),pausedResult,'pause freezes numbers and pulse');pauseFrames++;
   if(pauseFrames===2){fire(document.querySelector('#scene'),'pointerdown',{clientX:600,clientY:300});fire(document,'pointermove',{clientX:680,clientY:340});fire(document,'pointerup',{});assert.ok(app.camera().position.distanceTo(pausedView)>1);}
   if(pauseFrames===3)app.api.togglePause();
  }
 }});
 const M=app.c.NeuralPrologue,doc=app.document;doc.querySelector('#welcome').close();app.api.switchLevel(levelId);const state=M.solutions(levelId);app.api.setState(state);app.api.getWorld().rebuild();const saved=JSON.stringify(state);
 assert.ok(blank());await app.api.run();assert.equal(JSON.stringify(state),saved);assert.equal(doc.querySelectorAll('#test-results .pass').length,3);
 for(const {word}of M.samplesFor(levelId)){assert.ok(seen.has('cleared:'+word));assert.ok(seen.has('travelling:'+word));const times=holds.get(word);assert.ok(times.at(-1)-times[0]>=1600,'the slowest setting holds each complete matrix long enough to read');}
 assert.ok(Math.max(...pulses)>.3);assert.ok(pulses.includes(0));assert.equal(pauseFrames,3);
 assert.deepEqual(outlet().flatMap(r=>r.values),Array.from(M.evaluate(state,M.samplesFor(levelId).at(-1).v).output,v=>String(Number(v.toFixed(2)))));assert.ok(outlet().every(r=>r.status==='✓'),'the final matrix remains after completion');
 mode='stop';const replay=app.api.run();assert.ok(blank());await replay;assert.ok(stopped);
 mode='disconnected';state.sink=null;app.api.getWorld().rebuild();await app.api.run();assert.equal(doc.querySelectorAll('#test-results .fail').length,1);assert.ok(outlet().every(r=>r.flash===0));
 }
});

test('score outlet previews the actual wrong matrix, with a static hold for reduced motion',async()=>{
 let app,seen=false;const times=[];
 app=await boot({reduced:true,frameMs:80,onFrame:({now,document})=>{
  if(!app||!document.body.classList.contains('network-running'))return;
  const out=[],status=[],flash=[];app.scenes.at(-1).traverse(o=>{if(o.userData.sinkRow!=null){const text=o.children.filter(c=>c.material?.map).map(c=>c.material.map.image.drawnText);out[o.userData.sinkRow]=text.slice(2);status.push(text[1]);flash.push(o.children[0].material.emissiveIntensity);}});
  if(out.flat().includes('·')||status.some(s=>s!=='·'))return;
  const M=app.c.NeuralPrologue,x=M.samplesFor('scores')[0].v,actual=M.evaluate(app.api.getState(),x).output;
  assert.deepEqual(out.flat(),Array.from(actual,v=>String(Number(v.toFixed(2)))));assert.notDeepEqual(out.flat(),Array.from(M.reference(x,'scores'),v=>String(Number(v.toFixed(2)))));
  assert.ok(flash.every(v=>v===0));seen=true;times.push(now);
 }});
 const M=app.c.NeuralPrologue,s=M.solutions('scores');s.nodes.find(n=>n.id==='score').links.reverse();app.api.switchLevel('scores');app.api.setState(s);app.api.getWorld().rebuild();await app.api.run();
 assert.ok(seen);assert.ok(times.at(-1)-times[0]>=1600);assert.ok(app.api.getFault());assert.equal(app.document.querySelectorAll('#test-results .fail').length,1);
 app.c.I18n.setPreference('en');const names=[];app.scenes.at(-1).traverse(o=>{if(o.userData.scoreKey!=null)names[o.userData.scoreKey]=o.material.map.image.drawnText;});assert.deepEqual(names,M.samplesFor('scores')[0].word.split(' / '));assert.ok(names.every(s=>!/[\u3400-\u9fff]/.test(s)));
 app.api.switchLevel('qkv');let axes=0;app.scenes.at(-1).traverse(o=>{if(o.userData.scoreKey!=null||o.userData.scoreQuery!=null)axes++;});assert.equal(axes,0,'pair axes do not leak onto vector feature outputs');
});
test('a full restart replaces visible residual wires with the tutorial start and undo restores the scene',async()=>{
 const app=await boot(),doc=app.document;app.api.switchLevel('residual');
 app.api.setState(app.c.NeuralPrologue.solutions('residual'));app.api.getWorld().rebuild();app.api.refresh();
 const board=JSON.stringify(app.api.getState()),connections=doc.querySelectorAll('.port.input.connected').length;
 assert.ok(connections>0);
 doc.querySelector('#progress-restart').onclick();doc.querySelector('#progress-reset-apply').onclick();
 assert.equal(doc.querySelectorAll('.port.input.connected').length,0);assert.equal(doc.querySelectorAll('.node-label').length,0);assert.equal(doc.querySelector('#undo').disabled,true);assert.equal(app.api.getState().sink,null);
 doc.querySelector('#progress-restart-undo').onclick();assert.equal(JSON.stringify(app.api.getState()),board);assert.equal(doc.querySelectorAll('.port.input.connected').length,connections);assert.equal(doc.querySelectorAll('.node-label').length,2);
});

test('the ready-to-run full Transformer renders and completes every basic operator without editing',async()=>{
 const app=await boot({frameMs:5000}),M=app.c.NeuralPrologue;
 app.api.switchLevel('transformer-demo');const before=JSON.stringify(app.api.getState());
 assert.equal(app.document.querySelectorAll('.node-label').length,67);
 assert.equal(app.document.querySelectorAll('.demo-section-label').length,4);
 app.api.chooseOutput('R');app.api.chooseInput('sink',0);assert.equal(JSON.stringify(app.api.getState()),before);
 await app.api.run();
 assert.equal(app.api.getFault(),null);assert.equal(app.document.querySelector('#level-complete').open,true);
 assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);
 assert.deepEqual(Array.from(app.document.querySelectorAll('#demo-transcript .generated-token'),el=>el.textContent),['你','爱','狗']);
 assert.equal(JSON.stringify(app.api.getState()),before);assert.equal(app.document.querySelectorAll('.packet:not([hidden])').length,0);
 app.document.querySelector('#complete-replay').onclick();assert.equal(app.document.querySelector('#level-complete').open,false);
 assert.equal(JSON.stringify(app.api.getState()),before);
 app.c.I18n.setPreference('en');assert.deepEqual([...app.c.I18n.missing],[]);
 app.api.switchLevel('wire');assert.equal(app.document.querySelectorAll('.demo-section-label').length,0);assert.equal(app.document.querySelector('#demo-views').hidden,true);
});
test('floating demonstration supports full orbit, pan and cursor zoom without editing',async()=>{
 const app=await boot(),doc=app.document,T=app.c.T;app.api.switchLevel('transformer-demo');
 const saved=JSON.stringify(app.api.getState()),scene=app.scenes.at(-1),host=doc.querySelector('#scene');
 host.getBoundingClientRect=()=>({left:0,top:0,width:1400,height:760});
 assert.ok(app.camera().isPerspectiveCamera);assert.equal(scene.children.find(o=>o.userData.environmentFloor).children.length,0);
 const assembly=scene.children.find(o=>o.children.some(n=>n.userData.node==='demo-up'));
 assert.equal(assembly.children.find(o=>o.userData.node==='demo-up').position.y,6);assert.equal(assembly.children.find(o=>o.userData.node==='demo-up-2').position.y,-6);
 const fire=(target,type,props)=>{const e=new app.window.Event(type,{bubbles:true,cancelable:true});Object.assign(e,props);target.dispatchEvent(e);};
 const drag=(dx,dy,props={})=>{fire(host,'pointerdown',{clientX:700,clientY:360,...props});fire(doc,'pointermove',{clientX:700+dx,clientY:360+dy});fire(doc,'pointerup',{clientX:700+dx,clientY:360+dy});};
 const direction=()=>app.camera().getWorldDirection(new T.Vector3());
 const before=direction();drag(700,0);assert.ok(direction().angleTo(before)>1,'orbit is no longer limited to the old board tilt');
 const position=app.camera().position.clone(),heading=direction();drag(240,100,{shiftKey:true});assert.ok(app.camera().position.distanceTo(position)>10);assert.ok(direction().distanceTo(heading)<1e-9,'pan preserves heading');
 doc.querySelector('#demo-pan').onclick();assert.equal(doc.querySelector('#demo-pan').getAttribute('aria-pressed'),'true');const panStart=app.camera().position.clone();drag(-150,0);assert.ok(app.camera().position.distanceTo(panStart)>5);
 doc.querySelector('#camera-reset').onclick();doc.querySelector('#demo-pan').onclick();
 const cursor=new T.Vector2(.45,-.1),ray=new T.Raycaster();ray.setFromCamera(cursor,app.camera());
 const anchor=ray.ray.intersectPlane(new T.Plane().setFromNormalAndCoplanarPoint(direction(),new T.Vector3()),new T.Vector3());
 fire(host,'wheel',{clientX:(cursor.x+1)*700,clientY:(1-cursor.y)*380,deltaY:-800});
 const projected=anchor.clone().project(app.camera());assert.ok(Math.abs(projected.x-cursor.x)<1e-9&&Math.abs(projected.y-cursor.y)<1e-9,'zoom stays anchored under the cursor');
 assert.equal(JSON.stringify(app.api.getState()),saved);app.api.switchLevel('wire');assert.ok(app.camera().isOrthographicCamera);assert.ok(scene.children.find(o=>o.userData.environmentFloor).visible);assert.equal(doc.querySelector('#demo-pan').hidden,false);assert.equal(doc.querySelector('#demo-follow-view').hidden,true);
});
test('follow switch cycles live data while paused and follows the selected branch through handoffs',async()=>{
 let app,phase=0,frozen=null;const selectedNode='masked-a-q-2',targets=new Set();
 app=await boot({reduced:false,frameMs:150,onFrame:()=>{
  if(!app||phase===3)return;
  const world=app.api.getWorld(),control=app.document.querySelector('#demo-follow-next'),view=world.inspectFollow();
  if(!phase&&view.active.some(p=>p.node?.id===selectedNode)){
   app.api.togglePause();assert.equal(control.disabled,false);
   const count=view.active.length;assert.ok(count>3);
   for(let i=0;i<count;i++){control.onclick();const current=world.inspectFollow();targets.add(current.packet);assert.equal(app.document.querySelector('#demo-follow').getAttribute('aria-pressed'),'true');}
   assert.equal(targets.size,count,'every active flow can be selected without a name or number');
   for(let i=0;i<count&&world.inspectFollow().packet.node?.id!==selectedNode;i++)control.onclick();
   assert.equal(world.inspectFollow().packet.node.id,selectedNode);
   frozen={packet:world.inspectFollow().packet,position:world.inspectFollow().packet.mesh.position.clone(),target:world.inspectFollow().target.clone(),time:app.time()};phase=1;
  }else if(phase===1){
   assert.equal(view.packet,frozen.packet);assert.ok(view.packet.mesh.position.distanceTo(frozen.position)<1e-9,'switching the camera does not advance paused data');
   if(app.time()-frozen.time>=800){assert.ok(view.target.distanceTo(frozen.target)>.1,'camera glides to the selection while data stays paused');app.api.togglePause();phase=2;}
  }else if(phase===2&&view.packet?.node?.id==='mask'){
   assert.ok(targets.has(frozen.packet));phase=3;app.api.run();
  }
 }});
 app.api.switchLevel('transformer-demo');const doc=app.document,world=app.api.getWorld(),control=doc.querySelector('#demo-follow-next');
 assert.equal(control.hidden,false);assert.equal(control.disabled,false);const saved=JSON.stringify(app.api.getState());
 control.onclick();const first=world.inspectFollow();control.onclick();const second=world.inspectFollow();
 assert.notEqual(first.seed,second.seed);assert.equal(first.target.distanceTo(second.target),0,'rapid clicks retarget without an immediate camera jump');
 const speed=doc.querySelector('#playback-speed');speed.value='1';speed.dispatchEvent(new app.window.Event('input'));
 await app.api.run();assert.equal(phase,3,'selected decoder Q continues through its score to its causal mask');assert.equal(JSON.stringify(app.api.getState()),saved);
 assert.equal(control.disabled,false);assert.equal(world.inspectFollow().packet,null,'stopping removes the old follow target');
 app.c.I18n.setPreference('en');assert.equal(control.textContent,'Switch flow');assert.deepEqual([...app.c.I18n.missing],[]);
 app.api.switchLevel('wire');assert.equal(control.hidden,true);
});

test('switching follows a smooth camera flight and consecutive clicks retarget from the current view',async()=>{
 const app=await boot({reduced:false,frameMs:50});app.api.switchLevel('transformer-demo');
 const world=app.api.getWorld(),button=app.document.querySelector('#demo-follow-next'),before=world.inspectFollow().target.clone(),saved=JSON.stringify(app.api.getState());
 const nextFrame=()=>new Promise(resolve=>setImmediate(resolve));
 button.onclick();assert.equal(world.inspectFollow().target.distanceTo(before),0);assert.equal(button.disabled,false);
 for(let i=0;i<3;i++)await nextFrame();
 const midway=world.inspectFollow().target.clone();assert.ok(midway.distanceTo(before)>0);assert.ok(world.inspectFollow().gliding);
 button.onclick();assert.equal(world.inspectFollow().target.distanceTo(midway),0,'a second click starts from the currently rendered position');
 for(let i=0;i<30&&world.inspectFollow().gliding;i++)await nextFrame();
 assert.equal(world.inspectFollow().gliding,false);assert.ok(world.inspectFollow().target.distanceTo(midway)>1);
 button.onclick();app.document.querySelector('#demo-follow').onclick();const stopped=world.inspectFollow().target.clone();await nextFrame();assert.equal(world.inspectFollow().gliding,false);assert.equal(world.inspectFollow().target.distanceTo(stopped),0);
 assert.equal(JSON.stringify(app.api.getState()),saved);
});

test('playback and zoom sliders keep the selected flow and viewing distance while paused and running',async()=>{
 let app,phase=0,frames=0,frozen,chosenZoom,heldTarget;
 const change=(selector,value)=>{const el=app.document.querySelector(selector);el.value=String(value);el.dispatchEvent(new app.window.Event('input'));};
 app=await boot({reduced:false,frameMs:100,onFrame:()=>{
  if(!app||phase===4)return;
  const doc=app.document,world=app.api.getWorld(),view=world.inspectFollow();
  if(phase===0&&view.active.length){
   doc.querySelector('#demo-follow-next').onclick();app.api.togglePause();
   const selected=world.inspectFollow();frozen={packet:selected.packet,p:selected.packet.p.clone()};heldTarget=selected.target.clone();
   change('#playback-speed',1);change('#camera-zoom',5);chosenZoom=world.inspectFollow().zoom;
   assert.equal(world.inspectFollow().packet,frozen.packet);assert.equal(world.inspectFollow().target.distanceTo(heldTarget),0);phase=1;
  }else if(phase===1){
   assert.ok(view.packet.p.distanceTo(frozen.p)<1e-10);assert.equal(view.zoom,chosenZoom);
   if(++frames===2){app.api.togglePause();phase=2;frames=0;}
  }else if(phase===2){
   assert.equal(view.zoom,chosenZoom,'automatic framing must not undo the chosen zoom');
   if(++frames===8){
    assert.ok(view.target.distanceTo(heldTarget)>.01,'camera still tracks the moving data');
    const selected=view.packet;change('#playback-speed',2);change('#camera-zoom',9);chosenZoom=world.inspectFollow().zoom;
    assert.equal(world.inspectFollow().packet,selected);phase=3;frames=0;
   }
  }else if(phase===3){
   assert.equal(view.zoom,chosenZoom);
   if(++frames===8){phase=4;app.api.run();}
  }
  assert.equal(doc.querySelector('#demo-follow').getAttribute('aria-pressed'),'true');
 }});
 app.api.switchLevel('transformer-demo');const saved=JSON.stringify(app.api.getState());await app.api.run();
 assert.equal(phase,4);assert.equal(JSON.stringify(app.api.getState()),saved);
 const profile=app.document.querySelector('#demo-follow-view');profile.value='close';profile.onchange({target:profile});assert.equal(app.api.getWorld().inspectFollow().followZoom,null,'explicit distance preset replaces the custom distance');
 app.api.switchLevel('wire');app.api.switchLevel('transformer-demo');assert.equal(app.api.getWorld().inspectFollow().followZoom,null);
});

test('automatic following stays with encoder transfers instead of detouring to decoder work or parked residuals',async()=>{
 let app,watching=false,stopped=false;const transfers=new Set();
 app=await boot({reduced:false,frameMs:100,onFrame:()=>{
  if(!app||stopped)return;
  const world=app.api.getWorld(),view=world.inspectFollow();
  if(view.active.some(p=>p.node?.id==='softmax'))watching=true;
  if(!watching)return;
  const tracking=world.inspectTracking();
  assert.ok(tracking.length);
  for(const p of tracking){
   assert.equal(app.c.NeuralPrologue.demo.sections[p.node?.id||p.edge?.to],'encoder','in-flight encoder data retains the camera while decoder operators run');
   if(p.motion==='wire'){
    assert.notEqual(p.edge.to,'demo-add','parked residual input must not pull the camera toward a future operation');
    transfers.add(p.edge.from);
   }
  }
  if(view.active.some(p=>p.node?.id==='original-output')){stopped=true;app.api.run();}
 }});
 app.api.switchLevel('transformer-demo');const saved=JSON.stringify(app.api.getState());
 const speed=app.document.querySelector('#playback-speed');speed.value='1';speed.dispatchEvent(new app.window.Event('input'));
 await app.api.run();assert.ok(stopped);
 for(const id of ['softmax','softmax-2','aggregate','aggregate-2','wide-concat'])assert.ok(transfers.has(id),'followed transfer from '+id);
 assert.equal(JSON.stringify(app.api.getState()),saved);
});

test('demo Softmax hands off while the independent masked branch still runs, and joins wait for their inputs',async()=>{
 let app,stopped=false,overlap=false,softStart=null,softEnd=null;const began=new Map(),finished=new Map(),previous=new Set();
 app=await boot({reduced:false,frameMs:100,onFrame:({now,document})=>{
  if(!app||stopped)return;
  const scene=app.scenes.at(-1),units=scene.children.flatMap(g=>g.children).filter(g=>g.userData.node),active=new Set(units.filter(g=>g.userData.processing).map(g=>g.userData.node));
  for(const id of previous)if(!active.has(id)&&!finished.has(id))finished.set(id,now);
  const state=app.api.getState();
  for(const id of active)if(!began.has(id)){
   began.set(id,now);const n=state.nodes.find(n=>n.id===id);
   for(const from of n.links)if(!['R','Z'].includes(from))assert.ok(finished.has(from)&&now>finished.get(from),id+' must receive '+from+' before starting');
  }
  if(active.has('softmax-2')&&softStart===null)softStart=now;
  const outgoing=document.querySelector('.packet[data-motion="wire"][data-from="softmax-2"][data-to="aggregate-2"]:not([hidden])');
  if(outgoing&&softEnd===null){softEnd=now;overlap=active.has('mask');}
  previous.clear();for(const id of active)previous.add(id);
  // Reach the encoder/decoder join as well, then stop this focused timing run.
  if(active.has('score-5')){stopped=true;app.api.run();}
 }});
 app.api.switchLevel('transformer-demo');const speed=app.document.querySelector('#playback-speed');speed.value='1';speed.dispatchEvent(new app.window.Event('input'));const saved=JSON.stringify(app.api.getState());await app.api.run();
 assert.ok(stopped);assert.ok(overlap,'encoder Softmax leaves while decoder causal masking is still processing');
 assert.ok(Math.abs(softEnd-softStart-650)<=100,'Softmax uses its own one-step clock, without the depth-wide hold');
 for(const id of ['aggregate-3','wide-concat','demo-add','score-5'])assert.ok(began.has(id),'join checked: '+id);
 assert.equal(JSON.stringify(app.api.getState()),saved);assert.equal(app.document.querySelectorAll('.packet,.projection-calculation').length,0);
});

test('floating demonstration preserves the operator clock, all operators and paused navigation',async()=>{
 let app,pausedOnce=false,initialCamera;const processed=new Set();
 app=await boot({reduced:false,frameMs:1500,onFrame:({document})=>{
  if(!app)return;for(const scene of [app.scenes.at(-1)])for(const group of scene.children)for(const n of group.children)if(n.userData.node&&n.userData.processing)processed.add(n.userData.node);
  if(!pausedOnce&&processed.size>8){pausedOnce=true;assert.ok(app.camera().position.distanceTo(initialCamera)>5,'the camera follows work without user input');app.api.togglePause();const camera=app.camera().position.clone();document.querySelector('[data-demo-view="cross"]').click();assert.ok(app.camera().position.distanceTo(camera)>1);assert.equal(document.querySelector('#demo-follow').getAttribute('aria-pressed'),'false','choosing a region takes manual control');document.querySelector('#demo-follow').onclick();assert.equal(document.querySelector('#demo-follow').getAttribute('aria-pressed'),'true');app.api.togglePause();}
 }});
 app.api.switchLevel('transformer-demo');initialCamera=app.camera().position.clone();const saved=JSON.stringify(app.api.getState()),start=app.time();await app.api.run();const elapsed=app.time()-start;
 assert.ok(elapsed>600000,'slow playback is never squeezed into a fixed one-minute runtime');assert.equal(processed.size,67);assert.ok(pausedOnce);assert.ok(app.camera().position.distanceTo(initialCamera)<1e-8,'follow returns to the model center before completion');assert.equal(app.document.querySelectorAll('#demo-transcript .generated-token').length,3);
 assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);assert.equal(JSON.stringify(app.api.getState()),saved);assert.equal(app.document.querySelectorAll('.packet').length,0);
});
test('generation feedback can pause, change follow distance and stop without starting another round',async()=>{
 let app,feedbackFrames=0,position,stopped=false;
 app=await boot({frameMs:500,onFrame:({document})=>{
  if(!app||!document.querySelector('.feedback-packet'))return;
  const packet=app.scenes.at(-1).children.find(o=>o.isMesh&&o.geometry?.type==='SphereGeometry');
  feedbackFrames++;
  if(feedbackFrames===1){app.api.togglePause();position=packet.position.clone();}
  if(feedbackFrames===3){assert.ok(packet.position.distanceTo(position)<1e-10);const select=document.querySelector('#demo-follow-view');select.value='close';select.onchange({target:select});assert.equal(document.querySelector('#demo-follow').getAttribute('aria-pressed'),'true');app.api.togglePause();}
  if(feedbackFrames===5){stopped=true;app.api.run();}
 }});
 app.api.switchLevel('transformer-demo');const graph=JSON.stringify(app.api.getState());
 await app.api.run();assert.ok(stopped);assert.equal(app.document.querySelector('#level-complete').open,undefined);assert.equal(app.document.querySelectorAll('.packet').length,0);assert.equal(app.document.querySelectorAll('#demo-transcript .generated-token').length,1);assert.equal(JSON.stringify(app.api.getState()),graph);
 app.api.switchLevel('wire');assert.equal(app.document.querySelector('#demo-generation').hidden,true);
});
test('demo slow playback exposes each column for one second, respects speed changes and reduced motion',async()=>{
 for(const reduced of [false,true]){
  let app,previousColumn=null,lastColumnAt=0,slowSpan=0,fastSpan=0,spedUp=false;
  app=await boot({reduced,frameMs:80,onFrame:({now,document})=>{
   if(!app)return;
   const active=app.scenes.at(-1).children.flatMap(g=>g.children).find(g=>g.userData.node==='masked-a-q');
   const col=active?.userData.activeColumn;
   if(col==null)return;
   if(previousColumn!==col){
    if(!spedUp&&previousColumn===0&&col===1){slowSpan=now-lastColumnAt;const input=document.querySelector('#playback-speed');input.value='1';input.dispatchEvent(new app.window.Event('input'));spedUp=true;}
    else if(spedUp&&previousColumn===0&&col===1){fastSpan=now-lastColumnAt;app.api.run();}
    previousColumn=col;lastColumnAt=now;
   }
  }});
  app.api.switchLevel('transformer-demo');await app.api.run();
  assert.ok(slowSpan>=960&&slowSpan<=1120,'0.25x gives one second per column even with reduced motion: '+slowSpan);
  assert.ok(fastSpan>=180&&fastSpan<=340,'1x gives about 260 ms per column: '+fastSpan);
 }
});
test('demonstration cards show the selected instance and return to observation controls',async()=>{
 const app=await boot(),M=app.c.NeuralPrologue,doc=app.document;
 app.api.switchLevel('transformer-demo');const s=app.api.getState(),saved=JSON.stringify(s),ev=M.evaluate(s,M.samplesFor(s.levelId)[0].v);
 for(const language of ['zh','en']){
  app.c.I18n.setPreference(language);
  for(const id of ['demo-up','demo-up-2']){
   app.api.getWorld().focus(id);
   const values=[...doc.querySelectorAll('#card-example-values .result i')].map(el=>Number(el.textContent));
   assert.deepEqual(values,Array.from(ev.values[id],x=>Number(x.toFixed(2))));
   assert.equal(doc.querySelector('.card-path').hidden,true,'do not imply that all 67 parallel operators form one serial chain');
   assert.doesNotMatch(doc.querySelector('#card-return').textContent,/搭建|building/);
   doc.querySelector('#card-close').onclick();assert.doesNotMatch(doc.querySelector('#scene-help').textContent,/摆放|删除|move modules|delete/);
  }
 }
 assert.equal(JSON.stringify(app.api.getState()),saved);assert.deepEqual([...app.c.I18n.missing],[]);
});
test('a solved network gathers into its matching component and flies into the shelf before completion',async()=>{
 let app,root,assembly,originalDisposals=0,temporaryDisposals=0,landed=false;const phases=new Set();
 const resources=group=>{const set=new Set();group.traverse(o=>{if(o.geometry)set.add(o.geometry);for(const material of o.material?(Array.isArray(o.material)?o.material:[o.material]):[]){set.add(material);if(material.map)set.add(material.map);}});return set;};
 app=await boot({reduced:false,frameMs:120,onFrame:({document})=>{
  if(!app)return;const scene=app.scenes.at(-1),effect=scene.children.find(o=>o.userData.fusion);if(!effect)return;
  const shelf=document.querySelector('[data-add-module="single-attention"]');shelf.getBoundingClientRect=()=>({left:24,top:150,width:104,height:60});
  assert.equal(document.querySelector('#component-fusion').hidden,false);assert.ok(!document.querySelector('#level-complete').open);
  assert.equal(document.querySelectorAll('#test-results .pass').length,app.c.NeuralPrologue.levels['self-attention'].sampleCount);
  assert.equal(document.querySelector('#module-shelf').hasAttribute('open'),true);
  phases.add(effect.userData.fusion.stage);
  if(!root){
   root=effect;assembly=scene.children.find(o=>o.children.some(child=>child.userData.node==='q'));
   const original=resources(assembly);for(const resource of original)resource.addEventListener('dispose',()=>originalDisposals++);
   for(const resource of resources(root))if(!original.has(resource))resource.addEventListener('dispose',()=>temporaryDisposals++);
   assert.equal(root.userData.fusion.count,app.api.getState().nodes.length);
   assert.equal(root.children[0].children[0].children.length,assembly.children.length,'the snapshot contains the player’s actual board');
  }
  assert.equal(assembly.visible,false);const packed=effect.children.find(o=>o.userData.node==='fusion-preview');assert.ok(packed);
  if(effect.userData.fusion.phase>.93){
   const screen=packed.position.clone().project(app.camera()),x=(screen.x+1)*700,y=(1-screen.y)*380;
   assert.ok(Math.abs(x-76)<2&&Math.abs(y-180)<2,'the new component reaches the real shelf tile');landed=true;
   assert.ok(shelf.classList.contains('fusion-arrival'));assert.match(document.querySelector('#fusion-stage').textContent,/可拖出使用/);
  }
 }});
 const doc=app.document;doc.querySelector('#welcome').close();app.api.switchLevel('self-attention');app.api.setState(app.c.NeuralPrologue.solutions('self-attention'));app.api.getWorld().rebuild();
 doc.querySelector('#playback-speed').value='3';doc.querySelector('#playback-speed').dispatchEvent(new app.window.Event('input'));
 doc.querySelector('#module-shelf').removeAttribute('open');const saved=JSON.stringify(app.api.getState());await app.api.run();
 assert.deepEqual([...phases],['gather','formed','shelf']);assert.ok(landed);assert.equal(root.parent,null);assert.equal(assembly.visible,true);
 assert.equal(originalDisposals,0,'the celebration must never dispose live board resources');assert.ok(temporaryDisposals>0,'temporary celebration resources are released');
 assert.equal(doc.querySelector('#component-fusion').hidden,true);assert.equal(doc.body.classList.contains('synthesizing-component'),false);assert.equal(doc.querySelectorAll('.fusion-awaiting,.fusion-arrival').length,0);
 assert.equal(doc.querySelector('#level-complete').open,true);assert.equal(JSON.stringify(app.api.getState()),saved);
});
test('fusion supports pause, camera movement, language changes and skipping; stop removes every transient effect',async()=>{
 let app,phase=0,pausedAt,pausedFrames=0,stopping=false,stopped,skipped=false,root;
 app=await boot({reduced:false,frameMs:120,onFrame:({document,window})=>{
  if(!app)return;const scene=app.scenes.at(-1),effect=scene.children.find(o=>o.userData.fusion);if(!effect)return;root=effect;
  if(stopping){app.api.togglePause();stopped=app.api.run();app.api.switchLevel('weighted-sum');return;}
  if(!phase&&effect.userData.fusion.phase>.1){pausedAt=effect.userData.fusion.phase;app.api.togglePause();phase=1;return;}
  if(phase===1){
   if(!pausedFrames)pausedAt=effect.userData.fusion.phase;else assert.equal(effect.userData.fusion.phase,pausedAt,'pause freezes the synthesis');
   if(!pausedFrames){
    const before=app.view().position.slice();const event=(type,x,y)=>Object.assign(new window.Event(type,{bubbles:true}),{clientX:x,clientY:y});
    document.querySelector('#scene').dispatchEvent(event('pointerdown',600,300));document.dispatchEvent(event('pointermove',675,325));document.dispatchEvent(event('pointerup',675,325));assert.notDeepEqual(app.view().position,before);
    app.c.I18n.setPreference('en');assert.match(document.querySelector('#fusion-name').textContent,/Single-head self-attention/);assert.equal(effect.parent,scene,'relocalizing the original board does not dispose its celebration snapshot');
   }
   if(++pausedFrames===3){app.api.togglePause();phase=2;}return;
  }
  if(phase===2&&effect.userData.fusion.phase>.7){document.querySelector('#fusion-skip').onclick();skipped=true;phase=3;}
 }});
 const doc=app.document;doc.querySelector('#welcome').close();app.api.switchLevel('self-attention');app.api.setState(app.c.NeuralPrologue.solutions('self-attention'));app.api.getWorld().rebuild();
 doc.querySelector('#playback-speed').value='3';doc.querySelector('#playback-speed').dispatchEvent(new app.window.Event('input'));
 const saved=JSON.stringify(app.api.getState());await app.api.run();assert.ok(skipped);assert.equal(doc.querySelector('#level-complete').open,true);assert.equal(root.parent,null);assert.equal(JSON.stringify(app.api.getState()),saved);
 stopping=true;await app.api.run();await stopped;
 assert.equal(app.api.getState().levelId,'weighted-sum');assert.equal(doc.querySelector('#component-fusion').hidden,true);assert.equal(root.parent,null);assert.equal(doc.body.classList.contains('synthesizing-component'),false);
 assert.equal(doc.querySelector('#level-complete').open,false);assert.equal(doc.querySelector('#pause').getAttribute('aria-pressed'),'false');assert.equal(doc.querySelectorAll('.fusion-awaiting,.fusion-arrival,.packet').length,0);
 assert.ok(app.scenes.at(-1).children.filter(o=>o.children.some(n=>n.userData.node)).every(o=>o.visible));assert.deepEqual([...app.c.I18n.missing],[]);
});
for(const [mode,hash,screen] of [['running','#home?lesson=projection','home'],['paused','#menu?lesson=projection&track=main&group=workbench&from=game','levels']]){
 test(`browser history cancels a ${mode} run without replacing the current board or publishing stale results`,{timeout:15000},async()=>{
  let app,cancelled=false,pauseStarted=false,resultsAfterNavigation;
  app=await boot({url:'https://example.test/games/neural-atelier/?preview=1'+hash,reduced:false,frameMs:80,onFrame:({document})=>{
   if(!app||cancelled||!document.querySelector('.projection-calculation:not([hidden])'))return;
   if(mode==='paused'&&!pauseStarted){pauseStarted=true;app.api.togglePause();return;}
   cancelled=true;if(mode==='paused')assert.equal(document.querySelector('#pause').getAttribute('aria-pressed'),'true');
   app.c.history.back();
   assert.equal(app.api.readNavigation(app.c.location.hash).screen,screen);
   assert.equal(document.body.classList.contains('network-running'),false);assert.equal(document.querySelector('#pause').getAttribute('aria-pressed'),'false');
   assert.equal(document.querySelectorAll('.packet,.projection-calculation,.projection-result').length,0);
   resultsAfterNavigation=document.querySelector('#test-results').innerHTML;
  }});
  const doc=app.document;app.api.setState(app.c.NeuralPrologue.solutions('projection'));app.api.getWorld().rebuild();app.api.refresh();
  const state=app.api.getState(),saved=JSON.stringify(state);
  app.api.navigate(app.api.readNavigation('#projection'),{source:'user'});const historyLength=app.c.history.length;
  await app.api.run();assert.ok(cancelled);assert.equal(app.api.getState(),state);assert.equal(JSON.stringify(state),saved);
  assert.equal(doc.querySelector('#test-results').innerHTML,resultsAfterNavigation);assert.equal(doc.querySelector('#run-report').hidden,true);assert.ok(!doc.querySelector('#level-complete').open);assert.equal(app.api.getFault(),null);assert.equal(app.api.isCompleted('projection'),false);
  app.c.history.forward();assert.equal(app.c.location.hash,'#projection');assert.equal(app.c.history.length,historyLength,'paired history events do not append entries');
  assert.equal(app.api.getState(),state);assert.equal(JSON.stringify(state),saved);assert.equal(doc.querySelector('#test-results').innerHTML,resultsAfterNavigation);assert.equal(doc.querySelector('#run-report').hidden,true);assert.ok(!doc.querySelector('#level-complete').open);
  assert.equal(doc.querySelector('#pause').disabled,true);assert.equal(doc.querySelectorAll('.packet,.projection-calculation,.projection-result').length,0);
 });
}
test('browser history during fusion retains the saved completion and never overlays the restored lesson',async()=>{
 let app,cancelled=false,root,resultsAfterNavigation;
 app=await boot({url:'https://example.test/games/neural-atelier/#wire',reduced:false,frameMs:120,onFrame:({document})=>{
  if(!app||cancelled)return;const effect=app.scenes.at(-1).children.find(o=>o.userData.fusion);if(!effect)return;
  cancelled=true;root=effect;assert.equal(app.api.isCompleted('self-attention'),true,'completion is saved before the celebration');
  app.c.history.back();
  assert.equal(app.api.getState().levelId,'wire');assert.equal(root.parent,null);assert.equal(document.querySelector('#component-fusion').hidden,true);
  resultsAfterNavigation=document.querySelector('#test-results').innerHTML;
 }});
 const doc=app.document;app.api.setState(app.c.NeuralPrologue.solutions('wire'));app.api.getWorld().rebuild();const wire=JSON.stringify(app.api.getState());
 app.api.switchLevel('self-attention');app.api.setState(app.c.NeuralPrologue.solutions('self-attention'));app.api.getWorld().rebuild();const solved=JSON.stringify(app.api.getState());
 doc.querySelector('#playback-speed').value='3';doc.querySelector('#playback-speed').dispatchEvent(new app.window.Event('input'));
 await app.api.run();assert.ok(cancelled);assert.equal(JSON.stringify(app.api.getState()),wire);assert.equal(app.api.isCompleted('self-attention'),true);
 assert.equal(doc.querySelector('#test-results').innerHTML,resultsAfterNavigation);assert.equal(doc.querySelectorAll('#test-results .pass').length,0);assert.equal(doc.querySelector('#run-report').hidden,true);assert.ok(!doc.querySelector('#level-complete').open);
 assert.equal(doc.body.classList.contains('network-running'),false);assert.equal(doc.body.classList.contains('synthesizing-component'),false);assert.equal(doc.querySelector('#pause').getAttribute('aria-pressed'),'false');
 assert.equal(doc.querySelectorAll('.fusion-awaiting,.fusion-arrival,.packet,.projection-calculation').length,0);assert.equal(root.parent,null);
 app.c.history.forward();assert.equal(app.c.location.hash,'#self-attention');assert.equal(JSON.stringify(app.api.getState()),solved);assert.equal(app.api.isCompleted('self-attention'),true);
 assert.ok(!doc.querySelector('#level-complete').open);assert.equal(doc.querySelector('#component-fusion').hidden,true);assert.equal(app.api.getFault(),null);
 assert.ok(app.scenes.at(-1).children.filter(o=>o.children.some(n=>n.userData.node)).every(o=>o.visible));
});
test('A × V animates actual coefficients, V columns and stored sums at the projection pace',async()=>{
 let app,level='weighted-sum';const seen=new Set(),starts=new Map(),checks=new Set();
 const factor=v=>v<0?'('+Number(v.toFixed(2))+')':String(Number(v.toFixed(2)));
 app=await boot({reduced:false,frameMs:100,onFrame:({now,document})=>{
  if(!app)return;const M=app.c.NeuralPrologue,sample=M.samplesFor(level).find(s=>s.word===document.querySelector('#sample-name').textContent);if(!sample)return;
  const state=app.api.getState(),ev=M.evaluate(state,sample.v),scene=app.scenes.at(-1);
  scene.traverseVisible(unit=>{
   const data=unit.userData.aggregation;if(!data)return;
   const id=unit.userData.node,n=state.nodes.find(n=>n.id===id),width=ev.values[id].length/3,a=ev.values[n.links[0]],v=ev.values[n.links[1]];
   assert.deepEqual([...data.attention],[...a]);assert.deepEqual([...data.values],[...v]);assert.deepEqual([...data.output],[...ev.values[id]]);
   assert.equal(data.queryWord,sample.word.split(' / ')[data.query]);
   if(level==='cross-attention'){assert.deepEqual([...data.valueWords],Array.from(M.levels[level].secondaryTokenIds,i=>M.math.vocab[i]));checks.add('distinct source words');}
   for(const cell of unit.children.filter(o=>o.userData.aggregationValue)){
    const {row,column,value}=cell.userData.aggregationValue;assert.equal(value,v[row*width+column]);assert.ok(cell.material.opacity>.2&&cell.material.opacity<=1,'V remains visible even when its column is inactive');
    if(unit.userData.calculationHighlight>.999){if(column===unit.userData.activeColumn)assert.ok(cell.material.opacity>.999,'the current V column is clear');else assert.ok(cell.material.opacity<.35,'other V columns recede');}
   }
   if(unit.children[0].material.opacity<.13){assert.equal(document.querySelector('.node-label[data-node="'+id+'"]').hidden,true);checks.add('faded scaffold');}
   const formula=document.querySelector('.aggregation-calculation[data-node="'+id+'"]');
   if(!formula.hidden){
    const query=Number(formula.dataset.projectionRow),column=Number(formula.dataset.column),key=[level,sample.word,query,column].join('|');seen.add(key);if(!starts.has(key)&&unit.userData.calculationHighlight>.2)starts.set(key,now);
    assert.equal(formula.querySelector('.projection-equation>b').textContent,'O · '+sample.word.split(' / ')[query]+' · ');
    assert.deepEqual([...formula.querySelectorAll('.projection-equation>span')].map(el=>el.textContent),[0,1,2].map(i=>factor(a[query*3+i])+' × '+factor(v[i*width+column])));
   }
   for(const spark of unit.children.filter(o=>o.userData.contribution&&o.visible&&o.material.opacity>.05)){
    const d=spark.userData;assert.equal(d.inputValue,a[data.query*3+d.input]);assert.equal(d.weight,v[d.input*width+d.output]);assert.equal(d.value,d.inputValue*d.weight);assert.ok(Math.abs(d.value)>1e-9);assert.equal(d.output,unit.userData.activeColumn);
    checks.add('products');if(d.value<0)checks.add('negative product');
   }
   for(const cue of unit.children.filter(o=>o.userData.projectionColumn!=null&&o.visible))assert.equal(cue.userData.projectionColumn,unit.userData.activeColumn);
   const incoming=scene.children.find(o=>o.userData.processingSecondary===id);
   if(incoming?.visible){checks.add('V enters its cells');incoming.children.filter(o=>o.userData.aggregationInput).forEach(bar=>assert.equal(bar.userData.value,v[bar.userData.source*width+bar.userData.column]));}
   const out=scene.children.find(o=>o.userData.processingOutput===id);
   if(out?.visible){
    for(const bar of out.children.filter(o=>o.userData.feature!=null&&o.visible&&o.material.opacity>.01)){assert.equal(bar.userData.value,ev.values[id][bar.userData.feature]);if(bar.userData.transfer===1&&out.userData.gather===0)checks.add('sum waits in bank');}
    if(out.userData.gather>0){assert.equal(out.children.filter(o=>o.userData.feature!=null&&o.visible).length,width*3);checks.add('whole output gathers');}
   }
  });
 }});
 const M=app.c.NeuralPrologue;
 for(level of ['weighted-sum','aggregation','cross-attention']){
  app.api.switchLevel(level);app.api.setState(M.solutions(level));app.api.getWorld().rebuild();const saved=JSON.stringify(app.api.getState());await app.api.run();
  assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);assert.equal(JSON.stringify(app.api.getState()),saved);
  const width=level==='cross-attention'?4:2;
  for(const sample of M.samplesFor(level))for(let query=0;query<3;query++)for(let column=0;column<width;column++){
   const key=[level,sample.word,query,column].join('|');assert.ok(seen.has(key),key);
   if(column)assert.ok(Math.abs(starts.get(key)-starts.get([level,sample.word,query,column-1].join('|'))-1040)<=110,'each output feature keeps the same 0.4-step calculation time');
  }
  assert.equal(app.document.querySelectorAll('.packet,.projection-calculation,.projection-result').length,0);
  app.scenes.at(-1).traverse(o=>{assert.equal(o.userData.aggregation,undefined);if(o.userData.projectionBank!=null)assert.equal(o.visible,false);});
 }
 for(const key of ['distinct source words','faded scaffold','products','negative product','V enters its cells','sum waits in bank','whole output gathers'])assert.ok(checks.has(key),key);
});

test('A × V handles zero and signed contributions in both widths and replaces the live matrix on the next sample',async()=>{
 const app=await boot(),M=app.c.NeuralPrologue,T=app.c.T,scene=new T.Scene(),art=app.c.createSceneArt(T,scene,{aggregate:0x64a68f,'original-aggregate':0x64a68f});
 const a=[1,0,0,.25,.75,0,.2,.3,.5],words=['我','爱','猫'],valueWords=['猫','我','爱'];
 for(const kind of ['aggregate','original-aggregate']){
  const d=M.definitions[kind],width=d.out/3,unit=art.module(scene,{id:kind,x:0,z:0,links:['a','v']},d,M.weights),rowDuration=art.projectionFrame(0,width).duration;
  for(const multiplier of [1,-2]){
   const v=Array.from({length:3},(_,i)=>Array.from({length:width},(_,j)=>i===j?0:((i+1)*(j+1)-3)*multiplier)).flat(),output=M.operations[kind]([a,v]);
   for(let query=0;query<3;query++)for(let column=0;column<width;column++){
    art.processing(kind,query*rowDuration+.53+column*.4,a,output,0x64a68f,{other:v,words,valueWords});art.focusCalculations();
    assert.equal(unit.userData.aggregation.query,query);assert.equal(unit.userData.activeColumn,column);
    const particles=unit.children.filter(o=>o.userData.contribution&&o.visible);
    assert.deepEqual(particles.map(o=>o.userData.input),[0,1,2].filter(i=>Math.abs(a[query*3+i]*v[i*width+column])>1e-9));
    assert.ok(Math.abs(particles.reduce((sum,o)=>sum+o.userData.value,0)-output[query*width+column])<1e-9,'visible contributions sum to this output');
    for(const cell of unit.children.filter(o=>o.userData.aggregationValue)){const {row,column,value}=cell.userData.aggregationValue;assert.equal(value,v[row*width+column]);}
   }
   art.clearProcessing();art.focusCalculations();assert.equal(unit.userData.aggregation,undefined);
   unit.traverse(o=>{if(o.userData.aggregationValue)assert.equal(o.userData.aggregationValue.value,undefined);if(o.userData.contribution)assert.equal(o.visible,false);});
  }
 }
});

test('an extra A × V module uses its own two wires instead of a reference V projection',async()=>{
 let app;const seen=new Set();
 app=await boot({reduced:false,frameMs:180,onFrame:({document})=>{
  if(!app)return;const M=app.c.NeuralPrologue,sample=M.samplesFor('weighted-sum').find(s=>s.word===document.querySelector('#sample-name').textContent);if(!sample)return;
  const ev=M.evaluate(app.api.getState(),sample.v);
  app.scenes.at(-1).traverseVisible(o=>{if(o.userData.node!=='aggregate-2'||!o.userData.aggregation)return;const data=o.userData.aggregation;assert.deepEqual([...data.values],[...ev.values.aggregate]);assert.notDeepEqual([...data.values],[...ev.values.v]);assert.deepEqual([...data.output],[...ev.values['aggregate-2']]);seen.add(data.query);});
 }});
 const M=app.c.NeuralPrologue;app.api.switchLevel('weighted-sum');const state=M.initial('weighted-sum'),b=M.bounds(M.levels['weighted-sum']);
 for(const [kind,x,z]of [['original-weights',b.left,b.back],['v',b.left,b.front],['aggregate',b.right,b.back],['aggregate',b.right,b.front]])assert.ok(M.addNode(state,kind,x,z),kind);
 M.connect(state,'R','original-weights',0);M.connect(state,'R','v',0);M.connect(state,'original-weights','aggregate',0);M.connect(state,'v','aggregate',1);M.connect(state,'original-weights','aggregate-2',0);M.connect(state,'aggregate','aggregate-2',1);M.connect(state,'aggregate-2','sink',0);
 app.api.setState(state);app.api.getWorld().rebuild();const saved=JSON.stringify(state);await app.api.run();assert.equal(seen.size,3);assert.equal(JSON.stringify(state),saved);
});

test('A × V pauses, rescales and relocalizes without losing arithmetic, then cleans up on stop',async()=>{
 let app,phase=0,frames=0,stop,before;
 const unit=()=>app.scenes.at(-1).children.flatMap(g=>g.children).find(g=>g.userData.node==='aggregate');
 app=await boot({reduced:false,frameMs:100,onFrame:({document})=>{
  if(!app)return;const formula=document.querySelector('.aggregation-calculation:not([hidden])');
  if(!phase&&formula?.dataset.projectionRow==='1'&&formula.dataset.column==='1'){app.api.togglePause();before=JSON.stringify(unit().userData.aggregation);phase=1;}
  else if(phase===1&&++frames===3){
   assert.equal(JSON.stringify(unit().userData.aggregation),before);const scale=Number(formula.style.getPropertyValue('--calculation-scale')),wheel=new app.window.Event('wheel',{bubbles:true,cancelable:true});wheel.deltaY=2000;document.querySelector('#scene').dispatchEvent(wheel);
   assert.ok(Number(formula.style.getPropertyValue('--calculation-scale'))<scale);assert.equal(formula.dataset.detail,'abstract');app.c.I18n.setPreference('en');assert.match(formula.textContent,/O · love ·/);assert.equal(unit().userData.aggregation.queryWord,'love');assert.ok(unit().children[0].material.opacity<.13);
   const speed=document.querySelector('#playback-speed');speed.value='1';speed.dispatchEvent(new app.window.Event('input'));assert.equal(formula.dataset.projectionRow,'1');app.api.togglePause();phase=2;
  }else if(phase===2&&formula?.dataset.projectionRow==='2'){stop=app.api.run();phase=3;}
 }});
 app.document.querySelector('#welcome').close();app.api.switchLevel('weighted-sum');app.api.setState(app.c.NeuralPrologue.solutions('weighted-sum'));app.api.getWorld().rebuild();const saved=JSON.stringify(app.api.getState());await app.api.run();await stop;
 assert.equal(phase,3);assert.equal(JSON.stringify(app.api.getState()),saved);assert.equal(app.document.querySelectorAll('.packet,.projection-calculation,.projection-result').length,0);assert.equal(unit().userData.aggregation,undefined);assert.equal(unit().children[0].material.opacity,1);assert.equal(app.document.querySelector('.node-label[data-node="aggregate"]').hidden,true,'stopping keeps titles hidden at the distant zoom');
 app.document.querySelector('#camera-reset').onclick();assert.equal(app.document.querySelector('.node-label[data-node="aggregate"]').hidden,false,'resetting the camera restores the title');
 app.api.switchLevel('wire');assert.equal(app.document.querySelector('.aggregation-calculation'),null);assert.deepEqual([...app.c.I18n.missing],[]);
});

test('calculation focus fades the projection scaffold for every token while keeping matrices, highlights and contributions readable',async()=>{
 let app;const seen=new Set(),sparks=new Set();
 const units=()=>app.scenes.at(-1).children.flatMap(g=>g.children).filter(g=>['q','k','v'].includes(g.userData.node));
 app=await boot({reduced:false,frameMs:120,onFrame:({document})=>{
  if(!app)return;
  for(const unit of units()){
   const data=unit.userData.sequenceProjection;if(!data||unit.children[0].material.opacity>.13)continue;
   const id=unit.userData.node;seen.add(id+'/'+data.row);
   assert.equal(document.querySelector('.node-label[data-node="'+id+'"]').hidden,true,'the name stays out of the running calculation');
   for(const base of unit.children.slice(0,2)){assert.ok(base.material.opacity<=.13);assert.equal(base.material.depthWrite,false);assert.equal(base.castShadow,false);}
   const faces=unit.children.filter(o=>o.userData.detailText);assert.equal(faces.length,4);
   faces.forEach(face=>{
    assert.ok(face.material.opacity>.3&&face.material.opacity<=1,'inactive numbers remain visible');
    if(face.userData.matrixColumn!==unit.userData.activeColumn)assert.ok(face.material.opacity<.35,'only the current column is emphasized');
    else if(unit.userData.calculationHighlight>.999)assert.ok(face.material.opacity>.999,'the multiplying column is fully clear');
   });
   unit.children.filter(o=>o.geometry?.type==='TubeGeometry').forEach(path=>assert.ok(path.material.opacity<.081,'the static weight web recedes'));
   const active=unit.children.find(o=>o.userData.projectionColumn===unit.userData.activeColumn);if(active)assert.ok(active.visible,'column emphasis remains visible');
   if(unit.children.some(o=>o.userData.contribution&&o.visible&&o.material.opacity>.4))sparks.add(id);
   app.scenes.at(-1).traverse(o=>{const edge=o.userData.wire;if(edge&&(edge.from===id||edge.to===id))assert.ok(o.material.opacity<.15,'nearby permanent wires recede without changing moving packets');});
  }
 }});
 const M=app.c.NeuralPrologue;app.api.switchLevel('qkv');app.api.setState(M.solutions('qkv'));app.api.getWorld().rebuild();const saved=JSON.stringify(app.api.getState());await app.api.run();
 assert.equal(seen.size,9);assert.deepEqual([...sparks].sort(),['k','q','v']);assert.equal(JSON.stringify(app.api.getState()),saved);
 for(const unit of units()){
  assert.equal(app.document.querySelector('.node-label[data-node="'+unit.userData.node+'"]').hidden,false);
  for(const base of unit.children.slice(0,2)){assert.equal(base.material.opacity,1);assert.equal(base.material.transparent,false);assert.equal(base.material.depthWrite,true);assert.equal(base.castShadow,true);}
 }
 app.scenes.at(-1).traverse(o=>{if(o.userData.wire){assert.equal(o.material.opacity,1);assert.equal(o.material.transparent,false);assert.equal(o.material.depthWrite,true);}});
 assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);
});

test('the same focus treatment restores all module families exactly and does not accumulate opacity loss',async()=>{
 const app=await boot(),M=app.c.NeuralPrologue,T=app.c.T,scene=new T.Scene(),colors=Object.fromEntries(Object.keys(M.definitions).map(id=>[id,0x6e9e90])),art=app.c.createSceneArt(T,scene,colors);
 for(const kind of ['project','gate','silu','multiply','add','norm','layernorm2','wide-embedding','position','score','mask','softmax','aggregate','ropeq','self-heads']){
  const levelId=M.exampleLevel(kind,'wire'),level=M.levels[levelId],d=M.definitions[kind],ex=M.extraExample(kind,levelId)||app.c.exampleFor(M,kind,levelId);
  const unit=art.module(scene,{id:kind,x:0,z:0,links:d.ins.map(()=>null)},d,M.weights);
  const snapshot=()=>{const values=[];unit.traverse(o=>{if(o.userData.computationBackdrop)values.push([o.material.opacity,o.material.transparent,o.material.depthWrite,o.castShadow,o.receiveShadow]);});return values;};
  const idle=snapshot(),options={sequence:!!level.sequence,detailed:d.type==='linear',paired:!level.sequence&&['multiply','add'].includes(d.type),other:ex.inputs[1]?.[1],words:M.samplesFor(levelId)[ex.sampleIndex||0].word.split(' / ')};
  art.processing(kind,.55,ex.inputs[0][1],ex.output[1],colors[kind],options);assert.ok(art.focusCalculations().has(kind),kind);
  assert.ok(unit.children[0].material.opacity<.13,kind+' base fades');const focused=snapshot();
  for(let i=0;i<4;i++){art.focusCalculations();assert.deepEqual(snapshot(),focused,kind+' camera render must not compound alpha');}
  if(d.type==='linear'||d.type==='mix')unit.children.filter(o=>o.userData.matrixColumn!=null).forEach(cell=>{
   if(cell.userData.matrixColumn===unit.userData.activeColumn)assert.equal(cell.material.opacity,1);
   else{assert.ok(cell.material.opacity>.2&&cell.material.opacity<.35);assert.equal(cell.material.depthWrite,false);assert.equal(cell.castShadow,false);}
  });
  if(d.type==='activation')unit.children.filter(o=>o.geometry?.type==='TubeGeometry').forEach(curve=>assert.equal(curve.material.opacity,1,'the activation curve explains the operation'));
  art.processing(kind,.55,ex.inputs[0][1],ex.output[1],colors[kind],options);art.focusCalculations();assert.deepEqual(snapshot(),focused,kind+' repeated animation time preserves focus');
  art.clearProcessing();assert.equal(art.focusCalculations().size,0);assert.deepEqual(snapshot(),idle,kind+' stop restores materials and shadows');
 }
});

test('paused calculation focus survives camera and language changes, then clears immediately on stop and level change',async()=>{
 let app,phase='waiting',frames=0,stop;
 const unit=()=>app.scenes.at(-1).children.flatMap(g=>g.children).find(g=>g.userData.node==='project');
 const event=(target,type,props={})=>{const e=new app.window.Event(type,{bubbles:true,cancelable:true});Object.assign(e,props);target.dispatchEvent(e);};
 app=await boot({reduced:false,frameMs:80,onFrame:({document})=>{
  if(!app||!unit())return;
  if(phase==='waiting'&&unit().children[0].material.opacity<.13){app.api.togglePause();phase='paused';return;}
  if(phase!=='paused')return;
  assert.ok(unit().children[0].material.opacity<.13);assert.equal(document.querySelector('.node-label[data-node="project"]').hidden,true);
  if(++frames===2){event(document.querySelector('#scene'),'wheel',{deltaY:-100});app.c.I18n.setPreference('en');assert.ok(unit().children[0].material.opacity<.13);assert.equal(document.querySelector('.node-label[data-node="project"]').hidden,true);}
  if(frames===5){phase='stopped';stop=app.api.run();assert.equal(unit().children[0].material.opacity,1);assert.equal(document.querySelector('.node-label[data-node="project"]').hidden,false);}
 }});
 app.api.switchLevel('projection');app.api.setState(app.c.NeuralPrologue.solutions('projection'));app.api.getWorld().rebuild();const saved=JSON.stringify(app.api.getState());await app.api.run();await stop;
 assert.equal(phase,'stopped');assert.equal(JSON.stringify(app.api.getState()),saved);app.api.switchLevel('scores');assert.equal(app.document.querySelector('.node-label[hidden]'),null);assert.equal(app.document.querySelectorAll('.processing-packet').length,0);
});

test('QKV projections animate every token and weight column at the gate-projection pace, then gather the actual results',async()=>{
 let app;const starts=new Map(),seen=new Set(),paths=new Map();
 app=await boot({reduced:false,frameMs:100,onFrame:({now,document})=>{
  if(!app)return;const M=app.c.NeuralPrologue,sample=M.samplesFor('qkv').find(s=>s.word===document.querySelector('#sample-name').textContent);if(!sample)return;
  for(const el of document.querySelectorAll('.sequence-projection-calculation:not([hidden])')){const key=[el.dataset.node,sample.word,el.dataset.projectionRow,el.dataset.column].join('|');if(!starts.has(key))starts.set(key,now);assert.equal(el.querySelector('.projection-equation>b').textContent,sample.word.split(' / ')[Number(el.dataset.projectionRow)]+' · ');}
  app.scenes.at(-1).traverseVisible(o=>{
   const data=o.userData.sequenceProjection;if(data){
    assert.deepEqual([...data.input],[...sample.v]);const active=o.userData.activeColumn;
    for(const cue of o.children.filter(c=>c.userData.projectionColumn!=null&&c.visible))assert.equal(cue.userData.projectionColumn,active);
    seen.add(o.userData.node);for(const spark of o.children.filter(c=>c.userData.contribution&&c.visible)){
     const d=spark.userData;assert.equal(d.inputValue,sample.v[data.row*2+d.input]);assert.equal(d.value,d.inputValue*d.weight);assert.equal(d.output,active);if(d.value<0)seen.add('negative contribution');
    }
   }
   if(o.userData.sequenceProjectionInput&&o.material.opacity>.01){assert.equal(o.userData.mappedInput,o.userData.feature%2);seen.add('features routed');}
   if(['q','k','v'].includes(o.userData.processingOutput)){
    const expected=M.math.mat(sample.v,M.weights[o.userData.processingOutput]);
    for(const bar of o.children.filter(c=>c.userData.feature!=null&&c.visible&&c.material.opacity>.01)){
     assert.equal(bar.userData.value,expected[bar.userData.feature]);assert.equal(bar.userData.projectionToken,Math.floor(bar.userData.feature/2));
     const key=[o.userData.processingOutput,sample.word,bar.userData.feature].join('|');if(!paths.has(key))paths.set(key,[]);paths.get(key).push(bar.getWorldPosition(new app.c.T.Vector3()));
     if(bar.userData.transfer===1&&o.userData.gather===0)seen.add('stored before gathering');
    }
    if(o.userData.gather>0){assert.equal(o.children.filter(c=>c.userData.feature!=null&&c.visible).length,6);seen.add('complete gather');}
   }
  });
 }});
 const M=app.c.NeuralPrologue;app.api.switchLevel('qkv');app.api.setState(M.solutions('qkv'));app.api.getWorld().rebuild();await app.api.run();
 assert.equal(starts.size,54);for(const node of ['q','k','v'])for(const sample of M.samplesFor('qkv'))for(let row=0;row<3;row++){
  const first=starts.get([node,sample.word,row,0].join('|')),second=starts.get([node,sample.word,row,1].join('|'));assert.ok(Math.abs(second-first-1040)<=200,'all output columns keep the default gate-projection pace');
 }
 for(const key of ['q','k','v','negative contribution','features routed','stored before gathering','complete gather'])assert.ok(seen.has(key),key);
 for(const points of paths.values())assert.ok(points.every((p,i)=>!i||p.distanceTo(points[i-1])<.8),'results move continuously from the neuron to their token bank and outgoing tray');
 assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);assert.equal(app.document.querySelectorAll('.projection-calculation,.projection-result,.packet').length,0);
 app.scenes.at(-1).traverse(o=>{assert.equal(o.userData.sequenceProjection,undefined);if(o.userData.projectionBank!=null)assert.equal(o.visible,false);});
});
test('sequence projection uses actual upstream features for extra copies and four-feature source/target streams',async()=>{
 let app,id='self-attention';const seen=new Set();
 app=await boot({reduced:false,frameMs:160,onFrame:({document})=>{
  if(!app)return;const M=app.c.NeuralPrologue,sample=M.samplesFor(id).find(s=>s.word===document.querySelector('#sample-name').textContent);if(!sample)return;
  const state=app.api.getState(),ev=M.evaluate(state,sample.v);
  app.scenes.at(-1).traverseVisible(o=>{const p=o.userData.sequenceProjection;if(!p)return;const n=state.nodes.find(n=>n.id===o.userData.node);assert.deepEqual([...p.input],[...ev.values[n.links[0]]]);seen.add(n.id);
   if(n.id==='q-2'){assert.notDeepEqual([...p.input],[...sample.v]);seen.add('upstream projection');}
   if(id==='cross-attention'&&['original-k','original-v'].includes(n.kind)){assert.equal(p.input.length,12);assert.equal(p.word,M.math.vocab[M.levels[id].secondaryTokenIds[p.row]]);seen.add('source token');}
  });
 }});
 const M=app.c.NeuralPrologue;app.api.switchLevel(id);const state=M.initial(id);assert.ok(M.addNode(state,'q',-2,0));assert.ok(M.addNode(state,'q',2,0));M.connect(state,'R','q',0);M.connect(state,'q','q-2',0);M.connect(state,'q-2','sink',0);app.api.setState(state);app.api.getWorld().rebuild();await app.api.run();
 id='cross-attention';app.api.switchLevel(id);app.api.setState(M.solutions(id));app.api.getWorld().rebuild();await app.api.run();assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);
 for(const key of ['q','q-2','upstream projection','original-q','original-k','original-v','source token'])assert.ok(seen.has(key),key);
});
test('paused QKV calculation keeps its token and arithmetic while zooming, switching language, changing speed and stopping',async()=>{
 let app,phase=0,frames=0,stop,before;
 app=await boot({reduced:false,frameMs:100,onFrame:({document})=>{
  if(!app)return;const formula=document.querySelector('.sequence-projection-calculation[data-node="q"]:not([hidden])');
  if(!phase&&formula?.dataset.projectionRow==='1'&&formula.dataset.column==='1'){app.api.togglePause();before=formula.textContent;phase=1;}
  else if(phase===1&&++frames===3){
   assert.equal(formula.textContent,before);const scale=Number(formula.style.getPropertyValue('--calculation-scale')),wheel=new app.window.Event('wheel',{bubbles:true,cancelable:true});wheel.deltaY=2000;document.querySelector('#scene').dispatchEvent(wheel);
   assert.ok(Number(formula.style.getPropertyValue('--calculation-scale'))<scale);assert.equal(formula.dataset.detail,'abstract');app.c.I18n.setPreference('en');assert.match(formula.textContent,/love ·/);
   const speed=document.querySelector('#playback-speed');speed.value='1';speed.dispatchEvent(new app.window.Event('input'));assert.equal(formula.dataset.projectionRow,'1');app.api.togglePause();phase=2;
  }else if(phase===2&&formula?.dataset.projectionRow==='2'){stop=app.api.run();phase=3;}
 }});
 app.document.querySelector('#welcome').close();app.api.switchLevel('qkv');app.api.setState(app.c.NeuralPrologue.solutions('qkv'));app.api.getWorld().rebuild();await app.api.run();await stop;
 assert.equal(phase,3);assert.equal(app.document.querySelectorAll('.packet,.projection-calculation,.projection-result').length,0);assert.ok(!app.document.querySelector('#level-complete').open);
 app.scenes.at(-1).traverse(o=>{assert.equal(o.userData.sequenceProjection,undefined);if(o.userData.projectionBank!=null)assert.equal(o.visible,false);});assert.deepEqual([...app.c.I18n.missing],[]);
});
test('V-only animation passes without Q/K and an actual outlet error remains dismissible after running',async()=>{
 let app;app=await boot({reduced:false,frameMs:250,onFrame:({document})=>{if(app&&document.body.classList.contains('network-running')){assert.equal(app.api.getFault(),null);assert.equal(document.querySelector('.local-fault'),null);}}});
 app.api.switchLevel('qkv');const M=app.c.NeuralPrologue,state=M.initial('qkv');assert.ok(M.addNode(state,'v',0,0));M.connect(state,'R','v',0);M.connect(state,'v','sink',0);
 app.api.setState(state);app.api.getWorld().rebuild();await app.api.run();assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);assert.equal(app.api.getFault(),null);assert.equal(app.document.querySelector('#level-complete').open,true);
 state.sink='R';app.api.getWorld().rebuild();await app.api.run();assert.equal(app.api.getFault().id,'sink');assert.ok(app.document.querySelector('.local-fault[data-owner="sink"]'));
 app.c.I18n.setPreference('en');assert.match(app.document.querySelector('.local-fault strong').textContent,/Final result/);
 const before=JSON.stringify(app.api.getState());app.document.querySelector('.fault-close').dispatchEvent(new app.window.Event('click',{bubbles:true}));assert.equal(app.document.querySelector('.local-fault'),null);assert.equal(app.api.getFault(),null);assert.equal(JSON.stringify(app.api.getState()),before);
});
test('four-feature readout lessons animate their actual feature and score widths and inspect the shared head',async()=>{
 let app;const widths=new Set();
 app=await boot({reduced:false,frameMs:1000,onFrame:()=>{
  if(!app)return;
  for(const o of app.scenes.at(-1).children){
   if(!o.visible||!o.userData.packet)continue;const bars=o.children.filter(c=>c.userData.feature!=null);
   if(bars.length)widths.add(bars.length/(o.userData.row==='sequence'?3:1));
  }
 }});
 const M=app.c.NeuralPrologue,doc=app.document;doc.querySelector('#welcome').close();
 for(const id of ['vocabulary-scores','vocabulary']){
  app.api.switchLevel(id);app.api.setState(M.solutions(id));app.api.getWorld().rebuild();const saved=JSON.stringify(app.api.getState());
  await app.api.run();assert.equal(doc.querySelectorAll('#test-results .pass').length,3);assert.equal(JSON.stringify(app.api.getState()),saved);
  assert.equal(doc.querySelectorAll('.packet').length,0);doc.querySelector('#complete-close').onclick();
  app.api.getWorld().focusShelf('original-head');assert.match(doc.querySelector('#card-stats').textContent,/4 × 6/);assert.match(doc.querySelector('#card-purpose').textContent,/17.*18.*19/s);doc.querySelector('#card-close').onclick();
 }
 assert.ok(widths.has(4),'decoder features retain all four coordinates');assert.ok(widths.has(6),'projection carries all six candidate scores');
});
test('vocabulary projections show real weights and biases; selection compares every candidate and carries the winning IDs',async()=>{
 let app,level='vocabulary',paused=false,pauseFrames=0,frozen;const columns=new Set(),candidates=new Set(),flags=new Set(),times=new Map();
 app=await boot({reduced:false,frameMs:100,onFrame:({now,document})=>{
  if(!app)return;const M=app.c.NeuralPrologue,state=app.api.getState(),sample=M.samplesFor(level).find(s=>s.word===document.querySelector('#sample-name').textContent);if(!sample)return;
  const ev=M.evaluate(state,sample.v),scene=app.scenes.at(-1);
  scene.traverseVisible(unit=>{
   const id=unit.userData.node,n=state.nodes.find(n=>n.id===id);if(!n)return;
   const kind=M.nodeKind(n),projection=unit.userData.sequenceProjection;
   if(kind==='original-head'&&projection){
    const {row,column}=projection,key=[sample.word,row,column].join('|');columns.add(key);
    assert.deepEqual([...projection.input],[...ev.values[n.links[0]]]);
    assert.equal(unit.children.filter(o=>o.userData.matrixColumn!=null).length,48);
    const formula=document.querySelector('.projection-calculation[data-node="'+id+'"]');
    if(!formula.hidden){
     const bias=formula.querySelector('.projection-bias');assert.equal(Number(bias.dataset.value),M.definitions[kind].bias[column]);
     assert.match(formula.textContent,new RegExp(M.math.vocab[column]));
     if(column===2||column===5){assert.ok(formula.textContent.includes('(-0.75)'));flags.add('bias shown');}
     if(!times.has(key))times.set(key,now);
    }
    if(unit.userData.calculationHighlight>.999){
     for(const cell of unit.children.filter(o=>o.userData.matrixColumn!=null))assert.ok(cell.userData.matrixColumn===column?cell.material.opacity>.999:cell.material.opacity<.35);
     assert.equal(document.querySelector('.node-label[data-node="'+id+'"]').hidden,true);flags.add('matrix focus');
    }
   }
   const selection=unit.userData.selection;if(!selection)return;
   const scores=ev.values[n.links[0]].slice(selection.row*6,(selection.row+1)*6);
   assert.deepEqual([...selection.scores],[...scores]);assert.equal(selection.best,scores.slice(0,selection.column+1).indexOf(Math.max(...scores.slice(0,selection.column+1))));
   if(selection.stage==='compare')candidates.add([level,sample.word,selection.row,selection.column].join('|'));
   for(const carrier of unit.children.filter(o=>o.userData.selectionWinner&&o.visible)){
    const {row,winner}=carrier.userData;assert.equal(winner,ev.values[id][row]);flags.add('winner flies');
   }
   if(level==='vocabulary'&&selection.column===2&&selection.stage==='compare'&&!paused){app.api.togglePause();paused=true;frozen=JSON.stringify(selection);return;}
   if(paused&&pauseFrames<3){assert.equal(JSON.stringify(selection),frozen,'pausing freezes comparisons');if(++pauseFrames===3)app.api.togglePause();}
   const out=scene.children.find(o=>o.userData.processingOutput===id);
   if(out?.visible&&out.userData.gather>0){
    assert.deepEqual(out.children.filter(o=>o.userData.feature!=null).map(o=>o.userData.value),[...ev.values[id]]);flags.add('IDs gather');
   }
  });
 }});
 const M=app.c.NeuralPrologue;
 for(level of ['vocabulary','autoregressive']){
  app.api.switchLevel(level);const state=M.solutions(level);
  if(level==='vocabulary'){const rename=id=>id==='original-head'?'original-head-2':id==='pick'?'pick-2':id;state.nodes.forEach(n=>{n.id=rename(n.id);n.links=n.links.map(rename);});state.sink=rename(state.sink);}
  app.api.setState(state);app.api.getWorld().rebuild();const before=JSON.stringify(state);await app.api.run();assert.equal(JSON.stringify(state),before);
  assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);assert.equal(app.document.querySelectorAll('.selection-calculation,.projection-result,.packet').length,0);
  for(const sample of M.samplesFor(level))for(let row=0;row<3;row++)for(let column=0;column<6;column++)assert.ok(candidates.has([level,sample.word,row,column].join('|')));
  app.scenes.at(-1).traverse(unit=>assert.equal(unit.userData.selection,undefined));
 }
 assert.equal(columns.size,54);assert.equal(pauseFrames,3);assert.deepEqual([...flags].sort(),['IDs gather','bias shown','matrix focus','winner flies']);
 const first=M.samplesFor('vocabulary')[1].word;for(let column=1;column<6;column++){const delta=times.get([first,0,column].join('|'))-times.get([first,0,column-1].join('|'));assert.ok(Math.abs(delta-1040)<=200,'column '+column+' starts after '+delta+' ms');}
 app.c.I18n.setPreference('en');app.api.getWorld().focusShelf('original-head');assert.doesNotMatch(app.document.querySelector('#card-meaning').textContent,/[\u3400-\u9fff]/);app.document.querySelector('#card-close').onclick();
 assert.deepEqual([...app.c.I18n.missing],[]);
});
test('token outlet reveals completed selections one at a time and clears them on stop or restart',async()=>{
 let app,mode='complete',paused=false,pauseFrames=0,stopped=false;const seen=new Set();
 const outlet=()=>{const out=[];app.scenes.at(-1).traverse(o=>{if(o.userData.sinkRow!=null){const text=o.children.filter(c=>c.material?.map).map(c=>c.material.map.image.drawnText);out[o.userData.sinkRow]={word:text.at(-1),status:text.at(-2)};}});return out;};
 app=await boot({reduced:false,frameMs:80,onFrame:({document})=>{
  if(!app)return;const M=app.c.NeuralPrologue,sample=M.samplesFor('vocabulary').find(s=>s.word===document.querySelector('#sample-name').textContent),ev=M.evaluate(app.api.getState(),sample.v);
  let selection;app.scenes.at(-1).traverse(o=>{if(o.userData.node==='pick-2')selection=o.userData.selection;});
  if(!selection||selection.stage!=='compare')return;
  const displayed=outlet();
  if(mode==='disconnected'){assert.deepEqual(displayed.map(r=>r.word),['','','']);seen.add('unwired picker stays private');return;}
  assert.deepEqual(displayed.map(r=>r.word),[0,1,2].map(i=>i<selection.row?M.math.vocab[ev.output[i]]:''),'only earlier completed positions are visible');
  assert.ok(displayed.every(r=>r.status==='·'),'no correctness verdict before the whole trial finishes');
  seen.add(sample.word+'|'+selection.row);
  if(mode==='stop'&&selection.row===1){
   stopped=true;mode='stopped';app.api.run();assert.deepEqual(outlet().map(r=>r.word),['','','']);return;
  }
  if(mode==='complete'&&selection.row===1&&!paused){paused=true;app.api.togglePause();}
  if(paused&&pauseFrames<3){
   if(++pauseFrames===2)app.c.I18n.setPreference('en');
   if(pauseFrames===3){assert.equal(outlet().filter(r=>r.word).length,1);app.c.I18n.setPreference('zh');app.api.togglePause();}
  }
 }});
 const M=app.c.NeuralPrologue,doc=app.document;doc.querySelector('#welcome').close();app.api.switchLevel('vocabulary');const state=M.solutions('vocabulary');
 state.nodes.find(n=>n.id==='pick').id='pick-2';state.sink='pick-2';app.api.setState(state);app.api.getWorld().rebuild();const saved=JSON.stringify(state);
 assert.deepEqual(outlet().map(r=>r.word),['','','']);await app.api.run();assert.equal(JSON.stringify(state),saved);
 for(const sample of M.samplesFor('vocabulary'))for(let i=0;i<3;i++)assert.ok(seen.has(sample.word+'|'+i));
 assert.deepEqual(outlet().map(r=>r.word),Array.from(M.evaluate(state,M.samplesFor('vocabulary').at(-1).v).output,id=>M.math.vocab[id]));assert.ok(outlet().every(r=>r.status==='✓'));assert.equal(pauseFrames,3);
 mode='stop';const run=app.api.run();assert.deepEqual(outlet().map(r=>r.word),['','',''],'a replay clears the previous words immediately');await run;assert.ok(stopped);assert.deepEqual(outlet().map(r=>r.word),['','','']);
 mode='disconnected';state.sink=null;app.api.getWorld().rebuild();await app.api.run();assert.ok(seen.has('unwired picker stays private'));assert.deepEqual(outlet().map(r=>r.word),['','','']);
 assert.deepEqual([...app.c.I18n.missing],[]);
});
test('append outlet retains two old words then reveals the new token before transport ends',async()=>{
 let app,mode='complete',paused=false,pauseFrames=0,stopped=false;const seen=new Set();
 const outlet=()=>{const out=[];app.scenes.at(-1).traverse(o=>{if(o.userData.sinkRow!=null){const text=o.children.filter(c=>c.material?.map).map(c=>c.material.map.image.drawnText);out[o.userData.sinkRow]={word:text.at(-1),status:text.at(-2)};}});return out;};
 app=await boot({reduced:false,frameMs:80,onFrame:({document})=>{
  if(!app||mode==='stopped')return;
  const M=app.c.NeuralPrologue,sample=M.samplesFor('autoregressive').find(s=>s.word===document.querySelector('#sample-name').textContent);if(!sample)return;
  const ev=M.evaluate(app.api.getState(),sample.v),displayed=outlet();
  const process=document.querySelector('.processing-packet[data-node="feedback-2"]:not([hidden])');
  const outbound=document.querySelector('.packet[data-motion="wire"][data-from="feedback-2"][data-to="sink"]:not([hidden])');
  if(mode==='disconnected'){
   if(process){assert.deepEqual(displayed.map(r=>r.word),['','','']);seen.add('unused append stays private');}return;
  }
  if(process?.dataset.phase==='enter'){
   assert.deepEqual(displayed.map(r=>r.word),[...ev.output.slice(0,2).map(i=>M.math.vocab[i]),'']);
   assert.ok(displayed.every(r=>r.status==='·'));seen.add('retained words');
   if(mode==='stop'){stopped=true;mode='stopped';app.api.run();assert.deepEqual(outlet().map(r=>r.word),['','','']);return;}
   if(!paused){paused=true;app.api.togglePause();}
   if(pauseFrames<3){
    if(++pauseFrames===2)app.c.I18n.setPreference('en');
    if(pauseFrames===3){assert.equal(outlet().filter(r=>r.word).length,2);app.c.I18n.setPreference('zh');app.api.togglePause();}
   }
  }
  if(outbound){
   assert.deepEqual(displayed.map(r=>r.word),Array.from(ev.output,i=>M.math.vocab[i]));
   assert.ok(displayed.every(r=>r.status==='·'),'preview must not grade the result');
   assert.equal(outbound.querySelector('.packet-values').textContent,M.readTokens(ev.output));seen.add(mode+' new token in transit');
  }
 }});
 const M=app.c.NeuralPrologue,doc=app.document;doc.querySelector('#welcome').close();app.api.switchLevel('autoregressive');
 const state=M.solutions('autoregressive');state.nodes.find(n=>n.id==='feedback').id='feedback-2';state.sink='feedback-2';
 app.api.setState(state);app.api.getWorld().rebuild();const saved=JSON.stringify(state);await app.api.run();
 assert.equal(JSON.stringify(state),saved);assert.equal(pauseFrames,3);assert.ok(seen.has('retained words'));assert.ok(seen.has('complete new token in transit'));
 assert.equal(doc.querySelectorAll('#test-results .pass').length,3);
 mode='stop';const run=app.api.run();assert.deepEqual(outlet().map(r=>r.word),['','','']);await run;assert.ok(stopped);
 mode='wrong';state.nodes.find(n=>n.id==='feedback-2').links=['R'];app.api.getWorld().rebuild();await app.api.run();
 assert.ok(seen.has('wrong new token in transit'));assert.equal(doc.querySelectorAll('#test-results .fail').length,1);
 assert.deepEqual(outlet().map(r=>r.word),Array.from(M.evaluate(state,M.samplesFor('autoregressive')[0].v).output,i=>M.math.vocab[i]));
 mode='disconnected';state.sink=null;app.api.getWorld().rebuild();await app.api.run();assert.ok(seen.has('unused append stays private'));assert.deepEqual(outlet().map(r=>r.word),['','','']);
 assert.deepEqual([...app.c.I18n.missing],[]);
});
test('moving trays preserve feature columns and sequence rows through projection',async()=>{
 let app;const checked=new Set();
 app=await boot({reduced:false,frameMs:100,onFrame:()=>{
  if(!app)return;
  for(const mesh of app.scenes.at(-1).children){
   if(mesh.userData.routing||!mesh.visible||(!mesh.userData.packet&&!mesh.userData.processing&&!mesh.userData.processingOutput))continue;
   const bars=mesh.children.filter(o=>o.userData.feature!=null);if(!bars.length)continue;
   const rows=mesh.userData.row==='sequence'?3:1,cols=bars.length/rows;
   assert.ok(mesh.children.some(o=>o.userData.packetTray),'each vector has one shared tray');
   for(let row=0;row<rows;row++){
    const features=bars.slice(row*cols,(row+1)*cols);assert.ok(features.every(o=>o.position.z===features[0].position.z));
    assert.ok(features.every((o,i)=>!i||o.position.x>features[i-1].position.x),'features stay in left-to-right order');
    if(row){assert.ok(features[0].position.z>bars[(row-1)*cols].position.z);assert.equal(features[0].position.x,bars[0].position.x);}
   }
   if(rows===1)assert.equal(bars[0].position.z,0);
   checked.add(rows+'x'+cols);
  }
 }});
 for(const id of ['projection','scores']){app.api.switchLevel(id);app.api.setState(app.c.NeuralPrologue.solutions(id));app.api.getWorld().rebuild();await app.api.run();assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);}
 for(const shape of ['1x2','1x4','3x2','3x3'])assert.ok(checked.has(shape),shape+' should retain its orientation');
});
test('port badges fade around moving and fanned-out features, then recover as vectors leave',async()=>{
 let app;const seen=new Set();
 const opacity=el=>Number(el.style.opacity||1);
 app=await boot({reduced:false,frameMs:100,onFrame:()=>{
  if(!app)return;
  const input=app.document.querySelector('[data-input="project:0"]'),output=app.document.querySelector('[data-output="project"]'),sink=app.document.querySelector('[data-input="sink:0"]');
  if(!input)return;
  if(opacity(input)<.12){seen.add('input');if(opacity(sink)===1)seen.add('distant stays opaque');}
  if(opacity(output)<.12)seen.add('output');
  if(opacity(input)>.12&&opacity(input)<.95)seen.add('gradual fade');
  const projecting=app.scenes.at(-1).children.find(o=>o.visible&&o.userData.processing==='project'&&o.userData.routing);
  if(projecting&&projecting.children.some(o=>o.visible&&o.userData.featureValue!=null)&&opacity(input)<.12)seen.add('fanned features');
  if(seen.has('input')&&opacity(input)===1&&app.document.querySelector('.packet:not([hidden])'))seen.add('departed');
 }});
 app.api.switchLevel('projection');app.api.setState(app.c.NeuralPrologue.solutions('projection'));app.api.getWorld().rebuild();await app.api.run();
 for(const mark of ['input','output','distant stays opaque','gradual fade','fanned features','departed'])assert.ok(seen.has(mark),mark);
 assert.ok([...app.document.querySelectorAll('.port')].every(el=>opacity(el)===1));
 assert.equal(app.document.querySelector('#level-complete').open,true);
});

test('paused port fading follows the camera and stopping restores every badge',async()=>{
 let app,phase=0,frames=0,before,stop;
 app=await boot({reduced:false,frameMs:100,onFrame:()=>{
  if(!app)return;const port=app.document.querySelector('[data-input="project:0"]');if(!port)return;
  if(!phase&&Number(port.style.opacity||1)<.12){app.api.togglePause();before=port.style.opacity;phase=1;}
  else if(phase===1&&++frames===3){
   assert.equal(port.style.opacity,before,'pause freezes the vector and its fading');
   const scene=app.document.querySelector('#scene'),event=(target,type,props)=>{const e=new app.window.Event(type,{bubbles:true,cancelable:true});Object.assign(e,props);target.dispatchEvent(e);};
   const view=app.view();event(scene,'pointerdown',{clientX:600,clientY:300});event(app.document,'pointermove',{clientX:800,clientY:350});event(app.document,'pointerup',{});
   assert.notDeepEqual(app.view().position,view.position);
   assert.ok([...app.document.querySelectorAll('.port')].every(el=>Number.isFinite(Number(el.style.opacity||1))));
   stop=app.api.run();phase=2;
  }
 }});
 app.document.querySelector('#welcome').close();app.api.switchLevel('projection');app.api.setState(app.c.NeuralPrologue.solutions('projection'));app.api.getWorld().rebuild();await app.api.run();await stop;
 assert.equal(phase,2);assert.ok([...app.document.querySelectorAll('.port')].every(el=>Number(el.style.opacity||1)===1));assert.ok(!app.document.querySelector('#level-complete').open);
});

test('camera drag and zoom remain live during a run without editing the circuit',async()=>{
 let app,checked=false;
 const event=(target,type,properties)=>{const e=new app.window.Event(type,{bubbles:true,cancelable:true});Object.assign(e,properties);target.dispatchEvent(e);};
 app=await boot({reduced:false,frameMs:100,onFrame:({document})=>{
  if(!app||checked||!document.querySelector('.packet:not([hidden])'))return;checked=true;
  const original=JSON.stringify(app.api.getState()),start=app.view();
  event(document.querySelector('#scene'),'pointerdown',{clientX:600,clientY:300});event(document,'pointermove',{clientX:680,clientY:350});event(document,'pointerup',{});
  assert.notDeepEqual(app.view().position,start.position,'dragging the workbench moves the camera');
  const second=app.view(),label=document.querySelector('[data-node="project"]');
  event(label,'pointerdown',{clientX:680,clientY:350});event(document,'pointermove',{clientX:730,clientY:390});event(document,'pointerup',{});
  assert.notDeepEqual(app.view().position,second.position,'dragging a module label during a run also orbits');
  const beforeZoom=app.view().width;event(label,'wheel',{deltaY:-100});assert.ok(app.view().width<beforeZoom);
  const slider=document.querySelector('#camera-zoom'),speed=document.querySelector('#playback-speed').value;assert.equal(slider.disabled,false);slider.value='0';event(slider,'input',{});assert.ok(app.view().width>beforeZoom);assert.equal(document.querySelector('#playback-speed').value,speed);
  app.api.chooseOutput('R');app.api.chooseInput('sink',0);assert.equal(JSON.stringify(app.api.getState()),original,'camera gestures and port clicks must not rewire or move modules');
 }});
 app.document.querySelector('#welcome').close();app.api.switchLevel('projection');app.api.setState(app.c.NeuralPrologue.solutions('projection'));app.api.getWorld().rebuild();await app.api.run();
 assert.ok(checked);assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);assert.equal(app.document.querySelectorAll('.packet').length,0);assert.equal(app.document.body.classList.contains('network-running'),false);
});
test('construct every 3D level, run its data through scene updates, open and close every module card',async()=>{
 let app,active;const collected=new Set();app=await boot({onFrame:()=>{if(app&&app.scenes.at(-1).children.some(o=>o.userData.fusion))collected.add(active);}});const M=app.c.NeuralPrologue;
 for(const id of Object.keys(M.levels)){
  active=id;
  app.api.switchLevel(id);const s=M.solutions(id);if(M.levels[id].engine)s.settings.precision=16;app.api.setState(s);app.api.getWorld().rebuild();app.api.refresh();
  await app.api.run();assert.equal(app.document.querySelector('#loading').hidden,true);assert.equal(app.document.querySelectorAll('.node-value').length,0,'completed runs clear transient vector labels');
  assert.equal(app.document.querySelectorAll('[data-node]').length,M.levels[id].nodes.length);
  for(const n of s.nodes){app.api.getWorld().focus(n.id);assert.equal(app.document.querySelector('#module-card').open,true);assert.ok(app.document.querySelector('#card-name').textContent);app.document.querySelector('#card-close').onclick();assert.equal(app.document.querySelector('#module-card').open,false);assert.ok(app.document.querySelector('.arena > #scene'));}
  if(M.fusions[id]){
   const before=JSON.stringify(app.api.getState()),kind=M.fusions[id].kind;app.document.querySelector('#complete-close').onclick();app.api.getWorld().focusShelf(kind);assert.equal(app.document.querySelector('#module-card').open,true);assert.equal(app.document.querySelector('#card-name').textContent,M.definitions[kind].name);app.document.querySelector('#card-close').onclick();assert.equal(JSON.stringify(app.api.getState()),before);
  }
 }
 assert.deepEqual([...collected].sort(),Object.keys(M.fusions).sort(),'every collection milestone reaches the actual 3D animation');
});

test('original ReLU visibly stops negative features at zero while keeping positive features unchanged',async()=>{
 let app;const seen=new Set();
 app=await boot({reduced:false,frameMs:100,onFrame:()=>{
  if(!app)return;app.scenes.at(-1).traverse(o=>{
   if(o.userData.activation!=='relu')return;
   o.children.filter(c=>c.userData.activationFeature!=null&&c.visible).forEach(spark=>{
    const {inputValue,outputValue,phase}=spark.userData;
    if(inputValue<0&&phase>.5){seen.add('negative');assert.equal(outputValue,0);assert.ok(spark.position.x<=1e-9,'negative carrier stops at the zero threshold');}
    if(inputValue>0&&phase>.5){seen.add('positive');assert.equal(outputValue,inputValue);}
   });
  });
 }});
 app.api.switchLevel('relu');app.api.setState(app.c.NeuralPrologue.solutions('relu'));app.api.getWorld().rebuild();await app.api.run();
 assert.deepEqual([...seen].sort(),['negative','positive']);
 app.api.getWorld().focus('relu');assert.match(app.document.querySelector('#card-formula').textContent,/max\(0/);
 assert.match(app.document.querySelector('#card-example-values').textContent,/ReLU/);assert.doesNotMatch(app.document.querySelector('#card-example-values').textContent,/注意力|SiLU/);
 app.document.querySelector('#card-close').onclick();assert.equal(app.document.querySelectorAll('.packet').length,0);
});

test('cross-attention uses separate four-feature source racks and distinct travelling source and target labels',async()=>{
 let app;const seen=new Set();
 app=await boot({reduced:false,frameMs:160,languages:['en-US'],onFrame:({document})=>{
  if(!app)return;
  for(const el of document.querySelectorAll('.packet')){
   if(el.hidden)continue;
   if(el.dataset.from==='Z'){seen.add('source');assert.equal(el.querySelector('.packet-title').textContent,'Source sequence');}
   if(el.dataset.from==='R'){seen.add('target');assert.equal(el.querySelector('.packet-title').textContent,'Target features');}
  }
 }});
 const M=app.c.NeuralPrologue;app.api.switchLevel('cross-attention');
 assert.equal(app.document.querySelector('#sample-value').textContent,'3 positions × 4 features');
 const groups={R:[],Z:[]};app.scenes.at(-1).traverse(o=>{if(o.userData.sourceId)groups[o.userData.sourceId].push(o);});
 assert.equal(groups.R.length,3);assert.equal(groups.Z.length,3);assert.ok(Math.min(...groups.R.map(o=>o.position.z))>Math.max(...groups.Z.map(o=>o.position.z)));
 assert.ok([...app.document.querySelectorAll('.source-label')].some(el=>/Target features D/.test(el.textContent)&&/previous LayerNorm/.test(el.textContent)));
 assert.match(app.document.querySelector('[data-output="R"]').textContent,/D · 3 × 4/);assert.match(app.document.querySelector('[data-output="Z"]').textContent,/E · 3 × 4/);
 app.api.setState(M.solutions('cross-attention'));app.api.getWorld().rebuild();await app.api.run();assert.deepEqual([...seen].sort(),['source','target']);
 app.api.getWorld().focus('original-k');assert.match(app.document.querySelector('#card-formula').textContent,/E/);
 const example=app.document.querySelector('#card-example-values');assert.doesNotMatch(example.textContent,/original-|\bwide-/);assert.ok(example.querySelector('[style*="repeat(4"]'));
 app.document.querySelector('#card-close').onclick();assert.equal(app.document.querySelectorAll('.packet').length,0);assert.deepEqual([...app.c.I18n.missing],[]);
});

test('LayerNorm cards show the actual per-token centering and scaling as one basic operator',async()=>{
 const app=await boot(),M=app.c.NeuralPrologue,doc=app.document;
 for(const language of ['zh','en']){
  app.c.I18n.setPreference(language);
  for(const [level,kind,width]of [['layernorm','layernorm2',2],['encoder-layer','ln-attn',4],['decoder-layer','ln-cross',4]]){
   app.api.switchLevel(level);app.api.setState(M.solutions(level));app.api.getWorld().rebuild();app.api.getWorld().focus(kind);
   const ex=M.extraExample(kind,level),n=ex.normalization,input=Array.from(ex.inputs[0][1]).slice(0,width),centered=input.map(x=>x-n.mean),result=centered.map(x=>x/Math.sqrt(n.variance+n.epsilon));
   assert.equal(n.width,width);assert.ok(Math.abs(centered.reduce((a,b)=>a+b,0))<1e-9);
   assert.ok(result.every((x,i)=>Math.abs(x-ex.output[1][i])<1e-9));
   assert.equal(doc.querySelector('#card-package').hidden,true);assert.equal(M.packageOrigin(kind),null);
   assert.equal(doc.querySelectorAll('.card-layernorm-steps>b').length,3);
   const rendered=doc.querySelector('.card-layernorm-steps').textContent;
   for(const text of ['μ =','σ² =','γ','β','0.000001'])assert.ok(rendered.includes(text),text);
   assert.doesNotMatch(rendered,/undefined|NaN/);assert.doesNotMatch(doc.querySelector('#card-stats').textContent,/已封装|packaged/);
   if(language==='en')assert.doesNotMatch(rendered+doc.querySelector('#card-family').textContent,/[\u3400-\u9fff]/);
   if(kind==='layernorm2'){assert.equal(doc.querySelector('#card-name').textContent,'LayerNorm');assert.ok(rendered.includes('[0.5, −0.5]')||rendered.includes('[0.5, -0.5]'));}
   doc.querySelector('#card-close').onclick();
  }
 }
 assert.deepEqual([...app.c.I18n.missing],[]);
});

test('original LayerNorm animates centering and scaling on four features and clears its highlights after stopping',async()=>{
 let app,checked=false;
 app=await boot({reduced:false,frameMs:140,onFrame:()=>{
  if(!app)return;app.scenes.at(-1).traverse(o=>{
   const n=o.userData.normalization;if(!o.userData.processing||!n)return;
   checked=true;assert.equal(n.centered.length,4);assert.ok(Math.abs(n.centered.reduce((a,b)=>a+b,0))<1e-10);assert.ok(n.phase>=0&&n.phase<=2);
  });
 }});
 app.api.switchLevel('encoder-layer');app.api.setState(app.c.NeuralPrologue.solutions('encoder-layer'));app.api.getWorld().rebuild();await app.api.run();assert.ok(checked);
 app.scenes.at(-1).traverse(o=>assert.equal(o.userData.normalization,undefined));
 app.api.getWorld().focus('ln-attn');assert.match(app.document.querySelector('#card-example-values').textContent,/μ =/);assert.doesNotMatch(app.document.querySelector('.card-example-title').textContent,/“我”/);
 app.document.querySelector('#card-close').onclick();
});
test('LayerNorm moves actual features through mean, centering, squared deviations and division, including zero variance',async()=>{
 let app;const seen=new Set(),paths=new Map();
 app=await boot({reduced:false,frameMs:80,onFrame:({document})=>{
  if(!app)return;const formula=document.querySelector('.layernorm-calculation:not([hidden])');if(formula)seen.add(formula.dataset.normStage);
  const word=document.querySelector('#sample-name').textContent;
  app.scenes.at(-1).traverseVisible(o=>{
   const flow=o.userData.layerNorm;
   if(flow){assert.equal(flow.groups.length,1);const g=flow.groups[0],mean=(g.values[0]+g.values[1])/2,variance=g.values.reduce((s,x)=>s+(x-mean)**2,0)/2;
    assert.equal(g.mean,mean);assert.equal(g.variance,variance);assert.equal(g.divisor,Math.sqrt(variance+1e-6));g.output.forEach((v,i)=>assert.ok(Math.abs(v-(g.values[i]-mean)/g.divisor)<1e-12));
    if(!variance){seen.add('zero variance');assert.equal(g.divisor,.001);if(formula?.dataset.normStage==='variance')assert.match(formula.textContent,/0\.001/);}
   }
   const d=o.userData;
   if(d.lnMean&&o.visible){assert.equal(d.value,d.input);seen.add('mean contributions');}
   if(d.lnShift&&o.visible){assert.equal(d.value,-d.mean);seen.add('mean broadcast');}
   if(d.lnSquare&&o.visible){assert.equal(d.value,d.centered**2);if(d.centered<0)seen.add('negative centered squared');if(d.value===0)assert.equal(o.material.wireframe,true);}
   if(d.lnInput&&o.material.opacity>.01){
    const key=word+'|'+d.feature;if(!paths.has(key))paths.set(key,[]);paths.get(key).push(o.getWorldPosition(new app.c.T.Vector3()));
    if(d.centering===1&&d.scaling===0){assert.ok(Math.abs(d.displayValue-d.centered)<1e-12);seen.add('centered held');}
    if(d.scaling===1){assert.ok(Math.abs(d.displayValue-d.centered/d.divisor)<1e-12);seen.add('scaled');}
    if(d.displayValue===0){seen.add('zero carrier');assert.equal(o.material.wireframe,true);}
   }
   if(d.processingOutput==='layernorm2'&&o.visible&&o.userData.gather>0){assert.equal(o.children.filter(c=>c.userData.feature!=null&&c.visible&&c.material.opacity>.99).length,2);seen.add('gather');}
  });
 }});
 app.api.switchLevel('layernorm');app.api.setState(app.c.NeuralPrologue.solutions('layernorm'));app.api.getWorld().rebuild();await app.api.run();
 for(const mark of ['mean','center','variance','scale','mean contributions','mean broadcast','negative centered squared','centered held','scaled','zero variance','zero carrier','gather'])assert.ok(seen.has(mark),mark);
 for(const points of paths.values())assert.ok(points.every((p,i)=>!i||p.distanceTo(points[i-1])<.6),'features do not teleport between calculations');
 assert.equal(app.document.querySelectorAll('#test-results .pass').length,app.c.NeuralPrologue.samplesFor('layernorm').length);assert.equal(app.document.querySelectorAll('.layernorm-calculation,.projection-result,.packet').length,0);
 app.scenes.at(-1).traverse(o=>{assert.equal(o.userData.layerNorm,undefined);if(o.userData.lnMean||o.userData.lnShift||o.userData.lnSquare)assert.equal(o.visible,false);});
});

test('four-feature LayerNorm handles every token at the same pace and duplicate nodes use their actual input',async()=>{
 let app;const seen=new Set(),starts=new Map();
 app=await boot({reduced:false,frameMs:110,onFrame:({now,document})=>{
  if(!app)return;const M=app.c.NeuralPrologue,state=app.api.getState(),sample=M.samplesFor('encoder-layer').find(s=>s.word===document.querySelector('#sample-name').textContent);if(!sample)return;
  const ev=M.evaluate(state,sample.v);
  for(const formula of document.querySelectorAll('.layernorm-calculation:not([hidden])')){const key=formula.dataset.node+'|'+sample.word+'|'+formula.dataset.normRow;if(!starts.has(key))starts.set(key,now);}
  app.scenes.at(-1).traverseVisible(o=>{
   const f=o.userData.layerNorm;if(!f)return;const node=state.nodes.find(n=>n.id===o.userData.node),input=ev.values[node.links[0]];
   assert.equal(f.groups.length,3);assert.deepEqual([...f.groups.flatMap(g=>g.values)],[...input]);seen.add(node.id);
   for(const g of f.groups){assert.equal(g.values.length,4);const mean=g.values.reduce((s,x)=>s+x,0)/4,variance=g.values.reduce((s,x)=>s+(x-mean)**2,0)/4;assert.equal(g.mean,mean);assert.equal(g.variance,variance);assert.ok(Math.abs(g.centered.reduce((a,b)=>a+b,0))<1e-12);}
   if(node.id==='ln-attn-2'){assert.ok(Math.abs(f.groups[0].mean)<1e-12);seen.add('already normalized');}
  });
 }});
 const M=app.c.NeuralPrologue;app.api.switchLevel('encoder-layer');const state=M.initial('encoder-layer');assert.ok(M.addNode(state,'ln-attn',-2.3,0));assert.ok(M.addNode(state,'ln-attn',2.3,0));
 // Historical invalid wiring still animates before the final role error. New connections reject it.
 assert.match(M.connect(state,'R','ln-attn',0),/组件用途/);state.nodes[0].links=['R'];state.nodes[1].links=['ln-attn'];state.sink='ln-attn-2';app.api.setState(state);app.api.getWorld().rebuild();await app.api.run();
 for(const mark of ['ln-attn','ln-attn-2','already normalized'])assert.ok(seen.has(mark),mark);assert.equal(starts.size,6);
 // This deliberately incomplete encoder stops after the first sample fails,
 // but both modules must finish all three tokens before showing the verdict.
 for(const node of state.nodes){const sample=M.samplesFor('encoder-layer')[0],a=starts.get(node.id+'|'+sample.word+'|0'),b=starts.get(node.id+'|'+sample.word+'|1'),c=starts.get(node.id+'|'+sample.word+'|2');assert.ok(b>a&&c>b);assert.ok(Math.abs((b-a)-(c-b))<=220);}
 assert.equal(app.document.querySelectorAll('#test-results .fail').length,1);
 assert.ok(!app.document.querySelector('#level-complete').open,'two normalizations do not replace the encoder');assert.equal(app.document.querySelectorAll('.layernorm-calculation,.packet,.projection-result').length,0);
});

test('paused LayerNorm keeps its arithmetic and camera controls while language, speed and stop remain usable',async()=>{
 let app,phase=0,frames=0,stop,before,original;
 app=await boot({reduced:false,frameMs:90,onFrame:({document})=>{
  if(!app)return;const formula=document.querySelector('.layernorm-calculation:not([hidden])');
  if(!phase&&formula?.dataset.normStage==='variance'){app.api.togglePause();before=formula.textContent;original=JSON.stringify(app.api.getState());phase=1;}
  else if(phase===1&&++frames===3){
   assert.equal(formula.textContent,before);const scale=Number(formula.style.getPropertyValue('--calculation-scale')),wheel=new app.window.Event('wheel',{bubbles:true,cancelable:true});wheel.deltaY=2000;document.querySelector('#scene').dispatchEvent(wheel);
   assert.ok(Number(formula.style.getPropertyValue('--calculation-scale'))<scale);assert.equal(formula.dataset.detail,'abstract');
   app.c.I18n.setPreference('en');assert.match(formula.querySelector('.projection-equation').textContent,/I · s =/);
   const speed=document.querySelector('#playback-speed');speed.value='1';speed.dispatchEvent(new app.window.Event('input'));assert.equal(formula.dataset.normStage,'variance');
   app.api.togglePause();phase=2;
  }else if(phase===2&&formula?.dataset.normStage==='scale'){stop=app.api.run();phase=3;}
 }});
 app.document.querySelector('#welcome').close();app.api.switchLevel('layernorm');app.api.setState(app.c.NeuralPrologue.solutions('layernorm'));app.api.getWorld().rebuild();await app.api.run();await stop;
 assert.equal(phase,3);assert.equal(JSON.stringify(app.api.getState()),original);assert.equal(app.document.querySelectorAll('.packet,.layernorm-calculation,.projection-result').length,0);assert.ok(!app.document.querySelector('#level-complete').open);
 app.scenes.at(-1).traverse(o=>{assert.equal(o.userData.layerNorm,undefined);if(o.userData.lnMean||o.userData.lnShift||o.userData.lnSquare)assert.equal(o.visible,false);});assert.deepEqual([...app.c.I18n.missing],[]);
});

test('RMSNorm visibly reduces and rescales each token independently with equal timing and signed values',async()=>{
 let app;const seen=new Set(),starts=new Map();
 app=await boot({reduced:false,frameMs:80,onFrame:({now,document})=>{
  if(!app)return;
  const formula=document.querySelector('.rms-calculation:not([hidden])');
  if(formula){
   const key=document.querySelector('#sample-name').textContent+'|'+formula.dataset.normRow;
   if(!starts.has(key))starts.set(key,now);seen.add(formula.dataset.normStage);
   assert.ok(formula.querySelector('.projection-equation').textContent.length);assert.ok(Number(formula.style.getPropertyValue('--calculation-scale'))>0);
  }
  app.scenes.at(-1).traverse(o=>{
   if(o.userData.rms){
    const {groups}=o.userData.rms;assert.equal(groups.length,3);
    for(const g of groups){
     const r=Math.sqrt(g.values.reduce((sum,v)=>sum+v*v,0)/g.values.length+1e-6);
     assert.ok(Math.abs(g.divisor-r)<1e-12);
     g.output.forEach((value,i)=>assert.ok(Math.abs(value-g.values[i]/r)<1e-12));
    }
    if(groups[0].divisor!==groups[2].divisor)seen.add('separate divisors');
   }
   if(o.visible&&o.userData.rmsSquare){assert.equal(o.userData.value,o.userData.input**2);if(o.userData.input<0)seen.add('negative squared');}
   if(o.visible&&o.userData.rmsScaled){assert.ok(Math.abs(o.userData.value-o.userData.input/o.userData.divisor)<1e-12);if(o.userData.value<0)seen.add('negative preserved');if(o.userData.value===0){seen.add('zero');assert.equal(o.material.wireframe,true);}}
  });
 }});
 app.api.switchLevel('normalize');app.api.setState(app.c.NeuralPrologue.solutions('normalize'));app.api.getWorld().rebuild();await app.api.run();
 for(const event of ['squares','rms','scale','separate divisors','negative squared','negative preserved','zero'])assert.ok(seen.has(event),event);
 assert.equal(starts.size,9);
 for(const sample of app.c.NeuralPrologue.samplesFor('normalize')){
  const a=starts.get(sample.word+'|0'),b=starts.get(sample.word+'|1'),c=starts.get(sample.word+'|2');
  assert.ok(b>a&&c>b);assert.ok(Math.abs((b-a)-(c-b))<=160,'each token gets the same visible processing time');
 }
 assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);assert.equal(app.document.querySelectorAll('.rms-calculation,.projection-result').length,0);
 app.scenes.at(-1).traverse(o=>{assert.equal(o.userData.rms,undefined);if(o.userData.rmsSquare||o.userData.rmsScaled)assert.equal(o.visible,false);});
});

test('paused RMSNorm keeps its computation while zooming and changing language, and stopping clears it',async()=>{
 let app,phase=0,frames=0,stop,before;
 app=await boot({reduced:false,frameMs:80,onFrame:({document})=>{
  if(!app)return;
  const formula=document.querySelector('.rms-calculation:not([hidden])');
  if(!phase&&formula?.dataset.normStage==='scale'){
   app.api.togglePause();before=formula.querySelector('.projection-equation').textContent;phase=1;
  }else if(phase===1&&++frames===3){
   assert.equal(formula.querySelector('.projection-equation').textContent,before);
   const oldScale=Number(formula.style.getPropertyValue('--calculation-scale'));
   const wheel=new app.window.Event('wheel',{bubbles:true,cancelable:true});wheel.deltaY=2000;document.querySelector('#scene').dispatchEvent(wheel);
   assert.ok(Number(formula.style.getPropertyValue('--calculation-scale'))<oldScale);assert.equal(formula.dataset.detail,'abstract');
   app.c.I18n.setPreference('en');assert.match(formula.querySelector('.projection-equation').textContent,/I · /);
   stop=app.api.run();phase=2;
  }
 }});
 app.document.querySelector('#welcome').close();app.api.switchLevel('normalize');app.api.setState(app.c.NeuralPrologue.solutions('normalize'));app.api.getWorld().rebuild();await app.api.run();await stop;
 assert.equal(phase,2);assert.equal(app.document.querySelectorAll('.packet,.rms-calculation,.projection-result').length,0);assert.ok(!app.document.querySelector('#level-complete').open);
 app.scenes.at(-1).traverse(o=>{assert.equal(o.userData.rms,undefined);if(o.userData.rmsSquare||o.userData.rmsScaled)assert.equal(o.visible,false);});
 assert.deepEqual([...app.c.I18n.missing],[]);
});
test('embedding lookup selects actual IDs, copies complete table entries and keeps token order at equal pace',async()=>{
 let app;const seen=new Set(),starts=new Map(),paths=new Map();
 app=await boot({reduced:false,frameMs:80,onFrame:({now,document})=>{
  if(!app)return;const formula=document.querySelector('.token-calculation[data-operation="lookup"]:not([hidden])'),name=document.querySelector('#sample-name').textContent;
  if(formula){const key=name+'|'+formula.dataset.tokenRow;if(!starts.has(key))starts.set(key,now);seen.add(formula.dataset.tokenStage);assert.ok(formula.querySelector('.projection-equation').textContent.length);}
  app.scenes.at(-1).traverse(o=>{
   const flow=o.userData.tokenFlow;
   if(flow?.operation==='lookup'){
    for(const group of flow.groups)assert.deepEqual([...group.output],[...app.c.NeuralPrologue.math.emb[group.lookupId]]);
    if(flow.local>.28&&flow.local<.84){const cues=o.children.filter(c=>c.userData.lookupEntry!=null&&c.visible);assert.deepEqual(cues.map(c=>c.userData.lookupEntry),[flow.groups[flow.row].lookupId]);}
   }
   if(o.visible&&o.userData.tokenResult){
    const d=o.userData;assert.equal(d.value,app.c.NeuralPrologue.math.emb[d.lookupId][d.feature]);seen.add('id '+d.lookupId);
    if(d.value<0)seen.add('negative');if(d.value===0){seen.add('zero');assert.equal(o.material.wireframe,true);}
    const key=name+'|'+d.row+'|'+d.feature;if(!paths.has(key))paths.set(key,[]);paths.get(key).push(o.position.clone());
   }
   if(o.visible&&o.userData.processingOutput==='embedding'){
    const bars=o.children.filter(c=>c.userData.feature!=null&&c.visible&&c.material.opacity>.99);
    if(bars.length&&bars.length<6)seen.add('waiting results');
    if(o.userData.gather>0){assert.equal(bars.length,6,'gather waits for all three lookups');seen.add('gather');}
   }
  });
 }});
 const M=app.c.NeuralPrologue,before=JSON.stringify(M.math.emb);
 app.api.switchLevel('embedding');app.api.setState(M.solutions('embedding'));app.api.getWorld().rebuild();await app.api.run();
 for(const mark of ['address','read','copy','negative','zero','waiting results','gather',...Array.from({length:6},(_,i)=>'id '+i)])assert.ok(seen.has(mark),mark);
 assert.equal(starts.size,9);
 for(const sample of M.samplesFor('embedding')){const a=starts.get(sample.word+'|0'),b=starts.get(sample.word+'|1'),c=starts.get(sample.word+'|2');assert.ok(b>a&&c>b);assert.ok(Math.abs((b-a)-(c-b))<=160,'same reading time for each token');}
 for(const positions of paths.values()){assert.ok(positions.length>3);assert.ok(positions.every((p,i)=>!i||p.distanceTo(positions[i-1])<.5),'copies move continuously from table to output');}
 assert.equal(JSON.stringify(M.math.emb),before,'looking up a word never consumes or changes its table entry');
 assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);assert.equal(app.document.querySelectorAll('.token-calculation,.projection-result,.packet').length,0);
 app.scenes.at(-1).traverse(o=>{assert.equal(o.userData.tokenFlow,undefined);if(o.userData.tokenResult||o.userData.lookupEntry!=null)assert.equal(o.visible,false);});
});

test('sinusoidal positions visibly scale features, sample the correct frequencies, then add before gathering',async()=>{
 let app,levelId='positions';const seen=new Set();
 app=await boot({reduced:false,frameMs:100,onFrame:({document})=>{
  if(!app)return;const width=levelId==='positions'?2:4,formula=document.querySelector('.token-calculation[data-operation="position"]:not([hidden])');
  if(formula){seen.add(width+':'+formula.dataset.tokenStage);assert.ok(Number(formula.style.getPropertyValue('--calculation-scale'))>0);}
  app.scenes.at(-1).traverse(o=>{
   const flow=o.userData.tokenFlow;
   if(flow?.operation==='position'){
    for(const group of flow.groups){
     assert.equal(group.values.length,width);
     group.values.forEach((x,j)=>{const angle=group.row/10000**(2*Math.floor(j/2)/width),code=(j%2?Math.cos:Math.sin)(angle);assert.equal(group.scaled[j],x*Math.sqrt(width));assert.equal(group.codes[j],code);assert.ok(Math.abs(group.output[j]-group.scaled[j]-code)<1e-12);});
    }
   }
   if(o.visible&&o.userData.positionCode){const d=o.userData;assert.equal(d.value,(d.feature%2?Math.cos:Math.sin)(d.angle));if(d.value<0)seen.add('negative code');if(d.value===0){seen.add('zero code');assert.equal(o.material.wireframe,true);}if(width===4&&d.row===2&&d.feature===2){assert.equal(d.angle,.02);seen.add('lower frequency');}}
   if(o.visible&&o.userData.tokenResult&&o.userData.operation==='position'){const d=o.userData;assert.ok(Math.abs(d.value-d.scaled-d.code)<1e-12);seen.add(width+':sum moves');}
   if(o.visible&&o.userData.tokenInput&&o.userData.lookupId==null&&o.userData.mappingProgress===1){assert.ok(Math.abs(o.userData.displayValue-o.userData.value*Math.sqrt(width))<1e-12);seen.add(width+':input scales');}
   if(o.visible&&o.userData.processingOutput===(width===2?'position':'wide-position')&&o.userData.gather>0){assert.equal(o.children.filter(c=>c.userData.feature!=null&&c.visible&&c.material.opacity>.99).length,width*3);seen.add(width+':gather');}
  });
 }});
 for(const stage of ['positions','wider-features']){levelId=stage;app.api.switchLevel(stage);app.api.setState(app.c.NeuralPrologue.solutions(stage));app.api.getWorld().rebuild();await app.api.run();assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);}
 for(const mark of ['negative code','zero code','lower frequency',...[2,4].flatMap(w=>['scale','position','add','sum moves','input scales','gather'].map(s=>w+':'+s))])assert.ok(seen.has(mark),mark);
 assert.equal(app.document.querySelectorAll('.token-calculation,.projection-result,.packet').length,0);app.scenes.at(-1).traverse(o=>{assert.equal(o.userData.tokenFlow,undefined);if(o.userData.positionCode||o.userData.tokenResult)assert.equal(o.visible,false);});
});

test('repeated IDs keep separate positions and an extra position module animates its actual already-positioned input',async()=>{
 let app;const seen=new Set();
 app=await boot({reduced:false,frameMs:120,onFrame:()=>{
  if(!app)return;app.scenes.at(-1).traverse(o=>{
   const f=o.userData.tokenFlow;if(!f)return;
   if(f.operation==='lookup'){assert.ok(f.groups.every(g=>g.lookupId===0));assert.ok(f.groups.every(g=>g.output.join(',')==='1,0'));seen.add('repeat '+f.row);}
   if(o.userData.node==='position-2'){
    const expected=app.c.NeuralPrologue.original.position([1,0,1,0,1,0]);assert.deepEqual([...f.groups.flatMap(g=>g.values)],[...expected]);
    assert.notDeepEqual([...f.groups[0].output],[...f.groups[1].output]);seen.add('actual second input');
   }
  });
 }});
 const M=app.c.NeuralPrologue;for(const sample of M.samplesFor('positions')){sample.v=[0,0,0];sample.word='我 / 我 / 我';}
 app.api.switchLevel('positions');const state=M.solutions('positions');state.nodes.push({id:'position-2',kind:'position',x:4,z:2,links:['position']});state.sink='position-2';app.api.setState(state);app.api.getWorld().rebuild();await app.api.run();
 for(const mark of ['repeat 0','repeat 1','repeat 2','actual second input'])assert.ok(seen.has(mark),mark);
 assert.ok(!app.document.querySelector('#level-complete').open,'an extra position addition is still an incorrect circuit');assert.ok(app.document.querySelectorAll('#test-results .fail').length);
});

for(const [levelId,operation]of [['embedding','lookup'],['positions','position']])test(`paused ${operation} retains its step while zooming, changing language and speed, then cleans up on stop`,async()=>{
 let app,phase=0,frames=0,stop,before;
 app=await boot({reduced:false,frameMs:90,onFrame:({document})=>{
  if(!app)return;const formula=document.querySelector(`.token-calculation[data-operation="${operation}"]:not([hidden])`);
  if(!phase&&formula?.dataset.tokenRow==='0'&&formula.dataset.tokenStage===(operation==='lookup'?'copy':'add')){app.api.togglePause();before=formula.querySelector('.projection-equation').textContent;phase=1;}
  else if(phase===1&&++frames===3){
   assert.equal(formula.querySelector('.projection-equation').textContent,before);
   const scale=Number(formula.style.getPropertyValue('--calculation-scale')),wheel=new app.window.Event('wheel',{bubbles:true,cancelable:true});wheel.deltaY=2000;document.querySelector('#scene').dispatchEvent(wheel);
   assert.ok(Number(formula.style.getPropertyValue('--calculation-scale'))<scale);assert.equal(formula.dataset.detail,'abstract');
   app.c.I18n.setPreference('en');assert.match(formula.querySelector('.projection-equation').textContent,/I · /);
   const speed=document.querySelector('#playback-speed');speed.value='1';speed.dispatchEvent(new app.window.Event('input'));assert.equal(formula.dataset.tokenRow,'0');
   app.api.togglePause();phase=2;
  }else if(phase===2&&formula?.dataset.tokenRow==='1'){stop=app.api.run();phase=3;}
 }});
 app.document.querySelector('#welcome').close();app.api.switchLevel(levelId);app.api.setState(app.c.NeuralPrologue.solutions(levelId));app.api.getWorld().rebuild();await app.api.run();await stop;
 assert.equal(phase,3);assert.equal(app.document.querySelectorAll('.packet,.token-calculation,.projection-result').length,0);assert.ok(!app.document.querySelector('#level-complete').open);
 app.scenes.at(-1).traverse(o=>{assert.equal(o.userData.tokenFlow,undefined);if(o.userData.tokenResult||o.userData.positionCode||o.userData.lookupEntry!=null)assert.equal(o.visible,false);});assert.deepEqual([...app.c.I18n.missing],[]);
});

test('extended overview shrinks the board and fades titles, then restores labels and the saved circuit',async()=>{
 const app=await boot(),doc=app.document;
 const wheel=deltaY=>{const e=new app.window.Event('wheel',{bubbles:true,cancelable:true});e.deltaY=deltaY;doc.querySelector('#scene').dispatchEvent(e);};
 for(const id of ['projection','masked-multihead']){
  app.api.switchLevel(id);app.api.setState(app.c.NeuralPrologue.solutions(id));app.api.getWorld().rebuild();
  const saved=JSON.stringify(app.api.getState()),initial=app.view().width,base=Math.max(1.35,initial/23);
  assert.equal(doc.querySelector('#camera-zoom-rate').textContent,'100%');assert.equal(doc.querySelector('#camera-zoom').disabled,false);
  const titles=()=>[...doc.querySelectorAll('.node-label,.source-label,.port')];
  const checkScale=()=>{for(const el of titles()){const scale=Number(el.style.transform.match(/scale\(([^)]+)\)/)?.[1]);assert.ok(Math.abs(scale-initial/app.view().width)<1e-8,'board labels follow the same projected scale as components');}};
  checkScale();
  assert.ok(titles().every(el=>!el.hidden&&Number(el.style.opacity||1)===1),'initial titles stay legible even on the large board');
  wheel((base-initial/23)/.0008);const oldWidth=app.view().width;
  wheel(base*.3/.0008);
  assert.ok(app.view().width>oldWidth,'the whole board continues shrinking past the former limit');checkScale();
  assert.ok(titles().every(el=>!el.hidden&&Number(el.style.opacity)>0&&Number(el.style.opacity)<1));
  const halfway=app.view().width,slider=doc.querySelector('#camera-zoom');
  assert.ok(Number(slider.value)>0);slider.value='0';slider.dispatchEvent(new app.window.Event('input'));assert.ok(app.view().width>halfway);
  assert.ok(titles().every(el=>el.hidden&&Number(el.style.opacity)===0),'invisible titles no longer intercept pointer or keyboard input');
  assert.ok([...doc.querySelectorAll('.port')].every(el=>el.disabled),'invisible ports cannot receive input');
  const furthest=app.view().width;wheel(10000);assert.equal(app.view().width,furthest,'overview has a finite limit');
  app.c.I18n.setPreference('en');assert.ok(titles().every(el=>el.hidden),'a label rebuild preserves the distant view');
  const node=app.api.getState().nodes[0].id;app.api.getWorld().focus(node);await new Promise(setImmediate);
  doc.querySelector('#card-close').onclick();assert.equal(app.view().width,furthest);assert.ok(titles().every(el=>el.hidden));
  slider.value='100';slider.dispatchEvent(new app.window.Event('input'));assert.ok(app.view().width<initial);assert.ok(titles().every(el=>!el.hidden&&Number(el.style.opacity||1)===1),'zooming in restores all titles and port labels');checkScale();
  doc.querySelector('#camera-reset').onclick();assert.equal(doc.querySelector('#camera-zoom-rate').textContent,'100%');assert.ok(Number(slider.value)>0&&Number(slider.value)<100);assert.ok(Math.abs(app.view().width-initial)<1e-8);assert.ok(titles().every(el=>!el.hidden));
  assert.equal(JSON.stringify(app.api.getState()),saved);
 }
});

test('every ordinary lesson keeps numeric detail at its default 100% even in a narrower desktop viewport',async()=>{
 const viewport={width:1100,height:700},app=await boot({viewport}),M=app.c.NeuralPrologue,doc=app.document;
 const slider=doc.querySelector('#camera-zoom'),reset=()=>doc.querySelector('#camera-reset').onclick();
 for(const id of Object.values(M.routes).flatMap(route=>route.ids).filter(id=>!M.levels[id].floating)){
  app.api.switchLevel(id);app.api.setState(M.solutions(id));app.api.getWorld().rebuild();const saved=JSON.stringify(app.api.getState());
  const faces=[];app.scenes.at(-1).traverse(o=>{if(o.userData.detailText)faces.push(o);});
  assert.equal(doc.querySelector('#camera-zoom-rate').textContent,'100%',id);
  assert.ok(faces.every(o=>o.visible),id+' default shows all matrix values');
  slider.value='0';slider.dispatchEvent(new app.window.Event('input'));
  assert.ok(faces.every(o=>!o.visible),id+' distant view abstracts all matrices together');
  reset();assert.equal(doc.querySelector('#camera-zoom-rate').textContent,'100%');assert.ok(faces.every(o=>o.visible),id+' reset restores numeric detail');
  assert.equal(JSON.stringify(app.api.getState()),saved,id+' camera changes do not edit the circuit');
 }
});

test('single- and two-head attention show formulas and matrix values at 100% and restore them after zooming during a pause',async()=>{
 let app,phase=0;const seen=new Set(),viewport={width:1100,height:700};
 app=await boot({viewport,reduced:false,frameMs:200,onFrame:({document})=>{
  if(!app)return;
  const formulas=[...document.querySelectorAll('.sequence-projection-calculation:not([hidden])')];
  for(const formula of formulas){
   assert.equal(formula.dataset.detail,'full','the default view never hides active arithmetic');
   assert.equal(Number(formula.style.getPropertyValue('--calculation-scale')),1,'100% uses the normal formula font size');
   assert.match(formula.querySelector('.projection-equation').textContent,/×/);seen.add(formula.dataset.node);
  }
  const formula=formulas.find(el=>el.dataset.node==='aggregate');if(phase||!formula)return;phase=1;app.api.togglePause();
  let unit;app.scenes.at(-1).traverse(o=>{if(o.userData.node==='aggregate')unit=o;});
  const arithmetic=JSON.stringify(unit.userData.aggregation),text=formula.textContent,faces=unit.children.filter(o=>o.userData.detailText);
  assert.ok(faces.length&&faces.every(o=>o.visible));
  const slider=document.querySelector('#camera-zoom');slider.value='0';slider.dispatchEvent(new app.window.Event('input'));
  assert.equal(formula.dataset.detail,'abstract');assert.ok(faces.every(o=>!o.visible));assert.ok(Number(formula.style.getPropertyValue('--calculation-scale'))<1);
  document.querySelector('#camera-reset').onclick();
  assert.equal(document.querySelector('#camera-zoom-rate').textContent,'100%');assert.equal(formula.dataset.detail,'full');assert.ok(faces.every(o=>o.visible));
  viewport.width=1400;document.querySelector('#camera-reset').onclick();assert.equal(formula.dataset.detail,'full');assert.equal(Number(formula.style.getPropertyValue('--calculation-scale')),1);
  assert.equal(formula.textContent,text);assert.equal(JSON.stringify(unit.userData.aggregation),arithmetic);app.api.togglePause();
 }});
 const M=app.c.NeuralPrologue;app.api.switchLevel('self-attention');const state=M.initial('self-attention');
 for(const [kind,x,z]of [['original-weights',-4,-2],['v',-4,2],['aggregate',0,0],['output',4,0]])assert.ok(M.addNode(state,kind,x,z));
 for(const [from,to,slot]of [['R','original-weights',0],['R','v',0],['original-weights','aggregate',0],['v','aggregate',1],['aggregate','output',0],['output','sink',0]])assert.equal(M.connect(state,from,to,slot),null);
 app.api.setState(state);app.api.getWorld().rebuild();const saved=JSON.stringify(state);await app.api.run();
 assert.deepEqual([...seen].sort(),['aggregate','output','v']);assert.equal(phase,1);assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);assert.equal(JSON.stringify(state),saved);
 seen.clear();app.api.switchLevel('multihead');const multi=M.solutions('multihead'),multiSaved=JSON.stringify(multi);app.api.setState(multi);app.api.getWorld().rebuild();await app.api.run();
 assert.deepEqual([...seen],['output']);assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);assert.equal(JSON.stringify(multi),multiSaved);
});

test('small matrices keep their blocks while numbers return when zoomed in',async()=>{
 const app=await boot();app.api.switchLevel('original-ffn');app.api.setState(app.c.NeuralPrologue.solutions('original-ffn'));app.api.getWorld().rebuild();
 const state=JSON.stringify(app.api.getState());
 let down;app.scenes.at(-1).traverse(o=>{if(o.userData.node==='down')down=o;});
 const faces=down.children.filter(o=>o.userData.detailText);assert.equal(faces.length,8);
 const wheel=new app.window.Event('wheel',{bubbles:true,cancelable:true});wheel.deltaY=2000;app.document.querySelector('#scene').dispatchEvent(wheel);
 assert.ok(faces.every(o=>!o.visible),'unreadable matrix numbers are hidden');
 assert.ok(down.children.filter(o=>o.geometry?.type==='BoxGeometry'&&o.visible).length>=8,'the colored weight blocks stay visible');
 await app.api.run();assert.equal(app.document.querySelectorAll('#test-results .pass').length,app.c.NeuralPrologue.levels['original-ffn'].sampleCount,'abstract rendering does not change the calculation');
 app.api.getWorld().focus('down');await new Promise(setImmediate);
 assert.ok(faces.every(o=>o.visible),'the close-up restores the actual numbers');
 app.document.querySelector('#card-close').onclick();assert.ok(faces.every(o=>!o.visible),'returning to the distant view restores abstraction immediately');
 assert.equal(JSON.stringify(app.api.getState()),state);
});

test('gate and reused linear projection cards compare the same input with a worked feature in both languages and levels',async()=>{
 const app=await boot(),M=app.c.NeuralPrologue,doc=app.document,round=v=>Number(v.toFixed(2));
 for(const language of ['zh','en']){
  app.c.I18n.setPreference(language);
  for(const level of ['gating','ffn']){
   app.api.switchLevel(level);const state=M.solutions(level);app.api.setState(state);app.api.getWorld().rebuild();
   const saved=JSON.stringify(state),ev=M.evaluate(state,[1,0]),expected={U:ev.values.project,G:ev.values.gate,S:ev.values.silu,H:ev.values.multiply};let previous=null;
   for(const kind of ['gate','project']){
    app.api.getWorld().focus(kind);assert.equal(doc.querySelector('#module-card').open,true);
    const comparison=doc.querySelector('.card-comparison');assert.ok(comparison);
    assert.match(doc.querySelector('#card-meaning').textContent,/Wg/);assert.match(doc.querySelector('#card-meaning').textContent,/\bW\b/);
    assert.equal(doc.querySelector('#card-name').textContent,M.definitions[kind].name);assert.doesNotMatch(doc.querySelector('#module-card').textContent,/内容投影|Content projection/);
    assert.match(doc.querySelector('#card-purpose').textContent,/SiLU/);
    assert.equal(doc.querySelector('#card-formula').textContent,(kind==='gate'?'G':'U')+' = '+(level==='ffn'?'Z':'X')+' × W'+(kind==='gate'?'g':''));
    assert.equal(doc.querySelector('#card-example-input').textContent,(level==='ffn'?'Z = [1.41, 0]':'X = [1, 0]'));
    for(const [symbol,values]of Object.entries(expected)){
     const cells=[...comparison.querySelectorAll(`[data-symbol="${symbol}"] i`)];assert.deepEqual(cells.map(c=>Number(c.textContent)),Array.from(values,round));
     assert.deepEqual(cells.map((c,i)=>c.classList.contains('example-focus')?i:-1).filter(i=>i>=0),[2],'the same feature is highlighted at every stage');
    }
    for(const [role,symbol]of [['content','U'],['scale','S'],['result','H']])assert.equal(Number(comparison.querySelector(`[data-role="${role}"] b`).textContent),round(expected[symbol][2]));
    assert.equal(comparison.querySelector('.card-gate-step').textContent,`SiLU(${Number(expected.G[2].toFixed(4))}) ≈ ${round(expected.S[2])}`);
    if(previous)assert.equal(comparison.textContent,previous,'switching branches keeps one comparable example');previous=comparison.textContent;
    if(language==='en')assert.doesNotMatch(doc.querySelector('#module-card').textContent,/[\u3400-\u9fff]/);
    doc.querySelector('#card-close').onclick();assert.equal(JSON.stringify(app.api.getState()),saved);
   }
  }
 }
 assert.deepEqual([...app.c.I18n.missing],[]);
});
test('SiLU cards connect each input to its output and carry the same example into multiplication',async()=>{
 const app=await boot(),M=app.c.NeuralPrologue,doc=app.document,round=v=>Number(v.toFixed(2));
 const values=(root,symbol)=>[...root.querySelectorAll(`[data-symbol="${symbol}"] i`)].map(el=>Number(el.textContent));
 for(const language of ['zh','en']){
  app.c.I18n.setPreference(language);
  for(const level of ['activation','gating','ffn']){
   app.api.switchLevel(level);const state=M.solutions(level);app.api.setState(state);app.api.getWorld().rebuild();
   const saved=JSON.stringify(state),source=level==='activation'?[1,1]:[1,0],ev=M.evaluate(state,source),symbol=level==='activation'?'P':'G',input=ev.values[symbol==='P'?'project':'gate'],output=ev.values.silu;
   app.api.getWorld().focus('silu');const example=doc.querySelector('.card-silu-example');assert.ok(example);
   assert.equal(doc.querySelector('#card-formula').textContent,`S = SiLU(${symbol})`);
   assert.deepEqual(values(example,symbol),Array.from(input,round));assert.deepEqual(values(example,'S'),Array.from(output,round));
   for(const selector of [`[data-symbol="${symbol}"] i`,'.card-silu-link i','[data-symbol="S"] i']){
    const cells=[...example.querySelectorAll(selector)];assert.equal(cells.length,input.length);
    assert.deepEqual(cells.flatMap((el,i)=>el.classList.contains('example-focus')?[i]:[]),[2],'the input, SiLU arrow and output highlight one matching feature');
   }
   const equation=example.querySelector('.card-silu-worked code'),shownInput=Number(equation.textContent.match(/^SiLU\(([^)]+)\)/)[1]);
   assert.equal(shownInput,Number(input[2].toFixed(4)));assert.equal(Number(equation.querySelector('sup').textContent),-shownInput);
   assert.equal(round(M.silu(shownInput)),round(output[2]),'the displayed scalar calculation rounds to the displayed output');
   assert.ok(output.some(v=>v<0),'negative SiLU outputs remain visible');
   const product=example.querySelector('.card-gate-result');assert.equal(!!product,level!=='activation');
   if(product)for(const [role,expected]of [['content',ev.values.project[2]],['scale',output[2]],['result',ev.values.multiply[2]]])assert.equal(Number(product.querySelector(`[data-role="${role}"] b`).textContent),round(expected));
   if(language==='en')assert.doesNotMatch(doc.querySelector('#module-card').textContent,/[\u3400-\u9fff]/);
   doc.querySelector('#card-close').onclick();
   if(level!=='activation'){
    app.api.getWorld().focus('multiply');assert.equal(doc.querySelector('.card-silu-example'),null);
    assert.deepEqual(values(doc.querySelector('#card-example-values'),'S'),Array.from(output,round),'the next card uses the SiLU output from the same input');
    doc.querySelector('#card-close').onclick();
   }
   assert.equal(JSON.stringify(app.api.getState()),saved,'inspection does not change the puzzle');
  }
 }
 assert.deepEqual([...app.c.I18n.missing],[]);
});
test('residual cards and the lesson explain baseline plus update with the actual shared example',async()=>{
 const app=await boot(),M=app.c.NeuralPrologue,doc=app.document,round=v=>Number(v.toFixed(2));
 for(const language of ['zh','en']){
  app.c.I18n.setPreference(language);
  for(const level of ['residual','ffn']){
   app.api.switchLevel(level);const state=M.solutions(level);app.api.setState(state);app.api.getWorld().rebuild();
   const saved=JSON.stringify(state),ev=M.evaluate(state,[1,0]),source=level==='residual'?'X':'R',update=ev.values[level==='residual'?'relu-ffn':'down'];
   app.api.getWorld().focus('add');const card=doc.querySelector('.card-add-example');assert.ok(card);
   assert.equal(doc.querySelector('#card-formula').textContent,`Y = ${source} + F`+(level==='residual'?'; F = FFN(X)':''));
   for(const [symbol,values]of [[source,ev.values.R],['F',update],['Y',ev.values.add]]){
    const cells=[...card.querySelectorAll(`[data-symbol="${symbol}"] i`)];assert.deepEqual(cells.map(c=>Number(c.textContent)),Array.from(values,round));
    assert.deepEqual(cells.flatMap((c,i)=>c.classList.contains('example-focus')?[i]:[]),[0]);
   }
   assert.equal(Number(card.querySelector('[data-role="baseline"]').textContent),1);assert.equal(Number(card.querySelector('[data-role="update"]').textContent),round(update[0]));
   assert.equal(Number(card.querySelector('.card-add-step b').textContent),round(ev.values.add[0]));assert.equal(card.querySelector('.card-add-zero code').textContent,`F = [0, 0] → Y = ${source}`);
   const reason=doc.querySelector('#mission-reason');assert.equal(reason.hidden,false);
   if(level==='residual')for(const [role,values]of [['baseline',ev.values.R],['update',ev.values['relu-ffn']],['result',ev.values.add]])assert.deepEqual(JSON.parse(reason.querySelector(`[data-role="${role}"] code`).textContent),Array.from(values,round));
   else assert.equal(reason.querySelector('.residual-example'),null);
   if(language==='en'){assert.doesNotMatch(doc.querySelector('#module-card').textContent,/[\u3400-\u9fff]/);assert.doesNotMatch(reason.textContent,/[\u3400-\u9fff]/);}
   doc.querySelector('#card-close').onclick();assert.equal(JSON.stringify(app.api.getState()),saved);
  }
 }
 assert.deepEqual([...app.c.I18n.missing],[]);
});
test('all matrix labels switch abstraction together and a close-up preserves the board mode',async()=>{
 const app=await boot();app.api.switchLevel('original-ffn');app.api.setState(app.c.NeuralPrologue.solutions('original-ffn'));app.api.getWorld().rebuild();
 const faces=[];app.scenes.at(-1).traverse(o=>{if(o.userData.detailText)faces.push(o);});
 assert.ok(new Set(faces.map(o=>o.userData.detailText.fontFraction)).size>1,'short numbers and decimals have different fitted sizes');
 const wheel=deltaY=>{const e=new app.window.Event('wheel',{bubbles:true,cancelable:true});e.deltaY=deltaY;app.document.querySelector('#scene').dispatchEvent(e);};
 const uniform=()=>{assert.equal(new Set(faces.map(o=>o.visible)).size,1,'no partly abstracted matrix or neighboring module');return faces[0].visible;};
 assert.equal(uniform(),true,'normal-size modules keep every number, including decimals');
 wheel(-2000);assert.equal(uniform(),true);
 let steps=0;while(uniform()&&steps++<50)wheel(25);
 assert.equal(uniform(),false,'a small module footprint abstracts every matrix');
 // Return slightly across the entry threshold: hysteresis keeps the board
 // abstract, and opening a detailed card must not reset that decision.
 wheel(-25);assert.equal(uniform(),false);
 app.api.getWorld().focus('project');await new Promise(setImmediate);
 assert.ok(faces.filter(o=>o.parent.userData.node==='project').every(o=>o.visible));
 app.document.querySelector('#card-close').onclick();assert.equal(uniform(),false);
 while(!uniform()&&steps-- >0)wheel(-25);
 assert.equal(uniform(),true,'all numbers return together when readable');
});
test('3D fault bubble and pin dismiss without losing recorded failure',async()=>{
 const app=await boot();app.api.switchLevel('vocabulary');await app.api.run();assert.ok(app.api.getFault());const b=app.document.querySelector('[data-dismiss-result]');assert.ok(b);b.dispatchEvent(new app.window.Event('click',{bubbles:true}));assert.equal(app.api.getFault(),null);assert.equal(app.document.querySelectorAll('.local-fault,.fault-pin').length,0);assert.ok(app.document.querySelector('#test-results .fail'));await app.api.run();assert.ok(app.document.querySelector('[data-dismiss-result]'));
});

test('dual input plates, titles and outlets remain separated and aligned across all dual-source lessons',async()=>{
 const app=await boot(),M=app.c.NeuralPrologue,T=app.c.T,doc=app.document;
 for(const id of ['cross-attention','decoder-layer','encoder-decoder']){
  app.api.switchLevel(id);app.api.setState(M.solutions(id));app.api.getWorld().rebuild();
  const groups={R:[],Z:[]};app.scenes.at(-1).traverse(o=>{if(o.userData.sourceId)groups[o.userData.sourceId].push(o);});
  const centers=Object.fromEntries(['R','Z'].map(key=>[key,groups[key].find(g=>g.userData.sourceRow===0).position.z+1.05]));
  assert.ok(centers.R-centers.Z-3.8>=2,'clear gap between the two plates');
  for(const key of ['R','Z']){
   const label=doc.querySelector('[data-source-label="'+key+'"]');
   const pos=new T.Vector3(M.levels[id].sourceX,.95,centers[key]-2.05).project(app.camera());
   assert.ok(Math.abs(parseFloat(label.style.top)-(1-pos.y)*380)<.1,'title follows its own plate');
   const port=doc.querySelector('[data-output="'+key+'"]'),dim=M.levels[id].sourceDim,x=M.levels[id].sourceX+(dim===12?1.65:1.15),p=new T.Vector3(x,.65,centers[key]).project(app.camera());
   assert.ok(Math.abs(parseFloat(port.style.top)-(1-p.y)*380)<.1,'outlet follows its plate');
  }
  const lowerTitle=Number.parseFloat(doc.querySelector('[data-source-label="R"]').style.top),upperFront=new T.Vector3(M.levels[id].sourceX,.6,centers.Z+1.9).project(app.camera());
  assert.ok(lowerTitle>(1-upperFront.y)*380+20,'D title clears the E plate in the default view');
 }
});

test('select a cable, delete it with the visible action or keyboard, and undo it',async()=>{
 const app=await boot();app.document.querySelector('#welcome').close();app.api.switchLevel('gating');app.api.setState(app.c.NeuralPrologue.solutions('gating'));app.api.getWorld().rebuild();
 const selectedPath=()=>app.document.querySelector('.wire-targets path[data-wire-to="multiply"][data-wire-port="0"]');
 selectedPath().dispatchEvent(new app.window.Event('click',{bubbles:true}));assert.ok(app.document.querySelector('[data-delete-wire]'));assert.equal(app.api.getState().nodes.find(n=>n.id==='multiply').links[0],'silu');
 app.document.querySelector('[data-delete-wire]').dispatchEvent(new app.window.Event('click',{bubbles:true}));assert.equal(selectedPath(),null);assert.equal(app.document.querySelector('.wire-actions'),null);assert.equal(app.api.getState().nodes.find(n=>n.id==='multiply').links[1],'project');
 app.document.querySelector('#undo').onclick();selectedPath().dispatchEvent(new app.window.Event('click',{bubbles:true}));const event=new app.window.Event('keydown',{bubbles:true});event.key='Delete';app.document.dispatchEvent(event);assert.equal(selectedPath(),null);
});
test('completed and cancelled wiring clear the dashed drag preview',async()=>{
 const app=await boot();app.document.querySelector('#welcome').close();app.api.switchLevel('projection');app.document.querySelector('#scene').getBoundingClientRect=()=>({left:0,top:0,width:1400,height:760,right:1400,bottom:760});
 const previews=()=>app.scenes.at(-1).children.filter(o=>o.material?.type==='LineDashedMaterial');
 app.api.chooseOutput('R');app.api.getWorld().previewAt(650,340);assert.equal(previews().length,1);app.api.chooseInput('project',0);assert.equal(previews().length,0);
 app.api.chooseOutput('project');app.api.getWorld().previewAt(800,380);assert.equal(previews().length,1);const event=new app.window.Event('keydown',{bubbles:true});event.key='Escape';app.document.dispatchEvent(event);assert.equal(previews().length,0);
});
test('SiLU carriers follow curved channels and shrink by the actual sigmoid gate',async()=>{
 let app;const observed=new Map();let snapshots=0;
 app=await boot({reduced:false,frameMs:20,onFrame:({document})=>{
  const label=document.querySelector('.processing-packet[data-node="silu"]');if(!label||label.hidden)return;
  const key=label.dataset.row;app.scenes.at(-1).traverse(o=>{
   if(o.userData.activationFeature==null||!o.visible)return;const d=o.userData,id=key+':'+d.activationFeature;if(!observed.has(id))observed.set(id,[]);
   assert.ok(Math.abs(d.outputValue-d.inputValue*d.sigmoid)<1e-12);assert.ok(Math.abs(d.value)<=Math.abs(d.inputValue)+1e-12);if(d.inputValue<0)assert.ok(d.value<0,'SiLU must not zero or flip negative inputs');
   observed.get(id).push({scale:o.scale.x,x:o.position.x,y:o.position.y,phase:d.phase});snapshots++;
  });
 }});
 app.api.switchLevel('activation');app.api.setState(app.c.NeuralPrologue.solutions('activation'));app.api.getWorld().rebuild();await app.api.run();assert.ok(snapshots>30);
 const negative=observed.get('1:3');assert.ok(negative.length>4);assert.ok(negative.at(-1).scale<negative[0].scale*.6);assert.ok(negative.at(-1).x>negative[0].x);assert.ok(negative.at(-1).y>negative[0].y+.3);
 app.scenes.at(-1).traverse(o=>{if(o.userData.activationFeature!=null)assert.equal(o.visible,false,'no lingering activation effect');});assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);
});

test('first-level dashed guide demonstrates the connection without creating one and hides once used',async()=>{
 const app=await boot(),visible=()=>{const g=app.document.querySelector('.connection-guide');return !!g&&!g.hasAttribute('hidden');};
 assert.ok(visible());assert.equal(app.api.getState().sink,null);
 assert.match(app.document.querySelector('.connection-guide path').getAttribute('d'),/^M/);
 // Exercise the port actions directly: LinkeDOM caches dataset before later setAttribute calls.
 app.api.chooseOutput('R');assert.equal(visible(),false);assert.equal(app.api.getState().sink,null);
 app.api.chooseInput('sink',0);assert.equal(app.api.getState().sink,'R');assert.equal(visible(),false);
 await app.api.run();assert.equal(visible(),false);assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);
 app.document.querySelector('#reset').onclick();assert.ok(visible());app.api.switchLevel('projection');assert.equal(visible(),false);
});

test('real packet timeline gives every row equal duration and scales with the slider',async()=>{
 const app=await boot({reduced:false,frameMs:50}),world=app.api.getWorld(),original=world.animate,slider=app.document.querySelector('#playback-speed'),totals=[];
 app.api.setState(app.c.NeuralPrologue.solutions('wire'));world.rebuild();
 for(const rate of [.5,1,2]){
  slider.value=String(rate);slider.dispatchEvent(new app.window.Event('input',{bubbles:true}));const durations=[];
  world.animate=async(...args)=>{const start=app.time();await original(...args);durations.push(app.time()-start);};
  await app.api.run();assert.equal(durations.length,3);assert.ok(durations.every(ms=>ms===durations[0]));assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);totals.push(durations[0]);
 }
 assert.ok(Math.abs(totals[0]-2*totals[1])<=50);assert.ok(Math.abs(totals[1]-2*totals[2])<=50);
});
test('changing speed mid-flight keeps packets moving forward and still completes',async()=>{
 let changed=false,firstPacket=null,positions=[];
 const app=await boot({reduced:false,frameMs:50,onFrame:({now,document,window})=>{
  const packet=document.querySelector('.packet');if(!firstPacket&&packet)firstPacket=packet;
  if(packet===firstPacket&&packet){const left=parseFloat(packet.style.left);if(Number.isFinite(left))positions.push(left);}
  if(now>=200&&!changed&&packet){changed=true;const slider=document.querySelector('#playback-speed');slider.value='.25';slider.dispatchEvent(new window.Event('input',{bubbles:true}));}
 }});
 app.api.setState(app.c.NeuralPrologue.solutions('wire'));app.api.getWorld().rebuild();await app.api.run();
 assert.ok(changed);assert.ok(positions.length>10);assert.ok(positions.every((x,i)=>i===0||x>=positions[i-1]));assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);
});

test('stationary rows pulse while labelled vectors follow invisible departure and arrival paths',async()=>{
 let app,tubeCount;const seen=new Map(),arrivals=new Map(),projected=new Map(),rowTones=new Map();
 const tubes=scene=>{let count=0;scene.traverse(o=>{if(o.geometry?.type==='TubeGeometry')count++;});return count;};
 app=await boot({reduced:false,frameMs:10,onFrame:({document})=>{
  const scene=app.scenes.at(-1);if(!scene)return;
  assert.equal(tubes(scene),tubeCount,'running must not draw extra wires inside either rack');
  scene.traverse(o=>{if(o.userData.sourceRow!=null){assert.equal(o.position.y,.65,'input trays stay still');if(o.userData.active){const i=o.userData.sourceRow;if(!rowTones.has(i))rowTones.set(i,new Set());rowTones.get(i).add(o.children[0].material.color.getHex());}}if(o.userData.sinkRow!=null)assert.equal(o.position.y,.66,'output trays stay still');});
  for(const el of document.querySelectorAll('.packet[data-motion="wire"]')){
   const row=Number(el.dataset.row),mesh=scene.children.find(o=>o.userData.packet&&o.userData.from===el.dataset.from);
   if(el.dataset.from==='R'&&!seen.has(row)){
    assert.equal(el.querySelector('.packet-title').textContent,['我','爱','猫'][row]);
    assert.equal(mesh.position.z,-1.05+row*.96);
    const active=[];scene.traverse(o=>{if(o.userData.sourceRow!=null&&o.userData.active)active.push(o.userData.sourceRow);});assert.deepEqual(active,[row]);
    const features=mesh.children.filter(o=>o.userData.feature!=null);
    seen.set(row,{tone:el.style.getPropertyValue('--row-color'),heights:features.map(o=>o.geometry.parameters.height)});
   }
   if(el.dataset.from==='project'&&!el.hidden){
    projected.set(row,JSON.parse(el.querySelector('.packet-values').textContent));
    arrivals.set(row,Math.min(arrivals.get(row)??Infinity,Math.abs(mesh.position.z-(-1.05+row*.96))));
   }
  }
 }});
 app.api.switchLevel('projection');app.api.setState(app.c.NeuralPrologue.solutions('projection'));app.api.getWorld().rebuild();tubeCount=tubes(app.scenes.at(-1));await app.api.run();
 assert.equal(seen.size,3);assert.equal(new Set([...seen.values()].map(r=>r.tone)).size,3);
 assert.ok(seen.get(0).heights[0]>4*seen.get(0).heights[1]);assert.ok(seen.get(1).heights[1]>4*seen.get(1).heights[0]);assert.equal(seen.get(2).heights[0],seen.get(2).heights[1]);
 assert.deepEqual([...projected.values()],[[1,0,1,.5],[0,1,.5,-1],[1,1,1.5,-.5]]);
 assert.ok([...rowTones.values()].every(tones=>tones.size>1),'each active row has a visible colour pulse');
 assert.ok([...arrivals.values()].every(d=>d<.15));assert.equal(app.document.querySelectorAll('.packet,.row-marker').length,0);
});

test('sequence packets represent all three rows, and extra validation samples are not a fourth or fifth token row',async()=>{
 let app,mode='sequence',sequenceFrames=0,validationFrames=0;
 app=await boot({reduced:false,frameMs:200,onFrame:({document})=>{
  const el=document.querySelector('.packet');if(!el)return;
  if(mode==='sequence'){
   sequenceFrames++;assert.equal(el.dataset.row,'sequence');assert.match(el.textContent,/整段输入/);assert.equal(document.querySelector('.row-marker'),null);
   const active=[];app.scenes.at(-1).traverse(o=>{if(o.userData.sourceRow!=null&&o.userData.active)active.push(o.userData.sourceRow);});assert.deepEqual(active,[0,1,2]);
  }else if(el.dataset.row==='validation'){
   validationFrames++;assert.match(el.querySelector('.packet-title').textContent,/^新输入 [AB]$/);
   const active=[];app.scenes.at(-1).traverse(o=>{if(o.userData.sourceRow!=null&&o.userData.active)active.push(o.userData.sourceRow);});assert.deepEqual(active,[0]);
  }
 }});
 for(const id of ['attention','gating']){mode=id==='attention'?'sequence':'row';app.api.switchLevel(id);app.api.setState(app.c.NeuralPrologue.solutions(id));app.api.getWorld().rebuild();await app.api.run();}
 assert.ok(sequenceFrames>0);assert.ok(validationFrames>0);
});

test('extra scalar validation cases replace the first input and output matrix rows',async()=>{
 let app;const seen=new Set();
 app=await boot({reduced:false,frameMs:40,onFrame:({document})=>{
  if(!app)return;const name=document.querySelector('#sample-name').textContent;if(!/^新输入 [AB]$/.test(name))return;
  const M=app.c.NeuralPrologue,example=M.samplesFor('workshop').find(item=>item.word===name),scene=app.scenes.at(-1),source=[],output=[];
  scene.traverse(o=>{
   const text=o.children?.filter(c=>c.material?.map).map(c=>c.material.map.image.drawnText);
   if(o.userData.sourceRow!=null&&o.userData.active)source.push([o.userData.sourceRow,...text]);
   if(o.userData.sinkRow!=null&&o.userData.active)output.push([o.userData.sinkRow,...text]);
  });
  assert.equal(source.length,1);assert.equal(output.length,1);
  assert.deepEqual(source[0],[0,name,...example.v.map(value=>String(Number(value.toFixed(2))))]);
  assert.equal(output[0][0],0);assert.equal(output[0][1],name.at(-1));
  if(output[0].at(-1)==='·')return;
  assert.deepEqual(output[0].slice(3),Array.from(M.reference(example.v,'workshop'),value=>String(Number(value.toFixed(2)))));
  seen.add(name);
 }});
 app.api.switchLevel('workshop');app.api.setState(app.c.NeuralPrologue.solutions('workshop'));app.api.getWorld().rebuild();await app.api.run();
 assert.deepEqual([...seen].sort(),['新输入 A','新输入 B']);
});

test('projection travels continuously through the matrix and animates only nonzero weighted contributions',async()=>{
 let app;const paths=new Map(),phases=new Set(),contributions=new Map();
 app=await boot({reduced:false,frameMs:10,onFrame:({document})=>{
  const scene=app.scenes.at(-1);
  for(const row of ['0','1','2']){
   const candidates=Array.from(document.querySelectorAll(`.packet[data-row="${row}"]`));if(!candidates.length)continue;
   const visible=candidates.filter(el=>!el.hidden);
   if(!visible.length){assert.ok(!paths.has(row),'no blank frame between input, processing and output');continue;}
   assert.equal(visible.length,1,'one continuous label for this projection');
   const el=visible[0],inside=el.dataset.motion==='process',mesh=scene.children.find(o=>inside?o.userData.processing==='project':o.userData.packet&&o.userData.from===el.dataset.from);
   if(!paths.has(row))paths.set(row,[]);paths.get(row).push(mesh.position.clone());
   if(inside){phases.add(el.dataset.phase);assert.equal(document.querySelector('.node-value[data-owner="project"]'),null,'result appears only after processing');}
   if(!contributions.has(row))contributions.set(row,new Set());
   scene.traverse(o=>{if(o.userData.contribution&&o.visible)contributions.get(row).add(`${o.userData.input}:${o.userData.output}`);});
  }
 }});
 app.api.switchLevel('projection');app.api.setState(app.c.NeuralPrologue.solutions('projection'));app.api.getWorld().rebuild();await app.api.run();
 assert.deepEqual([...phases],['enter','compute','gather']);
 for(const positions of paths.values())assert.ok(positions.every((p,i)=>!i||p.distanceTo(positions[i-1])<.5),'no jump across the module');
 assert.deepEqual([...contributions.get('0')].sort(),['0:0','0:2','0:3']);
 assert.deepEqual([...contributions.get('1')].sort(),['1:1','1:2','1:3']);
 assert.equal(contributions.get('2').size,6);
});

test('two-input processing waits for both branches to reach its ports',async()=>{
 let app,arrived=new Set(),waiting=0,processed=0,lastRow;
 app=await boot({reduced:false,frameMs:10,onFrame:({document})=>{
  const proc=document.querySelector('.processing-packet[data-node="multiply"]');if(!proc)return;
  if(proc.dataset.row!==lastRow){lastRow=proc.dataset.row;arrived=new Set();}
  const n=app.api.getState().nodes.find(n=>n.id==='multiply'),scene=app.scenes.at(-1);
  for(const el of document.querySelectorAll('.packet[data-motion="wire"][data-to="multiply"]')){
   if(el.hidden)continue;const mesh=scene.children.find(o=>o.userData.packet&&o.userData.from===el.dataset.from&&o.userData.to==='multiply'),index=n.links.indexOf(el.dataset.from);
   if(Math.hypot(mesh.position.x-(n.x-1.12),mesh.position.z-(n.z+(index? .55:-.55)))<.2){arrived.add(el.dataset.from);if(proc.hidden)waiting++;}
  }
  if(!proc.hidden){processed++;assert.equal(arrived.size,2,'both inputs arrive before processing');}
 }});
 app.api.switchLevel('gating');app.api.setState(app.c.NeuralPrologue.solutions('gating'));app.api.getWorld().rebuild();await app.api.run();
 assert.ok(waiting>10);assert.ok(processed>0);
});

test('elementwise multiplication pairs both real inputs and keeps each product until the output vector gathers',async()=>{
 let app;const products=new Map(),trails=new Map(),counts=new Set(),cases=new Set();let gathering=0;
 app=await boot({reduced:false,frameMs:40,onFrame:({document})=>{
  if(!app)return;const label=document.querySelector('.processing-packet[data-node="multiply"]');if(!label||label.hidden)return;
  const sample=Number.parseInt(document.querySelector('#pass-count').textContent),M=app.c.NeuralPrologue,ev=M.evaluate(app.api.getState(),M.samples[sample].v);
  const scene=app.scenes.at(-1),unit=scene.children.flatMap(o=>o.children).find(o=>o.userData.node==='multiply'),meshes=[scene.children.find(o=>o.userData.processing==='multiply'),scene.children.find(o=>o.userData.processingSecondary==='multiply')];
  const inputs=[ev.values.silu,ev.values.project],calc=document.querySelector('.projection-calculation[data-node="multiply"]'),out=scene.children.find(o=>o.userData.processingOutput==='multiply');
  for(const [slot,mesh]of meshes.entries()){
   if(!mesh.visible)continue;
   for(const {bar}of mesh.userData.featureRoutes){
    const i=bar.userData.feature;assert.equal(bar.userData.value,inputs[slot][i],'each colored input carries the value from its own wire');
    if(bar.material.opacity<=0)continue;
    const pos=bar.getWorldPosition(new app.c.T.Vector3()),key=sample+':'+slot+':'+i;
    if(!trails.has(key))trails.set(key,[]);trails.get(key).push(pos);
    if(bar.userData.mappingProgress===1){
     assert.ok(Math.abs(pos.z-(unit.position.z+(i-1.5)*.43))<1e-9,'same-index inputs share one channel');
     assert.equal(bar.userData.mappedInput,i);
     const partner=meshes[1-slot].userData.featureRoutes[i].bar;
     assert.notEqual(bar.material.color.getHex(),partner.material.color.getHex(),'the two inputs retain distinct colors');
     if(bar.userData.pairProgress>.99)assert.ok(pos.distanceTo(partner.getWorldPosition(new app.c.T.Vector3()))<.015,'the two factors meet at the same multiplication ring');
    }
   }
  }
  unit.traverse(o=>{
   if(o.userData.multiplyFeature==null||!o.visible)return;const d=o.userData,i=d.multiplyFeature;
   assert.equal(d.inputA,inputs[0][i]);assert.equal(d.inputB,inputs[1][i]);assert.equal(d.value,d.inputA*d.inputB);
   products.set(sample+':'+i,d.value);
   if(d.value===0){cases.add('zero');assert.ok(o.material.wireframe);assert.ok(o.scale.x>=.6,'a zero still has a visible hollow carrier');}
   if(d.value<0)cases.add('negative');if(d.inputA<0&&d.inputB<0){cases.add('negative times negative');assert.ok(d.value>0);}
   if(Math.abs(d.inputA)>1||Math.abs(d.inputB)>1)cases.add('not binary');
  });
  if(!calc.hidden){
   const i=Number(calc.dataset.column),fmt=v=>Number(v.toFixed(2)).toString(),factor=v=>v<0?'('+fmt(v)+')':fmt(v),values=[inputs[0][i],inputs[1][i],ev.values.multiply[i]];
   const rounded=values.some(v=>Math.abs(v-Number(v.toFixed(2)))>1e-9);
   assert.equal(calc.querySelector('.projection-equation').textContent,factor(values[0])+' × '+factor(values[1])+(rounded?' ≈ ':' = ')+fmt(values[2]));
   assert.equal(calc.querySelector('.projection-sum-symbol').textContent,'×');
   assert.equal(document.querySelectorAll('.projection-result[data-node="multiply"]:not([hidden])').length,i,'only already-computed products remain visible');
   if(Number(calc.style.getPropertyValue('--calculation-focus'))>.9){
    assert.equal(unit.userData.activePair,i);
    const sockets=unit.children.filter(o=>o.geometry?.type==='SphereGeometry'&&o.position.x===.68&&o.userData.multiplyFeature==null);
    assert.ok(sockets[i].material.emissiveIntensity>.8);assert.ok(sockets.filter((_,j)=>i!==j).every(o=>o.material.emissiveIntensity<.1));
    for(const selector of ['[data-input="multiply:0"]','[data-input="multiply:1"]','[data-output="multiply"]'])assert.ok(Number(document.querySelector(selector).style.opacity)<.12,'dimension badges do not cover the calculation');
   }
  }
  if(out.visible){
   const bars=out.children.filter(o=>o.visible&&o.userData.feature!=null);counts.add(bars.length);
   if(out.userData.gather>0){gathering++;assert.equal(bars.length,4,'wait for all products before transporting the vector');}
   for(const bar of bars){const i=bar.userData.feature;assert.equal(bar.userData.value,ev.values.multiply[i]);if(out.userData.gather===0)assert.ok(Math.abs(bar.getWorldPosition(new app.c.T.Vector3()).z-(unit.position.z+(i-1.5)*.43))<1e-9);}
  }
 }});
 app.api.switchLevel('gating');app.api.setState(app.c.NeuralPrologue.solutions('gating'));app.api.getWorld().rebuild();await app.api.run();
 assert.equal(products.size,20);for(const kind of ['zero','negative','negative times negative','not binary'])assert.ok(cases.has(kind),kind);
 assert.deepEqual([...counts].filter(Boolean).sort(),[1,2,3,4]);assert.ok(gathering>0);
 for(const trail of trails.values())assert.ok(trail.every((p,i)=>!i||p.distanceTo(trail[i-1])<.3),'factor carriers never teleport between sockets');
 assert.equal(app.document.querySelectorAll('#test-results .pass').length,5);
 assert.equal(app.document.querySelectorAll('.packet,.projection-calculation,.projection-result').length,0);
 app.scenes.at(-1).traverse(o=>{if(o.userData.multiplyFeature!=null)assert.equal(o.visible,false);if(o.userData.node==='multiply')assert.equal(o.userData.activePair,null);assert.equal(o.userData.processingSecondary,undefined,'the extra input tray is removed at completion');});
});

test('residual addition joins both actual inputs, shows each sum and gathers the completed vector',async()=>{
 let app;const sums=new Map(),cases=new Set(),arrivals=new Map(),trails=new Map(),counts=new Set();let gathering=0;
 app=await boot({reduced:false,frameMs:50,onFrame:({document})=>{
  if(!app)return;const label=document.querySelector('.processing-packet[data-node="add"]');if(!label)return;
  const scene=app.scenes.at(-1),unit=scene.children.flatMap(o=>o.children).find(o=>o.userData.node==='add'),node=app.api.getState().nodes.find(n=>n.id==='add');
  if(!unit)return;
  const sample=label.dataset.row==='validation'?Number(document.querySelector('#pass-count').textContent.split('/')[0]):Number(label.dataset.row);
  for(const el of document.querySelectorAll('.packet[data-motion="wire"][data-to="add"]')){
   if(el.hidden)continue;const slot=node.links.indexOf(el.dataset.from),mesh=scene.children.find(o=>o.userData.packet&&o.userData.from===el.dataset.from&&o.userData.to==='add');
   const port=new app.c.T.Vector3(-1.12,.68,slot?.55:-.55).add(unit.position);
   if(mesh.position.distanceTo(port)<.2){if(!arrivals.has(sample))arrivals.set(sample,new Set());arrivals.get(sample).add(slot);}
  }
  if(label.hidden)return;assert.equal(arrivals.get(sample)?.size,2,'both incoming vectors reach their own ports before addition');
  const M=app.c.NeuralPrologue,ev=M.evaluate(app.api.getState(),M.samples[sample].v),inputs=node.links.map(id=>ev.values[id]);
  const meshes=[scene.children.find(o=>o.userData.processing==='add'),scene.children.find(o=>o.userData.processingSecondary==='add')],calc=document.querySelector('.projection-calculation[data-node="add"]'),out=scene.children.find(o=>o.userData.processingOutput==='add');
  for(const [slot,mesh]of meshes.entries()){
   if(!mesh.visible)continue;
   for(const {bar}of mesh.userData.featureRoutes){
    const i=bar.userData.feature;assert.equal(bar.userData.value,inputs[slot][i]);if(bar.material.opacity<=0)continue;
    const pos=bar.getWorldPosition(new app.c.T.Vector3()),key=sample+':'+slot+':'+i;
    if(!trails.has(key))trails.set(key,[]);trails.get(key).push(pos);
    if(bar.userData.mappingProgress===1){
     assert.ok(Math.abs(pos.z-(unit.position.z+(i-.5)*.65))<1e-9);
     const partner=meshes[1-slot].userData.featureRoutes[i].bar;assert.notEqual(bar.material.color.getHex(),partner.material.color.getHex());
     if(bar.userData.pairProgress>.99)assert.ok(pos.distanceTo(partner.getWorldPosition(new app.c.T.Vector3()))<.015,'matching addends meet at the same + ring');
    }
   }
  }
  unit.traverse(o=>{
   if(o.userData.additionFeature==null||!o.visible)return;const d=o.userData,i=d.additionFeature;
   assert.equal(d.operation,'add');assert.equal(d.inputA,inputs[0][i]);assert.equal(d.inputB,inputs[1][i]);assert.equal(d.value,d.inputA+d.inputB);
   sums.set(sample+':'+i,d.value);if(d.inputB<0)cases.add('decrease');if(d.inputB>0)cases.add('increase');if(d.inputA===0)cases.add('zero baseline');
  });
  if(!calc.hidden){
   const i=Number(calc.dataset.column),fmt=v=>Number(v.toFixed(2)).toString(),term=v=>v<0?'('+fmt(v)+')':fmt(v),values=[inputs[0][i],inputs[1][i],ev.values.add[i]],rounded=values.some(v=>Math.abs(v-Number(v.toFixed(2)))>1e-9);
   assert.equal(calc.dataset.operation,'add');assert.equal(calc.querySelector('.projection-sum-symbol').textContent,'+');
   assert.equal(calc.querySelector('.projection-equation').textContent,term(values[0])+' + '+term(values[1])+(rounded?' ≈ ':' = ')+fmt(values[2]));
   assert.equal(document.querySelectorAll('.projection-result[data-node="add"]:not([hidden])').length,i);
   if(Number(calc.style.getPropertyValue('--calculation-focus'))>.9){
    assert.equal(unit.userData.activePair,i);
    for(const selector of ['[data-input="add:0"]','[data-input="add:1"]','[data-output="add"]'])assert.ok(Number(document.querySelector(selector).style.opacity)<.12);
   }
  }
  if(out.visible){
   const bars=out.children.filter(o=>o.visible&&o.userData.feature!=null);counts.add(bars.length);
   if(out.userData.gather>0){gathering++;assert.equal(bars.length,2,'both sums are ready before the outgoing vector gathers');}
   for(const bar of bars){assert.equal(bar.userData.value,ev.values.add[bar.userData.feature]);if(out.userData.gather===0)assert.ok(Math.abs(bar.getWorldPosition(new app.c.T.Vector3()).z-(unit.position.z+(bar.userData.feature-.5)*.65))<1e-9);}
  }
 }});
 // Include a negative FFN update so the addition animation covers both signs.
 app.c.NeuralPrologue.samples[4].v=[-1,-2];
 app.api.switchLevel('residual');app.api.setState(app.c.NeuralPrologue.solutions('residual'));app.api.getWorld().rebuild();await app.api.run();
 assert.equal(sums.size,10);assert.deepEqual([...cases].sort(),['decrease','increase','zero baseline']);assert.deepEqual([...counts].filter(Boolean).sort(),[1,2]);assert.ok(gathering>0);
 for(const trail of trails.values())assert.ok(trail.every((p,i)=>!i||p.distanceTo(trail[i-1])<.4),'addends move continuously into their sockets');
 assert.equal(app.document.querySelectorAll('#test-results .pass').length,5);assert.equal(app.document.querySelectorAll('.packet,.projection-calculation,.projection-result').length,0);
 app.scenes.at(-1).traverse(o=>{if(o.userData.additionFeature!=null)assert.equal(o.visible,false);if(o.userData.node==='add')assert.equal(o.userData.activePair,null);assert.equal(o.userData.processingSecondary,undefined);});
});

for(const [stage,kind,featureKey,inputKey,count,glyph]of [['gating','multiply','multiplyFeature','multiplyInput',4,'×'],['residual','add','additionFeature','additionInput',2,'+']]){
test(`every ${kind} pair uses the same pace and obeys the speed slider`,async()=>{
 let app,runIndex=0;const runs=[new Map(),new Map()],departures=[[],[]];
 app=await boot({reduced:false,frameMs:20,onFrame:({document,now})=>{
  if(!app)return;const label=document.querySelector(`.processing-packet[data-node="${kind}"][data-row="2"]`),wire=document.querySelector(`.packet[data-motion="wire"][data-from="${kind}"][data-row="2"]`);
  if(wire&&!wire.hidden)departures[runIndex].push(now);
  if(!label||label.hidden)return;
  app.scenes.at(-1).traverse(o=>{if(!o.visible||o.userData[featureKey]==null)return;const i=o.userData[featureKey];if(!runs[runIndex].has(i))runs[runIndex].set(i,[]);runs[runIndex].get(i).push(now);});
 }});
 app.api.switchLevel(stage);app.api.setState(app.c.NeuralPrologue.solutions(stage));app.api.getWorld().rebuild();
 for(const rate of [.25,.5]){
  const slider=app.document.querySelector('#playback-speed');slider.value=String(rate);slider.dispatchEvent(new app.window.Event('input',{bubbles:true}));await app.api.run();
  const spans=runs[runIndex];assert.equal(spans.size,count);const widths=[...spans.values()].map(a=>a.at(-1)-a[0]);assert.ok(Math.max(...widths)-Math.min(...widths)<=20);
  for(let i=1;i<count;i++)assert.ok(spans.get(i-1).at(-1)<spans.get(i)[0]);
  assert.ok(departures[runIndex][0]>spans.get(count-1).at(-1));runIndex++;
 }
 for(let i=0;i<count;i++){const width=r=>runs[r].get(i).at(-1)-runs[r].get(i)[0];assert.ok(Math.abs(width(0)-2*width(1))<=40);}
});

test(`paused ${kind} keeps inputs, results and formulas stable across camera and language changes`,async()=>{
 let app,phase=0,frames=0,snapshot,restart;
 const visual=()=>{const data=[];app.scenes.at(-1).traverse(o=>{if(o.userData[featureKey]!=null||o.userData[inputKey]!=null)data.push([o.visible,o.position.toArray(),o.scale.toArray(),o.material.opacity,o.material.emissiveIntensity]);});return JSON.stringify(data);};
 const wheel=deltaY=>{const e=new app.window.Event('wheel',{bubbles:true,cancelable:true});e.deltaY=deltaY;app.document.querySelector('#scene').dispatchEvent(e);};
 app=await boot({reduced:false,frameMs:40,onFrame:({document})=>{
  if(!app)return;const calc=document.querySelector(`.projection-calculation[data-node="${kind}"]`);if(!calc||calc.hidden)return;
  if(!phase&&calc.dataset.column==='1'&&Number(calc.querySelector('b').style.opacity)>.1){phase=1;app.api.togglePause();snapshot=visual();return;}
  if(phase!==1)return;assert.equal(visual(),snapshot);frames++;
  if(frames===2){const scale=Number(calc.style.getPropertyValue('--calculation-scale'));wheel(2000);assert.equal(calc.dataset.detail,'abstract');assert.ok(Number(calc.style.getPropertyValue('--calculation-scale'))<scale);assert.equal(calc.querySelector('.projection-sum-symbol').textContent,glyph);}
  if(frames===3){wheel(-2000);assert.equal(calc.dataset.detail,'full');app.c.I18n.setPreference('en');assert.equal(visual(),snapshot,'language rebuild restores the same pair and product');}
  if(frames===5){
   const saved=JSON.stringify(app.api.getState());app.api.run();assert.equal(app.document.querySelectorAll('.packet,.projection-calculation,.projection-result').length,0);
   app.scenes.at(-1).traverse(o=>{if(o.userData[featureKey]!=null)assert.equal(o.visible,false);assert.equal(o.userData.processingSecondary,undefined);});
   assert.equal(JSON.stringify(app.api.getState()),saved);phase=2;restart=app.api.run();
  }
 }});
 app.api.switchLevel(stage);app.api.setState(app.c.NeuralPrologue.solutions(stage));app.api.getWorld().rebuild();await app.api.run();await restart;
 assert.equal(phase,2);assert.equal(app.document.querySelectorAll('#test-results .pass').length,5);
});
}

test('duplicated multiplication modules animate their actual wiring even in an incorrect circuit',async()=>{
 let app;const seen=new Set();
 app=await boot({reduced:false,frameMs:80,onFrame:({document})=>{
  if(!app)return;
  for(const id of ['multiply','multiply-2']){
   const label=document.querySelector(`.processing-packet[data-node="${id}"]`);if(!label||label.hidden)continue;
   const M=app.c.NeuralPrologue,state=app.api.getState(),node=state.nodes.find(n=>n.id===id),ev=M.evaluate(state,M.samples[0].v),calc=document.querySelector(`.projection-calculation[data-node="${id}"]`);
   assert.ok(calc,'each operator instance has its own two-factor equation');
   app.scenes.at(-1).children.flatMap(o=>o.children).find(o=>o.userData.node===id).traverse(o=>{if(o.userData.multiplyFeature==null||!o.visible)return;const i=o.userData.multiplyFeature;assert.equal(o.userData.inputA,ev.values[node.links[0]][i]);assert.equal(o.userData.inputB,ev.values[node.links[1]][i]);assert.equal(o.userData.value,ev.values[id][i]);seen.add(id);});
  }
 }});
 app.api.switchLevel('ffn');const M=app.c.NeuralPrologue,s=M.initial('ffn');
 const coords=[[-4.7,-2],[-1.4,-2],[1.9,-2],[5.2,-2],[-4.7,2]];
 ['project','gate','multiply','multiply','down'].forEach((kind,i)=>assert.ok(M.addNode(s,kind,...coords[i])));
 for(const edge of [['R','project',0],['R','gate',0],['gate','multiply',0],['project','multiply',1],['multiply','multiply-2',0],['gate','multiply-2',1],['multiply-2','down',0],['down','sink',0]])assert.equal(M.connect(s,...edge),null);
 app.api.setState(s);app.api.getWorld().rebuild();await app.api.run();assert.deepEqual([...seen],['multiply','multiply-2']);assert.equal(app.document.querySelectorAll('#test-results .fail').length,1);
});

test('duplicate additions show their actual addends and a visible zero sum in an incorrect circuit',async()=>{
 let app;const seen=new Set(),zeros=new Set();
 app=await boot({reduced:false,frameMs:80,onFrame:({document})=>{
  if(!app)return;const M=app.c.NeuralPrologue,state=app.api.getState();
  for(const id of ['add','add-2']){
   const label=document.querySelector(`.processing-packet[data-node="${id}"]`);if(!label||label.hidden)continue;
   const node=state.nodes.find(n=>n.id===id),ev=M.evaluate(state,M.samples[0].v),calc=document.querySelector(`.projection-calculation[data-node="${id}"]`);
   assert.equal(calc.dataset.operation,'add');
   app.scenes.at(-1).children.flatMap(o=>o.children).find(o=>o.userData.node===id).traverse(o=>{
    if(o.userData.additionFeature==null||!o.visible)return;const d=o.userData,i=d.additionFeature;
    assert.equal(d.inputA,ev.values[node.links[0]][i]);assert.equal(d.inputB,ev.values[node.links[1]][i]);assert.equal(d.value,ev.values[id][i]);seen.add(id);
    if(d.value===0){assert.ok(o.material.wireframe);assert.ok(o.scale.x>=.6);zeros.add(id);}
   });
  }
 }});
 app.api.switchLevel('ffn');const M=app.c.NeuralPrologue,state=M.initial('ffn');
 assert.ok(M.addNode(state,'add',-4.7,-2));assert.ok(M.addNode(state,'add',-1.4,-2));
 for(const edge of [['R','add',0],['Z','add',1],['add','add-2',0],['R','add-2',1],['add-2','sink',0]])assert.equal(M.connect(state,...edge),null);
 app.api.setState(state);app.api.getWorld().rebuild();await app.api.run();
 assert.deepEqual([...seen],['add','add-2']);assert.deepEqual([...zeros],['add','add-2']);assert.equal(app.document.querySelectorAll('#test-results .fail').length,1);
});
test('stopping and immediately restarting during processing clears effects without cancelling the new run',async()=>{
 let app,restarted=false,restart;
 app=await boot({reduced:false,frameMs:20,onFrame:({document})=>{
  const proc=document.querySelector('.processing-packet');
  if(!restarted&&proc&&!proc.hidden&&proc.dataset.phase==='compute'){
   restarted=true;app.api.run();assert.equal(document.querySelectorAll('.packet').length,0);
   app.scenes.at(-1).traverse(o=>{if(o.userData.contribution)assert.equal(o.visible,false);});
   restart=app.api.run();
  }
 }});
 app.api.switchLevel('projection');app.api.setState(app.c.NeuralPrologue.solutions('projection'));app.api.getWorld().rebuild();await app.api.run();await restart;
 assert.ok(restarted);assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);assert.equal(app.document.querySelectorAll('.packet').length,0);
});

test('English scene labels and module cards work, and live language switching keeps a moving run valid',async()=>{
 let app,changed=false;
 app=await boot({reduced:false,frameMs:80,languages:['en-US'],onFrame:({document})=>{
  if(app&&!changed&&document.querySelector('.packet')){changed=true;app.c.I18n.setPreference('zh');app.c.I18n.setPreference('en');}
 }});
 const M=app.c.NeuralPrologue;
 for(const id of Object.keys(M.levels)){
  // Keep a fine-grained live language switch in the first lesson. The remaining
  // sweep checks translated labels/cards; dedicated scene tests cover pacing.
  app.setFrameMs(id==='wire'?80:2500);
  app.api.switchLevel(id);const s=M.solutions(id);if(M.levels[id].engine)s.settings.precision=16;app.api.setState(s);app.api.getWorld().rebuild();await app.api.run();
  for(const n of s.nodes){app.api.getWorld().focus(n.id);assert.doesNotMatch(app.document.querySelector('#card-meaning').textContent,/[\u3400-\u9fff]/);app.document.querySelector('#card-close').onclick();}
  assert.doesNotMatch(app.document.querySelector('#labels').textContent,/[\u3400-\u9fff]/);
 }
 assert.ok(changed);assert.deepEqual([...app.c.I18n.missing],[]);assert.equal(app.document.querySelectorAll('.packet').length,0);
});

test('down projection routes four feature blocks to matching weight rows and explains each output',async()=>{
 let app;const rows=new Set(),intervals=new Map(),trails=new Map(),formulas=new Set();
 app=await boot({reduced:false,frameMs:40,onFrame:({document,now})=>{
  if(!app)return;const label=document.querySelector('.processing-packet[data-node="down"]');if(!label||label.hidden)return;
  const scene=app.scenes.at(-1),mesh=scene.children.find(o=>o.userData.processing==='down'),unit=scene.children.flatMap(o=>o.children).find(o=>o.userData.node==='down');
  const bars=mesh.children.filter(o=>o.userData.feature!=null);
  const row=label.dataset.row;
  for(const bar of bars){
   const id=document.querySelector('#pass-count').textContent+':'+bar.userData.feature,pos=bar.getWorldPosition(new app.c.T.Vector3());
   if(!trails.has(id))trails.set(id,[]);trails.get(id).push(pos);
   if(bar.userData.mappingProgress!==1||!mesh.visible)continue;
   rows.add(row);const i=bar.userData.feature,expectedZ=unit.position.z+(i-1.5)*1.5/4;
   assert.ok(Math.abs(pos.z-expectedZ)<1e-9,'each coefficient aligns with the matching weight row');
   assert.ok(Math.abs(pos.x-(unit.position.x-1.12))<1e-9,'all four coefficients face their own row');
   assert.equal(bar.userData.mappedInput,i);
   if(row==='0')assert.equal(bar.userData.value,[1,0,1,.5][i]);
  }
  if(row!=='0')return;
  const calc=document.querySelector('.projection-calculation[data-node="down"]');
  if(calc&&!calc.hidden){const column=Number(calc.dataset.column);assert.equal(calc.querySelector('.projection-equation').textContent,['1 × 0.4 + 0 × 0 + 1 × 0.2 + 0.5 × (-0.1) = 0.55','1 × 0 + 0 × 0.3 + 1 × 0.2 + 0.5 × 0.3 = 0.35'][column]);formulas.add(column);}
  unit.traverse(o=>{if(!o.userData.contribution||!o.visible)return;const d=o.userData;
   assert.equal(d.value,[1,0,1,.5][d.input]*app.c.NeuralPrologue.weights.down[d.input][d.output]);
   assert.notEqual(d.value,0,'zero values still align but do not send a nonzero contribution');
   if(!intervals.has(d.output))intervals.set(d.output,[]);intervals.get(d.output).push(now);
  });
 }});
 app.api.switchLevel('original-ffn');app.api.setState(app.c.NeuralPrologue.solutions('original-ffn'));app.api.getWorld().rebuild();await app.api.run();
 assert.equal(formulas.size,2);assert.equal(rows.size,4,'includes three named samples and the extra validation samples');
 for(const points of trails.values())assert.ok(points.every((p,i)=>!i||p.distanceTo(points[i-1])<.35),'components fan out without teleporting');
 assert.ok(Math.max(...intervals.get(0))<Math.min(...intervals.get(1)),'first output is explained before the others');
 const duration=i=>Math.max(...intervals.get(i))-Math.min(...intervals.get(i));assert.ok(Math.abs(duration(0)-duration(1))<=40,'both outputs use the same travel time');
 assert.equal(app.document.querySelectorAll('.projection-calculation,.packet').length,0);
 assert.equal(app.document.querySelectorAll('#test-results .pass').length,5);
});

test('projection keeps computed values by their own output spheres before gathering, with zoomable formulas',async()=>{
 let app,zoomChecked=false,zoomPhase=0;const columns=new Map(),counts=new Set(),trails=new Map();let gathered=0;
 const wheel=deltaY=>{const e=new app.window.Event('wheel',{bubbles:true,cancelable:true});e.deltaY=deltaY;app.document.querySelector('#scene').dispatchEvent(e);};
 app=await boot({reduced:false,frameMs:40,onFrame:({document})=>{
  if(!app)return;
  const calc=document.querySelector('.projection-calculation[data-node="project"]'),out=app.scenes.at(-1).children.find(o=>o.userData.processingOutput==='project');
  if(!calc||!out)return;
  const unit=app.api.getState().nodes.find(n=>n.id==='project'),labels=[...document.querySelectorAll('.projection-result[data-node="project"]')];
  if(!calc.hidden&&out.userData.row==='0'){
   const column=Number(calc.dataset.column);columns.set(column,parseFloat(calc.style.top));
   assert.equal(labels.filter(el=>!el.hidden).length,column,'previous results persist; future results are not revealed');
   assert.ok(Number(document.querySelector('[data-output="project"]').style.opacity)<.12,'the output dimension badge stays transparent while computing');
   if(column===2&&!zoomChecked){
    zoomChecked=true;zoomPhase=1;app.api.togglePause();const scale=Number(calc.style.getPropertyValue('--calculation-scale'));
    wheel(2000);assert.equal(calc.dataset.detail,'abstract');assert.ok(Number(calc.style.getPropertyValue('--calculation-scale'))<scale);
    assert.ok(labels.every(el=>el.style.visibility==='hidden'),'abstract view retains the 3D result blocks instead of tiny digits');
   }else if(zoomPhase===1){
    zoomPhase=2;wheel(-2000);assert.equal(calc.dataset.detail,'full');assert.ok(Math.abs(Number(calc.style.getPropertyValue('--calculation-scale'))-1)<1e-9,'returning to 100% restores the normal formula size');app.api.togglePause();
   }
  }
  if(!out.visible)return;
  const bars=out.children.filter(o=>o.userData.feature!=null),visible=bars.filter(o=>o.visible),gather=out.userData.gather;
  counts.add(visible.length);
  if(gather>0){gathered++;assert.equal(visible.length,4,'gather starts only after all four results exist');}
  for(const bar of visible){
   const j=bar.userData.feature,pos=bar.getWorldPosition(new app.c.T.Vector3()),key=out.userData.row+':'+j;
   if(!trails.has(key))trails.set(key,[]);trails.get(key).push(pos);
   if(gather===0){assert.ok(Math.abs(pos.x-(unit.x+.98))<1e-9);assert.ok(Math.abs(pos.z-(unit.z+(j-1.5)*.4))<1e-9,'computed value stays at its corresponding sphere');}
   if(out.userData.row==='0')assert.equal(labels[j].textContent,String([1,0,1,.5][j]));
  }
 }});
 app.api.switchLevel('projection');app.api.setState(app.c.NeuralPrologue.solutions('projection'));app.api.getWorld().rebuild();await app.api.run();
 assert.equal(columns.size,4);assert.equal(new Set(columns.values()).size,4,'the equation follows each sphere, not a fixed panel below');
 assert.deepEqual([...counts].sort(),[1,2,3,4]);assert.ok(gathered>0);assert.equal(zoomPhase,2);
 for(const trail of trails.values())assert.ok(trail.every((v,i)=>!i||v.distanceTo(trail[i-1])<.3),'registered results move continuously into the outgoing vector');
 assert.equal(app.document.querySelectorAll('.projection-result,.projection-calculation,.packet').length,0);
 assert.equal(Number(app.document.querySelector('[data-output="project"]').style.opacity||1),1);
});

test('pause freezes processing while camera and speed controls stay live; resume never catches up wall time',async()=>{
 let app,phase='waiting',snapshot,frames=0,resumedPosition;
 const event=(target,type,properties={})=>{const e=new app.window.Event(type,{bubbles:true,cancelable:true});Object.assign(e,properties);target.dispatchEvent(e);};
 const visual=()=>{const data=[];app.scenes.at(-1).traverse(o=>{if(o.userData.feature!=null||o.userData.contribution)data.push([o.visible,o.position.toArray(),o.scale.toArray(),o.material.opacity]);});return JSON.stringify(data);};
 app=await boot({reduced:false,frameMs:40,onFrame:({document})=>{
  if(!app)return;const label=document.querySelector('.processing-packet');if(!label||label.hidden)return;
  const mesh=app.scenes.at(-1).children.find(o=>o.userData.processing==='project');
  if(phase==='waiting'&&mesh.userData.featureRoutes?.[0].bar.userData.mappingProgress===1){
   phase='paused';document.querySelector('#pause').onclick();assert.equal(document.querySelector('#pause').textContent,'▶ 继续');snapshot=visual();return;
  }
  if(phase==='paused'){
   assert.equal(visual(),snapshot,'animation positions, scale and opacity remain frozen');frames++;
   if(frames===3){
    const before=app.view(),circuit=JSON.stringify(app.api.getState());
    event(document.querySelector('#scene'),'pointerdown',{clientX:500,clientY:300});event(document,'pointermove',{clientX:600,clientY:340});event(document,'pointerup');event(label,'wheel',{deltaY:-100});
    assert.notDeepEqual(app.view().position,before.position);assert.ok(app.view().width<before.width);
    app.api.chooseOutput('R');app.api.chooseInput('sink',0);assert.equal(JSON.stringify(app.api.getState()),circuit);
    const slider=document.querySelector('#playback-speed');slider.value='.5';event(slider,'input');
    assert.equal(visual(),snapshot,'camera and settings cannot advance processing');
   }
   if(frames===6){app.c.I18n.setPreference('en');assert.equal(document.querySelector('#pause').textContent,'▶ Resume');assert.equal(visual(),snapshot,'changing language preserves paused effects');}
   if(frames===15){resumedPosition=mesh.position.clone();document.querySelector('#pause').onclick();phase='resuming';}
  }else if(phase==='resuming'){
   assert.ok(mesh.position.distanceTo(resumedPosition)<.1,'resume does not include the paused wall-clock interval');phase='done';
  }
 }});
 app.document.querySelector('#welcome').close();app.api.switchLevel('projection');app.api.setState(app.c.NeuralPrologue.solutions('projection'));app.api.getWorld().rebuild();await app.api.run();
 assert.equal(phase,'done');assert.equal(app.document.querySelector('#pause').disabled,true);assert.equal(app.document.querySelector('#pause').getAttribute('aria-pressed'),'false');assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);
});

test('stopping a paused run clears its effects and the next run starts unpaused',async()=>{
 let app,restart,stopped=false;
 app=await boot({reduced:false,frameMs:80,onFrame:({document})=>{
  const calc=document.querySelector('.projection-calculation:not([hidden])');if(!app||stopped||!calc)return;
  stopped=true;app.api.togglePause();app.api.run();
  assert.equal(document.querySelector('#pause').disabled,true);assert.equal(document.querySelectorAll('.packet,.projection-calculation,.projection-result').length,0);
  restart=app.api.run();assert.equal(document.querySelector('#pause').getAttribute('aria-pressed'),'false');
 }});
 app.api.switchLevel('projection');app.api.setState(app.c.NeuralPrologue.solutions('projection'));app.api.getWorld().rebuild();await app.api.run();await restart;
 assert.ok(stopped);assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);
});

test('dragging a module card previews the operator, places one copy and dismisses the tutorial guide',async()=>{
 for(const left of [0,210]){
 const app=await boot(),doc=app.document;doc.querySelector('#welcome').close();app.api.switchLevel('workshop');
 const scene=()=>app.scenes.at(-1),ghost=()=>scene().children.find(o=>o.userData.placementGhost),guide=()=>doc.querySelector('.placement-guide');
 const event=(target,type,properties={})=>{const e=new app.window.Event(type,{bubbles:true,cancelable:true});Object.assign(e,properties);target.dispatchEvent(e);};
 doc.querySelector('#scene').getBoundingClientRect=()=>({left,top:0,width:1400,height:760,right:left+1400,bottom:760});
 assert.equal(guide().hasAttribute('hidden'),false);const card=doc.querySelector('[data-add-module="relu-ffn"]');
 event(card,'pointerdown',{clientX:120,clientY:160});event(doc,'pointermove',{clientX:left+700,clientY:380});
 assert.ok(ghost());assert.ok(ghost().children.length>10,'preview is the packaged FFN module, not a placeholder cube');assert.equal(app.api.getState().nodes.length,0);
 event(doc,'pointerup',{clientX:left+700,clientY:380});card.dispatchEvent(new app.window.Event('click',{bubbles:true}));
 assert.equal(ghost(),undefined);assert.equal(app.api.getState().nodes.length,1);assert.equal(guide().hasAttribute('hidden'),true);assert.equal(doc.querySelector('#module-count').textContent,'1 / 1');
 app.api.chooseOutput('R');app.api.chooseInput('relu-ffn',0);app.api.chooseOutput('relu-ffn');app.api.chooseInput('sink',0);await app.api.run();assert.equal(doc.querySelectorAll('#test-results .pass').length,5);
 app.document.querySelector('#reset').onclick();assert.equal(guide().hasAttribute('hidden'),false);
 event(doc.querySelector('[data-add-module="relu-ffn"]'),'pointerdown',{clientX:120,clientY:160});event(doc,'pointermove',{clientX:left+700,clientY:380});event(doc,'pointerup',{clientX:left+1500,clientY:380});assert.equal(ghost(),undefined);assert.equal(app.api.getState().nodes.length,0,'dropping outside does not create or spend a copy');
 if(left){
  event(doc.querySelector('[data-add-module="relu-ffn"]'),'pointerdown',{clientX:120,clientY:160});event(doc,'pointermove',{clientX:180,clientY:380});event(doc,'pointerup',{clientX:180,clientY:380});
  assert.equal(app.api.getState().nodes.length,0,'dropping in the reserved shelf space does not create a component');
 }
 }
});

function noticeTimers(){
 let now=0,id=0;const tasks=new Map(),callbacks=[];
 return {callbacks,setTimeout(fn,delay){callbacks.push(fn);tasks.set(++id,{fn,at:now+delay});return id;},clearTimeout(id){tasks.delete(id);},
  tick(ms){now+=ms;for(const [id,t]of [...tasks])if(t.at<=now){tasks.delete(id);t.fn();}},get pending(){return tasks.size;}};
}
test('named sublayer cards and rejected connections explain their roles in both languages',async()=>{
 const timers=noticeTimers(),app=await boot({timers}),doc=app.document,M=app.c.NeuralPrologue;
 doc.querySelector('#welcome').close();app.api.switchLevel('decoder-layer');app.api.setState(M.solutions('decoder-layer'));app.api.getWorld().rebuild();
 const saved=JSON.stringify(app.api.getState());
 for(const language of ['zh','en']){
  app.c.I18n.setPreference(language);
  for(const kind of ['ln-attn','ln-cross','ln-ffn','encoder-add','cross-add','ffn-add']){
   app.api.getWorld().focusShelf(kind);assert.equal(doc.querySelector('#card-purpose').textContent,M.componentUse(kind));
   if(language==='en')assert.doesNotMatch(doc.querySelector('#card-purpose').textContent,/[\u3400-\u9fff]/);
   doc.querySelector('#card-close').onclick();
  }
  app.api.chooseOutput('masked-heads');app.api.chooseInput('ln-ffn',0);
  const toast=doc.querySelector('.connection-toast');assert.ok(toast);assert.match(toast.textContent,language==='en'?/Module role mismatch.*FFN/s:/组件用途不匹配.*FFN/s);
  assert.equal(JSON.stringify(app.api.getState()),saved);timers.tick(3600);assert.equal(doc.querySelector('.connection-toast'),null);
 }
 assert.deepEqual([...app.c.I18n.missing],[]);
});
test('a borrowed sublayer normalizer animates before its dismissible local role verdict',async()=>{
 let app,frames=0;
 app=await boot({reduced:false,frameMs:1000,onFrame:({document})=>{
  if(!app||!document.querySelector('.packet'))return;frames++;
  assert.equal(app.api.getFault(),null);assert.equal(document.querySelector('.local-fault'),null);
 }});
 const doc=app.document,M=app.c.NeuralPrologue;doc.querySelector('#welcome').close();app.api.switchLevel('decoder-layer');
 const state=M.solutions('decoder-layer'),borrowed=state.nodes.find(n=>n.id==='ln-attn');borrowed.id='ln-ffn-2';borrowed.kind='ln-ffn';
 for(const n of state.nodes)n.links=n.links.map(id=>id==='ln-attn'?'ln-ffn-2':id);
 app.api.setState(state);app.api.getWorld().rebuild();const saved=JSON.stringify(state);await app.api.run();
 assert.ok(frames>0);assert.equal(JSON.stringify(state),saved);assert.ok(!doc.querySelector('#level-complete').open);
 for(const selector of ['.local-fault','.fault-pin'])assert.equal(doc.querySelector(selector).dataset.owner,'ln-ffn-2');
 assert.match(doc.querySelector('.local-fault').textContent,/组件用途不匹配.*FFN/s);assert.equal(doc.querySelectorAll('.packet').length,0);
 doc.querySelector('[data-dismiss-result]').dispatchEvent(new app.window.Event('click',{bubbles:true}));
 assert.equal(app.api.getFault(),null);assert.equal(doc.querySelector('.local-fault'),null);assert.ok(doc.querySelector('#test-results .fail'));
});
test('rejected clicks show a port-local toast, track the camera, and dismiss without changing the circuit',async()=>{
 const timers=noticeTimers(),app=await boot({timers}),doc=app.document,M=app.c.NeuralPrologue;
 doc.querySelector('#welcome').close();app.api.switchLevel('weighted-sum');app.api.setState(M.solutions('weighted-sum'));app.api.getWorld().rebuild();
 const saved=JSON.stringify(app.api.getState()),event=(target,type,props={})=>{const e=new app.window.Event(type,{bubbles:true,cancelable:true});Object.assign(e,props);target.dispatchEvent(e);};
 event(doc.querySelector('[data-output="original-weights"]'),'click');event(doc.querySelector('[data-input="v:0"]'),'click');
 let toast=doc.querySelector('.connection-toast');assert.ok(toast);assert.equal(toast.parentElement.id,'labels');
 assert.equal(toast.getAttribute('role'),'status');assert.match(toast.textContent,/3 × 3.*3 × 2/);assert.equal(JSON.stringify(app.api.getState()),saved);
 assert.ok(doc.querySelector('[data-input="v:0"]').classList.contains('rejected'));
 Object.defineProperties(toast,{offsetWidth:{value:260},offsetHeight:{value:70}});app.api.getWorld().labels();
 const at=[toast.style.left,toast.style.top];event(doc.querySelector('#scene'),'wheel',{deltaY:150});assert.notDeepEqual([toast.style.left,toast.style.top],at);
 assert.ok(parseFloat(toast.style.left)>=12&&parseFloat(toast.style.left)<=1400-260-12);
 assert.ok(parseFloat(toast.style.top)>=12&&parseFloat(toast.style.top)<=760-70-12);
 assert.equal(doc.querySelectorAll('.connection-toast').length,1,'label refresh retains a single notice');
 const obsolete=timers.callbacks.at(-1);timers.tick(3200);app.api.chooseInput('v',0);toast=doc.querySelector('.connection-toast');obsolete();
 assert.equal(doc.querySelector('.connection-toast'),toast,'an older timer cannot remove the next rejection');timers.tick(3599);assert.ok(doc.querySelector('.connection-toast'));timers.tick(1);
 assert.equal(doc.querySelector('.connection-toast'),null);assert.equal(doc.querySelector('.port.rejected'),null);assert.equal(JSON.stringify(app.api.getState()),saved);
 assert.match(doc.querySelector('#feedback').textContent,/3 × 3.*3 × 2/,'the footer keeps the explanation for later reading');
 app.api.chooseInput('v',0);assert.ok(doc.querySelector('.connection-toast'));app.api.chooseInput('aggregate',0);
 assert.equal(doc.querySelector('.connection-toast'),null);assert.equal(doc.querySelector('.port.rejected'),null);assert.equal(timers.pending,0);
 assert.equal(JSON.stringify(app.api.getState()),saved,'a failed attempt never removed or replaced the old wire');
});

test('drag rejection and self-wires use the same localized notice and clear on cancellation, inspection, run or level change',async()=>{
 for(const language of ['zh-CN','en-US']){
  const timers=noticeTimers(),app=await boot({timers,languages:[language]}),doc=app.document,M=app.c.NeuralPrologue;
  doc.querySelector('#welcome').close();app.api.switchLevel('weighted-sum');app.api.setState(M.solutions('weighted-sum'));app.api.getWorld().rebuild();
  doc.querySelector('#scene').getBoundingClientRect=()=>({left:0,top:0,width:1400,height:760,right:1400,bottom:760});
  const event=(target,type,props={})=>{const e=new app.window.Event(type,{bubbles:true,cancelable:true});Object.assign(e,props);target.dispatchEvent(e);};
  const saved=JSON.stringify(app.api.getState());doc.elementFromPoint=()=>doc.querySelector('[data-input="v:0"]');
  event(doc.querySelector('[data-output="original-weights"]'),'pointerdown',{clientX:650,clientY:350});event(doc,'pointermove',{clientX:830,clientY:350});event(doc,'pointerup',{clientX:830,clientY:350});timers.tick(0);
  assert.ok(doc.querySelector('.connection-toast'));assert.match(doc.querySelector('.connection-toast').textContent,/3 × 3.*3 × 2/);assert.equal(JSON.stringify(app.api.getState()),saved);
  if(language==='en-US')assert.doesNotMatch(doc.querySelector('.connection-toast').textContent,/[\u3400-\u9fff]/);
  event(doc,'pointercancel');assert.equal(doc.querySelector('.connection-toast'),null);
  app.api.chooseOutput('v');app.api.chooseInput('v',0);assert.ok(doc.querySelector('.connection-toast').textContent.includes(app.c.I18n.t('不能把输出接回自己')));event(doc,'keydown',{key:'Escape'});assert.equal(doc.querySelector('.connection-toast'),null);
  const reject=()=>{app.api.chooseOutput('v');app.api.chooseOutput('original-weights');app.api.chooseInput('v',0);assert.ok(doc.querySelector('.connection-toast'));};
  reject();app.api.getWorld().focusShelf('aggregate');assert.equal(doc.querySelector('.connection-toast'),null);doc.querySelector('#card-close').onclick();
  reject();app.c.I18n.setPreference(language==='en-US'?'zh':'en');assert.equal(doc.querySelector('.connection-toast'),null);
  reject();const running=app.api.run();assert.equal(doc.querySelector('.connection-toast'),null);await running;
  assert.equal(doc.querySelectorAll('#test-results .pass').length,3);doc.querySelector('#complete-close').onclick();
  reject();app.api.switchLevel('scores');assert.equal(doc.querySelector('.connection-toast'),null);assert.equal(timers.pending,0);
 }
});

test('AV cards explain provenance and whole-vector contributions in both languages and feature widths',async()=>{
 for(const language of ['zh-CN','en-US']){
  const app=await boot({languages:[language]}),doc=app.document,M=app.c.NeuralPrologue;doc.querySelector('#welcome').close();
  for(const [level,kind,width]of [['weighted-sum','aggregate',2],['cross-attention','original-aggregate',4],['attention','aggregate',2],['multihead','aggregate',2]]){
   app.api.switchLevel(level);const before=JSON.stringify(app.api.getState()),exampleLevel=M.exampleLevel(kind,level),stage=M.levels[exampleLevel];app.api.getWorld().focusShelf(kind);
   const expected=app.c.aggregationExample(M,M.extraExample(kind,exampleLevel));
   assert.equal(doc.querySelector('.av-example').dataset.width,String(width));assert.equal(doc.querySelector('#card-attention-flow').hidden,false);
   assert.match(doc.querySelector('.av-a').textContent,/Softmax/);assert.match(doc.querySelector('.av-v').textContent,/V/);assert.equal(doc.querySelector('#card-formula').textContent,'O = A × V');
   assert.match(doc.querySelector('#card-purpose').textContent,/Transformer/);assert.equal(doc.querySelectorAll('.av-simple tbody tr').length,2);assert.match(doc.querySelector('.av-simple .av-sum').textContent,/\[0.5, 0\] \+ \[0, 3\] = \[0.5, 3\]/);
   assert.equal(doc.querySelector('.av-actual').hasAttribute('open'),false);
   assert.deepEqual([...doc.querySelectorAll('.av-actual tr[data-weight]')].map(el=>Number(el.dataset.weight)),Array.from(expected.weights));
   assert.ok(doc.querySelector('.av-example-level').textContent.includes(stage.title));
   const query=M.samplesFor(exampleLevel)[0].word.split(' / ')[2];assert.equal(doc.querySelector('.av-query>b').textContent,query);
   if(stage.dualSource)assert.deepEqual([...doc.querySelectorAll('.av-word')].map(el=>el.textContent),Array.from(stage.secondaryTokenIds,id=>M.math.vocab[id]));
   if(language==='en-US')assert.doesNotMatch(doc.querySelector('#module-card').textContent,/[\u3400-\u9fff]/);
   doc.querySelector('#card-close').onclick();assert.equal(JSON.stringify(app.api.getState()),before);
  }
  app.api.switchLevel('weighted-sum');app.api.getWorld().focusShelf('v');assert.equal(doc.querySelector('#card-attention-flow').hidden,true);assert.equal(doc.querySelector('.av-example'),null);doc.querySelector('#card-close').onclick();
 }
});

test('scoring cards explain the divisor and compare downstream Softmax using real values in both languages',async()=>{
 for(const language of ['zh-CN','en-US']){
  const app=await boot({languages:[language]}),doc=app.document,M=app.c.NeuralPrologue;doc.querySelector('#welcome').close();
  for(const [level,kind,width]of [['scores','score',2],['cross-attention','original-score',4]]){
   app.api.switchLevel(level);const before=JSON.stringify(app.api.getState());app.api.getWorld().focusShelf(kind);
   assert.equal(doc.querySelector('#module-card').open,true);assert.equal(doc.querySelector('.score-example').dataset.width,String(width));
   const expected=app.c.scoreComparison(M,M.extraExample(kind,level));
   for(const mode of ['before','after'])assert.deepEqual([...doc.querySelectorAll('.score-bar[data-mode="'+mode+'"]')].map(el=>Number(el.dataset.weight)),Array.from(expected[mode]));
   assert.match(doc.querySelector('#card-formula').textContent,/√dₖ/);assert.match(doc.querySelector('.score-shortcut').textContent,/Softmax\(\[0, 2\]/);
   assert.ok(doc.querySelector('.score-theory').textContent.length>200);assert.equal(doc.querySelectorAll('.score-table tbody tr').length,3);
   assert.equal(doc.querySelectorAll('.score-full .card-vector').length,3);
   if(level==='cross-attention')assert.equal(doc.querySelector('.score-pair .key b').textContent.endsWith(M.math.vocab[M.levels[level].secondaryTokenIds[0]]),true);
   if(language==='en-US')assert.doesNotMatch(doc.querySelector('#module-card').textContent,/[\u3400-\u9fff]/);
   doc.querySelector('#card-close').onclick();assert.equal(JSON.stringify(app.api.getState()),before);
  }
 }
});

test('shelf cards inspect every available kind without placing a node, including exhausted stock',async()=>{
 const app=await boot({languages:['en-US']}),doc=app.document,M=app.c.NeuralPrologue;doc.querySelector('#welcome').close();
 const event=(target,type,properties={})=>{const e=new app.window.Event(type,{bubbles:true,cancelable:true});Object.assign(e,properties);target.dispatchEvent(e);};
 for(const level of ['workshop','scores','add-norm','pack-scores','multihead','encoder-layer','encoder-decoder','ffn','attention','block']){
  app.api.switchLevel(level);const saved=JSON.stringify(app.api.getState()),view=app.view();
  for(const kind of Object.keys(M.inventory(M.levels[level]))){
   if(M.shelfKind(M.levels[level],kind)!==kind){assert.ok(doc.querySelector(`[data-add-module="${M.shelfKind(M.levels[level],kind)}"]`),'identical operators share a real shelf entry');continue;}
   if(!doc.querySelector(`[data-add-module="${kind}"]`)){assert.equal(M.fusions[level]?.kind,kind,'only this lesson’s unearned composite is hidden');continue;}
   const card=doc.querySelector(`[data-add-module="${kind}"]`);event(card,'click');event(card,'dblclick');await new Promise(setImmediate);
   assert.equal(doc.querySelector('#module-card').open,true,kind);assert.equal(doc.querySelector('#card-name').textContent,M.moduleName(M.levels[level],kind));
   const origin=M.packageOrigin(kind),number=doc.querySelector('#card-number');
   assert.equal(number.textContent,origin?'From: '+origin.chapter:M.levels[level].chapter);
   assert.equal(number.dataset.originLevel,origin?.id);
   assert.doesNotMatch(doc.querySelector('#module-card').textContent,/[\u3400-\u9fff]/,kind);
   assert.ok(app.scenes.at(-1).children.some(o=>o.userData.shelfPreview));assert.equal(JSON.stringify(app.api.getState()),saved);
   doc.querySelector('#camera-reset').onclick();await new Promise(setImmediate);doc.querySelector('#card-close').onclick();
   assert.equal(app.scenes.at(-1).children.some(o=>o.userData.shelfPreview),false);assert.deepEqual(app.view(),view);assert.equal(app.api.placeModule(0,0),false,'inspection leaves no placement gesture pending');
  }
 }
 app.api.switchLevel('workshop');app.api.setState(M.solutions('workshop'));app.api.getWorld().rebuild();app.api.refresh();
 const card=doc.querySelector('[data-add-module="relu-ffn"]');assert.equal(card.dataset.addDisabled,'true');assert.equal(card.getAttribute('aria-disabled'),'false');
 const saved=JSON.stringify(app.api.getState());event(card,'dblclick');await new Promise(setImmediate);assert.equal(doc.querySelector('#module-card').open,true);doc.querySelector('#card-close').onclick();assert.equal(JSON.stringify(app.api.getState()),saved);
 await app.api.run();assert.equal(doc.querySelectorAll('#test-results .pass').length,5,'preview did not replace the real operator used for execution');
});

test('clicking a placed module offers deletion, preserves other wiring and remains compatible with double-click',async()=>{
 const app=await boot(),doc=app.document,M=app.c.NeuralPrologue;doc.querySelector('#welcome').close();app.api.switchLevel('ffn');app.api.setState(M.solutions('ffn'));app.api.getWorld().rebuild();app.api.refresh();
 const event=(target,type,properties={})=>{const e=new app.window.Event(type,{bubbles:true,cancelable:true});Object.assign(e,properties);target.dispatchEvent(e);};
 doc.querySelector('#scene').getBoundingClientRect=()=>({left:0,top:0,width:1400,height:760,right:1400,bottom:760});
 const saved=JSON.stringify(app.api.getState());event(doc.querySelector('[data-node="down"]'),'click');assert.ok(doc.querySelector('[data-delete-node="down"]'));
 event(doc.querySelector('[data-close-node]'),'click');assert.equal(doc.querySelector('.node-actions'),null);assert.equal(JSON.stringify(app.api.getState()),saved);
 event(doc.querySelector('[data-node="down"]'),'click');event(doc.querySelector('[data-delete-node="down"]'),'click');
 assert.equal(app.api.getState().nodes.some(n=>n.id==='down'),false);assert.deepEqual(Array.from(app.api.getState().nodes.find(n=>n.id==='add').links),['R',null]);assert.equal(app.api.getState().nodes.find(n=>n.id==='multiply').links[1],'project');assert.equal(M.stock(app.api.getState(),'down'),2);
 doc.querySelector('#undo').onclick();assert.equal(JSON.stringify(app.api.getState()),saved);
 for(let i=0;i<2;i++){event(doc.querySelector('[data-node="down"]'),'pointerdown',{clientX:650,clientY:350});event(doc,'pointerup',{clientX:650,clientY:350});if(!i)assert.ok(doc.querySelector('.node-actions'));}
 await new Promise(setImmediate);assert.equal(doc.querySelector('#module-card').open,true);assert.equal(doc.querySelector('.node-actions'),null);doc.querySelector('#card-close').onclick();assert.equal(JSON.stringify(app.api.getState()),saved);
 app.api.switchLevel('projection');event(doc.querySelector('[data-node="project"]'),'click');assert.equal(doc.querySelector('.node-actions'),null,'fixed introductory modules remain fixed');
});

test('duplicate modules render cards, distinct processing and correctly sized weight matrices',async()=>{
 let app;const seen=new Set();
 app=await boot({reduced:false,frameMs:120,onFrame:({document})=>{for(const el of document.querySelectorAll('.processing-packet'))if(!el.hidden)seen.add(el.dataset.node);}});
 app.api.switchLevel('ffn');const s=app.c.NeuralPrologue.initial('ffn');
 const coords=[[-4.7,-2],[-1.4,-2],[1.9,-2],[5.2,-2],[-4.7,2],[-1.4,2]];['project','down','project','down','add','add'].forEach((kind,i)=>app.c.NeuralPrologue.addNode(s,kind,...coords[i]));
 for(const edge of [['R','project',0],['project','down',0],['down','project-2',0],['project-2','down-2',0],['down','add',0],['down-2','add',1],['add','add-2',0],['R','add-2',1],['add-2','sink',0]])app.c.NeuralPrologue.connect(s,...edge);
 app.api.setState(s);app.api.getWorld().rebuild();app.api.refresh();
 for(const id of ['project-2','down-2','add-2']){app.api.getWorld().focus(id);assert.ok(app.document.querySelector('#card-name').textContent.endsWith(' · 2'));app.document.querySelector('#card-close').onclick();}
 await app.api.run();assert.equal(app.document.querySelector('#loading').hidden,true);for(const id of s.nodes.map(n=>n.id))assert.ok(seen.has(id),id+' has an independent processing event');assert.equal(app.document.querySelectorAll('#test-results .fail').length,1,'a freely built incorrect circuit still receives real numerical validation');
});

test('all projection outputs get equal travel time and only the speed slider changes their pace',async()=>{
 let app,runIndex=0;const runs=[new Map(),new Map()],departures=[[],[]];
 app=await boot({reduced:false,frameMs:20,onFrame:({document,now})=>{
  const proc=document.querySelector('.processing-packet[data-node="project"][data-row="2"]');
  if(!app)return;
  const wire=document.querySelector('.packet[data-motion="wire"][data-from="project"][data-row="2"]');
  if(wire&&!wire.hidden)departures[runIndex].push(now);
  if(!proc||proc.hidden)return;
  app.scenes.at(-1).traverse(o=>{if(!o.visible||!o.userData.contribution)return;const d=o.userData;
   if(!runs[runIndex].has(d.output))runs[runIndex].set(d.output,[]);runs[runIndex].get(d.output).push(now);
  });
 }});
 app.api.switchLevel('projection');app.api.setState(app.c.NeuralPrologue.solutions('projection'));app.api.getWorld().rebuild();
 for(const rate of [.25,.5]){
  const slider=app.document.querySelector('#playback-speed');slider.value=String(rate);slider.dispatchEvent(new app.window.Event('input',{bubbles:true}));
  await app.api.run();const spans=runs[runIndex];assert.equal(spans.size,4);
  const widths=[...spans.values()].map(times=>Math.max(...times)-Math.min(...times));
  assert.ok(Math.max(...widths)-Math.min(...widths)<=20,'no later output is compressed into a short flash');
  for(let j=1;j<4;j++)assert.ok(Math.max(...spans.get(j-1))<Math.min(...spans.get(j)),'outputs are shown one at a time');
  assert.ok(Math.min(...departures[runIndex])>Math.max(...spans.get(3)),'the outgoing vector waits for the last output');
  assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);runIndex++;
 }
 for(let j=0;j<4;j++){const width=i=>Math.max(...runs[i].get(j))-Math.min(...runs[i].get(j));assert.ok(Math.abs(width(0)-2*width(1))<=40,'the speed slider scales every output equally');}
});

test('transport takes two-thirds of its former time while matrix processing keeps its slow duration',async()=>{
 let app;const spans=new Map();
 app=await boot({reduced:false,frameMs:20,onFrame:({document,now})=>{
  if(!app)return;
  for(const el of document.querySelectorAll('.packet:not([hidden])')){
   const key=[el.dataset.row,el.dataset.motion,el.dataset.from].join(':');
   if(!spans.has(key))spans.set(key,{start:now,end:now});spans.get(key).end=now;
  }
 }});
 app.api.switchLevel('projection');app.api.setState(app.c.NeuralPrologue.solutions('projection'));app.api.getWorld().rebuild();await app.api.run();
 assert.equal(app.document.querySelector('#playback-speed').value,'0.25');
 for(const row of ['0','1','2']){
  for(const from of ['R','project']){const span=spans.get(row+':wire:'+from);assert.ok(span);assert.ok(Math.abs(span.end-span.start-650/.25/1.5)<40,'travel is faster for every input');}
  const process=spans.get(row+':process:project');assert.ok(Math.abs(process.end-process.start-2.2*650/.25)<40,'computation keeps its former duration');
 }
});

test('one whole weight column, its output sphere and its attached formula highlight together',async()=>{
 let app,pausedFrames=0,snapshot=null;const columns=new Set(),directions=new Set();
 const allUnits=()=>app.scenes.at(-1).children.flatMap(o=>o.children).filter(o=>o.userData.node==='project');
 app=await boot({reduced:false,frameMs:40,onFrame:({document})=>{
  if(!app)return;const calc=document.querySelector('.projection-calculation[data-node="project"]');if(!calc||calc.hidden||Number(calc.style.getPropertyValue('--calculation-focus'))<.9)return;
  const unit=allUnits()[0],col=Number(calc.dataset.column),cues=unit.children.filter(o=>o.userData.projectionColumn!=null),spheres=unit.children.filter(o=>o.geometry?.type==='SphereGeometry'&&o.position.x===.7);
  columns.add(col);assert.equal(unit.userData.activeColumn,col);assert.deepEqual(cues.filter(o=>o.visible).map(o=>o.userData.projectionColumn),[col]);
  const active=cues[col],faces=unit.children.filter(o=>o.userData.detailText&&Math.abs(o.position.x-active.position.x)<1e-9);
  assert.equal(faces.length,2,'both weights of the column participate, including zeros');
  assert.ok(faces.every(o=>o.material.color.getHex()!==0xffffff));
  assert.ok(unit.children.filter(o=>o.userData.detailText&&!faces.includes(o)).every(o=>o.material.color.getHex()===0xffffff));
  assert.ok(spheres[col].material.emissiveIntensity>.8);assert.ok(spheres.filter((_,i)=>i!==col).every(o=>o.material.emissiveIntensity<.1),'only the sphere being computed glows');
  assert.equal(calc.style.getPropertyValue('--calculation-color'),'#'+active.children[1].material.color.getHexString());
  const screen=spheres[col].getWorldPosition(new app.c.T.Vector3()).project(app.camera()),x=(screen.x+1)*700,y=(1-screen.y)*380,scale=Number(calc.style.getPropertyValue('--calculation-scale'));
  const radius=.115*spheres[col].scale.x*1400/app.view().width,side=calc.dataset.side;
  directions.add(side);assert.ok(Math.abs(parseFloat(calc.style.top)-y)<1e-7,'formula stays level with its sphere');
  assert.ok(Math.abs(Math.abs(parseFloat(calc.style.left)-x)-radius-9*scale)<1e-7,'the pointer reaches the sphere edge');
  if(col===1&&pausedFrames<3){
   if(!pausedFrames){app.api.togglePause();snapshot={col,opacity:active.children[0].material.opacity,scale};}
   assert.equal(col,snapshot.col);assert.equal(active.children[0].material.opacity,snapshot.opacity,'pause also freezes the highlight');
   const e=new app.window.Event('wheel',{bubbles:true,cancelable:true});e.deltaY=pausedFrames===0?-150:pausedFrames===1?300:0;document.querySelector('#scene').dispatchEvent(e);
   if(!pausedFrames)assert.ok(Number(calc.style.getPropertyValue('--calculation-scale'))>snapshot.scale,'formula grows with the board');
   if(pausedFrames===1)assert.ok(Number(calc.style.getPropertyValue('--calculation-scale'))<snapshot.scale,'formula shrinks with the board');
   if(++pausedFrames===3)app.api.togglePause();
  }
 }});
 app.api.switchLevel('projection');app.api.setState(app.c.NeuralPrologue.solutions('projection'));app.api.getWorld().rebuild();await app.api.run();
 assert.deepEqual([...columns].sort(),[0,1,2,3]);assert.equal(pausedFrames,3);assert.ok(directions.has('right'));
 for(const unit of allUnits()){assert.equal(unit.userData.activeColumn,null);assert.ok(unit.children.filter(o=>o.userData.projectionColumn!=null).every(o=>!o.visible));}
 assert.equal(app.document.querySelectorAll('.projection-calculation,.projection-result').length,0);
});
test('cloud modules render real shard matrices and animate both inputs of gathering and reduction',async()=>{
 let app;const seen=new Set();app=await boot({reduced:false,frameMs:700,onFrame:({document})=>{
  if(!app)return;const root=app.scenes.at(-1);root?.traverse(o=>{if(o.visible&&o.userData.processingSecondary==='tp-gather')seen.add('gather-secondary');if(o.visible&&o.userData.additionFeature!=null)seen.add('partial-sum');});
 }});
 const M=app.c.NeuralPrologue;
 for(const id of M.cloud.ids){app.api.switchLevel(id);app.api.setState(M.solutions(id));app.api.getWorld().rebuild();app.api.refresh();await app.api.run();assert.equal(app.api.getFault(),null);assert.equal(app.document.querySelector('#level-complete').open,true,id);}
 assert.ok(seen.has('gather-secondary'));assert.ok(seen.has('partial-sum'));
 app.api.select('tp-reduce');app.api.getWorld().focus('tp-reduce');assert.equal(app.document.querySelector('#card-merge-contrast').hidden,true,'communication is not described as a residual');
});

test('board surfaces and grid accept the stage palette without tinting the input vectors',async()=>{
 const app=await boot(),T=app.c.T,M=app.c.NeuralPrologue,scene=new T.Scene(),art=app.c.createSceneArt(T,scene,{});
 const input=()=>scene.children[1].children.filter(o=>o.isMesh).map(o=>o.material?.color?.getHex());
 art.surroundings(M.samplesFor('wire'),M.levels.wire);const original=input();
 art.surroundings(M.samplesFor('wire'),M.levels.wire,{surface:'#f4e3eb',edge:'#dfbecd',accent:'#8c4d62'});
 const floor=scene.children[0];assert.equal(floor.children[2].material.color.getHexString(),'f4e3eb');assert.equal(floor.children[3].material.color.getHexString(),'dfbecd');
 assert.deepEqual(input(),original,'the input and output geometry keeps its own semantic colors');
});

test('source and output captions fit the compact board without moving connection ports',async()=>{
 const app=await boot({viewport:{width:530,height:550},languages:['en-US']}),doc=app.document;
 app.api.switchLevel('encoder-layer');const state=JSON.stringify(app.api.getState());
 const positions=()=>[...doc.querySelectorAll('.port')].map(el=>[el.style.left,el.style.top]);
 const before=positions(),labels=[...doc.querySelectorAll('.source-label')];
 for(const el of labels)Object.defineProperties(el,{offsetWidth:{value:205},offsetHeight:{value:66}});
 doc.querySelector('#camera-reset').onclick();
 for(const el of labels){const x=Number.parseFloat(el.style.left);assert.ok(x>=110.5&&x<=419.5,'caption remains within the scene');}
 assert.deepEqual(positions(),before,'only captions shift; ports stay on their projected coordinates');
 assert.equal(JSON.stringify(app.api.getState()),state);
});

test('attention outlets distinguish attention weights from aggregated and projected features',async()=>{
 const app=await boot(),M=app.c.NeuralPrologue;
 for(const [id,weights,note]of [['causal',true,/未来词的比例为 0/],['softmax',true,/读取对应词 V 的比例/],['weighted-sum',false,/O = A × V，尚未经过输出投影/],['aggregation',false,/O = A × V.*尚未投影/],['attention',false,/汇合与输出投影；尚未加残差/]]){
  app.api.switchLevel(id);
  const label=[...app.document.querySelectorAll('.source-label')].find(el=>el.querySelector('b')?.textContent===M.levels[id].outputTitle);
  assert.ok(label,id);assert.equal(label.querySelector('b').textContent.includes('权重'),weights,id);
  assert.ok(label.textContent.includes(M.levels[id].outputNote),id);assert.match(label.textContent,note,id);
 }
});

test('causal masking routes real scores through shutters, keeps allowed values and gathers all outputs',async()=>{
 let app,paused=false,pauseFrames=0,frozen;const rows=new Set(),stages=new Set(),blocked=new Set(),allowed=new Set();let shutters=false,gather=false,frames=0;
 app=await boot({reduced:false,frameMs:120,onFrame:({document})=>{
  if(!app)return;const scene=app.scenes.at(-1);let flow;
  scene.traverse(o=>{if(o.userData.tokenFlow?.operation==='mask')flow=o.userData.tokenFlow;
   if(o.userData.maskShutter&&o.visible&&o.material.opacity>.1)shutters=true;
   if(o.userData.operation==='mask'&&o.visible){const d=o.userData;assert.equal(d.blocked,d.feature>d.row);if(d.blocked){assert.equal(d.value,-Infinity);blocked.add(d.row+':'+d.feature);}else{assert.equal(d.value,d.input);allowed.add(d.row+':'+d.feature);}}
   if(o.userData.processingOutput==='mask'&&o.visible&&o.userData.gather>0)gather=true;
  });
  if(!flow)return;frames++;rows.add(flow.row);stages.add(flow.stage);
  const label=document.querySelector('.token-calculation[data-operation="mask"]');assert.ok(label);assert.doesNotMatch(label.textContent,/NaN|undefined/);
  if(!paused&&flow.row===1&&flow.stage==='restrict'){paused=true;frozen=JSON.stringify(flow);app.api.togglePause();}
  else if(paused&&pauseFrames<3){assert.equal(JSON.stringify(flow),frozen);if(++pauseFrames===3)app.api.togglePause();}
 }});
 const M=app.c.NeuralPrologue;app.api.switchLevel('causal');app.api.setState(M.solutions('causal'));app.api.getWorld().rebuild();await app.api.run();
 assert.deepEqual([...rows].sort(),[0,1,2]);for(const stage of ['arrive','restrict','replace','ready'])assert.ok(stages.has(stage));
 assert.equal(blocked.size,3);assert.equal(allowed.size,6);assert.ok(shutters&&gather&&frames>20);assert.equal(pauseFrames,3);
 assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);
 assert.equal(app.document.querySelectorAll('.token-calculation[data-operation="mask"]').length,0);
 app.scenes.at(-1).traverse(o=>{assert.ok(o.userData.tokenFlow?.operation!=='mask');if(o.userData.maskShutter||o.userData.maskResult)assert.equal(o.visible,false);});
 const t=app.c.createSceneArt(app.c.T,new app.c.T.Scene(),{}).tokenFrame;
 assert.equal(t(.22,'mask').row,0);assert.equal(t(.22+1.8,'mask').row,1);assert.equal(t(.22+3.6,'mask').row,2);
 assert.equal(t(0,'mask').step,1.8);
});

test('masked bridge animates both expanded heads and packages the whole board before completion',async()=>{
 let app;const masks=new Set(),projections=new Set(),phases=new Set();
 app=await boot({reduced:false,frameMs:180,onFrame:({document})=>{
  if(!app)return;const scene=app.scenes.at(-1);scene.traverse(o=>{
   if(o.userData.tokenFlow?.operation==='mask')masks.add(o.userData.node);
   if(o.userData.sequenceProjection)projections.add(o.userData.node);
  });
  const effect=scene.children.find(o=>o.userData.fusion);if(effect){
   assert.ok(!document.querySelector('#level-complete').open);assert.equal(effect.userData.fusion.count,16);phases.add(effect.userData.fusion.stage);
   document.querySelector('[data-add-module="masked-heads"]').getBoundingClientRect=()=>({left:24,top:160,width:104,height:60});
  }
 }});
 app.api.switchLevel('masked-multihead');app.api.setState(app.c.NeuralPrologue.solutions('masked-multihead'));app.api.getWorld().rebuild();await app.api.run();
 assert.equal(app.api.getFault(),null);assert.deepEqual([...masks].sort(),['mask','mask-2']);
 for(const h of ['a','b'])for(const r of ['q','k','v'])assert.ok(projections.has('masked-'+h+'-'+r),'projection '+h+' '+r);
 assert.ok(phases.size>=3,'board gathering, module formation and shelf landing');assert.ok(app.document.querySelector('#level-complete').open);
 assert.equal(app.document.querySelectorAll('#test-results .pass').length,3);assert.equal(app.api.getState().nodes.length,16);
 assert.equal(app.document.querySelectorAll('.token-calculation,.projection-calculation,.packet').length,0);
});

test('generation module cards show computed words and the retained context in both languages',async()=>{
 const app=await boot(),doc=app.document,M=app.c.NeuralPrologue;doc.querySelector('#welcome').close();app.api.switchLevel('autoregressive');
 const saved=JSON.stringify(app.api.getState());
 for(const lang of ['zh','en']){
  app.c.I18n.setPreference(lang);
  for(const kind of ['step','pick','feedback']){
   app.api.getWorld().focusShelf(kind);
   assert.ok(doc.querySelector('#module-card').open);
   assert.equal(doc.querySelector('#card-name').textContent,M.definitions[kind].name);
   assert.ok(doc.querySelector('#card-example-values .generation-example'));
   if(kind==='step')assert.deepEqual([...doc.querySelectorAll('#card-example-values td')].map(el=>Number(el.textContent)),Array.from(M.autoregressiveExample().scores.slice(-6)));
   if(kind==='pick'&&lang==='en')assert.match(doc.querySelector('#card-meaning').textContent,/not three newly generated tokens/);
   if(kind==='feedback'){
    assert.equal(doc.querySelectorAll('#card-example-values .generation-new').length,2);
    assert.equal(doc.querySelector('#card-example-values .generation-dropped').dataset.tokenId,'0');
   }
   if(lang==='en')assert.doesNotMatch(doc.querySelector('#module-card').textContent,/[\u3400-\u9fff]/);
   doc.querySelector('#card-close').onclick();assert.equal(JSON.stringify(app.api.getState()),saved);
  }
 }
});

test('failed module placement shows a localized drop-point toast for clicks and drags without consuming stock',async()=>{
 for(const language of ['zh-CN','en-US']){
  const timers=noticeTimers(),app=await boot({timers,languages:[language]}),doc=app.document,M=app.c.NeuralPrologue,T=app.c.T;
  doc.querySelector('#welcome').close();app.api.switchLevel('ffn');app.api.chooseModule('project');assert.equal(app.api.placeModule(-3,0),true);
  const host=doc.querySelector('#scene'),world=app.api.getWorld(),saved=JSON.stringify(app.api.getState()),stock=M.stock(app.api.getState(),'project');
  host.getBoundingClientRect=()=>({left:0,top:0,width:1400,height:760,right:1400,bottom:760});
  const fire=(target,type,props={})=>{const e=new app.window.Event(type,{bubbles:true,cancelable:true});Object.assign(e,props);target.dispatchEvent(e);};
  const drop=()=>{const p=new T.Vector3(-3,0,0).project(app.camera());return {clientX:(p.x+1)*700,clientY:(1-p.y)*380};};
  const check=()=>{const toast=doc.querySelector('.connection-toast');assert.ok(toast);assert.equal(toast.dataset.errorKind,'placement');assert.equal(toast.getAttribute('role'),'status');assert.equal(toast.parentElement.id,'labels');assert.match(toast.textContent,language==='en-US'?/Module not placed.*empty spot/:/组件未放置.*这里放不下/);assert.equal(doc.querySelector('.port.rejected'),null);assert.equal(JSON.stringify(app.api.getState()),saved);assert.equal(M.stock(app.api.getState(),'project'),stock);return toast;};
  app.api.chooseModule('project');fire(host,'pointerdown',drop());let toast=check();
  Object.defineProperties(toast,{offsetWidth:{value:260},offsetHeight:{value:70}});world.labels();
  const before=[toast.style.left,toast.style.top];fire(host,'wheel',{deltaY:150});assert.notDeepEqual([toast.style.left,toast.style.top],before,'notice follows its world-space drop point');
  assert.equal(doc.querySelectorAll('.connection-toast').length,1);timers.tick(3599);check();timers.tick(1);assert.equal(doc.querySelector('.connection-toast'),null);
  fire(doc.querySelector('[data-add-module="project"]'),'pointerdown',{clientX:60,clientY:100});fire(doc,'pointermove',drop());fire(doc,'pointerup',drop());timers.tick(0);check();
  assert.ok(!app.scenes.at(-1).children.some(o=>o.userData.placementGhost),'drag preview is cleared but the rejection remains');
  fire(doc,'keydown',{key:'Escape'});assert.equal(doc.querySelector('.connection-toast'),null);assert.equal(timers.pending,0);
  app.api.chooseModule('project');const bounds=M.bounds(M.levels.ffn);assert.equal(app.api.placeModule(bounds.right+2,bounds.front+2),false);toast=check();
  Object.defineProperties(toast,{offsetWidth:{value:260},offsetHeight:{value:70}});world.labels();
  assert.ok(parseFloat(toast.style.left)>=12&&parseFloat(toast.style.left)<=1400-260-12);assert.ok(parseFloat(toast.style.top)>=12&&parseFloat(toast.style.top)<=760-70-12);
  assert.equal(app.api.placeModule(2,0),true,'a rejected click can be retried at a valid spot');assert.equal(doc.querySelector('.connection-toast'),null);assert.equal(timers.pending,0);assert.equal(app.api.getState().nodes.length,2);
  assert.deepEqual([...app.c.I18n.missing],[]);
 }
});
test('module card prose marks curated terms and keeps them explained in both languages',async()=>{
 const app=await boot(),doc=app.document,M=app.c.NeuralPrologue;
 for(const language of ['zh','en']){
  app.c.I18n.setPreference(language);
  app.api.switchLevel('ffn');
  const solution=M.solutions('ffn');app.api.setState(solution);app.api.getWorld().rebuild();app.api.refresh();
  app.api.getWorld().focus(solution.nodes.find(n=>n.kind==='silu').id);
  assert.equal(doc.querySelector('#module-card').open,true);
  const card=doc.querySelector('#module-card'),term=card.querySelector('.glossary-term');
  assert.ok(term,'the module card prose marks curated terms');
  assert.equal(term.getAttribute('tabindex'),'0');
  assert.equal(term.getAttribute('aria-describedby'),'glossary-popover');
  const popover=doc.querySelector('#glossary-popover');
  term.dispatchEvent(new app.window.Event('focusin',{bubbles:true}));
  assert.equal(popover.hidden,false,'focus opens the glossary popover');
  assert.match(popover.textContent,/激活|Activation/);
  if(language==='en')assert.doesNotMatch(card.textContent,/[\u3400-\u9fff]/);
  doc.querySelector('#card-close').onclick();
 }
 assert.deepEqual([...app.c.I18n.missing],[]);
});
