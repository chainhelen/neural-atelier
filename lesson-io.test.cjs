const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
test('every lesson has explicit bilingual boundaries without changing its puzzle or arithmetic',()=>{
 const c={console,navigator:{languages:['zh-CN']}};c.window=c;vm.createContext(c);
 const load=f=>vm.runInContext(fs.readFileSync(__dirname+'/'+f,'utf8'),c,{filename:f});
 for(const f of ['locales/en.js','i18n.js','model.js','campaign.js','engine-model.js','original-curriculum.js','transformer-demo.js','cloud-curriculum.js'])load(f);
 vm.runInContext('Math.random=()=>0.5',c);
 const M=c.NeuralPrologue,ids=Object.values(M.routes).flatMap(r=>r.ids);
 const puzzle=id=>JSON.stringify({initial:M.initial(id),solution:M.solutions(id),inventory:M.inventory(M.levels[id]),samples:M.samplesFor(id),answers:M.samplesFor(id).map(s=>M.reference(s.v,id))});
 const before=new Map(ids.map(id=>[id,puzzle(id)]));
 load('lesson-io.js');load('structure-map.js');c.I18n.activate();
 assert.deepEqual([...M.lessonIO.audited].sort(),[...ids].sort(),'new lessons must declare what their outlet contains');
 for(const id of ids)assert.equal(puzzle(id),before.get(id),id+' keeps its inputs, answers, initial circuit and inventory');
 for(const language of ['zh','en']){
  c.I18n.setPreference(language);
  for(const id of ids){
   const l=M.levels[id];assert.ok(l.sourceTitle&&l.sourceNote&&l.routeInput&&l.outputTitle&&l.outputNote&&l.resultNote,id+' has complete boundary descriptions');
   assert.doesNotMatch(l.outputTitle,/^(结果特征|完整计算结果|本层特征 Y|输出特征 \/ 预测词|Result features|Complete result|Layer features Y)$/);
   assert.equal(M.structureMap.get(id).output,l.outputTitle+' · '+l.outputNote,id+' shares the board description with its hint');
   if(language==='en')assert.doesNotMatch(l.sourceTitle+l.sourceNote+l.routeInput+l.outputTitle+l.outputNote+l.resultNote,/[\u3400-\u9fff]/,id+' translated boundaries');
  }
 }
 const x=M.samplesFor('weighted-sum')[0].v,a=M.reference(x,'softmax'),v=M.reference(x,'qkv');
 const o=M.math.mix(a,v),y=M.math.mat(o,M.weights.output);
 assert.deepEqual(Array.from(M.reference(x,'weighted-sum')),Array.from(o));
 assert.deepEqual(Array.from(M.reference(x,'self-attention')),Array.from(y));
 assert.notDeepEqual(Array.from(o),Array.from(y),'the aggregation and projected attention outlets are distinct');
 assert.equal(M.levels['pack-scores'].chapter,'Original · reuse after 07');
 assert.equal(M.levels['masked-multihead'].chapter,'Original · assembly practice');
 for(const field of ['summary','description']){
  assert.match(M.levels.softmax[field],/Q\/K/);assert.match(M.levels.softmax[field],/Softmax/);
  assert.match(M.levels.softmax[field],/sum to 1/);assert.match(M.levels.softmax[field],/without masking future positions|no future mask/);
 }
 assert.ok(M.levels.softmax.summary.length<M.levels.softmax.description.length,'the menu summarizes rather than repeats the wiring instructions');
});
