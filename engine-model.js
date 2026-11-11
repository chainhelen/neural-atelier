/* Educational cost model, in simulated ms / MB. Not a benchmark of Apple hardware. */
(function(M){
M.engineDefaults={batch:16,layout:'channels',precision:8,cache:true,fusion:true,resident:true,prefix:false,split:0,context:2048};
M.engineControls={
 layout:{label:I18n.t("张量布局"),values:[['sequence',I18n.t("序列优先 B/S/C")],['channels',I18n.t("通道优先 B/C/1/S")]]},
 batch:{label:I18n.t("Prefill 每批位置数"),values:[1,4,8,16,32,64,128].map(n=>[n,String(n)])},
 cache:{label:I18n.t("历史 K/V"),values:[[false,I18n.t("每步重算")],[true,I18n.t("保留并追加")]]},
 precision:{label:I18n.t("权重精度"),values:[[16,'16 bit'],[8,'8 bit'],[4,'4 bit'],[2,'2 bit']]},
 fusion:{label:I18n.t("激活与门控"),values:[[false,I18n.t("分别写回内存")],[true,I18n.t("融合执行")]]},
 resident:{label:I18n.t("权重驻留"),values:[[false,I18n.t("重复从存储读入")],[true,I18n.t("保留在统一内存")]]},
 split:{label:I18n.t("Prefill 投影 / FFN 的 ANE 分担"),values:[0,20,40,60,80,100].map(n=>[n,n+'%'])},
 prefix:{label:I18n.t("第二轮相同前缀"),values:[[false,I18n.t("重新计算")],[true,I18n.t("复用匹配状态")]]},
 context:{label:I18n.t("KV 容量（位置数）"),values:[512,1024,2048,4096,8192].map(n=>[n,String(n)])}
};
const quant=(w,bits)=>{if(bits===16)return w;const q=2**(bits-1)-1,max=Math.max(...w.flat().map(Math.abs)),scale=max/q;return w.map(r=>r.map(v=>Math.round(v/scale)*scale));};
function engineCompute(x,bits){
 const {mat,rotate,scores,causal,probabilities,mix,normalize,rows}=M.math;
 const Wq=quant(M.weights.q,bits),Wk=quant(M.weights.k,bits),Wv=quant(M.weights.v,bits),Wo=quant(M.weights.output,bits);
 const z=normalize(x),a=mat(mix(probabilities(causal(scores(rotate(mat(z,Wq)),rotate(mat(z,Wk))))),mat(z,Wv)),Wo);
 const f=rows(a).map(r=>{const n=M.norm(r),g=M.linear(n,quant(M.weights.gate,bits)).map(M.silu),u=M.linear(n,quant(M.weights.up,bits));return M.linear(g.map((v,i)=>v*u[i]),quant(M.weights.down,bits));}).flat();
 return a.map((v,i)=>v+f[i]);
}
// Quantize actual miniature weights; correctness is not determined by a setting flag.
M.operations.enqkv=([x],ctx)=>{const {mat,normalize}=M.math,b=ctx.state.settings.precision,z=normalize(x);return [...mat(z,quant(M.weights.q,b)),...mat(z,quant(M.weights.k,b)),...mat(z,quant(M.weights.v,b))];};
M.operations.enattn=([qkv],ctx)=>{const {mat,rotate,scores,causal,probabilities,mix}=M.math,b=ctx.state.settings.precision;return mat(mix(probabilities(causal(scores(rotate(qkv.slice(0,6)),rotate(qkv.slice(6,12))))),qkv.slice(12,18)),quant(M.weights.output,b));};
M.operations.enffn=([a],ctx)=>{const b=ctx.state.settings.precision,f=M.math.rows(a).map(r=>{const n=M.norm(r),g=M.linear(n,quant(M.weights.gate,b)).map(M.silu),u=M.linear(n,quant(M.weights.up,b));return M.linear(g.map((v,i)=>v*u[i]),quant(M.weights.down,b));}).flat();return a.map((v,i)=>v+f[i]);};
const taskBase={load:6,enembed:2,enqkv:24,enattn:10,enffn:32,enhead:3};
const heavy=new Set(['enqkv','enffn']);
function report(state,level,work={prompt:128,context:2048}){
 const s={...M.engineDefaults,...state.settings},errors=[],timeline=[],prompt=work.prompt||128,context=work.context||2048,batches=Math.ceil(prompt/s.batch),scale=prompt/128,share=s.split/100;
 let cursor=0,transfer=0,aneWork=0,computeWork=0,last='cpu';
 for(const n of state.nodes){
  const device=n.device||'gpu';
  if(device==='ane'&&!heavy.has(n.id))errors.push(M.definitions[n.id].name+I18n.t(" 不在本关 ANE 支持集合内"));
  if((device==='ane'||heavy.has(n.id)&&share>0)&&s.layout!=='channels')errors.push(I18n.t("ANE 接收的张量布局不兼容"));
  const base=taskBase[n.id]*(n.id==='load'?1:scale),cpuFactor=n.id==='load'||n.id==='enembed'||n.id==='enhead'?1:8;
  let duration=base*(device==='cpu'?cpuFactor:device==='ane'?.68:1),effective=device;
  if(heavy.has(n.id)&&device==='gpu'&&share>0){duration=Math.max(base*(1-share),base*share*.68)+2;aneWork+=base*share;effective='gpu+ane';}
  else if(device==='ane')aneWork+=base;
  computeWork+=base;
  if(n.id==='load'&&!s.resident)duration+=28;
  if(heavy.has(n.id)||n.id==='enattn')duration+=batches*(device==='ane'?1.2:.65);
  if(n.id==='enffn'&&!s.fusion)duration+=14*scale;
  if(n.id==='enqkv'&&s.layout==='sequence')duration+=6*scale;
  if(last!==device){const cost=device==='ane'||last==='ane'?4:1.5;transfer+=cost;timeline.push({id:'transfer',name:I18n.t("同步 / 布局交接"),device:'memory',start:cursor,duration:cost});cursor+=cost;}
  if(n.id==='enffn'&&!s.fusion)transfer+=5;
  timeline.push({id:n.id,name:M.definitions[n.id].name,device:effective,start:cursor,duration});cursor+=duration;last=device;
 }
 const memory=27*s.precision/8*1.08+s.context*.0012+1.2; // Decimal GB, 27B weights + metadata + simplified KV + runtime.
 const scratch=s.batch*12+80;
 // Decode mostly reads weights, plus KV; unlike prefill it has little reuse per new token.
 const bandwidth=(state.nodes.some(n=>heavy.has(n.id)&&n.device==='cpu')?70:1700);
 const weightRead=27*s.precision/8*1000/bandwidth;
 const decodeTransfer=state.nodes.filter(n=>n.device==='ane').length*5;
 const decodeMs=weightRead+context*.0006+2+decodeTransfer+(s.cache?0:context/128*7)+(s.resident?0:28);
 const ttft=cursor,second=ttft*(s.prefix&&s.cache?.22:1)+8,total=ttft+decodeMs*4;
 const accuracy=M.samplesFor(level.id).map(t=>{const actual=engineCompute(t.v,s.precision),expected=engineCompute(t.v,16);return Math.max(...actual.map((x,i)=>Math.abs(x-expected[i])));});
 const error=Math.max(...accuracy);
 const metrics={ttft,total,tps:1000/decodeMs,decode:decodeMs,second,memory,scratch,transfer,aneMin:100*aneWork/computeWork,error};
 if(s.context<context)errors.push(I18n.t("KV 容量不足：当前 ")+s.context+I18n.t("，负载需要 ")+context+I18n.t(" 个位置"));
 const limits=level.limits||{},checks=Object.entries(limits).map(([key,target])=>({key,target,value:metrics[key],pass:key==='tps'||key==='aneMin'?metrics[key]>=target:metrics[key]<=target}));
 return {metrics,timeline,errors:[...new Set(errors)],checks,pass:!errors.length&&checks.every(c=>c.pass),work};
}
M.engineReport=(state,level=M.levels[state.levelId])=>{
 const reports=(level.workloads||[{prompt:128,context:2048}]).map(w=>report(state,level,w));
 return {cacheTrace:M.cachedAttention(M.math.normalize(M.samplesFor(level.id)[0].v),state.settings.cache),runs:reports,pass:reports.every(r=>r.pass),metrics:reports[0].metrics,timeline:reports[0].timeline,errors:reports.flatMap(r=>r.errors)};
};
M.metricLabels={ttft:[I18n.t("首词"),'ms'],total:[I18n.t("首词 + 4 次解码"),'ms'],tps:[I18n.t("解码"),'token/s'],second:[I18n.t("第二轮首词"),'ms'],memory:[I18n.t("常驻内存"),'GB'],scratch:[I18n.t("临时内存"),'MB'],transfer:[I18n.t("交接成本"),'ms'],aneMin:[I18n.t("ANE 工作占比"),'%'],error:[I18n.t("最大数值误差"),'']};
M.engineCompute=engineCompute;M.quantize=quant;
})(window.NeuralPrologue);
