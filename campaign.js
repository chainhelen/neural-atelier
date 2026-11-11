/* Deterministic teaching model. Shapes, operators and budgets are deliberately small. */
(function(M){
const flat=rows=>rows.flat(),rows=(v,n=2)=>Array.from({length:v.length/n},(_,i)=>v.slice(i*n,i*n+n));
const mat=(v,w)=>flat(rows(v).map(r=>M.linear(r,w)));
const softmax=v=>{const max=Math.max(...v),e=v.map(x=>Math.exp(x-max)),sum=e.reduce((a,b)=>a+b,0);return e.map(x=>x/sum);};
const rotate=v=>flat(rows(v).map(([x,y],p)=>[x*Math.cos(p)-y*Math.sin(p),x*Math.sin(p)+y*Math.cos(p)]));
const scores=(q,k)=>flat(rows(q).map(a=>rows(k).map(b=>a.reduce((s,x,i)=>s+x*b[i],0)/Math.sqrt(2))));
const causal=s=>s.map((x,i)=>i%3>Math.floor(i/3)?-Infinity:x);
const probabilities=s=>flat(rows(s,3).map(softmax));
const mix=(a,v)=>flat(rows(a,3).map(r=>[0,1].map(j=>r.reduce((sum,p,i)=>sum+p*rows(v)[i][j],0))));
const normalize=v=>flat(rows(v).map(M.norm));
const ffn=v=>flat(rows(v).map(r=>{const z=M.norm(r),g=M.linear(z,M.weights.gate).map(M.silu),u=M.linear(z,M.weights.up);return M.linear(g.map((x,i)=>x*u[i]),M.weights.down);}));
const Wq=[[1,.25],[.5,1]],Wk=[[.8,-.5],[.2,1]],Wv=[[1,0],[0,2]],Wo=[[.9,.1],[-.2,.8]];
const emb=[[1,0],[0,1],[1,1],[-1,0],[0,-1],[-1,-1]];
const vocab=[I18n.t("我"),I18n.t("爱"),I18n.t("猫"),I18n.t("你"),I18n.t("恨"),I18n.t("狗")];
const head=[[1,0,1,-1,0,-1],[0,1,1,0,-1,-1]],bias=[0,0,-.75,0,0,-.75];
function logits(v){return flat(rows(v).map(r=>M.linear(r,head).map((x,i)=>x+bias[i])));}
function sample(v){return rows(v,6).map(r=>r.indexOf(Math.max(...r)));}
const decode=v=>sample(logits(v));
const attention=x=>mat(mix(probabilities(causal(scores(rotate(mat(x,Wq)),rotate(mat(x,Wk))))),mat(x,Wv)),Wo);
const attentionWeights=x=>probabilities(causal(scores(rotate(mat(x,Wq)),rotate(mat(x,Wk)))));
const block=x=>{const a=attention(normalize(x)),r=x.map((v,i)=>v+a[i]),f=ffn(r);return r.map((v,i)=>v+f[i]);};
const serial=(ids,source='R')=>Object.fromEntries(ids.map((id,i)=>[id,[i?ids[i-1]:source]]));
const def=(name,ins,out,type,hint)=>({name,ins,out,type,hint,x:0,z:0});
Object.assign(M.weights,{q:Wq,k:Wk,v:Wv,output:Wo,lmhead:head});
Object.assign(M.definitions,{
 embedding:def(I18n.t("词向量表"),[3],6,'embedding',I18n.t("Token ID 是词表中的编号。查表取出每个编号对应的两维向量。")),
 norm:def('RMSNorm',[6],6,'norm',I18n.t("把每个位置的一组数按同一比例缩放，整理整体尺度，这叫 RMSNorm。每个向量除以自己的均方根（含 ε）；本例缩放权重设为 1，维度与位置顺序不变。")),
 q:def(I18n.t("Q · 查询投影"),[6],6,'linear',I18n.t("每个位置生成查询。相同的权重用于所有位置。")),
 k:def(I18n.t("K · 键投影"),[6],6,'linear',I18n.t("每个位置生成供查询匹配的键。")),
 v:def(I18n.t("V · 值投影"),[6],6,'linear',I18n.t("每个位置生成被注意力加权汇合的内容。")),
 score:def(I18n.t("QKᵀ · 打分"),[6,6],9,'score',I18n.t("把每个 Q 与每个 K 点乘，再除以 √2，得到 3 × 3 分数。")),
 mask:def(I18n.t("因果遮罩"),[9],9,'mask',I18n.t("未来位置设为 −∞；Softmax 后这些位置的权重为 0。")),
 softmax:def('Softmax',[9],9,'softmax',I18n.t("每个查询位置的分数转成和为 1 的非负权重。")),
 ropeq:def(I18n.t("RoPE · 旋转 Q"),[6],6,'rope',I18n.t("让不同位置的 Q 转过不同角度，这是 RoPE 的旋转位置编码。本例位置 p 的向量旋转 p 弧度；真实模型有多对特征和不同频率。")),
 ropek:def(I18n.t("RoPE · 旋转 K"),[6],6,'rope',I18n.t("K 也按位置转动，才能与旋转后的 Q 一起比较相对位置。它使用与 Q 相同的 RoPE 规则；V 保持原样。")),
 aggregate:def(I18n.t("AV · 汇合内容"),[9,6],6,'mix',I18n.t("每个查询位置用注意力权重对各位置的 V 加权求和，得到该位置的输出向量。")),
 attentionweights:def(I18n.t("已完成的注意力权重"),[6],9,'attention',I18n.t("把前面学过的 Q/K 投影、RoPE、打分、遮罩和 Softmax 收在一起，输出 3 × 3 权重。每个查询位置的三个权重之和为 1。")),
 output:def(I18n.t("输出投影 Wo"),[6],6,'linear',I18n.t("汇合结果再乘 Wo，映射回残差流的特征宽度。")),
 attention:def(I18n.t("已完成的 Attention"),[6],6,'attention',I18n.t("让每个位置按比例读入自己和前面的内容，这里封装为 Attention。内部包括 Q/K/V、RoPE、因果注意力与输出投影；不含残差。双击可查看。")),
 residual1:def(I18n.t("Attention 残差"),[6,6],6,'residual',I18n.t("把 Attention 的变化量逐项加到原输入。")),
 ffnblock:def(I18n.t("已完成的 FFN"),[6],6,'ffn',I18n.t("各个位置分别加工自己的特征，不从其他位置取数，这是逐位置 FFN。本模块独立执行归一化、双路投影、SiLU、门控与降维，输出变化量 F，尚未加残差。")),
 residual2:def(I18n.t("FFN 残差"),[6,6],6,'residual',I18n.t("把 FFN 变化量加到 Attention 后的残差流。")),
 lmhead:def(I18n.t("词表投影"),[6],18,'vocabulary',I18n.t("每个位置的特征乘词表权重，加偏置，得到六个候选词的分数。")),
 pick:def(I18n.t("取最高分"),[18],3,'pick',I18n.t("演示采用贪心选择：每个位置选最高分的词表 ID。真实生成也可按概率采样。")),
 invert:def(I18n.t("反向投影"),[6],6,'invert',I18n.t("将每个特征乘 −1。本关用这个干扰模块练习定位计算错误。")),
 step:def(I18n.t("下一词模型"),[3],18,'decode',I18n.t("教学小模型：每个词的下一词分布由固定转移表计算。")),
 feedback:def(I18n.t("追加新词"),[3],3,'cache',I18n.t("将最新预测追加到上下文末尾，保留最后三个位置。")),
 load:def(I18n.t("读取权重"),[6],6,'storage',I18n.t("把权重从存储准备到统一内存，供后续计算重复读取。")),
 enembed:def('Embedding',[6],6,'embedding',I18n.t("读取词向量。在这个引擎关卡中输入已映射为两维特征。")),
 enqkv:def(I18n.t("QKV 投影"),[6],18,'attention',I18n.t("本关把三组投影作为一个调度任务，处理三个输入位置。")),
 enattn:def('Attention / KV',[18],6,'attention',I18n.t("按因果注意力汇合内容。解码时只为新位置计算 Q/K/V，旧 K/V 可缓存。")),
 enffn:def('SwiGLU FFN',[6],6,'ffn',I18n.t("对每个位置做两路投影、激活、门控和降维，再接回残差。")),
 enhead:def(I18n.t("词表与采样"),[6],6,'vocabulary',I18n.t("得到词表分数并取最高分；调度任务仍返回数值，结果区显示预测。"))
});
M.extraColors={embedding:0x7d9ea5,norm:0xb8a36a,q:0x789abf,k:0xa394bd,v:0x5da995,score:0x879bb3,mask:0xb4a27d,softmax:0xc9a068,ropeq:0x789abf,ropek:0xa394bd,aggregate:0x64a68f,output:0x6c9a92,attention:0x729ca3,residual1:0xc3a26f,ffnblock:0x9a94b4,residual2:0xc3a26f,lmhead:0xb79b6b,pick:0x639b7a,invert:0xb88475,step:0x7f9aa7,feedback:0xb49769,load:0xa8a18c,enembed:0x7d9ea5,enqkv:0x789abf,enattn:0xa394bd,enffn:0x5da995,enhead:0xb79b6b};
M.extraColors.attentionweights=0xc9a068;
M.definitions.lmhead.bias=bias;
const next=[1,2,0,4,5,3];
M.operations={embedding:([ids])=>flat(ids.map(i=>{if(!emb[i])throw {message:I18n.t("词表中找不到该 Token ID。请使用词表内的编号。")};return emb[i];})),norm:([v])=>normalize(v),q:([x])=>mat(x,Wq),k:([x])=>mat(x,Wk),v:([x])=>mat(x,Wv),score:([q,k])=>scores(q,k),mask:([s])=>causal(s),softmax:([s])=>probabilities(s),ropeq:([q])=>rotate(q),ropek:([k])=>rotate(k),aggregate:([a,v])=>mix(a,v),output:([x])=>mat(x,Wo),attention:([x])=>attention(x),residual1:([a,b])=>a.map((x,i)=>x+b[i]),ffnblock:([x])=>ffn(x),residual2:([a,b])=>a.map((x,i)=>x+b[i]),lmhead:([x])=>logits(x),pick:([x])=>sample(x),invert:([x])=>x.map(n=>-n),step:([ids])=>flat(ids.map(i=>vocab.map((_,j)=>j===next[i]?4:-1))),feedback:([ids],ctx)=>[...ctx.input.slice(1),ids[2]],load:([x])=>x,enembed:([x])=>x,enqkv:([x])=>{const z=normalize(x);return [...mat(z,Wq),...mat(z,Wk),...mat(z,Wv)];},enattn:([x])=>mat(mix(probabilities(causal(scores(rotate(x.slice(0,6)),rotate(x.slice(6,12))))),x.slice(12,18)),Wo),enffn:([x])=>x.map((v,i)=>v+ffn(x)[i]),enhead:([x])=>x};
const sequenceSamples=[{word:I18n.t("我 / 爱 / 猫"),v:[1,0,0,1,1,1]},{word:I18n.t("你 / 恨 / 狗"),v:[-1,0,0,-1,-1,-1]},{word:I18n.t("猫 / 我 / 爱"),v:[1,1,1,0,0,1]}];
M.operations.attentionweights=([x])=>attentionWeights(x);
const idSamples=[{word:I18n.t("我 / 爱 / 猫"),v:[0,1,2]},{word:I18n.t("你 / 恨 / 狗"),v:[3,4,5]},{word:I18n.t("猫 / 我 / 爱"),v:[2,0,1]}];
const extra={};
function add(id,number,title,nodes,mission,description,options={}){
 const count=nodes.length,positions={};
 nodes.forEach((k,i)=>{positions[k]=count<=3?[(i-(count-1)/2)*3.5,0]:count<=6?[-4.5+(i%3)*4.3,Math.floor(i/3)*4-2]:[-5.5+(i%4)*3.4,Math.floor(i/4)*3.5-2.6];});
 extra[id]={id,chapter:I18n.t("网络 / ")+String(number).padStart(2,'0'),title,nodes,sources:['R'],sourceDim:6,sinkDim:6,sampleCount:3,sequence:true,sourceX:-8.8,sinkX:9,zoom:1.07,boardWidth:23,boardDepth:11,positions,mission,description,summary:description,concept:title,sourceTitle:I18n.t("输入特征 X"),sourceNote:I18n.t("三个输入位置各有一份特征"),outputTitle:I18n.t("结果特征"),shape:'3 × 2',boardText:'NETWORK / '+String(number).padStart(2,'0'),next:null,hints:[I18n.t("双击模块查看输入、输出和计算。运行网络后，在出口比较实际值与目标值。")],...options};
}
add('embedding',1,I18n.t("按 Token ID 查找词向量"),['embedding'],I18n.t("输出三个 Token ID 对应的两维词向量"),I18n.t("连接词向量表，查找三个 ID 的对应向量。编号范围是 0 至 5；出口应得到每词两个特征。"),{outputArrivalHold:450,sourceDim:3,sinkDim:6,sourceTitle:'Token ID',sourceNote:I18n.t("词表编号：0 至 5"),outputTitle:I18n.t("词向量 X"),shape:I18n.t("3 个 ID → 3 × 2")});
add('normalize',2,I18n.t("先整理数值尺度"),['norm'],I18n.t("调整每组数的尺度，保留特征间的比例"),I18n.t("每个位置独立做 RMSNorm，调整数值尺度，不混合不同位置的特征。"),{initialSink:'R',outputTitle:I18n.t("归一化特征 Z")});
add('qkv',3,I18n.t("同一份输入，三个角色"),['q','k','v'],I18n.t("把内容向量送到出口"),I18n.t("Q 用来发起查询，K 用来匹配，V 提供内容。本关目标是 V；三块权重各不相同。"),{initialLinks:{q:['R'],k:['R'],v:['R']},initialSink:'q',outputTitle:I18n.t("内容 V")});
add('scores',4,I18n.t("查询在找谁"),['q','k','score'],I18n.t("得到每个 Q 对每个 K 的分数"),I18n.t("将三个位置的 Q 分别与三个位置的 K 配对，算出九个相关分数。输入口顺序是 Q、K。"),{initialLinks:{q:['R'],k:['R']},sinkDim:9,outputTitle:I18n.t("相关分数 S"),shape:'3 × 2 → 3 × 3'});
add('causal',5,I18n.t("不要偷看未来"),['q','k','score','mask','softmax'],I18n.t("让未来位置的注意力为零"),I18n.t("一次处理整句时，第一个位置也不能读取后面的词。让每个位置只分配权重给自己和过去。"),{initialLinks:{q:['R'],k:['R'],score:['q','k'],softmax:['score']},initialSink:'softmax',sinkDim:9,outputTitle:I18n.t("注意力权重 A"),shape:'3 × 3',attentionReadout:true});
add('rope',6,I18n.t("让位置进入匹配"),['q','k','ropeq','ropek','score'],I18n.t("让匹配也考虑位置差异"),I18n.t("同样的特征，放在不同位置会怎样匹配？试着把 RoPE 加入 Q/K 的比较，得到 3 × 3 分数；本例每个位置增加 1 弧度，V 不参与旋转。"),{initialLinks:{q:['R'],k:['R'],score:['q','k']},initialSink:'score',sinkDim:9,outputTitle:I18n.t("带位置的分数"),shape:'3 × 2 → 3 × 3'});
add('attention',7,I18n.t("把关注变成内容"),['q','k','v','ropeq','ropek','score','mask','softmax','aggregate','output'],I18n.t("让每个位置收到它能参考的内容"),I18n.t("把前面试过的小计算组合成一次因果 Attention。这次出口不再是关注比例，而是每个位置汇入内容后的两个特征；包含输出投影，暂不加残差。"),{initialLinks:{q:['R'],k:['R'],v:['R'],ropeq:['q'],ropek:['k'],score:['ropeq','ropek'],mask:['score']},attentionReadout:true,zoom:1.22,boardWidth:26,boardDepth:14,sourceX:-10.3,sinkX:10.3});
add('block',8,I18n.t("组装现代解码器块"),['norm','attention','residual1','ffnblock','residual2'],I18n.t("交流信息、加工特征，再把变化加回来"),I18n.t("用已完成的 Attention 与 FFN 搭出一层，让每段的变化都能接回原路。注意两个模块的边界：Attention 不含归一化，FFN 已含；出口要保留两次更新。"),{initialLinks:{norm:['R'],attention:['norm']},outputTitle:I18n.t("本层特征 Y")});
add('vocabulary',9,I18n.t("让特征真正变成词"),['lmhead','pick','invert'],I18n.t("让词表分数读出「我爱猫」"),I18n.t("这一关才把特征变成文字。词表投影为每个候选词算分，取最高分。改变特征会改变文字；词一样也不保证全部数值正确。"),{initialLinks:{invert:['R'],lmhead:['invert'],pick:['lmhead']},initialSink:'pick',sinkDim:3,outputTitle:I18n.t("预测 Token ID"),shape:'3 × 2 → 3 × 6 → 3',tokenOutput:true});
add('autoregressive',10,I18n.t("把刚生成的词送回来"),['step','pick','feedback'],I18n.t("让上下文前进一步"),I18n.t("连接上下文更新步骤，让出口得到追加新词后的最近三个词，而不是三个独立预测。连续运行可观察解码循环。"),{sourceDim:3,initialLinks:{step:['R'],pick:['step']},initialSink:'pick',sinkDim:3,outputTitle:I18n.t("更新后的上下文"),shape:I18n.t("上下文 → 下一词 → 追加"),tokenOutput:true});
add('softmax',4,I18n.t("把分数变成比例"),['q','k','score','softmax'],I18n.t("让每组匹配分数变成和为 1 的权重"),I18n.t("Q/K 和打分已经接好。本关只加入 Softmax，把可能为负的分数变成非负比例，先不遮住未来位置。"),{initialLinks:{q:['R'],k:['R'],score:['q','k']},initialSink:'score',sinkDim:9,outputTitle:I18n.t("注意力权重 A"),shape:'3 × 3 → 3 × 3',attentionReadout:true});
add('rotation',6,I18n.t("先转动一份 Q"),['q','ropeq'],I18n.t("让不同位置的 Q 转过不同角度"),I18n.t("Q 投影已经接好。试着让第一个位置不转，后两个位置分别转 1、2 弧度，看看数值怎样变化。本关只处理 Q，不计算匹配分数。"),{initialLinks:{q:['R']},initialSink:'q',outputTitle:I18n.t("旋转后的 Q"),shape:'3 × 2 → 3 × 2',boardWidth:18,boardDepth:8,sourceX:-6.5,sinkX:6.5,zoom:.9});
add('aggregation',8,I18n.t("按比例汇入内容"),['attentionweights','v','aggregate'],I18n.t("把各处内容按比例取来，加在一起"),I18n.t("比例已经算好，V 也准备好了。现在让每个位置按自己的比例取来内容，合成两个新特征；暂不做输出投影。"),{initialLinks:{attentionweights:['R'],v:['R']},outputTitle:I18n.t("汇合特征 O"),shape:'3 × 3 + 3 × 2 → 3 × 2',attentionReadout:true,boardWidth:20,boardDepth:9,sourceX:-7.5,sinkX:7.5,zoom:.96});
add('vocabulary-scores',11,I18n.t("先给候选词打分"),['lmhead'],I18n.t("让每个位置得到六个候选词分数"),I18n.t("词表投影仍然是矩阵运算，只是每个输出对应一个候选词。先看清这些分数，下一关再决定选哪个词。"),{sinkDim:18,outputTitle:I18n.t("词表分数 L"),shape:'3 × 2 → 3 × 6',boardWidth:18,boardDepth:8,sourceX:-6.5,sinkX:6.5,zoom:.9});
const engineNodes=['load','enembed','enqkv','enattn','enffn','enhead'];
const engineBase={engine:true,sequence:true,initialLinks:serial(engineNodes),initialSink:'enhead',devices:{load:'cpu',enembed:'cpu',enqkv:'gpu',enattn:'gpu',enffn:'gpu',enhead:'cpu'},settings:{},sampleCount:3,boardWidth:25,boardDepth:13,zoom:1.2,sourceX:-10.5,sinkX:10.5,outputTitle:I18n.t("输出特征 / 预测词"),sourceTitle:I18n.t("提示词特征"),sourceNote:I18n.t("调度不同，计算目标相同"),shape:'CPU · GPU · ANE',tokenReadout:true};
function engine(id,num,title,mission,description,options){add(id,num,title,engineNodes,mission,description,{...engineBase,...options});extra[id].chapter=I18n.t("番外 / ")+String(num).padStart(2,'0');extra[id].boardText='ENGINE / '+String(num).padStart(2,'0');}
engine('placement',1,I18n.t("把工作交给合适的芯片"),I18n.t("让第一个词早点出现，结果不变"),I18n.t("把任务交给 CPU、GPU 或 ANE，看看怎样分工能更早得到第一个词。比较完整流程的模拟耗时；结果要正确，设备也要能接下分到的任务。\n\n模拟目标：结果正确，模拟首词时间 ≤ 110 ms。"),{controls:[],settings:{batch:16,layout:'channels',precision:8,cache:true,fusion:true,resident:true,prefix:false,split:0},limits:{ttft:110},devices:Object.fromEntries(engineNodes.map(k=>[k,'cpu']))});
engine('layout',2,I18n.t("让 ANE 读懂布局"),I18n.t("让 ANE 用上排列合适的数据"),I18n.t("同样一组数，换个排列方式，ANE 能不能接着算？调整张量布局，满足模拟首词时间和 ANE 工作占比目标。本关只做教学模型的布局检查，不执行真实设备编译。\n\n模拟目标：布局兼容，模拟首词时间 ≤ 110 ms，ANE 工作占比 ≥ 10%。"),{controls:['layout'],devices:{...engineBase.devices,enqkv:'ane',enffn:'ane'},settings:{batch:16,layout:'sequence',precision:8,cache:true,fusion:true,resident:true,split:0},limits:{ttft:110,aneMin:10}});
engine('batching',3,I18n.t("一次多算几个位置"),I18n.t("一次多算一些，也别撑满临时内存"),I18n.t("读入提示词时，一批算几个位置更合适？调整 Prefill 批次，比较模拟首词时间与临时内存，还要达到 ANE 工作占比目标。小批次派发更多，大批次要放下更多中间结果。\n\n模拟目标：首词时间 ≤ 105 ms，临时内存 ≤ 650 MB，ANE 工作占比 ≥ 10%。"),{controls:['batch'],devices:{...engineBase.devices,enqkv:'ane',enffn:'ane'},settings:{batch:1,layout:'channels',precision:8,cache:true,fusion:true,resident:true,split:0},limits:{ttft:105,scratch:650,aneMin:10}});
engine('kv-cache',4,I18n.t("过去的 K/V 留下来"),I18n.t("少算重复的部分，更快生成下一个词"),I18n.t("每次只生成一个新词，前面算过的内容还要重算吗？比较保留与重算历史 K/V 的方案，让输出一致，并达到模拟解码速度目标。\n\n模拟目标：解码速度 ≥ 45 token/s，输出必须一致。"),{controls:['cache'],settings:{batch:16,layout:'channels',precision:8,cache:false,fusion:true,resident:true,split:0},limits:{tps:45}});
engine('quantization',5,I18n.t("把权重装进内存"),I18n.t("少占内存，也别让数值偏得太远"),I18n.t("每个权重少用几位，能省下多少空间，又会差多少？调整精度与上下文长度，同时满足模拟内存和数值误差限制。内存包含权重、元数据、KV 与运行空间；教学权重会实际量化，候选词排名可能变化。\n\n模拟目标：常驻内存 ≤ 24 GB，最大数值误差 ≤ 0.30。"),{controls:['precision','context'],settings:{batch:16,layout:'channels',precision:16,cache:true,fusion:true,resident:true,split:0,context:2048},limits:{memory:24,error:.30}});
engine('fusion',6,I18n.t("少写一次，少搬一趟"),I18n.t("少搬数据，让整段计算早点完成"),I18n.t("相邻计算能不能少写一次中间结果，权重能不能留着继续用？试试融合与权重驻留，满足模拟总时间和交接限制。统一内存也不代表读写、同步和布局转换免费。\n\n模拟目标：首词时间 + 4 次解码 ≤ 180 ms，交接耗时 ≤ 14 ms。"),{controls:['fusion','resident'],settings:{batch:16,layout:'channels',precision:8,cache:true,fusion:false,resident:false,split:0},limits:{total:180,transfer:14}});
engine('parallel',7,I18n.t("GPU 与 ANE 一起工作"),I18n.t("让两块芯片分担工作，一起早点完成"),I18n.t("两块芯片一起算，怎样分工才不互相等待？调整投影与 FFN 的 ANE 分担比例及 Prefill 批次，让模拟首词时间和 ANE 工作占比都达标；同步也会花时间。\n\n模拟目标：首词时间 ≤ 78 ms，ANE 工作占比 ≥ 20%。"),{controls:['split','batch'],settings:{batch:16,layout:'channels',precision:8,cache:true,fusion:true,resident:true,split:0},limits:{ttft:78,aneMin:20}});
engine('session',8,I18n.t("下一轮不用从头读"),I18n.t("沿用相同的开头，让第二轮更快"),I18n.t("两轮提示有相同的开头，上一轮的计算能留下多少？KV 缓存已开启，试试前缀复用，再调整上下文长度与精度，让模拟第二轮首词时间、内存和误差都达标。\n\n模拟目标：第二轮首词时间 ≤ 60 ms，内存 ≤ 24 GB，最大数值误差 ≤ 0.30。"),{controls:['prefix','context','precision'],settings:{batch:16,layout:'channels',precision:8,cache:true,fusion:true,resident:true,split:40,prefix:false,context:2048},limits:{second:60,memory:24,error:.30}});
engine('capstone',9,I18n.t("换一种负载也能跑"),I18n.t("让两种负载都满足正确性、内存和速度限制"),I18n.t("提示更长、上下文更多，刚才的办法还管用吗？调整分工与内存设置，让两种负载都达标。模拟限制：首词 ≤ 110 ms，解码 ≥ 45 token/s，内存 ≤ 24 GB，误差 ≤ 0.30，第二轮首词 ≤ 65 ms，ANE 工作占比 ≥ 15%，临时内存 ≤ 650 MB。"),{controls:['layout','batch','cache','precision','fusion','resident','split','prefix','context'],settings:{batch:1,layout:'sequence',precision:16,cache:false,fusion:false,resident:false,split:0,prefix:false,context:2048},limits:{ttft:110,tps:45,memory:24,error:.30,second:65,aneMin:15,scratch:650},workloads:[{prompt:128,context:2048},{prompt:256,context:4096}],devices:Object.fromEntries(engineNodes.map(k=>[k,'cpu']))});
const rationales={
 softmax:{title:I18n.t("为什么分数不能直接当比例？"),text:I18n.t("打分可能为负，大小和总和也不固定。Softmax 先把它们变成正数，再除以总和，让每个查询位置的权重加起来为 1。"),takeaway:I18n.t("相同的三个分数会得到各约 1/3；其中一个更高，它分到的比例就更大。本关只计算比例，不限制可见位置；因果遮罩才负责排除未来位置。")},
 rotation:{title:I18n.t("旋转给向量带来了什么？"),text:I18n.t("旋转保留向量长度，并可改变方向。例如 [1, 0] 转过 90° 会成为 [0, 1]。按位置使用不同角度，让相同特征可以带上位置差异。"),takeaway:I18n.t("本关只看旋转本身。下一关同时旋转 Q 和 K，它们的夹角差才会把相对位置带入匹配分数。"),scope:I18n.t("RoPE 是后来的位置编码方法。2017 年原始 Transformer 使用加到词向量上的正弦位置编码。")},
 aggregation:{title:I18n.t("有了比例，为什么还要汇合？"),text:I18n.t("权重只回答“各取多少”，V 才提供要取的特征。把每份 V 乘上它的权重，再将这些向量相加，这叫加权求和。当前位置由此收到可见位置带来的内容。"),takeaway:I18n.t("例如取 [2, 0] 的 25%，再取 [0, 4] 的 75%，得到 [0.5, 3]。这里需要把不同来源加起来，和前面只配对、不求和的逐元素相乘不同。")},
 'vocabulary-scores':{title:I18n.t("为什么要先得到整组候选分数？"),text:I18n.t("网络中的两个特征还不是文字。词表投影为每个候选词准备一组权重，计算这个特征向量对各个候选词的分数。"),takeaway:I18n.t("六个分数对应“我、爱、猫、你、恨、狗”六个候选。分数可以为负，尚未做选择，也还不是概率。")},
 embedding:{
  title:I18n.t("为什么编号要换成向量？"),
  text:I18n.t("Token ID 只是查表地址，编号相邻不代表含义相近。词向量表把编号换成可参与矩阵计算的特征，后面的层才能处理这些特征之间的关系。"),
  takeaway:I18n.t("真实词向量来自训练；这里的六个向量用于演示查表，不代表已经学会这些词的含义。")
 },
 normalize:{
  title:I18n.t("为什么要整理数值尺度？"),
  text:I18n.t("把一组数一起放大或缩小，可以调整尺度而保留它们的比例。RMSNorm（均方根归一化）用每个向量自己的均方根做这件事，为后续投影提供较一致的尺度。本例缩放权重为 1。"),
  takeaway:I18n.t("例如 [3, 4] 除以 √((9 + 16) / 2)，约为 [0.85, 1.13]。特征间的比例保留，整体尺度得到调整。")
 },
 qkv:{
  title:I18n.t("为什么同一份输入要分成 Q/K/V？"),
  text:I18n.t("两个位置如何匹配，和匹配后传递哪些内容，需要不同的特征表示。Q/K 用来比较，V 则提供真正要汇入的内容；三套权重可以分别学习这两种用途。"),
  takeaway:I18n.t("给某个位置很高的注意力权重时，汇入的是它的 V。Q/K 的数参与计算权重，不直接充当这份内容。")
 },
 scores:{
  title:I18n.t("为什么先算匹配分数？"),
  text:I18n.t("当前位置需要决定从哪些位置取信息。它的 Q 与各位置的 K 做点积，得到一组随输入变化的匹配分数，为后面的权重分配提供依据。"),
  takeaway:I18n.t("点积再除以 √维度，控制高维点积的尺度，避免 Softmax 过度集中。分数可以为负，还不是概率。")
 },
 causal:{
  title:I18n.t("为什么不能读取未来的词？"),
  text:I18n.t("生成下一个词时，模型只有已经出现的内容。即使训练或 Prefill 一次处理整段文本，每个位置也要遵守同样的限制，否则会用到生成时尚不存在的信息。"),
  takeaway:I18n.t("在“我 / 爱 / 猫”中，“爱”这个位置可以参考“我”和自己。未来的“猫”被遮住，经过 Softmax 后其权重为 0。")
 },
 rope:{
  title:I18n.t("为什么位置会影响匹配？"),
  text:I18n.t("让相同特征随位置转过不同角度，就是 RoPE（旋转位置编码）的思路。Q 和 K 都按位置旋转后，它们的点积能包含相对位置的信息，为注意力提供词序线索，而不只比较特征本身。"),
  takeaway:I18n.t("相隔一格和两格，旋转角度差不同，匹配分数可能不同，但不保证每次变化。Q/K 的位置同时平移时，相对角度不变。本关用一对特征演示；真实 RoPE 在多对特征上使用不同频率。")
 },
 attention:{
  title:I18n.t("为什么算完权重还要汇合 V？"),
  text:I18n.t("注意力权重决定各位置的信息占比，但权重本身还不是新的词特征。按这些比例加权 V，当前位置才真正得到其他位置带来的内容。"),
  takeaway:I18n.t("例如权重 [0.2, 0.8, 0] 表示取第一份 V 的 20%、第二份的 80%。输出投影再把汇合结果转换成下一步需要的特征表示。")
 },
 block:{
  title:I18n.t("为什么一层里要有两种加工？"),
  text:I18n.t("先从别的位置取信息，再加工自己已有的特征：前者是 Attention，后者是逐位置 FFN。两段都把修改量通过残差加回各自的输入，形成可以交给下一层的新特征。"),
  takeaway:I18n.t("第二次残差以 Attention 更新后的特征为基础。如果又接回最早的输入，就会漏掉前一步已经加入的修改。")
 },
 vocabulary:{
  title:I18n.t("为什么最后还要经过词表？"),
  text:I18n.t("网络里流动的是特征向量，文字要在输出端通过词表读出。词表投影把特征换成每个候选 token 的分数，再由选词规则决定输出哪个。"),
  takeaway:I18n.t("本关总选最高分，用来观察特征如何改变文字。真实生成也可以按概率采样，因此同一组分数不一定每次选出同一个词。")
 },
 autoregressive:{
  title:I18n.t("为什么生成的词要送回来？"),
  text:I18n.t("下一词的预测依赖当前上下文，而刚生成的词会成为下一次预测的条件。把新词追加回来，模型才能继续生成句子。"),
  takeaway:I18n.t("首次处理提示词称为 Prefill；随后反复预测并追加一个新词，就是 Decode。本关用三词窗口和固定转移规则演示这个循环。")
 },
 placement:{
  title:I18n.t("为什么不能全交给最快的芯片？"),
  text:I18n.t("把任务交给擅长它的设备，并安排好交接，叫异构调度。一次推理既有矩阵运算，也有取数和控制；某一步算得更快，收益可能被派发与同步抵消，所以要比较完整链路的首词时间。"),
  takeaway:I18n.t("本关把 CPU、GPU、ANE 的算子支持和交接成本分别计入，让你比较不同分工带来的总耗时。")
 },
 layout:{
  title:I18n.t("为什么形状对了还不够？"),
  text:I18n.t("一组数怎样按轴排列、怎样放进内存，叫张量布局。算子按约定的布局取数；布局不匹配时，即使元素数量相同，也可能需要重排数据、换算子或回退，增加开销。"),
  takeaway:I18n.t("本关用 B/C/1/S 演示 ANE 投影适配。相邻算子采用兼容的布局，才有机会省去反复转换，让加速单元真正用起来。")
 },
 batching:{
  title:I18n.t("为什么 Prefill 可以成批计算？"),
  text:I18n.t("把提示词的多个位置放在一批里算，就是这里的 Prefill 批处理。提示词各位置已经给定，投影可以让它们一起使用同一套权重，复用权重、摊薄派发成本，提高矩阵单元的利用率。"),
  takeaway:I18n.t("批次越大，同时存放的中间特征也越多。本关需要在减少派发次数和控制临时内存之间取舍。")
 },
 'kv-cache':{
  title:I18n.t("为什么过去的 K/V 可以复用？"),
  text:I18n.t("把过去算好的键和值留下来，叫 KV 缓存。因果注意力中，追加一个词不会改变过去位置已经算出的 K/V；每步只补新位置，就能避免重复投影整段历史。"),
  takeaway:I18n.t("新 Q 仍要读取历史 K/V 来取信息；缓存节省的是重复计算，同时占用内存。这是用空间换解码速度。")
 },
 quantization:{
  title:I18n.t("为什么降低位数能省内存？"),
  text:I18n.t("把权重近似到更少位数能表示的数值，叫量化。27B 个权重仅按 16 位存放，就约需 54 GB，还没计入 K/V 和运行空间；降低位数能减少存储容量和读取它们的数据量。"),
  takeaway:I18n.t("可表示的数值变少，就需要舍入。误差会传到后续计算，甚至改变候选词的排名，所以省内存的同时还要检查输出误差。")
 },
 fusion:{
  title:I18n.t("为什么少写一次内存也能更快？"),
  text:I18n.t("让相邻计算一起完成、少把中间结果写回内存，叫算子融合。激活和门控分开执行时，前一步写出的结果要被下一步重新读取；融合可以减少这些写回、读取和派发，即使乘加次数相近也可能更快。"),
  takeaway:I18n.t("权重驻留则省去反复从存储加载的开销。它保留的是内存中的权重，每一步计算仍然要读取这些权重。")
 },
 parallel:{
  title:I18n.t("为什么两块芯片一起做未必最快？"),
  text:I18n.t("把可拆分的投影或 FFN 分给 GPU 和 ANE 同时做，就是异构并行。但后续步骤要等两边都结束；分配不均时，快的一边会空等，同步还会额外花时间。"),
  takeaway:I18n.t("比较分担比例时，要看两边各自的完成时间和交接成本。ANE 工作占比变高，并不自动意味着首词更快。")
 },
 session:{
  title:I18n.t("为什么新一轮能接着上次算？"),
  text:I18n.t("新请求开头相同，就沿用那一段已经算好的 K/V，这叫前缀复用。它要求 KV 缓存已开启，而且模型、token 前缀和位置都匹配，才能省去新一轮对应部分的 Prefill。"),
  takeaway:I18n.t("这把生成过程中的 K/V 复用延伸到多轮请求。保留更长前缀需要更多 K/V 内存，第二轮更快也必须装得下。")
 },
 capstone:{
  title:I18n.t("为什么要用两种负载验收？"),
  text:I18n.t("长提示更考验 Prefill，多轮长上下文更考验 K/V 容量和复用。只在一种输入上表现好的配置，换一种负载就可能变慢或超出内存。"),
  takeaway:I18n.t("首词时间、逐词速度、内存和误差需要一起检查。每项优化都有代价，完整引擎要让这些约束在两种负载下同时成立。")
 }
};
for(const [id,rationale]of Object.entries(rationales)){
 if(extra[id].engine)rationale.scope=I18n.t("耗时、内存和芯片支持范围采用教学模型，用于比较方案，不是设备实测。");
 extra[id].rationale=rationale;
}
Object.assign(M.levels,extra);
for(const [id,maxModules]of [['ffn',8],['attention',12],['block',7]]){
 const level=M.levels[id];level.buildable=true;level.maxModules=maxModules;
 level.inventory=Object.fromEntries(level.nodes.map(kind=>[kind,2]));
}
// Combine the two formerly identical stocks without reducing existing circuits' capacity.
M.levels.ffn.inventory.project=4;
// Teach individual operations before assembling modern FFNs and Attention.
// Level IDs and saved circuits remain stable when their chapter changes.
const chapters=[
 {prefix:I18n.t("序章 / "),board:'PROLOGUE',ids:['wire','projection','activation','down-projection','plain-ffn','residual','workshop']},
 {prefix:I18n.t("FFN 进阶 / "),board:'FFN',ids:['gate-projection','pairwise','gating','swiglu','normalize','ffn']},
 {prefix:I18n.t("网络 / "),board:'NETWORK',ids:['embedding','qkv','scores','softmax','causal','rotation','rope','aggregation','attention','block','vocabulary-scores','vocabulary','autoregressive']}
];
chapters.forEach(({prefix,board,ids})=>ids.forEach((id,i)=>{M.levels[id].chapter=prefix+String(i+1).padStart(2,'0');M.levels[id].boardText=board+' / '+String(i+1).padStart(2,'0');M.levels[id].practice=board==='PROLOGUE';}));
M.levels.residual.concept=I18n.t("在已有结果上加入残差");
M.levels.residual.description=I18n.t("用两块熟悉的投影算出修改量 F，再用残差相加得到 X + F。桌上的模块都还没有接线，五组输入都要匹配。");
M.levels.residual.rationale.scope=I18n.t("本关用线性支路提供修改量，重点练 X + F；也可以把这条支路换成上一关的 FFN。真实网络中的直连路径还有助于训练信号回传。");
M.levels.gating.summary=I18n.t("让门分支产生随输入变化的系数，逐项调节内容特征。");
M.levels.gating.description=I18n.t("两条投影支路已经接好。加入 SiLU，让一条支路算出的系数调节另一条的内容。试试激活放在哪一路，五组输入都要匹配目标。");
M.levels.normalize.description=I18n.t("三个位置各有一组数。试试 RMSNorm 怎样整理它们的尺度：各位置独立缩放，维度和位置顺序不变；本关缩放权重为 1。");
M.levels.normalize.rationale.scope=I18n.t("这里学的是 RMSNorm，方便理解下一关的 Z 从哪里来。2017 年原始 Transformer 使用 LayerNorm，两者不是同一个运算。");
M.levels.ffn.description=I18n.t("把已学过的 SwiGLU、归一化和残差组合起来。R 是原输入，Z = RMSNorm(R) 已由上游准备好；从组件架搭建，让出口得到更新后的特征。");
M.levels.ffn.sourceNote=I18n.t("R 是原输入，Z 是它归一化后的结果");
M.levels.ffn.rationale.text=I18n.t("先用归一化输入加工出 FFN 的修改量，再加回原输入。这里将前几关各自完成的计算接起来；后面学习 Attention 后，还会把这段放进完整网络层。");
M.routes={main:{name:I18n.t("主线 · Transformer"),description:I18n.t("从向量与矩阵开始，亲手搭出 Attention、FFN 和逐词生成。"),ids:Object.keys(M.levels).filter(id=>!M.levels[id].engine)},engine:{name:I18n.t("番外 · 端侧推理"),description:I18n.t("以 Qwen 27B、Kapitsa 与 Apple ANE 为背景的独立挑战，使用教学成本模型。"),ids:Object.keys(M.levels).filter(id=>M.levels[id].engine)}};
M.routes.main.ids=chapters.flatMap(chapter=>chapter.ids);
M.routes.main.description=I18n.t("先练单路计算与残差，再进入 SwiGLU 进阶，最后组装 Attention 与逐词生成。每段先学单个运算，再做综合关。");
for(const [track,route]of Object.entries(M.routes))route.ids.forEach((id,i)=>{M.levels[id].track=track;M.levels[id].next=route.ids[i+1]||null;});
M.samplesFor=id=>M.levels[id]?.sequence?(M.levels[id].sourceDim===3?idSamples:sequenceSamples):M.samples;
const oldReference=M.reference;
M.reference=(v,id)=>{
 if(!extra[id])return oldReference(v,id);
 const q=mat(v,Wq),k=mat(v,Wk),vv=mat(v,Wv);
 if(id==='embedding')return flat(v.map(i=>emb[i]));
 if(id==='normalize')return normalize(v);
 if(id==='qkv')return vv;
 if(id==='scores')return scores(q,k);
 if(id==='softmax')return probabilities(scores(q,k));
 if(id==='causal')return probabilities(causal(scores(q,k)));
 if(id==='rotation')return rotate(q);
 if(id==='rope')return scores(rotate(q),rotate(k));
 if(id==='aggregation')return mix(attentionWeights(v),vv);
 if(id==='attention')return attention(v);
 if(id==='block')return block(v);
 if(id==='vocabulary')return decode(v);
 if(id==='vocabulary-scores')return logits(v);
 if(id==='autoregressive')return [...v.slice(1),next[v[2]]];
 if(M.levels[id].engine){const a=attention(normalize(v));return a.map((x,i)=>x+ffn(a)[i]);}
 throw new Error('Unknown level '+id);
};
// Incremental attention caches only K and V. Absolute position keeps RoPE consistent.
M.cachedAttention=(input,useCache=true)=>{
 const {rows}=M.math;let keys=[],values=[],processed=0,outputs=[];
 rows(input).forEach((row,p)=>{
  const rotation=(a,pos)=>[a[0]*Math.cos(pos)-a[1]*Math.sin(pos),a[0]*Math.sin(pos)+a[1]*Math.cos(pos)];
  const history=rows(input).slice(0,p+1);
  if(useCache){keys.push(rotation(M.linear(row,Wk),p));values.push(M.linear(row,Wv));processed++;}
  else{keys=history.map((r,j)=>rotation(M.linear(r,Wk),j));values=history.map(r=>M.linear(r,Wv));processed+=history.length;}
  const q=rotation(M.linear(row,Wq),p),a=softmax(keys.map(k=>(q[0]*k[0]+q[1]*k[1])/Math.sqrt(2)));
  outputs.push(M.linear([0,1].map(j=>a.reduce((sum,w,i)=>sum+w*values[i][j],0)),Wo));
 });return {output:outputs.flat(),processed,keys,values};
};
M.readTokens=v=>I18n.joinWords(v.map(i=>vocab[i]??'?'));
M.rollout=(state,input,count=6)=>{let context=[...input],trace=[];for(let i=0;i<count;i++){const ev=M.evaluate(state,context);if(ev.error||!ev.output)return {trace,error:ev.error||I18n.t("未生成更新后的上下文。请检查出口连线。")};const token=ev.output[2];trace.push({context:[...context],token,next:[...ev.output]});context=ev.output;}return {trace,text:I18n.joinWords([...input.map(i=>vocab[i]),...trace.map(t=>vocab[t.token])])};};
M.decodeFeatures=v=>M.readTokens(decode(v));
M.attentionWeights=ev=>ev.values.softmax||null;
M.extraCard=(id)=>{const d=M.definitions[id];if(!d)return null;return {number:'',seal:d.name[0],english:id.toUpperCase(),family:I18n.t("计算模块"),color:'#477f78',ability:d.name,formula:{q:'Q = XWq',k:'K = XWk',v:'V = XWv',score:'S = QKᵀ / √2',mask:'S[i,j] = −∞  (j > i)',softmax:'Aᵢ = exp(Sᵢ) / Σ exp(Sⱼ)',ropeq:'Q[p] ← R(p) Q[p]',ropek:'K[p] ← R(p) K[p]',aggregate:'O = AV',norm:'x / √(mean(x²) + ε)',output:'O ← OWo',lmhead:I18n.t("L = XW词表 + b"),pick:'token = argmax(L)',invert:'Y = −X',residual1:'R = X + Attention(Norm(X))',residual2:'Y = R + FFN(R)',embedding:'X[p] = E[token_id[p]]'}[id]||I18n.t("输入 → ")+d.name+I18n.t(" → 输出"),meaning:d.hint,purpose:I18n.t("本模块使用教学规模的数值计算。连线决定数据来源；运行轨迹显示实际算出的中间值。"),weight:M.weights[id]?M.weights[id].length+' × '+M.weights[id][0].length:I18n.t("无 / 已封装"),trait:extra[id]?.engine?I18n.t("调度任务"):I18n.t("数值计算"),stage:0};};
M.extraExample=(id,levelId)=>{
 if(!M.operations[id])return null;const s=M.solutions(levelId),ev=M.evaluate(s,M.samplesFor(levelId)[0].v),n=s.nodes.find(n=>n.id===id);if(!n)return null;
 if(id==='norm'){
  const values=ev.values[n.links[0]],width=values.length/3,meanSquare=values.slice(0,width).reduce((sum,x)=>sum+x*x,0)/width;
  return {inputs:[['X',values]],output:['Z',ev.values[id]],rms:{width,meanSquare,divisor:Math.sqrt(meanSquare+1e-6)}};
 }
 return {inputs:n.links.map((k,i)=>[I18n.t("输入 ")+(i+1),ev.values[k]||[]]),output:[I18n.t("结果"),ev.values[id]||[]]};
};
// Reference circuits are for automated verification only, never auto-filled by UI.
M.solutions=id=>{const s=M.initial(id,{complete:true}),links={workshop:{project:['R'],sink:'project'},
 wire:{sink:'R'},projection:{project:['R'],sink:'project'},activation:{project:['R'],silu:['project'],sink:'silu'},
 'down-projection':{project:['R'],down:['project'],sink:'down'},'plain-ffn':{project:['R'],silu:['project'],down:['silu'],sink:'down'},'gate-projection':{gate:['R'],sink:'gate'},pairwise:{project:['R'],gate:['R'],multiply:['gate','project'],sink:'multiply'},swiglu:{gate:['R'],project:['R'],silu:['gate'],multiply:['silu','project'],down:['multiply'],sink:'down'},
 gating:{gate:['R'],project:['R'],silu:['gate'],multiply:['silu','project'],sink:'multiply'},residual:{project:['R'],down:['project'],add:['R','down'],sink:'add'},ffn:{gate:['Z'],project:['Z'],silu:['gate'],multiply:['silu','project'],down:['multiply'],add:['R','down'],sink:'add'},
 embedding:{embedding:['R'],sink:'embedding'},normalize:{norm:['R'],sink:'norm'},qkv:{q:['R'],k:['R'],v:['R'],sink:'v'},scores:{q:['R'],k:['R'],score:['q','k'],sink:'score'},
 causal:{q:['R'],k:['R'],score:['q','k'],mask:['score'],softmax:['mask'],sink:'softmax'},rope:{q:['R'],k:['R'],ropeq:['q'],ropek:['k'],score:['ropeq','ropek'],sink:'score'},
 softmax:{q:['R'],k:['R'],score:['q','k'],softmax:['score'],sink:'softmax'},rotation:{q:['R'],ropeq:['q'],sink:'ropeq'},aggregation:{attentionweights:['R'],v:['R'],aggregate:['attentionweights','v'],sink:'aggregate'},'vocabulary-scores':{lmhead:['R'],sink:'lmhead'},
 attention:{q:['R'],k:['R'],v:['R'],ropeq:['q'],ropek:['k'],score:['ropeq','ropek'],mask:['score'],softmax:['mask'],aggregate:['softmax','v'],output:['aggregate'],sink:'output'},
 block:{norm:['R'],attention:['norm'],residual1:['R','attention'],ffnblock:['residual1'],residual2:['residual1','ffnblock'],sink:'residual2'},
 vocabulary:{lmhead:['R'],pick:['lmhead'],invert:['R'],sink:'pick'},autoregressive:{step:['R'],pick:['step'],feedback:['pick'],sink:'feedback'}};
 const l=links[id]||{...serial(engineNodes),sink:'enhead'};
 for(const n of s.nodes)n.links=l[n.id]||n.links;s.sink=l.sink;return s;
};
M.math={rows,mat,rotate,scores,causal,probabilities,mix,logits,decode,normalize,attention,block,emb,vocab};
})(window.NeuralPrologue);
