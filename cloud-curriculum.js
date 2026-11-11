/* Cloud deployment foundations. Small exact matrices, not GPU performance claims. */
(function(M){
const L=(zh,en)=>{globalThis.AtelierEnglish[zh]=en;return I18n.t(zh);};
const scope=L('单机两卡的数学缩尺：每个组件标明所属 GPU，芯片归属固定，台面位置可自由摆放。本批只验证分片计算与通信结果，不模拟显存容量、真实并发时间或真实 GLM / Kimi 权重。','A mathematical miniature of two GPUs in one server. Module labels fix GPU ownership while workbench positions remain free. These lessons verify sharded computation and communication results, not memory capacity, real concurrency or GLM / Kimi weights.');
const W=[[1,0,1,.5],[0,1,.5,-1]],D=[[.5,.1],[.2,.6],[.3,-.2],[-.1,.4]],relu=x=>Math.max(0,x),plus=(a,b)=>a.map((x,i)=>x+b[i]);
const singleScope=L('单机单卡的内存位置教学：输入从 CPU 内存开始，GPU 算子只能读取 GPU 显存中的张量，结果交回 CPU。权重已在显存，拷贝统计只计算输出依赖路径上的特征向量，每个数按 FP32 的 4 字节计；不代表真实硬件耗时。','One-server, one-GPU memory-location lesson. Inputs start in CPU memory, GPU operators require GPU tensors, and the result returns to the CPU. Weights are already resident. Copy counts cover feature vectors on the output dependency path at 4 bytes per FP32 value; they do not represent real hardware timings.');
const specs={
 'sg-upload2':[L('上传输入 · CPU → GPU','Upload input · CPU → GPU'),[2],2,'transfer',null,L('把两个输入数从 CPU 内存复制到 GPU 显存，数值不变。GPU 才能读取这份输入。两个 FP32 数共 8 字节；这里权重已经在显存中。','Copy two input values from CPU memory to GPU memory without changing them. The GPU can then read this input. Two FP32 values occupy 8 bytes; weights are already on the GPU.'),'Xcpu → Xgpu'],
 'sg-download4':[L('取回四维结果 · GPU → CPU','Read four features · GPU → CPU'),[4],4,'transfer',null,L('把四个 GPU 结果复制回 CPU，数值不变，共 16 字节。CPU 算子不能直接读取本关的 GPU 张量，需要先取回。','Copy four GPU values back to the CPU unchanged: 16 bytes. CPU operators cannot directly read GPU tensors in this lesson.'),'Hgpu → Hcpu'],
 'sg-upload4':[L('上传四维结果 · CPU → GPU','Upload four features · CPU → GPU'),[4],4,'transfer',null,L('把 CPU 处理过的四个特征送回 GPU，复制 16 字节。若能让后续算子直接在 GPU 处理，就能省去这次往返。','Send four CPU-processed features back to the GPU, copying 16 bytes. Running the operation on the GPU can avoid this round trip.'),'Hcpu → Hgpu'],
 'sg-download2':[L('取回输出 · GPU → CPU','Read output · GPU → CPU'),[2],2,'transfer',null,L('把最终两个特征复制回 CPU，复制 8 字节。GPU 上算完并不等于 CPU 已经拿到结果。','Copy the final two features to the CPU: 8 bytes. Finishing GPU computation does not mean the CPU already has its result.'),'Ygpu → Ycpu'],
 'sg-project':[L('GPU · 完整投影 W','GPU · complete projection W'),[2],4,'linear',W,L('一张 GPU 存整块 2 × 4 权重 W，一次算出四个特征。输入必须先上传到显存；例：[1, 1] 得到 [1, 1, 1.5, −0.5]。','One GPU holds the complete 2 × 4 weight matrix W and computes all four features. Upload the input first. Example: [1, 1] becomes [1, 1, 1.5, −0.5].'),'Hgpu = Xgpu W'],
 'sg-down':[L('GPU · 完整投影 D','GPU · complete projection D'),[4],2,'linear',D,L('一张 GPU 用完整的 4 × 2 权重 D，把四个中间特征组合成两个输出。它需要的是 GPU 张量，不是仍在 CPU 中的结果。','One GPU uses the complete 4 × 2 matrix D to combine four intermediate features into two outputs. It requires a GPU tensor, not a result still in CPU memory.'),'Ygpu = Agpu D'],
 'sg-cpu-relu':[L('CPU · ReLU','CPU · ReLU'),[4],4,'activation',null,L('在 CPU 上逐项计算 max(0, x)。第二关的运行环境只提供这个 CPU 版本，所以中间结果必须先从 GPU 取回。实际系统若有 GPU 内核，就可以避免这种回退。','Compute max(0, x) elementwise on the CPU. The second lesson provides only this CPU implementation, so intermediate results must be downloaded. A real system can avoid this fallback when a GPU kernel is available.'),'Acpu = ReLU(Hcpu)'],
 'sg-gpu-relu':[L('GPU · ReLU','GPU · ReLU'),[4],4,'activation',null,L('做与 CPU ReLU 完全相同的运算，但直接读写 GPU 显存。前后两个投影都在 GPU 时，中间四个特征可以一直留在显存，省去 16 + 16 字节的往返。','Compute exactly the same ReLU as the CPU version, but read and write GPU memory directly. Between GPU projections, all four features stay resident, avoiding a 16 + 16 byte round trip.'),'Agpu = ReLU(Hgpu)'],
 'tp-col-a':[L('GPU 0 · 前半列','GPU 0 · first columns'),[2],2,'linear',W.map(r=>r.slice(0,2)),L('两卡收到同一个输入 X。GPU 0 只存 W 的前两列，算出输出的前两个特征；不是把输入砍成一半。例：[1, 1] 得到 [1, 1]。','Both GPUs receive the same X. GPU 0 stores the first two columns of W and computes the first two output features; the input is not halved. Example: [1, 1] produces [1, 1].'),'Y₀ = XW₀'],
 'tp-col-b':[L('GPU 1 · 后半列','GPU 1 · last columns'),[2],2,'linear',W.map(r=>r.slice(2)),L('GPU 1 存 W 的后两列。输入与 GPU 0 相同，但权重不同，所以输出特征不同。例：[1, 1] 得到 [1.5, −0.5]。','GPU 1 stores the last two columns of W. Its input matches GPU 0, but its weights differ. Example: [1, 1] produces [1.5, −0.5].'),'Y₁ = XW₁'],
 'tp-gather':[L('拼接收集 · Gather','Concatenate · Gather'),[2,2],4,'concat',null,L('两个入口分别接前半、后半输出，把 [a, b] 和 [c, d] 拼成 [a, b, c, d]，不相加。本关在出口收集一份；All-Gather 则会让每张卡都得到完整结果。','The first input is the first half and the second is the last half. Join [a, b] and [c, d] into [a, b, c, d], without addition. This lesson collects one copy at the outlet; All-Gather would distribute a complete result to every GPU.'),'Y = concat(Y₀, Y₁)'],
 'tp-expand':[L('已知投影 · 生成 H','Known projection · make H'),[2],4,'linear',W,L('先得到四个特征 H，作为本关待拆分的输入。下一步两卡会各取其中两个特征，并乘对应的权重行。','Produce the four-feature H that this lesson will split. Each GPU then takes two features and multiplies their matching weight rows.'),'H = XW'],
 'tp-slice-a':[L('取前半特征','Take first features'),[4],2,'linear',[[1,0],[0,1],[0,0],[0,0]],L('取 H 的前两个特征。例如 [1, 1, 1.5, −0.5] 取出 [1, 1]；投影矩阵中的 0 / 1 只负责选取，不做新的特征混合。','Take the first two features of H. For [1, 1, 1.5, −0.5], return [1, 1]. The 0/1 matrix only selects coordinates; it does not learn a new feature mixture.'),'H₀ = H[:2]'],
 'tp-slice-b':[L('取后半特征','Take last features'),[4],2,'linear',[[0,0],[0,0],[1,0],[0,1]],L('取 H 的后两个特征，顺序保持不变。例如 [1, 1, 1.5, −0.5] 取出 [1.5, −0.5]。','Take the last two features of H in their original order. For [1, 1, 1.5, −0.5], return [1.5, −0.5].'),'H₁ = H[2:]'],
 'tp-row-a':[L('GPU 0 · 前半权重行','GPU 0 · first weight rows'),[2],2,'linear',D.slice(0,2),L('用前半特征乘 D 的前两行。得到的是每个输出特征的一部分贡献，不能直接当作完整输出。例：[1, 1] 得到 [0.7, 0.7]。','Multiply the first features by the first rows of D. The result is a partial contribution to every output feature, not a complete output. Example: [1, 1] produces [0.7, 0.7].'),'P₀ = H₀D₀'],
 'tp-row-b':[L('GPU 1 · 后半权重行','GPU 1 · last weight rows'),[2],2,'linear',D.slice(2),L('用后半特征乘 D 的后两行。它和另一张卡计算的是同一组输出坐标的不同贡献，需要逐项相加。例：[1.5, −0.5] 得到 [0.5, −0.5]。','Multiply the last features by the last rows of D. Both GPUs contribute to the same output coordinates, so their partial values must be added. Example: [1.5, −0.5] produces [0.5, −0.5].'),'P₁ = H₁D₁'],
 'tp-reduce':[L('部分和归并 · Reduce','Sum partials · Reduce'),[2,2],2,'add',null,L('把两张卡对同一输出坐标的贡献相加：[0.7, 0.7] + [0.5, −0.5] = [1.2, 0.2]。这里是矩阵乘法的部分和，不是残差；两个入口都不是绕过计算的原始输入。All-Reduce 还会把总和分发给所有卡。','Add both GPUs’ contributions to the same output coordinates: [0.7, 0.7] + [0.5, −0.5] = [1.2, 0.2]. These are partial matrix-product sums, not a residual: neither input is a bypass of the original input. All-Reduce also distributes the sum to every GPU.'),'Y = P₀ + P₁'],
 'tp-relu-a':[L('GPU 0 · 局部 ReLU','GPU 0 · local ReLU'),[2],2,'activation',null,L('ReLU 对每个数单独计算 max(0, x)，不需要知道另一张卡的特征。因此前半特征可留在 GPU 0 上处理，无需先收集完整向量。','ReLU computes max(0, x) independently for each number. It does not need the other GPU’s features, so the first half can stay on GPU 0 without gathering the full vector.'),'A₀ = ReLU(H₀)'],
 'tp-relu-b':[L('GPU 1 · 局部 ReLU','GPU 1 · local ReLU'),[2],2,'activation',null,L('后半特征在 GPU 1 本地经过 ReLU，再交给后半权重行。例：[1.5, −0.5] 变成 [1.5, 0]。这只适用于逐元素运算，不能据此把跨特征归一化也拆开独立计算。','Apply ReLU to the last features on GPU 1, then feed the last weight rows. Example: [1.5, −0.5] becomes [1.5, 0]. This works for elementwise operations; it does not make normalization across features independently separable.'),'A₁ = ReLU(H₁)']
};
const cards={};
for(const [id,[name,ins,out,type,w,hint,formula]]of Object.entries(specs)){
 M.definitions[id]={name,ins,out,type,hint,x:0,z:0,...(type==='activation'?{activation:'relu'}:{})};if(w)M.weights[id]=w;
 M.extraColors[id]=id.endsWith('-b')?0x9b88c4:id.includes('gather')||id.includes('reduce')?0xc7a166:0x659bab;
 cards[id]={number:'',seal:id.startsWith('sg-')?'GPU':'TP',english:id.startsWith('sg-')?'SINGLE GPU':'TENSOR PARALLEL',family:id.startsWith('sg-')?L('云端推理 · 单机单卡','Cloud inference · one GPU'):L('云端推理 · 两卡协作','Cloud inference · two GPUs'),ability:name,meaning:hint,purpose:id.startsWith('sg-')?singleScope:scope,formula,weight:w?w.length+' × '+w[0].length:L('无权重','No weights'),trait:id.startsWith('sg-')?L('内存位置','Memory location'):L('分片计算','Sharded computation'),stage:0,color:'#'+M.extraColors[id].toString(16)};
}
for(const id of Object.keys(specs).filter(id=>id.startsWith('sg-'))){
 const d=M.definitions[id],upload=id.startsWith('sg-upload'),download=id.startsWith('sg-download');
 d.memoryInput=upload||id==='sg-cpu-relu'?'cpu':'gpu';d.memoryOutput=download||id==='sg-cpu-relu'?'cpu':'gpu';
 if(d.type==='transfer'){d.transfer=true;d.glyphs=[upload?'CPU':'GPU',upload?'GPU':'CPU'];M.operations[id]=([x])=>[...x];}
 if(d.type==='activation')M.operations[id]=([x])=>x.map(relu);
}
M.definitions['tp-reduce'].communication=true;
M.operations['tp-gather']=([a,b])=>[...a,...b];M.operations['tp-reduce']=([a,b])=>plus(a,b);
for(const id of ['tp-relu-a','tp-relu-b'])M.operations[id]=([x])=>x.map(relu);
const definitions=[
 ['single-upload',L('先让一张 GPU 算起来','Start with one GPU'),['sg-upload2','sg-project','sg-download4'],4,
 L('把 CPU 的输入交给 GPU，再拿回结果','Send CPU input to the GPU, then read the result'),
 L('用一张 GPU 的完整权重 W 做投影。输入起初在 CPU 内存，出口也在 CPU；让五组输入都得到正确结果。','Project with the complete W on one GPU. Both the input and the output live on the CPU. Produce the correct result for all five inputs.'),
 L('CPU 负责准备输入和发起工作，GPU 在自己的显存中读取数据、做矩阵计算。连上数学算子还不够，输入必须先到达它能读取的内存。','The CPU prepares input and launches work; the GPU reads data and computes in its own memory. Wiring the mathematical operation is not enough: its input must reach memory it can read.'),
 L('上传和取回不改变数值，只改变数据在哪里。第一次先学会这个往返，随后再看哪些中间结果不必取回。','Uploading and downloading preserve values but change where data resides. First learn this round trip, then decide which intermediate results can stay on the GPU.'),
 {'sg-upload2':['R'],'sg-project':['sg-upload2'],'sg-download4':['sg-project'],sink:'sg-download4'},x=>M.linear(x,W)],
 ['single-fallback',L('算到一半，要回 CPU？','A CPU detour in the middle?'),['sg-upload2','sg-project','sg-download4','sg-cpu-relu','sg-upload4','sg-down','sg-download2'],2,
 L('遇到 CPU 算子，接通一次往返','Connect a round trip for a CPU operator'),
 L('本关只提供 CPU 版 ReLU，两次投影在 GPU 上完成。搭出 W → ReLU → D；不要让算子读取另一侧内存中的张量。','This lesson provides only a CPU ReLU; both projections run on the GPU. Build W → ReLU → D without asking an operator to read a tensor on the other device.'),
 L('部署时某个算子若缺少可用的 GPU 实现，执行可能回退到 CPU。虽然公式没变，GPU 与 CPU 之间却多了中间结果的传输。','An operator without an available GPU implementation may fall back to the CPU. The formula stays the same, but intermediate tensors must cross between GPU and CPU memory.'),
 L('这里的 CPU 回退是教学限制，不是 ReLU 天生只能在 CPU 算。观察最终报告：中间四个特征取回又上传，额外搬运 32 字节。','CPU fallback is a constraint of this lesson, not an inherent limitation of ReLU. The final report shows four intermediate values downloaded and uploaded: 32 extra bytes.'),
 {'sg-upload2':['R'],'sg-project':['sg-upload2'],'sg-download4':['sg-project'],'sg-cpu-relu':['sg-download4'],'sg-upload4':['sg-cpu-relu'],'sg-down':['sg-upload4'],'sg-download2':['sg-down'],sink:'sg-download2'},x=>M.linear(M.linear(x,W).map(relu),D)],
 ['single-resident',L('让中间结果留在显存','Keep intermediate results on the GPU'),['sg-upload2','sg-project','sg-gpu-relu','sg-down','sg-download2'],2,
 L('结果相同，搬运更少','Same result, fewer copies'),
 L('现在多了 GPU 版 ReLU。输出要与上一关相同，但每组输入的特征搬运量最多 16 字节。CPU 版和拷贝组件仍可选择。','A GPU ReLU is now available. Match the previous output while copying at most 16 bytes of features per input. CPU and copy components remain available as alternatives.'),
 L('W、ReLU、D 都能直接使用显存中的结果，就不需要为了中间一步返回 CPU。少搬数据不会改变模型公式，也不是把三个算子合成一次计算。','When W, ReLU and D can use GPU-resident results, the middle step no longer needs a CPU detour. Avoiding copies preserves the formula and does not fuse three operators into one.'),
 L('单卡可以完成整段网络，但完整权重、中间结果和缓存都要占显存。下一阶段再学习拆分权重的方法；多卡不是单卡的起点。','One GPU can execute the whole segment, but complete weights, intermediates and caches consume its memory. The next stage teaches how to split weights; multiple GPUs are not the starting point.'),
 {'sg-upload2':['R'],'sg-project':['sg-upload2'],'sg-gpu-relu':['sg-project'],'sg-down':['sg-gpu-relu'],'sg-download2':['sg-down'],sink:'sg-download2'},x=>M.linear(M.linear(x,W).map(relu),D)],
 ['tp-columns',L('一块矩阵，两张卡','One matrix, two GPUs'),['tp-col-a','tp-col-b','tp-gather'],4,
 L('把输出特征分给两张卡','Split output features between GPUs'),
 L('两卡分别拿到一半权重列。让完整输出与未拆分的 XW 一致，五组输入都要匹配。','Each GPU holds half the weight columns. Make the complete output match unsharded XW for all five inputs.'),
 L('前面一个投影组件保存整块 W；现在 W 按输出特征拆成两份。两卡各算不同的输出坐标，可以分担权重存储和计算。','Previously one projection held all of W. Now split W by output features. Each GPU computes different output coordinates, sharing weight storage and arithmetic.'),
 L('完整输入要给两张卡，输出才分成两半。拼接顺序决定特征顺序；这里不是把两份结果相加。','Both GPUs need the full input; their outputs are different halves. Concatenation order determines feature order. Do not add those halves.'),
 {'tp-col-a':['R'],'tp-col-b':['R'],'tp-gather':['tp-col-a','tp-col-b'],sink:'tp-gather'},x=>M.linear(x,W)],
 ['tp-rows',L('两份部分和，一个结果','Two partial sums, one result'),['tp-expand','tp-slice-a','tp-slice-b','tp-row-a','tp-row-b','tp-reduce'],2,
 L('输入分开后，如何还原一次投影？','How do split inputs recover one projection?'),
 L('先得到四维 H，再拆给两张卡，各乘对应的两行 D。让输出与未拆分的 HD 一致。','Produce four-feature H, split it across the GPUs, and multiply each half by its matching rows of D. Match the unsharded HD result.'),
 L('每个输出都是四个输入特征的加权和。每张卡只拿到其中两个特征，因此只能算出同一输出的一部分贡献。','Each output is a weighted sum of four input features. A GPU holding only two features can compute only part of that same output.'),
 L('按列拆：输出坐标不同，要拼接。按行拆：输出坐标相同、贡献不完整，要相加。Reduce 与残差都做加法，但数据来源不同。','Column split: different output coordinates must be concatenated. Row split: incomplete contributions to the same coordinates must be summed. Reduce and residual addition both add, but their inputs mean different things.'),
 {'tp-expand':['R'],'tp-slice-a':['tp-expand'],'tp-slice-b':['tp-expand'],'tp-row-a':['tp-slice-a'],'tp-row-b':['tp-slice-b'],'tp-reduce':['tp-row-a','tp-row-b'],sink:'tp-reduce'},x=>M.linear(M.linear(x,W),D)],
 ['tp-ffn',L('让特征留在各自的卡上','Keep features on their GPUs'),['tp-col-a','tp-col-b','tp-relu-a','tp-relu-b','tp-row-a','tp-row-b','tp-reduce'],2,
 L('两张卡完成同一个 FFN','Complete one FFN across two GPUs'),
 L('把前两关的分片组合起来，得到与 XW → ReLU → D 相同的结果。两卡各处理自己的特征，最后收集输出。','Combine both sharding methods to match XW → ReLU → D. Each GPU processes its own features; collect the output at the end.'),
 L('第一层按列拆，恰好把中间特征分在两张卡。ReLU 逐项运算，可留在原卡；第二层按行拆，直接接各卡已有的特征，再归并部分和。','A column-parallel first layer leaves the intermediate features split across GPUs. ReLU is elementwise and stays local. A row-parallel second layer consumes those local features, then reduces partial sums.'),
 L('不必每经过一个组件就拼回完整向量。这是 FFN 张量并行的一种常见组合。真实多卡执行还要考虑通信、同步和显存，后续分开学习。','There is no need to reconstruct the full vector after every module. This is a common tensor-parallel FFN pattern. Real execution also requires communication, synchronization and memory planning, introduced separately later.'),
 {'tp-col-a':['R'],'tp-col-b':['R'],'tp-relu-a':['tp-col-a'],'tp-relu-b':['tp-col-b'],'tp-row-a':['tp-relu-a'],'tp-row-b':['tp-relu-b'],'tp-reduce':['tp-row-a','tp-row-b'],sink:'tp-reduce'},x=>M.linear(M.linear(x,W).map(relu),D)]
];
const links={},refs={},known=new Set();
for(const [i,[id,title,nodes,sinkDim,mission,description,why,takeaway,solution,ref]]of definitions.entries()){
 const single=id.startsWith('single-');if(id==='tp-columns')known.clear();
 nodes.forEach(k=>known.add(k));const inventory=Object.fromEntries([...known].map(k=>[k,2]));
 M.levels[id]={id,title,nodes,sources:['R'],sourceDim:2,sinkDim,sampleCount:5,sequence:false,sourceX:-10.3,sinkX:10.3,zoom:1.22,boardWidth:26,boardDepth:14,sourceTitle:single?L('CPU 输入 X','CPU input X'):L('输入特征 X','Input features X'),sourceNote:'',outputTitle:single?L('CPU 中的结果','Result on the CPU'):L('完整计算结果','Complete result'),shape:'2 → '+sinkDim,mission,description,summary:description,concept:title,hints:[],rationale:{title:single?L('先理解一张 GPU 怎么工作','First understand one GPU'):L('从矩阵运算到两卡协作','From matrix operations to two GPUs'),text:why,takeaway,scope:single?singleScope:scope},track:'cloud',cloud:true,cloudSingle:single,...(id==='single-resident'?{copyLimit:16}:{}),chapter:L('云端推理 / 0'+(i+1),'Cloud / 0'+(i+1)),boardText:'CLOUD / 0'+(i+1)+(single?' · 1 GPU':' · 2 GPUs'),buildable:true,requirePlacement:true,startWithModules:false,maxModules:nodes.length+2,inventory,next:definitions[i+1]?.[0]||null,
 shelf:{packageKinds:single?['sg-upload2','sg-upload4','sg-download2','sg-download4']:['tp-gather','tp-reduce'],currentKinds:[...known],order:[...known],current:L('两卡计算组件','Two-GPU modules'),packages:L('通信与收集','Communication and collection'),basics:single?L('CPU / GPU 算子','CPU / GPU operators'):L('分片与局部计算','Shards and local computation'),earlier:L('已学零件','Earlier parts'),note:L('已学组件保留为可选方案，不需要全部使用。GPU 编号表示所属芯片，不由摆放位置决定。','Learned modules remain available as alternatives; not every module is needed. GPU labels identify chip ownership, independently of placement.')}};
 links[id]=solution;refs[id]=ref;
}
M.routes={main:M.routes.main,modern:M.routes.modern,cloud:{name:L('实践 · 云端推理','Practice · Cloud inference'),description:L('学习单卡数据搬运、CPU 回退与显存驻留，再将矩阵和 FFN 拆到两张 GPU。','Learn one-GPU transfers, CPU fallback and memory residency, then split matrices and an FFN across two GPUs.'),ids:definitions.map(d=>d[0])},engine:M.routes.engine};
M.routes.cloud.groups=[
 {id:'single',label:L('单机单卡','One GPU'),title:L('先让一张 GPU 工作','Start with one GPU'),description:L('认识 CPU 与显存、算子回退和中间结果驻留。','Learn CPU/GPU memory, operator fallback and resident intermediate results.'),ids:['single-upload','single-fallback','single-resident']},
 {id:'tensor-parallel',label:L('单机两卡','Two GPUs'),title:L('再把计算拆到两张卡','Then split work across two GPUs'),description:L('沿用同样的矩阵和 FFN，学习按列、按行分片及结果收集。','Keep the same matrices and FFN while learning column and row sharding and result collection.'),ids:['tp-columns','tp-rows','tp-ffn']}
];I18n.prepare(M.routes.cloud.groups);
const priorEvaluate=M.evaluate;
M.evaluate=(state,input)=>{
 const ev=priorEvaluate(state,input),l=M.levels[state.levelId];if(!l?.cloudSingle||ev.error)return ev;
 const nodes=new Map(state.nodes.map(n=>[n.id,n])),used=new Set(),visit=id=>{if(id==='R'||used.has(id))return;used.add(id);nodes.get(id)?.links.forEach(visit);};visit(state.sink);
 const locations={R:'cpu'},copies=[];let failure=null;
 for(const id of ev.order.filter(id=>used.has(id))){
  const n=nodes.get(id),d=M.definitions[M.nodeKind(n)];
  if(!d.memoryInput||n.links.some(from=>locations[from]!==d.memoryInput)){failure={id,message:L('内存位置不匹配：','Memory location mismatch: ')+d.name+L(' 需要读取 ',' requires ')+(d.memoryInput||'GPU').toUpperCase()+L(' 中的张量，请检查上传或取回连接。',' tensors. Check the upload or readback connections.')};break;}
  locations[id]=d.memoryOutput;if(d.transfer)copies.push({id,from:d.memoryInput,to:d.memoryOutput,bytes:ev.values[id].length*4});
 }
 const bytes=copies.reduce((sum,c)=>sum+c.bytes,0);ev.cloudMemory={bytes,copies,locations,limit:l.copyLimit??null};
 if(!failure&&locations[state.sink]!=='cpu')failure={id:'sink',message:L('结果还在 GPU 显存：出口在 CPU，请先取回输出。','The result is still in GPU memory. Read it back before sending it to the CPU outlet.')};
 if(!failure&&l.copyLimit!=null&&bytes>l.copyLimit)failure={id:'sink',message:L('数值已算出，但特征搬运超过本关上限：','Values were computed, but feature copies exceed this lesson’s limit: ')+bytes+' / '+l.copyLimit+' B'};
 if(failure){ev.error=failure.message;ev.bad={id:failure.id};}return ev;
};
M.cloudCopyLabels={title:L('本组输入的特征搬运','Feature copies for this input'),limit:L('上限','Limit'),note:L('仅统计输出依赖路径；权重已驻留，FP32 每个数 4 字节，不是硬件耗时。','Output dependency path only. Weights are resident; each FP32 value is 4 bytes. This is not a hardware timing.')};I18n.prepare(M.cloudCopyLabels);
const oldExampleLevel=M.exampleLevel;
M.exampleLevel=(kind,id)=>cards[kind]?Object.keys(links).find(l=>links[l][kind]):oldExampleLevel(kind,id);
const oldSamples=M.samplesFor,oldRef=M.reference,oldSolutions=M.solutions,oldCard=M.extraCard,oldExample=M.extraExample;
M.samplesFor=id=>refs[id]?M.samples:oldSamples(id);
M.reference=(v,id)=>refs[id]?refs[id](v):oldRef(v,id);
M.solutions=id=>{if(!links[id])return oldSolutions(id);const s=M.initial(id,{complete:true});for(const n of s.nodes)n.links=[...links[id][n.id]];s.sink=links[id].sink;return s;};
M.extraCard=(kind,id)=>cards[kind]||oldCard?.(kind,id);
M.extraExample=(kind,id,nodeId)=>{if(!cards[kind])return oldExample?.(kind,id,nodeId);const stage=Object.keys(links).find(l=>links[l][kind]),s=M.solutions(stage),input=M.samples[2].v,ev=M.evaluate(s,input),n=s.nodes.find(n=>n.id===kind);return {sampleIndex:2,exampleSource:['X',input],inputs:n.links.map(k=>[k==='R'?'X':M.definitions[k].name,ev.values[k]]),output:[M.definitions[kind].name,ev.values[kind]]};};
M.cloud={weights:W,down:D,ids:definitions.map(d=>d[0])};
I18n.prepare({cards,levels:Object.fromEntries(M.cloud.ids.map(id=>[id,M.levels[id]])),route:M.routes.cloud,definitions:Object.fromEntries(Object.keys(specs).map(id=>[id,M.definitions[id]]))});
})(window.NeuralPrologue);
