/* Cloud deployment foundations. Small exact matrices, not GPU performance claims. */
(function(M){
const L=(zh,en)=>{globalThis.AtelierEnglish[zh]=en;return I18n.t(zh);};
const scope=L('本组关卡用小矩阵演示单机两卡计算。模块的 GPU 归属固定，不随台面位置改变。这里只验证分片计算与通信结果，不模拟显存容量、真实并发时间或真实 GLM / Kimi 权重。','These lessons use small matrices to demonstrate two GPUs in one server. Each module belongs to a fixed GPU, regardless of its workbench position. Only sharded computation and communication results are checked. GPU memory capacity, real concurrency timings and real GLM / Kimi weights are not simulated.');
const W=[[1,0,1,.5],[0,1,.5,-1]],D=[[.5,.1],[.2,.6],[.3,-.2],[-.1,.4]],relu=x=>Math.max(0,x),plus=(a,b)=>a.map((x,i)=>x+b[i]);
const singleScope=L('本组关卡演示单机单卡的内存位置。输入从 CPU 内存开始，GPU 算子只能读取显存中的张量，出口在 CPU。权重已驻留显存。搬运量仅统计出口所需计算路径上的特征向量，每个 FP32 数按 4 字节计；不代表真实硬件耗时。','These lessons demonstrate memory locations with one GPU in one server. Input starts in CPU memory. GPU operators can only read tensors in GPU memory, and the outlet is on the CPU. Weights are already resident on the GPU. Copy counts include only feature vectors on the computation path required by the outlet, at 4 bytes per FP32 value. They do not represent real hardware timings.');
const specs={
 'sg-upload2':[L('上传输入 · CPU → GPU','Upload input · CPU → GPU'),[2],2,'transfer',null,L('把两个输入数从 CPU 内存复制到 GPU 显存，数值不变。GPU 才能读取这份输入。两个 FP32 数共 8 字节；这里权重已经在显存中。','Copy two input values from CPU memory to GPU memory without changing them. The GPU can then read this input. Two FP32 values occupy 8 bytes; weights are already on the GPU.'),'Xcpu → Xgpu'],
 'sg-download4':[L('取回四维结果 · GPU → CPU','Download four features · GPU → CPU'),[4],4,'transfer',null,L('把四个 GPU 结果复制回 CPU，数值不变，共 16 字节。CPU 算子不能直接读取本关的 GPU 张量，需要先取回。','Copy four GPU values back to the CPU unchanged: 16 bytes. CPU operators cannot directly read GPU tensors in this lesson. Download them first.'),'Hgpu → Hcpu'],
 'sg-upload4':[L('上传四维结果 · CPU → GPU','Upload four features · CPU → GPU'),[4],4,'transfer',null,L('把 CPU 处理后的四个特征复制回 GPU，共 16 字节。若改用 GPU 算子处理中间特征，就能省去取回和上传这次往返。','Copy four CPU-processed features back to the GPU: 16 bytes. Processing these intermediate features with a GPU operator avoids the download and upload round trip.'),'Hcpu → Hgpu'],
 'sg-download2':[L('取回输出 · GPU → CPU','Download output · GPU → CPU'),[2],2,'transfer',null,L('把最终两个特征复制回 CPU，复制 8 字节。GPU 上算完并不等于 CPU 已经拿到结果。','Copy the final two features to the CPU: 8 bytes. Finishing GPU computation does not mean the CPU already has its result.'),'Ygpu → Ycpu'],
 'sg-project':[L('GPU · 完整投影 W','GPU · complete projection W'),[2],4,'linear',W,L('一张 GPU 存整块 2 × 4 权重 W，一次算出四个特征。输入必须先上传到显存；例：[1, 1] 得到 [1, 1, 1.5, −0.5]。','One GPU holds the complete 2 × 4 weight matrix W and computes all four features. Upload the input first. Example: [1, 1] becomes [1, 1, 1.5, −0.5].'),'Hgpu = Xgpu W'],
 'sg-down':[L('GPU · 完整投影 D','GPU · complete projection D'),[4],2,'linear',D,L('一张 GPU 用完整的 4 × 2 权重 D，把四个中间特征组合成两个输出。它需要的是 GPU 张量，不是仍在 CPU 中的结果。','One GPU uses the complete 4 × 2 matrix D to combine four intermediate features into two outputs. It requires a GPU tensor, not a result still in CPU memory.'),'Ygpu = Agpu D'],
 'sg-cpu-relu':[L('CPU · ReLU','CPU · ReLU'),[4],4,'activation',null,L('在 CPU 上逐项计算 max(0, x)。第二关的运行环境只提供这个 CPU 版本，所以中间结果必须先从 GPU 取回。实际系统若有 GPU 内核，就可以避免这种回退。','Compute max(0, x) elementwise on the CPU. The second lesson provides only this CPU implementation, so intermediate results must be downloaded. A real system can avoid this fallback when a GPU kernel is available.'),'Acpu = ReLU(Hcpu)'],
 'sg-gpu-relu':[L('GPU · ReLU','GPU · ReLU'),[4],4,'activation',null,L('做与 CPU ReLU 完全相同的运算，但直接读写 GPU 显存。前后两个投影都在 GPU 时，中间四个特征可以一直留在显存，省去 16 + 16 字节的往返。','Compute exactly the same ReLU as the CPU version, but read and write GPU memory directly. Between GPU projections, all four features stay resident, avoiding a 16 + 16 byte round trip.'),'Agpu = ReLU(Hgpu)'],
 'tp-col-a':[L('GPU 0 · 前两列权重','GPU 0 · first two weight columns'),[2],2,'linear',W.map(r=>r.slice(0,2)),L('两卡接收同一份完整输入 X。GPU 0 只保存 W 的前两列，计算前两个输出特征。分片的是权重列，不是输入。例：[1, 1] 得到 [1, 1]。','Both GPUs receive the same complete input X. GPU 0 stores only the first two columns of W and computes the first two output features. The weight columns are split, not the input. Example: [1, 1] produces [1, 1].'),'Y₀ = XW₀'],
 'tp-col-b':[L('GPU 1 · 后两列权重','GPU 1 · last two weight columns'),[2],2,'linear',W.map(r=>r.slice(2)),L('GPU 1 存 W 的后两列。输入与 GPU 0 相同，但权重不同，所以输出特征不同。例：[1, 1] 得到 [1.5, −0.5]。','GPU 1 stores the last two columns of W. Its input matches GPU 0, but its weights differ. Example: [1, 1] produces [1.5, −0.5].'),'Y₁ = XW₁'],
 'tp-gather':[L('拼接收集 · Gather','Concatenate · Gather'),[2,2],4,'concat',null,L('第一个入口接前半输出，第二个入口接后半输出。将 [a, b] 和 [c, d] 拼成 [a, b, c, d]，不相加。本关在出口收集一份完整结果；All-Gather 则会让每张卡都得到完整结果。','Connect the first half to the first input port and the last half to the second. Join [a, b] and [c, d] into [a, b, c, d], without addition. This lesson collects one complete result at the outlet. All-Gather would give every GPU the complete result.'),'Y = concat(Y₀, Y₁)'],
 'tp-expand':[L('已知投影 · 生成 H','Known projection · compute H'),[2],4,'linear',W,L('先得到四个特征 H，作为本关待拆分的输入。下一步两卡会各取其中两个特征，并乘对应的权重行。','Produce the four-feature H that this lesson will split. Each GPU then takes two features and multiplies their matching weight rows.'),'H = XW'],
 'tp-slice-a':[L('取前两个特征','Take the first two features'),[4],2,'linear',[[1,0],[0,1],[0,0],[0,0]],L('取 H 的前两个特征。例如从 [1, 1, 1.5, −0.5] 取出 [1, 1]。矩阵中的 0 / 1 只选择坐标，不混合特征。','Take the first two features of H. For [1, 1, 1.5, −0.5], return [1, 1]. The 0 / 1 matrix only selects coordinates; it does not mix features.'),'H₀ = H[:2]'],
 'tp-slice-b':[L('取后两个特征','Take the last two features'),[4],2,'linear',[[0,0],[0,0],[1,0],[0,1]],L('取 H 的后两个特征，顺序保持不变。例如 [1, 1, 1.5, −0.5] 取出 [1.5, −0.5]。','Take the last two features of H in their original order. For [1, 1, 1.5, −0.5], return [1.5, −0.5].'),'H₁ = H[2:]'],
 'tp-row-a':[L('GPU 0 · 前两行权重','GPU 0 · first two weight rows'),[2],2,'linear',D.slice(0,2),L('用前半特征乘 D 的前两行。得到的是每个输出特征的一部分贡献，不能直接当作完整输出。例：[1, 1] 得到 [0.7, 0.7]。','Multiply the first features by the first rows of D. The result is a partial contribution to every output feature, not a complete output. Example: [1, 1] produces [0.7, 0.7].'),'P₀ = H₀D₀'],
 'tp-row-b':[L('GPU 1 · 后两行权重','GPU 1 · last two weight rows'),[2],2,'linear',D.slice(2),L('用后半特征乘 D 的后两行。它和另一张卡计算的是同一组输出坐标的不同贡献，需要逐项相加。例：[1.5, −0.5] 得到 [0.5, −0.5]。','Multiply the last features by the last rows of D. Both GPUs contribute to the same output coordinates, so their partial values must be added. Example: [1.5, −0.5] produces [0.5, −0.5].'),'P₁ = H₁D₁'],
 'tp-reduce':[L('部分和归并 · Reduce','Sum partials · Reduce'),[2,2],2,'add',null,L('把两张卡对同一输出坐标的贡献相加：[0.7, 0.7] + [0.5, −0.5] = [1.2, 0.2]。这里是矩阵乘法的部分和，不是残差；两个入口都不是绕过计算的原始输入。All-Reduce 还会把总和分发给所有卡。','Add both GPUs’ contributions to the same output coordinates: [0.7, 0.7] + [0.5, −0.5] = [1.2, 0.2]. These are partial matrix-product sums, not a residual: neither input is a bypass of the original input. All-Reduce also distributes the sum to every GPU.'),'Y = P₀ + P₁'],
 'tp-relu-a':[L('GPU 0 · 局部 ReLU','GPU 0 · local ReLU'),[2],2,'activation',null,L('ReLU 对每个数单独计算 max(0, x)，不需要知道另一张卡的特征。因此前半特征可留在 GPU 0 上处理，无需先收集完整向量。','ReLU computes max(0, x) independently for each number. It does not need the other GPU’s features, so the first half can stay on GPU 0 without gathering the full vector.'),'A₀ = ReLU(H₀)'],
 'tp-relu-b':[L('GPU 1 · 局部 ReLU','GPU 1 · local ReLU'),[2],2,'activation',null,L('在 GPU 1 上对后半特征应用 ReLU，再将结果交给后两行权重的投影模块。例：[1.5, −0.5] 变成 [1.5, 0]。这只适用于逐元素运算；跨特征归一化不能直接按同样方式独立计算。','Apply ReLU to the last features on GPU 1, then send the result to the projection module with the last two weight rows. Example: [1.5, −0.5] becomes [1.5, 0]. This works for elementwise operations. Normalization across features cannot be computed independently in the same way.'),'A₁ = ReLU(H₁)']
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
 L('把输入送去计算，再带回结果','Send the input to the GPU and bring the result back'),
 L('输入和出口都在 CPU，投影模块却在 GPU 上。接好它们之间的路，让五组输入都得到目标结果。','The input and outlet are on the CPU, but the projection module is on the GPU. Connect a path between them and match the target for all five inputs.'),
 L('GPU 要先拿到数据，才能用完整权重 W 计算 XW。CPU 使用的内存与 GPU 使用的显存是两个位置；把输入复制过去叫“上传”，把结果复制回来叫“取回”。这就是本关练习的数据搬运。','The GPU needs the data before it can compute XW with the complete weights W. CPU memory and GPU memory are separate locations. Copying input to the GPU is an upload; copying the result back is a download. These are the data transfers you connect in this lesson.'),
 L('上传和取回都保留数值，只改变数据所在的内存。先完成这次往返，后面再判断哪些中间结果可以留在 GPU。','Upload and download operations preserve values and change only their memory location. Complete this round trip first. Later, decide which intermediate results can stay on the GPU.'),
 {'sg-upload2':['R'],'sg-project':['sg-upload2'],'sg-download4':['sg-project'],sink:'sg-download4'},x=>M.linear(x,W)],
 ['single-fallback',L('算到一半，要回 CPU？','A CPU detour in the middle?'),['sg-upload2','sg-project','sg-download4','sg-cpu-relu','sg-upload4','sg-down','sg-download2'],2,
 L('让两次 GPU 计算接上中间的 CPU 工作','Connect two GPU calculations through a CPU step'),
 L('两次投影在 GPU 上，中间的 ReLU 却只有 CPU 版本。怎样把它们接起来？让五组输入都匹配目标。每个模块只能读取自己所在设备的数据。','Both projections run on the GPU, but the ReLU between them has only a CPU version. How can you connect them? Match the targets for all five inputs. Each module can only read data on its own device.'),
 L('这次绕回 CPU 的路叫“CPU 回退”：某一步没有可用的 GPU 实现，就交给 CPU。运算仍是 W → ReLU → D，但中间数据需要在 GPU 显存与 CPU 内存之间搬运。','This detour is called CPU fallback: when a step has no available GPU implementation, the CPU takes over. The calculation is still W → ReLU → D, but intermediate data must move between GPU and CPU memory.'),
 L('CPU 回退是本关的教学限制，并非 ReLU 只能在 CPU 上计算。查看运行报告：四个中间特征先取回、再上传，额外搬运 32 字节。','CPU fallback is a teaching constraint, not a requirement of ReLU. Check the run report: downloading and uploading the four intermediate features copies 32 extra bytes.'),
 {'sg-upload2':['R'],'sg-project':['sg-upload2'],'sg-download4':['sg-project'],'sg-cpu-relu':['sg-download4'],'sg-upload4':['sg-cpu-relu'],'sg-down':['sg-upload4'],'sg-download2':['sg-down'],sink:'sg-download2'},x=>M.linear(M.linear(x,W).map(relu),D)],
 ['single-resident',L('让中间结果留在显存','Keep intermediate results on the GPU'),['sg-upload2','sg-project','sg-gpu-relu','sg-down','sg-download2'],2,
 L('少搬数据，算出一样的结果','Move less data, get the same result'),
 L('现在多了 GPU 版 ReLU。试着减少来回搬运，让五组输入仍得到上一关的结果，每组最多搬运 16 字节。CPU 版 ReLU 和搬运模块也还在，不必全部使用。','A GPU version of ReLU is now available. Try reducing the round trips while keeping the previous lesson’s results for all five inputs. Copy at most 16 bytes per input. CPU ReLU and transfer modules are still available; you do not need to use them all.'),
 L('把中间结果留在 GPU 显存，供后续模块直接使用，叫“显存驻留”。W、ReLU、D 都能读取这些结果时，就不必绕回 CPU。省去搬运不改变模型公式，也不等于把三个算子融合成一次计算。','Keeping intermediate results in GPU memory for later modules to use is called GPU memory residency. When W, ReLU and D can all read these results, no CPU detour is needed. Avoiding copies preserves the formula; it does not fuse the three operators into one computation.'),
 L('单卡可以执行整段网络，但完整权重、中间结果和缓存都会占用显存。下一组关卡再学习拆分权重，把计算分给两张 GPU。','One GPU can execute the whole network segment, but complete weights, intermediate results and caches all use GPU memory. The next lessons split weights and distribute computation across two GPUs.'),
 {'sg-upload2':['R'],'sg-project':['sg-upload2'],'sg-gpu-relu':['sg-project'],'sg-down':['sg-gpu-relu'],'sg-download2':['sg-down'],sink:'sg-download2'},x=>M.linear(M.linear(x,W).map(relu),D)],
 ['tp-columns',L('一块矩阵，两张卡','One matrix, two GPUs'),['tp-col-a','tp-col-b','tp-gather'],4,
 L('各算一半特征，拼出完整结果','Compute half the features on each GPU, then join them'),
 L('每张 GPU 保存一半权重列，两张卡接收同一份完整输入。把两卡算出的特征拼在一起，让五组输入都得到与整块权重计算相同的结果。','Each GPU holds half the weight columns and receives the same complete input. Join the features they compute and match the full-matrix result for all five inputs.'),
 L('把 W 沿着列拆开，分给两张 GPU，叫“列并行”。每张卡计算不同的输出特征，合起来仍是 XW。它们分担的是权重存储和矩阵计算，输入没有被拆开。','Splitting W by columns across two GPUs is called column parallelism. Each GPU computes different output features; together they form XW. The GPUs share weight storage and matrix computation, but the input is not split.'),
 L('按顺序把两半特征接起来，叫“拼接”，这里由 Gather 模块完成。顺序决定最终特征的位置；不是把两份结果相加。','Joining the two halves in order is called concatenation, performed here by the Gather module. The order determines each feature’s position. The two halves are not added together.'),
 {'tp-col-a':['R'],'tp-col-b':['R'],'tp-gather':['tp-col-a','tp-col-b'],sink:'tp-gather'},x=>M.linear(x,W)],
 ['tp-rows',L('两份贡献，一个结果','Two contributions, one result'),['tp-expand','tp-slice-a','tp-slice-b','tp-row-a','tp-row-b','tp-reduce'],2,
 L('各算一部分，加出完整结果','Compute a contribution on each GPU, then add them'),
 L('这次，两张 GPU 各拿到四个特征中的两个，以及对应的权重行。把两卡算出的贡献加起来，让五组输入都得到与整块权重计算相同的结果。','This time, each GPU gets two of the four features and the matching weight rows. Add their contributions to match the full-matrix result for all five inputs.'),
 L('把 D 沿着行拆开，让每张 GPU 处理对应的输入特征，叫“行并行”。每个输出本来是四项乘积之和，每张卡只算其中两项。这些不完整的贡献叫“部分和”，相加后才得到完整的 HD。','Splitting D by rows so each GPU handles the matching input features is called row parallelism. Each output sums four products, but each GPU computes only two of them. These incomplete contributions are partial sums; adding them gives the complete HD.'),
 L('按列拆：输出坐标不同，要拼接。按行拆：输出坐标相同、贡献不完整，要相加。Reduce 与残差都做加法，但数据来源不同。','Column split: different output coordinates must be concatenated. Row split: incomplete contributions to the same coordinates must be summed. Reduce and residual addition both add, but their inputs mean different things.'),
 {'tp-expand':['R'],'tp-slice-a':['tp-expand'],'tp-slice-b':['tp-expand'],'tp-row-a':['tp-slice-a'],'tp-row-b':['tp-slice-b'],'tp-reduce':['tp-row-a','tp-row-b'],sink:'tp-reduce'},x=>M.linear(M.linear(x,W),D)],
 ['tp-ffn',L('让特征留在各自的卡上','Keep features on their GPUs'),['tp-col-a','tp-col-b','tp-relu-a','tp-relu-b','tp-row-a','tp-row-b','tp-reduce'],2,
 L('两张卡合作，完成同一段网络','Let two GPUs work together on one network segment'),
 L('用前两关学过的方法，让两张 GPU 各自继续处理手里的特征。最后合成结果，五组输入都要与 XW → ReLU → D 的计算一致。','Use the methods from the last two lessons so each GPU keeps working on its own features. Combine the results at the end and match XW → ReLU → D for all five inputs.'),
 L('这段协作叫“FFN 张量并行”：第一层按列拆开，ReLU 在各卡上逐项计算，第二层按行拆开，再归并部分和。你把同一段网络的矩阵计算分给了两张卡，而不是让两张卡各算一遍完整网络。','This collaboration is tensor parallelism for an FFN: split the first layer by columns, apply ReLU locally, split the second layer by rows, then reduce partial sums. The GPUs share the matrix computation within one network segment; neither computes the whole network alone.'),
 L('不必在每个模块之后都拼回完整向量。这是 FFN 张量并行的一种常见组合。真实多卡执行还要考虑通信、同步和显存；本关不模拟这些成本。','You do not need to reconstruct the full vector after every module. This is a common tensor-parallel FFN pattern. Real execution must also account for communication, synchronization and GPU memory. This lesson does not simulate those costs.'),
 {'tp-col-a':['R'],'tp-col-b':['R'],'tp-relu-a':['tp-col-a'],'tp-relu-b':['tp-col-b'],'tp-row-a':['tp-relu-a'],'tp-row-b':['tp-relu-b'],'tp-reduce':['tp-row-a','tp-row-b'],sink:'tp-reduce'},x=>M.linear(M.linear(x,W).map(relu),D)]
];
const links={},refs={},known=new Set();
for(const [i,[id,title,nodes,sinkDim,mission,description,why,takeaway,solution,ref]]of definitions.entries()){
 const single=id.startsWith('single-');if(id==='tp-columns')known.clear();
 nodes.forEach(k=>known.add(k));const inventory=Object.fromEntries([...known].map(k=>[k,2]));
 M.levels[id]={id,title,nodes,sources:['R'],sourceDim:2,sinkDim,sampleCount:5,sequence:false,sourceX:-10.3,sinkX:10.3,zoom:1.22,boardWidth:26,boardDepth:14,sourceTitle:single?L('CPU 输入 X','CPU input X'):L('输入特征 X','Input features X'),sourceNote:'',outputTitle:single?L('CPU 中的结果','Result on the CPU'):L('完整计算结果','Complete result'),shape:'2 → '+sinkDim,mission,description,summary:description,concept:title,hints:[],rationale:{title:single?L('理解 CPU 与 GPU 的内存位置','Understand CPU and GPU memory locations'):L('理解两卡分片计算','Understand sharded computation across two GPUs'),text:why,takeaway,scope:single?singleScope:scope},track:'cloud',cloud:true,cloudSingle:single,...(id==='single-resident'?{copyLimit:16}:{}),chapter:L('云端推理 / 0'+(i+1),'Cloud / 0'+(i+1)),boardText:'CLOUD / 0'+(i+1)+(single?' · 1 GPU':' · 2 GPUs'),buildable:true,requirePlacement:true,startWithModules:false,maxModules:nodes.length+2,inventory,next:definitions[i+1]?.[0]||null,
 shelf:{packageKinds:single?['sg-upload2','sg-upload4','sg-download2','sg-download4']:['tp-gather','tp-reduce'],currentKinds:[...known],order:[...known],current:L('计算模块','Compute modules'),packages:L('数据搬运与收集','Data transfers and collection'),basics:single?L('CPU / GPU 算子','CPU / GPU operators'):L('分片与局部计算','Shards and local computation'),earlier:L('已学模块','Earlier modules'),note:L('已学模块仍可选择，不必全部使用。GPU 编号表示模块所属的芯片，不随台面位置改变。','Previously learned modules remain available. You do not need to use them all. The GPU number identifies each module’s chip, regardless of its workbench position.')}};
 links[id]=solution;refs[id]=ref;
}
M.routes={main:M.routes.main,modern:M.routes.modern,cloud:{name:L('实践 · 云端推理','Practice · Cloud inference'),description:L('先让一张 GPU 算起来，看看数据为什么要来回搬运；再试着让两张 GPU 分工，算出同样的结果。','Start with one GPU and see why data moves back and forth. Then share the work across two GPUs to get the same result.'),ids:definitions.map(d=>d[0])},engine:M.routes.engine};
M.routes.cloud.groups=[
 {id:'single',label:L('单机单卡','One GPU'),title:L('单卡搬运与计算','Transfers and computation on one GPU'),description:L('把输入送到 GPU，再把结果带回来。哪些中间结果可以不用来回搬？','Send input to the GPU and bring results back. Which intermediate results can stay there?'),ids:['single-upload','single-fallback','single-resident']},
 {id:'tensor-parallel',label:L('单机两卡','Two GPUs'),title:L('两卡分片计算','Sharded computation across two GPUs'),description:L('把一块矩阵分给两张卡：什么时候把结果拼起来，什么时候要相加？','Share a matrix across two GPUs: when should their results be joined, and when should they be added?'),ids:['tp-columns','tp-rows','tp-ffn']}
];I18n.prepare(M.routes.cloud.groups);
const priorEvaluate=M.evaluate;
M.evaluate=(state,input)=>{
 const ev=priorEvaluate(state,input),l=M.levels[state.levelId];if(!l?.cloudSingle||ev.error)return ev;
 const nodes=new Map(state.nodes.map(n=>[n.id,n])),used=new Set(),visit=id=>{if(id==='R'||used.has(id))return;used.add(id);nodes.get(id)?.links.forEach(visit);};visit(state.sink);
 const locations={R:'cpu'},copies=[];let failure=null;
 for(const id of ev.order.filter(id=>used.has(id))){
  const n=nodes.get(id),d=M.definitions[M.nodeKind(n)];
  if(!d.memoryInput||n.links.some(from=>locations[from]!==d.memoryInput)){failure={id,message:L('内存位置不匹配：','Memory location mismatch: ')+d.name+L(' 需要读取 ',' requires ')+(d.memoryInput||'GPU').toUpperCase()+L(' 中的张量。请用上传或取回模块搬运数据，并检查连线。',' tensors. Use an upload or download module to transfer the data, and check the wires.')};break;}
  locations[id]=d.memoryOutput;if(d.transfer)copies.push({id,from:d.memoryInput,to:d.memoryOutput,bytes:ev.values[id].length*4});
 }
 const bytes=copies.reduce((sum,c)=>sum+c.bytes,0);ev.cloudMemory={bytes,copies,locations,limit:l.copyLimit??null};
 if(!failure&&locations[state.sink]!=='cpu')failure={id:'sink',message:L('结果仍在 GPU 显存中。出口在 CPU，请用取回模块复制输出。','The result is still in GPU memory. The outlet is on the CPU. Copy the output with a download module.')};
 if(!failure&&l.copyLimit!=null&&bytes>l.copyLimit)failure={id:'sink',message:L('特征搬运量超过上限，请减少中间结果的搬运。当前 / 上限：','Feature copies exceed the limit. Reduce transfers of intermediate results. Current / limit: ')+bytes+' / '+l.copyLimit+' B'};
 if(failure){ev.error=failure.message;ev.bad={id:failure.id};}return ev;
};
M.cloudCopyLabels={title:L('本组输入的特征搬运量','Feature bytes copied for this input'),limit:L('上限','Limit'),note:L('只统计出口所需计算路径上的特征搬运。权重已驻留显存，每个 FP32 数按 4 字节计。这不是硬件耗时。','Counts include only feature copies on the computation path required by the outlet. Weights are resident on the GPU, and each FP32 value is 4 bytes. This is not a hardware timing.')};I18n.prepare(M.cloudCopyLabels);
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
