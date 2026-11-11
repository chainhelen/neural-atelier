(function(scope){
const weights={gate:[[1,-1,.5,0],[0,.5,1,1]],project:[[1,0,1,.5],[0,1,.5,-1]],down:[[.4,0],[0,.3],[.2,.2],[-.1,.3]]};
// "up" names the FFN weight's role, not a separate placeable operator.
weights.up=weights.project;
const norm=v=>{const rms=Math.sqrt(v.reduce((s,x)=>s+x*x,0)/v.length+1e-6);return v.map(x=>x/rms);};
const linear=(v,w)=>w[0].map((_,j)=>v.reduce((s,x,i)=>s+x*w[i][j],0));
const silu=x=>x/(1+Math.exp(-x));
const definitions={
 project:{name:I18n.t("线性投影"),hint:I18n.t("每一列权重生成一个新特征：一个输入向量中的 2 个数，组合成 4 个数。"),ins:[2],out:4,type:'linear',x:0,z:0},
 gate:{name:I18n.t("门投影"),hint:I18n.t("输入向量中的 2 个特征，乘权重 Wg，展开成 4 个。"),ins:[2],out:4,type:'linear',x:-4.6,z:-2.2},
 silu:{name:'SiLU',hint:I18n.t("对每个数做 x / (1 + exp(−x))，改变数值关系，不改变向量维度。"),ins:[4],out:4,type:'activation',x:-.9,z:-2.6},
 multiply:{name:I18n.t("逐元素相乘"),hint:I18n.t("两个 4 维向量对应位置相乘，得到四个乘积；不把这些乘积相加。"),ins:[4,4],out:4,type:'multiply',x:.1,z:1.8},
 down:{name:I18n.t("降维投影"),hint:I18n.t("4 维向量乘 4 × 2 权重矩阵 Wd，把四个特征组合成两个。"),ins:[4],out:2,type:'linear',x:3.5,z:1.5},
 add:{name:I18n.t("残差相加"),hint:I18n.t("原有特征加上支路算出的修改量。正的修改量让值增加，负的让值减少，零则保持该值。"),ins:[2,2],out:2,type:'add',x:6.4,z:-1.5}
};
const levels={
 wire:{id:'wire',chapter:I18n.t("序章 / 01"),title:I18n.t("让信号抵达"),nodes:[],sources:['R'],sinkDim:2,sampleCount:3,sourceX:-3.5,sinkX:3.5,zoom:.65,
  mission:I18n.t("把输入原样送到出口"),description:I18n.t("让每个输入位置的两个数完整抵达右边。先接一条线，再试跑三个位置。"),summary:I18n.t("只有输入和出口，先熟悉接线与试跑。"),concept:I18n.t("向量与连线"),sourceTitle:I18n.t("输入 X"),sourceNote:'',outputTitle:I18n.t("输出特征 Y"),boardText:'SIGNAL / 01',shape:'2 → 2',next:'projection',
  rationale:{title:I18n.t("为什么先传一份向量？"),text:I18n.t("计算模块接收的是表示 token 的一组数。连线决定下一个模块从哪里取数；先让向量原样抵达，才能看清后续变化来自哪一步计算。"),takeaway:I18n.t("[1, 0] 和 [0, 1] 是不同的特征。传递时保留数值和顺序，后面的权重才能作用到对应的特征上。")},
  hints:[I18n.t("圆口发送数据，方口接收数据。点两个端口，或从圆口直接拖线。")]},
 projection:{id:'projection',chapter:I18n.t("序章 / 02"),title:I18n.t("经过一次投影"),nodes:['project'],sources:['R'],sinkDim:4,sampleCount:3,sourceX:-5,sinkX:5.2,zoom:.75,
  mission:I18n.t("把 2 个特征变成 4 个"),description:I18n.t("用桌上的权重矩阵处理每个输入向量，让出口得到 4 个特征。三个输入位置共用同一套权重。"),summary:I18n.t("加入一块投影模块，观察矩阵如何改变向量。"),concept:I18n.t("一次矩阵乘法"),sourceTitle:I18n.t("输入 X"),sourceNote:'',outputTitle:I18n.t("投影特征 P"),boardText:'PROJECTION / 02',shape:'2 → 4',next:'activation',
  rationale:{title:I18n.t("为什么要重新组合特征？"),text:I18n.t("后续计算需要从输入中提取不同的组合。矩阵的每一列用一组权重组合原有特征，得到一种新的表示；真实模型会通过训练学习这些权重。"),takeaway:I18n.t("本关把 2 个数展开成 4 个，为后面的非线性变换提供更多组合。数变多了，信息仍来自同一份输入。")},
  hints:[I18n.t("端口上的数字是向量维度。双击模块，可以查看它的权重和输入输出。")]},
 activation:{id:'activation',chapter:I18n.t("序章 / 03"),title:I18n.t("让数值有弯曲"),nodes:['project','silu'],sources:['R'],sinkDim:4,sampleCount:3,sourceX:-6,sinkX:6,zoom:.83,boardWidth:17,boardDepth:7,
  positions:{project:[-2,0],silu:[2,0]},
  mission:I18n.t("改变数值，保留四个特征"),description:I18n.t("桌上的投影与 SiLU 还没有接线。让出口得到投影结果经过 SiLU 后的四个数，三个输入位置都要匹配。"),summary:I18n.t("新增 SiLU。维度相同的两份向量，数值不一定相同。"),concept:I18n.t("投影与非线性"),sourceTitle:I18n.t("输入 X"),sourceNote:'',outputTitle:I18n.t("激活特征 S"),boardText:'ACTIVATION / 03',shape:'2 → 4 → 4',next:'gating',
  rationale:{title:I18n.t("为什么矩阵后面还要有 SiLU？"),text:I18n.t("假设串联两个线性投影：A 是输入，B、C 是固定的权重矩阵。\n\n(A × B) × C = A × (B × C)\n\n先把 B、C 相乘得到新矩阵，就能代替这两层；对任何输入 A 都成立。\n\n用一个数举例：先乘 3，再乘 4，等于直接乘 12。\n分开算：(2 × 3) × 4 = 24\n合并算：2 × (3 × 4) = 2 × 12 = 24"),takeaway:I18n.t("所以，光堆线性层，仍然只是一次线性变换。SiLU 让缩放比例随数值变化：1 变为约 0.73，−1 变为约 −0.27。插在两层之间后，通常就不能合并为一个固定矩阵，从而能表达单纯线性变换做不到的关系。")},
  hints:[I18n.t("可以先试跑现有线路，比较出口的实际值与预期值。"),I18n.t("SiLU 对每个数单独计算，不改变向量长度；双击卡片可查看函数。")]},
 'down-projection':{id:'down-projection',title:I18n.t("把四个特征收回来"),nodes:['project','down'],sources:['R'],sinkDim:2,sampleCount:3,sourceX:-6,sinkX:6,zoom:.83,boardWidth:17,boardDepth:7,
  mission:I18n.t("把投影得到的 4 个特征变回 2 个"),description:I18n.t("桌上有熟悉的线性投影和新的降维投影，都还没有接线。让展开后的四个特征重新组合成两个输出。"),summary:I18n.t("只加入降维投影，先看清 4 → 2 的矩阵乘法。"),concept:I18n.t("降维投影"),sourceTitle:I18n.t("输入 X"),sourceNote:'',outputTitle:I18n.t("投影结果 F"),shape:'2 → 4 → 2',
  rationale:{title:I18n.t("为什么要回到两个特征？"),text:I18n.t("后面的模块仍使用两个特征。降维投影用两组权重分别组合前面的四个数，得到两个输出；它不是只留下前两个数。"),takeaway:I18n.t("4 × 2 矩阵的每一列都要乘上输入的四个数，再求和。现在只练这一步，后面再加入激活和残差。")},hints:[]},
 'plain-ffn':{id:'plain-ffn',title:I18n.t("先走完一条 FFN"),nodes:['project','silu','down'],sources:['R'],sinkDim:2,sampleCount:3,sourceX:-7.5,sinkX:7.5,zoom:.96,boardWidth:20,boardDepth:8,
  mission:I18n.t("展开、激活，再回到原宽度"),description:I18n.t("把前面分别玩过的三个模块串起来。目标是先展开特征，再经过 SiLU，最后得到两个输出。"),summary:I18n.t("组合三个熟悉的模块，完成一段单支路 FFN。"),concept:I18n.t("单支路 FFN"),sourceTitle:I18n.t("输入 X"),sourceNote:'',outputTitle:I18n.t("FFN 输出 F"),shape:'2 → 4 → 4 → 2',
  rationale:{title:I18n.t("这三步合起来有什么用？"),text:I18n.t("第一块矩阵把输入组合成更多特征，激活改变这些特征的数值关系，最后一块矩阵再将它们组合回原宽度。中间有非线性，整段通常不能合并成一块固定矩阵。"),takeaway:I18n.t("FFN 对每个 token 的特征分别加工；这一段没有从其他 token 取信息，也还没有残差相加。"),scope:I18n.t("本关是使用 SiLU 的单支路教学版。2017 年原始 Transformer 的 FFN 使用 ReLU；后面再学习双路的 SwiGLU。")},hints:[]},
 'gate-projection':{id:'gate-projection',title:I18n.t("换一套权重看看"),nodes:['gate'],sources:['R'],sinkDim:4,sampleCount:3,sourceX:-5,sinkX:5.2,zoom:.75,boardWidth:15,boardDepth:7,
  mission:I18n.t("用另一套权重产生四个特征"),description:I18n.t("门投影仍然做熟悉的 2 → 4 矩阵乘法。本关只算它的输出 G，先不加入 SiLU 或相乘。"),summary:I18n.t("认识第二套投影权重，先不搭双路网络。"),concept:I18n.t("同种运算，不同权重"),sourceTitle:I18n.t("输入 X"),sourceNote:'',outputTitle:I18n.t("门投影结果 G"),shape:'2 → 4',
  rationale:{title:I18n.t("为什么还要另一套投影？"),text:I18n.t("内容特征和调节它的系数可以依赖不同的输入组合。因此两条支路各有一套权重；同一份输入经过它们，会得到不同的数。"),takeaway:I18n.t("例如输入 [1, 0]，熟悉的投影得到 [1, 0, 1, 0.5]，门投影得到 [1, −1, 0.5, 0]。G 之后经过 SiLU 才会用于门控，本关先学会把 G 算出来。"),scope:I18n.t("从这里开始学习 SwiGLU 这种现代 FFN 设计。它增加了门分支，2017 年原始 Transformer 的 FFN 没有这一支路。")},hints:[]},
 pairwise:{id:'pairwise',title:I18n.t("先把两组数配对"),nodes:['project','gate','multiply'],sources:['R'],sinkDim:4,sampleCount:3,sourceX:-7.5,sinkX:7.5,zoom:.96,boardWidth:20,boardDepth:9,
  initialLinks:{project:['R'],gate:['R']},mission:I18n.t("得到两组数对应位置的乘积"),description:I18n.t("两块投影已各自接到输入。只需把它们算出的两组数交给逐元素相乘，得到四个乘积。"),summary:I18n.t("只新增逐元素相乘，暂时没有 SiLU。"),concept:I18n.t("对应位置相乘"),sourceTitle:I18n.t("输入 X"),sourceNote:'',outputTitle:I18n.t("相乘结果 H"),shape:'2 → 4 + 4 → 4',
  rationale:{title:I18n.t("相乘到底改变了什么？"),text:I18n.t("每一项都有自己的乘数：乘 0 会消去这一项，乘 0.5 会减半，乘 2 会加倍。四对数分别相乘，不把不同位置的乘积加起来。"),takeaway:I18n.t("例如 [2, 4, 1, −2] 与 [0.5, 0, 2, 1] 相乘，得到 [1, 0, 2, −2]。先看懂配对，下一关再让一条支路经过 SiLU。"),scope:I18n.t("本关直接相乘两组投影结果，是逐元素运算练习，还不是 SwiGLU。")},hints:[]},
 gating:{id:'gating',chapter:I18n.t("序章 / 04"),title:I18n.t("让两路相遇"),nodes:['gate','project','silu','multiply'],sources:['R'],sinkDim:4,sampleCount:5,sourceX:-8,sinkX:8,zoom:1,boardWidth:21,boardDepth:10,
  positions:{gate:[-4,-2],project:[-4,2],silu:[0,-2],multiply:[3.5,1.2]},initialLinks:{gate:['R'],project:['R']},
  mission:I18n.t("把两组数一一相乘"),description:I18n.t("两块投影接收同一份输入，分别算出 4 个数。用 SiLU 处理其中一组，再让两组数一一相乘。接好模块，让 5 组输入都得到目标结果。"),summary:I18n.t("新增两路分支和逐元素相乘。自己决定哪条支路经过激活。"),concept:I18n.t("两路门控"),sourceTitle:I18n.t("输入 X"),sourceNote:'',outputTitle:I18n.t("门控特征 H"),boardText:'GATING / 04',shape:'2 → 4 + 4 → 4',next:'residual',
  rationale:{text:I18n.t("同样强度的特征，不一定对每种输入都同样有用。内容分支算出特征，门分支根据当前输入算调节系数；相乘后，就能对各个特征分别减弱或放大。"),inputs:[[1,0],[0,2]],feature:2,takeaway:I18n.t("同样的内容值，一个被减弱，另一个被放大。输入改变了系数，两套权重没有变。"),scope:I18n.t("真实模型通过训练学习怎样产生有用的特征和系数。本关用固定小矩阵演示 SwiGLU，这是 FFN 的一种设计。")},
  hints:[I18n.t("端口维度相同只说明能接上，不代表计算结果正确。"),I18n.t("逐元素相乘把两条支路对应的数相乘；激活放在哪一条支路会改变结果。")]},
 residual:{id:'residual',chapter:I18n.t("序章 / 05"),title:I18n.t("留一条回来的路"),nodes:['project','down','add'],sources:['R'],sinkDim:2,sampleCount:5,sourceX:-7.5,sinkX:7.5,zoom:.96,boardWidth:20,boardDepth:9,
  positions:{project:[-3.8,1.3],down:[0,1.3],add:[3.8,-.3]},
  mission:I18n.t("在原有特征上做调整"),description:I18n.t("桌上两块投影算出 2 维修改量 F。出口要得到 X + F，在原有特征 X 的基础上完成一次调整。接好模块，让 5 组输入都匹配目标。"),summary:I18n.t("用残差学习修改量：在已有特征上增加或减少，而无需重新复制每个原值。"),concept:I18n.t("降维与残差"),sourceTitle:I18n.t("输入 X"),sourceNote:'',outputTitle:I18n.t("残差特征 Y"),boardText:'RESIDUAL / 05',shape:'2 → 4 → 2 + 2',next:'ffn',
  rationale:{kind:'residual',title:I18n.t("为什么要加回输入？"),input:[1,0],text:I18n.t("假如这一层只需要调整部分特征，支路就只需学习“加多少或减多少”。无需修改的值沿直连路径继续传递，支路不用再学一遍如何复制它们。"),takeaway:I18n.t("修改量为 0 时，输出就是原输入；修改量为负时，对应的值会减小。相加后的数值仍可能改变，甚至抵消到 0。"),scope:I18n.t("真实模型通过训练学习修改量。直连路径也为训练信号提供回传路线，有助于训练更深的网络。本关的矩阵是固定的。")},
  hints:[I18n.t("一个圆形输出口可以连接多条线，不会消耗原来的向量。"),I18n.t("相加的两份向量需要同样长度；一份是原输入，一份是刚算出的变化。")]},
 swiglu:{id:'swiglu',title:I18n.t("给门控接上出口"),nodes:['gate','project','silu','multiply','down'],sources:['R'],sinkDim:2,sampleCount:3,sourceX:-8.5,sinkX:8.5,zoom:1.05,boardWidth:23,boardDepth:11,
  initialLinks:{gate:['R'],project:['R'],silu:['gate'],multiply:['silu','project']},mission:I18n.t("让门控特征回到原宽度"),description:I18n.t("上关的两路门控已经接好。把四个门控特征组合成两个输出，先完成 FFN 本身。"),summary:I18n.t("把已完成的门控接到熟悉的降维投影，还不加残差。"),concept:I18n.t("SwiGLU FFN"),sourceTitle:I18n.t("输入 X"),sourceNote:'',outputTitle:I18n.t("FFN 输出 F"),shape:'2 → 4 + 4 → 4 → 2',
  rationale:{title:I18n.t("这一步为什么才是完整 FFN？"),text:I18n.t("门控先调整展开后的四个特征；降维投影再把它们组合成后续网络需要的两个特征。双路投影、SiLU、相乘和降维合起来，就是这里的 SwiGLU FFN。"),takeaway:I18n.t("先前的单支路 FFN 直接激活内容特征；这里用另一条支路产生系数来调节内容。残差相加在 FFN 外，下一步再把输出用作修改量。"),scope:I18n.t("SwiGLU 是 FFN 的一种现代实现，不是整个 Transformer，也不是 2017 年原始 FFN 的固定结构。")},hints:[]},
 workshop:{id:'workshop',chapter:I18n.t("序章 / 06"),title:I18n.t("从架上拿一块"),nodes:['project'],inventory:{project:1},maxModules:1,buildable:true,placementTutorial:true,sources:['R'],sinkDim:4,sampleCount:3,sourceX:-5.5,sinkX:5.5,zoom:.8,boardWidth:16,boardDepth:7,
  mission:I18n.t("拖进一块投影，再接上线"),description:I18n.t("台面上少了一块熟悉的投影。从组件架拖一块进来，连接输入和出口，然后试跑。"),summary:I18n.t("只学一个新操作：从组件架拖入模块。"),concept:I18n.t("自己放置组件"),sourceTitle:I18n.t("输入 X"),sourceNote:'',outputTitle:I18n.t("投影特征 P"),boardText:'WORKSHOP / 06',shape:'2 → 4',next:'ffn',
  rationale:{title:I18n.t("为什么开始自己拿组件？"),text:I18n.t("后面的完整网络会有分支和合流，需要你决定使用哪些模块、各放几块以及如何取数。本关先用熟悉的投影练习放置，为独立搭建计算图做准备。"),takeaway:I18n.t("台面上的摆放位置方便你组织线路；真正决定计算结果的是模块的运算、权重和接线。")},
  hints:[I18n.t("按住组件架上的卡片，拖到台面松开。再把输入、投影、出口连起来。") ]},
 ffn:{id:'ffn',chapter:I18n.t("正式关卡 / 01"),title:I18n.t("让一层醒来"),nodes:['gate','project','silu','multiply','down','add'],sources:['Z','R'],sinkDim:2,sampleCount:5,sourceX:-8.55,sinkX:9,zoom:1,
  mission:I18n.t("搭出一段 FFN"),description:I18n.t("把序章学过的投影、门控、降维和残差组合起来。Z 是归一化后的输入，R 是原输入；自行搭建，让出口匹配目标结果。"),summary:I18n.t("把双路门控与降维组成 SwiGLU FFN，再用残差更新输入。"),concept:I18n.t("SwiGLU FFN + 残差"),sourceTitle:I18n.t("层输入 R"),sourceNote:I18n.t("上游 Attention 的输出特征"),outputTitle:I18n.t("本层特征 Y"),boardText:'FFN / SWIGLU',shape:'2 → 4 → 2',next:null,
  initialLinks:{gate:['Z'],project:['Z']},
  rationale:{kind:'swiglu',title:I18n.t("为什么还需要一段 FFN？"),text:I18n.t("Attention 从其他位置汇入信息后，每个位置还需要加工自己的特征。FFN 根据当前向量产生修改量，再通过残差更新这个位置的表示。"),takeaway:I18n.t("降维投影输出 FFN 的修改量；残差相加在 FFN 外把它加回原输入。同一套 FFN 权重分别处理每个位置，不在这里混合不同 token。"),scope:I18n.t("Swi 来自 Swish（这里就是 SiLU），GLU 指“门控线性单元”。SiLU 产生的系数也可以为负或超过 1。")},
  hints:[I18n.t("圆口是输出，方口是输入；输入可以分叉给多个模块。"),I18n.t("SiLU 把门投影的数变成系数；系数与线性投影的内容特征逐元素相乘。"),I18n.t("降维后的变换 F，需要加回进入 FFN 前的原始 R。")]}
};
function levelFor(state){return levels[state.levelId]||levels.ffn;}
const nodeKind=n=>n?.kind||n?.id;
const kindOf=(state,id)=>nodeKind(state.nodes.find(n=>n.id===id))||id;
const definitionFor=(state,id)=>definitions[kindOf(state,id)];
const inventory=level=>level.inventory||Object.fromEntries(level.nodes.map(k=>[k,1]));
const moduleLimit=level=>level.maxModules??level.nodes.length;
// Some lessons reuse an identical operator at several named positions. Keep
// those saved kinds intact, but offer one shelf entry with a shared copy limit.
const shelfKind=(level,kind)=>level.componentAliases?.[kind]||kind;
const moduleName=(level,kind)=>level.componentNames?.[shelfKind(level,kind)]||definitions[kind]?.name||kind;
function stock(state,kind){const level=levelFor(state),representative=shelfKind(level,kind);return Math.max(0,(inventory(level)[kind]?(inventory(level)[representative]||0):0)-state.nodes.filter(n=>shelfKind(level,nodeKind(n))===representative).length);}
function canAdd(state,kind){const level=levelFor(state);return !!level.buildable&&stock(state,kind)>0&&state.nodes.length<moduleLimit(level);}
function nextId(state,kind){if(!state.nodes.some(n=>n.id===kind))return kind;let i=2;while(state.nodes.some(n=>n.id===kind+'-'+i))i++;return kind+'-'+i;}
function validNodeId(id,kind){return typeof id==='string'&&typeof kind==='string'&&(id===kind||new RegExp('^'+kind+'-(?:[2-9]|[1-9][0-9]+)$').test(id));}
function nodeName(state,id){const kind=kindOf(state,id),d=definitions[kind];if(!d)return id;const level=levelFor(state),representative=shelfKind(level,kind),copies=state.nodes.filter(n=>shelfKind(level,nodeKind(n))===representative),grouped=Object.values(level.componentAliases||{}).includes(representative);return moduleName(level,kind)+(copies.length>1?' · '+(grouped?copies.findIndex(n=>n.id===id)+1:id===kind?'1':id.slice(kind.length+1)):'');}

function bounds(level){return {left:level.sourceX+2.8,right:level.sinkX-2.8,back:-(level.boardDepth||(level.id==='ffn'?11:7))/2+1.6,front:(level.boardDepth||(level.id==='ffn'?11:7))/2-1.6};}
function shuffled(items,random=Math.random){const list=[...items];for(let i=list.length-1;i>0;i--){const j=Math.floor(random()*(i+1));[list[i],list[j]]=[list[j],list[i]];}return list;}
function arrange(state,random=Math.random){
 const b=bounds(levelFor(state)),count=state.nodes.length;if(!count)return state;
 const maxCols=Math.max(1,Math.floor((b.right-b.left)/3.25)+1),cols=Math.min(maxCols,Math.ceil(Math.sqrt(count*1.5))),rows=Math.ceil(count/cols);
 const dx=cols>1?Math.min(3.8,(b.right-b.left)/(cols-1)):0,dz=rows>1?Math.min(3.5,(b.front-b.back)/(rows-1)):0;
 const slots=shuffled(Array.from({length:cols*rows},(_,i)=>[(b.left+b.right)/2+(i%cols-(cols-1)/2)*dx,(Math.floor(i/cols)-(rows-1)/2)*dz]),random);
 state.nodes.forEach((n,i)=>{n.x=Math.max(b.left,Math.min(b.right,slots[i][0]+(random()-.5)*.24));n.z=Math.max(b.back,Math.min(b.front,slots[i][1]+(random()-.5)*.24));});return state;
}
function initial(levelId='ffn',{complete=false,random=Math.random}={}){const level=levels[levelId]||levels.ffn,empty=level.buildable&&!level.startWithModules&&!complete;
 const copies={};
 const state={version:1,levelId:level.id,...(level.requirePlacement?{placementRevision:1}:{}),palette:Object.keys(inventory(level)),...(level.engine?{settings:{...api.engineDefaults,...level.settings}}:{}),nodes:(empty?[]:level.nodes).map(kind=>{const d=definitions[kind],copy=copies[kind]=(copies[kind]||0)+1,id=copy===1?kind:kind+'-'+copy;return {id,kind,x:d.x??0,z:d.z??0,...(level.engine?{device:level.devices?.[id]||'gpu'}:{}),links:level.initialLinks?.[id]?[...level.initialLinks[id]]:d.ins.map(()=>null)};}),sink:empty?null:level.initialSink||null};return arrange(state,random);
}
function placement(state,kind,x,z){const level=levelFor(state),b=bounds(level);if(!canAdd(state,kind))return null;if(!Number.isFinite(x)||!Number.isFinite(z)||x<b.left||x>b.right||z<b.back||z>b.front)return null;if(state.nodes.some(n=>Math.abs(n.x-x)<2.65&&Math.abs(n.z-z)<2.9))return null;return {id:nextId(state,kind),kind,x,z,...(level.engine?{device:level.devices?.[kind]||'gpu'}:{}),links:definitions[kind].ins.map(()=>null)};}
function addNode(state,id,x,z){const n=placement(state,id,x,z);if(!n)return false;state.nodes.push(n);return true;}
function removeNode(state,id){if(!levelFor(state).buildable||!state.nodes.some(n=>n.id===id))return false;state.nodes=state.nodes.filter(n=>n.id!==id);state.nodes.forEach(n=>n.links=n.links.map(k=>k===id?null:k));if(state.sink===id)state.sink=null;return true;}
function disconnect(state,to,port){if(to==='sink'){if(!state.sink)return false;state.sink=null;return true;}const n=state.nodes.find(n=>n.id===to);if(!n?.links[port])return false;n.links[port]=null;return true;}
const samples=[{word:I18n.t("我"),v:[1,0]},{word:I18n.t("爱"),v:[0,1]},{word:I18n.t("猫"),v:[1,1]},{word:I18n.t("新输入 A"),v:[-1,2]},{word:I18n.t("新输入 B"),v:[2,-.5]}];
function reference(v,levelId='ffn'){
 if(levelId==='wire')return [...v];
 const p=linear(v,weights.project);
 if(levelId==='projection'||levelId==='workshop')return p;
 if(levelId==='activation')return p.map(silu);
 if(levelId==='down-projection')return linear(p,weights.down);
 if(levelId==='plain-ffn')return linear(p.map(silu),weights.down);
 if(levelId==='gate-projection')return linear(v,weights.gate);
 if(levelId==='pairwise'){const g=linear(v,weights.gate);return p.map((x,i)=>x*g[i]);}
 if(levelId==='residual')return linear(p,weights.down).map((x,i)=>x+v[i]);
 const z=['gating','swiglu'].includes(levelId)?v:norm(v),g=linear(z,weights.gate).map(silu),u=linear(z,weights.up),h=g.map((x,i)=>x*u[i]);
 if(levelId==='gating')return h;
 const f=linear(h,weights.down);return levelId==='swiglu'?f:v.map((x,i)=>x+f[i]);
}
const sourceValues=(state,v)=>Object.fromEntries(levelFor(state).sources.map(id=>[id,id==='Z'?norm(v):v]));
function evaluate(state,v){const level=levelFor(state),context={state,level,input:v},values=api.sourceValues(state,v),depths=Object.fromEntries(level.sources.map(id=>[id,0])),visiting=new Set(),order=[];let error=null,bad=null;
 function run(id){if(id==null)throw {message:I18n.t("有一个输入口还没有接线。")};if(values[id])return values[id];const node=state.nodes.find(n=>n.id===id),kind=nodeKind(node),d=definitions[kind];if(!node||!d)throw {message:I18n.t("线路指向了不存在的模块。")};if(visiting.has(id))throw {message:I18n.t("线路绕成了环，数值走不出来。"),node:id};visiting.add(id);
 const inputs=d.ins.map((dim,i)=>{if(!node.links[i])throw {message:`「${moduleName(level,kind)}${I18n.t("」的输入 ")}${i+1}${I18n.t(" 还没接上。")}`,node:id,port:i};const a=run(node.links[i]);if(a.length!==dim)throw {message:`「${moduleName(level,kind)}${I18n.t("」需要 ")}${dim}${I18n.t(" 维，收到 ")}${a.length}${I18n.t(" 维。")}`,node:id,port:i};return a;});
 let out;if(api.operations?.[kind])out=api.operations[kind](inputs,context);else if(weights[kind])out=linear(inputs[0],weights[kind]);else if(kind==='silu')out=inputs[0].map(silu);else out=inputs[0].map((x,i)=>kind==='multiply'?x*inputs[1][i]:x+inputs[1][i]);
 values[id]=out;depths[id]=1+Math.max(...node.links.map(k=>depths[k]));order.push(id);visiting.delete(id);return out;}
 const dim=levelFor(state).sinkDim;
 let output=null;try{if(!state.sink)throw {message:I18n.t("出口还没有接上。"),node:'sink',port:0};output=run(state.sink);if(output.length!==dim)throw {message:`${I18n.t("出口需要 ")}${dim}${I18n.t(" 维向量。")}`,node:'sink'};}catch(e){error=e.message;bad={id:e.node,port:e.port};}
 // Every runnable branch participates in the visual execution, even when the
 // final outlet is disconnected or another branch cannot finish. Keep the
 // original final-output validation separate from this execution trace.
 for(const node of state.nodes){visiting.clear();try{run(node.id);}catch{}}
 return {output,values,depths,order,error,bad};}
function valueForKind(state,ev,kind){
 if(!ev)return null;const pending=[state.sink],seen=new Set();
 while(pending.length){const id=pending.shift();if(!id||seen.has(id))continue;seen.add(id);const n=state.nodes.find(n=>n.id===id);if(!n)continue;if(nodeKind(n)===kind)return ev.values[id]||null;pending.push(...n.links);}
 return null;
}
function connect(state,from,to,port){const level=levelFor(state);if(!level.sources.includes(from)&&!state.nodes.some(n=>n.id===from))return I18n.t("没有找到这个端口");if(to!=='sink'&&!state.nodes.some(n=>n.id===to))return I18n.t("没有找到这个端口");const out=from==='R'||from==='Z'?(level.sourceDim||2):definitionFor(state,from)?.out,dim=to==='sink'?level.sinkDim:definitionFor(state,to)?.ins[port];if(!out||!dim)return I18n.t("没有找到这个端口");if(from===to)return I18n.t("不能把输出接回自己");if(out!==dim)return `${I18n.t("维度接不上：输出 ")}${out}${I18n.t(" 维，入口需要 ")}${dim}${I18n.t(" 维。")}`;if(to==='sink')state.sink=from;else state.nodes.find(n=>n.id===to).links[port]=from;return null;}
const api={sourceValues,valueForKind,nodeKind,kindOf,definitionFor,inventory,moduleLimit,shelfKind,moduleName,stock,canAdd,validNodeId,nodeName,weights,norm,linear,silu,definitions,levels,levelFor,initial,samples,reference,evaluate,connect,disconnect,bounds,arrange,placement,addNode,removeNode};if(typeof module!=='undefined'&&module.exports)module.exports=api;else scope.NeuralPrologue=api;
})(globalThis);
