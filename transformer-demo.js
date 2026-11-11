/* A prepared, fully expanded original Transformer. No packaged operators. */
(function(M){
const L=(zh,en)=>{AtelierEnglish[zh]=en;return I18n.t(zh);};
const O=M.original,id='transformer-demo',flat=x=>x.flat();
const copy=x=>JSON.parse(JSON.stringify(x));
const definitions={
 'demo-up':{name:L('线性投影 · 4 → 8','Linear projection · 4 → 8'),ins:[12],out:24,type:'linear',hint:L('每个词的四个特征乘 W1，展开为八个特征。三个位置各自计算，共用同一套权重。','Multiply each token’s four features by W1 to expand to eight. The three positions are processed separately with the same weights.')},
 'demo-relu':{name:'ReLU',ins:[24],out:24,type:'activation',activation:'relu',hint:L('对三个词各自的八个特征逐个计算 max(0, x)，负数归零，正数保留。','Apply max(0, x) to each of the eight features at all three positions: zero negatives and keep positives.')},
 'demo-down':{name:L('降维投影 · 8 → 4','Down projection · 8 → 4'),ins:[24],out:12,type:'linear',hint:L('每个词的八个激活特征乘 W2，组合回模型使用的四个特征。','Multiply each token’s eight activated features by W2 to return to the model’s four features.')},
 'demo-add':{name:L('残差相加','Residual addition'),ins:[12,12],out:12,type:'residual',hint:L('把这一段算出的更新，加回进入这一段时的特征。两边都是三个词、每词四个特征。','Add the stage’s computed update back to its input: three tokens with four features on each side.')},
 'demo-ln':{name:'LayerNorm',ins:[12],out:12,type:'layernorm',hint:L('每个词单独减特征均值，再除以标准差。本例 γ = 1、β = 0，位于残差相加之后。','Center each token’s features and divide by their standard deviation. Here γ = 1 and β = 0; this follows residual addition.')},
 'demo-head':{...M.definitions['original-head'],bias:[0,0,0,0,0,0],hint:L('把每个目标位置的四个特征乘以 4 × 6 词表矩阵，得到六个候选词分数。这关使用专门挑选的固定教学权重，让三轮生成能展示不同词；仍由后面的 Softmax 和选词决定输出。','Multiply each target position’s four features by a 4 × 6 vocabulary matrix to score six candidates. This lesson uses fixed teaching weights chosen to produce different tokens across three rounds; Softmax and token selection still determine the output.')},
 'demo-probabilities':{name:L('Softmax · 词表概率','Softmax · vocabulary probabilities'),ins:[18],out:18,type:'softmax',columns:6,hint:L('每个目标位置的六个候选分数分别做 Softmax，得到合计为 1 的六个概率。这里分配的是候选词概率，不是注意力比例。','Apply Softmax to the six candidate scores at each target position to get six probabilities summing to one. These are vocabulary probabilities, not attention weights.')}
};
Object.assign(M.definitions,definitions);
M.weights['demo-up']=O.wide.up;M.weights['demo-down']=O.wide.down;
// Fixed readout columns aligned with the decoder states for the three demo
// prefixes (and their opposites). Every round uses this same matrix; no
// round counter, transition table or substitution changes the selected token.
M.weights['demo-head']=[[-4.46,-4.48,2.4,4.46,4.48,-2.4],[-.14,2.82,-4.66,.14,-2.82,4.66],[3.96,-.98,2.86,-3.96,.98,-2.86],[.63,2.64,-.61,-.63,-2.64,.61]];
const softmax=v=>{const max=Math.max(...v),e=v.map(x=>Math.exp(x-max)),sum=e.reduce((a,b)=>a+b,0);return e.map(x=>x/sum);};
Object.assign(M.operations,{
 'demo-up':([x])=>O.project(x,O.wide.up),'demo-down':([x])=>O.project(x,O.wide.down),'demo-relu':([x])=>x.map(O.relu),
 'demo-add':([a,b])=>a.map((x,i)=>x+b[i]),'demo-ln':([x])=>flat(M.math.rows(x,4).map(O.ln)),
 'demo-head':([x])=>O.project(x,M.weights['demo-head']),
 'demo-probabilities':([x])=>flat(M.math.rows(x,6).map(softmax))
});
for(const [kind,d]of Object.entries(definitions))M.extraColors[kind]=({linear:0x789abf,activation:0xc6a574,residual:0xc3a26f,layernorm:0xb8a36a,softmax:0xb79b6b,vocabulary:0xb79b6b})[d.type];
const nodes=[],counts={},sections={},sourceNodes=new Set(['Z']);
function put(kind,links,x,z,section){
 const number=counts[kind]=(counts[kind]||0)+1,nodeId=kind+(number===1?'':'-'+number);
 nodes.push({id:nodeId,kind,links,x,z});sections[nodeId]=section;if(section==='encoder'||links.length===1&&sourceNodes.has(links[0])&&M.definitions[kind].type==='linear')sourceNodes.add(nodeId);return nodeId;
}
function attention(query,kv,z,section,masked=false){
 const heads=['a','b'].map((head,i)=>{
  const hz=z+(i?2.3:-2.3);
  const q=put('masked-'+head+'-q',[query],-25,hz,section),k=put('masked-'+head+'-k',[kv],-21,hz,section),v=put('masked-'+head+'-v',[kv],-17,hz,section);
  const scores=put('score',[q,k],-13,hz,section),s=masked?put('mask',[scores],-9,hz,section):scores;
  const a=put('softmax',[s],masked?-5:-9,hz,section);
  return put('aggregate',[a,v],masked?-1:-5,hz,section);
 });
 const c=put('wide-concat',heads,masked?3:-1,z,section);
 return put('original-output',[c],masked?7:3,z,section);
}
function residual(input,update,x,z,section){return put('demo-ln',[put('demo-add',[input,update],x,z,section)],x+4,z,section);}
function ffn(input,x,z,section){
 const up=put('demo-up',[input],x,z,section),a=put('demo-relu',[up],x+4,z,section),down=put('demo-down',[a],x+8,z,section);
 return residual(input,down,x+12,z,section);
}
const source=put('wide-embedding',['Z'],-33,-15,'encoder'),xp=put('wide-position',[source],-29,-15,'encoder');
const enc=ffn(residual(xp,attention(xp,xp,-15,'encoder'),7,-15,'encoder'),15,-15,'encoder');
const target=put('wide-embedding',['R'],-33,0,'masked'),tp=put('wide-position',[target],-29,0,'masked');
const masked=residual(tp,attention(tp,tp,0,'masked',true),11,0,'masked');
const crossed=residual(masked,attention(masked,enc,15,'cross'),7,15,'cross');
const decoded=ffn(crossed,15,15,'cross');
const logits=put('demo-head',[decoded],23,0,'readout'),prob=put('demo-probabilities',[logits],27,0,'readout'),picked=put('pick',[prob],31,0,'readout');
for(const n of nodes)n.y=({encoder:6,masked:0,cross:-6,readout:0})[sections[n.id]];
const title=L('运行完整的原版 Transformer','Run the complete original Transformer');
const demo=M.levels[id]={id,title,chapter:L('原版基础 / 21','Original / 21'),boardText:'ORIGINAL / 21',track:'main',original:true,demonstration:true,floating:true,generationRounds:3,
 nodes:nodes.map(n=>n.kind),inventory:counts,buildable:false,sequence:true,wide:true,sources:['R','Z'],sourceDim:3,sinkDim:3,sampleCount:3,
 dualSource:true,sourceOffsets:{Z:-15,R:0},sourceHeights:{Z:6,R:0},secondaryTokenIds:[...O.sourceIds],sourceX:-38,sinkX:37,boardWidth:83,boardDepth:45,zoom:4.3,labelZoom:1.5,
 sourcePorts:{R:'D ID',Z:'S ID'},sourceNames:{R:L('目标前缀 ID','Target prefix IDs'),Z:L('源序列 ID','Source sequence IDs')},
 sourceTitle:L('目标前缀 ID','Target prefix IDs'),sourceNote:L('解码器读取已生成的词','The decoder reads the existing prefix'),secondaryTitle:L('编码器读取完整源句','The encoder reads the complete source'),
 outputTitle:L('各目标位置的预测词','Predictions at target positions'),outputNote:L('最后一个位置预测下一词','The final position predicts the next token'),tokenOutput:true,shapes:{24:'3 × 8'},shape:L('四维 · 双头 · 编码器、解码器各一层','4 features · 2 heads · one encoder and decoder layer'),next:null,
 mission:L('点一次试跑，看完整模型流动','Press Run to watch the complete model'),
 description:L('点一次试跑，连续生成三个新词。每轮只把最后位置的预测词追加到文本并送回目标输入。可调速或暂停；最慢档逐列计算。镜头跟随数据，拖动或滚轮缩放可接管视角。','Press Run to generate three new tokens. Each round appends only the final position’s prediction and feeds it back to the target input. Adjust speed or pause; the slowest setting computes column by column. The camera follows data; drag or scroll to take control.'),
 summary:L('只需运行。展开完整模型，连续观察三轮生成与回送。','Just run. Unfold the complete model and watch three rounds of generation and feedback.'),concept:title,hints:[],
 rationale:{title:L('之前的组件，怎样组成一个模型？','How do the familiar modules form a model?'),
 text:L('编码器先整理源句。解码器先读自己的目标前缀，再通过交叉注意力读取编码器输出；两边的 FFN 分别加工各词的特征。每个子层后都接残差相加和 LayerNorm。','The encoder processes the source. The decoder first reads its own target prefix, then reads the encoder output through cross-attention. Each side’s FFN processes token features. Every sublayer is followed by residual addition and LayerNorm.'),
 takeaway:L('每个注意力头的 Q、K、V、缩放打分、Softmax 和 A × V 都单独摆出；两个头拼接后经过 Wo。词表投影与词表 Softmax 在模型末端；取最高分是额外的选词步骤。','Each head’s Q, K, V, scaled scores, Softmax and A × V are separate modules. Concatenate both heads and apply Wo. Vocabulary projection and vocabulary Softmax end the model; taking the maximum is an additional selection step.'),
 scope:L('按 2017 年原版推理结构缩尺：编码器、解码器各一层，四维、双头，FFN 为 4 → 8 → 4，关闭 dropout。为保持矩阵大小，本演示每轮读取最近三个目标词，完整文字保留在上方；这是演示窗口，不是原版模型的上下文限制。源句保持不变，每轮重算，没有 KV Cache。词表投影使用本关专用的固定教学权重，让三轮选出不同词；未经过语言训练，不代表翻译能力。','A scaled 2017 inference structure: one encoder and decoder layer, four features, two heads, 4 → 8 → 4 FFNs, dropout off. To keep matrix sizes fixed, each round reads the latest three target tokens; the full text remains above. This is a display window, not the original model’s context limit. The source stays unchanged and is recomputed each round, without a KV cache. The vocabulary projection uses fixed weights chosen for this lesson so the three rounds select different tokens. It is not language-trained and does not demonstrate translation ability.')},
 menu:{kind:'connect',label:L('完整演示 · 只需运行','Full demonstration · just run'),description:L('观察已连好的完整编码器—解码器连续生成三个词，并将每轮预测送回输入。','Watch the fully wired encoder–decoder generate three tokens, feeding each round’s prediction back to the input.')},
 demoViews:[{key:'all',label:L('全景','Overview'),x:0,z:0,zoom:4.3},{key:'encoder',label:L('编码器','Encoder'),x:-1,y:6,z:-15,zoom:3.3},{key:'masked',label:L('因果自注意力','Masked self-attention'),x:-10,z:0,zoom:2.5},{key:'cross',label:L('交叉注意力 + FFN','Cross-attention + FFN'),x:0,y:-6,z:15,zoom:3.3},{key:'readout',label:L('词表与选词','Vocabulary & selection'),x:29,z:0,zoom:1.05}],
 demoSections:[{label:L('编码器 · 自注意力 → 残差与归一化 → FFN','ENCODER · Self-attention → Add & Norm → FFN'),x:-4,y:6,z:-20.8,color:'#4e8877'},
 {label:L('解码器 ① · 因果自注意力 → 残差与归一化','DECODER 1 · Masked attention → Add & Norm'),x:-10,z:-5.8,color:'#8d779e'},
 {label:L('解码器 ② · 读取编码器 → 残差与归一化 → FFN','DECODER 2 · Read encoder → Add & Norm → FFN'),x:-4,y:-6,z:9.2,color:'#8d779e'},
 {label:L('模型末端 · 词表分数 → 概率 → 选词','OUTPUT · Vocabulary scores → probabilities → selection'),x:28,z:-4.5,color:'#a18743'}]
};
demo.summary=demo.menu.description;demo.menu.summary=demo.menu.description;
M.routes.main.ids.push(id);M.routes.main.groups.at(-1).ids.push(id);M.levels.autoregressive.next=id;
const group=M.routes.main.groups.at(-1);group.description=L('先学词表打分，再连接完整模型、选词与回送，最后观察三轮生成。','Learn vocabulary scoring, then assemble the model, select tokens and feed them back before observing three generation rounds.');
const initial=M.initial,solutions=M.solutions,samples=M.samplesFor,sources=M.sourceValues,reference=M.reference,example=M.extraExample,card=M.extraCard,moduleShape=M.moduleShape;
M.initial=(levelId,options)=>levelId===id?{version:1,levelId:id,palette:[],nodes:copy(nodes),sink:picked}:initial(levelId,options);
M.solutions=levelId=>levelId===id?M.initial(id):solutions(levelId);
const sampleFor=v=>({word:v.map(i=>M.math.vocab[i]).join(' / '),v:[...v]});
const nextInput=(v,output)=>[...v.slice(1),output.at(-1)];
const readout=v=>M.operations['demo-head']([O.decoder(O.position(O.embed(v,4)),O.encoder(O.position(O.embed(O.sourceIds,4))))]);
M.samplesFor=levelId=>{
 if(levelId!==id)return samples(levelId);
 const rounds=[];let v=samples('embedding')[0].v;
 for(let i=0;i<demo.generationRounds;i++){rounds.push(sampleFor(v));v=nextInput(v,M.operations.pick([readout(v)]));}
 return rounds;
};
M.sourceValues=(state,v)=>state.levelId===id?{R:[...v],Z:[...O.sourceIds]}:sources(state,v);
M.reference=(v,levelId)=>levelId===id?M.operations.pick([readout(v)]):reference(v,levelId);
M.moduleShape=(kind,dim)=>dim===24?'3 × 8':moduleShape(kind,dim);
M.demo={id,sections,sourceNodes:[...sourceNodes],encoder:enc,masked,crossed,decoded,logits,prob,picked,nextInput,sampleFor,readout};
M.extraCard=(kind,levelId)=>{
 if(!definitions[kind])return card(kind,levelId);
 const d=definitions[kind],w=M.weights[kind];
 return {number:'',seal:d.name.slice(0,1),english:kind.slice(5).toUpperCase(),family:L('原版 Transformer · 基础运算','Original Transformer · basic operator'),color:'#'+M.extraColors[kind].toString(16),ability:d.name,meaning:d.hint,purpose:demo.rationale.text,
 formula:({'demo-up':'P = XW1','demo-down':'F = AW2','demo-relu':'A = max(0, P)','demo-add':'R = X + F(X)','demo-ln':'LN(x) = (x − μ) / √(variance(x) + ε)','demo-head':'L = DWvocab','demo-probabilities':'p_j = exp(L_j − max(L)) / Σ exp(L_k − max(L))'})[kind],weight:w?w.length+' × '+w[0].length:L('无','None'),trait:L('基础运算','Basic operation'),stage:0};
};
M.extraExample=(kind,levelId,nodeId)=>{
 if(levelId!==id)return example(kind,levelId,nodeId);
 const s=M.initial(id),v=M.samplesFor(id)[0].v,ev=M.evaluate(s,v),n=s.nodes.find(n=>n.id===nodeId)||s.nodes.find(n=>n.kind===kind);if(!n)return null;
 const sourceWords=O.sourceIds.map(i=>M.math.vocab[i]),targetWords=v.map(i=>M.math.vocab[i]),queryWords=sourceNodes.has(n.id)?sourceWords:targetWords,keyWords=sections[n.id]==='cross'?sourceWords:queryWords;
 return {sampleIndex:0,queryWords,keyWords,cross:sections[n.id]==='cross',masked:sections[n.id]==='masked',exampleSource:['D ID',v],inputs:n.links.map(from=>[from==='Z'?'S ID':from==='R'?'D ID':M.definitions[s.nodes.find(x=>x.id===from).kind].name,ev.values[from]]),output:[M.definitions[n.kind].name,ev.values[n.id]]};
};
L(' 模块，双击查看',' module, double-click to inspect');L('收起卡片，继续查看 ↗','Close card and continue exploring ↗');L('收起卡片，继续搭建 ↗','Close card and continue building ↗');
L('拖动旋转 · Shift / 右键拖动平移 · 滚轮缩放','Drag to orbit · Shift / right-drag to pan · scroll to zoom');L('平移视角','Pan view');L('跟随数据','Follow data');L('运行时跟随数据；拖动场景或滚轮缩放可接管视角','Follow data during a run; drag the scene or scroll to take manual control');L('切换平移模式，也可按住 Shift 或右键拖动','Toggle pan mode, or hold Shift or the right mouse button while dragging');
L('模型区域','Model regions');L('完整演示','Full demonstration');L('拖动转动视角 · 双击查看基础组件','Drag to orbit · double-click a basic module');
L('完整模型已运行完毕。','The complete model has finished running.');
L('跟随距离','Follow distance');L('远景','Wide');L('中景','Medium');L('近景','Close');
L('完整模型已连续运行三轮。','The complete model has run for three rounds.');L('新生成：','New tokens: ');L('正在生成','Generating');L('已生成','Generated');L('接回输入','Feed back');L('完整文字','Full text');L('目标输入 · 最近三个词','Target input · latest three tokens');
I18n.prepare({definitions,demo,group});
L('切换跟随','Switch flow');L('每点一次，跟随另一股数据；暂停时也可切换','Follow another data flow with each click, including while paused');
L('全屏','Fullscreen');L('退出全屏','Exit fullscreen');L('退出全屏（Esc）','Exit fullscreen (Esc)');L('全屏查看，保留运行与调速','Go fullscreen with playback controls');
})(window.NeuralPrologue);
