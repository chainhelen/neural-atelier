/* A curriculum-wide locator. This describes architecture, never the player's wiring. */
(function(M){
const translations=new Map();
const L=(zh,en)=>{translations.set(zh,en);return zh;};
const tr=text=>I18n.language()==='en'?(translations.get(text)||text):text;
function prepare(value,seen=new WeakSet()){
 if(!value||typeof value!=='object'||seen.has(value))return;seen.add(value);
 for(const key of Object.keys(value)){
  const item=value[key];
  if(typeof item==='string'&&translations.has(item))Object.defineProperty(value,key,{enumerable:true,configurable:true,get:()=>tr(item)});
  else if(item&&typeof item==='object')prepare(item,seen);
 }
}
// Only the static HTML labels need entries in the shared catalog.
for(const [zh,en]of [['本关提示','Level hint'],['关闭提示','Close hint']])globalThis.AtelierEnglish[zh]??=en;
const copy={
 title:L('本关提示','Level hint'),close:L('关闭提示','Close hint'),
 scope:L('本关所在位置','Lesson location'),exit:L('本关出口','Lesson outlet'),next:L('在完整结构中，接下来','Next in the complete structure'),
 focus:L('本关重点','Lesson focus'),reference:L('结构参考 · 不是棋盘接线答案','Architecture reference · not a wiring solution'),
 legend:L('步骤图中，着色的是本关练习的操作；描边的是本关出口。','In the step diagram, color marks the operations practiced here; an outline marks the lesson outlet.'),
 original:L('原版 Transformer · 编码器与解码器','Original Transformer · encoder and decoder'),
 modern:L('现代扩展 · 解码器结构','Modern extensions · decoder architecture'),
 cloud:L('云端部署 · 计算与数据搬运','Cloud deployment · computation and data movement'),
 engine:L('端侧推理 · 执行视图','On-device inference · execution view'),
 units:L('特征','features'),ids:L('词编号','token IDs'),
};
const node=(id,zh,en)=>({id,label:L(zh,en)});
const flows={
 connection:[node('input','输入向量','Input vector'),node('wire','传递到下一块','Route to a module'),node('result','仍是向量','Still a vector')],
 classicFFN:[node('up','线性展开','Expand features'),node('activation','ReLU','ReLU'),node('down','线性压回','Project back'),node('result','FFN 输出 F','FFN output F')],
 residual:[node('branch','支路计算 F(X)','Branch F(X)'),node('add','加回本段输入 X','Add this stage’s X'),node('norm','LayerNorm（原版）','LayerNorm (original)')],
 input:[node('embedding','词编号 → 词向量','Token IDs → embeddings'),node('position','加正弦位置编码','Add sinusoidal positions'),node('result','送入注意力层','Into attention layers')],
 attention:[node('qkv','Q / K / V 投影','Q / K / V projections'),node('score','QKᵀ 缩放打分','Scaled QKᵀ scores'),node('softmax','Softmax → 权重 A','Softmax → weights A'),node('av','A × V → 汇合特征','A × V → mixed features'),node('wo','多头时拼接，再投影 Wo','Concat if multi-head, then Wo')],
 causal:[node('qkv','Q / K / V 投影','Q / K / V projections'),node('score','QKᵀ 缩放打分','Scaled QKᵀ scores'),node('mask','遮住未来分数','Mask future scores'),node('softmax','Softmax → 权重 A','Softmax → weights A'),node('av','A × V → 汇合特征','A × V → mixed features'),node('wo','多头时拼接，再投影 Wo','Concat if multi-head, then Wo')],
 encoder:[node('attention','自注意力','Self-attention'),node('add1','加回注意力的输入','Add attention’s input'),node('norm1','LayerNorm','LayerNorm'),node('ffn','ReLU FFN','ReLU FFN'),node('add2','加回 FFN 的输入','Add the FFN’s input'),node('norm2','LayerNorm → 本层输出','LayerNorm → layer output')],
 decoder:[node('attention','带遮罩的自注意力','Masked self-attention'),node('add1','加回自注意力的输入','Add self-attention’s input'),node('norm1','LayerNorm','LayerNorm'),node('cross','交叉注意力：读取编码器 E','Cross-attention: read encoder E'),node('add2','加回交叉注意力的输入','Add cross-attention’s input'),node('norm2','LayerNorm','LayerNorm'),node('ffn','ReLU FFN','ReLU FFN'),node('add3','加回 FFN 的输入','Add the FFN’s input'),node('norm3','LayerNorm → 本层输出','LayerNorm → layer output')],
 cross:[node('encoded','源句 → 编码器 → E','Source → encoder → E'),node('prepared','上一关：目标前缀 → 因果自注意力 + 残差 + LayerNorm → D','Previous lesson: target prefix → masked attention + residual + LayerNorm → D'),node('q','目标特征 D → Q 投影 → Q','Target features D → Q projection → Q'),node('kv','编码器输出 E → K / V 投影 → K / V','Encoder output E → K / V projections → K / V'),node('score','QKᵀ 缩放打分','Scaled QKᵀ scores'),node('softmax','Softmax → 权重 A','Softmax → weights A'),node('av','A × V → 源句信息','A × V → source information'),node('wo','输出投影 Wo','Output projection Wo'),node('add','加回交叉注意力的输入','Add cross-attention’s input'),node('norm','LayerNorm','LayerNorm')],
 generation:[node('layers','编码器与解码器','Encoder and decoder'),node('logits','词表投影 → 候选分数','Vocabulary projection → logits'),node('select','选出词编号','Select a token ID'),node('append','追加到目标前缀，再计算','Append to target prefix; repeat')],
 autoregressive:[node('predict','模型打分（本关用查表代替）','Model scoring (replaced by lookup here)'),node('select','从分数中选词','Select tokens from scores'),node('append','只把最后一个预测追加到旧上下文','Append only the final prediction to the old context')],
 modernFFN:[node('up','内容投影 U = XW','Content projection U = XW'),node('gate','并行门投影 G = XWg','Parallel gate projection G = XWg'),node('silu','门值 SiLU(G)','Gate values SiLU(G)'),node('multiply','对应相乘 U ⊙ SiLU(G)','Pairwise U ⊙ SiLU(G)'),node('down','压回 → FFN 输出 F','Project back → FFN output F')],
 singleSiLU:[node('up','线性展开','Expand features'),node('silu','SiLU','SiLU'),node('down','压回 → FFN 输出 F','Project back → FFN output F')],
 modernAttention:[node('qkv','Q / K / V 投影','Q / K / V projections'),node('rope','旋转 Q/K：RoPE','Rotate Q/K: RoPE'),node('score','QKᵀ 缩放打分','Scaled QKᵀ scores'),node('mask','因果遮罩','Causal mask'),node('softmax','Softmax → 权重 A','Softmax → weights A'),node('av','A × V → 汇合特征','A × V → mixed features'),node('wo','输出投影 Wo','Output projection Wo')],
 modernLayer:[node('norm1','RMSNorm','RMSNorm'),node('attention','带 RoPE 的因果注意力','Causal attention with RoPE'),node('add1','加回这一段的输入','Add this stage’s input'),node('norm2','RMSNorm','RMSNorm'),node('ffn','SwiGLU FFN','SwiGLU FFN'),node('add2','加回 FFN 段的输入','Add the FFN stage’s input')],
 singleGPU:[node('input','CPU 中的输入','Input in CPU memory'),node('upload','传入 GPU 显存','Copy into GPU memory'),node('compute','投影 / 激活 / FFN','Projection / activation / FFN'),node('readback','结果读回 CPU','Read result back to CPU')],
 tensorParallel:[node('split','划分矩阵或特征','Partition weights or features'),node('compute','两张卡各算自己的部分','Each GPU computes its part'),node('merge','收集分量或相加部分和','Gather features or reduce sums'),node('result','得到完整特征','Complete feature vector')],
 engine:[node('memory','权重与 KV 的存储','Weight and KV storage'),node('prefill','Prefill：处理已有前缀','Prefill: process the prefix'),node('decode','Decode：逐次处理新位置','Decode: process each new position'),node('check','本关数值与性能验收','Numerical and performance checks'),node('next','后续层、词表与选词','Later layers, vocabulary, selection')],
};
// Each lesson has an explicit exit. Similar-shaped tensors can have different meanings.
const lessons={};
function locate(ids,family,regions,flow,focus,exit,where,output,next,note=''){
 for(const id of ids.split(' '))lessons[id]={family,regions,flow,focus,exit,where,output,next,note};
}
const classic=L('原版 · FFN 内部','Original · inside an FFN'),attn=L('原版 · 编码器的自注意力内部','Original · inside encoder self-attention');
const reuse=L('这是可复用的局部练习。原版编码器和解码器都使用 FFN；当前棋盘不是完整的一层。','This is a reusable local exercise. Both original encoders and decoders use FFNs; this board is not a complete layer.');
const valuesBypass=L('V 不经过打分或 Softmax，直接供 A × V 使用。单头无需拼接。','V bypasses scoring and Softmax and feeds A × V directly. A single head needs no concatenation.');
const residualNote=L('残差从每一段自己的输入绕过支路；图中的“加回输入”不是加回整模型最初的输入。','Each residual bypasses its own branch from that stage’s input, not the initial input of the entire model.');
locate('wire','original',['wire'],'connection',['wire'],'result',L('通用操作 · 向量传递','General operation · vector routing'),L('原样传出的向量','Unchanged vector'),L('可送入投影、注意力或 FFN。','Can feed a projection, attention or FFN.'),L('连线只把数据送到下一处，不改变数值和维度。它出现在整个网络中，不是 Transformer 中某个专属层。','A wire carries data to the next place without changing its values or dimensions. Wires appear throughout a network; they are not a dedicated Transformer layer.'));
locate('projection','original',['encoder','decoder'],'classicFFN',['up'],'up',classic,L('展开后的特征 P','Expanded features P'),L('在原版 FFN 中，接下来经过 ReLU。','In an original FFN, ReLU follows.'),reuse);
locate('relu','original',['encoder','decoder'],'classicFFN',['activation'],'activation',classic,L('ReLU 激活特征 A','ReLU activations A'),L('送入线性压回投影，得到 FFN 输出。','Feed the down projection to produce the FFN output.'),reuse);
locate('down-projection','original',['encoder','decoder'],'classicFFN',['down'],'down',classic,L('压回后的特征 F','Projected-back features F'),L('在完整 FFN 中，这个宽度可与本段原输入相加。','In a complete FFN, this width can be added to the stage’s input.'),L('本关只练压回投影，输入支路没有 ReLU；后续关再组装完整 FFN。','This lesson isolates the down projection without ReLU in its input branch; later lessons assemble the full FFN.'));
locate('original-ffn workshop','original',['encoder','decoder'],'classicFFN',['up','activation','down'],'result',classic,L('FFN 的变换 F','FFN transformation F'),L('加回 FFN 的输入，然后 LayerNorm（原版）。','Add the FFN’s input, then LayerNorm (original).'),reuse);
locate('residual','original',['encoder','decoder'],'residual',['branch','add'],'add',L('原版 · FFN 与外部残差','Original · FFN and its residual'),L('相加后的特征 X + FFN(X)','Features after X + FFN(X)'),L('原版结构中接下来是 LayerNorm，本关还没加入。','LayerNorm follows in the original architecture; it is not part of this lesson yet.'),L('支路就是前两关的原版 FFN，ReLU 仍在内部。相加在组件外面，把 FFN 的结果加回同一份输入。','The branch is the previous two lessons’ original FFN, with ReLU still inside. Addition sits outside the package and adds the FFN result to the same input.'));
locate('layernorm','original',['encoder','decoder'],'residual',['norm'],'norm',L('原版 · 残差相加后的 LayerNorm','Original · LayerNorm after residual addition'),L('归一化后的特征','Normalized features'),L('在完整网络中交给下一个子层或下一层。','In the full network, pass to the next sublayer or layer.'),L('地图标出它在完整网络中的位置。本关只练 LayerNorm，输入由游戏直接给定，未经过残差；组合关再接 X + FFN(X) → LayerNorm。','The map marks its place in the full network. This lesson isolates LayerNorm with game-supplied inputs that have not passed through a residual addition. The combined lesson later connects X + FFN(X) → LayerNorm.'));
locate('add-norm','original',['encoder','decoder'],'residual',['branch','add','norm'],'norm',L('原版 · FFN 子层与外部残差','Original · FFN sublayer and its residual'),L('FFN + 残差 + LayerNorm 的结果','Result of FFN + residual + LayerNorm'),L('送往下一层；Attention 子层在本关之外。','Pass to the next layer; attention is outside this lesson.'),residualNote);
locate('embedding','original',['source','target'],'input',['embedding'],'embedding',L('原版 · 模型输入准备','Original · input preparation'),L('查表得到的词向量 X','Looked-up embeddings X'),L('加入正弦位置编码，再进入网络层。','Add sinusoidal positions before entering the layers.'));
locate('positions wider-features','original',['source','target'],'input',['embedding','position'],'position',L('原版 · 模型输入准备','Original · input preparation'),L('带位置的词特征','Position-enriched token features'),L('用于注意力的 Q/K/V 投影。','Feed Q/K/V projections in attention.'),L('位置加在输入词向量上；这里不是旋转 Q/K 的 RoPE。','Positions are added to input embeddings; this is not RoPE rotation of Q/K.'));
locate('qkv','original',['encoder'],'attention',['qkv'],'qkv',attn,L('值向量矩阵 V = X × Wᵥ','Value matrix V = X × Wᵥ'),L('V 是三个词各自的计算结果，Wᵥ 才是权重。后续 Q/K 打分、Softmax 算出参考比例 A，再用 A × V 把这些值向量加权汇合。','V contains the computed vectors of three tokens; Wᵥ holds the weights. Later Q/K scoring and Softmax produce attention weights A, and A × V forms weighted sums of these value vectors.'),valuesBypass);
locate('scores','original',['encoder'],'attention',['qkv','score'],'score',attn,L('匹配分数矩阵 S = QKᵀ / √2','Matching-score matrix S = QKᵀ / √2'),L('每个格子是一对查询词 Q 与被比较词 K 的分数。下一关 Softmax 将同一查询的三个分数变成参考比例 A。','Each cell scores one query token Q against one key token K. Next, Softmax converts a query’s three scores into attention weights A.'),L('输入仍是上一关的 X，分别乘权重 Wq、Wk 得到 Q、K，再配对打分。V 不进入打分计算；之后 A × V 才用到它。','The input is still the previous lesson’s X. Multiply by Wq and Wk to obtain Q and K, then score every pair. V is used later in A × V, not in scoring.'));
locate('softmax pack-scores','original',['encoder'],'attention',['softmax'],'softmax',attn,L('注意力权重 A（中间结果）','Attention weights A (intermediate)'),L('与 V 做加权求和 A × V。','Weight and sum V using A × V.'),valuesBypass);
locate('pack-scores','original',['encoder'],'attention',['qkv','score','softmax'],'softmax',attn,L('注意力权重矩阵 A：读取各词 V 的比例','Attention-weight matrix A: proportions for reading V'),L('下一关将 A 与 V 相乘：每个查询按自己的三个比例，将三个 V 向量加权相加，得到新的内容向量。','Next, multiply A by V: each query uses its three proportions to form a weighted sum of the three V vectors, producing a new content vector.'),L('输入仍是 X，组合组件内部完成 Q/K 投影、打分和 Softmax。A 是由本次输入算出的比例，不是固定模型参数，也还不是汇合后的内容特征。','The input is still X. The package computes Q/K projections, scores and Softmax internally. A contains proportions computed from this input, not fixed model parameters or aggregated content features.'));
locate('weighted-sum','original',['encoder'],'attention',['av'],'av',attn,L('按比例汇合后的特征 O','Weighted-sum features O'),L('单头接输出投影 Wo；多头先拼接再投影。','Apply Wo for one head; concatenate before Wo for multiple heads.'),valuesBypass);
locate('self-attention multihead','original',['encoder'],'attention',['qkv','score','softmax','av','wo'],'wo',attn,L('Attention 输出（尚未加残差）','Attention output (before residual)'),L('加回 Attention 的输入，再做 LayerNorm。','Add attention’s input, then apply LayerNorm.'),valuesBypass);
locate('encoder-layer','original',['encoder','memory'],'encoder',['attention','add1','norm1','ffn','add2','norm2'],'norm2',L('原版 · 一层完整编码器','Original · one complete encoder layer'),L('本层输出 E','Encoder layer output E'),L('继续下一编码器层；最后一层的 E 供解码器交叉注意力读取。','Continue to the next encoder layer; the final E supplies decoder cross-attention.'),residualNote);
locate('causal','original',['decoder'],'causal',['mask'],'softmax',L('原版 · 解码器 → 带遮罩的自注意力','Original · decoder → masked self-attention'),L('注意力权重 A（中间结果）','Attention weights A (intermediate)'),L('与 V 汇合，再做输出投影；之后才是残差、归一化。','Mix V, then output projection; residual and normalization follow.'),valuesBypass);
locate('masked-multihead','original',['decoder'],'causal',['qkv','score','mask','softmax','av','wo'],'wo',L('原版 · 组装双头因果注意力','Original · assemble two-head causal attention'),L('双头因果注意力输出（尚未加残差）','Two-head causal attention output (before residual)'),L('收成组件；下一关加回输入，再做 LayerNorm。','Package as a module; add the input and apply LayerNorm in the next lesson.'),valuesBypass);
locate('decoder-self','original',['decoder'],'decoder',['attention','add1','norm1'],'norm1',L('原版 · 解码器的第一个子层','Original · first decoder sublayer'),L('自注意力 + 残差 + LayerNorm 的结果','Self-attention + residual + LayerNorm result'),L('送入交叉注意力，读取编码器的 E。','Feed cross-attention to read encoder E.'),residualNote);
locate('cross-attention','original',['memory','decoder'],'cross',['q','kv','score','softmax','av','wo'],'wo',L('原版 · 解码器 → 交叉注意力子层','Original · decoder → cross-attention sublayer'),L('交叉注意力输出（尚未加残差）','Cross-attention output (before residual)'),L('先加回交叉注意力的输入并做 LayerNorm，再进入 FFN。','Add cross-attention’s input and apply LayerNorm before entering the FFN.'),L('棋盘下方 D 是上一关已算好的目标特征，上方 E 是编码器输出。D / E 都不是现成的 Q / K / V，必须分别投影。源句完整可读；结果仍对应目标位置。','The lower board input D contains target features already computed by the previous lesson; the upper input E is the encoder output. Neither is a ready-made Q, K or V: they need separate projections. The whole source is readable; results still correspond to target positions.'));
locate('decoder-layer','original',['decoder'],'decoder',['attention','add1','norm1','cross','add2','norm2','ffn','add3','norm3'],'norm3',L('原版 · 一层完整解码器','Original · one complete decoder layer'),L('本层输出 D','Decoder layer output D'),L('送往下一解码器层；最后一层之后才接词表投影。','Feed the next decoder layer; vocabulary projection follows the final layer.'),residualNote);
locate('encoder-decoder','original',['source','encoder','memory','target','decoder','head'],'generation',['layers','logits'],'logits',L('原版 · 编码器、解码器与词表头','Original · encoder, decoder and vocabulary head'),L('词表分数 L（还没有选词）','Vocabulary logits L (no token selected yet)'),L('从词表分数选出下一个词，再追加到目标前缀。','Select the next token from logits, then append it to the target prefix.'));
locate('transformer-demo','original',['source','encoder','memory','target','decoder','head'],'generation',['layers','logits','select'],'select',L('完整演示 · 原版 Transformer 全部展开','Full demonstration · original Transformer unfolded'),L('各目标位置选出的词（最后一个预测下一词）','Selected tokens at target positions (the last predicts the next token)'),L('真实生成会把最后一个预测追加到目标前缀，再跑下一次。','Generation appends the final prediction to the target prefix before the next pass.'),L('棋盘上的每个头都展开为 Q/K/V、打分、Softmax、A × V；FFN 展开为投影、ReLU、降维。连接已就绪，只需运行一次。','Each board head expands into Q/K/V, scores, Softmax and A × V; each FFN expands into projection, ReLU and down projection. All wires are ready; just run.'));
locate('vocabulary-scores','original',['head'],'generation',['logits'],'logits',L('第 16 关解码器之后 · 学习词表投影','After level 16’s decoder · learn vocabulary projection'),L('每个候选词的分数 L','A score L for each candidate token'),L('第 18 关把这一步放回完整模型，第 19 关再从分数选词。','Place this step into the full model in level 18, then select tokens from scores in 19.'));
locate('vocabulary','original',['head'],'generation',['select'],'select',L('完整模型末端 · 选词','End of the model · token selection'),L('预测词的编号（本练习每个位置一个）','Predicted token IDs (one per position in this exercise)'),L('实际自回归生成通常取最后位置的新词，追加后继续。','Autoregressive generation normally appends the new token from the last position and continues.'));
locate('autoregressive','original',['head','target'],'autoregressive',['predict','select','append'],'append',L('模型外部 · 逐词生成循环','Outside the model · generation loop'),L('追加一个新词后的最近三词','Latest three tokens after appending one new token'),L('新上下文交给同一个预测器，开始下一次预测。','Feed the new context to the same predictor for the next prediction.'),L('本关用固定转移表代替模型打分；它与完整模型是替代关系，不是前后两层。','A fixed transition table replaces model scoring here; it is an alternative to the full model, not a layer after it.'));
const modern=L('现代扩展 · FFN 内部','Modern extension · inside an FFN');
locate('activation','modern',['layers'],'singleSiLU',['silu'],'silu',modern,L('SiLU 激活特征 S','SiLU activations S'),L('线性压回后才是 FFN 输出。','The down projection produces the FFN output.'),L('这是单支路激活练习，还没有 SwiGLU 的门分支。','This is a single-branch activation exercise, without a SwiGLU gate branch.'));
locate('plain-ffn','modern',['layers'],'singleSiLU',['up','silu','down'],'down',modern,L('单支路 SiLU FFN 的输出 F','Single-branch SiLU FFN output F'),L('可作为残差支路的变换。','Can serve as the transformation in a residual branch.'),L('这是一种单支路变体；后续 SwiGLU 才加入门分支。','This is a single-branch variant; the later SwiGLU introduces a gate branch.'));
locate('gate-projection','modern',['layers'],'modernFFN',['gate'],'gate',modern,L('门投影结果 G（还未激活）','Gate projection G (not yet activated)'),L('先算 SiLU(G)，再调节内容特征 U。','Apply SiLU(G), then modulate the content features U.'));
locate('pairwise gating','modern',['layers'],'modernFFN',['multiply'],'multiply',modern,L('对应相乘后的特征 H','Elementwise product features H'),L('经过压回投影得到 FFN 输出。','Pass through the down projection for the FFN output.'),L('内容 U 与门系数配对相乘；不是跨词汇合，也不是残差相加。','Content U is multiplied by gate coefficients, not mixed across tokens or added as a residual.'));
locate('swiglu','modern',['layers'],'modernFFN',['up','gate','silu','multiply','down'],'down',modern,L('SwiGLU FFN 的输出 F','SwiGLU FFN output F'),L('加回这一段的原输入。','Add this stage’s original input.'));
locate('normalize','modern',['layers'],'modernLayer',['norm1'],'norm1',L('现代扩展 · 子层计算前的归一化','Modern extension · normalization before a sublayer'),L('RMSNorm 后的特征 Z','Features Z after RMSNorm'),L('供下一段 Attention 或 FFN 使用。','Feed the following attention or FFN.'),L('RMSNorm 可用在两个子层前；这里借注意力入口定位。原版则把 LayerNorm 放在残差相加后。','RMSNorm can precede either sublayer; the attention entrance is shown here. The original places LayerNorm after residual addition.'));
locate('ffn','modern',['layers'],'modernLayer',['ffn','add2'],'add2',L('现代扩展 · FFN 半层','Modern extension · the FFN half of a layer'),L('SwiGLU + 残差后的特征','Features after SwiGLU + residual'),L('送往后续层；本关没有 Attention 半层。','Feed later layers; the attention half is absent here.'),L('Z 已在本关之外经过 RMSNorm；棋盘计算 SwiGLU(Z)，再加回未归一化的 R。','Z is already RMS-normalized outside this lesson; the board computes SwiGLU(Z), then adds the unnormalized R.'));
locate('rotation','modern',['layers'],'modernAttention',['rope'],'rope',L('现代扩展 · 注意力的位置处理','Modern extension · positions inside attention'),L('旋转后的 Q（本关只练 Q）','Rotated Q (only Q is exercised here)'),L('完整 RoPE 还要旋转 K，再算 Q/K 分数。','Full RoPE also rotates K, then computes Q/K scores.'),valuesBypass);
locate('rope','modern',['layers'],'modernAttention',['rope','score'],'score',L('现代扩展 · 注意力的位置处理','Modern extension · positions inside attention'),L('带相对位置信息的分数 S','Scores S with relative-position information'),L('因果遮罩后送入 Softmax。','Apply the causal mask, then Softmax.'),valuesBypass);
locate('aggregation','modern',['layers'],'modernAttention',['av'],'av',L('现代扩展 · 注意力内部','Modern extension · inside attention'),L('按比例汇合的特征 O','Weighted-sum features O'),L('接输出投影 Wo，再进入残差。','Apply output projection Wo, then the residual.'),valuesBypass);
locate('attention','modern',['layers'],'modernAttention',['qkv','rope','score','mask','softmax','av','wo'],'wo',L('现代扩展 · 完整注意力支路','Modern extension · complete attention branch'),L('Attention 输出（尚未加残差）','Attention output (before residual)'),L('加回本段输入，然后进入 FFN 半层。','Add the stage’s input, then enter the FFN half.'),valuesBypass);
locate('block','modern',['layers'],'modernLayer',['norm1','attention','add1','norm2','ffn','add2'],'add2',L('现代扩展 · 一层完整解码器块','Modern extension · one complete decoder block'),L('本层输出 Y','Layer output Y'),L('继续后续层；模型末端还有最终归一化和词表投影。','Continue through later layers; the model ends with final normalization and vocabulary projection.'),residualNote);
const cloud=L('云端 · 线性层 / FFN 的执行方式','Cloud · executing linear layers / FFNs');
locate('single-upload','cloud',['cpu','gpu','result'],'singleGPU',['upload','compute','readback'],'readback',cloud,L('读回 CPU 的投影特征','Projected features read back to CPU'),L('真实网络中可供激活或其他算子继续计算。','Can feed an activation or another operator in a real network.'),L('这里练设备间传递与线性投影，不是完整 Transformer 推理。','This exercises device transfers and a linear projection, not complete Transformer inference.'));
locate('single-fallback single-resident','cloud',['cpu','gpu','result'],'singleGPU',['compute'],'readback',cloud,L('读回 CPU 的 ReLU FFN 特征','ReLU FFN features read back to CPU'),L('在网络结构中可接残差；本关另验收内存位置与搬运量。','Can feed a residual in the network; this lesson also checks memory locations and copy volume.'),L('比较中途回 CPU 与留在 GPU 的执行路径，FFN 的数学目标不变。','Compare CPU fallback with GPU residency; the FFN’s mathematical target stays the same.'));
locate('tp-columns','cloud',['gpu','gpu2','result'],'tensorParallel',['split','compute','merge'],'result',cloud,L('拼齐的投影特征（不同分量）','Gathered projected features (distinct components)'),L('可交给激活；不是将两份部分和相加。','Can feed an activation; these are not partial sums to add.'));
locate('tp-rows','cloud',['gpu','gpu2','result'],'tensorParallel',['split','compute','merge'],'result',cloud,L('相加部分和后的完整特征','Complete features after summing partials'),L('可供后续算子使用；Reduce 不是残差。','Can feed subsequent operators; Reduce is not a residual.'));
locate('tp-ffn','cloud',['gpu','gpu2','result'],'tensorParallel',['split','compute','merge'],'result',cloud,L('两卡合算的 ReLU FFN 输出','ReLU FFN output computed across two GPUs'),L('在完整层中接入残差。','Feed a residual in a complete layer.'),L('拆的是 FFN 的计算与存储，不是在网络里增加一种新层。','This partitions FFN computation and storage; it does not add a new kind of network layer.'));
const engineNote=L('这是教学执行流水线与成本模型，不是某款真实模型的完整计算图；当前数值出口仍为特征，未做词表选词。','This is a teaching execution pipeline and cost model, not a real model’s complete graph. Its numerical outlet still contains features; vocabulary selection is not performed.');
for(const [ids,focus,where]of [
 ['placement layout parallel',['prefill'],L('端侧 · 算子放在哪里执行','On-device · where operators execute')],
 ['batching',['prefill'],L('端侧 · Prefill 的批处理','On-device · batched prefill')],
 ['kv-cache',['decode'],L('端侧 · Decode 复用历史 K/V','On-device · decoding with cached K/V')],
 ['quantization',['memory'],L('端侧 · 权重与 KV 内存','On-device · weight and KV memory')],
 ['fusion',['prefill','decode'],L('端侧 · 合并执行与减少搬运','On-device · fused execution and fewer transfers')],
 ['session',['memory','prefill'],L('端侧 · 多轮前缀复用','On-device · reuse across turns')],
 ['capstone',['memory','prefill','decode'],L('端侧 · 综合执行验收','On-device · complete execution challenge')]
])locate(ids,'engine',['cpu','gpu','ane'],'engine',focus,'check',where,L('教学流水线输出特征','Teaching pipeline output features'),L('真实模型还需要后续层及词表投影，再选出新词。','A real model still needs later layers and vocabulary projection before selecting a token.'),engineNote);
// Pairwise multiplication is deliberately taught before SiLU is added.
flows.pairwise=flows.modernFFN.filter(n=>n.id!=='silu').map(n=>n.id==='multiply'?node('multiply','对应相乘 U ⊙ G','Pairwise U ⊙ G'):n);
lessons.pairwise={...lessons.pairwise,flow:'pairwise',note:L('这里只练 U ⊙ G，门值尚未经过 SiLU。地图展示它在门控 FFN 中的位置；下一关再补激活。','This isolates U ⊙ G without SiLU on the gate. The map locates it in a gated FFN; the next lesson adds the activation.')};
for(const id of ['quantization','session','kv-cache'])lessons[id].regions=['memory','gpu'];
for(const data of [copy,flows,lessons])prepare(data);
function get(id){const entry=lessons[id];if(!entry)throw new Error('Missing structure map for '+id);const l=M.levels[id];return {...entry,output:l.outputTitle+' · '+l.outputNote,steps:flows[entry.flow],shape:l.tokenOutput?l.sinkDim+' '+copy.ids:(l.sequence?'3 × '+l.sinkDim/3:'1 × '+l.sinkDim)};}
const esc=s=>String(tr(s)).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
function drilldown(title,steps,note=''){
 return `<details class="architecture-drilldown"><summary>${esc(title)}</summary><ol>${steps.map(text=>`<li>${esc(text)}</li>`).join('')}</ol>${note?`<p>${esc(note)}</p>`:''}</details>`;
}
function attentionInside({cross=false,causal=false,rope=false}={}){
 return drilldown(L('展开 Attention 内部','Expand Attention'),[
  cross?'Q = DWq; K = EWk; V = EWv':'Q = XWq; K = XWk; V = XWv',
  ...(rope?[L('RoPE：按位置旋转 Q 和 K，V 不旋转','RoPE: rotate Q and K by position; leave V unchanged')]:[]),
  L('S = QKᵀ / √dₖ；dₖ 是每个头的 Q/K 特征数','S = QKᵀ / √dₖ; dₖ is the Q/K feature count per head'),
  ...(causal?[L('因果遮罩：未来位置的分数置为 −∞','Causal mask: set future-position scores to −∞')]:[]),
  'A = Softmax(S)',
  L('A × V：按 A 的比例汇合各位置的 V','A × V: sum V across positions using the proportions in A'),
  L('多个头先拼接，再乘输出投影 Wo','Concatenate multiple heads, then apply output projection Wo')
 ],L('Wq / Wk / Wv / Wo 是模型参数；A 是根据本次输入算出的读取比例，每次输入都可能不同。','Wq / Wk / Wv / Wo are model parameters; A contains reading proportions computed from this input and can change with every input.'));
}
function ffnInside(kind='relu'){
 const gated=kind==='gate'||kind==='swiglu';
 return drilldown(L('展开 FFN 内部','Expand FFN'),gated?[
  L('同一份输入 Z 分成并行两路：U = ZWup，G = ZWgate','Split the same Z into parallel branches: U = ZWup, G = ZWgate'),
  kind==='swiglu'?'S = SiLU(G); H = U ⊙ S':'H = U ⊙ G',
  'F = HWdown'
 ]:['H = XWup + bup',kind==='silu'?'A = SiLU(H)':'A = ReLU(H)','F = AWdown + bdown'],L('每个位置独立使用同一套权重；FFN 加工特征，不在词之间汇合信息。残差在 FFN 外面相加。','Each position independently uses the same weights. The FFN transforms features without mixing tokens; residual addition is outside the FFN.'));
}
function normInside(rms=false,title=null){
 return drilldown(title||(rms?L('展开 RMSNorm','Expand RMSNorm'):L('展开残差与 LayerNorm','Expand residual and LayerNorm')),rms?[
  'r = √(mean(x²) + ε)','z = (x / r) ⊙ γ'
 ]:[L('先相加：r = 本段输入 + 本段计算结果','First add: r = stage input + stage result'),'μ = mean(r); σ² = mean((r − μ)²)','z = ((r − μ) / √(σ² + ε)) ⊙ γ + β'],L('均值与尺度都在当前词的特征维度上计算；γ、β 是模型参数，RMSNorm 只使用 γ。','Means and scales are computed over one token’s feature dimensions. γ and β are model parameters; RMSNorm uses only γ.'));
}
// Keep model hierarchy separate from the operator-level route below it.
function originalArchitecture(entry){
 const e=esc,focus=new Set(),all=['encoder-attention','encoder-ffn','decoder-self','decoder-cross','decoder-ffn'];
 if(entry.flow==='classicFFN')focus.add('encoder-ffn').add('decoder-ffn');
 if(entry.flow==='attention')focus.add('encoder-attention');
 if(entry.flow==='causal')focus.add('decoder-self');
 if(entry.flow==='cross')focus.add('decoder-cross');
 if(entry.flow==='encoder')focus.add('encoder-attention').add('encoder-ffn');
 if(entry.flow==='decoder')for(const [name,steps]of [['decoder-self',['attention','add1','norm1']],['decoder-cross',['cross','add2','norm2']],['decoder-ffn',['ffn','add3','norm3']]])if(steps.some(id=>entry.focus.includes(id)))focus.add(name);
 if(entry.flow==='input')focus.add('encoder-input').add('decoder-input');
 if(entry.flow==='generation'){
  if(entry.focus.includes('layers'))for(const key of [...all,'encoder-input','decoder-input'])focus.add(key);
  if(entry.focus.includes('logits')||entry.focus.includes('select'))focus.add('readout');
 }
 const legend=entry.flow==='connection'?L('仅作完整模型参考；连线没有专属的层位置。','Full-model reference only; wiring has no dedicated layer location.'):
  entry.flow==='residual'?L('着色处展示这类操作的常见使用位置；本关只练下方列出的步骤。','Color shows common sites for this operation; this lesson practices only the steps listed below.'):
  entry.referenceOnly?L('着色部分是本关查表所替代的打分过程。','Color shows the scoring process replaced by the lookup in this lesson.'):
  L('着色框标出相关模块；本关具体练习哪些操作，见下方步骤。','Colored blocks identify related modules; the steps below show exactly what this lesson practices.');
 const active=key=>focus.has(key)?' architecture-focus':'';
 const addNorm=L('加回本段输入 → LayerNorm','Add this stage’s input → LayerNorm');
 const stage=(key,label,role,extra='')=>`<div class="architecture-stage${active(key)}" data-architecture-node="${key}">${extra}<b>${e(label)}</b><span>${e(role)}</span>${key.endsWith('ffn')?ffnInside():attentionInside({cross:key==='decoder-cross',causal:key==='decoder-self'})}${key==='decoder-cross'&&entry.flow==='cross'?`<span class="architecture-boundary">${e(L('本关在 Wo 后结束；下面的残差与归一化在第 16 关接上。','This lesson ends after Wo; level 16 adds the residual and normalization below.'))}</span>`:''}<div class="architecture-addnorm${entry.flow==='residual'?' architecture-focus':''}">${normInside(false,addNorm)}</div></div>`;
 const input=(key,label,example)=>`<div class="architecture-input" data-text-input="${key}"><b>${e(label)}</b><span>${e(example)}</span></div><div class="architecture-prep${active(key+'-input')}" data-architecture-node="${key}-input">${e(L('词向量 + 正弦位置编码','Embeddings + sinusoidal positions'))}</div>`;
 return `<section class="architecture-detail" aria-label="${e(L('展开编码器与解码器','Inside the encoder and decoder'))}">
 <div class="architecture-heading"><h4>${e(L('Attention 与 FFN 都在层里面','Attention and FFN live inside each layer'))}</h4><span>${e(legend)}</span></div>
 <div class="architecture-columns">
 <section class="architecture-lane" data-architecture-lane="encoder">
 ${input('encoder',L('源句：用户要翻译的输入','Source: the user’s text to translate'),L('例如：我爱猫','Example: “J’aime les chats”'))}
 <details class="architecture-layer" open><summary>${e(L('编码器 · 每一层','Encoder · each layer'))}</summary>
 ${stage('encoder-attention',L('自注意力 · Self-Attention','Self-Attention'),L('让源句中的词互相读取信息','Source tokens read one another'))}
 ${stage('encoder-ffn','FFN',L('线性展开 → ReLU → 线性压回','Linear up → ReLU → linear down'))}
 </details><div class="architecture-memory"><b>${e(L('最后一层的输出 E','Final encoder output E'))}</b><span>${e(L('已经包含编码器的 Attention 和 FFN 计算','Includes the encoder’s Attention and FFN computations'))}</span></div>
 </section>
 <section class="architecture-lane" data-architecture-lane="decoder">
 ${input('decoder',L('目标前缀：已经生成的那部分译文','Target prefix: the translation generated so far'),L('例如：I love（第一次只有起始标记）','Example: “I love” (only a start marker at the first step)'))}
 <details class="architecture-layer" open><summary>${e(L('解码器 · 每一层','Decoder · each layer'))}</summary>
 ${stage('decoder-self',L('因果自注意力 · Masked Self-Attention','Masked Self-Attention'),L('① 整理目标前文；第 14 关在 LayerNorm 后得到 D','① Process the target prefix; level 14 produces D after LayerNorm'))}
 ${stage('decoder-cross',L('交叉注意力 · Cross-Attention','Cross-Attention'),L('② 用 D 查询源句，取回相关信息（第 15 关）','② Use D to retrieve relevant source information (level 15)'),`<div class="architecture-memory-link" data-architecture-edge="E-to-cross">${e(L('来自编码器的 E 提供 K / V；D 提供 Q','Encoder E supplies K / V; D supplies Q'))}</div>`)}
 ${stage('decoder-ffn','FFN',L('③ 逐个加工融合后的目标特征（第 16 关接入）','③ Transform each combined target feature (added in level 16)'))}
 </details><div class="architecture-readout${active('readout')}" data-architecture-node="readout"><b>${e(L('最后一层后：词表投影 → 选下一个词','After the final layer: vocabulary projection → next token'))}</b><span>${e(L('例如选出 cats，追加为 I love cats，再继续预测','For example, select “cats”, append it to “I love”, then predict again'))}</span>${readoutInside()}</div>
 </section></div>
 <p class="architecture-depth">${e(L('上图展开的是每一层；原版编码器与解码器分别堆叠多层。本游戏完整层用一层示范，E 送入各解码器层的交叉注意力。第 16 关会组装右侧完整的一层。','The diagram expands one layer; the original encoder and decoder each stack multiple layers. This game demonstrates one complete layer. E feeds cross-attention in every decoder layer; level 16 assembles the full layer on the right.'))}</p>
 <div class="architecture-weights" data-model-weights><b>${e(L('模型权重在哪里？','Where are the model weights?'))}</b><p>${e(L('词向量表、各处的 Wq / Wk / Wv / Wo、FFN 矩阵和 LayerNorm 参数都是模型权重，加载在组件内部；源句、目标前缀以及中间的 D / E 是流经组件的数据。推理时权重固定，文本和中间特征随请求与生成步骤变化。','Embedding tables, Wq / Wk / Wv / Wo, FFN matrices and LayerNorm parameters are model weights loaded inside the modules. The source, target prefix and intermediate D / E are data flowing through them. Weights stay fixed during inference; text and features change with each request and generation step.'))}</p><small>${e(L('翻译句子只是含义示例；游戏的手写小矩阵没有训练成翻译模型。千问等 decoder-only 模型使用“用户输入 + 已生成文本”这一条序列，没有这里单独的编码器与交叉注意力支路。','The translation is an illustration; the game’s hand-written matrices are not a trained translator. Decoder-only models such as Qwen use one sequence of prompt plus generated text, without this separate encoder and cross-attention branch.'))}</small></div>
 </section>`;
}
function readoutInside(){return drilldown(L('展开词表与生成循环','Expand vocabulary and generation'),[
 L('最后位置的特征 × 词表权重 → 每个候选 token 的分数','Last-position features × vocabulary weights → a score for each candidate token'),
 L('选下一个 token；按概率采样时先经过 Softmax','Select the next token; probability sampling first applies Softmax'),
 L('把新 token 追加到已有文本，再运行同一个模型','Append the token to the existing text and run the same model again')]);}
function modernArchitecture(entry){
 const e=esc,layer=entry.flow==='modernLayer',ffn=['modernFFN','singleSiLU','pairwise'].includes(entry.flow),attn=entry.flow==='modernAttention';
 const block=(key,label,role,inner,active)=>`<div class="architecture-stage${active?' architecture-focus':''}" data-architecture-node="${key}"><b>${e(label)}</b><span>${e(role)}</span>${inner}</div>`;
 const kind=entry.flow==='singleSiLU'?'silu':entry.flow==='pairwise'?'gate':'swiglu';
 return `<section class="architecture-detail"><div class="architecture-heading"><h4>${e(L('现代解码器：同一条文本序列，反复经过多层','Modern decoder: one text sequence through multiple layers'))}</h4></div><div class="architecture-input"><b>${e(L('用户输入 + 已生成的文本 → 词向量','User prompt + generated text → embeddings'))}</b><span>${e(L('模型权重存放在各组件内；新生成的词会追加回这条序列。','Model weights live inside the modules; new tokens are appended to this sequence.'))}</span></div>
 <details class="architecture-layer modern-layer" open><summary>${e(L('展开一层：Attention → FFN，两段各有残差','One layer: Attention → FFN, each with its own residual'))}</summary>
 ${block('modern-norm1','RMSNorm',L('先归一化，再进入 Attention','Normalize before Attention'),normInside(true),layer&&entry.focus.includes('norm1'))}
 ${block('modern-attention',L('因果自注意力 · Attention','Causal Self-Attention'),L('Q/K 加入 RoPE；不读取未来位置','Apply RoPE to Q/K; never read future positions'),attentionInside({causal:true,rope:true}),attn||layer&&entry.focus.includes('attention'))}
 ${block('modern-add1',L('加回 Attention 的输入','Add the Attention input'),'R₁ = X + Attention(RMSNorm(X))','',layer&&entry.focus.includes('add1'))}
 ${block('modern-norm2','RMSNorm',L('先归一化，再进入 FFN','Normalize before the FFN'),normInside(true),layer&&entry.focus.includes('norm2'))}
 ${block('modern-ffn',kind==='silu'?'SiLU FFN':kind==='gate'?L('门控 FFN · 尚未加 SiLU','Gated FFN · before adding SiLU'):'SwiGLU FFN',L('逐个加工目标词的特征','Transform each target token’s features'),ffnInside(kind),ffn||layer&&entry.focus.includes('ffn'))}
 ${block('modern-add2',L('加回 FFN 的输入','Add the FFN input'),'Y = R₁ + FFN(RMSNorm(R₁))','',layer&&entry.focus.includes('add2'))}
 </details><div class="architecture-readout"><b>${e(L('后续层 → 最终 RMSNorm → 词表投影 → 选词','Later layers → final RMSNorm → vocabulary projection → token selection'))}</b>${readoutInside()}</div><p class="architecture-depth">${e(L('这是现代扩展章节的参考结构。单支路 SiLU、裸门值相乘等早期练习会标出尚未补齐的部分；并非每关都已经是完整 SwiGLU 或完整解码器块。','This is the reference for the modern-extension lessons. Earlier exercises show their single SiLU branch or raw gate product explicitly; not every lesson is already a complete SwiGLU or decoder block.'))}</p></section>`;
}
function executionArchitecture(entry,id){
 const e=esc,box=(key,label,inner,active=true)=>`<div class="architecture-stage${active?' architecture-focus':''}" data-architecture-node="${key}"><b>${e(label)}</b>${inner}</div>`;
 let title,body;
 if(entry.family==='cloud'){
  title=L('同一段矩阵计算，放到哪些设备执行？','Where does the same matrix computation execute?');
  if(entry.flow==='singleGPU'){
   const steps=id==='single-upload'?['H = XW']:id==='single-fallback'?['H = XW',L('H 取回 CPU → ReLU → 再上传 GPU','Read H back to CPU → ReLU → upload to GPU'),'Y = ReLU(H)D']:['H = XW',L('ReLU 留在 GPU，特征不往返 CPU','Keep ReLU on the GPU; no intermediate CPU round trip'),'Y = ReLU(H)D'];
   body=`<div class="execution-input">${e(L('CPU 输入 → 上传 GPU','CPU input → upload to GPU'))}</div>${box('single-gpu',id==='single-fallback'?L('GPU 投影 → CPU 激活 → GPU 投影','GPU projection → CPU activation → GPU projection'):'GPU',drilldown(L('展开本关的设备路径','Expand this lesson’s device path'),steps,L('W 和 D 是已驻留在 GPU 的权重。搬运特征不改变数值，改变的是它在哪块内存里。','W and D are weights already resident on the GPU. Copying features changes their memory location, not their values.')))}<div class="execution-input">${e(L('取回 CPU → 本关输出特征','Read back to CPU → lesson output features'))}</div>`;
  }else{
   const columns=id==='tp-columns',rows=id==='tp-rows';
   const initial=rows?L('先算 H = XW，再分成前半 H₀ / 后半 H₁','Compute H = XW, then split it into H₀ / H₁'):L('两张 GPU 都读取完整输入 X','Both GPUs read the full input X');
   body=`<div class="execution-input">${e(initial)}</div><div class="execution-parallel">${[0,1].map(i=>box('gpu'+i,'GPU '+i,drilldown(L('展开这张卡的计算','Expand this GPU’s computation'),columns?[`Y${i} = XW${i}`,L('只持有 W 的一部分列 → 得到一部分输出特征','Store some columns of W → produce some output features')]:rows?[`P${i} = H${i}D${i}`,L('只持有 D 的对应行 → 得到每个输出坐标的部分和','Store matching rows of D → produce a partial sum for each output coordinate')]:[`H${i} = XW${i}`,`A${i} = ReLU(H${i})`,`P${i} = A${i}D${i}`]))).join('')}</div>${box('merge',columns?'Gather · concat(Y₀, Y₁)':'Reduce · P₀ + P₁',drilldown(L('展开通信的含义','Expand the communication step'),[columns?L('两卡给出不同输出坐标，按顺序拼接','The GPUs produce different output coordinates; concatenate them in order'):L('两卡给出同一坐标的不同贡献，逐项相加','The GPUs contribute to the same output coordinates; add their partial values')],L('本关在出口收集一份结果；不是把结果再分发到每张卡的 All-Gather / All-Reduce。','This lesson collects one result at the outlet; it does not redistribute the result to every GPU as All-Gather / All-Reduce would.')))}`;
  }
  body+=`<p class="architecture-depth">${e(L('这些关卡只展开线性层或 ReLU FFN 的执行。完整 Transformer 的 Attention、外部残差和词表头不在本关棋盘里；设备数改变了分工，没有增加一种神经网络层。','These lessons expand execution of a linear layer or ReLU FFN. Attention, outer residuals and vocabulary heads are outside this board; more devices change the division of work, not the neural-network layers.'))}</p>`;
 }else{
  title=L('端侧：同一模型的两种运行阶段','On-device: two execution phases of the same model');
  body=`${box('engine-memory',L('模型权重 ≠ KV 缓存','Model weights ≠ KV cache'),drilldown(L('展开两种存储','Expand both kinds of storage'),[L('权重：训练得到的矩阵，推理时固定，供各次计算读取','Weights: trained matrices, fixed during inference and read by repeated computations'),L('KV 缓存：这次文本在各层算出的 K / V，随前缀增长','KV cache: K / V computed for this text at each layer, growing with the prefix')]),entry.focus.includes('memory'))}
  <div class="execution-parallel">${box('engine-prefill','Prefill',drilldown(L('展开提示词处理','Expand prompt processing'),[L('一起处理用户已给出的多个位置','Process the multiple positions already provided by the user'),L('每层算 Q/K/V → 因果 Attention → FFN','At each layer: Q/K/V → causal Attention → FFN'),L('保留这些位置的 K/V，为生成准备缓存','Keep these positions’ K/V to prepare the generation cache')]),entry.focus.includes('prefill'))}
  ${box('engine-decode','Decode',drilldown(L('展开逐词生成','Expand token-by-token generation'),[L('每次处理一个新位置，仍使用同一套层和权重','Process one new position at a time using the same layers and weights'),L('新 Q 读取历史和当前 K/V；当前 K/V 追加到缓存','New Q reads past and current K/V; append current K/V to the cache'),L('经过 FFN 和后续层，词表头预测下一词','Pass through the FFN and later layers; the vocabulary head predicts the next token')]),entry.focus.includes('decode'))}</div>
  ${box('engine-device','CPU / GPU / ANE',drilldown(L('展开设备分工与交接','Expand device placement and handoffs'),[L('CPU 负责调度；投影、Attention、FFN 按支持能力安排执行','The CPU schedules work; projections, Attention and FFN run on supported devices'),L('张量布局、同步与中间结果交接都有成本','Tensor layouts, synchronization and intermediate handoffs have costs'),L('本游戏只允许投影和 FFN 分担到 ANE；不是所有算子都能放上去','This game allows projections and FFN work on ANE; not every operator can be assigned there')]),!entry.focus.includes('memory'))}
  <p class="architecture-depth">${e(L('上面解释真实推理阶段的含义；本关棋盘与成本数字是教学模型，数值出口仍是特征，未运行完整模型的词表生成。GPU 与 ANE 分工也不是某款硬件的实测保证。','The diagram explains real inference phases. This board and its costs are a teaching model: the numerical outlet still contains features, without full-model vocabulary generation. GPU/ANE placement is not a hardware benchmark guarantee.'))}</p>`;
 }
 return `<section class="architecture-detail execution-detail"><div class="architecture-heading"><h4>${e(title)}</h4></div>${body}</section>`;
}
function diagram(entry){
 const n=(id,x,y,zh,en)=>({id,x,y,label:L(zh,en)});
 let nodes,paths,height=122;
 if(entry.flow==='connection'){
  height=52;nodes=[n('input',3,12,'输入向量','Input'),n('wire',81,12,'连线传递','Wire'),n('result',159,12,'输出向量','Output')];
  paths=['M61 26H79m-4 -3l4 3 -4 3','M139 26H157m-4 -3l4 3 -4 3'];
 }else if(entry.family==='original'){
  nodes=[n('source',3,4,'源句输入','Source'),n('encoder',81,4,'编码器','Encoder'),n('memory',159,4,'源表征 E','Memory E'),n('target',3,78,'目标前缀','Target'),n('decoder',81,78,'解码器','Decoder'),n('head',159,78,'词表/选词','Readout')];
  paths=['M61 18H79m-4 -3l4 3 -4 3','M139 18H157m-4 -3l4 3 -4 3','M188 32V51H110V76m-3 -4l3 4 3 -4','M61 92H79m-4 -3l4 3 -4 3','M139 92H157m-4 -3l4 3 -4 3','M188 106V116H32V108m-3 4l3 -4 3 4'];
 }else if(entry.family==='modern'){
  nodes=[n('input',3,4,'词向量','Embedding'),n('layers',81,4,'解码器层','Layers'),n('head',159,4,'Norm/词表','Norm/head'),n('selection',159,78,'选词','Select'),n('prefix',3,78,'更新前缀','New prefix')];
  paths=['M61 18H79m-4 -3l4 3 -4 3','M139 18H157m-4 -3l4 3 -4 3','M188 32V76m-3 -4l3 4 3 -4','M159 92H63m4 -3l-4 3 4 3','M32 78V34m-3 4l3 -4 3 4'];
 }else if(entry.family==='cloud'&&entry.flow==='singleGPU'){
  height=52;nodes=[n('cpu',3,12,'CPU 内存','CPU RAM'),n('gpu',81,12,'单张 GPU','One GPU'),n('result',159,12,'CPU 结果','CPU result')];
  paths=['M61 26H79m-4 -3l4 3 -4 3','M139 26H157m-4 -3l4 3 -4 3'];
 }else if(entry.family==='cloud'){
  nodes=[n('cpu',3,41,'CPU 内存','CPU RAM'),n('gpu',81,4,'GPU 0','GPU 0'),n('gpu2',81,78,'GPU 1','GPU 1'),n('result',159,41,'完整结果','Full result')];
  paths=['M61 55H70V18H79m-4 -3l4 3 -4 3','M70 55V92H79m-4 -3l4 3 -4 3','M139 18H148V55H157m-4 -3l4 3 -4 3','M139 92H148V55'];
 }else{
  nodes=[n('memory',81,4,'权重 / KV','Weights/KV'),n('cpu',3,67,'CPU','CPU'),n('gpu',81,67,'GPU','GPU'),n('ane',159,67,'ANE','ANE')];
  paths=['M110 32V45H32V65m-3 -4l3 4 3 -4','M110 45V65m-3 -4l3 4 3 -4','M110 45H188V65m-3 -4l3 4 3 -4'];
 }
 return `<svg class="structure-overview" viewBox="0 0 220 ${height}" role="img" aria-label="${esc(copy[entry.family]+' · '+entry.where)}"><g class="structure-edges">${paths.map(d=>`<path d="${d}"/>`).join('')}</g>${nodes.map(n=>{const inside=entry.family==='original'&&['encoder','decoder'].includes(n.id)||entry.family==='modern'&&n.id==='layers';return `<g data-map-region="${n.id}" class="${entry.regions.includes(n.id)?'map-region-active':''}"><rect x="${n.x}" y="${n.y}" width="58" height="28" rx="6"/><text x="${n.x+29}" y="${n.y+(inside?11:18)}" text-anchor="middle">${esc(n.label)}</text>${inside?`<text class="structure-mini-inside" x="${n.x+29}" y="${n.y+22}" text-anchor="middle">Attention · FFN</text>`:''}</g>`;}).join('')}</svg>`;
}
function generationExample(kind='all'){
 const e=esc,{context,scores,predictions,next}=M.autoregressiveExample();
 const word=id=>e(M.math.vocab[id]),tokens=(ids,extra='')=>ids.map((id,i)=>`<span class="generation-token ${extra&&i===ids.length-1?extra:''}" data-token-id="${id}">${word(id)}</span>`).join('');
 if(kind==='step')return `<div class="generation-example"><p>${e(L('只看最后的“猫”：查到的六个候选分数','Look at the final Cat: six scores from the lookup'))}</p><table><thead><tr>${M.math.vocab.map((w,i)=>`<th class="${i===predictions[2]?'generation-chosen':''}">${e(w)}</th>`).join('')}</tr></thead><tbody><tr>${scores.slice(-6).map((score,i)=>`<td class="${i===predictions[2]?'generation-chosen':''}">${score}</td>`).join('')}</tr></tbody></table><p>${e(L('查表只提供分数；真正选出词的是“取最高分”。','The lookup only provides scores; Pick highest selects the token.'))}</p></div>`;
 const comparison=`<table class="generation-predictions"><thead><tr><th>${e(L('对应输入词','Input token'))}</th><th>${e(L('该位置预测的后一个词','Predicted next token'))}</th><th>${e(L('生成时如何使用','Use during generation'))}</th></tr></thead><tbody>${context.map((id,i)=>`<tr class="${i===2?'generation-chosen':''}" data-prediction-index="${i}"><td>${word(id)}</td><td>${word(predictions[i])}</td><td>${e(i===2?L('采用：续写整段文本','Use: continues the whole text'):L('不追加：文本中已有后续词','Skip: a following token already exists'))}</td></tr>`).join('')}</tbody></table>`;
 const update=`<div class="generation-window" data-generation-window><div><small>${e(L('原上下文','Old context'))}</small><div>${tokens(context)}</div></div><div><small>${e(L('只追加一个新词','Append just one token'))}</small><div><span class="generation-token generation-dropped" data-token-id="${context[0]}">${word(context[0])}</span>${tokens(context.slice(1))}<i>＋</i>${tokens([predictions[2]],'generation-new')}</div></div><div><small>${e(L('本关出口：最近三词','Lesson outlet: latest three'))}</small><div>${tokens(next,'generation-new')}</div></div></div><p>${e(L('划掉的旧词只因本关窗口装得下三个词而移出；“追加”本身不是替换整段文本。','The crossed-out token leaves only because this exercise’s window holds three tokens. Appending does not replace the whole text.'))}</p>`;
 return `<div class="generation-example">${kind==='feedback'?'':comparison}${kind==='pick'?'':update}</div>`;
}
function generationMap(entry,outlet){
 const e=esc;
 const roles=[
  ['predict','step',L('承担“模型＋词表投影”的打分工作','Stands in for model + vocabulary scoring'),L('前面用完整模型算分；本关查固定表，输出同样形状的候选分数。只替换打分环节，方便专注后面的循环。','Earlier lessons computed scores with the full model. Here a lookup supplies candidate scores of the same shape, isolating the loop that follows.')],
  ['select','pick',L('承担选词工作','Selects tokens'),L('六个候选中挑最高分。每个输入位置都会有一个预测，真正续写整段文本时只用最后位置的预测。','Choose the best of six candidates. Each input position has a prediction; only the final position’s prediction continues the whole text.')],
  ['append','feedback',L('承担更新上下文的工作','Updates the context'),L('组件记住本次原上下文，把最后一个预测追加进去，再保留最近三词。它是模型外的文本操作，不是 FFN 或 KV 缓存。','The module remembers the trial’s original context, appends the final prediction and keeps the latest three tokens. This is text handling outside the model, not an FFN or KV cache.')]
 ];
 return `<div class="structure-context"><p class="structure-family">${e(entry.where)}</p><h3>${e(L('这一关搭的是“下一次输入”，不是新的一层网络','Build the next input, not another network layer'))}</h3><p class="structure-reference">${e(L('前面已经学会模型怎样输出分数；现在看选出的新词如何成为下一次输入。','You have learned how the model produces scores. Now see how a selected token becomes part of the next input.'))}</p></div><div class="generation-map-grid"><section class="structure-local"><h4>${e(L('地图中的步骤 ↔ 棋盘组件','Map steps ↔ board modules'))}</h4><ol class="structure-flow" data-map-flow="autoregressive">${roles.map(([id,kind,role,detail])=>`<li class="map-step-focus ${id==='append'?'map-step-exit':''}" data-map-step="${id}" data-generation-module="${kind}"><strong>${e(M.definitions[kind].name)}</strong><span>${e(role)}</span><p>${e(detail)}</p>${id==='append'?`<b>${e(copy.exit)}</b>`:''}</li>`).join('')}</ol><p class="structure-note">${e(entry.note)}</p></section><section><h4>${e(L('用“我 / 爱 / 猫”看一次更新','One update of I / Love / Cat'))}</h4>${generationExample()}${outlet}</section></div><p class="generation-loop-note">↶ ${e(M.levels.autoregressive.rationale.takeaway)}</p><details class="generation-reference"><summary>${e(L('对照：真实 Transformer 的打分部分在哪里？','Reference: where does a real Transformer compute scores?'))}</summary><p>${e(L('“教学预测器”替代下面的词向量、编码器／解码器和词表投影这一整段打分。真实模型用整个可见前缀生成分数；本关的表只看各位置的词，所以不会学语言。','The teaching predictor replaces the scoring stage below: embeddings, encoder/decoder and vocabulary projection. A real model uses the visible prefix to compute scores; this table looks only at the token at each position, so it does not learn language.'))}</p>${originalArchitecture({...entry,flow:'generation',focus:['layers','logits'],referenceOnly:true})}</details>`;
}
function render(id){
 const entry=get(id),e=esc;
 const outlet=`<div class="structure-outlet"><small>${e(copy.exit)}</small><strong>${e(entry.output)}</strong><code>${e(entry.shape)}</code></div>`;
 if(id==='autoregressive')return generationMap(entry,outlet);
 const step=(n)=>`<li data-map-step="${n.id}" class="${entry.focus.includes(n.id)?'map-step-focus ':''}${entry.exit===n.id?'map-step-exit':''}"><span>${e(n.label)}</span>${entry.exit===n.id?`<b>${e(copy.exit)}</b>`:''}</li>`;
 let route=entry.steps.map(step).join('');
 if(entry.flow==='cross'){
  const ns=Object.fromEntries(entry.steps.map(n=>[n.id,n]));
  route=`<li class="map-branch-pair"><ol>${step(ns.prepared)}${step(ns.q)}</ol><ol>${step(ns.encoded)}${step(ns.kv)}</ol></li>`+entry.steps.filter(n=>!['prepared','encoded','q','kv'].includes(n.id)).map(step).join('');
 }
 if(entry.flow==='modernFFN'||entry.flow==='pairwise'){
  const ns=Object.fromEntries(entry.steps.map(n=>[n.id,n]));
  route=`<li class="map-branch-pair"><ol>${step(ns.up)}</ol><ol>${step(ns.gate)}${ns.silu?step(ns.silu):''}</ol></li>${step(ns.multiply)}${step(ns.down)}`;
 }
 if(entry.flow==='connection')return `<div class="structure-context"><p class="structure-family">${e(L('从连接组件开始','Start by connecting modules'))}</p><h3>${e(entry.where)}</h3><p>${e(entry.note)}</p></div><div class="structure-expanded-grid"><section class="structure-global">${diagram(entry)}<p>${e(L('例如：“我”的 [1, 0] 经过连线，出口仍是 [1, 0]。','For example, I enters as [1, 0] and leaves the wire as [1, 0].'))}</p>${outlet}</section><section class="structure-local"><h4>${e(copy.focus)}</h4><ol class="structure-flow" data-map-flow="connection">${route}</ol><p class="structure-map-legend">${e(copy.legend)}</p></section></div><details class="generation-reference wiring-reference"><summary>${e(L('参考：连线在完整模型里做什么？','Reference: what do wires do in a full model?'))}</summary><p>${e(L('每个模块算完，都需要把结果交给后面的模块。这里的箭头表示这种传递；Attention 和 FFN 的具体计算会在后续关卡学习。','Each module passes its result to another module. The arrows represent that transfer; later lessons introduce the computations inside Attention and FFN.'))}</p>${originalArchitecture(entry)}</details>`;
 const mapNote=entry.family==='original'?L('源句进入编码器；目标前缀与编码器 E 一起进入解码器。图中一个框可代表多层。','The encoder reads the source; the decoder reads the target prefix and encoder E. Each box can represent multiple layers.'):entry.family==='modern'?L('这里是解码器型扩展，没有原版的编码器交叉注意力支路。选出的新词会追加回前缀。','These are decoder-style extensions without the original encoder cross-attention branch. Selected tokens are appended to the prefix.'):L('设备位置图，不表示固定的算子先后顺序。数学结果与执行成本分别验收。','Device locations, not a fixed operator sequence. Mathematical results and execution costs are checked separately.');
 return `<div class="structure-context"><p class="structure-family">${e(copy[entry.family])}</p><h3>${e(entry.where)}</h3><p class="structure-reference">${e(copy.reference)}</p></div>${entry.family==='original'?originalArchitecture(entry):entry.family==='modern'?modernArchitecture(entry):executionArchitecture(entry,id)}<div class="structure-expanded-grid"><section class="structure-global">${entry.family==='original'?'':diagram(entry)}<p>${e(mapNote)}</p>${outlet}<div class="structure-next"><small>${e(copy.next)}</small><p>${e(entry.next)}</p></div></section><section class="structure-local"><h4>${e(copy.focus)}</h4><ol class="structure-flow" data-map-flow="${entry.flow}">${route}</ol><p class="structure-map-legend">${e(copy.legend)}</p>${entry.note?`<p class="structure-note">${e(entry.note)}</p>`:''}</section></div>`;
}
M.structureMap={copy,get,lessons,flows,render,generationExample};
})(window.NeuralPrologue);
