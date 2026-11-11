/* One description of each lesson boundary, shared by the board and its hint.
 * These labels describe the authored task; they never alter its arithmetic. */
(function(M){
const L=(zh,en)=>{globalThis.AtelierEnglish[zh]=en;return I18n.t(zh);};
const audited=new Set();
function output(ids,zh,en,noteZh,noteEn){
 for(const id of ids.split(' ')){
  const l=M.levels[id];if(!l)throw new Error('Unknown lesson boundary: '+id);
  l.outputTitle=L(zh,en);l.outputNote=L(noteZh,noteEn);
  l.resultNote=l.outputTitle+'：'+l.outputNote+'。';
  globalThis.AtelierEnglish[l.resultNote]=en+': '+noteEn+'.';
  audited.add(id);
 }
}
function input(ids,zh,en,noteZh,noteEn){
 for(const id of ids.split(' ')){
  const l=M.levels[id];l.sourceTitle=L(zh,en);l.sourceNote=L(noteZh,noteEn);
  l.sourceNames={...l.sourceNames,R:l.sourceTitle};l.routeInput=l.sourceTitle;
 }
}
input('wire projection relu down-projection original-ffn workshop residual activation plain-ffn gate-projection pairwise gating swiglu','输入向量 X','Input vector X','游戏直接给定的单个词练习向量','A single-token practice vector supplied directly by the game');
input('embedding positions wider-features','词表编号 Token ID','Vocabulary token IDs','三个词的词表编号，尚未查找词向量','Vocabulary IDs for three tokens, before embedding lookup');
// The isolated normalization lessons deliberately do not inherit prior outputs.
for(const id of ['layernorm','add-norm'])M.levels[id].routeInput=M.levels[id].sourceTitle;
input('qkv scores softmax pack-scores weighted-sum self-attention multihead causal rotation rope aggregation attention block normalize','输入向量矩阵 X','Input vector matrix X','游戏给定：每个词两个特征','Game-supplied input: two features per token');
M.levels.scores.sourceNote=L('与上一关相同的 X，供 Q/K 投影','The same X as before, for Q/K projections');
M.levels['pack-scores'].sourceNote=L('与上一关相同的 X，重新计算 A','The same X as before; compute A again');
input('encoder-layer','带位置的四维词向量 X','Four-feature positional embeddings X','四维词向量查表、缩放并加入位置后的结果','After four-feature embedding lookup, scaling and position encoding');
input('masked-multihead decoder-self','目标前缀特征 X','Target-prefix features X','目标词已查表并加入位置，每词四个特征','Target embeddings plus positions; four features per token');
input('decoder-layer','目标前缀特征 D','Target-prefix features D','目标词已查表并加入位置，每词四个特征','Target embeddings plus positions; four features per token');
// D has already passed through the masked-attention sublayer in level 14.
M.levels['cross-attention'].routeInput=L('目标上下文 D + 编码器输出 E','Target context D + encoder output E');
M.levels['cross-attention'].secondaryTitle=L('编码器输出，用于 K/V 投影','Encoder output for K/V projections');
M.levels['decoder-layer'].sourceNames.Z=L('编码器输出 E','Encoder output E');
M.levels['decoder-layer'].routeInput=L('目标前缀特征 D + 编码器输出 E','Target-prefix features D + encoder output E');
M.levels['decoder-layer'].secondaryTitle=L('编码器输出，供交叉注意力读取','Encoder output for cross-attention');
for(const id of ['vocabulary-scores','vocabulary','autoregressive'])M.levels[id].routeInput=M.levels[id].sourceTitle;
for(const id of ['encoder-decoder','transformer-demo']){
 M.levels[id].routeInput=L('源序列 ID + 目标前缀 ID','Source IDs + target-prefix IDs');
}
M.levels['encoder-decoder'].sourceNote=L('已有的目标前缀，尚未查找词向量','Existing target prefix, before embedding lookup');
M.levels['encoder-decoder'].secondaryTitle=L('完整源句的词表编号，供编码器读取','Full source token IDs for the encoder');
M.levels.ffn.routeInput=L('原输入 R + 已归一化输入 Z','Original input R + normalized input Z');

output('wire','原样传递的向量 Y','Unchanged vector Y','Y = X，连线只传递数值','Y = X; the wire only transfers values');
output('projection','线性展开结果 P','Expanded projection P','P = XW₁，每词从两个特征展开到四个','P = XW₁; two features expand to four');
output('relu','ReLU 激活结果 A','ReLU activations A','A = ReLU(XW₁)，负数变为零','A = ReLU(XW₁); negative values become zero');
output('down-projection','降维投影结果 F','Down-projection result F','F = (XW₁)W₂，本关未加入激活','F = (XW₁)W₂; this lesson has no activation');
output('original-ffn workshop','ReLU FFN 输出 F','ReLU FFN output F','F = ReLU(XW₁)W₂，尚未加残差','F = ReLU(XW₁)W₂, before residual addition');
output('residual','残差相加结果 Y','Residual sum Y','Y = X + FFN(X)，尚未做 LayerNorm','Y = X + FFN(X), before LayerNorm');
output('embedding','词向量矩阵 X','Embedding matrix X','按 ID 查表得到，每词两个特征','Looked up by ID; two features per token');
output('positions','带位置的词向量 Xp','Positional embeddings Xp','Xp = √2 × 词向量 + 正弦位置编码','Xp = √2 × embeddings + sinusoidal positions');
output('layernorm','LayerNorm 结果 Y','LayerNorm result Y','每个练习向量独立减均值、缩放','Each practice vector is centered and scaled independently');
output('add-norm','FFN 残差归一化结果 Y','Normalized FFN residual Y','Y = LayerNorm(X + FFN(X))','Y = LayerNorm(X + FFN(X))');
output('qkv','值向量矩阵 V','Value matrix V','V = X × Wᵥ','V = X × Wᵥ');
output('scores','匹配分数矩阵 S','Matching-score matrix S','S = QKᵀ / √2，每格对应一对查询词与键词','S = QKᵀ / √2; one score per query–key pair');
output('softmax pack-scores','注意力权重矩阵 A','Attention-weight matrix A','每个数是读取对应词 V 的比例','Each value is a proportion for reading a token’s V');
output('weighted-sum','加权汇合结果 O','Weighted-sum result O','O = A × V，尚未经过输出投影','O = A × V, before output projection');
output('self-attention','单头注意力输出 Y','Single-head attention output Y','Y = (A × V)Wo，尚未加残差','Y = (A × V)Wo, before residual addition');
output('multihead','双头注意力输出 Y','Two-head attention output Y','Y = Concat(H₁, H₂)Wo，尚未加残差','Y = Concat(H₁, H₂)Wo, before residual addition');
output('wider-features','带位置的四维词向量 Xp','Four-feature positional embeddings Xp','Xp = 2 × 四维词向量 + 正弦位置编码','Xp = 2 × four-feature embeddings + sinusoidal positions');
output('encoder-layer','编码器层输出 E','Encoder layer output E','已完成注意力、FFN 及两次残差与 LayerNorm','After attention, FFN and both residual/LayerNorm stages');
output('causal','因果注意力权重 A','Causal attention weights A','每个查询的权重和为 1，未来词的权重为 0','Each query’s weights sum to 1; future tokens receive zero weight');
output('masked-multihead','双头因果注意力输出 Y','Two-head causal-attention output Y','两头分别遮罩和汇合，再拼接并投影；尚未加残差','Each head masks and aggregates separately; concatenate and project the head outputs before residual addition');
output('decoder-self','目标上下文特征 D','Target-context features D','因果自注意力 + 残差 + LayerNorm 的结果','After causal self-attention, residual addition and LayerNorm');
output('cross-attention','交叉注意力输出 Y','Cross-attention output Y','按目标查询汇合源句信息，尚未加残差','Source information aggregated for target queries, before residual addition');
output('decoder-layer','解码器层输出 D','Decoder layer output D','已完成三个子层，各自先加残差再做 LayerNorm','After all three sublayers, each with residual addition followed by LayerNorm');
output('vocabulary-scores encoder-decoder','候选词分数 L','Candidate-token scores L','每个位置的六个候选词分数，尚未选词','Six candidate scores per position, before token selection');
output('vocabulary','预测词的 Token ID','Predicted token IDs','每个位置输出最高分词的 ID；最后一个预测下一词','The highest-scoring token ID at each position; the last predicts the next token');
output('autoregressive','更新后的三词上下文','Updated three-token context','追加一个新词后，只保留最近三个词','Append one token, then keep only the latest three');
output('transformer-demo','各目标位置的预测词','Predictions at target positions','最后一个位置预测下一词','The final position predicts the next token');

output('activation','SiLU 激活结果 S','SiLU activations S','S = SiLU(XW₁)，尚未降维','S = SiLU(XW₁), before down projection');
output('plain-ffn','SiLU FFN 输出 F','SiLU FFN output F','F = SiLU(XW₁)W₂，尚未加残差','F = SiLU(XW₁)W₂, before residual addition');
output('gate-projection','门投影结果 G','Gate-projection result G','G = XWg，尚未经过 SiLU','G = XWg, before SiLU');
output('pairwise','两路逐元素乘积 H','Elementwise branch product H','H = (XW₁) ⊙ (XWg)，本关未加入 SiLU','H = (XW₁) ⊙ (XWg); no SiLU in this lesson');
output('gating','门控后的展开特征 H','Gated expanded features H','H = (XW₁) ⊙ SiLU(XWg)，尚未降维','H = (XW₁) ⊙ SiLU(XWg), before down projection');
output('swiglu','SwiGLU FFN 输出 F','SwiGLU FFN output F','门控结果已降回两个特征，尚未加残差','Gated values projected back to two features; before residual');
output('normalize','RMSNorm 结果 Z','RMSNorm result Z','每个词独立按均方根缩放，仍有两个特征','Each token is scaled by its RMS, retaining two features');
output('ffn','SwiGLU 残差结果 Y','SwiGLU residual result Y','Y = R + SwiGLU(Z)，Z 是已归一化的 R','Y = R + SwiGLU(Z); Z is the normalized R');
output('rotation','旋转后的查询 Q′','Rotated queries Q′','Q′ = RoPE(XWq)，本关还未计算匹配分数','Q′ = RoPE(XWq); scores are not computed here');
output('rope','RoPE 匹配分数 S','RoPE matching scores S','旋转后的 Q/K 打分，尚未经过遮罩或 Softmax','Scores from rotated Q/K, before masking or Softmax');
output('aggregation','加权汇合结果 O','Weighted-sum result O','O = A × V，按因果注意力权重汇合；尚未投影','O = A × V with causal attention weights, before output projection');
output('attention','现代注意力输出 Y','Modern attention output Y','已完成 RoPE、因果遮罩、汇合和输出投影；尚未加残差','After RoPE, causal masking, aggregation and output projection; before residual addition');
output('block','现代解码器块输出 Y','Modern decoder-block output Y','已完成注意力、SwiGLU 及各自的残差','After attention, SwiGLU and both residual additions');

input('single-upload single-fallback single-resident','CPU 输入向量 X','CPU input vector X','两个特征，等待上传到 GPU','Two features awaiting upload to the GPU');
input('tp-columns tp-rows tp-ffn','输入向量 X','Input vector X','同一份输入，用于比较两张 GPU 拆分与完整计算','Shared input for comparing two-GPU sharding with full computation');
output('single-upload','读回 CPU 的投影结果 P','Projection P read back to CPU','P = XW，每词四个特征','P = XW; four features per token');
output('single-fallback single-resident','读回 CPU 的 ReLU FFN 结果 F','ReLU FFN result F read back to CPU','F = ReLU(XW)D，CPU/GPU 分工不改变数学目标','F = ReLU(XW)D; CPU/GPU placement preserves the target');
output('tp-columns','拼接后的投影结果 H','Gathered projection H','H = XW，两卡各提供不同的特征','H = XW; the GPUs supply distinct features');
output('tp-rows','归并后的降维结果 F','Reduced down-projection result F','F = (XW)D，两卡部分和相加；未加入 ReLU','F = (XW)D; add partial sums, with no ReLU');
output('tp-ffn','两卡 ReLU FFN 结果 F','Two-GPU ReLU FFN result F','F = ReLU(XW)D，与未拆分的 FFN 相同','F = ReLU(XW)D, matching the unsharded FFN');
const engineIds=M.routes.engine.ids.join(' ');
input(engineIds,'提示词向量矩阵 X','Prompt-vector matrix X','本练习直接提供词向量，每词两个特征','This exercise supplies embeddings directly, two features per token');
output(engineIds,'推理末端特征 Y','Final inference features Y','三个词各有两个特征；预测词由运行报告另行读出','Two features per token for three tokens; predictions are read out separately in the run report');
M.definitions.enhead.name=L('末端特征交接','Final feature handoff');
M.definitions.enhead.hint=L('本调度任务原样传递三个词的两维特征。运行报告用词表投影和取最高分读出预测词。关卡出口的数值不是词表分数，也不是 Token ID。','This scheduling task passes through two-feature vectors for three tokens. The run report applies vocabulary projection and maximum selection to produce predicted tokens. Values at the lesson outlet are neither vocabulary scores nor token IDs.');

for(const id of ['scores','softmax','pack-scores','causal','rope'])M.levels[id].outputPairAxes=true;
const rename=(id,title,en)=>{const l=M.levels[id];l.title=L(title,en);l.concept=l.title;};
rename('pack-scores','用上组好的打分模块','Put the scoring module to work');
M.levels['pack-scores'].concept=L('注意力权重的封装与复用','Packaging and reusing attention weights');
M.levels['pack-scores'].boardText='ORIGINAL / 07 · REUSE';
M.levels['pack-scores'].chapter=L('原版基础 · 07 后复用练习','Original · reuse after 07');
rename('masked-multihead','把遮罩放进两个头','Put the mask inside both heads');
M.levels['masked-multihead'].concept=L('双头因果注意力','Two-head causal attention');
M.levels['masked-multihead'].boardText='ORIGINAL / ASSEMBLY';
M.levels['masked-multihead'].chapter=L('原版基础 · 组合练习','Original · assembly practice');
rename('qkv','先准备好可读取的内容','Prepare the content to read');
M.levels.qkv.concept=L('值向量 V','Value vectors V');
rename('scores','看看词与词有多匹配','See how tokens match');
M.levels.scores.concept=L('Q/K 缩放点积打分','Q/K scaled dot-product scoring');
rename('weighted-sum','按比例取回内容','Read content in proportion');
M.levels['weighted-sum'].concept=L('注意力加权求和','Attention weighted sum');
rename('ffn','把新变化加回原路','Add the update to the original path');
M.levels.ffn.concept=L('SwiGLU 残差连接','SwiGLU residual connection');
M.lessonIO={audited:[...audited]};
I18n.prepare(M.levels);I18n.prepare(M.definitions);
})(window.NeuralPrologue);
