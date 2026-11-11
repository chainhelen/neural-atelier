/* Original Transformer first. Existing modern circuits keep their IDs and arithmetic. */
(function(M){
// Keep each new lesson's English copy next to its Chinese source.
const L=(zh,en)=>{globalThis.AtelierEnglish[zh]=en;return I18n.t(zh);};
L('原版基础','Original');L('现代扩展','Modern');L('源序列','Source sequence');L('目标前缀','Target prefix');
L('3 个位置 × 4 个特征','3 positions × 4 features');
L('这里只展开第一个位置的归一化，其他位置分别计算。','The calculation below expands the first position; other positions are normalized independently.');
L('残差相加结果','Residual sum');
const {rows,mat,scores,probabilities,mix,causal}=M.math,flat=x=>x.flat(),plus=(a,b)=>a.map((x,i)=>x+b[i]);
const relu=x=>Math.max(0,x),eps=1e-6;
const wide={
 q:[[1,.2,0,-.1],[.1,1,.3,0],[.2,-.1,.8,.1],[0,.2,.1,1]],
 k:[[.8,0,.1,-.2],[-.1,1,.2,.1],[.3,.1,1,0],[0,-.2,.1,.9]],
 v:[[1,.1,0,.2],[0,.9,.2,0],[.1,0,.8,.1],[-.2,.1,0,1]],
 output:[[.9,.1,0,.1],[-.1,.8,.2,0],[.2,0,.9,-.1],[0,.1,.2,.8]],
 up:[[1,0,0,.5,1,-.5,.2,.4],[0,1,.5,0,.2,1,-.3,.1],[.2,0,1,-.2,.5,.2,1,-.3],[0,.3,-.2,1,-.1,.5,.2,1]],
 down:[[.4,0,.1,0],[0,.3,0,.1],[.1,.1,.4,0],[-.1,.2,.1,.3],[.2,.1,-.1,.2],[0,.2,.1,-.1],[.1,-.2,.3,.1],[-.1,.1,0,.2]],
 embedding:[[1,0,.3,-.4],[0,1,-.2,.5],[1,1,.4,.2],[-1,0,-.3,.4],[0,-1,.2,-.5],[-1,-1,-.4,-.2]],
 vocabulary:[[1,0,1,-1,0,-1],[0,1,1,0,-1,-1],[.3,-.2,.1,-.3,.2,-.1],[-.1,.3,.2,.1,-.3,-.2]]
};
const project=(v,w)=>flat(rows(v,w.length).map(r=>M.linear(r,w)));
function ln(v){const mean=v.reduce((a,b)=>a+b,0)/v.length,variance=v.reduce((s,x)=>s+(x-mean)**2,0)/v.length;return v.map(x=>(x-mean)/Math.sqrt(variance+eps));}
const lnSequence=v=>flat(rows(v,v.length/3).map(ln));
const classicFFN=v=>M.linear(M.linear(v,v.length===4?wide.up:M.weights.project).map(relu),v.length===4?wide.down:M.weights.down);
const siluFFN=v=>M.linear(M.linear(v,M.weights.project).map(M.silu),M.weights.down);
const swigluFFN=v=>{const g=M.linear(v,M.weights.gate).map(M.silu),u=M.linear(v,M.weights.project);return M.linear(g.map((x,i)=>x*u[i]),M.weights.down);};
const sequenceFFN=v=>flat(rows(v,v.length/3).map(classicFFN));
const position=v=>{const width=v.length/3;return flat(rows(v,width).map((r,p)=>r.map((x,j)=>Math.sqrt(width)*x+(j%2?Math.cos:Math.sin)(p/10000**(2*Math.floor(j/2)/width)))));};
const embed=(ids,width=2)=>flat(ids.map(i=>(width===4?wide.embedding:M.math.emb)[i]));
const unmaskedWeights=x=>probabilities(scores(mat(x,M.weights.q),mat(x,M.weights.k)));
const scoreWide=(q,k)=>flat(rows(q,4).map(a=>rows(k,4).map(b=>a.reduce((s,x,i)=>s+x*b[i],0)/2)));
const mixWide=(a,v)=>flat(rows(a,3).map(r=>[0,1,2,3].map(j=>r.reduce((s,p,i)=>s+p*rows(v,4)[i][j],0))));
const vocabularyBias=[0,0,-.75,0,0,-.75];
const vocabulary=v=>flat(rows(v,4).map(r=>M.linear(r,wide.vocabulary).map((x,i)=>x+vocabularyBias[i])));
const singleAttention=(qSource,kvSource=qSource)=>qSource.length===12?project(mixWide(probabilities(scoreWide(project(qSource,wide.q),project(kvSource,wide.k))),project(kvSource,wide.v)),wide.output):mat(mix(probabilities(scores(mat(qSource,M.weights.q),mat(kvSource,M.weights.k))),mat(kvSource,M.weights.v)),M.weights.output);
function head(qSource,kvSource,index,masked=false){
 const width=qSource.length/3,headWidth=width/2,w=width===4?wide:M.weights;
 const split=(v,m)=>rows(project(v,m),width).map(r=>r.slice(index*headWidth,(index+1)*headWidth)),q=split(qSource,w.q),k=split(kvSource,w.k),v=split(kvSource,w.v);
 const s=flat(q.map(a=>k.map(b=>a.reduce((sum,x,j)=>sum+x*b[j],0)/Math.sqrt(headWidth)))),a=rows(probabilities(masked?causal(s):s),3);
 return flat(a.map(row=>Array.from({length:headWidth},(_,j)=>row.reduce((sum,p,i)=>sum+p*v[i][j],0))));
}
const concat=(a,b)=>flat(rows(a,a.length/3).map((r,i)=>[...r,...rows(b,b.length/3)[i]]));
const multi=(q,kv=q,masked=false)=>project(concat(head(q,kv,0,masked),head(q,kv,1,masked)),q.length===12?wide.output:M.weights.output);
function encoder(x){const a=lnSequence(plus(x,multi(x)));return lnSequence(plus(a,sequenceFFN(a)));}
function decoderSelf(x){return lnSequence(plus(x,multi(x,x,true)));}
function decoder(x,e){const a=decoderSelf(x),b=lnSequence(plus(a,multi(a,e)));return lnSequence(plus(b,sequenceFFN(b)));}
const sourceIds=[2,0,1],sourceFeatures=()=>position(embed(sourceIds,4)),memory=()=>encoder(sourceFeatures());
const def=(name,ins,out,type,hint,extra={})=>({name,ins,out,type,hint,x:0,z:0,...extra});
const labels={
 relu:L('ReLU · 负数归零','ReLU · zero negative values'),
 ln:'LayerNorm',
 ffn:L('原版 FFN · ReLU','Original FFN · ReLU'),
 attn:L('双头自注意力','Two-head self-attention'),
 masked:L('双头因果注意力','Two-head masked attention'),
 cross:L('双头交叉注意力','Two-head cross-attention'),
 position:L('正弦位置编码','Sinusoidal positions')
};
const lnHint=L('LayerNorm 是基础算子，不是由前面的模块封装而成。它对每个词的特征单独计算：先减均值，再除以标准差。本例固定 γ = 1、β = 0。','LayerNorm is a basic operator, not a package of earlier modules. It processes each token’s features separately: subtract the mean, then divide by the standard deviation. This example fixes γ = 1 and β = 0.');
const ffnHint=L('每个 token 分别经过 Linear → ReLU → Linear。两次投影的偏置在本例设为 0；内部没有门分支和归一化。','Each token follows Linear → ReLU → Linear. Both biases are zero in this example; there is no gate branch or normalization inside.');
const positionHint=L('原版先把词向量乘 √d_model，再加正弦/余弦位置编码。本例 d_model = 2，位置 p 加 [sin(p), cos(p)]。','The original scales embeddings by √d_model, then adds sinusoidal positions. Here d_model = 2, and position p adds [sin(p), cos(p)].');
const widePositionHint=L('四维词向量乘 √4，再加 [sin(p), cos(p), sin(p/100), cos(p/100)]。两对特征使用不同频率，仍是原版的相加位置编码。','Scale four-feature embeddings by √4, then add [sin(p), cos(p), sin(p/100), cos(p/100)]. The two pairs use different frequencies, still using the original additive encoding.');
Object.assign(M.definitions,{
 relu:def(labels.relu,[4],4,'activation',L('每个数计算 max(0, x)：负数变 0，正数保持原值。维度不变。','Apply max(0, x) to each value: negatives become zero and positives stay unchanged. Dimensions are preserved.'),{activation:'relu'}),
 layernorm2:def(labels.ln,[2],2,'layernorm',lnHint),
 'relu-ffn':def(labels.ffn,[2],2,'ffn',ffnHint,{glyphs:['W1','ReLU','W2']}),
 position:def(labels.position,[6],6,'position',positionHint),
 'original-weights':def(L('已学的打分与 Softmax','Scores and Softmax'),[6],9,'attention',L('把 Q/K 投影、缩放点积和 Softmax 封装起来。三个位置彼此可见，没有 RoPE 和因果遮罩。','Packages Q/K projections, scaled dot products and Softmax. All three positions are visible, without RoPE or a causal mask.')),
 'single-attention':def(L('单头自注意力','Single-head self-attention'),[6],6,'attention',L('包含本关的 Q/K/V、打分、Softmax、A × V 和 Wo。计算与七个独立模块相同。每个词仍有两个特征。不含残差、归一化或 FFN。','Contains this lesson’s Q/K/V, scoring, Softmax, A × V and Wo. The computation matches the seven separate modules. Each token retains two features. Residuals, normalization and FFNs are not included.')),
 'two-head-attention':def(L('双头自注意力 · 两维','Two-head self-attention · two features'),[6],6,'attention',L('两个头各算一个特征，再拼接并乘 Wo。与两维的双头关卡使用同样的权重；完整编码器里的四维版本另有四维端口。','Each head computes one feature; concatenate the heads and apply Wo. Uses the same weights as the two-feature lesson. The four-feature version in the complete encoder has separate four-feature ports.')),
 'single-cross-attention':def(L('单头交叉注意力','Single-head cross-attention'),[12,12],12,'attention',L('第一个入口提供目标特征，第二个入口提供编码器的源特征。内部保留四维 Q/K/V、打分、Softmax、A × V 和 Wo，与刚搭好的单头交叉注意力一致。','The first input supplies target features; the second supplies source features from the encoder. Contains four-feature Q/K/V, scoring, Softmax, A × V and Wo, matching the single-head cross-attention circuit.')),
 'head-a':def(L('注意力头 A','Attention head A'),[6],3,'attention',L('用第一组 Q/K/V 权重各投影到一个特征。本头独立打分、计算 Softmax 并汇合 V。每个位置输出一个数。','Project to one feature with the first set of Q/K/V weights. This head independently scores, applies Softmax and aggregates V. It outputs one value per position.'),{glyphs:['Q₁','K₁','V₁']}),
 'head-b':def(L('注意力头 B','Attention head B'),[6],3,'attention',L('用第二组 Q/K/V 权重做独立注意力。输入相同，权重不同，因此可以得到另一种信息组合。','Use a second set of Q/K/V weights for independent attention. The same input with different weights can yield a different combination of information.'),{glyphs:['Q₂','K₂','V₂']}),
 concat:def(L('按特征拼接','Concatenate features'),[3,3],6,'concat',L('把同一位置的头 A、头 B 输出并排放成两个特征。保留三个 token 的顺序，不相加、不把 token 接到序列尾部。','Place head A and head B side by side at each position. Keep token order; do not add the heads or append tokens to the sequence.')),
 'self-heads':def(labels.attn,[6],6,'attention',L('两个头分别计算 Q/K/V、缩放点积、Softmax 和汇合。将两头结果按特征拼接，再乘 Wo。所有位置互相可见。','Each head independently computes Q/K/V, scaled dot products, Softmax and aggregation. Concatenate the head outputs along the feature dimension, then apply Wo. All positions can read one another.')),
 'masked-heads':def(labels.masked,[6],6,'attention',L('双头注意力在 Softmax 前遮住未来位置。当前 token 只能读取自己和前面的目标 token。','Both heads mask future positions before Softmax. Each token may read only itself and earlier target tokens.'),{causal:true}),
 'cross-heads':def(labels.cross,[6,6],6,'attention',L('第一个输入提供目标特征，第二个输入提供编码器的源特征。模块内部从目标特征投影出 Q，从源特征投影出 K/V；每个目标位置都可读取完整源序列，无需遮住源序列的后面。','The first input supplies target features; the second supplies source features from the encoder. Inside the module, target features are projected into Q and source features into K/V. Every target position can read the whole source sequence without a future mask on the source.')),
 'ln-attn':def(L('Attention 后的 LayerNorm','LayerNorm after Attention'),[6],6,'layernorm',lnHint),
 'ln-ffn':def(L('FFN 后的 LayerNorm','LayerNorm after FFN'),[6],6,'layernorm',lnHint),
 'ln-cross':def(L('交叉注意力后的 LayerNorm','LayerNorm after cross-attention'),[6],6,'layernorm',lnHint),
 'original-ffn-block':def(labels.ffn,[6],6,'ffn',ffnHint,{glyphs:['W1','ReLU','W2']}),
 'cross-add':def(L('交叉注意力残差','Cross-attention residual'),[6,6],6,'residual',L('将交叉注意力得到的更新加回发起查询的解码器特征。不要加到编码器的源特征上。','Add the cross-attention update back to the decoder features that supplied the queries, not to the encoder source features.')),
 'original-encoder':def(L('已完成的原版编码器','Completed original encoder'),[6],6,'encoder',L('双头自注意力 → 残差相加 → LayerNorm → ReLU FFN → 残差相加 → LayerNorm。本例只用一层。','Two-head self-attention → residual addition → LayerNorm → ReLU FFN → residual addition → LayerNorm. This example uses one layer.')),
 'original-decoder':def(L('已完成的原版解码器','Completed original decoder'),[6,6],6,'decoder',L('目标输入先做因果自注意力，再以编码器输出为 K/V 做交叉注意力，最后经过 ReLU FFN。三段都先加残差，再做 LayerNorm。','The target first uses masked self-attention, then cross-attends to encoder K/V, then uses a ReLU FFN. Each sublayer adds its residual before LayerNorm.')),
 'source-embed':def(L('源词向量与位置','Source embeddings and positions'),[3],6,'position',positionHint),
 'target-embed':def(L('目标词向量与位置','Target embeddings and positions'),[3],6,'position',positionHint)
});
for(const id of ['self-heads','masked-heads','cross-heads','ln-attn','ln-ffn','ln-cross','original-ffn-block','cross-add','original-encoder','original-decoder']){
 const d=M.definitions[id];d.ins=d.ins.map(()=>12);d.out=12;
}
for(const id of ['source-embed','target-embed']){M.definitions[id].out=12;M.definitions[id].hint=widePositionHint;}
Object.assign(M.definitions,{
 'wide-embedding':def(L('词向量表 · 四维','Embedding table · four features'),[3],12,'embedding',L('按 token ID 查表，为每个 token 取出四个数，组成一个四维向量。四个数共同表示同一个 token；每一维是数值坐标，不一定有单独可命名的含义。','Look up each token ID to retrieve four numbers forming one four-dimensional vector. Together they represent the same token; a coordinate does not necessarily have a separately named meaning.')),
 'wide-position':def(labels.position,[12],12,'position',widePositionHint),
 'encoder-add':def(L('Attention 残差','Attention residual'),[12,12],12,'residual',L('把注意力更新加到当前子层输入，随后交给 LayerNorm。','Add the attention update to the current sublayer input, then send the sum to LayerNorm.')),
 'ffn-add':def(L('FFN 残差','FFN residual'),[12,12],12,'residual',L('将 FFN 更新加到进入 FFN 前的特征，随后交给 LayerNorm。','Add the FFN update to the features entering the FFN, then send the sum to LayerNorm.')),
 'original-score':def(L('QKᵀ · 四维打分','QKᵀ · four-feature scores'),[12,12],9,'score',L('每个四维 Q 与每个四维 K 点乘，再除以 √4。三个目标位置对三个源位置得到九个分数。','Dot each four-feature Q with each four-feature K and divide by √4. Three target positions against three source positions produce nine scores.')),
 'original-aggregate':def(L('AV · 汇合四维内容','AV · aggregate four-feature values'),[9,12],12,'mix',L('每个目标位置按三个注意力权重加权汇合源 V，保留四个输出特征。','Each target position uses its three attention weights to combine source V while retaining four output features.')),
 'original-head':def(L('词表投影 · 四维','Vocabulary projection · four features'),[12],18,'vocabulary',L('为每个目标位置的四维特征计算六个候选词分数，并加入固定偏置。','Compute six candidate-token scores from each target’s four features and add the fixed bias.')),
 'wide-invert':def(L('反向投影 · 四维','Negation · four features'),[12],12,'invert',L('把每个特征乘 −1。它会改变送入词表投影的数值，供你比较候选分数怎样变化。','Multiply each feature by −1. This changes the input to the vocabulary projection so you can compare the resulting candidate scores.'))
});
for(const [kind,name]of [['q',L('Q · 查询投影','Q · query projection')],['k',L('K · 键投影','K · key projection')],['v',L('V · 值投影','V · value projection')],['output',L('输出投影 Wo','Output projection Wo')]]){
 M.definitions['original-'+kind]=def(name,[12],12,'linear',L('每个位置的四个特征乘同一套 4 × 4 权重，产生四个新特征。不会在这里混合不同 token。','Each position’s four features use the same 4 × 4 weights to produce four new features. Tokens are not mixed by this projection.'));
 M.weights['original-'+kind]=wide[kind];
}
Object.assign(M.definitions,{
 'silu-ffn':def(L('单支路 FFN · SiLU','Single-branch FFN · SiLU'),[2],2,'ffn',L('封装线性投影、SiLU 和降维投影。每个词的两个数先展开为四个。经过 SiLU 后，再组合回两个。不含门分支、归一化或残差。','Packages linear projection, SiLU and down projection. Each token’s two values expand to four. After SiLU, they are projected back to two. No gate branch, normalization or residual is included.'),{glyphs:['W1','SiLU','W2']}),
 'swiglu-ffn':def(L('SwiGLU FFN','SwiGLU FFN'),[2],2,'ffn',L('同一输入分别产生内容和门值。门值经过 SiLU 后，与内容逐元素相乘。乘积再经过降维投影。这里只输出修改量 F；归一化与残差仍在模块外。','Project the same input into content and gate values. Apply SiLU to the gate, then multiply it elementwise with the content. Project the product down. This module outputs only the update F; normalization and the residual remain outside.'),{glyphs:['W / Wg','SiLU ×','Wd']}),
 'swiglu-residual':def(L('SwiGLU + 残差','SwiGLU + residual'),[2,2],2,'ffn',L('保留 R、Z 两个入口。用 Z 计算 SwiGLU 的修改量，再加到 R。计算与 SwiGLU 残差关相同。Z 的归一化仍由模块外提供。','Keeps separate R and Z inputs. Compute the SwiGLU update from Z, then add it to R. This matches the SwiGLU residual lesson. Normalization that produces Z remains outside the module.'),{glyphs:['Z → FFN','+ R']}),
 'original-transformer':def(L('原版编码器—解码器','Original encoder–decoder'),[3,3],18,'decoder',L('保留目标与源 Token ID 两个入口。内部包含两边的词向量与位置、编码器、解码器及词表投影。输出是每个目标位置的候选词分数。选词与生成循环仍在模块外。','Keeps separate target and source token-ID inputs. Includes both embedding and position stages, the encoder, the decoder and vocabulary projection. Outputs candidate-token scores at each target position. Token selection and the generation loop remain outside the module.'),{glyphs:['Embed','Enc / Dec','Scores']}),
 'modern-block':def(L('现代解码器层','Modern decoder layer'),[6],6,'decoder',L('包含 RMSNorm、带 RoPE 的因果 Attention、SwiGLU 和两次残差。每段先归一化，再把更新加回自己的输入；三个 token 各保留两个特征。','Contains RMSNorm, causal attention with RoPE, SwiGLU and both residuals. Each sublayer normalizes first, then adds its update to its own input. All three tokens retain two features each.'),{glyphs:['Norm / Attn','+','Norm / FFN','+']})
});
M.weights['original-head']=wide.vocabulary;
M.definitions['original-head'].bias=vocabularyBias;
Object.assign(M.operations,{
 relu:([x])=>x.map(relu),layernorm2:([x])=>ln(x),'relu-ffn':([x])=>classicFFN(x),position:([x])=>position(x),
 'original-weights':([x])=>unmaskedWeights(x),'single-attention':([x])=>singleAttention(x),'two-head-attention':([x])=>multi(x),'single-cross-attention':([q,kv])=>singleAttention(q,kv),'head-a':([x])=>head(x,x,0),'head-b':([x])=>head(x,x,1),concat:([a,b])=>concat(a,b),
 'self-heads':([x])=>multi(x),'masked-heads':([x])=>multi(x,x,true),'cross-heads':([q,kv])=>multi(q,kv),
 'ln-attn':([x])=>lnSequence(x),'ln-ffn':([x])=>lnSequence(x),'ln-cross':([x])=>lnSequence(x),
 'original-ffn-block':([x])=>sequenceFFN(x),'cross-add':([a,b])=>plus(a,b),
 'original-encoder':([x])=>encoder(x),'original-decoder':([x,e])=>decoder(x,e),
 'source-embed':([ids])=>position(embed(ids,4)),'target-embed':([ids])=>position(embed(ids,4)),
 'wide-embedding':([ids])=>embed(ids,4),'wide-position':([x])=>position(x),
 'encoder-add':([a,b])=>plus(a,b),'ffn-add':([a,b])=>plus(a,b),
 'original-score':([q,k])=>scoreWide(q,k),'original-aggregate':([a,v])=>mixWide(a,v),'original-head':([v])=>vocabulary(v),'wide-invert':([v])=>v.map(x=>-x),
 'original-q':([x])=>project(x,wide.q),'original-k':([x])=>project(x,wide.k),'original-v':([x])=>project(x,wide.v),'original-output':([x])=>project(x,wide.output)
});
Object.assign(M.operations,{
 'silu-ffn':([x])=>siluFFN(x),'swiglu-ffn':([x])=>swigluFFN(x),'swiglu-residual':([r,z])=>plus(r,swigluFFN(z)),
 'original-transformer':([target,source])=>vocabulary(decoder(position(embed(target,4)),encoder(position(embed(source,4))))),
 'modern-block':([x])=>M.math.block(x)
});
// Expand the four-feature masked block using the exact two-feature slice of each head.
const maskedProjectionIds=[];
for(const [headIndex,headName]of ['A','B'].entries())for(const role of ['q','k','v']){
 const id='masked-'+headName.toLowerCase()+'-'+role,w=wide[role].map(row=>row.slice(headIndex*2,headIndex*2+2));
 const name=L(role.toUpperCase()+' · 头 '+headName,role.toUpperCase()+' · head '+headName);
 M.definitions[id]=def(name,[12],6,'linear',L('四维输入乘本头的 4 × 2 权重，每个词得到两个特征。A、B 头使用不同的权重列；它们都读取同一份输入。','Multiply four input features by this head’s 4 × 2 weights to obtain two features per token. Heads A and B use different weight columns and read the same input.'));
 M.weights[id]=w;M.operations[id]=([x])=>project(x,w);maskedProjectionIds.push(id);
}
M.definitions['wide-concat']=def(L('按特征拼接 · 四维','Concatenate features · four features'),[6,6],12,'concat',L('每个词把头 A 的两个数放在前面、头 B 的两个数放在后面，合成四个特征。不是相加，也不增加词数。','For each token, place head A’s two numbers before head B’s two numbers to form four features. Do not add them or increase the token count.'));
M.operations['wide-concat']=([a,b])=>concat(a,b);
const operatorIds=['relu','layernorm2','relu-ffn','position','original-weights','head-a','head-b','concat','self-heads','masked-heads','cross-heads','ln-attn','ln-ffn','ln-cross','original-ffn-block','cross-add','original-encoder','original-decoder','source-embed','target-embed','wide-embedding','wide-position','encoder-add','ffn-add','original-score','original-aggregate','original-head','original-q','original-k','original-v','original-output'];
operatorIds.push(...maskedProjectionIds,'wide-concat','single-attention','two-head-attention','single-cross-attention','silu-ffn','swiglu-ffn','swiglu-residual','original-transformer','modern-block','wide-invert');
for(const id of operatorIds){
 const d=M.definitions[id];M.extraColors[id]=({linear:0x789abf,score:0x879bb3,mix:0x64a68f,vocabulary:0xb79b6b,embedding:0x7d9ea5,activation:0xc6a574,layernorm:0xb8a36a,ffn:0x759c88,position:0xb49b6a,attention:0x789abf,concat:0x9b86b4,residual:0xc3a26f,encoder:0x6a9c93,decoder:0x9387ad})[d.type];
}
M.extraColors['head-b']=0xa394bd;
M.extraColors['wide-invert']=0xb88475;
for(const id of maskedProjectionIds)M.extraColors[id]=({q:0x789abf,k:0xa394bd,v:0x5da995})[id.slice(-1)];
M.extraColors['original-k']=0xa394bd;M.extraColors['original-v']=0x5da995;
for(const id of ['mask','attention','attentionweights','enattn'])M.definitions[id].causal=true;
const zeroBias=L('使用原版的运算顺序；为了看清数值，特征宽度缩为 2、FFN 展开到 4，偏置设为 0。固定教学权重不代表训练好的语言模型。','Uses the original operation order with width 2, FFN width 4 and zero biases so the numbers are readable. Fixed teaching weights do not represent a trained language model.');
const scope=L('这是原版结构的缩尺推理：基础算子用两维，完整编码器和解码器用四维、两个头、一层；LayerNorm 的 γ = 1、β = 0，FFN 偏置为 0。省略训练和推理时关闭的 dropout，不复刻论文规模。','Scaled-down original inference: two features for basic operators, then four features, two heads and one layer for the encoder and decoder. LayerNorm uses γ = 1 and β = 0, and FFN biases are zero. Training and inference-disabled dropout are omitted; the paper’s scale is not reproduced.');
const added={},solutions={},references={};
function lesson(id,title,nodes,mission,description,why,takeaway,links,reference,options={}){
 const scalar=options.scalar,count=nodes.length;
 added[id]={id,title,nodes,sources:['R'],sourceDim:scalar?2:6,sinkDim:scalar?2:6,sampleCount:scalar?5:3,sequence:!scalar,
  sourceX:count>6?-10.3:-8.8,sinkX:count>6?10.3:9,zoom:count>6?1.22:1.07,boardWidth:count>6?26:23,boardDepth:count>6?14:11,
  sourceTitle:scalar?I18n.t('输入 X'):I18n.t('输入特征 X'),sourceNote:'',outputTitle:I18n.t('结果特征'),shape:scalar?'2 → 2':'3 × 2',
  mission,description,summary:description,concept:title,hints:[],rationale:{title:L('为什么需要这一步？','Why is this step needed?'),text:why,takeaway,scope:scalar?zeroBias:scope},...options};
 solutions[id]=links;references[id]=reference;
}
lesson('relu',L('让负数停在零','Stop negative values at zero'),['project','relu'],
 L('让负数变成 0，正数保持原样','Turn negatives into zero and keep positives unchanged'),
 L('连起投影和 ReLU，让每个负数变为 0，正数保持原值。五组输入都要匹配。','Wire the projection and ReLU so negative values become zero and positive values stay unchanged. Match all five inputs.'),
 L('连续两次线性投影可合并成一块矩阵：(XW1)W2 = X(W1W2)。ReLU 按输入的正负选择性地归零，使整个变换通常无法再合并成固定矩阵。','Two linear projections collapse into one matrix: (XW1)W2 = X(W1W2). ReLU selectively zeros values according to their sign, so the complete transformation generally cannot collapse into one fixed matrix.'),
 L('例如 [2, −3, 0, 1] 变成 [2, 0, 0, 1]。这是 2017 年原版 FFN 使用的激活；此时没有门投影。','For example, [2, −3, 0, 1] becomes [2, 0, 0, 1]. This is the activation used by the original 2017 FFN, without a gate projection.'),
 {project:['R'],relu:['project'],sink:'relu'},x=>M.linear(x,M.weights.project).map(relu),{scalar:true,sinkDim:4,shape:'2 → 4 → 4',sourceX:-6,sinkX:6,boardWidth:17,boardDepth:7,zoom:.83,outputTitle:L('ReLU 输出 A','ReLU output A')});
lesson('original-ffn',L('接出原版 FFN','Build the original FFN'),['project','relu','down'],
 L('让特征展开、变化，再回到两个数','Expand and transform the features, then return to two values'),
 L('把投影、ReLU 和降维投影组合起来，让出口的两个数匹配目标。每个词单独加工，本关还没有残差和归一化。','Combine projection, ReLU and down projection so the two output values match the target. Process each token separately; residuals and normalization come later.'),
 L('先展开、再变化、再收回的这段网络，叫前馈网络（FFN）。第一次投影产生更多特征组合，ReLU 加入非线性，第二次投影把结果组合回后续网络需要的宽度。','This expand–transform–return circuit is a feed-forward network (FFN). The first projection creates more feature combinations, ReLU adds nonlinearity, and the second projection returns the result to the width needed downstream.'),
 L('FFN(x) = ReLU(xW1 + b1)W2 + b2。本例 b1、b2 为 0。FFN 加工同一 token 的特征，Attention 才负责跨 token 取信息。','FFN(x) = ReLU(xW1 + b1)W2 + b2. Here b1 and b2 are zero. The FFN processes features within each token; Attention reads across tokens.'),
 {project:['R'],relu:['project'],down:['relu'],sink:'down'},classicFFN,{scalar:true,shape:'2 → 4 → 4 → 2'});
lesson('workshop',L('用上组好的 FFN','Put the assembled FFN to work'),['relu-ffn'],
 L('从组件架取出原版 FFN','Take the original FFN from the module shelf'),
 L('从组件架拖入“原版 FFN · ReLU”，连接输入和出口，试试一块模块能否复现分步搭建的结果。','Drag Original FFN · ReLU from the module shelf and connect its input and output. Can one module reproduce the expanded circuit?'),
 L('把一段网络收成一块模块，后面就不必反复重搭，这叫封装。外形变简单，内部仍依次执行线性展开、ReLU 和降维投影；权重和结果都没有变。','Putting a circuit into one reusable module is called encapsulation. The outside becomes simpler, but expansion, ReLU and down projection still run inside, with the same weights and results.'),
 L('本关练习从组件架拖入模块。双击 FFN 可查看内部三步。残差相加仍在 FFN 模块外。','Practice dragging a module from the module shelf. Double-click the FFN to inspect its three steps. Residual addition remains outside the FFN module.'),
 {'relu-ffn':['R'],sink:'relu-ffn'},classicFFN,{scalar:true,buildable:true,requirePlacement:true,placementTutorial:true,inventory:{'relu-ffn':1},maxModules:1,packages:['relu-ffn'],retiredCircuitKinds:['project'],sourceX:-6,sinkX:6,boardWidth:17,boardDepth:7,zoom:.83,outputTitle:L('FFN 输出 F','FFN output F')});
lesson('residual',L('留一条回来的路','Keep a path back'),['relu-ffn','add'],
 L('把 FFN 结果加回输入','Add the FFN result to the input'),
 L('给输入留一条不经过 FFN 的路，让它也能参与最后的结果。用“原版 FFN · ReLU”和“残差相加”试着搭出这条旁路。','Leave the input a path around the FFN so it can contribute to the final result too. Try building that bypass with Original FFN · ReLU and Residual addition.'),
 L('把一段计算的输出加回这段的输入，这叫残差连接：Y = X + FFN(X)。FFN 的输出不再替换原输入，而是作为修改量。训练时，支路只需学习加多少或减多少；不需要修改时，输出 0 就能让原输入继续传下去，不必再学一次复制输入。','Adding a circuit’s output back to its input is called a residual connection: Y = X + FFN(X). The FFN output becomes an update instead of a replacement. During training, the branch only needs to learn how much to add or subtract. A zero update passes the input onward without learning to copy it again.'),
 L('残差不保证数值不变：修改量可以增加、减少，甚至抵消原值。ReLU 仍在 FFN 内部；新加的只是外面的直连路径和相加。','A residual does not keep values unchanged: the update may increase, decrease or cancel them. ReLU remains inside the FFN; only the bypass and addition are new.'),
 {'relu-ffn':['R'],add:['R','relu-ffn'],sink:'add'},x=>plus(x,classicFFN(x)),{scalar:true,buildable:true,requirePlacement:true,inventory:{'relu-ffn':1,add:1},maxModules:2,packages:['relu-ffn'],retiredCircuitKinds:['project','down'],shape:'X + FFN(X) → 2'});
Object.assign(added.residual.rationale,{kind:'residual',input:[1,0]});
lesson('layernorm',L('LayerNorm：整理特征的尺度','LayerNorm: normalize feature scale'),['layernorm2'],
 L('让一组数围绕 0，整理彼此的差距','Center each vector around zero and rescale its differences'),
 L('将两维练习向量接入 LayerNorm。运行后，观察每组数如何减去自身均值，再除以标准差。输入由游戏直接给定，不是前一关的计算结果。','Wire the two-feature practice vectors into LayerNorm. Run the circuit to observe each vector being centered and divided by its standard deviation. The game supplies these inputs directly; they are not the previous lesson’s results.'),
 L('经过投影和残差相加，同一组特征可能整体偏大，或彼此差距变大。LayerNorm 去掉这组数的共同偏移和尺度，让后续层在较稳定的数值范围内处理相对差异，有助于训练稳定。每个 token 单独计算，不把整句话的数混在一起。','Projections and residual addition can shift a token’s features together or spread them farther apart. LayerNorm removes that shared offset and scale so later layers can work with relative differences in a more stable numerical range, helping training stability. Each token is normalized separately, not together with the whole sentence.'),
 L('[1, 3] → 减均值 2 → [−1, 1] → 除以标准差 1 → 约 [−1, 1]；[6, 10] 减均值 8、再除以标准差 2，也得到约 [−1, 1]。这些都是 LayerNorm 内部步骤。本关取 γ = 1、β = 0；真实模型还会学习每个特征的缩放 γ 和偏移 β。','[1, 3] → subtract mean 2 → [−1, 1] → divide by standard deviation 1 → approximately [−1, 1]. For [6, 10], subtract mean 8 and divide by standard deviation 2 to get the same result. These are LayerNorm’s internal steps. This lesson uses γ = 1, β = 0; a real model learns a scale γ and offset β for each feature.'),
 {layernorm2:['R'],sink:'layernorm2'},ln,{scalar:true,
  sourceTitle:L('练习向量 X','Practice vector X'),sourceNames:{R:L('练习向量 X','Practice vector X')},routeInput:L('练习向量 X','Practice vector X'),
  sourceNote:L('本关直接给定，未经过残差','Given here; before any residual addition'),outputTitle:L('LayerNorm 结果 Y','LayerNorm result Y')});
added.layernorm.rationale.scope=L('原版 Transformer 将残差相加的结果送入 LayerNorm，例如 X + FFN(X)。本关单独练习 LayerNorm。“先相加，再归一化”关再组合这两步。','The original Transformer feeds the residual sum into LayerNorm, for example X + FFN(X). This lesson isolates LayerNorm. Add first, normalize second combines the two steps.');
lesson('add-norm',L('先相加，再归一化','Add first, normalize second'),['relu-ffn','add','layernorm2'],
 L('先更新输入，再整理数值尺度','Update the input, then tidy up its scale'),
 L('将 FFN 的修改量加回原输入。再把相加结果送入 LayerNorm。FFN 内部计算保持不变。','Add the FFN update to the original input. Then send the sum to LayerNorm. Keep the FFN’s internal computation unchanged.'),
 L('一条路保留输入，一条路带回修改量；相加后再用 LayerNorm 整理尺度。合起来就是 LayerNorm(X + FFN(X))，原版中常称为 Add & Norm。残差连接和归一化各有分工，不能互相代替。','One path carries the input, another brings an update; LayerNorm adjusts the scale after they are added. Together this is LayerNorm(X + FFN(X)), often called Add & Norm in the original model. Residual connections and normalization have different jobs.'),
 L('2017 年原版使用 Post-LN：先子层计算，再加残差，最后 LayerNorm。把归一化提前会改变计算，现代 Pre-LN 留到扩展章。','The 2017 model uses Post-LN: sublayer computation, residual addition, then LayerNorm. Moving normalization earlier changes the computation; modern Pre-LN comes in the extensions.'),
 {'relu-ffn':['R'],add:['R','relu-ffn'],layernorm2:['add'],sink:'layernorm2'},x=>ln(plus(x,classicFFN(x))),{scalar:true,
  sourceTitle:L('练习向量 X','Practice vector X'),sourceNames:{R:L('练习向量 X','Practice vector X')},
  sourceNote:L('本关给定，用于比较残差效果','Given here to compare residual effects')});
lesson('positions',L('同一个词也有不同位置','The same word at different positions'),['embedding','position'],
 L('让同一个词在不同位置带上不同的数','Give the same token different values at different positions'),
 L('查表取词向量，再加入原版的正弦位置编码。改变词的顺序，位置相加后的特征也应改变。','Look up embeddings, then add the original sinusoidal positions. Reordering tokens must change the position-enriched features.'),
 L('同一个词换个位置，意思可能就变了；仅靠词向量却看不出它在哪里。把确定的位置模式加进输入，这叫位置编码。原版使用正弦和余弦，后续 Q/K/V 就能同时利用词和位置。','Moving a token can change the meaning, but its embedding alone does not say where it is. Adding a deterministic position pattern is called position encoding. The original uses sine and cosine, letting later Q/K/V projections use both the token and its position.'),
 L('本例只有一对特征：PE(p) = [sin(p), cos(p)]。原版更宽的向量使用多组频率；这是相加，不是 RoPE 对 Q/K 做旋转。','This example has one feature pair: PE(p) = [sin(p), cos(p)]. Wider original models use multiple frequencies. This adds to embeddings; it does not rotate Q/K as RoPE does.'),
 {embedding:['R'],position:['embedding'],sink:'position'},ids=>position(embed(ids)),{outputArrivalHold:450,sourceDim:3,sourceTitle:'Token ID',shape:'3 ID → 3 × 2'});
lesson('weighted-sum',L('按比例取回内容','Read content in proportion'),['original-weights','v','aggregate'],
 L('按分好的比例，把各词的内容取回来','Bring back token content in the assigned proportions'),
 L('比例已经会算了，但内容还没有取回来。复用“已学的打分与 Softmax”，再配合 V 和汇合模块，让每个位置收到按比例组合的两个数。','The proportions are ready, but the content has not arrived. Reuse Scores and Softmax alongside V and aggregation so each position receives two values combining content in those proportions.'),
 L('分数只回答更关注哪里，还没有拿回信息。按比例取一份份 V，再把贡献加起来，这叫加权求和，写作 O = A × V。A 是注意力权重，V 提供内容；当前词由此得到上下文特征。','Scores only say where to attend; they have not retrieved information. Taking a share of each V and adding the contributions is a weighted sum, written O = A × V. A holds attention weights and V supplies content, giving the current token context features.'),
 L('取 [2, 0] 的 25% 和 [0, 4] 的 75%，得到 [0.5, 3]。这里在不同 token 之间加权求和，和后面门控的逐元素相乘不同。','Taking 25% of [2, 0] and 75% of [0, 4] gives [0.5, 3]. This weighted sum across tokens differs from the elementwise multiplication used in later gating.'),
 {'original-weights':['R'],v:['R'],aggregate:['original-weights','v'],sink:'aggregate'},x=>mix(unmaskedWeights(x),mat(x,M.weights.v)),{attentionReadout:true});
lesson('self-attention',L('拼成单头自注意力','Assemble single-head self-attention'),['q','k','v','score','softmax','aggregate','output'],
 L('让每个词从整句话里取回信息','Let each token retrieve information from the sentence'),
 L('匹配谁、各取多少、取回什么，现在把这些步骤连成一段完整计算。自由选取组件架里的模块，让每个词得到目标中的两个输出特征；所有位置互相可见。','Whom to match, how much to take, and what to bring back: connect these steps into one circuit. Choose modules from the shelf so each token gets the target pair of output features. All positions can read one another.'),
 L('同一句话里的词互相读取信息，这叫自注意力。Q/K 决定从谁读取，Softmax 把分数变成比例，V 提供内容，Wo 把汇合结果组合到网络需要的特征空间。本关用一套这样的计算，也就是一个注意力头。','Tokens reading from the same sentence is called self-attention. Q/K decide whom to read, Softmax turns scores into proportions, V supplies content, and Wo maps the aggregate into the feature space needed downstream. One such computation is one attention head.'),
 L('这是原版编码器的单头计算，不含 RoPE、因果遮罩或 FFN。双头注意力让两组投影权重分别计算，再拼接结果。','This is one head of original encoder attention, without RoPE, a causal mask or an FFN. Two-head attention uses separate projection weights for each head, then concatenates their results.'),
 {q:['R'],k:['R'],v:['R'],score:['q','k'],softmax:['score'],aggregate:['softmax','v'],output:['aggregate'],sink:'output'},x=>singleAttention(x),{buildable:true,maxModules:9,attentionReadout:true});
lesson('multihead',L('让两个头各看一遍','Let two heads look independently'),['head-a','head-b','concat','output'],
 L('让两种读取方式共同更新每个词','Use two ways of reading to update each token'),
 L('同一句话交给头 A 和头 B，各自会带回什么？组合它们的结果，让出口仍是每个词两个特征。两头用同一份输入，但不共用投影权重。','What will heads A and B bring back from the same sentence? Combine their results while keeping two output features per token. Both heads read the same input but use different projection weights.'),
 L('一个头有一套读取比例，多个头就能各用自己的投影权重形成不同的读取方式，这叫多头注意力。把同一个词的两头结果并排放在一起，叫按特征拼接；Wo 再组合它们带回的信息。','One head has one reading pattern; several heads can use their own projection weights to form different patterns. This is multi-head attention. Placing both results side by side for the same token is feature concatenation; Wo then recombines what they brought back.'),
 L('本例 d_model = 2、头数 = 2，每头 d_k = d_v = 1。缩放因此除以 √1。拼接沿特征方向进行，三个 token 仍然是三个。','Here d_model = 2 with two heads, so each has d_k = d_v = 1 and scales by √1. Concatenation is along features; three tokens remain three tokens.'),
 {'head-a':['R'],'head-b':['R'],concat:['head-a','head-b'],output:['concat'],sink:'output'},x=>multi(x),{shapes:{3:'3 × 1'}});
lesson('encoder-layer',L('接成一层原版编码器','Build one original encoder layer'),['self-heads','residual1','ln-attn','original-ffn-block','residual2','ln-ffn'],
 L('先交流，再加工','Exchange, then process'),
 L('用双头自注意力让词之间交换信息，再用原版 FFN 加工每个词。两段各自沿用学过的“先相加，再归一化”。','Use two-head self-attention to exchange information between tokens, then an original FFN to process each token. Both stages reuse the familiar “add first, normalize second” pattern.'),
 L('输入 X 已包含词向量和位置：仍是“我 / 爱 / 猫”，每个词有 4 个特征。这一层更新的是这些特征。','Input X already contains embeddings and positions: still I / love / cats, with four features per token. This layer updates those features.'),
 L('两段都把计算结果当作修改量，加回本段输入，再用 LayerNorm 整理特征尺度。第二段收到的是交流后的特征，它的残差也从这里取。','Each stage adds its computed update to that stage’s input, then uses LayerNorm to adjust the feature scale. The second stage receives the features after exchange, so its residual starts there too.'),
 {'self-heads':['R'],residual1:['R','self-heads'],'ln-attn':['residual1'],'original-ffn-block':['ln-attn'],residual2:['ln-attn','original-ffn-block'],'ln-ffn':['residual2'],sink:'ln-ffn'},encoder,{buildable:true,maxModules:8,positioned:true});
const encoderLesson=added['encoder-layer'];
encoderLesson.rationale.kind='encoder-layer';
encoderLesson.rationale.title=L('两段，每段都要收尾','Two stages, each with its own finish');
encoderLesson.description=L('让词先互相交流，再各自加工得到的新信息。用熟悉的双头自注意力和原版 FFN 搭成两段，想想每一段该怎样保留输入、整理尺度。出口仍是三个词，每词四个特征。','Let tokens exchange information, then process what each has received. Build two stages with two-head self-attention and the original FFN. How should each stage retain its input and adjust its scale? The output still has three tokens with four features each.');
encoderLesson.rationale.scope=L('沿用 2017 原版的 Post-LN 顺序。输入、两段之间、输出都是 3 个词 × 4 个特征；整句互相可见。','Uses the original 2017 Post-LN order. The input, intermediate state and output all have three tokens × four features; every token can read the whole sentence.');
// Named wrappers keep their sublayer roles and independent shelf stock.
encoderLesson.shelf={
 featureWidth:12,
 order:['self-heads','encoder-add','ln-attn','original-ffn-block','ffn-add','ln-ffn','wide-position'],
 current:L('四维模块','Four-feature modules'),
 earlier:L('已学模块','Previously learned modules'),
 note:L('已学模块仍可查看和拖入棋盘。两维端口不能直接连接本关的四维特征。','You can still inspect learned modules and drag them onto the board. Two-feature ports cannot connect directly to this lesson’s four-feature tensors.')
};
encoderLesson.packageLabel=L('已学模块的四维版本','Four-feature version of a learned module');
M.encoderGuide={
 tokens:['我','爱','猫'].map(word=>I18n.t(word)),
 exchange:{title:L('自注意力：读取其他词的特征','Self-attention: read features across tokens'),tag:labels.attn,text:L('每个词按注意力比例读取整句话的内容。例如，“猫”的特征会参与更新“爱”的特征。','Each token reads from the whole sentence in attention-weighted proportions. For example, features of cats contribute to updating love.')},
 process:{title:L('FFN：分别处理每个词的特征','FFN: process each token’s features'),tag:L('原版 FFN','Original FFN'),text:L('再分别加工交流后的每个词：4 → 8 → ReLU → 4。同一套权重用在所有词上，这一步不再从别的词取数。','Process each token after the exchange: 4 → 8 → ReLU → 4. All tokens use the same weights; this step does not read other tokens.')},
 stageNotes:[
  {path:L('双头自注意力 → 残差相加 → LayerNorm','Two-head attention → Residual addition → LayerNorm'),source:L('相加的另一端：原始输入 X。','Other input to addition: original input X.'),finish:L('这一段归一化后，才交给 FFN。','Normalize this stage before passing it to the FFN.')},
  {path:L('原版 FFN → 残差相加 → LayerNorm','Original FFN → Residual addition → LayerNorm'),source:L('相加的另一端：第一段 LayerNorm 的输出。','Other input to addition: the first LayerNorm’s output.'),finish:L('将最后的相加结果归一化，再接到关卡出口。','Normalize the final sum, then wire it to the lesson outlet.')}
 ],
 pattern:L('两段都复用同一种结构','Both stages reuse one pattern'),
 operations:[L('计算修改量','Compute an update'),L('残差相加','Residual addition'),'LayerNorm'],
 formulaTitle:L('用公式对照两段','Compare the two stages in formulas'),
 first:L('X1：自注意力、残差和 LayerNorm 的输出','X1: output of self-attention, residual addition and LayerNorm'),
 second:L('Y：FFN、残差和 LayerNorm 的输出','Y: output of the FFN, residual addition and LayerNorm'),
 formulaNote:L('LN 就是 LayerNorm。Attention 使用 Attention 残差和对应的 LayerNorm。FFN 使用 FFN 残差和对应的 LayerNorm。两段运算形式相同，但模块归属不同子层。','LN means LayerNorm. Attention uses its own residual and LayerNorm modules. The FFN uses its own residual and LayerNorm modules. Both stages use the same operation pattern, but the modules belong to different sublayers.')
};
I18n.prepare(M.encoderGuide);
lesson('masked-multihead',L('把遮罩放进两个头','Put the mask inside both heads'),
 ['masked-a-q','masked-a-k','masked-a-v','score','mask','softmax','aggregate','masked-b-q','masked-b-k','masked-b-v','score','mask','softmax','aggregate','wide-concat','original-output'],
 L('让两个头都只能看见自己和前面的词','Let both heads see only themselves and earlier tokens'),
 L('搭建头 A 和头 B，让两头都不能读取未来位置。两头读取同一份 X，但各有自己的 Q、K、V 权重。完成后组合两头结果。Q 和 K 不是两个头；A、B 也不是两组词。','Build heads A and B so neither can read future positions. Both read the same X, but each has its own Q, K and V weights. Combine the completed head outputs. Q and K are not two heads; A and B are not two groups of tokens.'),
 L('每个头先用 Q/K 计算分数，再用 Softmax 得到读取比例。V 提供要读取的内容。按比例汇合 V，得到该头的结果。两头使用不同的投影权重，因此可以得到不同的读取比例。','Each head computes Q/K scores, then applies Softmax to obtain reading weights. V supplies the content. Aggregate V using those weights to produce the head’s result. Different projection weights can produce different reading patterns.'),
 L('本关再给两套计算都加上“不能看未来”的限制：每个头在自己的 Softmax 前遮住未来分数。否则即使 A 遮住了“猫”，B 仍可能提前读到它。两头完成后，把结果拼接并经过 Wo，才收成新的双头因果注意力。','Add a no-looking-ahead restriction to both computations: mask future scores before each head’s own Softmax. Otherwise, even if A masks cats, B can still read it early. Concatenate both completed results and apply Wo to form the new two-head causal attention module.'),
 {'masked-a-q':['R'],'masked-a-k':['R'],'masked-a-v':['R'],score:['masked-a-q','masked-a-k'],mask:['score'],softmax:['mask'],aggregate:['softmax','masked-a-v'],
 'masked-b-q':['R'],'masked-b-k':['R'],'masked-b-v':['R'],'score-2':['masked-b-q','masked-b-k'],'mask-2':['score-2'],'softmax-2':['mask-2'],'aggregate-2':['softmax-2','masked-b-v'],'wide-concat':['aggregate','aggregate-2'],'original-output':['wide-concat'],sink:'original-output'},
 x=>multi(x,x,true),{sourceDim:12,sinkDim:12,shape:'3 × 4 → 3 × 4',wide:true,positioned:true,buildable:true,maxModules:18,boardWidth:32,boardDepth:18,sourceX:-13,sinkX:13,zoom:1.5,outputTitle:L('因果注意力输出','Causal attention output')});
added['masked-multihead'].rationale.kind='masked-heads';
added['masked-multihead'].rationale.title=L('一个头，里面有什么？','What is inside one head?');
added['masked-multihead'].rationale.scope='';
M.maskedHeadsGuide={
 input:L('同一份 X · 每个词 4 个数','Same X · 4 numbers per token'),
 heads:[L('头 A','Head A'),L('头 B','Head B')],
 result:L('每个词得到 2 个数','2 output numbers per token'),
 join:L('拼接两份结果：2 + 2 = 4 个数','Concatenate results: 2 + 2 = 4 numbers'),
 output:L('Wo 重组这 4 个数','Wo recombines these 4 numbers'),
 widthTitle:L('两个头，是把输入切成两半吗？','Do two heads split the input in half?'),
 width:L('不是。每个头的 Q/K/V 投影都读取完整的 4 个数。各自乘 4 × 2 权重后，得到 2 个数。两头汇合结果按 A 在前、B 在后拼接，再乘 4 × 4 的 Wo。残差和 LayerNorm 不在本模块内。','No. Each head’s Q/K/V projections read all four input values. Each applies its own 4 × 2 weights to produce two values. Concatenate the aggregate results with A before B, then apply the 4 × 4 Wo. Residual addition and LayerNorm are outside this module.'),
 oldTitle:L('组件架中的“双头自注意力”有何不同？','How does the module shelf’s two-head self-attention module differ?'),
 old:L('该模块已包含两个头、拼接和 Wo，但没有因果遮罩。不要把它作为 Q/K/V 的前置模块，否则会多算一轮注意力。本关在两个展开的头内加入遮罩，再封装成新模块。','That module contains both heads, concatenation and Wo, but no causal mask. Do not place it before Q/K/V; that would add another attention computation. This lesson adds masks inside the two expanded heads, then packages the result as a new module.'),
 exampleTitle:L('看一个例子：同一个“爱”，两头分别参考谁？','An example: how do the heads read for love?'),
 exampleNote:L('这是固定输入“我 / 爱 / 猫”在“爱”位置算出的参考比例。两套权重得到的比例不同，但“猫”在未来，两边都给它 0。头 A 的比例用于汇合 V_A，头 B 的比例用于汇合 V_B。','These are the reading weights at love for the fixed input I / love / cats. The two weight sets give different proportions, but the token “cats” is in the future, so both assign it zero. Head A’s proportions aggregate V_A; head B’s proportions aggregate V_B.')
};
I18n.prepare(M.maskedHeadsGuide);
lesson('decoder-self',L('先整理已经生成的词','First, process the words generated so far'),['masked-heads','residual1','ln-attn'],
 L('先让已有的词参考前文','Let the existing tokens use their earlier context'),
 L('接下来要续写文字，先整理已有的词。用双头因果注意力处理目标前文，再沿用“先相加，再归一化”的收尾。出口是目标特征 D，这一段还不读源句。','Before continuing the text, process the tokens already there. Use two-head causal attention for the target context and the familiar “add first, normalize second” finish. The output is target features D; this stage does not read the source yet.'),
 L('以翻译“我爱猫”为例，假设已生成 I love。预测下一个词前，love 先读取已有的 I，形成目标上下文。每个位置只读取自己和前面的词。这一步不读源句；交叉注意力负责读取源句。','Suppose a translation model has produced the target prefix I love. Before predicting the next token, love reads the earlier I to form target context. Each position reads only itself and earlier tokens. This step does not read the source; cross-attention does.'),
 L('目标输入表示已经给出的前缀，预测对齐下一词；不是把待预测的答案原样喂给当前位置。这里演示计算，不进行训练。','Target inputs represent the available prefix and predictions align with the next token. The answer is not fed into the position predicting it. This demonstrates computation, not training.'),
 {'masked-heads':['R'],residual1:['R','masked-heads'],'ln-attn':['residual1'],sink:'ln-attn'},decoderSelf,{positioned:true});
const dual={dualSource:true,sources:['R','Z'],secondaryTokenIds:sourceIds,sourcePorts:{R:'D',Z:'E'},sourceNames:{R:L('目标前缀 D','Target prefix D'),Z:L('编码器输出 E','Encoder output E')},sourceTitle:L('目标前缀 D','Target prefix D'),secondaryTitle:L('源序列 · 猫 / 我 / 爱','Source · cats / I / love'),positioned:true};
lesson('cross-attention',L('再带着上下文读取源句','Then, use that context to read the source'),['q','k','v','score','softmax','aggregate','output'],
 L('带着目标前文，从源句取回信息','Use the target context to retrieve source information'),
 L('下方 D 带着目标前文，上方 E 保存编码器整理的源句。试着让这两路在注意力中相遇：由目标侧发起查询，从源句取回内容。出口仍对应目标位置，不是源句位置。','The lower input D carries target context; the upper input E holds the encoded source. Bring these paths together in attention: let the target ask and retrieve content from the source. Output positions still belong to the target, not the source.'),
 L('只有 I love 这段目标前文，还不知道应该翻译成“爱猫”还是“爱狗”。上一关只整理目标前文，这一关让它去读编码器保存的原句。例如源句是“我爱猫”，交叉注意力就能提供与猫有关的信息，供后面的 FFN 和词表预测使用。游戏用固定小矩阵演示这种读取，不是真正的翻译模型。','The target prefix I love alone cannot tell whether the source says cats or dogs. The previous lesson processed that prefix; this one lets it read the source encoded in E. Cross-attention can retrieve cat-related information for the later FFN and vocabulary prediction. The game illustrates retrieval with fixed small matrices, not a trained translator.'),
 L('从一份序列查询另一份序列，这叫交叉注意力。D 是因果自注意力、残差和 LayerNorm 的结果：Q 从 D 投影，K/V 从编码器输出 E 投影。三个输出仍对应目标位置。源句已经完整给定，不套用目标侧的因果遮罩。本关先练单头，完整解码器使用双头。','Querying one sequence with another is called cross-attention. D comes from causal self-attention, residual addition and LayerNorm: project Q from D and K/V from encoder output E. The three outputs still belong to target positions. The whole source is available, without the target-side causal mask. Practice one head here; the full decoder uses two.'),
 {q:['R'],k:['Z'],v:['Z'],score:['q','k'],softmax:['score'],aggregate:['softmax','v'],output:['aggregate'],sink:'output'},x=>singleAttention(x,memory()),{...dual,sourceTitle:L('目标特征 D','Target features D'),sourceNote:L('因果自注意力子层的 LayerNorm 输出','LayerNorm output from the causal self-attention sublayer'),sourceNames:{R:L('目标特征 D','Target features D'),Z:dual.sourceNames.Z},targetFlowLabel:L('目标特征','Target features'),buildable:true,maxModules:9});
lesson('decoder-layer',L('加入 FFN，接成完整解码器层','Add the FFN and assemble a decoder layer'),['masked-heads','residual1','ln-attn','cross-heads','cross-add','ln-cross','original-ffn-block','residual2','ln-ffn'],
 L('先读目标前缀，再读源句，最后加工特征','Read the prefix, read the source, then process features'),
 L('目标前文和源句里的信息都准备好了，把“读取”和“加工”接成完整一层。使用双头因果注意力、双头交叉注意力和原版 FFN；三段都需要自己的残差与 LayerNorm。','The target context and source information are ready. Put reading and processing together into a complete layer with two-head causal attention, two-head cross-attention and the original FFN. Each stage needs its own residual and LayerNorm.'),
 L('前两关解决“从哪些词取信息”：先看目标前文，再读源句。FFN 接着对每个目标位置的特征做展开、ReLU 和压回，学习非线性组合；它不再从其他位置取信息。这样一层既能交流信息，也能加工交流后的特征。','The previous lessons decide which tokens to read: target context first, then the source. The FFN then expands each target feature vector, applies ReLU and projects it back, forming nonlinear combinations without reading other positions. A layer therefore both exchanges information and transforms the resulting features.'),
 L('每次残差都加回当前子层的输入。原版是三次 Post-LN；不要跳过交叉注意力，也不要把所有残差都接回最初的 D。','Each residual adds back that sublayer’s input. The original has three Post-LN steps: do not skip cross-attention or connect every residual to the initial D.'),
 {'masked-heads':['R'],residual1:['R','masked-heads'],'ln-attn':['residual1'],'cross-heads':['ln-attn','Z'],'cross-add':['ln-attn','cross-heads'],'ln-cross':['cross-add'],'original-ffn-block':['ln-cross'],residual2:['ln-cross','original-ffn-block'],'ln-ffn':['residual2'],sink:'ln-ffn'},x=>decoder(x,memory()),{...dual,buildable:true,maxModules:11});
lesson('vocabulary-scores',L('先给候选词打分','First, score the candidate tokens'),['original-head'],
 L('看看六个候选词各能得多少分','Find out how the six candidate tokens score'),
 L('输入是解码器最后一个 LayerNorm 的输出，每个目标位置有四个特征。放置词表投影并连线，计算每个位置的六个候选词分数。本关不选词。','The input is the decoder’s final LayerNorm output, with four features per target position. Place and wire vocabulary projection to compute six candidate scores at each position. This lesson does not select tokens.'),
 L('解码器输出的四个数是特征坐标，不是“猫”或“狗”的分数。词表投影的每列权重对应一个候选词。对四个特征加权求和，再加偏置，得到该词的分数。六列权重产生六个候选分数。','The decoder’s four values are feature coordinates, not scores for cats or dogs. Each vocabulary-projection column corresponds to one candidate. Take a weighted sum of all four features and add a bias to obtain that candidate’s score. Six columns produce six candidate scores.'),
 L('L = D × W_vocab + b：3 × 4 乘 4 × 6，得到 3 × 6。每个位置独立打分。六列依次对应“我、爱、猫、你、恨、狗”。分数不是概率。完整模型组装关会复用此模块。','L = D × W_vocab + b: 3 × 4 times 4 × 6 gives 3 × 6. Each position is scored independently. The six columns are I, love, cats, you, hate and dogs. Scores are not probabilities. The full-model assembly lesson reuses this module.'),
 {'original-head':['R'],sink:'original-head'},vocabulary,{sourceDim:12,sinkDim:18,wide:true,sourceTitle:L('解码器输出 D','Decoder output D'),sourceNote:L('解码器最后一个 LayerNorm 的输出','Output of the decoder’s final LayerNorm'),outputTitle:L('候选词分数 L','Candidate-token scores L'),shape:'3 × 4 → 3 × 6',buildable:true,maxModules:3,boardWidth:18,boardDepth:8,sourceX:-6.5,sinkX:6.5,zoom:.9,startWithModules:true,saveAliases:{lmhead:'original-head'}});
lesson('encoder-decoder',L('把原版两半接起来','Connect both halves of the original'),['source-embed','target-embed','original-encoder','original-decoder','lmhead'],
 L('从两份文字编号，一路算到候选词分数','Go from two sets of token IDs to candidate scores'),
 L('一边是完整源句，一边是还要继续写的目标前缀。把熟悉的词向量与位置、编码器、解码器和词表投影连起来，让两份 ID 最终得到每个目标位置的六个候选分数。本关还不选词。','One side has the whole source; the other has a target prefix waiting to be continued. Connect embeddings and positions, the encoder, decoder and vocabulary projection so both sets of IDs lead to six candidate scores per target position. This lesson does not select tokens yet.'),
 L('原版 Transformer 是编码器—解码器模型。编码器整理已给定的源句，解码器结合源句与目标前缀预测下一词；并非只有一条自注意力链。','The original Transformer is an encoder–decoder model. The encoder processes the given source; the decoder combines it with the target prefix to predict the next token. It is more than one self-attention chain.'),
 L('输出是每个目标位置的六个候选分数。真实生成会做 Softmax 和选词，并追加到目标前缀；本例只验证结构与数值，不宣称这些手工权重能翻译。','The output contains six candidate scores per target position. Generation applies Softmax and token selection, then appends to the prefix. This checks structure and arithmetic; these hand-written weights are not a translation model.'),
 {'source-embed':['Z'],'target-embed':['R'],'original-encoder':['source-embed'],'original-decoder':['target-embed','original-encoder'],lmhead:['original-decoder'],sink:'lmhead'},ids=>vocabulary(decoder(position(embed(ids,4)),memory())),{...dual,positioned:false,sourceDim:3,sinkDim:18,sourcePorts:{R:'D ID',Z:'S ID'},sourceNames:{R:L('目标前缀 ID','Target prefix IDs'),Z:L('源序列 ID','Source sequence IDs')},sourceTitle:L('目标前缀 ID','Target prefix IDs'),shape:'2 × 3 ID → 3 × 6',buildable:true,maxModules:7});
lesson('vocabulary',L('从分数中选出词','Select tokens from scores'),['original-head','pick','wide-invert'],
 L('让每个位置留下得分最高的词','Keep the highest-scoring token at each position'),
 L('复用解码器的四维输出和已学的词表投影。在候选分数后加入选词，将最高分对应的词表 ID 送到关卡出口。','Reuse the decoder’s four-feature output and the familiar vocabulary projection. Add token selection after candidate scoring to send the highest-scoring vocabulary ID to the lesson outlet.'),
 L('上一关的出口是候选分数，还不是词。取最高分把每组分数变成一个词表编号；每个目标位置得到一个预测。生成下一词时采用最后一个位置的预测，再追加到前缀。','The previous outlet contains candidate scores, not tokens. Taking the maximum converts each score group into one vocabulary ID, one prediction per target position. To generate the next token, use the final position’s prediction and append it to the prefix.'),
 L('取最高分与取 Softmax 后的最大概率会选中同一个词，所以本关直接按分数选择。手工教学权重不保证预测有语言意义，也不要求输出等于输入；三个位置都要按实际分数正确选词。','The largest logit and largest Softmax probability select the same token, so this lesson selects directly from scores. The hand-written teaching weights do not guarantee meaningful predictions or a copy of the input. Select the correct candidate at all three positions.'),
 {'original-head':['R'],pick:['original-head'],'wide-invert':['R'],sink:'pick'},v=>M.operations.pick([vocabulary(v)]),{sourceDim:12,sinkDim:3,wide:true,sourceTitle:L('解码器输出 D','Decoder output D'),sourceNote:L('与词表打分练习相同的解码器特征','The same decoder features as the vocabulary-scoring exercise'),outputTitle:L('预测 Token ID','Predicted token IDs'),shape:'3 × 4 → 3 × 6 → 3 ID',tokenOutput:true,buildable:true,maxModules:5,saveAliases:{lmhead:'original-head',invert:'wide-invert'}});
for(const id of ['vocabulary-scores','vocabulary'])Object.assign(added[id],{sourceNames:{R:added[id].sourceTitle},sourcePorts:{R:'D'}});
const readoutWidthNote=L('FFN 内部的 4 → 8 → 4，是把临时展开的特征收回模型宽度。这里的 4 → 6 则是为六个词各算一个分数：4 是每个位置的特征数，6 是词表大小，前面的 3 个位置始终不变。','Inside the FFN, 4 → 8 → 4 returns expanded features to the model width. Here, 4 → 6 computes one score for each of six tokens: 4 is the feature count per position, 6 is the vocabulary size, and the three positions stay unchanged.');
const withReadoutWidth=text=>L(text+' '+readoutWidthNote,globalThis.AtelierEnglish[text]+' '+globalThis.AtelierEnglish[readoutWidthNote]);
for(const id of ['vocabulary-scores','vocabulary'])added[id].rationale.text=withReadoutWidth(added[id].rationale.text);
for(const id of ['vocabulary-scores','encoder-decoder'])Object.assign(added[id],{
 outputTitle:L('候选词分数 L','Candidate-token scores L'),
 outputNote:L('每个位置的六个候选词分数','Six candidate scores per position'),
 resultNote:L('这里已经是候选词分数，还没有选词。每个目标位置分别有六个分数，下一步才从中选择 token。','These are candidate-token scores, with no token selected yet. Each target position has six scores; token selection comes next.')
});
lesson('wider-features',L('词不变，特征变宽','Same tokens, wider features'),['wide-embedding','wide-position'],
 L('还是三个词，这次让每个词带着四个数','Keep the three tokens, now with four values each'),
 L('前面每个 token 已经用两个数表示。本关重新从 token ID 查四维词表，改用四个数共同表示一个 token，再加入位置信息。三个 token 仍是三个，输出是 3 × 4。','Earlier, each token was already represented by two numbers. Here, start from token IDs and look up a four-feature embedding table, then add positions. Four numbers jointly represent each token. Three tokens remain three; the output is 3 × 4.'),
 L('维度就是一个向量里数值的个数。每一维是一个特征坐标，模型会学习怎样组合这些坐标来表达词和上下文的关系；不必把某一维固定理解成“动物”或“情感”。真实词向量由训练学习，本游戏用固定数值演示。','A vector’s dimension is its number of coordinates. The model learns to combine these feature coordinates to represent tokens and their context; a coordinate need not mean a fixed property such as “animal” or “emotion”. Real embeddings are learned during training; the game uses fixed teaching values.'),
 L('前面用两维是为了方便看计算。接下来一层里有多次 LayerNorm；两个数归一化后常接近 [1, −1] 或 [−1, 1]，许多差异看不出来。换成四维能保留更多相对差异，便于观察 Attention 和 FFN 的作用。','Two features made the earlier arithmetic easy to follow. The next layers apply LayerNorm repeatedly; normalizing two numbers often leaves approximately [1, −1] or [−1, 1], hiding many differences. Four coordinates retain more relative variation, making changes from Attention and FFNs easier to see.'),
 {'wide-embedding':['R'],'wide-position':['wide-embedding'],sink:'wide-position'},ids=>position(embed(ids,4)),{sourceDim:3,sinkDim:12,sourceTitle:'Token ID',shape:'3 ID → 3 × 4',wide:true});
Object.assign(added['wider-features'].rationale,{
 kind:'embedding-width',title:L('四个数，共同表示一个 token','Four coordinates, one token'),
 scope:L('这里是教学上换用更宽的词表，不是把上关的两维输出临时扩成四维。真实 Transformer 通常从第一张词向量表起就采用选定的模型宽度；原版 base 为 512 维。FFN 内部先展开再收回，是另一种宽度变化。','This switches to a wider embedding table for teaching; it does not expand the previous lesson’s two-feature output. A real Transformer typically uses its chosen model width from the initial embedding table; the original base model used 512 features. Expansion and contraction inside an FFN are a separate width change.')
});
M.embeddingWidthExample={
 token:M.math.vocab[2],beforeLabel:L('之前 · 两维词表','Earlier · two-feature table'),afterLabel:L('本关 · 四维词表','Here · four-feature table'),
 before:[...M.math.emb[2]],after:[...wide.embedding[2]]
};
I18n.prepare(M.embeddingWidthExample);
// Keep the early arithmetic tiny; complete Post-LN stacks need enough features
// to retain observable differences instead of collapsing to two signed values.
const wideAliases={residual1:'encoder-add',residual2:'ffn-add',q:'original-q',k:'original-k',v:'original-v',score:'original-score',aggregate:'original-aggregate',output:'original-output',lmhead:'original-head'};
for(const id of ['encoder-layer','decoder-self','cross-attention','decoder-layer','encoder-decoder']){
 const l=added[id],rename=k=>wideAliases[k]||k;l.wide=true;l.nodes=l.nodes.map(rename);
 const links=solutions[id];solutions[id]=Object.fromEntries(Object.entries(links).map(([k,v])=>[rename(k),Array.isArray(v)?v.map(rename):rename(v)]));
 if(id!=='encoder-decoder'){l.sourceDim=l.sinkDim=12;l.shape='3 × 4';}
}
lesson('pack-scores',L('用上组好的打分模块','Put the scoring module to work'),['original-weights'],
 L('用一块模块，算出同样的读取比例','Get the same reading proportions with one module'),
 L('把算分数、分比例的几步收在一块，能得到同样的结果吗？将 X 接入“已学的打分与 Softmax”，让出口复现注意力权重 A。也可用基础模块搭建；本关不生成新模块。','Can one module do the same scoring and allocation? Wire X into Scores and Softmax to reproduce attention weights A at the outlet. Familiar basic modules work too; this lesson creates no new module.'),
 L('关卡出口左侧列出查询词，上方列出被读取的词。每格表示读取对应词 V 向量的比例。同一查询的三个比例非负且合计为 1。A 随输入计算，不是固定的模型参数。V 汇合使用这些比例，但不在本关计算。','The lesson outlet lists query tokens on the left and tokens to read across the top. Each cell gives the proportion for reading one token’s V vector. Each query’s three proportions are nonnegative and sum to 1. A is computed from the input, not a fixed model parameter. V aggregation uses these proportions but is outside this lesson.'),
 L('例如输入“我 / 爱 / 猫”时，“我”对应约 [0.31, 0.26, 0.43]。汇合时，V我、V爱、V猫所占比例约为 31%、26%、43%。这里显示比例，不显示汇合后的向量。','For input I / love / cats, the query I has approximately [0.31, 0.26, 0.43]. During aggregation, V_I, V_love and V_cats contribute about 31%, 26% and 43%. This outlet shows proportions, not the aggregated vector.'),
 {'original-weights':['R'],sink:'original-weights'},unmaskedWeights,{sinkDim:9,shape:'3 × 2 → 3 × 3',buildable:true,maxModules:6,attentionReadout:true,packages:['original-weights'],
  sourceTitle:L('输入向量矩阵 X','Input vector matrix X'),sourceNames:{R:L('输入矩阵 X','Input matrix X')},routeInput:L('输入矩阵 X','Input matrix X'),sourceNote:L('与上一关相同的 X，重新计算 A','The same X as before; compute A again'),
  outputTitle:L('注意力权重矩阵 A','Attention-weight matrix A'),outputNote:L('每个数是读取对应词 V 的比例','Each value is a proportion for reading a token’s V'),outputPairAxes:true});
added['pack-scores'].rationale.title=L('出口的九个数在分配什么？','What do the nine output values allocate?');
Object.assign(M.levels,added);
M.levels['original-ffn'].summary=L('把展开、变化和收回连成原版 FFN，单独加工每个词。','Connect expansion, transformation and projection back into an original FFN for each token.');
M.levels.workshop.summary=L('从组件架取出 ReLU FFN，一块模块复现分步搭建的结果。','Take a ReLU FFN from the module shelf and reproduce the expanded circuit with one module.');
M.levels['pack-scores'].summary=L('用组好的打分模块复现读取比例；内容汇合留给下一步。','Reuse the scoring module to reproduce the reading proportions; content retrieval comes next.');
M.levels['relu'].concept=L('ReLU 激活','ReLU activation');
M.levels['relu'].summary=L('让负数停在零，看看这一步怎样改变投影结果。','Stop negative values at zero and see how that changes the projection.');
M.levels['original-ffn'].concept=L('原版前馈网络 FFN','Original feed-forward network FFN');
M.levels['workshop'].concept=L('模块封装与复用','Module encapsulation and reuse');
M.levels['residual'].concept=L('残差连接','Residual connection');
M.levels['residual'].summary=L('保留原输入，把 FFN 带回的修改量加上去。','Keep the original input and add the update brought back by the FFN.');
M.levels['positions'].concept=L('正弦位置编码','Sinusoidal position encoding');
M.levels['positions'].summary=L('同一个词换个位置，带上的数也不一样。','Move the same token to another position and its values change too.');
M.levels['layernorm'].concept=L('LayerNorm','LayerNorm');
M.levels['layernorm'].summary=L('观察每组数怎样围绕均值、调整尺度。','Watch each vector get centered around its mean and rescaled.');
M.levels['add-norm'].concept=L('Post-LN · Add & Norm','Post-LN · Add & Norm');
M.levels['add-norm'].summary=L('把输入与修改量合起来，再整理这组数。','Combine the input and update, then normalize the resulting values.');
M.levels['weighted-sum'].concept=L('注意力加权求和','Attention weighted sum');
M.levels['weighted-sum'].summary=L('让读取比例真正带回内容，而不只是一张比例表。','Use the reading proportions to bring back content, not just a table of weights.');
M.levels['self-attention'].concept=L('单头自注意力','Single-head self-attention');
M.levels['self-attention'].summary=L('把匹配、比例和内容接起来，完成一次词间交流。','Connect matching, proportions and content into one exchange between tokens.');
M.levels['multihead'].concept=L('多头注意力','Multi-head attention');
M.levels['multihead'].summary=L('同一句话读两遍，合用两个头带回的信息。','Read the same sentence in two ways and combine what the heads bring back.');
M.levels['wider-features'].concept=L('词向量维度','Embedding dimensions');
M.levels['wider-features'].summary=L('词还是那些词，换成每词四个数，为整层计算作准备。','Keep the same tokens, now with four values each, ready for a complete layer.');
M.levels['encoder-layer'].concept=L('原版编码器层','Original encoder layer');
M.levels['encoder-layer'].summary=L('让词互相交流，再分别加工带回的信息。','Let tokens exchange information, then process what each has received.');
M.levels['masked-multihead'].concept=L('双头因果注意力','Two-head causal attention');
M.levels['masked-multihead'].summary=L('同一句话交给两个头，两边都不能偷看未来。','Give the sentence to two heads without letting either peek ahead.');
M.levels['decoder-self'].concept=L('因果自注意力子层','Causal self-attention sublayer');
M.levels['decoder-self'].summary=L('续写之前，先整理目标前文；这一步还不读源句。','Before continuing the text, process the target context without reading the source yet.');
M.levels['cross-attention'].concept=L('交叉注意力','Cross-attention');
M.levels['cross-attention'].summary=L('带着目标前文去读源句，把内容带回目标侧。','Read the source using the target context and bring content back to the target.');
M.levels['decoder-layer'].concept=L('原版解码器层','Original decoder layer');
M.levels['decoder-layer'].summary=L('读目标前文、读源句，再加工信息，接成完整一层。','Read the target context, read the source and process the information in one layer.');
M.levels['vocabulary-scores'].concept=L('词表投影','Vocabulary projection');
M.levels['vocabulary-scores'].summary=L('看看每个目标位置为六个候选词打了多少分。','See how each target position scores the six candidate tokens.');
M.levels['encoder-decoder'].concept=L('原版编码器—解码器','Original encoder–decoder');
M.levels['encoder-decoder'].summary=L('把源句与目标前文两路接起来，一路走到候选词分数。','Connect the source and target-context paths all the way to candidate scores.');
M.levels['vocabulary'].concept=L('最高分选词 · Argmax','Maximum token selection · Argmax');
M.levels['vocabulary'].summary=L('从六个候选中选出得分最高的词，不把分数当词。','Choose the highest-scoring of six candidates: the score itself is not a token.');
// The FFN reuse exercise already lives in the prologue. Retired links continue
// at the next real lesson; their saved circuits remain under the old key.
M.levelRedirects={...(M.levelRedirects||{}),'pack-ffn':'add-norm'};
M.resolveLevelId=id=>{const target=Object.hasOwn(M.levelRedirects,id)?M.levelRedirects[id]:id;return Object.hasOwn(M.levels,target)?target:null;};
// Old scoring saves may contain parts that were formerly taught before this
// lesson. Preserve those placed parts without offering them early on the shelf.
M.levels['vocabulary-scores'].legacyInventory={'original-head':4,'target-embed':2,'original-transformer':2};
M.levels.vocabulary.legacyInventory={'original-head':4};
for(const level of Object.values(added))if(level.buildable)level.inventory??=Object.fromEntries(level.nodes.map(kind=>[kind,2]));
const originalPrologue=['wire','projection','relu','down-projection','original-ffn','workshop','residual'];
const originalNetwork=['embedding','positions','layernorm','add-norm','qkv','scores','softmax','weighted-sum','self-attention','multihead','wider-features','encoder-layer','causal','decoder-self','cross-attention','decoder-layer','vocabulary-scores','encoder-decoder','vocabulary','autoregressive'];
const modern=['activation','plain-ffn','gate-projection','pairwise','gating','swiglu','normalize','ffn','rotation','rope','aggregation','attention','block'];
for(const level of Object.values(M.levels))level.practice=false;
for(const [ids,prefix,board]of [[originalPrologue,L('序章 / ','Prologue / '),'PROLOGUE'],[originalNetwork,L('原版基础 / ','Original / '),'ORIGINAL'],[modern,L('现代扩展 / ','Modern / '),'MODERN']])ids.forEach((id,i)=>{
 const l=M.levels[id];l.chapter=prefix+String(i+1).padStart(2,'0');l.boardText=board+' / '+String(i+1).padStart(2,'0');l.practice=board==='PROLOGUE';l.original=board!=='MODERN';
 if(l.original){delete l.initialLinks;delete l.initialSink;}
});
// Bridge lessons have their own labels so existing chapter numbers and hashes stay stable.
const bridges={'weighted-sum':'pack-scores','decoder-self':'masked-multihead'};
Object.assign(M.levels['pack-scores'],{chapter:L('模块复用 / 01','Module reuse / 01'),boardText:'MODULE REUSE / 01',original:true});
Object.assign(M.levels['masked-multihead'],{chapter:L('模块组合 / 01','Packaging / 01'),boardText:'PACKAGING / 01',original:true});
M.routes.main={name:L('主线 · 原版基础','Main · Original Transformer'),description:L('从 ReLU FFN、LayerNorm 和正弦位置编码开始，逐步搭建注意力、编码器与解码器。搭建关由你连线，完整演示可直接运行。','Start with ReLU FFNs, LayerNorm and sinusoidal positions, then build attention, the encoder and the decoder. Wire the building lessons yourself; run the complete demonstration as prepared.'),ids:[...originalPrologue,...originalNetwork.flatMap(id=>bridges[id]?[bridges[id],id]:[id])]};
M.routes.modern={name:L('扩展 · 现代模型','Extensions · Modern models'),description:L('对比 SiLU / SwiGLU、RMSNorm / Pre-LN 与 RoPE 对原版 FFN、归一化和位置处理的改变。','Compare how SiLU / SwiGLU, RMSNorm / Pre-LN and RoPE change the original FFN, normalization and position handling.'),ids:modern};
M.routes={main:M.routes.main,modern:M.routes.modern,engine:M.routes.engine};
for(const [track,route]of Object.entries(M.routes))route.ids.forEach((id,i)=>{M.levels[id].track=track;M.levels[id].next=route.ids[i+1]||null;});
// Selection-page chapters describe learning dependencies without changing the
// playable route, stored IDs, or the order of any mathematical operations.
M.routes.main.description=L('从基础模块逐步搭建原版编码器—解码器，观察输入特征如何变成词表分数。','Build the original encoder–decoder from basic modules and trace input features through to vocabulary scores.');
M.routes.main.groups=[
 {id:'workbench',label:L('动手入门','Getting started'),title:L('序章 · 学会动手搭网络','Prologue · Learn to build'),description:L('学习投影与 ReLU，搭建原版 FFN。再从组件架复用 FFN，并在模块外加入残差。','Learn projections and ReLU to build the original FFN. Then reuse it from the module shelf and add a residual outside the module.'),ids:originalPrologue},
 {id:'foundations',label:L('词向量与归一化','Inputs and normalization'),title:L('基础准备 · 词向量、位置与归一化','Foundations · Embeddings, positions and normalization'),description:L('学习带位置的词向量，以及原版“先残差相加，再 LayerNorm”的结构。','Learn position-aware embeddings and the original pattern of residual addition followed by LayerNorm.'),ids:['embedding','positions','layernorm','add-norm']},
 {id:'attention',label:L('搭出注意力','Build attention'),title:L('注意力 · 从单头扩展到多头','Attention · From one head to several'),description:L('理解 Q/K 匹配与 V 汇合，逐步搭出单头和多头注意力。','Build single-head and multi-head attention by learning Q/K matching and V aggregation.'),ids:['qkv','scores','softmax','pack-scores','weighted-sum','self-attention','multihead']},
 {id:'encoder',label:L('组装编码器','Assemble the encoder'),title:L('编码器 · 准备输入，再组装整层','Encoder · Prepare the input, then assemble the layer'),description:L('准备每词四个特征的输入，再用双头注意力、FFN、残差和 LayerNorm 组装编码器层。','Prepare four features per token, then assemble an encoder layer with two-head attention, an FFN, residuals and LayerNorm.'),ids:['wider-features','encoder-layer'],connections:{sources:[{label:L('已学的计算','Familiar computations'),text:L('多头注意力 + FFN + 残差 + LayerNorm','Multi-head attention + FFN + residuals + LayerNorm')},{label:L('准备的输入','Prepared input'),text:L('四维词向量与位置','Four-feature embeddings and positions')}],target:{label:L('组合起来','Assemble'),text:L('原版编码器层','Original encoder layer')}}},
 {id:'decoder',label:L('开始搭解码器','Build the decoder'),title:L('解码器 · 因果自注意力与交叉注意力','Decoder · Causal self-attention and cross-attention'),description:L('先搭因果注意力处理目标前文，再读取源句、加入 FFN，组成解码器层。','Build causal attention for the target context, then read the source and add an FFN to complete the decoder layer.'),ids:['causal','masked-multihead','decoder-self','cross-attention','decoder-layer'],connections:{sources:[{label:L('目标侧','Target side'),text:L('因果遮罩 → 双头注意力 → 残差与归一化','Causal masks → two-head attention → residual and normalization')},{label:L('源句侧','Source side'),text:L('编码器输出 E','Encoder output E')}],target:{label:L('读取源句','Read the source'),text:L('交叉注意力','Cross-attention')}}},
 {id:'generation',label:L('打分、组合与选词','Score, assemble and select'),title:L('模型末端 · 先学打分，再连接完整模型','Model output · Learn scoring before assembling the model'),description:L('先把解码器特征投影为词表分数，再组装完整模型并练习选词。这些关卡使用相同的四维特征和词表权重。最后单独练习生成循环。','Project decoder features into vocabulary scores, then assemble the complete model and practice token selection. These lessons share the same four-feature states and vocabulary weights. Practice the generation loop separately at the end.'),ids:['vocabulary-scores','encoder-decoder','vocabulary','autoregressive']}
];
const menuLesson=(id,kind,label,summary)=>{M.levels[id].menu={kind,label,summary};};
menuLesson('multihead','extend',L('扩展模块','Extend a module'),L('同一句话读两遍，合用两个头带回的信息。','Read the same sentence in two ways and combine what the heads bring back.'));
menuLesson('wider-features','prepare',L('输入准备','Prepare input'),L('词还是那些词，换成每词四个数，为整层计算作准备。','Keep the same tokens, now with four values each, ready for a complete layer.'));
menuLesson('encoder-layer','assemble',L('组装整层','Assemble a layer'),L('让词互相交流，再分别加工带回的信息。','Let tokens exchange information, then process what each has received.'));
menuLesson('causal','branch',L('进入解码器','Enter the decoder'),L('整句都在眼前，也只能参考自己和前面的词。','Even with the whole sentence in view, read only yourself and earlier tokens.'));
menuLesson('masked-multihead','assemble',L('组合双头','Package both heads'),L('同一句话交给两个头，两边都不能偷看未来。','Give the sentence to two heads without letting either peek ahead.'));
menuLesson('decoder-self','reuse',L('第一段 · 读取目标前缀','Stage 1 · Read the target prefix'),L('续写之前，先整理目标前文；这一步还不读源句。','Before continuing the text, process the target context without reading the source yet.'));
menuLesson('cross-attention','source',L('第二段 · 读取源句','Stage 2 · Read the source'),L('带着目标前文去读源句，把内容带回目标侧。','Read the source using the target context and bring content back to the target.'));
menuLesson('decoder-layer','assemble',L('完整层 · 再加入 FFN','Complete layer · Add the FFN'),L('读目标前文、读源句，再加工信息，接成完整一层。','Read the target context, read the source and process the information in one layer.'));
M.levels['decoder-self'].menu.connection=L('本关输出 D → 交叉注意力的查询输入','Output D → query input for cross-attention');
M.levels['cross-attention'].menu.connection=L('目标上下文 D + 编码器输出 E → 交叉注意力更新；解码器层将更新加回目标特征','Target context D + encoder output E → cross-attention update; the decoder layer adds the update to the target features');
M.levels['decoder-layer'].menu.connection=L('因果自注意力 → 交叉注意力 → FFN，三个子层组成解码器层','Causal self-attention → cross-attention → FFN: three sublayers form the decoder layer');
menuLesson('vocabulary-scores','prepare',L('先学词表打分','Learn vocabulary scoring'),L('看看每个目标位置为六个候选词打了多少分。','See how each target position scores the six candidate tokens.'));
menuLesson('encoder-decoder','connect',L('组合已学模块','Assemble familiar modules'),L('把源句与目标前文两路接起来，一路走到候选词分数。','Connect the source and target-context paths all the way to candidate scores.'));
menuLesson('vocabulary','prepare',L('再从分数选词','Then select tokens'),L('从六个候选中选出得分最高的词，不把分数当词。','Choose the highest-scoring of six candidates: the score itself is not a token.'));
M.levels['vocabulary-scores'].menu.connection=L('解码器特征 → 候选词分数','Decoder features → candidate-token scores');
M.levels['encoder-decoder'].menu.connection=L('编码器 + 解码器 + 词表投影 → 完整模型','Encoder + decoder + vocabulary projection → complete model');
M.levels.vocabulary.menu.connection=L('词表投影的候选分数 → 预测词 ID','Vocabulary candidate scores → predicted token IDs');
M.levelMenuCopy={decoderSteps:L('因果自注意力 → 交叉注意力 → 完整解码器层','Causal self-attention → cross-attention → complete decoder layer'),stages:L('按学习阶段浏览','Browse learning stages'),progress:L('关已通过','levels completed'),connections:L('这些关卡怎样接起来','How these lessons connect')};
I18n.prepare(M.levelMenuCopy);
// Explain visibility separately from the Q/K scores whose values are checked.
Object.assign(M.levels.causal,{
 mission:L('让每个词只能参考自己和前面的词','Let each token read only itself and earlier tokens'),
 description:L('整句都在眼前，也不能偷看后面的词。试着在打分计算中用上因果遮罩，让出口的每组比例只分给自己和过去；未来位置必须是 0。本关不计算 V 或内容汇合。','Even with the whole sentence in view, do not peek ahead. Use a causal mask in the scoring circuit so each output row allocates weight only to itself and the past; future positions must get zero. This lesson does not compute V or value aggregation.'),
 rationale:{kind:'causal',
 title:L('为什么不能偷看后面的词？','Why not peek at later tokens?'),
 text:L('在完整解码器里，“爱”位置的结果用来预测它后面的词。如果它提前读到“猫”，就等于看了答案。这种只允许读取自己和过去的限制，叫因果遮罩。因此，即使一次送入整句，也要让“爱”只能参考“我、爱”。','In a complete decoder, the result at love predicts the following token. Reading cats there would reveal the answer. Restricting each position to itself and the past is called causal masking. Even when the whole sentence is supplied at once, love must read only I and love.'),
 takeaway:L('遮罩决定“哪些词不能参考”；Q/K 打分和 Softmax 决定“其余词各占多少比例”。本关检查整张 A，不能只把未来的权重变成 0：直接用 X 代替 Q/K，其他比例也会算错。','The mask decides which tokens are forbidden; Q/K scoring and Softmax decide how to divide weight among the rest. This lesson checks all of A, not just future zeros. Using X in place of Q/K changes the remaining proportions.'),
 scope:L('本关只计算参考比例，还没有汇合 V，也不会生成下一个词。','This lesson computes reference weights only. It does not yet aggregate V or generate the next token.')}
});
M.causalGuide={
 caption:L('以“我 / 爱 / 猫”为例：谁能参考谁？','For I / love / cats: who may read whom?'),
 reference:L('参考 →','Read →'),allowed:L('可参考','Allowed'),blocked:L('不可参考','Blocked'),
 legend:L('✓ 可参考，由分数决定比例；0 不可参考。','✓ Allowed: scores determine the weight. 0 Blocked.'),
 read:L('看“爱”这一组：三个数依次表示参考“我、爱、猫”的比例，加起来是 1。','For love, the three numbers are its weights for I, love and cats, in that order. They sum to 1.'),
 detail:L('遮罩在计算里改了什么？','What does the mask change in the calculation?'),
 before:L('“爱”的 Q/K 分数','love’s Q/K scores'),masked:L('遮住后面的“猫”','Mask the later cats'),after:L('Softmax 后的比例','Weights after Softmax'),
 why:L('把“猫”的分数改成 −∞，Softmax 才会给它 0 权重。不能只把分数改成 0，因为 exp(0) = 1，它仍会分到权重。其余分数保持不变，再由 Softmax 分配合计为 1 的比例。','Set the score for “cats” to −∞ so Softmax gives it zero weight. A score of 0 does not block it: exp(0) = 1 still receives weight. Keep all allowed scores unchanged; Softmax then normalizes their weights to sum to 1.'),
 weightsNote:L('每个位置参考各词的比例','Weights for reading each token')
};
I18n.prepare(M.causalGuide);I18n.prepare(M.levels.causal.rationale);
// After the placement tutorial, the shelf remains a usable part of the game.
// Preserve each lesson's prepared table and existing inventory; removing a
// module makes that copy available to place again without changing the puzzle.
for(const level of Object.values(M.levels))if(!level.practice&&!level.buildable){level.buildable=true;level.startWithModules=true;}
for(const id of [...originalNetwork,...Object.values(bridges)]){const l=M.levels[id];l.migratePreparedStart=!!l.startWithModules;l.startWithModules=false;l.requirePlacement=true;}
// A lesson's reference graph is not its shopping list. Keep learned modules in
// teaching order, without marking which ones occur in the reference solution.
// Scalar exercises and whole-sequence circuits have different port contracts.
// The engine track uses its own costed tasks, not raw arithmetic operators.
const sequenceKind=kind=>Math.max(...M.definitions[kind].ins,M.definitions[kind].out)>4;
// A reward has exactly the same inputs, weights and output as its lesson.
// Each new package has one introduction. Reuse lessons do not award it again.
M.fusions={
 'original-ffn':{kind:'relu-ffn',inputs:['R']},
 softmax:{kind:'original-weights',inputs:['R']},
 'self-attention':{kind:'single-attention',inputs:['R']},
 multihead:{kind:'two-head-attention',inputs:['R']},
 'wider-features':{kind:'source-embed',inputs:['R']},
 'encoder-layer':{kind:'original-encoder',inputs:['R']},
 'masked-multihead':{kind:'masked-heads',inputs:['R']},
 'cross-attention':{kind:'single-cross-attention',inputs:['R','Z']},
 'decoder-layer':{kind:'original-decoder',inputs:['R','Z']},
 'encoder-decoder':{kind:'original-transformer',inputs:['R','Z']},
 'plain-ffn':{kind:'silu-ffn',inputs:['R']},
 swiglu:{kind:'swiglu-ffn',inputs:['R']},
 ffn:{kind:'swiglu-residual',inputs:['R','Z']},
 attention:{kind:'attention',inputs:['R']},
 block:{kind:'modern-block',inputs:['R']}
};
const fusionOrigins={};for(const [id,{kind}]of Object.entries(M.fusions))fusionOrigins[kind]??=id;
M.fusionCopy={
 gathering:L('将已搭建的网络封装为模块','Package the completed network as a module'),
 formed:L('一个模块，内部计算不变','One module, unchanged computation'),
 stored:L('已加入左侧组件架，可拖入棋盘使用','Added to the left module shelf; drag it onto the board to use it'),
 storedLater:L('已加入组件架，后续搭建关可使用','Added to the module shelf for later building lessons'),
 useLater:L('后续搭建关可用','Use in later building lessons'),
 collected:L('已收录模块','Module collected'),
 collectedShort:L('已收录','Collected'),
 skip:L('跳过动画','Skip animation')
};
I18n.prepare(M.fusionCopy);
const learned=new Set(),origins={};
for(const [track,route]of Object.entries(M.routes)){
 const known=track==='engine'?new Set():learned;
 for(const id of route.ids){
  const l=M.levels[id],own={...M.inventory(l)};
  for(const kind of Object.keys(own)){known.add(kind);origins[kind]??=id;}
  if(M.fusions[id])known.add(M.fusions[id].kind);
  if(l.practice||l.engine)continue;
  const kinds=[...known].filter(kind=>sequenceKind(kind)===!!l.sequence||own[kind]);
  l.inventory=Object.fromEntries(kinds.map(kind=>[kind,Math.max(2,own[kind]||0)]));
  l.maxModules=Math.max(M.moduleLimit(l),l.nodes.length+2);
 }
}
// Cards for optional shelf modules use a real, dimensionally valid example
// from the lesson in which that module was taught, rather than a missing node.
M.exampleLevel=(kind,id)=>M.levels[id].nodes.includes(kind)?id:origins[kind]||fusionOrigins[kind]||id;
// Equal toy parameters do not make named sublayer instances interchangeable.
// Generic arithmetic (add, layernorm2) has no positional contract.
const residualRoles={'encoder-add':'attention',residual1:'attention','cross-add':'cross','ffn-add':'ffn',residual2:'ffn'};
const normRoles={'ln-attn':'attention','ln-cross':'cross','ln-ffn':'ffn'};
const updateRoles=Object.fromEntries([
 ...['single-attention','two-head-attention','self-heads','masked-heads','attention'].map(k=>[k,'attention']),
 ...['single-cross-attention','cross-heads'].map(k=>[k,'cross']),
 ...['relu-ffn','silu-ffn','swiglu-ffn','original-ffn-block','ffnblock'].map(k=>[k,'ffn'])
]);
const roleCopy={
 attention:{residual:L('一个入口接自注意力输出，另一个接该子层输入。此模块不能代替 FFN 或交叉注意力的残差。','Connect the self-attention output to one input and that sublayer’s input to the other. This module cannot replace an FFN or cross-attention residual.'),norm:L('只接 Attention 残差的输出。它属于自注意力这一段，不是 FFN 的归一化。','Accepts only the Attention residual output. It belongs to the self-attention stage, not the FFN stage.')},
 cross:{residual:L('一个入口接交叉注意力输出，另一个接发起查询的目标特征。','Connect the cross-attention output to one input and the target features that supplied its queries to the other.'),norm:L('只接受交叉注意力残差输出，不能用于自注意力或 FFN 子层。','Accepts only the cross-attention residual output, not a self-attention or FFN residual.')},
 ffn:{residual:L('一个入口接 FFN 输出，另一个接 FFN 输入。此相加模块内部不含 FFN。','Connect the FFN output to one input and the FFN input to the other. This addition module does not contain an FFN.'),norm:L('只接受 FFN 残差输出，不能用于注意力子层。','Accepts only the FFN residual output, not an attention residual.')}
};
I18n.prepare(roleCopy);
L('模块用途不匹配：','Module role mismatch: ');L('模块用途不匹配','Module role mismatch');
M.componentUse=kind=>{const role=normRoles[kind]||residualRoles[kind];return role?roleCopy[role][normRoles[kind]?'norm':'residual']:'';};
function componentIssues(state){
 const nodes=new Map(state.nodes.map(n=>[n.id,n])),sources=new Set(M.levels[state.levelId].sources);
 const node=id=>nodes.get(id),kind=id=>M.nodeKind(node(id)),completeMemo=new Map();
 function complete(id,seen=new Set()){
  if(sources.has(id))return true;if(!id||seen.has(id)||!node(id))return false;
  if(completeMemo.has(id))return completeMemo.get(id);
  const next=new Set(seen);next.add(id);const done=node(id).links.every(from=>complete(from,next));completeMemo.set(id,done);return done;
 }
 function projectionSource(id,channel,seen=new Set()){
  if(!id||seen.has(id))return null;seen.add(id);const n=node(id),k=kind(id);if(!n)return null;
  if(k==='rope'+channel)return projectionSource(n.links[0],channel,seen);
  return [channel,'original-'+channel,'masked-a-'+channel,'masked-b-'+channel].includes(k)?n.links[0]:null;
 }
 function scoreInputs(id,seen=new Set()){
  if(!id||seen.has(id))return null;seen.add(id);const n=node(id),k=kind(id);if(!n)return null;
  if(['softmax','mask'].includes(k))return scoreInputs(n.links[0],seen);
  if(['original-weights','attentionweights'].includes(k))return n.links[0]?[n.links[0],n.links[0]]:null;
  if(['score','original-score'].includes(k)){const q=projectionSource(n.links[0],'q'),key=projectionSource(n.links[1],'k');return q&&key?[q,key]:null;}
  return null;
 }
 // Follow an expanded attention computation as well as packaged heads. This
 // checks tensor provenance, not whether the player copied a reference graph.
 function attentionInputs(id,seen=new Set()){
  if(!id||seen.has(id))return null;seen.add(id);const n=node(id),k=kind(id);if(!n)return null;
  if(updateRoles[k]==='attention'||['head-a','head-b'].includes(k))return n.links[0]?[n.links[0],n.links[0]]:null;
  if(updateRoles[k]==='cross')return n.links.every(Boolean)?n.links.slice(0,2):null;
  if(['output','original-output'].includes(k))return attentionInputs(n.links[0],seen);
  if(['concat','wide-concat'].includes(k)){
   const heads=n.links.map(from=>attentionInputs(from,new Set(seen)));
   return heads.every(h=>h&&h[0]===heads[0]?.[0]&&h[1]===heads[0]?.[1])?heads[0]:null;
  }
  if(['aggregate','original-aggregate'].includes(k)){
   const pair=scoreInputs(n.links[0]),v=projectionSource(n.links[1],'v');return pair&&v===pair[1]?pair:null;
  }
  return null;
 }
 function updateRole(id){
  const k=kind(id);if(updateRoles[k])return updateRoles[k];
  if(['output','original-output'].includes(k)){const pair=attentionInputs(id);return pair?(pair[0]===pair[1]?'attention':'cross'):null;}
  if(k==='down'&&['relu','silu','multiply'].includes(kind(node(id).links[0])))return 'ffn';
  return null;
 }
 const issues=[];
 for(const n of state.nodes){
  const k=M.nodeKind(n),role=normRoles[k]||residualRoles[k];if(!role)continue;
  const mismatch=normRoles[k]?n.links[0]&&residualRoles[kind(n.links[0])]!==role:n.links.every(Boolean)&&n.links.every(id=>complete(id))&&!n.links.some(id=>updateRole(id)===role);
  if(mismatch)issues.push({id:n.id,port:normRoles[k]?0:undefined,message:I18n.t('模块用途不匹配：')+M.nodeName(state,n.id)+' · '+M.componentUse(k)});
 }
 return issues;
}
M.componentIssues=componentIssues;
const connectBeforeRoles=M.connect,evaluateBeforeRoles=M.evaluate;
M.connect=(state,from,to,port)=>{
 const next={...state,nodes:state.nodes.map(n=>({...n,links:[...n.links]}))},error=connectBeforeRoles(next,from,to,port);if(error)return error;
 // Validate descendants too: wiring an upstream node last must not bypass a
 // contract already connected farther downstream. Unfinished branches can wait.
 const affected=new Set([to]);let changed=true;
 while(changed){changed=false;for(const n of next.nodes)if(!affected.has(n.id)&&n.links.some(id=>affected.has(id))){affected.add(n.id);changed=true;}}
 const issue=componentIssues(next).find(issue=>affected.has(issue.id));if(issue)return issue.message;
 return connectBeforeRoles(state,from,to,port);
};
M.evaluate=(state,input)=>{
 const ev=evaluateBeforeRoles(state,input);if(ev.error)return ev;
 const used=new Set(),nodes=new Map(state.nodes.map(n=>[n.id,n])),visit=id=>{if(!id||used.has(id))return;used.add(id);nodes.get(id)?.links.forEach(visit);};visit(state.sink);
 const issues=componentIssues(state),issue=ev.order.map(id=>issues.find(i=>i.id===id)).find(i=>i&&used.has(i.id));
 if(issue){ev.error=issue.message;ev.bad={id:issue.id,port:issue.port};ev.componentIssue=issue;}
 return ev;
};
M.moduleShape=(kind,dim)=>{
 if(['head-a','head-b','concat'].includes(kind)&&dim===3)return '3 × 1';
 return sequenceKind(kind)?({3:I18n.t('3 个 ID'),6:'3 × 2',9:'3 × 3',12:'3 × 4',18:'3 × 6'}[dim]||dim):dim;
};
const moduleNames=ids=>ids.map(id=>M.definitions[id].name),pack=(stages,note)=>({stages,note});
const ffnStages=[moduleNames(['project']),['ReLU'],moduleNames(['down'])];
const matchingStages=[['Q','K'],['QKᵀ / √d_k'],['Softmax']];
const headStages=[['Q','K','V'],['softmax(QKᵀ / √d_k) V']];
M.packages={
 'silu-ffn':pack([moduleNames(['project']),['SiLU'],moduleNames(['down'])],M.definitions['silu-ffn'].hint),
 'swiglu-ffn':pack([['XW', 'SiLU(XWg)'],['⊙'],['Wd']],M.definitions['swiglu-ffn'].hint),
 'swiglu-residual':pack([['R', 'Z → SwiGLU'],['Y = R + F']],M.definitions['swiglu-residual'].hint),
 'original-transformer':pack([[L('源 ID → 词向量与位置 → 编码器','Source IDs → embeddings and positions → encoder')],[L('目标 ID → 词向量与位置','Target IDs → embeddings and positions'),L('编码器输出','Encoder output')],moduleNames(['original-decoder']),moduleNames(['original-head'])],M.definitions['original-transformer'].hint),
 'modern-block':pack([['RMSNorm → Attention'],['R1 = X + Attention'],['RMSNorm → SwiGLU'],['Y = R1 + FFN']],M.definitions['modern-block'].hint),
 'single-attention':pack([['Q','K','V'],['QKᵀ / √2'],['Softmax'],['A × V'],['Wo']],M.definitions['single-attention'].hint),
 'two-head-attention':pack([moduleNames(['head-a','head-b']),moduleNames(['concat']),['Wo']],M.definitions['two-head-attention'].hint),
 'single-cross-attention':pack([['Q ← D','K, V ← E'],['QKᵀ / √4'],['Softmax'],['A × V'],['Wo']],M.definitions['single-cross-attention'].hint),
 'relu-ffn':pack(ffnStages,L('与序章的三个模块使用相同权重：2 → 4 → 4 → 2。ReLU 仍在内部，因此不是一次线性变换。','Uses the same weights as the three prologue modules: 2 → 4 → 4 → 2. ReLU remains inside, so this is not one linear transformation.')),
 'original-weights':pack(matchingStages,L('只封装 Q/K 的匹配与比例，不包含 V、内容汇合、遮罩或 RoPE。','Packages only Q/K matching and proportions, without V, value aggregation, masking or RoPE.')),
 'head-a':pack(headStages,L('把已学的 Q/K/V、打分、Softmax、AV 收成一个头。这里各投影到 1 个特征，使用第一组权重；不含 Wo。','Packages Q/K/V, scores, Softmax and AV as one head. Here each projection produces one feature with the first set of weights; Wo is outside.')),
 'head-b':pack(headStages,L('与头 A 的步骤相同，但使用第二组权重，各投影到 1 个特征；两头各算各的比例，不共享 Softmax。','Uses the same steps as head A, with a second set of weights and one feature per projection. Each head computes its own Softmax weights.')),
 'self-heads':pack([moduleNames(['head-a','head-b']),moduleNames(['concat']),['Wo']],L('把两个头、拼接和 Wo 收在一起。完整层使用 4 个特征、每头 2 个；结构沿用前面的双头练习，权重也相应变宽。','Packages both heads, concatenation and Wo. Complete layers use four features, two per head: the same structure as the earlier two-head exercise, with wider weights.')),
 'masked-heads':pack([['Q','K','V'],['softmax(mask(QKᵀ / √d_k)) V'],['Concat(head A, head B) Wo']],L('沿用双头结构，在每个头的 Softmax 前加入因果遮罩。残差和 LayerNorm 仍在模块外。','Reuses the two-head structure with a causal mask before each head’s Softmax. Residual addition and LayerNorm remain outside the module.')),
 'cross-heads':pack([['Q ← D', 'K, V ← E'],['softmax(QKᵀ / √d_k) V'],['Concat(head A, head B) Wo']],L('把交叉注意力装进两个独立的头：D 提供目标特征，E 提供源特征，内部各自投影出 Q 和 K/V。现在每头 2 个特征，拼接后乘 Wo；不含残差或归一化。','Packages cross-attention into two independent heads. D supplies target features and E supplies source features; internal projections produce Q and K/V. Each head now has two features; concatenate and apply Wo. Residuals and normalization stay outside.')),
 'original-ffn-block':pack(ffnStages,L('内部仍是 Linear → ReLU → Linear。这次对三个 token 分别做 4 → 8 → 4；它把单个向量的练习推广到整段序列，不混合 token。','Still Linear → ReLU → Linear, now applying 4 → 8 → 4 to each of three tokens. This extends the single-vector exercise to a sequence without mixing tokens.')),
 'source-embed':pack(moduleNames(['wide-embedding','wide-position']).map(name=>[name]),L('用源 ID 查找四维词向量。先乘 √4，再加位置编码。计算与四维词向量练习相同。','Look up four-feature embeddings using source IDs. Scale them by √4, then add positions. The computation matches the four-feature embedding exercise.')),
 'target-embed':pack(moduleNames(['wide-embedding','wide-position']).map(name=>[name]),L('对目标前缀做同样的查表与位置相加。源与目标仍从不同的输入口取 ID。','Apply the same lookup and position addition to the target prefix. Source and target IDs still come from separate input ports.')),
 'original-encoder':pack([[labels.attn],['Add → LayerNorm'],[labels.ffn],['Add → LayerNorm']],L('收起已经搭过的一层编码器。每个 Add 都加回当前子层自己的输入；内部保留两次残差和两次 LayerNorm。','Packages the encoder layer already built. Each Add uses its own sublayer input; both residuals and both LayerNorm operations remain inside.')),
 'original-decoder':pack([[labels.masked],['Add → LayerNorm'],[labels.cross],['Add → LayerNorm'],[labels.ffn],['Add → LayerNorm']],L('收起已经搭过的解码器。保留目标和编码器输出两个入口，以及三段各自的残差与 LayerNorm。','Packages the decoder already built. It retains separate target and encoder-memory inputs, plus each of the three residual and LayerNorm wrappers.')),
 attentionweights:pack([['Q','K'],['RoPE'],['QKᵀ / √2'],[M.definitions.mask.name],['Softmax']],L('在原版匹配流程中加入旋转和遮罩。输出仍是注意力权重。V 投影和内容汇合不在模块内。','Adds rotation and masking to the original matching sequence. The output is still attention weights. V projection and value aggregation remain outside the module.')),
 attention:pack([[M.definitions.attentionweights.name,'V'],['AV'],['Wo']],L('收起现代 Attention 关搭过的整段：Q/K/V、Q/K 的 RoPE、因果遮罩、Softmax、AV 和 Wo。外部的 RMSNorm 与残差不包含在这里。','Packages the modern Attention circuit: Q/K/V, RoPE on Q/K, causal masking, Softmax, AV and Wo. The external RMSNorm and residual are not included.')),
 ffnblock:pack([['RMSNorm'],['U = XW', 'S = SiLU(XWg)'],['S ⊙ U'],[M.definitions.down.name]],L('内部先做 RMSNorm，再算双路 SwiGLU，输出修改量。相加的残差仍在外面；这是现代扩展，不是原版 ReLU FFN。','Runs RMSNorm then the two-branch SwiGLU to output an update. Residual addition remains outside; this is the modern extension, not the original ReLU FFN.')),
 enqkv:pack([['RMSNorm'],['Q','K','V']],L('在引擎章把归一化和三种投影当作一个可分配给芯片的任务；输出按 Q、K、V 分段保存。此处“组合任务”不意味着芯片必然执行为一个融合算子。','The engine chapter schedules normalization and all three projections as one task, with output segments for Q, K and V. A grouped task does not imply one fused hardware operator.')),
 enattn:pack([['RoPE(Q, K)'],['softmax(mask(QKᵀ / √2)) V'],['Wo']],L('读取上一任务的 Q/K/V，完成现代因果注意力。这里仍计算同样的数，后面的关卡才比较缓存和分配到不同芯片的成本。','Reads the previous task’s Q/K/V and completes modern causal attention. The arithmetic remains the same; later lessons compare caching and chip-assignment costs.')),
 enffn:pack([['RMSNorm → SwiGLU'],[M.definitions.add.name]],L('把现代 FFN 及外部残差合成一个调度任务，仍保留归一化、两套投影、SiLU、逐元素相乘、降维与相加。','Groups the modern FFN and its residual as one scheduling task. Normalization, both projections, SiLU, elementwise multiplication, down projection and addition all remain inside.'))
};
L('内部组成','Inside this module');L('新组合模块','New packaged module');
M.headDiagramCopy={
 shared:L('同一份输入分别进入两个头，使用两套投影权重。','Both heads receive the same full input and use separate projection weights.'),
 single:L('这个头用自己的 Q/K/V 权重处理完整输入。','This head applies its own Q/K/V weights to the full input.'),
 cross:L('两个头都读取：目标侧特征 → Q；编码器输出 → K/V。','Both heads read target-side features for Q and encoder outputs for K/V.'),
 projection:L('投影','Projection'),query:L('查询','Query'),key:L('键','Key'),value:L('值','Value'),target:L('目标侧','Target'),memory:L('编码器','Encoder'),
 score:L('Q/K 匹配打分','Q/K matching scores'),weights:L('Softmax · 读取比例','Softmax · reading weights'),
 bypass:L('直接送往汇合','Pass to aggregation'),mask:L('遮住未来位置','Mask future positions'),
 mix:L('按比例汇合 V','Mix V using the weights'),concat:L('按同一个词拼接两头的结果','Concatenate both results for the same token'),
 output:L('输出投影','Output projection'),
 roles:L('Q/K 决定从哪里读，V 提供内容；A 是本次计算得到的读取比例。','Q/K decide where to read; V supplies the content. A contains the reading weights computed for this input.')
};
for(const [kind,numbers]of Object.entries({'two-head-attention':[1,2],'self-heads':[1,2],'masked-heads':[1,2],'cross-heads':[1,2],'head-a':[1],'head-b':[2]})){
 const d=M.definitions[kind];
 M.packages[kind].attentionDiagram={numbers,inputWidth:d.ins[0]/3,headWidth:d.out/3/numbers.length,masked:kind==='masked-heads',cross:kind==='cross-heads'};
}
I18n.prepare(M.headDiagramCopy);
L('来源：','From: ');
// Provenance follows the first lesson that packages the computation, rather
// than a later packaging refresher or the currently selected example lesson.
const packageOrigins={};
for(const [id,{kind}]of Object.entries(M.fusions))packageOrigins[kind]??=id;
// Wider tensors and source/target roles reuse these already taught structures.
Object.assign(packageOrigins,{
 'self-heads':'multihead','original-ffn-block':'original-ffn',
 'target-embed':'wider-features','cross-heads':'cross-attention'
});
for(const kind of Object.keys(M.packages))packageOrigins[kind]??=origins[kind];
M.packageOrigin=kind=>{
 const id=M.packages[kind]&&packageOrigins[kind];
 return id?{id,chapter:M.levels[id].chapter}:null;
};
for(const [kind,p]of Object.entries(M.packages)){
 const l=M.levels[origins[kind]];if(l){l.packages??=[];if(!l.packages.includes(kind))l.packages.push(kind);}
}
I18n.prepare(M.packages);
// Group every learning shelf from the first placement lesson onward. Classify
// by interfaces and package metadata; keep compatible distractors and authored
// width-based ordering instead of using the reference wiring as a shopping list.
const groupedLearning=[...M.routes.main.ids,...M.routes.modern.ids];
for(const id of groupedLearning){
 if(!M.levels[id].buildable)continue;
 const l=M.levels[id],contract=kind=>JSON.stringify([M.definitions[kind].ins,M.definitions[kind].out]);
 const contracts=new Set(l.nodes.map(contract)),dimensions=new Set(l.nodes.flatMap(kind=>[...M.definitions[kind].ins,M.definitions[kind].out]));
 const kinds=Object.keys(M.inventory(l)),currentKinds=kinds.filter(kind=>contracts.has(contract(kind))||(M.packages[kind]&&[...M.definitions[kind].ins,M.definitions[kind].out].every(dim=>dimensions.has(dim))));
 l.shelf={...(l.shelf||{currentKinds,order:currentKinds,current:L('可用模块','Available modules'),earlier:L('已学模块','Previously learned modules'),note:L('已学模块仍可拖入棋盘或双击查看。端口尺寸不同的模块不能直接相连。上方包含可选模块，不必全部使用。','Drag learned modules onto the board or double-click to inspect them. Modules with different port sizes cannot connect directly. The groups above include optional modules; you do not need to use them all.')}),packages:L('组合模块','Packaged modules'),basics:L('基础模块','Basic modules')};
 I18n.prepare(l.shelf);
}

// Keep the packaged unmasked weights inspectable, but do not offer new copies here.
M.levels.causal.legacyInventory={'original-weights':M.levels.causal.inventory['original-weights']};
M.levels.causal.inventory['original-weights']=0;
M.levels.qkv.mission=L('为每个词准备一份可读取的内容','Prepare content that can be read from each token');
M.levels.qkv.description=L('每个词先准备好自己的内容，再谈向谁读取。选用 V 投影，让出口的三个词各得到两个数。它们都用同一套权重 Wᵥ；本关只检查 V，Q/K 的匹配计算留给后面。','Each token first prepares its own content; deciding whom to read comes later. Use V projection to produce two values for each of the three tokens. All use the same weights Wᵥ. This lesson checks only V; Q/K matching comes later.');
M.levels.qkv.outputTitle=L('值向量矩阵 V','Value matrix V');
M.levels.qkv.outputNote='V = X × Wᵥ';
M.levels.qkv.resultNote=L('V 是 X × Wᵥ 的计算结果，每个词各有一个值向量。Wᵥ 才是模块内部的权重矩阵。','V is the result of X × Wᵥ, with one value vector per token. Wᵥ is the weight matrix inside the module.');
Object.assign(M.levels.qkv.rationale,{
 title:L('V 是结果，Wᵥ 才是权重','V is the result; Wᵥ holds the weights'),
 text:L('这份可读取的内容叫值向量，V 来自 Value（值）。它由 V = X × Wᵥ 算出：Wᵥ 是模块内部的权重矩阵，V 是计算结果。本关 X 和 V 都是 3 × 2：三个词，每词两个数。每个词仍独立计算。','The content to be read is a value vector; V stands for Value. Compute it as V = X × Wᵥ: Wᵥ is the weight matrix inside the module, while V is the result. Both X and V are 3 × 2: three tokens, two values per token. Each token is still processed independently.'),
 takeaway:L('例如“爱”的输入 [0, 1] 乘本关 Wᵥ = [[1, 0], [0, 2]]，得到值向量 [0, 2]，它仍属于“爱”。后面 Q/K 打分，经 Softmax 变成参考比例，再把各词的 V 按比例相加，词之间才发生信息汇合。','For example, love starts with [0, 1]. Multiplying by this lesson’s Wᵥ = [[1, 0], [0, 2]] produces its value vector [0, 2], still belonging to love. Later, Q/K supply scores and Softmax converts them to proportions that weight the sum of the V vectors. That later step mixes information across tokens.')
});
M.levels.qkv.summary=L('先为每个词准备可读取的 V 内容，还不与其他词交流。','Prepare readable V content for each token, before exchanging information with other tokens.');
Object.assign(M.levels.scores,{
 sourceTitle:L('输入向量矩阵 X','Input vector matrix X'),sourceNames:{R:L('输入矩阵 X','Input matrix X')},routeInput:L('输入矩阵 X','Input matrix X'),
 sourceNote:L('与上一关相同的 X，供 Q/K 投影','The same X as before, for Q/K projections'),
 outputTitle:L('匹配分数矩阵 S','Matching-score matrix S'),outputNote:L('每格对应一对 Q 词与 K 词','One score per query–key token pair'),outputPairAxes:true,
 mission:L('看看每个词与其他词有多匹配','Find out how each token matches the others'),
 description:L('三个词互相比较，会得到怎样一张打分表？用 Q、K 和打分模块，让每对词的交点得到一个匹配分数。输入仍是 X；V 提供内容，不参与本关打分。','What score map will three tokens make when they compare with one another? Use Q, K and scoring so every token pair gets a matching score at its intersection. The input is still X; V supplies content and does not take part in this scoring.'),
 resultNote:L('S 的每格是一个查询词与一个键词的匹配分数。Softmax 将这些分数变成注意力权重 A。','Each cell of S scores one query token against one key token. Softmax turns these scores into attention weights A.')
});
M.levels.scores.summary=L('让三个词两两比较，得到九个匹配分数，还不是读取比例。','Compare three tokens against one another to get nine matching scores, not reading proportions yet.');
Object.assign(M.levels.scores.rationale,{
 title:L('这张打分表在比较什么？','What does this score map compare?'),
 text:L('发起比较的一侧叫查询（Query，Q），供它匹配的一侧叫键（Key，K）。两者从同一份 X 用不同权重算出：Q = XWq，K = XWk。出口左侧是查询词，上方是键词，交点记录 S = QKᵀ / √2 的匹配分数；这叫缩放点积打分。同一查询的分数越大，Softmax 后的注意力权重越高。此时还没有汇合 V。','The side asking is the query (Q); the side it matches is the key (K). Different weights project the same X into Q = XWq and K = XWk. Queries run down the outlet and keys across it; each intersection records a score from S = QKᵀ / √2, called scaled dot-product scoring. For one query, higher scores give larger attention weights after Softmax. V has not been aggregated yet.')
});
M.levels.softmax.description=L('分数有高有低，怎样变成“各取多少”？在熟悉的打分计算中用上 Softmax，让每个词分配出的比例合计为 1。本关所有位置都可见，不遮住未来。','Scores differ, but how much should each token contribute? Add Softmax to the familiar scoring calculation so each token’s proportions sum to 1. All positions are visible here, with no future mask.');
M.levels.softmax.summary=L('用 Softmax 把匹配分数变成“各取多少”的比例。','Use Softmax to turn matching scores into proportions for how much to read.');
M.levels.residual.rationale.scope=L('沿用已学的 ReLU FFN 和同一套教学权重。本关只增加残差。原版完整子层还会在相加后接 LayerNorm，本关不计算这一步。','Reuses the familiar ReLU FFN and the same teaching weights. Only the residual is new. A complete original sublayer applies LayerNorm after addition; this lesson does not compute that step.');
M.levels.ffn.description=L('一条路保留原输入 R，另一条路带回新的变化。用熟悉的 SwiGLU 和残差把它们合起来。关卡已给出整理过的 Z = RMSNorm(R)，让它负责计算修改量。','One path keeps the original input R; another brings back a change. Combine them with the familiar SwiGLU and residual. The lesson supplies the normalized Z = RMSNorm(R) to compute that update.');
M.levels.ffn.rationale.text=L('先把输入整理好再交给支路，这种安排叫 Pre-LN。这里用 RMSNorm 得到 Z，再用 SwiGLU 算出修改量，加回原输入 R；这条相加路径仍是残差连接。原版 Post-LN 则在相加后归一化，两者的激活、分支结构和归一化位置也不同。','Normalizing before the branch is called Pre-LN. Here RMSNorm produces Z, SwiGLU computes an update, and adding it to the original R forms the residual connection. Original Post-LN normalizes after addition; its activation, branch structure and normalization placement also differ.');
M.levels.block.title=L('让一层接起来','Connect a complete layer');
M.levels.block.rationale.scope=L('本关展示采用 Pre-LN、RMSNorm、SwiGLU 与 RoPE 的现代解码器块。它不含编码器交叉注意力，不是 2017 年原版编码器—解码器。','This lesson shows a modern decoder block using Pre-LN, RMSNorm, SwiGLU and RoPE. It has no encoder cross-attention and is not the original 2017 encoder–decoder.');
M.levels.vocabulary.rationale.scope=L('取最大分与取 Softmax 后的最大概率等价，本关直接按分数选词。概率采样会有不同的选择方式。','Selecting the largest logit is equivalent to selecting the largest Softmax probability, so this lesson selects directly from scores. Probability sampling can make different choices.');
L('用“我 / 爱 / 猫”看一次更新','One update of I / love / cats');
L('词与预测的对应关系','Tokens and their predictions');
L('它在生成循环中的作用','Its role in the generation loop');
L('固定查表的教学例子，不代表语言模型的真实预测。','A fixed-lookup teaching example, not predictions from a trained language model.');
M.definitions.step.name=L('教学预测器（查表）','Teaching predictor (lookup)');
M.definitions.step.hint=L('代替完整模型的打分工作：按输入词查固定表，为六个候选词给分。它不含 Attention 或 FFN，也不用接在完整模型后面。','Stands in for the full model’s scoring: look up each input token to score six candidates. It contains no Attention or FFN and does not follow the full model.');
M.definitions.feedback.hint=L('模块记住本次运行的原上下文。只追加入口三个预测中的最后一个词，再保留最近三个词。入口接收预测词，不接收原上下文。','The module remembers the original context for this run. Append only the last of the three incoming predictions, then keep the latest three tokens. The input receives predictions, not the original context.');
// This is a text-window update, not a KV cache. Avoid the cache glyphs K / V.
M.definitions.feedback.glyphs=['…','→','+1'];
Object.assign(M.levels.autoregressive,{
 sourceTitle:L('本次上下文','Current context'),sourceNote:L('已有的三个词，尚未追加新词','Three existing tokens, before appending'),
 outputNote:L('保留旧词，再追加一个新词','Keep old tokens and append one new token'),
 mission:L('让已有文本多出一个新词','Extend the existing text by one token'),
 description:L('上一关从分数里选词；这一关要把最后位置预测的新词接到已有文本后面。出口检查更新后的上下文，不是三个位置各自选出的词。','The previous lesson selected tokens from scores. Now extend the existing text with the last position’s prediction. The outlet checks the updated context, not all three selected predictions.'),
});
M.levels.autoregressive.summary=L('只把最后位置的预测词追加到文本，输出更新后的上下文。','Append only the final position’s predicted token and output the updated context.');
Object.assign(M.levels.autoregressive.rationale,{
 title:L('为什么只追加最后一个预测？','Why append only the last prediction?'),
 text:L('在因果模型中，“我”的位置预测它后面的词，“爱”的位置预测它后面的词，只有最后的“猫”预测的是整段“我爱猫”之后的新词。前面两个位置的后续词已经在上下文里，不应再追加一次。','In a causal model, I predicts the token after I, love predicts the token after love, and only the final token “cats” predicts a new token beyond the whole prefix. The tokens after earlier positions are already in the context; do not append them again.'),
 takeaway:L('新词变成下一次预测的已知条件，这就是自回归。棋盘只搭一次更新；通关后会用同一电路连续更新六次。循环发生在两次计算之间，不需要把出口连回入口。','The new token becomes a known condition for the next prediction: this is autoregression. Build one update on the board; after success, the same circuit repeats it six times. The loop is between evaluations, so no wire from outlet to inlet is needed.'),
 scope:L('为看清文本怎样更新，本关用固定查表代替完整模型打分。三词窗口也是教学限制；真实模型不会每生成一个词就固定删掉最早的一个词。','To isolate the text update, this lesson replaces model scoring with a fixed lookup. The three-token window is also a teaching limit; real models do not always drop the oldest token after each generation step.'),
});
// Derive the explanatory example from the same operators as the playable board.
M.autoregressiveExample=(input=[0,1,2])=>{
 const context=[...input],scores=M.operations.step([context]),predictions=M.operations.pick([scores]);
 return {context,scores,predictions,next:M.operations.feedback([predictions],{input:context})};
};
const previousSamples=M.samplesFor,previousSources=M.sourceValues,previousReference=M.reference,previousSolutions=M.solutions,previousExample=M.extraExample,previousCard=M.extraCard;
M.samplesFor=id=>{
 // Two-feature LayerNorm can hide a missing residual behind near-identical
 // +/-1 outputs. Include sign-changing and zero-update cases, not just epsilon differences.
 if(id==='add-norm')return [[1,-1],[-1,0],[1,1],[-1,2],[2,-.5]].map((v,i)=>({...previousSamples(id)[i],v}));
 if(['vocabulary-scores','vocabulary'].includes(id))return previousSamples('embedding').map(s=>({...s,v:decoder(position(embed(s.v,4)),memory())}));
 if(!M.levels[id]?.positioned)return previousSamples(id);
 return previousSamples('embedding').map(s=>{const positioned=position(embed(s.v,M.levels[id].wide?4:2));return {...s,v:id==='cross-attention'?decoderSelf(positioned):positioned};});
};
M.sourceValues=(state,v)=>M.levels[state.levelId]?.dualSource?{R:v,Z:state.levelId==='encoder-decoder'?[...sourceIds]:memory()}:previousSources(state,v);
M.reference=(v,id)=>references[id]?references[id](v):previousReference(v,id);
M.solutions=id=>{if(!solutions[id])return previousSolutions(id);const s=M.initial(id,{complete:true}),links=solutions[id];for(const n of s.nodes)n.links=[...links[n.id]];s.sink=links.sink;return s;};
const cards={};
for(const id of operatorIds){
 const d=M.definitions[id],w=M.weights[id];cards[id]={number:'',seal:id.slice(0,1).toUpperCase(),english:id.toUpperCase().replaceAll('-',' '),family:L('原版 Transformer','Original Transformer'),color:'#'+M.extraColors[id].toString(16).padStart(6,'0'),ability:d.name,meaning:d.hint,purpose:scope,weight:w?w.length+' × '+w[0].length:L('无 / 已封装','None / packaged'),trait:L('数值计算','Numerical computation'),stage:0};
}
Object.assign(cards.relu,{formula:'Aᵢ = max(0, Pᵢ)',purpose:L('正数保持原值，负数变为 0。结果取决于输入的正负，因此 ReLU 在线性层之间加入非线性。','Positive values stay unchanged; negative values become zero. The result depends on the input sign, so ReLU adds nonlinearity between linear layers.')});
for(const id of ['layernorm2','ln-attn','ln-ffn','ln-cross'])Object.assign(cards[id],{
 english:'LAYERNORM',family:L('基础算子 · 归一化','Basic operator · Normalization'),
 ability:L('去掉共同偏移，再统一尺度','Remove a shared offset, then normalize scale'),
 formula:'LN(x) = γ ⊙ ((x − μ) / √(σ² + ε)) + β',
 purpose:L('投影和残差相加会改变特征的尺度与偏移。LayerNorm 用每个词自己的均值和标准差调整特征，便于后续层处理。原版 Transformer 在残差相加后使用 LayerNorm。“先相加，再归一化”关练习这两步。','Projections and residual addition can change feature scale and offset. LayerNorm adjusts each token’s features using its own mean and standard deviation. The original Transformer applies LayerNorm after residual addition. Add first, normalize second practices these two steps.'),
 weight:L('γ = 1，β = 0（本例固定）','γ = 1, β = 0 (fixed here)'),
 trait:L('每个 token 独立归一化','Normalize each token independently')
});
M.layerNormGuide={
 center:L('1 · 减去均值','1 · Subtract the mean'),
 scale:L('2 · 除以标准差','2 · Divide by the standard deviation'),
 affine:L('3 · 按特征缩放与平移','3 · Scale and shift each feature'),
 parameters:L('γ 和 β 是可学习参数；本例固定为 1 和 0，所以这一步保持上面的结果。ε = 0.000001，防止所有特征相同时除以 0。','γ and β are learnable parameters. Here they are fixed at 1 and 0, so this step leaves the result above unchanged. ε = 0.000001 prevents division by zero when all features are equal.'),
 input:L('输入特征','Input features'),output:L('归一化结果','Normalized result'),parameterTitle:L('归一化参数','Normalization parameters')
};
I18n.prepare(M.layerNormGuide);
for(const id of ['relu-ffn','original-ffn-block'])cards[id].formula='FFN(x) = ReLU(xW1 + b1)W2 + b2';
for(const id of ['single-attention','single-cross-attention','two-head-attention']){
 cards[id].formula=id==='two-head-attention'?'Y = Concat(head A, head B) × Wo':'Y = softmax(QKᵀ / √'+(id==='single-attention'?'2':'4')+') × V × Wo';
 cards[id].purpose=M.packages[id].note;
}
for(const id of maskedProjectionIds){
 const role=id.slice(-1);cards[id].formula=role.toUpperCase()+'_'+id.split('-')[1].toUpperCase()+' = X × W (4 × 2)';
 cards[id].purpose=role==='q'?L('Q 表示当前词用来匹配的信息，与同一个头的 K 点乘得到打分。','Q represents the information a token uses for matching. Dot it with K from the same head to obtain scores.'):role==='k'?L('K 提供供查询匹配的信息；同一个头的 Q 与所有词的 K 比较，得到一张打分表。','K supplies information for query matching. Compare the same head’s Q against every token’s K to obtain a score map.'):L('V 提供要读取的内容。由同一个头的注意力权重加权汇合，不经过遮罩和 Softmax。','V supplies the content to read. Aggregate it using the same head’s attention weights; V bypasses the mask and Softmax.');
}
cards['wide-concat'].formula='C[p] = [headA[p], headB[p]]';
cards.position.formula='X[p] = √2 · Embedding[p] + [sin(p), cos(p)]';
for(const id of ['wide-position','source-embed','target-embed'])cards[id].formula='X[p] = 2 · Embedding[p] + [sin(p), cos(p), sin(p/100), cos(p/100)]';
cards['wide-embedding'].formula='X[p] = EmbeddingTable[token_id[p]]';
cards['wide-embedding'].purpose=L('本游戏中，“猫”的 ID 为 2，查表得到 [1, 1, 0.4, 0.2]。这是同一个词的四个坐标，不是四个词，也不是四个预先命名的属性。位置编码模块将位置信息加到这四个数上。','In this game, the token “cats” has ID 2 and retrieves [1, 1, 0.4, 0.2]. These are four coordinates of one token, not four tokens or predefined named properties. The position-encoding module adds position information to these four values.');
for(const id of ['silu-ffn','swiglu-ffn','swiglu-residual','modern-block']){cards[id].family=L('现代扩展','Modern extensions');cards[id].purpose=M.packages[id].note;}
cards['silu-ffn'].formula='F = SiLU(XW1)W2';
cards['swiglu-ffn'].formula='F = (SiLU(XWg) ⊙ XW)Wd';
cards['swiglu-residual'].formula='Y = R + SwiGLU(Z)';
cards['modern-block'].formula='R1 = X + Attention(RMSNorm(X)); Y = R1 + SwiGLU(RMSNorm(R1))';
cards['original-transformer'].formula='L = Vocabulary(Decoder(TargetEmbedding, Encoder(SourceEmbedding)))';cards['original-transformer'].purpose=M.packages['original-transformer'].note;
cards['encoder-add'].formula='R = X + MHA(X)';cards['ffn-add'].formula='R = A + FFN(A)';
cards['original-score'].formula='S = QKᵀ / √4';cards['original-aggregate'].formula='O = AV';cards['original-head'].formula='L = DW_vocab + b';
cards['original-head'].ability=L('用四个特征，为六个候选词打分','Use four features to score six candidates');
cards['original-head'].meaning=L('D 来自解码器最后一个 LayerNorm。每个目标位置的四个特征，分别与 W_vocab 的六列权重相乘求和，再加各列偏置；第 j 列的结果就是词表中第 j 个候选词的分数。','D comes from the decoder’s final LayerNorm. For each target position, take the dot product of its four features with each of the six W_vocab columns, then add that column’s bias. Column j produces the score for vocabulary candidate j.');
cards['original-head'].meaning=withReadoutWidth(cards['original-head'].meaning);
cards['original-head'].purpose=L('词表打分、完整模型组装和选词练习使用同一套 4 × 6 权重和偏置。词表投影只输出分数，不输出概率或已选词。取最高分是后续的选词步骤。','Vocabulary scoring, full-model assembly and token selection use the same 4 × 6 weights and biases. Vocabulary projection outputs scores, not probabilities or selected tokens. Maximum selection is a separate later step.');
cards['wide-invert'].formula='Y = −D';
cards.pick={...previousCard('pick'),seal:'max',ability:L('比较六个分数，留下得分最高的词','Compare six scores and keep the highest-scoring token'),
 meaning:L('每个目标位置有六个候选分数，按“我、爱、猫、你、恨、狗”的词表顺序排列。逐个比较并保留当前最高分，最后输出获胜候选的词表 ID；三个位置分别比较，不会把不同位置的分数混起来。','Each target position has six candidate scores, ordered I, love, cats, you, hate and dogs. Compare them while retaining the current maximum, then output the winning candidate’s vocabulary ID. Compare the three positions separately.'),
 purpose:L('例如 [1, 0, 0.25, −1, 0, −1.75] 中最高分是 1，对应“我”，因此输出它的 ID 0。输出的是候选编号，不是最高分本身；并列时保留词表中先出现的候选。取 Softmax 后再找最大值会选中同一个词，本关直接比较分数。','For [1, 0, 0.25, −1, 0, −1.75], the maximum is 1 for I, so the output is its ID 0. Output the candidate ID, not the score itself; ties keep the earlier vocabulary entry. Taking the maximum after Softmax would select the same token, so this lesson compares scores directly.')};
cards['original-q'].formula='Q = DWq';cards['original-k'].formula='K = EWk';cards['original-v'].formula='V = EWv';cards['original-output'].formula='Y = OWo';
cards['original-q'].meaning=L('从目标 D 的四个特征产生查询 Q，用来决定当前目标位置想从源句取什么信息。','Project queries Q from the target D’s four features to select source information for each target position.');
for(const id of ['original-k','original-v'])cards[id].meaning=L('从编码器输出 E 的四个特征产生键或值，供目标侧查询匹配和读取。Q 来自目标，K/V 来自源句。','Project keys or values from encoder E’s four features. Target queries match and read them: Q comes from the target, K/V from the source.');
cards['original-weights'].formula='A = softmax(QKᵀ / √2)';cards.concat.formula='C[p] = [headA[p], headB[p]]';
Object.assign(cards['original-weights'],{
 input:L('输入向量矩阵 X','Input vector matrix X'),output:L('注意力权重矩阵 A','Attention-weight matrix A'),
 ability:L('计算读取各词 V 的比例','Compute the proportions for reading each token’s V'),weight:L('内部 Wq、Wk · 各 2 × 2','Internal Wq, Wk · each 2 × 2'),
 meaning:L('接收 X，内部先算 Q = XWq、K = XWk，再对匹配分数 S = QKᵀ / √2 做 Softmax，得到 3 × 3 的 A。每个格子表示某个查询词读取某个词 V 向量的比例，同一查询的三个比例合计为 1。','Receives X, computes Q = XWq and K = XWk, then applies Softmax to matching scores S = QKᵀ / √2 to obtain the 3 × 3 matrix A. Each cell is the proportion for a query to read one token’s V vector; the three proportions for each query sum to 1.'),
 purpose:M.levels['pack-scores'].rationale.takeaway
});
for(const id of ['head-a','head-b'])cards[id].formula='head = softmax(QhKhᵀ / √1)Vh';
cards['self-heads'].formula='MHA(X) = Concat(headA, headB)Wo';cards['masked-heads'].formula='head = softmax(mask(QhKhᵀ / √2))Vh';
cards['cross-heads'].formula='Q = DWq; K = EWk; V = EWv';cards['cross-add'].formula='C = D + CrossAttention(D, E)';
cards['original-encoder'].formula='A = LN(X + MHA(X)); E = LN(A + FFN(A))';
cards['original-decoder'].formula='A = LN(D + MaskedMHA(D)); B = LN(A + Cross(A,E)); Y = LN(B + FFN(B))';
const scorePurpose=L('点积把多个特征的乘积相加。特征越多，分数通常波动越大；先除以 √dₖ，避免仅因维度变大就让后面的 Softmax 过早集中到一个位置。Softmax 接着把分数变成和为 1 的比例，两步作用不同。','A dot product adds feature-wise products. More features usually mean a wider score spread. Dividing by √dₖ avoids making Softmax overly concentrated merely because the head is wider. Softmax then turns scores into proportions that sum to 1; the two steps serve different purposes.');
for(const [id,width]of [['score',2],['original-score',4]]){
 const meaning=width===2?L('每个位置的 Q，分别与各位置的 K 对应相乘再求和。这里每个 Q/K 有 2 个特征，所以 dₖ = 2，再把每个点积除以 √2。输出 S 仍是匹配分数，可以为负。','For each position’s Q, multiply matching features with each K and sum. Here every Q/K has two features, so dₖ = 2 and each dot product is divided by √2. S still contains matching scores, which may be negative.'):
  L('每个位置的 Q，分别与各位置的 K 对应相乘再求和。这里每个 Q/K 有 4 个特征，所以 dₖ = 4，再把每个点积除以 √4 = 2。输出 S 仍是匹配分数，可以为负。','For each position’s Q, multiply matching features with each K and sum. Here every Q/K has four features, so dₖ = 4 and each dot product is divided by √4 = 2. S still contains matching scores, which may be negative.');
 cards[id]={...(cards[id]||previousCard(id)),ability:L('先算匹配，再缩放分数','Compute matches, then scale the scores'),formula:'S = QKᵀ / √dₖ   (dₖ = '+width+')',meaning,purpose:scorePurpose,trait:L('缩放点积','Scaled dot products')};
 M.definitions[id].hint=meaning;
}
cards.softmax={...previousCard('softmax'),ability:L('把一组分数变成一组比例','Turn scores into proportions'),formula:'Aⱼ = exp(Sⱼ) / Σₜ exp(Sₜ)',
 meaning:L('对同一个查询的三个分数分别取指数，再除以这些指数的总和。输出是非负比例，总和为 1，供后面按比例读取 V。','Exponentiate the three scores for one query, then divide each exponential by their sum. The outputs are nonnegative proportions summing to 1, ready to read V in proportion.'),
 purpose:L('Softmax 对分数差距敏感：输入 [0, 2] 得到约 [12%, 88%]，先除以 √2 再算则约为 [20%, 80%]。两次都和为 1，但比例不同；前一步缩放不会被这一步抵消。','Softmax responds to score differences: [0, 2] gives about [12%, 88%], while dividing by √2 first gives about [20%, 80%]. Both sum to 1 but allocate differently; Softmax does not cancel the preceding scale.')};
M.levels.scores.rationale.takeaway=L('例如 Q“我” = [1, 0.25]，K“爱” = [0.2, 1]，交点算 (1×0.2 + 0.25×1) / √2 ≈ 0.32。除以 √2 是因为每个 Q/K 有两个特征。0.32 是分数，不是 32% 的比例，也不是模型权重。','For example, Q for I is [1, 0.25] and K for love is [0.2, 1]. Their cell computes (1×0.2 + 0.25×1) / √2 ≈ 0.32. The divisor is √2 because each Q/K vector has two features. 0.32 is a score, not a 32% weight or a model parameter.');
M.scoreGuide={
 title:L('先看一个查询怎样给键打分','Follow one query as it scores the keys'),
 query:L('查询 Q','Query Q'),key:L('匹配的键 K','Matching key K'),dot:L('点积 Q · K','Dot product Q · K'),scaled:L('缩放分数 S','Scaled score S'),
 shape:L('Kᵀ 把 K 转置，让每个 Q 能一次匹配所有 K；结果的每个格对应一对查询与键。','Kᵀ transposes K so each Q can match every K at once. Each output cell corresponds to one query–key pair.'),
 dimension:L('dₖ 是每个注意力头中，一个 Q 或 K 向量的特征数。这里是','dₖ is the feature count of one Q or K vector in this attention head. Here it is'),
 compare:L('下一步对照：再经过 Softmax','Next-step comparison: apply Softmax'),
 before:L('不缩放 → Softmax','Unscaled → Softmax'),after:L('缩放后 → Softmax','Scaled → Softmax'),
 boundary:L('本模块只输出分数 S。下方将两组分数分别送入 Softmax，用于比较注意力权重。','This module outputs only scores S. Below, each set of scores is passed to Softmax to compare the attention weights.'),
 explanation:L('Softmax 先取指数，再除以指数的总和。因此，缩小分数差距会改变比例；它不是把原分数简单除以总和。两组比例都和为 1，但集中程度不同。','Softmax exponentiates scores before dividing by the sum of exponentials. Reducing score differences therefore changes the proportions; it is not simply dividing the original scores by their sum. Both distributions sum to 1 but differ in concentration.'),
 shortcut:L('一个更简单的对照：','A simpler comparison:'),
 reason:L('为什么用平方根？','Why the square root?'),
 theory:L('一个 dₖ 维点积是 dₖ 个乘积的和。假设 Q/K 各分量近似独立，均值为 0，方差为 1。那么点积的方差约为 dₖ，标准差约为 √dₖ。除以 √dₖ 后，方差回到约 1。缩放控制典型波动幅度，不保证每个分数都落在固定范围内。','A dₖ-dimensional dot product sums dₖ products. Assume approximately independent Q/K components with mean 0 and variance 1. The dot product then has variance about dₖ and standard deviation about √dₖ. Dividing by √dₖ brings the variance back to about 1. Scaling controls typical spread; it does not bound every score to a fixed range.'),
 training:L('Softmax 的比例过早接近 0 或 1 时，它对分数变化的响应会变小，训练也更难调整注意力。缩放是为了避免仅因维度增加就出现这种情况，并不要求注意力始终平均分配。','When Softmax proportions approach 0 or 1 too early, they become less responsive to score changes, making attention harder to adjust during training. Scaling avoids this happening merely because dimensionality increased; it does not require attention to stay uniform.'),
 full:L('查看完整 Q、K 和打分结果','View the full Q, K and score matrices')
};
I18n.prepare(M.scoreGuide);
const aggregateMeaning=L('A 是 Softmax 算出的注意力比例，V 是 V 投影产生的内容向量。AV 就是 A × V：一个比例乘一整组 V，再把这些贡献相加，得到输出 O。','A contains the attention proportions computed by Softmax. V contains the content vectors from the V projection. AV means A × V: scale each whole V vector by its proportion, then add the contributions to obtain O.');
const aggregatePurpose=L('这是原版 Transformer 的 Attention 公式里，Softmax 后面“乘 V”的一步。只做 V 投影时，各位置仍各自加工自己的特征；A × V 让当前查询能读取其他位置的内容。每个查询各有一组比例，分别算出自己的输出，之后才做输出投影 Wo。','This is the multiplication by V after Softmax in the original Transformer’s Attention formula. A V projection transforms each position separately; A × V lets a query read content from other positions. Each query has its own proportions and output. The output projection Wo comes afterward.');
for(const [id,wide]of [['aggregate',false],['original-aggregate',true]]){
 M.definitions[id].name=wide?L('A × V · 汇合四维内容','A × V · mix four-feature values'):L('A × V · 汇合内容','A × V · mix values');
 M.definitions[id].hint=aggregateMeaning;
 cards[id]={...(cards[id]||previousCard(id)),seal:'AV',english:'ATTENTION · WEIGHTED SUM',ability:L('按注意力权重汇合 V','Mix V with attention weights'),formula:'O = A × V',meaning:aggregateMeaning,purpose:aggregatePurpose,weight:L('无','None'),trait:L('矩阵乘法 · 加权求和','Matrix product · weighted sum')};
}
M.aggregateGuide={
 flowTitle:L('A × V 在注意力中的位置','Where A × V fits in attention'),
 a:L('A · 注意力权重','A · attention weights'),v:L('V · 内容向量','V · content vector'),o:L('O · 汇合结果','O · aggregation result'),
 scorePath:L('Q/K → 打分 → Softmax','Q/K → scores → Softmax'),
 crossScorePath:L('目标 Q / 源句 K → 打分 → Softmax','Target Q / source K → scores → Softmax'),
 maskedScorePath:L('Q/K → 打分 → 遮罩 → Softmax','Q/K → scores → mask → Softmax'),
 valuePath:L('输入特征 → V 投影','Input features → V projection'),
 crossValuePath:L('源句特征 → V 投影','Source features → V projection'),
 dynamic:L('A 是随输入算出的比例，不是另一块固定的投影权重。','A is computed from the input, not a fixed projection weight matrix.'),
 title:L('用两个 V 向量计算加权和','Compute a weighted sum of two V vectors'),
 simple:L('简化例子：假设某个查询得到的比例是 25% 和 75%。','Simplified example: suppose one query assigns 25% and 75%.'),
 proportion:L('A · 比例','A · proportion'),content:L('V · 内容向量','V · content vector'),contribution:L('这份贡献','Contribution'),
 sum:L('将各项贡献相加','Add the contributions'),
 whole:L('25% 乘第一个 V 的两个数，75% 乘第二个 V 的两个数；再把对应特征相加。结果是一组新向量，不是选出其中一个词。','Multiply both features of the first V by 25%, and both features of the second by 75%, then add matching features. The result is a new vector, not a selected word.'),
 actual:L('展开游戏里的三位置计算','Show the game’s three-position example'),
 rounding:L('上方为简化例子；展开后使用示例关卡的真实计算值 · 数字已四舍五入','The example above is simplified; expand for the example level’s actual calculation · Values are rounded'),
 query:L('正在为这个查询汇合：','Combining content for query:'),
 sources:L('内容来自：','Content sources:'),
 rest:L('上面只算这个查询的输出。其余查询各用自己的比例计算，三组输出合起来才是完整的 O。','The calculation above produces one query’s output. Each other query uses its own proportions; the three outputs together form O.'),
 dimensions:L('A 的三个数对应三个内容来源；每个比例乘完整的 V，输出保留 V 的特征数。','Each of the three proportions in A belongs to a content source. It scales the whole V vector, so the output keeps V’s feature count.')
};
I18n.prepare(M.aggregateGuide);
for(const [id,width]of [['output',2],['original-output',4]]){
 const meaning=width===2?
  L('O 是前一步 A × V 汇合得到的内容向量。Wo 用不同的权重组合 O 中的两个数，得到两个新的输出特征 Y。这里仍是 2 维 → 2 维，改变的是特征的组合方式。','O is the content vector produced by the preceding A × V step. Wo combines its two values with different weights to produce two new output features Y. The width stays 2 → 2; what changes is how the features are combined.'):
  L('O 是前一步 A × V 汇合得到的内容向量。Wo 用不同的权重组合 O 中的四个数，得到四个新的输出特征 Y。这里仍是 4 维 → 4 维，改变的是特征的组合方式。','O is the content vector produced by the preceding A × V step. Wo combines its four values with different weights to produce four new output features Y. The width stays 4 → 4; what changes is how the features are combined.');
 M.definitions[id].hint=meaning;
 cards[id]={...(cards[id]||previousCard(id)),seal:L('输','O'),english:'OUTPUT PROJECTION',family:I18n.t('投影'),ability:L('重新组合 Attention 取回的内容','Recombine the content retrieved by attention'),formula:'Y = O × Wo',meaning,
  purpose:L('Wo 是注意力末端的线性投影。每一列权重将输入特征组合成一个输出特征。真实模型通过训练学习这些权重，本游戏使用固定教学数值。Y 是注意力输出；完整子层还会将它加回该子层输入。','Wo is the linear projection at the end of attention. Each weight column combines input features into one output feature. Real models learn these weights during training; the game uses fixed teaching values. Y is the attention output. A complete sublayer adds it back to that sublayer’s input.'),
  trait:L('同一位置的线性组合','Linear combination within one position'),inputSymbol:'O'};
}
M.outputProjectionGuide={
 title:L('一列权重，组合出一个输出','One weight column produces one output'),
 flowTitle:L('V 汇合与输出投影的区别','V aggregation versus output projection'),
 aggregate:L('A × V：从不同词那里汇入内容。','A × V: gather content from different tokens.'),
 projection:L('Wo：重新组合当前词向量中的特征。','Wo: recombine features within the current token’s vector.'),
 concat:L('先把各个注意力头的输出拼接起来。','First concatenate the outputs of the attention heads.'),
 input:L('汇合得到的内容','Aggregated content'),concatInput:L('各头拼接的内容','Concatenated head outputs'),output:L('注意力输出','Attention output'),
 column:L('Wo 的列','Wo column'),feature:L('输出特征','Output feature'),word:L('对应词','Token'),
 recipe:L('每个输出特征都可以使用全部输入特征。各列使用不同的权重，因而表示不同的线性组合。','Each output feature can use all input features. Different columns use different weights and therefore represent different linear combinations.'),
 sameWeights:L('其他词各自做同样的计算，使用同一个 Wo。','Other tokens are processed independently with the same Wo.'),
 full:L('查看本关三个词的输入与输出','Show this lesson’s inputs and outputs for all three tokens'),
 multiheadTitle:L('Wo 如何组合多个头的特征','How Wo combines features from multiple heads'),
 multihead:L('多头注意力先把各个头的输出拼成一个向量，再乘 Wo。每个新的输出特征都可以同时使用多个头的分量。即使拼接后已经是模型需要的宽度，Wo 仍然负责重组各个头带回的信息。','Multi-head attention concatenates the head outputs into one vector, then multiplies by Wo. Each new output feature can use components from several heads. Even when concatenation already gives the model’s required width, Wo still recombines the information retrieved by the heads.')
};
I18n.prepare(M.outputProjectionGuide);
cards.mask={number:'',seal:L('遮','Mask'),english:'CAUSAL MASK',family:L('注意力','Attention'),color:'#9a813d',
 ability:L('阻止当前位置读取未来词','Block each position from reading future tokens'),
 formula:'Q/K scores → mask → Softmax',
 meaning:L('输入是 Q/K 算出的匹配分数，还不是注意力比例。对于“爱”这个位置，保留它对“我、爱”的分数，把对后面的“猫”的分数改成 −∞。其他位置各自按同样规则处理。','The input contains Q/K matching scores, not attention weights yet. For love, keep the scores for I and love and replace the score for the later cats with −∞. Apply the same rule independently at every position.'),
 purpose:L('例如分数 [2, 1, 3] 变成 [2, 1, −∞]，再经 Softmax 得到约 [0.73, 0.27, 0]。这是一组便于说明的数字，不是本关答案。遮罩限制可见范围，Softmax 才负责把可见分数变成合计为 1 的比例；分数设成 0 并不等于屏蔽。','For example, scores [2, 1, 3] become [2, 1, −∞], then Softmax produces about [0.73, 0.27, 0]. These illustrative numbers are not the lesson’s target. The mask limits visibility; Softmax converts allowed scores into weights summing to 1. Setting a score to 0 does not block it.'),
 input:L('Q/K 匹配分数','Q/K matching scores'),output:L('遮住未来后的分数','Scores with future positions masked'),weight:L('无','None'),trait:L('保留自己与过去','Keep self and earlier positions'),stage:0};
I18n.prepare(cards);
M.original={relu,ln,position,embed,wide,project,vocabulary,classicFFN,sequenceFFN,head,concat,multi,singleAttention,encoder,decoderSelf,decoder,sourceIds,
 exampleLabels(levelId){return {Oa:L('自注意力更新','Self-attention update'),Om:L('因果注意力更新','Masked attention update'),Oc:L('交叉注意力更新','Cross-attention update'),A:['relu','original-ffn'].includes(levelId)?L('ReLU 输出','ReLU output'):L('注意力权重','Attention weights'),E:L('编码器输出','Encoder output'),D:L('目标特征','Target features'),Xs:L('源词向量与位置','Source embeddings and positions'),Xt:L('目标词向量与位置','Target embeddings and positions'),Xp:L('含位置的特征','Position-enriched features'),H_A:L('头 A 的结果','Head A output'),H_B:L('头 B 的结果','Head B output'),C:L('拼接结果','Concatenated features'),X1:levelId==='encoder-layer'?L('自注意力子层的输出','Self-attention sublayer output'):L('第一次归一化','First normalized state'),X2:L('交叉注意力后的特征','Features after cross-attention'),R1:L('Attention 残差结果','Attention residual result'),R2:L('FFN 残差结果','FFN residual result'),Rc:L('交叉注意力残差结果','Cross-attention residual result')};},
 card(id,levelId){
  const generation=generationCard(id,levelId);if(generation)return generation;
  if(cards[id]){
   const c={...cards[id]},decoderLevel=levelId==='decoder-layer';
   if(['output','original-output'].includes(id)&&M.levels[levelId]?.nodes.includes('concat')){
    c.inputSymbol='C';c.formula='Y = C × Wo';
    c.meaning=L('C 是把多个注意力头的输出拼接成的向量。Wo 用每一列权重，把这些分量组合成一个新的输出特征。本关两个头各给一个数，拼成两维，再经过 2 × 2 的 Wo，仍输出两维。','C concatenates the outputs of the attention heads. Each column of Wo combines those components into one new output feature. Here two heads each supply one value, giving two features; a 2 × 2 Wo recombines them into two output features.');
   }
   if(id==='original-output'&&levelId==='masked-multihead'){c.inputSymbol='C';c.formula='Y = Concat(head A, head B) × Wo';c.meaning=L('两个头各提供两个特征，拼接后每个词有四个数。4 × 4 的 Wo 重新组合这些来自不同头的特征；输出仍是每个词四个数。','Each head provides two features, so concatenation gives four numbers per token. The 4 × 4 Wo recombines features from both heads; the output retains four numbers per token.');}
   const formulas={'self-heads':'Oa = MHA(X)','masked-heads':'Om = MaskedMHA('+(decoderLevel?'D':'X')+')','encoder-add':'R1 = '+(decoderLevel?'D': 'X')+' + '+(levelId==='encoder-layer'?'Oa':'Om'),'ln-attn':'X1 = LayerNorm(R1)','cross-heads':'Oc = CrossAttention(X1, E)','cross-add':'Rc = X1 + Oc','ln-cross':'X2 = LayerNorm(Rc)','original-ffn-block':'F = FFN('+(decoderLevel?'X2':'X1')+')','ffn-add':'R2 = '+(decoderLevel?'X2':'X1')+' + F','ln-ffn':'Y = LayerNorm(R2)'};
   if(formulas[id])c.formula=formulas[id];
   if(levelId==='cross-attention'&&['original-q','original-k','original-v'].includes(id)){
    c.formula={'original-q':'Q = D × Wq','original-k':'K = E × Wk','original-v':'V = E × Wv'}[id];
    c.meaning=id==='original-q'?L('D 是因果自注意力、残差和 LayerNorm 的结果，还不是 Q。每个目标位置用自己的 D 向量乘 Wq，得到读取源句的查询 Q。','D is the result of causal self-attention, residual addition and LayerNorm, not Q yet. Multiply each target position’s D vector by Wq to produce a query Q for reading the source.'):id==='original-k'?L('E 是编码器整理后的源句特征，还不是 K。每个源位置的 E 向量乘 Wk 得到键 K，让目标的 Q 能给各个源位置打匹配分数。','E contains source features from the encoder, not K yet. Multiply each source position’s E vector by Wk to produce a key K, so target queries Q can score source positions.'):L('E 和 K 分支使用同一份源句特征，但这里乘的是另一套权重 Wv，得到内容 V。注意力比例决定从各个源位置的 V 读取多少，输出仍按目标位置排列。','This branch reads the same source features E as the K branch, but uses different weights Wv to produce content V. Attention weights determine how much to retrieve from each source position’s V; outputs remain ordered by target position.');
    c.purpose=L('D / E 是投影前的两份输入，Q / K / V 是投影后的三个结果。D 来自目标侧，E 来自源句侧；它们恰好都是 3 × 4，并不代表含义相同。','D and E are the two inputs before projection; Q, K and V are the three results after projection. D belongs to the target and E to the source. Both happen to be 3 × 4, but they have different roles.');
   }
   if(levelId==='encoder-layer'){
    if(['encoder-add','ffn-add'].includes(id)){
     c.english='RESIDUAL ADDITION';c.ability=L('把修改量加回这一段的输入','Add the update to this stage’s input');
     c.meaning=L('两个入口按同一个词、同一个特征对齐相加。例如 [1, 2] 加修改量 [0.3, −0.5]，得到 [1.3, 1.5]。这条旁路让原输入也参与结果，而不必全部经过支路的变换。','Align both inputs by token and feature, then add. For example, [1, 2] plus the update [0.3, −0.5] gives [1.3, 1.5]. This bypass lets the original input contribute directly without passing entirely through the branch transformation.');
     c.purpose=L('两段都做相加，但分别使用 Attention 残差和 FFN 残差。第一段输入是 X；第二段输入是交流后的 X1。各自把本段修改量加回本段输入，再归一化。','Both stages add, but use their own Attention and FFN residual modules. The first starts with X; the second with X1, the features after exchange. Add each stage’s update back to its input, then normalize.');
    }else if(['ln-attn','ln-ffn'].includes(id)){
     c.english='LAYERNORM';c.ability=L('分别整理每个词的四个特征','Normalize each token’s four features');
     c.meaning=L('每个词单独取自己的 4 个特征，减去这 4 个数的均值，再除以标准差。例如 [1, 1, 3, 3] 约变成 [−1, −1, 1, 1]。三个词分别计算，不把不同词混在一起。','For each token, center its four features by their mean and divide by their standard deviation. For example, [1, 1, 3, 3] becomes approximately [−1, −1, 1, 1]. Process the three tokens separately.');
     c.purpose=L('残差相加后，数值的整体偏移和大小可能变化。LayerNorm 先整理这些尺度，再交给下一段。两段用的是同一种运算，本例两块都取 γ = 1、β = 0；真实模型可以分别学习各自的参数。','Residual addition can change the overall offset and scale. LayerNorm adjusts them before the next stage. Both copies use the same operation and γ = 1, β = 0 here; real models may learn different parameters for each copy.');
    }else if(id==='self-heads'){
     c.meaning=L('两个头都读取每个词的全部 4 个输入特征，各用自己的权重投影出 2 维 Q/K/V。每个头独立打分、计算读取比例并汇合 V；最后拼接两头的结果，再经过 Wo，输出仍是 3 个词、每词 4 个特征。','Both heads read all four input features of each token and use their own weights to produce two-feature Q/K/V. Each head independently scores, computes reading weights and mixes V. Concatenate the results and apply Wo; the output still has three tokens with four features each.');
     c.purpose=L('Q/K/V、打分、Softmax、A × V、拼接与 Wo 已经在这块内部。它负责跨词读取内容；残差相加和 LayerNorm 放在外面，由你搭建。','Q/K/V, scoring, Softmax, A × V, concatenation and Wo are inside this module. It reads across tokens; residual addition and LayerNorm remain outside for you to build.');
    }else if(id==='original-ffn-block'){
     c.meaning=L('把交流后的每个词分别送过 Linear → ReLU → Linear：4 个特征扩成 8 个，经 ReLU 再组合回 4 个。所有词用同一套权重，但各自计算；“爱”的 FFN 不会在这一步读取“猫”的向量。','Send each token after exchange through Linear → ReLU → Linear: expand four features to eight, apply ReLU, and combine back to four. All tokens share the weights but are processed independently; the FFN for “love” does not read the vector for “cats” at this step.');
     c.purpose=L('注意力已经带回上下文，FFN 再对这些特征做非线性加工，产生本词的修改量。用残差把修改量加回交流后的输入，再整理尺度，才完成这一层。','Attention has brought in context. The FFN applies nonlinear processing to those features and produces an update for that token. Add it back to the input after exchange and normalize the scale to finish the layer.');
    }
   }
   return c;
  }
  if(!M.levels[levelId]?.original)return null;
  if(id==='down'&&levelId==='original-ffn')return {number:'',seal:I18n.t('归'),english:'DOWN PROJECTION',family:I18n.t('投影'),color:'#637e63',ability:L('把特征组合回两个数','Combine features back into two values'),formula:'F = A × W2',meaning:L('四个输入特征与每列的四个权重相乘再求和，形成两个输出。这里没有门控；输入来自前面接入的投影或 ReLU。','Weight and sum all four input features for each column to form two outputs. There is no gating here; the input comes from the connected projection or ReLU.'),purpose:I18n.t(zeroBias),weight:'4 × 2',trait:I18n.t('线性变换')};
  if(id==='add'&&levelId==='residual')return {number:'',seal:I18n.t('续'),english:'RESIDUAL ADDITION',family:I18n.t('残差'),color:'#a17949',ability:L('把 FFN 的修改量加回 X','Add the FFN update back to X'),formula:'Y = X + F; F = FFN(X)',input:L('原输入 X 与 FFN 输出 F','Original input X and FFN output F'),output:L('更新结果 Y','Updated result Y'),meaning:L('F 来自上一关用过的原版 FFN，内部仍是 Linear → ReLU → Linear。这里把它与同一份原输入 X 对应相加。例如 [1, 0] + [0.55, 0.35] = [1.55, 0.35]。','F comes from the previous lesson’s original FFN, still Linear → ReLU → Linear inside. Add it feature by feature to the same original input X. For example, [1, 0] + [0.55, 0.35] = [1.55, 0.35].'),purpose:L('FFN 负责算修改量，相加负责更新原输入。修改量为 0 时保持原值；为负时让对应特征减小。相加在 FFN 外，不会替代里面的 ReLU。','The FFN computes the update; addition applies it to the input. A zero update preserves a value and a negative update reduces it. Addition sits outside the FFN and does not replace its ReLU.'),weight:I18n.t('无'),trait:I18n.t('对应位置相加')};
  if(id==='add'&&levelId==='add-norm')return {number:'',seal:I18n.t('续'),english:'RESIDUAL ADDITION',family:I18n.t('残差'),color:'#a17949',ability:L('把 FFN 的修改量加回 X','Add the FFN update back to X'),formula:'R = X + FFN(X)',meaning:M.definitions.add.hint,purpose:L('先相加，再把总和交给 LayerNorm。这样归一化处理的是残差合并后的特征。','Add first, then send the sum to LayerNorm so normalization processes the merged residual features.'),weight:I18n.t('无'),trait:I18n.t('对应位置相加')};
  return null;
 }
};
function generationCard(id,levelId){
 if(['step','feedback'].includes(id)||id==='pick'&&levelId==='autoregressive'){
  const c=previousCard(id,levelId);
  c.family=L('逐词生成 · 模型外的循环','Token generation · loop around the model');
  if(id==='step')Object.assign(c,{english:'TEACHING PREDICTOR',formula:L('输入词 → 查表 → 六个候选词分数','Input token → lookup → six candidate scores'),meaning:M.definitions.step.hint,purpose:L('完整模型用编码器—解码器和词表投影计算分数。本关用固定表代替打分，只练习选词后的上下文更新。例如“猫”之后，“我”得 4 分，其他候选都得 −1 分。这不是语言规律，也不是前面模型的预测。','The full model uses the encoder–decoder and vocabulary projection to compute scores. This lesson replaces scoring with a fixed table to isolate context updates after selection. For example, after cats, I scores 4 and every other candidate scores −1. This is neither a language rule nor a prediction from the previous model.'),weight:L('固定分数表','Fixed score table'),trait:L('查表打分','Lookup scoring')});
  if(id==='pick')Object.assign(c,{meaning:L('每个位置的六个候选分数中选一个最高分，得到三个预测词。三个预测分别回答“对应输入词后面是什么”，不是一次生成三个新词。','Select the highest of six candidate scores at each position, producing three predictions. Each predicts what follows its corresponding input token; they are not three newly generated tokens.'),purpose:L('继续写已有文本时，只采用最后一个位置的预测。前两个预测仍然显示，便于比较；“追加新词”负责取出最后一个。','To continue the existing text, use only the final position’s prediction. Earlier predictions stay visible for comparison; Append a word takes the last one.')});
  if(id==='feedback')Object.assign(c,{english:'APPEND ONE TOKEN',formula:L('新上下文 = 最近三词（原上下文 + 最后一个预测）','New context = last three (old context + final prediction)'),meaning:M.definitions.feedback.hint,purpose:L('例如原文是“我 / 爱 / 猫”，三个预测是“爱 / 猫 / 我”。只追加最后的“我”，得到“我 / 爱 / 猫 / 我”。本关只保留三词，关卡出口因此是“爱 / 猫 / 我”。模块自动记住本次运行的原输入，不需要额外连线。这不是 KV 缓存。','For context I / love / cats, the predictions are love / cats / I. Append only the final I to get I / love / cats / I. This lesson retains three tokens, so the lesson outlet is love / cats / I. The module remembers the original input for this run; no extra wire is needed. This is not a KV cache.'),weight:L('无','None'),trait:L('更新文本窗口','Update text window')});
  return c;
 }
 return null;
}
M.extraCard=(id,levelId)=>{
 const own=M.original.card(id,levelId);if(own)return own;
 const c=previousCard(id,levelId);if(!M.levels[levelId]?.original)return c;
 if(id==='v')Object.assign(c,{formula:'V = X × Wᵥ',meaning:M.levels.qkv.rationale.text,purpose:M.levels.qkv.rationale.takeaway,input:L('输入矩阵 X','Input matrix X'),output:M.levels.qkv.outputTitle});
 if(id==='output')c.formula='Y = O × Wo';
 if(['residual1','residual2'].includes(id)){c.formula='R = X_sublayer + Sublayer(X_sublayer)';c.meaning=L('加回当前子层自己的输入。相加之后才进入本段的 LayerNorm。','Add back the current sublayer’s own input. LayerNorm follows the addition.');}
 if(levelId==='cross-attention'&&['q','k','v'].includes(id)){c.formula={q:'Q = DWq',k:'K = EWk',v:'V = EWv'}[id];c.meaning=id==='q'?L('从目标 D 产生查询，决定这一步想从源句取什么。','Project queries from target D to select information from the source.'):L('从编码器输出 E 产生键或值，供目标侧查询匹配和读取。','Project keys or values from encoder output E so target queries can match and retrieve source information.');}
 return c;
};
M.extraExample=(id,levelId)=>{
 const reward=M.fusions[levelId];
 if(reward?.kind===id&&!M.levels[levelId].nodes.includes(id)){
  const s=M.solutions(levelId),sampleIndex=0,input=M.samplesFor(levelId)[sampleIndex].v,sources=M.sourceValues(s,input),values=reward.inputs.map(key=>sources[key]);
  const labels=levelId==='ffn'?{R:'R',Z:'Z'}:levelId==='encoder-decoder'?{R:'D ID',Z:'S ID'}:{R:reward.inputs.length===1?'X':'D',Z:'E'};
  return {sampleIndex,exampleSource:[labels.R,input],inputs:values.map((value,i)=>[labels[reward.inputs[i]],value]),output:[({'silu-ffn':'F','swiglu-ffn':'F','original-transformer':'L','source-embed':'Xp'})[id]||'Y',M.operations[id](values)]};
 }
 if(!M.levels[levelId]?.original)return previousExample(id,levelId);
 const s=M.solutions(levelId),sampleIndex=levelId==='relu'?2:0,input=M.samplesFor(levelId)[sampleIndex].v,ev=M.evaluate(s,input),n=s.nodes.find(n=>n.id===id);
 if(!n)return null;
 const names={R:M.levels[levelId].dualSource?'D':'X',Z:levelId==='encoder-decoder'?'Source ID':'E',project:'P',relu:'A',down:'F',add:'R',layernorm2:'Y',q:'Q',k:'K',v:'V',score:'S',softmax:'A',aggregate:'O',output:'Y','original-q':'Q','original-k':'K','original-v':'V','original-score':'S','original-aggregate':'O','original-output':'Y','original-head':'L','wide-embedding':'X','wide-position':'Xp',position:'Xp','head-a':'H_A','head-b':'H_B',concat:'C','original-weights':'A','relu-ffn':'F','original-ffn-block':'F','self-heads':'O','masked-heads':'O','cross-heads':'O','cross-add':'Rc','source-embed':'Xs','target-embed':'Xt','original-encoder':'E','original-decoder':'D','encoder-add':'R1','ffn-add':'R2','ln-attn':'X1','ln-cross':'X2','ln-ffn':'Y'};
 if(['vocabulary-scores','vocabulary'].includes(levelId))names.R='D';
 for(const kind of maskedProjectionIds)names[kind]=kind.slice(-1).toUpperCase()+'_'+kind.split('-')[1].toUpperCase();
 Object.assign(names,{'wide-concat':'C','score-2':'S_B','mask-2':'S_B masked','softmax-2':'A_B','aggregate-2':'H_B'});
 if(levelId==='masked-multihead')Object.assign(names,{aggregate:'H_A','original-output':'Om'});
 names['self-heads']='Oa';names['masked-heads']='Om';names['cross-heads']='Oc';
 names.add=levelId==='add-norm'?'R':'Y';
 const ex={sampleIndex,exampleSource:[names.R,input],inputs:n.links.map(k=>[names[k]||k,ev.values[k]]),output:[names[id]||id,ev.values[id]]};
 if(id==='relu')ex.activation={kind:'relu',feature:3};
 if(M.definitions[id].type==='add')ex.addition={feature:0};
 if(M.definitions[id].type==='layernorm'){const width=M.definitions[id].ins[0]===12?4:2,v=ev.values[n.links[0]].slice(0,width),mean=v.reduce((a,b)=>a+b,0)/width,variance=v.reduce((s,x)=>s+(x-mean)**2,0)/width;ex.normalization={width,mean,variance,epsilon:eps};}
 return ex;
};
})(window.NeuralPrologue);
