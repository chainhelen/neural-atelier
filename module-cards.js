// Card copy describes the same toy operators as model.js. Numbers are computed, not baked into the art.
export const moduleCards = {
  norm: {
    number:'',seal:'r',english:'RMS NORMALIZATION',family:I18n.t('归一化'),color:'#9a813d',
    ability:I18n.t('同一组特征，除以同一个均方根'),formula:'r = √(mean(X²) + ε); Zᵢ = (Xᵢ / r) · γᵢ',
    meaning:I18n.t('先把一个 token 的各个特征平方、求平均，再开平方，得到它自己的尺度 r。随后各特征都除以这个 r；另一个 token 单独算自己的 r。'),
    purpose:I18n.t('它调整整体数值尺度，不把各个特征变成同一个数，也不减去均值。本例 γ 全为 1，缩放后仍保留原来的正负与特征间比例；真实模型还会学习每个特征的 γ。'),
    weight:'γ = 1',trait:I18n.t('逐个 token 缩放'),stage:0
  },
  project: {
    number:'01',seal:I18n.t("映"),english:'LINEAR PROJECTION',family:I18n.t("投影"),color:'#397f71',
    ability:I18n.t("用权重组合出新特征"),formula:'P = X × W',
    meaning:I18n.t("输入向量包含 2 个数。取权重矩阵的一列，分别乘输入里的两个数，再把乘积相加，得到一个输出特征。"),
    purpose:I18n.t("矩阵有 4 列，就能生成 4 个特征。同一套权重可以处理每个 token 的向量，改变的是特征的表示方式。"),
    input:I18n.t("输入 X"),output:I18n.t("投影结果 P"),weight:'2 × 4',trait:I18n.t("线性变换"),stage:1
  },
  gate: {
    number:'01', seal:I18n.t("门"), english:'GATE PROJECTION', family:I18n.t("投影"), color:'#75628e',
    ability:I18n.t("为内容准备调节系数"), formula:'G = Z × Wg',
    meaning:I18n.t("门投影也做矩阵乘法，但权重 Wg 与线性投影的 W 不同。同一份输入乘 W 得到内容特征 U，乘 Wg 得到 G，供 SiLU 计算调节系数。"),
    purpose:I18n.t("G 还要经过 SiLU，才得到系数 S；S 与 U 的对应位置相乘，形成 H。系数随输入变化，可以缩小、放大或改变特征的符号。"),
    input:I18n.t("归一化输入 Z"), output:I18n.t("门分支 G"), weight:'2 × 4', trait:I18n.t("线性变换"), stage:0
  },
  silu: {
    number:'03', seal:I18n.t("启"), english:'SiLU ACTIVATION', family:I18n.t("激活"), color:'#a8793b',
    ability:I18n.t("把输入 G 算成输出 S"), formula:'S = SiLU(G)',
    meaning:I18n.t("SiLU 是函数名。G 是门投影算出的输入向量，S 是经过 SiLU 后的输出向量。G 中的每个数各算一次，得到 S 中对应的数。"),
    purpose:I18n.t("SiLU 会把较大的负数压近 0，较大的正数则接近原值。输出 S 接着与内容特征 U 相乘，成为逐项调节特征的系数。S 仍可以为负，也可以超过 1。"),
    input:I18n.t("门分支 G"), output:I18n.t("激活后的门值 S"), weight:I18n.t("无"), trait:I18n.t("逐项非线性"), stage:1
  },
  multiply: {
    number:'04', seal:I18n.t("合"), english:'ELEMENTWISE GATING', family:I18n.t("门控"), color:'#3e8078',
    ability:I18n.t("让门值作用于内容"), formula:'Hᵢ = Sᵢ × Uᵢ',
    meaning:I18n.t("把两个向量的对应特征配对：第 1 个乘第 1 个，第 2 个乘第 2 个。4 对乘积组成新的 4 维向量 H。"),
    purpose:I18n.t("U 给出内容特征，S 根据当前输入调节各项的贡献。两者来自同一个 token 的不同投影；这里不汇合不同 token。之后的降维投影才把各项乘积加权组合。S = 1 保持原值，S = 0 消去这一项。"),
    input:I18n.t("门值 S 与内容 U"), output:I18n.t("门控结果 H"), weight:I18n.t("无"), trait:I18n.t("对应位置相乘"), stage:2
  },
  down: {
    number:'05', seal:I18n.t("归"), english:'DOWN PROJECTION', family:I18n.t("投影"), color:'#637e63',
    ability:I18n.t("汇成两维变换"), formula:'F = H × Wd',
    meaning:I18n.t("把门控后的 4 个特征乘上 4 × 2 权重矩阵。每个输出都由 4 项乘积求和，最终得到 2 维的 FFN 变换 F。"),
    purpose:I18n.t("把扩展后的特征组合回原宽度。F 与最初的输入 R 都是 2 维，下一步才能按对应位置相加。"),
    input:I18n.t("门控结果 H"), output:I18n.t("FFN 变换 F"), weight:'4 × 2', trait:I18n.t("线性变换"), stage:3
  },
  add: {
    number:'06', seal:I18n.t("续"), english:'RESIDUAL ADDITION', family:I18n.t("残差"), color:'#a17949',
    ability:I18n.t("把变化加回原输入"), formula:'Y = R + F',
    meaning:I18n.t("R 是进入 FFN 前的特征，F 是支路算出的修改量。对应位置相加得到 Y：正的修改量让数值增加，负的让它减少。"),
    purpose:I18n.t("支路可以学习需要改动的部分。无需修改的位置只要输出 0，原值就能通过直连路径继续传递；不必让支路再学一遍如何复制原值。相加也可能抵消数值，因此并不保证原信息完整保留。"),
    input:I18n.t("原始输入 R 与变换 F"), output:I18n.t("本层输出 Y"), weight:I18n.t("无"), trait:I18n.t("对应位置相加"), stage:4
  }
};

I18n.prepare(moduleCards);

export function cardFor(id,levelId){
 const original=window.NeuralPrologue.original?.card(id,levelId);if(original)return original;
 if(!moduleCards[id])return null;
 const card={...moduleCards[id]};
 if(id==='project'&&['gating','swiglu','ffn'].includes(levelId)){
  Object.assign(card,{ability:I18n.t("用熟悉的投影产生内容特征"),formula:'U = Z × W',input:I18n.t("归一化输入 Z"),output:I18n.t("内容特征 U"),stage:0,
   meaning:I18n.t("这就是序章的线性投影，仍用同一套权重 W 做矩阵乘法。这里把输出叫 U，表示待调节的内容特征；门投影使用另一套权重 Wg，产生 G。"),
   purpose:I18n.t("“内容”描述它在这条支路的用途，运算仍是线性投影。门分支经 SiLU 得到系数 S，再与 U 逐项相乘；同一个内容值会随输入产生的系数而减弱或放大。")});
 }
 if(levelId==='ffn'){
  if(id==='gate'||id==='project')card.meaning=I18n.t("输入先归一化为 Z。")+' '+card.meaning;
  return card;
 }
 if(id==='gate'||(id==='project'&&['gating','swiglu'].includes(levelId))){
  card.formula=card.formula.replace('Z','X');
  card.input=I18n.t("输入 X");
 }
 if(id==='silu'&&['activation','plain-ffn'].includes(levelId)){
  card.ability=I18n.t("把输入 P 算成输出 S");card.formula='S = SiLU(P)';
  card.input=I18n.t("投影结果 P");card.output=I18n.t("激活特征 S");
  card.meaning=I18n.t("SiLU 是函数名。P 是线性投影算出的输入向量，S 是经过 SiLU 后的输出向量。P 中的每个数各算一次，得到 S 中对应的数。");
  card.purpose=I18n.t("乘一个固定比例会把所有数一起缩放；SiLU 的缩放比例随数值变化。较大的负数被压近 0，较大的正数接近原值，因此它能改变特征之间的数值关系。");
 }
 if(id==='down'&&levelId==='down-projection'){
  card.formula='F = P × Wd';card.input=I18n.t("投影结果 P");card.meaning=I18n.t("接过投影 P 中的 4 个数，乘上 4 × 2 权重矩阵。每个输出由 4 项乘积求和，得到 2 维变换 F。");card.purpose=I18n.t("降维仍是加权求和，四个特征都可以影响每个输出。它把特征宽度变回 2，方便后续模块继续使用。");
 }
 if(id==='down'&&levelId==='plain-ffn'){
  card.formula='F = S × Wd';card.input=I18n.t("激活特征 S");card.meaning=I18n.t("SiLU 已把投影结果 P 变成 S。降维投影接收这四个激活特征，各做一次四项加权求和，输出两个特征 F。");card.purpose=I18n.t("前面先展开并激活，这里再回到原宽度，组成单支路 FFN。这个输出还没有加回原输入。");
 }
 if(id==='down'&&levelId==='swiglu')card.purpose=I18n.t("降维仍是加权求和，四个特征都可以影响每个输出。它把特征宽度变回 2，方便后续模块继续使用。");
 if(id==='gate'&&['gate-projection','pairwise'].includes(levelId)){
  card.ability=I18n.t("用第二套权重提取特征");card.formula='G = X × Wg';card.input=I18n.t("输入 X");
  card.meaning=I18n.t("它和线性投影做相同的矩阵乘法，只是使用不同的权重 Wg。输入 [1, 0] 会得到 [1, −1, 0.5, 0]，先把这组输出记作 G。");
  card.purpose=levelId==='pairwise'?I18n.t("本关直接将 G 与另一组投影结果逐项相乘。这里只练对应配对，还没有 SiLU。"):I18n.t("本关只产生 G。后面的关卡才会用它计算调节系数，现在不需要搭出整个门控网络。");
 }
 if(id==='project'&&levelId==='pairwise'){
  card.formula='U = X × W';card.output=I18n.t("内容特征 U");card.purpose=I18n.t("把这组投影结果记作 U，接下来与另一组 G 对应相乘。这里仍使用最初的线性投影权重。");
 }
 if(id==='multiply'&&levelId==='pairwise'){
  card.ability=I18n.t("把两组数对应相乘");card.formula='Hᵢ = Gᵢ × Uᵢ';card.input=I18n.t("两组投影结果 G 与 U");card.output=I18n.t("相乘结果 H");
  card.purpose=I18n.t("每个输出只取一对输入的乘积，不在不同特征之间求和。例如 [2, 3] 与 [0.5, 2] 对应相乘得到 [1, 6]。");
 }
 return card;
}

// The comparison uses the example's actual Q/K and scored output. Softmax is
// shown as a downstream comparison; it is never added to the scoring operator.
export function scoreComparison(M,example,query=2,key=0){
 const q=example.inputs[0][1],k=example.inputs[1][1],width=q.length/3;
 const queries=M.math.rows(q,width),keys=M.math.rows(k,width);
 const raw=queries.flatMap(a=>keys.map(b=>a.reduce((sum,x,i)=>sum+x*b[i],0)));
 const scaled=example.output[1],offset=query*3;
 const softmax=values=>{const exp=values.map(x=>Math.exp(x-Math.max(...values))),sum=exp.reduce((a,b)=>a+b,0);return exp.map(x=>x/sum);};
 return {width,divisor:Math.sqrt(width),query,key,q:queries[query],k:keys[key],
  raw:raw.slice(offset,offset+3),scaled:scaled.slice(offset,offset+3),
  before:M.math.probabilities(raw).slice(offset,offset+3),after:M.math.probabilities(scaled).slice(offset,offset+3),
  simple:{raw:[0,2],scaled:[0,2/Math.sqrt(width)],before:softmax([0,2]),after:softmax([0,2/Math.sqrt(width)])}};
}

// Keep each source's whole V vector together: one attention coefficient
// scales all its features, then the contributions sum feature by feature.
export function aggregationExample(M,example,query=2){
 const describe=(weights,values)=>{
  const contributions=values.map((v,i)=>v.map(x=>weights[i]*x));
  return {weights,values,contributions,output:values[0].map((_,j)=>contributions.reduce((sum,v)=>sum+v[j],0))};
 };
 const width=example.inputs[1][1].length/3,values=M.math.rows(example.inputs[1][1],width),weights=M.math.rows(example.inputs[0][1],3)[query];
 return {width,query,...describe(weights,values),simple:describe([.25,.75],[[2,0],[0,4]])};
}

export function exampleFor(M,id,levelId='ffn'){
  if(['workshop','residual'].includes(levelId))return M.extraExample(id,levelId);
  if(['down-projection','plain-ffn','gate-projection','pairwise'].includes(levelId)){
   const input=[1,0],state=M.solutions(levelId),ev=M.evaluate(state,input),node=state.nodes.find(n=>n.id===id);
   const labels={R:'X',project:levelId==='pairwise'?'U':'P',gate:'G',silu:'S',multiply:'H',down:'F'};
   return {sampleIndex:0,exampleSource:['X',input],inputs:node.links.map(k=>[labels[k],ev.values[k]]),output:[labels[id],ev.values[id]],...(id==='silu'?{activation:{feature:2}}:{})};
  }
  const shared=['gating','swiglu','ffn'].includes(levelId),compare=['gate','project'].includes(id)&&['gating','swiglu','ffn'].includes(levelId),R=shared?[1,0]:[1,1],Z=levelId==='ffn'?M.norm(R):R,P=M.linear(R,M.weights.project);
  const meta={sampleIndex:shared?0:2,exampleSource:[levelId==='ffn'?'R':'X',R]};
  if(id==='project'&&!compare)return {...meta,inputs:[['X',R]],output:['P',P]};
  if(levelId==='activation')return {...meta,inputs:[['P',P]],output:['S',P.map(M.silu)],activation:{feature:2}};
  const G=M.linear(Z,M.weights.gate),U=M.linear(Z,M.weights.up),S=G.map(M.silu),H=S.map((x,i)=>x*U[i]),F=M.linear(H,M.weights.down),Y=R.map((x,i)=>x+F[i]),source=levelId==='ffn'?'Z':'X';
  const paths={gate:[[[source,Z]],['G',G]],project:[[[source,Z]],['U',U]],silu:[[['G',G]],['S',S]],multiply:[[['S',S],['U',U]],['H',H]],down:[[['H',H]],['F',F]],add:[[['R',R],['F',F]],['Y',Y]]};
  return {...meta,inputs:paths[id][0],output:paths[id][1],...(compare?{comparison:{input:[source,Z],rows:[['U',U],['G',G],['S',S],['H',H]],feature:2}}:{}),...(id==='silu'?{activation:{feature:2,content:U,product:H}}:{}),...(id==='add'?{addition:{feature:0}}:{})};
}
