// Scene objects represent weights, features and token rows; math stays in model.js.
export function createSceneArt(T,scene,colors,{anisotropy=1}={}){
 const context=new T.Group(),floor=new T.Group();floor.userData.environmentFloor=true;scene.add(floor,context);
 const units=new Map(),socketRings=new WeakMap(),sharedSphere=new T.SphereGeometry(.115,18,14),rows=[],tokens=[],outputHeaders=[];
 const rowColor=i=>[0x397aa2,0x8862aa,0x288778][i]??0x557e6b;
 const featureColor=i=>[0x397aa2,0x8862aa,0xb68036,0x288778][i%4];
 const projectionColor=0xb77724;
 const pairColors=[0x8b6ea5,0x368d77];
 const additionColors=[0x397aa2,0xb68036];
 const rowZ=i=>-1.05+i*.96;
 const sourceOffset=(layout,id)=>layout.sourceOffsets?.[id]??(layout.dualSource?(id==='Z'?-3:3):0);
 const tokenOperation=d=>d.type==='mask'&&d.out===9?'mask':d.type==='embedding'&&d.ins[0]===3?'lookup':d.type==='position'&&d.ins[0]===d.out?'position':null;
 const positionalValue=(p,j,width)=>(j%2?Math.cos:Math.sin)(p/10000**(2*Math.floor(j/2)/width));
 sharedSphere.userData.shared=true;
 const mat=(color,extra={})=>new T.MeshStandardMaterial({color,roughness:.5,metalness:.12,...extra});
 function box(parent,w,h,d,x,y,z,color,extra={}){const mesh=new T.Mesh(new T.BoxGeometry(w,h,d),mat(color,extra));mesh.position.set(x,y,z);mesh.castShadow=true;mesh.receiveShadow=true;parent.add(mesh);return mesh;}
 function slab(parent,w,d,h,x,y,z,color){const r=.12,s=new T.Shape(),a=-w/2,b=-d/2;s.moveTo(a+r,b);s.lineTo(a+w-r,b);s.quadraticCurveTo(a+w,b,a+w,b+r);s.lineTo(a+w,b+d-r);s.quadraticCurveTo(a+w,b+d,a+w-r,b+d);s.lineTo(a+r,b+d);s.quadraticCurveTo(a,b+d,a,b+d-r);s.lineTo(a,b+r);s.quadraticCurveTo(a,b,a+r,b);const g=new T.ExtrudeGeometry(s,{depth:h,bevelEnabled:true,bevelThickness:.035,bevelSize:.035,bevelSegments:2,steps:1});g.rotateX(-Math.PI/2);const mesh=new T.Mesh(g,mat(color));mesh.position.set(x,y,z);mesh.castShadow=true;mesh.receiveShadow=true;parent.add(mesh);return mesh;}
 function tube(parent,points,color,r=.022,opacity=1){const curve=new T.CatmullRomCurve3(points.map(p=>new T.Vector3(...p)));const mesh=new T.Mesh(new T.TubeGeometry(curve,18,r,6,false),mat(color,{transparent:opacity<1,opacity}));parent.add(mesh);return mesh;}
 function ring(parent,r,x,y,z,color,thickness=.018){const mesh=new T.Mesh(new T.TorusGeometry(r,thickness,8,36),mat(color));mesh.rotation.x=-Math.PI/2;mesh.position.set(x,y,z);parent.add(mesh);return mesh;}
 function neuron(parent,x,y,z,color){const b=new T.Mesh(sharedSphere,mat(color,{emissive:color,emissiveIntensity:.08}));b.position.set(x,y,z);b.castShadow=true;parent.add(b);socketRings.set(b,ring(parent,.155,x,y-.13,z,color,.012));return b;}
 function textTile(parent,text,w,d,x,y,z,{color='#314941',bg='#edf3e8',size=52}={}){
  // Match the physical tile's aspect ratio so letters are never stretched to fit.
  const canvas=document.createElement('canvas'),height=Math.min(384,2048*d/w);
  canvas.width=Math.round(height*w/d);canvas.height=Math.round(height);
  const ctx=canvas.getContext('2d');let fontFraction=size/128;
  function draw(value,tone){
   ctx.fillStyle=bg;ctx.fillRect(0,0,canvas.width,canvas.height);
   ctx.fillStyle=tone;ctx.textAlign='center';ctx.textBaseline='middle';
   let fontSize=canvas.height*size/128;
   const font=()=>`600 ${fontSize}px -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif`;
   ctx.font=font();
   fontSize*=Math.min(1,canvas.width*.88/Math.max(1,ctx.measureText(String(value)).width));
   fontFraction=fontSize/canvas.height;
   ctx.font=font();ctx.fillText(String(value),canvas.width/2,canvas.height*.52);
  }
  draw(text,color);
  const texture=new T.CanvasTexture(canvas);texture.colorSpace=T.SRGBColorSpace;texture.anisotropy=anisotropy;
  const mesh=new T.Mesh(new T.PlaneGeometry(w,d),new T.MeshBasicMaterial({map:texture,toneMapped:false}));
  mesh.rotation.x=-Math.PI/2;mesh.position.set(x,y,z);parent.add(mesh);
  if(parent.userData.node)mesh.userData.detailText={height:d,fontFraction};
  let previousText=text,previousTone=color;
  return {mesh,ctx,texture,write(value,tone=color){if(value===previousText&&tone===previousTone)return;previousText=value;previousTone=tone;draw(value,tone);if(mesh.userData.detailText)mesh.userData.detailText.fontFraction=fontFraction;texture.needsUpdate=true;}};
 }
 const detailBottom=new T.Vector3(),detailTop=new T.Vector3();
 const detailModes={board:true,card:true};let detailScope='board';
 function updateDetail(camera,width,height,boardScale=null){
  scene.updateMatrixWorld(true);camera.updateMatrixWorld();
  const labels=[],modules=new Set();let smallest=Infinity;
  scene.traverse(mesh=>{
   const detail=mesh.userData.detailText;if(!detail)return;
   // Hidden modules must not affect a card's close-up. The label's own
   // visibility is controlled below, so still measure it when abstracted.
   for(let parent=mesh.parent;parent;parent=parent.parent)if(!parent.visible)return;
   labels.push(mesh);modules.add(mesh.parent);
  });
  if(!labels.length)return;
  for(const module of modules){
   if(context.userData.layout?.floating&&detailScope==='board'){
    const center=new T.Vector3().setFromMatrixPosition(module.matrixWorld).project(camera);
    if(Math.abs(center.x)>1.2||Math.abs(center.y)>1.2||center.z< -1||center.z>1)continue;
   }
   const span=(x,z)=>{
    detailBottom.set(-x,.45,-z).applyMatrix4(module.matrixWorld).project(camera);
    detailTop.set(x,.45,z).applyMatrix4(module.matrixWorld).project(camera);
    return Math.hypot((detailTop.x-detailBottom.x)*width/2,(detailTop.y-detailBottom.y)*height/2);
   };
   smallest=Math.min(smallest,Math.max(span(1,0),span(0,1.2)));
  }
  // 100% is each lesson's authored learning view, even on a wider board.
  // Only abstract after zooming away from that view AND making plates small.
  // Cards and the floating full-model demo still use projected size alone.
  if(!Number.isFinite(smallest))return;
  const nearDefault=detailScope==='board'&&Number.isFinite(boardScale)&&boardScale>=(detailModes.board?.8:.9);
  const visible=nearDefault||smallest>=(detailModes[detailScope]?95:105);
  detailModes[detailScope]=visible;labels.forEach(mesh=>mesh.visible=visible);
 }
 function surroundings(samples,layout={},palette={}){
  for(const parent of [floor,context])for(const o of [...parent.children]){parent.remove(o);o.traverse(c=>{if(c.geometry&&!c.geometry.userData?.shared)c.geometry.dispose();c.material?.map?.dispose();c.material?.dispose?.();});}
  tokens.length=0;rows.length=0;outputHeaders.length=0;floor.visible=!layout.floating;context.visible=true;
  const sourceX=layout.sourceX??-8.55,sinkX=layout.sinkX??9,dim=layout.sinkDim||2,w=layout.boardWidth??(layout.id==='wire'?11.6:layout.id==='projection'?15:22.1),d=layout.boardDepth??(layout.id==='ffn'||!layout.id?11:7);
  const surface=palette.surface||'#eaf0e4',edge=palette.edge||'#b7c8b9',accent=palette.accent||'#526d5e';
  if(!layout.floating){
  slab(floor,w,d,.3,0,-.45,0,palette.edge||0xaabcb1);slab(floor,w-.2,d-.2,.09,0,-.12,0,palette.edge||0xd4dfd3);slab(floor,w-.45,d-.45,.035,0,-.01,0,surface);
  for(let x=Math.ceil(-w/2+1);x<w/2-.5;x++)tube(floor,[[x,.075,-d/2+.5],[x,.075,d/2-.5]],edge,.007,.55);
  for(let z=Math.ceil(-d/2+1);z<d/2-.5;z++)tube(floor,[[-w/2+.5,.075,z],[w/2-.5,.075,z]],edge,.007,.55);
  for(const z of [-d/2+.25,d/2-.25])box(floor,w-1.3,.05,.04,0,.08,z,0x92aa9b);
  for(const x of [-w/2+.45,w/2-.45])for(const z of [-d/2+.4,d/2-.4]){const screw=new T.Mesh(new T.CylinderGeometry(.065,.065,.03,12),mat(0x728a7c));screw.position.set(x,.1,z);floor.add(screw);}
  textTile(context,layout.boardText||'FFN / SWIGLU',3.2,.48,-w/2+3.3,.1,d/2-.8,{color:accent,bg:surface,size:58});
  textTile(context,layout.shape||'2 → 4 → 2',2.5,.45,w/2-3.25,.1,d/2-.8,{color:accent,bg:surface,size:62});
  }
  const seq=!!layout.sequence,inputCols=seq?(layout.sourceDim||6)/3:2,outCols=seq?(dim===3?1:dim/3):dim;
  const sourceW=inputCols>2?3.4:seq?2.4:1.8,rackW=seq?3.2:dim===4?2.6:1.8;
  for(const sourceId of layout.dualSource?['Z','R']:['R']){
  const offset=sourceOffset(layout,sourceId),first=context.children.length;
  slab(context,sourceW,3.8,.42,sourceX,.07,offset,sourceId==='Z'?0x9d91b0:0x7fa6af);slab(context,sourceW-.14,3.64,.08,sourceX,.5,offset,0xe8f1ea);
  [I18n.t("我"),I18n.t("爱"),I18n.t("猫")].forEach((word,i)=>{
   const g=new T.Group();g.position.set(sourceX,.65,rowZ(i)+offset);g.userData.sourceRow=i;g.userData.sourceId=sourceId;context.add(g);
   const panel=slab(g,sourceW-.25,.72,.12,0,0,0,0xbfd8d3);
   const stripe=box(g,sourceW-.38,.045,.055,0,.15,.31,rowColor(i),{emissive:rowColor(i),emissiveIntensity:.1});
   const label=textTile(g,seq?samples[0].word.split(' / ')[i]:word,.65,.3,inputCols>2?-1.22:seq?-.82:-.3,.17,-.02,{size:70});
   const cells=Array.from({length:inputCols},(_,j)=>{const x=inputCols>2?-.45+j*.48:seq?-.12+j*.48:.17+j*.34;box(g,seq?.4:.24,.12,.36,x,.18,0,0x73aaa6);return textTile(g,'·',seq?.4:.24,.34,x,.25,0,{color:'#fffdf4',bg:'#73aaa6',size:72});});tokens.push({g,cells,label,panel,stripe,sourceId,index:i});
  });
  context.children.slice(first).forEach(o=>o.position.y+=layout.sourceHeights?.[sourceId]||0);
  }
  const outputStart=context.children.length;
  slab(context,rackW,3.8,.42,sinkX,.07,0,0x608f7c);slab(context,rackW-.15,3.64,.08,sinkX,.5,0,0xe8f1df);
  const labelX=seq?-1.15:dim===4?-.87:-.47;
  const pairAxes=!!layout.outputPairAxes,cw=seq?Math.min(.7,1.95/outCols):.36;
  textTile(context,pairAxes?'Q↓ K→':I18n.t("位置"),pairAxes?.75:.48,.28,sinkX+labelX,.63,-1.65,{size:72,color:'#526d5e',bg:'#e8f1df'});
  if(pairAxes)for(let j=0;j<outCols;j++){
   const header=textTile(context,samples[0].word.split(' / ')[j],cw,.28,sinkX-.55+j*(cw+.03),.63,-1.65,{size:66,color:'#526d5e',bg:'#e8f1df'});
   header.mesh.userData.scoreKey=j;outputHeaders.push(header);
  }
  samples.slice(0,3).forEach((_,i)=>{const g=new T.Group();g.position.set(sinkX,.66,rowZ(i));g.userData.sinkRow=i;context.add(g);
   const panel=slab(g,rackW-.45,.72,.08,0,0,0,0xc8d9c5);
   const stripe=box(g,rackW-.55,.035,.045,0,.105,.32,rowColor(i),{emissive:rowColor(i),emissiveIntensity:.1});
   const label=textTile(g,pairAxes?samples[0].word.split(' / ')[i]:String(i),pairAxes?.5:.28,.32,labelX,.13,0,{size:70});
   if(pairAxes)label.mesh.userData.scoreQuery=i;
   const status=textTile(g,'·',.19,.22,rackW/2-.12,.14,0,{size:82,bg:'#e8f1df'});
   const cells=Array.from({length:outCols},(_,j)=>textTile(g,'·',cw,.43,seq?-.55+j*(cw+.03):(dim===4?-.42:-.02)+j*.4,.14,0,{color:'#558974',bg:'#e9f3e2',size:51}));rows.push({g,cells,status,panel,stripe,label});
  });
  context.children.slice(outputStart).forEach(o=>o.position.y+=layout.sinkY||0);
  context.userData={samples,layout};

 }
 function module(parent,n,d,weights,{register=true}={}){
  const g=new T.Group();g.position.set(n.x,n.y||0,n.z);g.userData.node=n.id;parent.add(g);const color=colors[n.kind||n.id];
  const backdrops=new Map();
  function backdrop(object,faint=.12,keep=()=>false){
   object?.traverse(mesh=>{if(!mesh.material)return;const m=mesh.material;mesh.userData.computationBackdrop=true;backdrops.set(mesh,{mesh,faint,keep,opacity:m.opacity,transparent:m.transparent,depthWrite:m.depthWrite,castShadow:mesh.castShadow,receiveShadow:mesh.receiveShadow});});return object;
  }
  function socketBackdrop(socket,faint=.16,keep=()=>false){backdrop(socket,faint,keep);backdrop(socketRings.get(socket),.08);}
  const pairOperation=['multiply','add'].includes(d.type)?d.type:null,pairTones=d.type==='add'?additionColors:pairColors;
  backdrop(slab(g,2.05,2.4,.26,0,.07,0,color));backdrop(slab(g,1.91,2.26,.08,0,.36,0,0xf4f5e9));
  const glow=box(g,1.6,.035,.055,0,.41,1.13,color,{emissive:color,emissiveIntensity:.18});
  const border=new T.Group();g.add(border);[[[-.96,.49,-1.12],[-.96,.49,-.78]],[[-.96,.49,-1.12],[-.62,.49,-1.12]],[[.96,.49,1.12],[.96,.49,.78]],[[.96,.49,1.12],[.62,.49,1.12]]].forEach(points=>tube(border,points,color,.018));
  backdrop(glow,.2);backdrop(border,.1);
  let outputs=[],inputs=[],connections=[],weightCells=[],biasCells=[],selectionRows=[],activationChannels=[],columnCues=[],projectionRows=[],pairChannels=[],normalizationStages=[],rmsRows=[],lnRows=[],tokenRows=[],lookupRows=[],positionWaves=[],valueLabels=[];
  const tokenKind=tokenOperation(d),tokenWidth=d.out/3;
  const mixing=d.type==='mix',vocabulary=d.type==='vocabulary'&&!!weights[n.kind||n.id],matrixProjection=d.type==='linear'||vocabulary||mixing;
  if(matrixProjection){
   // A × V uses the same dot-product choreography, but V is a live second
   // input rather than a learned weight matrix. Its cells start unfilled.
   const w=mixing?Array.from({length:3},()=>Array(d.out/3).fill(0)):weights[n.kind||n.id],a=w.length,b=w[0].length,cw=1.5/b,ch=1.5/a;
   for(let i=0;i<a;i++)for(let j=0;j<b;j++){const x=(j-(b-1)/2)*cw,z=(i-(a-1)/2)*ch,c=w[i][j]<0?0xc89d70:w[i][j]===0?0xd7dfd4:color;const tile=box(g,cw*.84,.055+Math.abs(w[i][j])*.065,ch*.82,x,.49,z,c);const face=textTile(g,mixing?'·':String(w[i][j]),cw*.8,ch*.76,x,.53+Math.abs(w[i][j])*.065,z,{bg:w[i][j]<0?'#d7b68b':w[i][j]===0?'#dfe5d9':'#d2dfcc',color:'#456052',size:63});if(mixing){tile.userData.aggregationValue={row:i,column:j};face.mesh.userData.aggregationValue={row:i,column:j};}weightCells.push({tile,face:face.mesh,write:face.write,i,j,weight:w[i][j]});}
   for(const {tile,face,j}of weightCells){
    const active=()=>g.userData.activeColumn===j?g.userData.calculationHighlight:0;
    tile.userData.matrixColumn=face.userData.matrixColumn=j;
    backdrop(tile,.22,active);backdrop(face,.32,active);
   }
   if(vocabulary)for(let j=0;j<b;j++){
    const x=(j-(b-1)/2)*cw,value=d.bias?.[j]||0,active=()=>g.userData.activeColumn===j?g.userData.calculationHighlight:0;
    backdrop(textTile(g,window.NeuralPrologue.math.vocab[j],cw*.94,.21,x,.6,-1.01,{size:78}).mesh,.3,active);
    const face=textTile(g,(value<0?'':'+')+value,cw*.94,.23,x,.58,1.01,{size:76,color:'#876732',bg:'#f2e7c7'});
    face.mesh.userData.vocabularyBias={column:j,value};backdrop(face.mesh,.3,active);biasCells.push({face:face.mesh,j,value});
   }
   if(mixing){
    textTile(g,'V',.32,.23,0,.55,.97,{size:78,color:'#367e68'});
    textTile(g,'A',.28,.23,-1.12,1.22,-.87,{size:78,color:'#996f32'});
    for(let i=0;i<a;i++)valueLabels.push(textTile(g,'·',.36,.24,.94,.56,(i-(a-1)/2)*ch,{size:76,color:'#'+rowColor(i).toString(16)}));
   }
   // One outline encloses the whole weight column, including zero entries.
   for(let j=0;j<b;j++){
    const cue=new T.Group(),top=.55+Math.max(...w.map(row=>Math.abs(row[j])))*.065;
    cue.position.set((j-(b-1)/2)*cw,top,0);cue.userData.projectionColumn=j;cue.visible=false;g.add(cue);
    const fill=new T.Mesh(new T.PlaneGeometry(cw*.96,1.57),new T.MeshBasicMaterial({color:projectionColor,transparent:true,opacity:.14,depthWrite:false,toneMapped:false}));fill.rotation.x=-Math.PI/2;cue.add(fill);
    const outline=new T.LineLoop(new T.BufferGeometry().setFromPoints([new T.Vector3(-cw*.48,.003,-.785),new T.Vector3(cw*.48,.003,-.785),new T.Vector3(cw*.48,.003,.785),new T.Vector3(-cw*.48,.003,.785)]),new T.LineBasicMaterial({color:projectionColor,transparent:true,opacity:1,depthWrite:false,toneMapped:false}));cue.add(outline);
    columnCues.push({cue,fill,outline});
   }
   for(let i=0;i<a;i++)inputs.push(neuron(g,-.7,.98,(i-(a-1)/2)*ch,0x6c9eb1));
   for(let j=0;j<b;j++)outputs.push(neuron(g,.7,1.28,(j-(b-1)/2)*.4,color));
   for(let i=0;i<a;i++)for(let j=0;j<b;j++){
    const mesh=tube(g,[[-.7,.98,(i-(a-1)/2)*ch],[0,1.24,(i-(a-1)/2)*.13+(j-(b-1)/2)*.13],[.7,1.28,(j-(b-1)/2)*.4]],w[i][j]<0?0xc89968:color,w[i][j]===0?.007:.015,w[i][j]===0?.2:.8);
    const spark=new T.Mesh(sharedSphere,mat(color,{emissive:color,emissiveIntensity:1.8,transparent:true,depthWrite:false}));spark.visible=false;spark.userData={contribution:true,input:i,output:j,weight:w[i][j]};g.add(spark);
    connections.push({mesh,spark,i,j,weight:w[i][j]});
   }
   const count=d.out/b;
   if(count>1)for(let row=0;row<count;row++){
    const bank=new T.Group(),z=(row-(count-1)/2)*(vocabulary?.84:.48),bankWidth=vocabulary?3.45:1.86;bank.visible=false;bank.userData.projectionBank=row;g.add(bank);
    backdrop(box(bank,bankWidth,.025,vocabulary?.68:.39,0,1.83,z,rowColor(row),{transparent:true,opacity:.14,depthWrite:false}),.12);
    const label=textTile(bank,'·',.33,.22,-bankWidth/2+.16,1.86,z,{size:65,color:'#496253'});
    const active=()=>g.userData.sequenceProjection?.row===row;
    backdrop(label.mesh,.35,active);
    const sockets=Array.from({length:b},(_,j)=>neuron(bank,.14+(j-(b-1)/2)*(vocabulary?.52:.34),1.92,z,rowColor(row)));
    sockets.forEach(socket=>socketBackdrop(socket,.12,active));
    sockets.forEach(o=>o.scale.setScalar(.62));projectionRows.push({row,g:bank,label,outputs:sockets});
   }
  }else if(d.type==='pick'){
   const model=window.NeuralPrologue,width=d.ins[0]/d.out;
   for(let j=0;j<width;j++)textTile(g,model.math.vocab[j],.29,.22,(j-(width-1)/2)*.3,.59,-1.02,{size:82});
   for(let row=0;row<d.out;row++){
    const z=(row-(d.out-1)/2)*.58,tone=rowColor(row),cells=[];
    const word=textTile(g,'·',.34,.23,-.96,1.43,z,{size:76});
    for(let j=0;j<width;j++){
     const x=(j-(width-1)/2)*.3,tile=box(g,.265,.045,.44,x,.53,z,tone,{transparent:true,opacity:.22});
     const face=textTile(g,'·',.26,.34,x,.558,z,{size:74}),cue=ring(g,.14,x,.59,z,projectionColor,.016),best=ring(g,.125,x,.62,z,0x368d77,.023);
     cue.visible=best.visible=false;tile.userData.selectionCell=face.mesh.userData.selectionCell={row,column:j};
     const active=()=>g.userData.selection?.row===row;
     backdrop(tile,.16,active);backdrop(face.mesh,.22,active);
     cells.push({j,tile,face,cue,best,point:new T.Vector3(x,.65,z)});
    }
    const output=neuron(g,.54,1.5,z,tone);outputs.push(output);
    const carrier=new T.Mesh(sharedSphere,mat(tone,{emissive:tone,emissiveIntensity:1.5,transparent:true,depthWrite:false}));carrier.visible=false;carrier.userData.selectionWinner=true;g.add(carrier);
    backdrop(word.mesh,.25,()=>g.userData.selection?.row===row);
    selectionRows.push({row,z,tone,cells,word,output,carrier});
   }
   textTile(g,'max → ID',1.25,.22,0,.53,1.03,{size:70,color:'#367e68'});
  }else if(d.type==='activation'){
   g.userData.activation=d.activation||'silu';
   const panel=backdrop(box(g,1.5,.025,1.7,0,.57,0,0xe4cc9d,{transparent:true,opacity:.34,depthWrite:false}),.12);
   for(let i=0;i<d.out;i++){
    const seqWidth=d.out===24?8:d.out,z=(i%seqWidth-(seqWidth-1)/2)*(d.out===24?.24:.42),rowY=d.out===24?Math.floor(i/8)*.5:0,pts=Array.from({length:33},(_,j)=>{const x=-3+j*6/32;return [x*.22,.92+rowY+.22*(d.activation==='relu'?Math.max(0,x):x/(1+Math.exp(-x))),z];});
    inputs.push(neuron(g,...pts[0],0xb6ab83));outputs.push(neuron(g,...pts[32],color));
    const channel=tube(g,pts,0xb38649,.026),gate=ring(g,d.out===24?.11:.19,0,.92+rowY,z,color,.025);
    const spark=new T.Mesh(sharedSphere,mat(color,{emissive:color,emissiveIntensity:1.6,transparent:true,depthWrite:false}));spark.visible=false;spark.userData={activationFeature:i};g.add(spark);
    activationChannels.push({channel,gate,spark,i});
   }
   textTile(g,d.activation==='relu'?'max(0, x)':'x · σ(x)',1.2,.22,0,.64,1.02,{size:67,color:'#8b6739',bg:'#f0e7c8'});
  }else if(pairOperation){
   for(let i=0;i<d.out;i++){
    const z=(i-(d.out-1)/2)*(d.out===2?.65:.43),left=neuron(g,-.65,.84,z,pairTones[0]),right=neuron(g,-.22,1.22,z,pairTones[1]);
    outputs.push(neuron(g,.68,1.05,z,color));
    const leftPath=tube(g,[[-.65,.84,z],[-.28,.89,z],[.12,.96,z]],pairTones[0],.023),rightPath=tube(g,[[-.22,1.22,z],[-.06,1.08,z],[.12,.96,z]],pairTones[1],.023),outPath=tube(g,[[.12,.96,z],[.4,1.01,z],[.68,1.05,z]],color,.025);
    const gate=ring(g,.16,.12,.96,z,color,.026);
    textTile(g,pairOperation==='add'?'+':'×',.23,.22,.12,.56,z,{size:72,color:pairOperation==='add'?'#876732':'#436f60',bg:'#f4f5e9'});
    const carrier=new T.Mesh(sharedSphere,mat(color,{emissive:color,emissiveIntensity:1.2,transparent:true,depthWrite:false}));carrier.visible=false;carrier.userData[pairOperation==='add'?'additionFeature':'multiplyFeature']=i;g.add(carrier);
    pairChannels.push({i,left,right,leftPath,rightPath,outPath,gate,carrier});
    const active=()=>g.userData.activePair===i?g.userData.calculationHighlight:0;
    socketBackdrop(left,.15,active);socketBackdrop(right,.15,active);
    [leftPath,rightPath,outPath,gate].forEach(mesh=>backdrop(mesh,.1,active));
   }
  }else if(d.type==='mask'){
   for(let row=0;row<3;row++){
    const z=(row-1)*.66,tone=rowColor(row),features=[];
    const band=box(g,1.7,.025,.55,0,.47,z,tone,{transparent:true,opacity:.06,depthWrite:false});
    const word=textTile(g,'·',.36,.24,-.9,1.83,z,{size:72,color:'#456052',bg:'#edf3e8'});
    for(let j=0;j<3;j++){
     const x=(j-1)*.49,blocked=j>row,tint=blocked?0xc58b65:tone;
     const tile=box(g,.43,.05,.45,x,.52,z,tint,{transparent:true,opacity:.28});
     const face=textTile(g,blocked?'−∞':'✓',.4,.35,x,.56,z,{size:68,color:blocked?'#8d5239':'#456052',bg:blocked?'#edddcf':'#e2e9da'});
     const input=neuron(g,x,.77,z,featureColor(j)),output=neuron(g,x,1.72,z,tint);inputs.push(input);outputs.push(output);
     const shutter=box(g,.43,.34,.07,x,1.15,z,tint,{transparent:true,opacity:.12,depthWrite:false,emissive:tint,emissiveIntensity:0});shutter.visible=false;shutter.userData.maskShutter={row,column:j};
     const carrier=new T.Mesh(new T.BoxGeometry(.15,.15,.15),mat(tint,{emissive:tint,emissiveIntensity:1.4,transparent:true,depthWrite:false}));carrier.visible=false;carrier.userData.maskResult=true;g.add(carrier);
     features.push({index:j,blocked,tile,face,input,output,shutter,carrier});
    }
    tokenRows.push({row,z,tone,band,word,features});
   }
  }else if(['score','softmax','attention'].includes(d.type)){
   // Three rows correspond to positions; crossed threads depict information mixing.
   for(let i=0;i<3;i++)for(let j=0;j<(d.columns||3);j++){
    const c=d.type==='mask'&&j>i?0xc3b6a5:color,x=(j-((d.columns||3)-1)/2)*(d.columns===6?.28:.47),z=(i-1)*.48;
    box(g,d.columns===6?.24:.39,.06,.39,x,.52,z,c);textTile(g,d.type==='mask'?(j>i?'−∞':'✓'):d.type==='softmax'?'p':String(i)+','+String(j),d.columns===6?.23:.35,.32,x,.56,z,{size:56,bg:'#e2e9da',color:'#4e6b5a'});
   }
   for(let i=0;i<3;i++){socketBackdrop(neuron(g,-.75,.96,(i-1)*.5,0x789abf));outputs.push(neuron(g,.75,1.05,(i-1)*.5,color));for(let j=0;j<3;j++)if(!d.causal||j<=i)backdrop(tube(g,[[-.75,.96,(j-1)*.5],[0,1.3,0],[.75,1.05,(i-1)*.5]],color,.012,.55),.1);}
  }else if(d.type==='rope'){
   for(let i=0;i<3;i++){const z=(i-1)*.6;ring(g,.24,0,.7,z,color);tube(g,[[0,.72,z],[Math.cos(i)*.25,.72,z+Math.sin(i)*.25]],color,.027);outputs.push(neuron(g,.7,1.02,z,color));}
  }else if(tokenKind){
   const width=tokenWidth,spacing=width===2?.76:.43;
   if(tokenKind==='lookup'){
    const model=window.NeuralPrologue,table=width===4?model.original.wide.embedding:model.math.emb;
    table.forEach((values,id)=>{
     const z=(id-2.5)*.29,cells=[];
     const word=textTile(g,model.math.vocab[id],.43,.24,-.67,.61,z,{size:88});
     values.forEach((value,j)=>{
      const cw=1.12/width,x=-.32+(j+.5)*cw,tone=value<0?0xd7b68b:0xd2dfcc;
      const tile=box(g,cw*.9,.075,.25,x,.52,z,tone),face=textTile(g,String(value),cw*.86,.23,x,.565,z,{size:88,color:'#456052',bg:value<0?'#d7b68b':'#d2dfcc'});
      cells.push({tile,face:face.mesh,value,point:new T.Vector3(x,.62,z)});
     });
     const cue=new T.Group();cue.position.set(0,.64,z);cue.visible=false;cue.userData.lookupEntry=id;g.add(cue);
     const fill=new T.Mesh(new T.PlaneGeometry(1.76,.275),new T.MeshBasicMaterial({color:projectionColor,transparent:true,opacity:.2,depthWrite:false,toneMapped:false}));fill.rotation.x=-Math.PI/2;cue.add(fill);
     const outline=new T.LineLoop(new T.BufferGeometry().setFromPoints([new T.Vector3(-.88,.005,-.138),new T.Vector3(.88,.005,-.138),new T.Vector3(.88,.005,.138),new T.Vector3(-.88,.005,.138)]),new T.LineBasicMaterial({color:projectionColor,transparent:true,depthWrite:false,toneMapped:false}));cue.add(outline);
     lookupRows.push({id,values,cells,word,cue,fill,outline,address:new T.Vector3(-.68,.83,z)});
    });
    textTile(g,'ID → E[ID]',1.45,.22,0,.5,1.02,{size:64});
   }else{
    for(let j=0;j<width;j++){
     const x=(j-(width-1)/2)*spacing,tone=j%2?0xa394bd:0x789abf;
     const points=Array.from({length:41},(_,i)=>{const p=i/20;return [x,.93+.22*positionalValue(p,j,width),(p-1)*.66];});
     const wave=tube(g,points,tone,.025,.35);wave.userData.positionWave=j;
     positionWaves.push(wave);
     textTile(g,(j%2?'cos':'sin')+'('+(j>=2?'p/100':'p')+')',spacing*.94,.23,x,.57,-1.02,{size:64,color:'#6b6285',bg:'#ece6f2'});
    }
    textTile(g,(width===2?'√2':'2')+' · X + PE',1.55,.22,0,.53,1.04,{size:62,color:'#74602f',bg:'#f2e7c7'});
   }
   for(let row=0;row<3;row++){
    const z=(row-1)*.66,tone=rowColor(row),features=[];
    const band=box(g,1.72,.025,.53,0,tokenKind==='lookup'?1.48:.46,z,tone,{transparent:true,opacity:.06,depthWrite:false});
    const word=textTile(g,'·',.37,.25,-.88,1.78,z,{size:72,color:'#456052',bg:'#edf3e8'});
    for(let j=0;j<width;j++){
     const x=(j-(width-1)/2)*spacing,output=neuron(g,x,1.7,z,tone);output.scale.setScalar(.7);outputs.push(output);
     const carrier=new T.Mesh(new T.BoxGeometry(.14,.14,.14),mat(tone,{emissive:tone,emissiveIntensity:1.5,transparent:true,depthWrite:false}));carrier.visible=false;carrier.userData.tokenResult=true;g.add(carrier);
     const feature={index:j,output,carrier};
     if(tokenKind==='position'){
      const input=neuron(g,x,.72,z,featureColor(j)),scaleGate=ring(g,.13,x,1.12,z,featureColor(j),.02),join=ring(g,.135,x,1.4,z,0xb68036,.023);
      const marker=neuron(g,x,.93+.22*positionalValue(row,j,width),z,0xb68036);marker.scale.setScalar(.6);
      const code=new T.Mesh(sharedSphere,mat(0xb68036,{emissive:0xb68036,emissiveIntensity:1.4,transparent:true,depthWrite:false}));code.visible=false;code.userData.positionCode=true;g.add(code);
      textTile(g,'+',.18,.18,x,1.415,z,{size:82,color:'#876732',bg:'#f2e7c7'});
      inputs.push(input);Object.assign(feature,{input,scaleGate,join,marker,code});
     }
     features.push(feature);
    }
    tokenRows.push({row,z,tone,band,word,features});
   }
  }else if(d.type==='position'){
   for(let feature=0;feature<2;feature++){
    const z=feature?.45:-.45,pts=Array.from({length:40},(_,i)=>{const x=i*2*Math.PI/39;return [(i/39-.5)*1.5,.85+.2*(feature?Math.cos(x):Math.sin(x)),z];});
    tube(g,pts,feature?0xa394bd:0x789abf,.026);outputs.push(neuron(g,.75,1.1,z,color));
    textTile(g,feature?'cos(p)':'sin(p)',.8,.23,0,.55,z,{size:58});
   }
   textTile(g,'X + PE',1,.25,0,.55,1.02,{size:60});
  }else if(d.type==='transfer'){
   const count=d.out;
   for(let i=0;i<count;i++){
    const z=(i-(count-1)/2)*.42,tone=d.memoryOutput==='gpu'?0x659bab:0xc7a166;
    inputs.push(neuron(g,-.62,.82,z,0xa3aca9));outputs.push(neuron(g,.62,.82,z,tone));
    tube(g,[[-.62,.82,z],[.62,.82,z]],tone,.025);
   }
   textTile(g,d.memoryInput.toUpperCase()+' → '+d.memoryOutput.toUpperCase(),1.65,.28,0,.5,.98,{size:62});
  }else if(d.type==='concat'&&d.out===4){
   for(let i=0;i<4;i++){
    const tone=i<2?0x659bab:0x9b88c4,x=(i-1.5)*.43;
    neuron(g,x,.72,0,tone);outputs.push(neuron(g,x,1.45,0,tone));
    tube(g,[[x,.72,0],[x,1.45,0]],tone,.025);
    textTile(g,i<2?'GPU 0':'GPU 1',.42,.22,x,.55,.65,{size:58});
   }
  }else if(d.type==='concat'){
   for(let i=0;i<3;i++){
    const z=(i-1)*.58;
    for(let head=0;head<2;head++){
     const tone=head?0xa394bd:0x789abf,x=head?.35:-.1;
     neuron(g,-.7,.75+head*.35,z,tone);outputs.push(neuron(g,x+.25,1.2,z,tone));
     tube(g,[[-.7,.75+head*.35,z],[x,1.02,z],[x+.25,1.2,z]],tone,.025);
     box(g,.36,.09,.35,x,.55,z,tone);textTile(g,head?'B':'A',.31,.28,x,.61,z,{size:65});
    }
   }
  }else if(d.type==='norm'){
   const width=d.ins[0]/3;
   for(let row=0;row<3;row++){
    const z=(row-1)*.65,tone=rowColor(row),features=[];
    const band=box(g,1.65,.025,.55,0,.47,z,tone,{transparent:true,opacity:.12,depthWrite:false});
    const hub=neuron(g,0,1.12,z,tone),halo=ring(g,.2,0,1.12,z,tone,.025);
    const divisor=textTile(g,'r',.64,.3,0,1.29,z,{size:76,color:'#74602f',bg:'#f2e7c7'});
    for(let j=0;j<width;j++){
     const x=(j-(width-1)/2)*.92,input=neuron(g,x,.72,z,featureColor(j)),output=neuron(g,x,1.64,z,tone);
     inputs.push(input);outputs.push(output);
     const label=textTile(g,'x',.4,.3,x,.86,z,{size:76,color:'#456052',bg:'#edf3e8'});
     const squarePath=tube(g,[[x,.72,z],[x*.8,.97,z],[0,1.12,z]],featureColor(j),.02,.2);
     const scalePath=tube(g,[[0,1.12,z],[x*.8,1.39,z],[x,1.64,z]],tone,.02,.2);
     const square=new T.Mesh(new T.BoxGeometry(.13,.13,.13),mat(featureColor(j),{emissive:featureColor(j),emissiveIntensity:1.4,transparent:true,depthWrite:false}));
     const scaled=new T.Mesh(sharedSphere,mat(tone,{emissive:tone,emissiveIntensity:1.4,transparent:true,depthWrite:false}));
     square.visible=scaled.visible=false;square.userData.rmsSquare=true;scaled.userData.rmsScaled=true;g.add(square,scaled);
     features.push({input,output,label,squarePath,scalePath,square,scaled,index:j});
    }
    rmsRows.push({row,band,hub,halo,divisor,features,tone});
   }
   textTile(g,'x² → r → x/r',1.55,.23,0,.53,1.02,{size:60,color:'#74602f',bg:'#f2e7c7'});
  }else if(d.type==='layernorm'){
   const count=d.out>4?3:1,width=d.out/count,spacing=width===2?.88:.43;
   for(let row=0;row<count;row++){
    const z=(row-(count-1)/2)*.7,tone=rowColor(row),features=[];
    const band=box(g,1.7,.025,count===1?1.5:.58,0,.47,z,tone,{transparent:true,opacity:.1,depthWrite:false});
    const meanHub=neuron(g,0,1.05,z-.17,0xb68036),scaleHub=neuron(g,0,1.45,z+.14,0x8862aa);
    const meanLabel=textTile(g,'μ',.75,.26,0,1.22,z-.17,{size:80,color:'#876732',bg:'#f2e7c7'}),scaleLabel=textTile(g,'s',.8,.26,0,1.62,z+.14,{size:80,color:'#695182',bg:'#ece6f2'});
    for(let j=0;j<width;j++){
     const x=(j-(width-1)/2)*spacing,input=neuron(g,x,.72,z,featureColor(j)),center=new T.Vector3(x,1.04,z),output=neuron(g,x,1.88,z,tone),gate=ring(g,.13,x,1.56,z,tone,.022);
     inputs.push(input);outputs.push(output);
     const meanPath=tube(g,[[x,.72,z],[x*.55,.96,z-.12],[0,1.05,z-.17]],featureColor(j),.019,.18),squarePath=tube(g,[[x,1.04,z],[x*.55,1.3,z+.1],[0,1.45,z+.14]],featureColor(j),.02,.18),scalePath=tube(g,[[x,1.04,z],[x,1.56,z],[x,1.88,z]],tone,.024,.18);
     const spark=(flag,tint,boxShape=false)=>{const mesh=new T.Mesh(boxShape?new T.BoxGeometry(.13,.13,.13):sharedSphere,mat(tint,{emissive:tint,emissiveIntensity:1.4,transparent:true,depthWrite:false}));mesh.visible=false;mesh.userData[flag]=true;g.add(mesh);return mesh;};
     features.push({index:j,input,center,output,gate,meanPath,squarePath,scalePath,meanSpark:spark('lnMean',featureColor(j)),shift:spark('lnShift',0xb68036),square:spark('lnSquare',featureColor(j),true)});
    }
    lnRows.push({row,z,tone,band,meanHub,scaleHub,meanLabel,scaleLabel,features});
   }
   textTile(g,'− μ → ÷ s',1.5,.23,0,.52,1.03,{size:65,color:'#74602f',bg:'#f2e7c7'});
  }else if(['embedding','vocabulary','pick','storage','cache','decode','ffn','invert'].includes(d.type)){
   const words=d.glyphs||(d.type==='vocabulary'||d.type==='pick'?[I18n.t("我"),I18n.t("爱"),I18n.t("猫"),I18n.t("你"),I18n.t("恨"),I18n.t("狗")]:d.type==='cache'?['K','V','+1']:d.type==='norm'?['x','RMS','÷']:d.type==='storage'?['W','W','W']:d.type==='ffn'?['G','×','U']:['0','1','2']);
   words.forEach((t,i)=>{const x=words.length>3?(i%2-.5)*.7:0,z=words.length>3?(Math.floor(i/2)-1)*.55:(i-1)*.58;box(g,.55,.09,.44,x,.55,z,color);textTile(g,t,.5,.38,x,.605,z,{size:64,bg:'#e9edde',color:'#496253'});});
   for(let i=0;i<3;i++)outputs.push(neuron(g,.7,1.08,(i-1)*.5,color));
  }else if(['encoder','decoder'].includes(d.type)){
   const parts=d.type==='encoder'?['MHA','LN','FFN','LN']:['Mask','Cross','FFN'];
   parts.forEach((word,i)=>{const z=(i-(parts.length-1)/2)*.45;box(g,1.4,.11,.35,0,.6+i*.05,z,color);textTile(g,word,1.25,.31,0,.66+i*.05,z,{size:60});outputs.push(neuron(g,.7,1.18,z,color));});
  }else{
   for(let i=0;i<2;i++){const z=(i-.5)*.65;neuron(g,-.6,.78,z,0x6b9fb1);neuron(g,-.3,1.25,z,0xb7a17d);outputs.push(neuron(g,.68,1.08,z,color));tube(g,[[-.6,.78,z],[.02,.96,z],[.68,1.08,z]],0x6b9fb1,.027);tube(g,[[-.3,1.25,z],[.02,.96,z],[.68,1.08,z]],color,.027);ring(g,.18,.06,.94,z,color);}
  }
  inputs.forEach(socket=>socketBackdrop(socket,matrixProjection?.12:.3));
  outputs.forEach((socket,i)=>socketBackdrop(socket,.14,()=>{
   if(matrixProjection)return g.userData.activeColumn===i?g.userData.calculationHighlight:0;
   if(selectionRows.length)return g.userData.selection?.row===i;
   if(pairOperation)return g.userData.activePair===i?g.userData.calculationHighlight:0;
   const flow=g.userData.tokenFlow||g.userData.layerNorm||g.userData.rms;
   return flow?Math.floor(i/(outputs.length/(d.out>4?3:1)))===flow.row:true;
  }));
  for(const row of rmsRows){const active=()=>g.userData.rms?.row===row.row;socketBackdrop(row.hub,.15,active);backdrop(row.halo,.15,active);backdrop(row.divisor.mesh,.3,active);row.features.forEach(f=>backdrop(f.label.mesh,.3,active));}
  for(const row of lnRows){const active=()=>g.userData.layerNorm?.row===row.row;[row.meanHub,row.scaleHub].forEach(o=>socketBackdrop(o,.15,active));[row.meanLabel,row.scaleLabel].forEach(t=>backdrop(t.mesh,.3,active));row.features.forEach(f=>backdrop(f.gate,.15,active));}
  for(const row of tokenRows){const active=()=>g.userData.tokenFlow?.row===row.row;backdrop(row.word.mesh,.35,active);for(const f of row.features)if(f.marker){socketBackdrop(f.marker,.2,active);backdrop(f.scaleGate,.15,active);backdrop(f.join,.15,active);}}
  d.ins.forEach((dim,i)=>{const z=d.ins.length===2?(i?.55:-.55):0;backdrop(box(g,.21,.17,.27,-1.08,.58,z,0x6e887c));for(let j=0;j<Math.min(dim,4);j++)backdrop(box(g,.12,.02,.035,-1.08,.68,z+(j-(Math.min(dim,4)-1)/2)*.055,0xc6d7c3));});
  const sock=new T.Mesh(new T.CylinderGeometry(.15,.15,.18,20),mat(color));sock.rotation.z=Math.PI/2;sock.position.set(1.05,.68,0);g.add(sock);
  backdrop(sock);
  if(register)units.set(n.id,{g,glow,outputs,inputs,connections,weightCells,biasCells,selectionRows,vocabulary,activationChannels,columnCues,projectionRows,pairChannels,pairTones,pairOperation,normalizationStages,rmsRows,lnRows,tokenKind,tokenWidth,tokenRows,lookupRows,positionWaves,mixing,valueLabels,backdrops,focusAmount:0,links:n.links,color});return g;
 }
 // Moving values stay clear. Matrix columns regain opacity with their own
 // calculation highlight; the surrounding context fades for the whole operation.
 function focusCalculations(){
  const active=new Map();
  for(const [id,u]of units){
   const amount=u.g.userData.processing?u.focusAmount:0;if(u.g.userData.processing)active.set(id,amount);
   for(const entry of u.backdrops.values()){
    const {mesh}=entry,m=mesh.material,keep=Number(entry.keep())||0,factor=1-amount*(1-entry.faint)*(1-keep),dimmed=factor<.999;
    const transparent=entry.transparent||dimmed;if(m.transparent!==transparent){m.transparent=transparent;m.needsUpdate=true;}
    m.opacity=entry.opacity*factor;m.depthWrite=dimmed?false:entry.depthWrite;
    mesh.castShadow=dimmed?false:entry.castShadow;mesh.receiveShadow=dimmed?false:entry.receiveShadow;
   }
  }
  return active;
 }
 function sync(values,depths,depth,selected,activeSample,results,busy,layout={},output=null,tokenPreview=null,timeline=null,arrivalPreview=null){
  if(!busy)clearProcessing();
  for(const [id,u]of units){const timing=timeline?.nodes.get(id),completed=values?.[id]&&(timing?timeline.time>=timing.end:(depths[id]||0)<=depth),active=busy&&(timing?timeline.time>=timing.start&&timeline.time<timing.end:depths[id]===depth),sel=id===selected;
   u.glow.material.emissiveIntensity=active?1.6:sel?.9:completed?.45:.12;
   const input=completed&&!layout.sequence?values[u.links[0]]:null;
   u.inputs.forEach((o,i)=>{const v=input?.[i];o.scale.setScalar(v==null?1:Math.abs(v)<1e-9?.55:1+Math.min(2,Math.abs(v))*.3);o.material.emissiveIntensity=active&&Number.isFinite(v)&&Math.abs(v)>1e-9?1.1:.08;});
   u.connections.forEach(({mesh,i,weight})=>{const lit=active&&input&&Math.abs(input[i]*weight)>1e-9;mesh.material.opacity=(input?(lit?.95:.08):(weight===0?.2:.8))*(1-.92*u.focusAmount);mesh.material.depthWrite=u.focusAmount===0;mesh.material.emissive.set(lit?u.color:0x000000);mesh.material.emissiveIntensity=lit?.6:0;});
   u.outputs.forEach((o,i)=>{const v=completed?values[id][i]:null;o.scale.setScalar(!Number.isFinite(v)?1:1+Math.min(2,Math.abs(v))*.18);o.material.color.set(Number.isFinite(v)&&v<0?0xc79868:u.color);o.material.emissiveIntensity=active?1.1:completed?.35:.08;});
  }
  const {samples}=context.userData,seq=!!layout.sequence,input=seq?samples[activeSample].v:null,validation=!seq&&busy&&activeSample>=3;
  const words=seq?samples[activeSample].word.split(' / '):samples.slice(0,3).map(s=>s.word);
  if(layout.outputPairAxes){rows.forEach((r,i)=>r.label.write(words[i]));outputHeaders.forEach((h,i)=>h.write(words[i]));}
  const sourceValues=layout.dualSource?window.NeuralPrologue.sourceValues({levelId:layout.id},input):null;
  tokens.forEach(t=>{const i=t.index,secondary=t.sourceId==='Z',v=secondary?sourceValues.Z:input,current=validation&&i===0?samples[activeSample]:samples[i];t.label.write(secondary?window.NeuralPrologue.math.vocab[layout.secondaryTokenIds[i]]:validation&&i===0?current.word:words[i]);t.g.userData.active=busy&&(seq||(validation?i===0:i===activeSample));t.cells.forEach((c,j)=>{const value=seq?v[i*t.cells.length+j]:current.v[j];c.write(Number.isFinite(value)?Number(value.toFixed(2)).toString():'·');});});
  pulseSource();
  outputReadout(activeSample,results,busy,layout,tokenPreview,arrivalPreview);
 }
 function outputReadout(activeSample,results,busy,layout={},tokenPreview=null,arrivalPreview=null){
  const seq=!!layout.sequence,validation=!seq&&busy&&activeSample>=3,flash=arrivalPreview?.pulse||0;
  rows.forEach((r,i)=>{
   const active=busy&&(seq||(validation?i===0:i===activeSample)),resultIndex=validation&&i===0?activeSample:i;r.g.userData.active=active;r.stripe.material.emissiveIntensity=layout.outputArrivalHold?(arrivalPreview ? .3+1.6*flash : .1):active?1.3:.1;
   r.panel.material.emissive.set(rowColor(i));r.panel.material.emissiveIntensity=.4*flash;
   const result=seq?results[activeSample]:results[resultIndex],offset=seq?i*r.cells.length:0;
   if(!seq&&!layout.outputPairAxes)r.label.write(validation&&i===0?(activeSample===3?'A':'B'):String(i));
   const actual=result?.actual||arrivalPreview?.values||(layout.tokenOutput?tokenPreview:null);
   const rowPass=result&&(seq?actual&&r.cells.every((_,j)=>Number.isFinite(actual[offset+j])&&Math.abs(actual[offset+j]-result.expected[offset+j])<(layout.engine?(layout.limits?.error||.06):1e-9)):result.pass);
   const tone=!result?'#558974':rowPass?'#397c61':'#a14d3d';
   r.panel.material.color.set(!result?0xc8d9c5:rowPass?0xbcd8c4:0xe6c1ac);
   r.status.write(!result?'·':rowPass?'✓':'×',tone);
   r.cells.forEach((cell,j)=>{const value=actual?.[offset+j];cell.write(layout.tokenOutput?(Number.isInteger(value)?window.NeuralPrologue.math.vocab[value]??'?':''):Number.isFinite(value)?Number(value.toFixed(2)).toString():result?'—':'·',tone);});
  });

 }
 const ramp=(x,a,b)=>{const t=Math.max(0,Math.min(1,(x-a)/(b-a)));return t*t*(3-2*t);};
 // Projection columns and elementwise pairs share the same per-result timing.
 // Compress the repeated middle section only for the overall tray movement.
 function projectionFrame(time,columns){
  const step=.4,extra=Math.max(0,columns-1)*step,duration=1+extra;
  const phase=time<.4?time:time<.67+extra?.4+(time-.4)*.27/(.27+extra):time-extra;
  const column=Math.max(0,Math.min(columns-1,Math.floor((time-.4)/step)));
  const offset=column*step,local=time-offset,focus=local>=.4&&local<.76?ramp(local,.4,.43)*(1-ramp(local,.71,.76)):0;
  return {duration,phase:Math.max(0,Math.min(1,phase)),column,offset,step,focus};
 }
 function sequenceProjectionFrame(time,columns,count=3){
  const rowDuration=projectionFrame(0,columns).duration,end=rowDuration*count,duration=end+.3;
  const row=Math.max(0,Math.min(count-1,Math.floor(time/rowDuration))),rowTime=Math.max(0,Math.min(rowDuration,time-row*rowDuration)),local=projectionFrame(rowTime,columns);
  const phase=time<end?time/end*.8:.8+(time-end)/.3*.2;
  return {...local,row,rowTime,rowDuration,end,duration,localPhase:local.phase,phase:Math.max(0,Math.min(1,phase))};
 }
 function tokenFrame(time,kind,count=3){
  const start=.22,step=kind==='mask'?1.8:kind==='lookup'?.9:1.2,end=start+count*step,duration=end+.3;
  const row=Math.max(0,Math.min(count-1,Math.floor((time-start)/step))),local=Math.max(0,Math.min(1,(time-start-row*step)/step));
  const phase=time<start?time/start*.18:time<end?.18+(time-start)/(count*step)*.62:.8+(time-end)/.3*.2;
  const stage=kind==='mask'?(local<.28?'arrive':local<.53?'restrict':local<.84?'replace':'ready'):kind==='lookup'?(local<.3?'address':local<.58?'read':local<.84?'copy':'ready'):(local<.28?'scale':local<.56?'position':local<.84?'add':'ready');
  return {row,local,start,step,end,duration,stage,phase:Math.max(0,Math.min(1,phase))};
 }
 function selectionFrame(time,columns=6,count=3){
  const timing=sequenceProjectionFrame(time,columns,count),finish=.67+(columns-1)*timing.step;
  return {...timing,finish,ready:finish+.27,stage:timing.rowTime<.4?'arrive':timing.rowTime<finish?'compare':'emit'};
 }
 function selectionChoice(scores,column){
  let previous=0;for(let j=1;j<column;j++)if(scores[j]>scores[previous])previous=j;
  const best=column===0||scores[column]>scores[previous]?column:previous;
  return {previous:column?previous:null,best};
 }
 function processingSelection(u,time,input,output,words){
  const width=input.length/output.length,timing=selectionFrame(time,width,output.length),active=time>=0&&time<timing.duration;
  u.g.userData.processing=active;if(!active){clearSelection(u);return;}
  u.glow.material.emissiveIntensity=.4+timing.focus;
  for(const row of u.selectionRows){
   const clock=time-row.row*timing.rowDuration,current=clock>=0&&clock<timing.rowDuration,done=clock>=timing.finish,scores=input.slice(row.row*width,(row.row+1)*width);
   const checked=Math.max(0,Math.min(width,Math.floor((clock-.61)/timing.step)+1)),best=checked?selectionChoice(scores,checked-1).best:null;
   row.word.write(words[row.row]||'·');
   for(const cell of row.cells){
    const scanning=current&&cell.j===timing.column&&timing.stage==='compare',winner=best===cell.j;
    cell.face.write(clock>=.3?String(Number(scores[cell.j].toFixed(2))):'·');
    Object.assign(cell.tile.userData.selectionCell,{value:scores[cell.j],checked:cell.j<checked,best:winner});
    cell.tile.material.emissive.set(winner?0x368d77:projectionColor);cell.tile.material.emissiveIntensity=scanning?1.05:winner?.65:0;
    cell.face.mesh.material.color.set(winner?0xb7efd0:0xffffff);
    cell.cue.visible=scanning;cell.cue.scale.setScalar(1+timing.focus*.16);cell.cue.material.emissive.set(projectionColor);cell.cue.material.emissiveIntensity=scanning?1:0;
    cell.best.visible=winner;cell.best.material.emissive.set(0x368d77);cell.best.material.emissiveIntensity=current?.8:.25;
   }
   row.output.material.emissiveIntensity=done?.7:.08;
   const transfer=ramp(clock,timing.finish,timing.finish+.27),winner=output[row.row];
   row.carrier.visible=current&&clock>=timing.finish&&clock<timing.finish+.3;
   row.carrier.position.copy(row.cells[winner].point).lerp(row.output.position,transfer);row.carrier.position.y+=Math.sin(Math.PI*transfer)*.3;
   row.carrier.material.opacity=ramp(clock,timing.finish,timing.finish+.035)*(1-ramp(clock,timing.finish+.24,timing.finish+.3));
   Object.assign(row.carrier.userData,{row:row.row,winner,value:scores[winner],transfer});
  }
  const scores=input.slice(timing.row*width,(timing.row+1)*width),choice=selectionChoice(scores,timing.column);
  u.g.userData.selection={row:timing.row,rowTime:timing.rowTime,column:timing.column,stage:timing.stage,scores:[...scores],...choice};
 }
 function clearSelection(u){
  delete u.g.userData.selection;
  for(const row of u.selectionRows){row.word.write('·');row.carrier.visible=false;for(const cell of row.cells){cell.face.write('·');cell.face.mesh.material.color.set(0xffffff);cell.tile.material.emissiveIntensity=0;cell.cue.visible=cell.best.visible=false;}}
 }
 function selectionAnchor(id,row,time){
  const u=units.get(id),timing=selectionFrame(time,u.selectionRows[0].cells.length,u.selectionRows.length),r=u.selectionRows[row];
  return (timing.stage==='emit'?r.output.position:r.cells[timing.column].point).clone().add(new T.Vector3(0,.35,0)).add(u.g.position);
 }
 function routeSelectionInput(id,mesh,time,origin,anchor){
  const u=units.get(id),width=u.selectionRows[0].cells.length,timing=selectionFrame(time,width,u.selectionRows.length);
  mesh.userData.routing=time>0;mesh.children.find(o=>o.userData.packetTray).material.opacity=1-ramp(time,0,.2);
  for(const {bar,pad,label,home,padHome,labelHome}of mesh.userData.featureRoutes){
   const index=bar.userData.feature,row=Math.floor(index/width),column=index%width,clock=time-row*timing.rowDuration,fan=ramp(clock,.02,.3),alpha=1-ramp(clock,.3,.4),target=u.selectionRows[row].cells[column].point.clone().add(u.g.position);
   for(const [object,start]of [[bar,home],[pad,padHome],[label,labelHome]]){object.position.copy(anchor).add(start).lerp(target.clone().add(new T.Vector3(0,object===label?.2:start.y,0)),fan).sub(origin);object.material.opacity=alpha;}
   label.visible=alpha>0&&fan>0&&detailModes[detailScope];Object.assign(bar.userData,{selectionInput:true,row,column,value:bar.userData.value});
  }
 }
 function routeSelectionOutput(id,mesh,time,origin){
  const u=units.get(id),timing=selectionFrame(time,u.selectionRows[0].cells.length,u.selectionRows.length),gather=ramp(timing.phase,.8,.97);
  mesh.userData.gather=gather;mesh.userData.routing=gather<1;
  mesh.children.forEach((o,i)=>{
   if(o.userData.packetTray){o.material.opacity=ramp(timing.phase,.8,.94);return;}
   const row=o.userData.feature??o.userData.featurePad,home=mesh.userData.outputHomes[i],ready=row*timing.rowDuration+timing.ready,alpha=ramp(time,ready,ready+.03);
   o.visible=alpha>0;o.material.opacity=alpha;const start=u.selectionRows[row].output.getWorldPosition(new T.Vector3()).add(new T.Vector3(0,home.y,0));
   o.position.copy(start.lerp(origin.clone().add(home),gather).sub(origin));Object.assign(o.userData,{selectedRow:row});
  });
 }
 function processingToken(u,time,input,output,words){
  if(u.tokenKind==='mask'){processingMask(u,time,input,output,words);return;}
  const timing=tokenFrame(time,u.tokenKind),active=time>=0&&time<timing.duration,width=u.tokenWidth;
  u.g.userData.processing=active;
  if(!active){clearToken(u);return;}
  u.glow.material.emissiveIntensity=1.1;
  const local=timing.local,lookup=u.tokenKind==='lookup',id=lookup?input[timing.row]:null;
  for(const entry of u.lookupRows){
   const strength=entry.id===id&&time>=timing.start?ramp(local,.08,.25)*(1-ramp(local,.84,1)):0;
   entry.cue.visible=strength>0;entry.fill.material.opacity=.22*strength;entry.outline.material.opacity=strength;
   entry.cells.forEach(({tile,face})=>{tile.material.emissive.set(projectionColor);tile.material.emissiveIntensity=.7*strength;face.material.color.set(0xffffff).lerp(new T.Color(projectionColor),strength*.23);});
  }
  u.positionWaves.forEach(wave=>{wave.material.opacity=timing.stage==='position'?.95:.35;wave.material.emissive.set(wave.material.color);wave.material.emissiveIntensity=timing.stage==='position'?.6:0;});
  const groups=[];
  for(const row of u.tokenRows){
   const clock=(time-timing.start-row.row*timing.step)/timing.step,focus=clock>=0&&clock<1,done=clock>=.84;
   const values=lookup?u.lookupRows[input[row.row]].values:input.slice(row.row*width,(row.row+1)*width),scaled=lookup?values:values.map(x=>x*Math.sqrt(width));
   const codes=lookup?null:values.map((_,j)=>positionalValue(row.row,j,width)),results=output.slice(row.row*width,(row.row+1)*width);
   row.band.material.opacity=focus?.25:done?.12:.06;row.word.write(words[row.row]||'·');
   for(const f of row.features){
    const j=f.index,value=results[j],zero=Math.abs(value)<1e-9,tone=value<0?0xc79868:row.tone;
    f.output.scale.setScalar(done?.6+Math.min(2,Math.abs(value))*.15:.7);f.output.material.color.set(done?tone:row.tone);f.output.material.emissiveIntensity=focus&&clock>.6?1.2:done?.3:.08;
    const start=lookup?u.lookupRows[input[row.row]].cells[j].point:new T.Vector3(f.output.position.x,1.4,row.z),begin=lookup?.4:.61,finish=.84,k=ramp(clock,begin,.82);
    f.carrier.visible=focus&&clock>=begin&&clock<finish;
    Object.assign(f.carrier.userData,{operation:u.tokenKind,row:row.row,feature:j,value,input:values[j],lookupId:lookup?input[row.row]:null,scaled:scaled[j],code:codes?.[j]});
    if(f.carrier.visible){
     f.carrier.position.copy(start).lerp(f.output.position,k);f.carrier.position.y+=Math.sin(Math.PI*k)*.18;
     f.carrier.scale.setScalar(zero?.65:.7+Math.min(2,Math.abs(value))*.3);f.carrier.material.wireframe=zero;
     f.carrier.material.color.set(tone);f.carrier.material.emissive.set(tone);f.carrier.material.opacity=ramp(clock,begin,begin+.035)*(1-ramp(clock,.80,.84));
    }
    if(!lookup){
     f.input.material.emissiveIntensity=focus&&clock<.36?.9:.08;
     f.scaleGate.material.emissive.set(featureColor(j));f.scaleGate.material.emissiveIntensity=focus&&clock<.34?1.4:0;
     f.join.material.emissive.set(0xb68036);f.join.material.emissiveIntensity=focus&&clock>=.5&&clock<.84?1.5:0;f.join.scale.setScalar(focus?1+.18*ramp(clock,.5,.61)*(1-ramp(clock,.8,.9)):1);
     f.marker.material.emissiveIntensity=focus&&clock>=.28&&clock<.6?1.5:.08;
     f.code.visible=focus&&clock>=.34&&clock<.65;Object.assign(f.code.userData,{row:row.row,feature:j,value:codes[j],angle:row.row/10000**(2*Math.floor(j/2)/width)});
     if(f.code.visible){
      f.code.position.copy(f.marker.position).lerp(new T.Vector3(f.output.position.x,1.4,row.z),ramp(clock,.34,.60));f.code.position.x+=Math.sin(Math.PI*ramp(clock,.34,.60))*.16;
      f.code.scale.setScalar(.6+Math.abs(codes[j])*.45);f.code.material.wireframe=Math.abs(codes[j])<1e-9;f.code.material.opacity=ramp(clock,.34,.38)*(1-ramp(clock,.59,.65));
     }
    }
   }
   groups.push({row:row.row,lookupId:lookup?input[row.row]:null,values:[...values],scaled:[...scaled],codes,output:results});
  }
  u.g.userData.tokenFlow={operation:u.tokenKind,row:timing.row,stage:timing.stage,local,groups};
 }
 function processingMask(u,time,input,output,words){
  const timing=tokenFrame(time,'mask');u.g.userData.processing=time>=0&&time<timing.duration;
  if(!u.g.userData.processing){clearToken(u);return;}
  const groups=[];
  for(const row of u.tokenRows){
   const clock=(time-timing.start-row.row*timing.step)/timing.step,active=clock>=0&&clock<1,done=clock>=.82;
   row.band.material.opacity=active?.28:done?.12:.06;row.word.write(words[row.row]||'·');
   for(const f of row.features){
    const index=row.row*3+f.index,value=output[index],tone=f.blocked?0xc58b65:row.tone;
    f.tile.material.opacity=active?.88:done?.48:.2;f.tile.material.emissive.set(tone);f.tile.material.emissiveIntensity=active?.65:0;
    f.face.write(clock>=.53?(value===-Infinity?'−∞':String(Number(value.toFixed(2)))):active?String(Number(input[index].toFixed(2))):f.blocked?'−∞':'✓');
    f.shutter.visible=f.blocked&&active&&clock>=.28&&clock<.86;f.shutter.position.y=1.45-.35*ramp(clock,.28,.46);f.shutter.material.opacity=.68*ramp(clock,.28,.42)*(1-ramp(clock,.77,.86));f.shutter.material.emissiveIntensity=.7;
    f.input.material.emissiveIntensity=active&&clock<.53?1:.08;
    f.output.material.color.set(tone);f.output.material.emissive.set(tone);f.output.material.emissiveIntensity=active&&clock>=.53?1.25:done?.3:.08;
    f.carrier.visible=active&&clock>=.53&&clock<.84;f.carrier.material.wireframe=f.blocked;f.carrier.material.color.set(tone);f.carrier.material.opacity=ramp(clock,.53,.57)*(1-ramp(clock,.79,.84));
    f.carrier.position.set(f.output.position.x,1.14,row.z).lerp(f.output.position,ramp(clock,.53,.82));
    Object.assign(f.carrier.userData,{operation:'mask',row:row.row,feature:f.index,input:input[index],value,blocked:f.blocked});
   }
   groups.push({row:row.row,input:input.slice(row.row*3,row.row*3+3),output:output.slice(row.row*3,row.row*3+3)});
  }
  u.g.userData.tokenFlow={operation:'mask',row:timing.row,stage:timing.stage,local:timing.local,groups};
 }
 function clearToken(u){
  delete u.g.userData.tokenFlow;
  u.lookupRows.forEach(entry=>{entry.cue.visible=false;entry.cells.forEach(({tile,face})=>{tile.material.emissiveIntensity=0;face.material.color.set(0xffffff);});});
  u.positionWaves.forEach(wave=>{wave.material.opacity=.35;wave.material.emissiveIntensity=0;});
  for(const row of u.tokenRows){
   row.band.material.opacity=.06;row.word.write('·');
   for(const f of row.features){f.carrier.visible=false;if(u.tokenKind==='mask'){f.shutter.visible=false;f.tile.material.opacity=.28;f.tile.material.emissiveIntensity=0;f.input.material.emissiveIntensity=.08;f.face.write(f.blocked?'−∞':'✓');}if(f.code){f.code.visible=false;f.input.material.emissiveIntensity=f.marker.material.emissiveIntensity=.08;f.scaleGate.material.emissiveIntensity=f.join.material.emissiveIntensity=0;f.join.scale.setScalar(1);}}
  }
 }
 function layerNormFrame(time,count=3){
  const start=.22,step=1.9,end=start+count*step,duration=end+.35;
  const row=Math.max(0,Math.min(count-1,Math.floor((time-start)/step))),local=Math.max(0,Math.min(1,(time-start-row*step)/step));
  const phase=time<start?time/start*.18:time<end?.18+(time-start)/(count*step)*.62:.8+(time-end)/.35*.2;
  return {row,local,start,step,end,duration,phase:Math.max(0,Math.min(1,phase)),stage:local<.23?'mean':local<.46?'center':local<.7?'variance':local<.92?'scale':'ready'};
 }
 function layerNormStats(values){
  const mean=values.reduce((a,b)=>a+b,0)/values.length,centered=values.map(x=>x-mean),squares=centered.map(x=>x*x),variance=squares.reduce((a,b)=>a+b,0)/values.length;
  return {values,mean,centered,squares,variance,divisor:Math.sqrt(variance+1e-6)};
 }
 function processingLayerNorm(u,time,input,output,color){
  const count=u.lnRows.length,width=input.length/count,timing=layerNormFrame(time,count),active=time>=0&&time<timing.duration;
  u.g.userData.processing=active;
  if(!active){clearLayerNorm(u);return;}
  u.glow.material.emissiveIntensity=1.2;const groups=[];
  for(const row of u.lnRows){
   const stats=layerNormStats(input.slice(row.row*width,(row.row+1)*width)),local=(time-timing.start-row.row*timing.step)/timing.step,focus=local>=0&&local<1,done=local>=.9;
   const {mean,centered,squares,divisor}=stats;if(count===1)row.tone=color;row.band.material.color.set(row.tone);
   row.band.material.opacity=focus?.35:done?.18:.1;
   row.meanHub.material.emissiveIntensity=focus&&local<.46?1.4:.08;row.scaleHub.material.emissiveIntensity=focus&&local>=.46&&local<.9?1.4:.08;
   row.meanLabel.write(local>=.18?'μ '+Number(mean.toFixed(3)):'μ');row.scaleLabel.write(local>=.67?'s '+Number(divisor.toFixed(4)):'s');
   for(const f of row.features){
    const j=f.index,value=input[row.row*width+j],result=output[row.row*width+j];
    f.input.material.emissiveIntensity=focus&&local<.46?1:.08;f.output.material.emissiveIntensity=focus&&local>=.7?1.2:done?.35:.08;
    f.output.scale.setScalar(done?.65+Math.min(2,Math.abs(result))*.25:1);f.output.material.color.set(done&&result<0?0xc79868:row.tone);
    f.gate.material.emissive.set(row.tone);f.gate.material.emissiveIntensity=focus&&local>=.7&&local<.9?1.5:0;f.gate.scale.setScalar(focus&&local>=.7?1+.18*Math.sin(Math.PI*ramp(local,.7,.9)):1);
    for(const [path,from,to]of [[f.meanPath,0,.46],[f.squarePath,.46,.7],[f.scalePath,.7,.92]]){const lit=focus&&local>=from&&local<to;path.material.opacity=lit?.85:.18;path.material.emissive.set(row.tone);path.material.emissiveIntensity=lit?.65:0;}
    for(const [mesh,from,to,v]of [[f.meanSpark,.03,.21,value],[f.shift,.25,.44,-mean],[f.square,.48,.69,squares[j]]]){
     mesh.visible=focus&&local>=from&&local<to;Object.assign(mesh.userData,{row:row.row,feature:j,value:v,input:value,mean,centered:centered[j],divisor});
     if(!mesh.visible)continue;
     const k=ramp(local,from,to-.02),point=mesh===f.meanSpark?f.meanPath.geometry.parameters.path.getPoint(k):mesh===f.square?f.squarePath.geometry.parameters.path.getPoint(k):row.meanHub.position.clone().lerp(f.center,k);
     mesh.position.copy(point);mesh.scale.setScalar(.55+Math.min(2,Math.abs(v))*.24);mesh.material.wireframe=Math.abs(v)<1e-9;mesh.material.color.set(v<0?0xc79868:mesh===f.shift?0xb68036:featureColor(j));mesh.material.opacity=ramp(local,from,from+.025)*(1-ramp(local,to-.025,to));
    }
   }
   groups.push({row:row.row,...stats,output:output.slice(row.row*width,(row.row+1)*width)});
  }
  u.g.userData.layerNorm={row:timing.row,stage:timing.stage,local:timing.local,groups};
  u.g.userData.normalization={...groups[timing.row],phase:timing.local<.46?0:timing.local<.7?1:2};
 }
 function clearLayerNorm(u){
  delete u.g.userData.layerNorm;delete u.g.userData.normalization;
  for(const row of u.lnRows){row.band.material.opacity=.1;row.meanHub.material.emissiveIntensity=row.scaleHub.material.emissiveIntensity=.08;row.meanLabel.write('μ');row.scaleLabel.write('s');for(const f of row.features){f.meanSpark.visible=f.shift.visible=f.square.visible=false;f.gate.scale.setScalar(1);f.gate.material.emissiveIntensity=0;f.input.material.emissiveIntensity=.08;for(const path of [f.meanPath,f.squarePath,f.scalePath]){path.material.opacity=.18;path.material.emissiveIntensity=0;}}}
 }
 function rmsFrame(time,count=3){
  const start=.22,step=.9,end=start+count*step,duration=end+.3;
  const row=Math.max(0,Math.min(count-1,Math.floor((time-start)/step))),local=Math.max(0,Math.min(1,(time-start-row*step)/step));
  const phase=time<start?time/start*.18:time<end?.18+(time-start)/(count*step)*.62:.8+(time-end)/.3*.2;
  return {row,local,start,step,end,duration,phase:Math.max(0,Math.min(1,phase)),stage:local<.3?'squares':local<.56?'rms':local<.84?'scale':'ready'};
 }
 function processingRMS(u,time,input,output,sequence){
  const count=sequence?3:1,width=input.length/count,timing=rmsFrame(time,count),active=time>=0&&time<timing.duration;
  u.g.userData.processing=active;u.glow.material.emissiveIntensity=.12+(active?1:0);
  const states=[];
  for(const row of u.rmsRows){
   const values=input.slice(row.row*width,(row.row+1)*width),squares=values.map(v=>v*v),meanSquare=squares.reduce((a,b)=>a+b,0)/width,r=Math.sqrt(meanSquare+1e-6);
   const local=(time-timing.start-row.row*timing.step)/timing.step,focus=active&&local>=0&&local<1,done=active&&local>=.84;
   row.band.material.opacity=focus?.42:done?.2:.12;row.halo.scale.setScalar(1+(focus?.18*Math.sin(Math.PI*Math.max(0,Math.min(1,local))):0));
   row.halo.material.emissive.set(row.tone);row.halo.material.emissiveIntensity=focus?1.3:0;row.hub.material.emissiveIntensity=focus?1.1:.08;
   row.divisor.write(active&&local>=.3?'÷ '+Number(r.toFixed(2)):'r');
   for(const f of row.features){
    const value=values[f.index],result=output[row.row*width+f.index],square=squares[f.index];
    f.label.write(active?String(Number(value.toFixed(2))):'x');
    f.input.scale.setScalar(active?.65+Math.min(2,Math.abs(value))*.4:1);f.input.material.emissiveIntensity=focus?.8:.08;
    f.output.scale.setScalar(done?.65+Math.min(2,Math.abs(result))*.4:1);f.output.material.color.set(done&&result<0?0xc79868:row.tone);f.output.material.emissiveIntensity=focus&&local>=.56?1.1:done?.35:.08;
    f.squarePath.material.opacity=focus&&local<.56?.85:.2;f.scalePath.material.opacity=focus&&local>=.56?.85:.2;
    f.square.visible=focus&&local>=.05&&local<.38;f.scaled.visible=focus&&local>=.58&&local<.86;
    Object.assign(f.square.userData,{row:row.row,feature:f.index,value:square,input:value});
    Object.assign(f.scaled.userData,{row:row.row,feature:f.index,value:result,input:value,divisor:r});
    if(f.square.visible){f.square.position.copy(f.squarePath.geometry.parameters.path.getPoint(ramp(local,.05,.34)));f.square.scale.setScalar(.6+Math.min(3,square)*.2);f.square.material.wireframe=square===0;f.square.material.opacity=1-ramp(local,.32,.38);}
    if(f.scaled.visible){f.scaled.position.copy(f.scalePath.geometry.parameters.path.getPoint(ramp(local,.58,.82)));f.scaled.scale.setScalar(.65+Math.min(3,Math.abs(result))*.4);f.scaled.material.color.set(result<0?0xc79868:row.tone);f.scaled.material.wireframe=Math.abs(result)<1e-9;f.scaled.material.opacity=1-ramp(local,.82,.86);}
   }
   states.push({row:row.row,values,squares,meanSquare,divisor:r,output:output.slice(row.row*width,(row.row+1)*width)});
  }
  if(active)u.g.userData.rms={row:timing.row,stage:timing.stage,local:timing.local,groups:states};else delete u.g.userData.rms;
 }
 function clearAggregation(u){
  if(!u.mixing)return;
  delete u.g.userData.aggregation;
  u.valueLabels.forEach(label=>label.write('·'));
  u.weightCells.forEach(cell=>{cell.weight=0;cell.write('·');cell.tile.material.color.set(0xd7dfd4);delete cell.tile.userData.aggregationValue.value;delete cell.face.userData.aggregationValue.value;});
  u.connections.forEach(c=>{c.weight=0;c.spark.userData.weight=0;});
 }
 function processing(id,progress,input,output,color,{sequence=false,detailed=false,paired=false,other=null,words=[],valueWords=[]}={}){
  const u=units.get(id);if(!u)return;
  if(u.mixing){sequence=true;detailed=true;}
  const duration=u.selectionRows.length?selectionFrame(progress,6,u.selectionRows.length).duration:u.lnRows.length?layerNormFrame(progress,u.lnRows.length).duration:u.tokenKind?tokenFrame(progress,u.tokenKind).duration:u.rmsRows.length?rmsFrame(progress,sequence?3:1).duration:detailed&&u.projectionRows.length?sequenceProjectionFrame(progress,u.outputs.length,u.projectionRows.length).duration:detailed||paired?projectionFrame(progress,output.length).duration:1;
  u.focusAmount=ramp(progress,0,.12)*(1-ramp(progress,duration-.18,duration));
  if(u.selectionRows.length){processingSelection(u,progress,input,output,words);return;}
  if(u.lnRows.length){processingLayerNorm(u,progress,input,output,color);return;}
  if(u.tokenKind){processingToken(u,progress,input,output,words);return;}
  if(u.rmsRows.length){processingRMS(u,progress,input,output,sequence);return;}
  const wholeTime=progress,sequenceTiming=detailed&&u.projectionRows.length?sequenceProjectionFrame(progress,u.outputs.length,u.projectionRows.length):null;
  if(u.mixing){
   if(progress>=0&&progress<duration&&other){
    const columns=u.outputs.length;
    u.valueLabels.forEach((label,i)=>label.write(valueWords[i]||'·'));
    for(const cell of u.weightCells){
     const value=other[cell.i*columns+cell.j];cell.weight=value;cell.write(String(Number(value.toFixed(2))),value<0?'#8b563c':'#456052');
     cell.tile.material.color.set(Math.abs(value)<1e-9?0xd7dfd4:value<0?0xc89d70:u.color);cell.tile.userData.aggregationValue.value=cell.face.userData.aggregationValue.value=value;
    }
    u.connections.forEach(c=>{c.weight=other[c.i*columns+c.j];c.spark.userData.weight=c.weight;});
    u.g.userData.aggregation={query:sequenceTiming.row,column:sequenceTiming.column,attention:[...input],values:[...other],output:[...output],queryWord:words[sequenceTiming.row],valueWords:[...valueWords]};
   }else clearAggregation(u);
  }
  if(sequenceTiming){
   const active=wholeTime>=0&&wholeTime<sequenceTiming.duration,width=u.inputs.length,columns=u.outputs.length,row=sequenceTiming.row;
   for(const bank of u.projectionRows){bank.g.visible=active;bank.label.write(words[bank.row]||'·');bank.outputs.forEach((o,j)=>{const ready=bank.row*sequenceTiming.rowDuration+.97+j*sequenceTiming.step;o.material.emissiveIntensity=wholeTime>=ready?.4:.03;});}
   if(active)u.g.userData.sequenceProjection={row,column:sequenceTiming.column,rowTime:sequenceTiming.rowTime,input:[...input],output:[...output],word:words[row]};else delete u.g.userData.sequenceProjection;
   input=input.slice(row*width,(row+1)*width);output=output.slice(row*columns,(row+1)*columns);progress=sequenceTiming.rowTime;sequence=false;
  }
  const worked=detailed||paired,timing=worked?projectionFrame(progress,output.length):null,active=sequenceTiming?wholeTime>=0&&wholeTime<sequenceTiming.duration:progress>=0&&progress<(timing?.duration||1),p=timing?timing.phase:Math.max(0,Math.min(1,progress));u.g.userData.processing=active;
  const entering=ramp(p,0,.23),gathering=ramp(p,.6,.9),pulse=active?Math.sin(Math.PI*p):0;
  const focus=active&&worked?timing.focus:0,accent=paired||sequenceTiming?u.color:projectionColor;
  u.g.userData.calculationHighlight=focus;
  u.g.userData.activeColumn=detailed&&focus>0?timing.column:null;
  u.g.userData.activePair=paired&&focus>0?timing.column:null;
  u.columnCues.forEach(({cue,fill,outline},j)=>{const strength=j===timing?.column?focus:0;cue.visible=strength>0;fill.material.color.set(accent);outline.material.color.set(accent);fill.material.opacity=.19*strength;outline.material.opacity=strength;});
  u.glow.material.emissiveIntensity=.12+1.5*pulse;
  u.normalizationStages.forEach((tile,i)=>{const local=p*3-i,strength=active&&local>=0&&local<=1?Math.sin(Math.PI*local):0;tile.material.emissive.set(color);tile.material.emissiveIntensity=strength*1.4;});
  if(u.normalizationStages.length){const features=input.slice(0,sequence?input.length/3:input.length),mean=features.reduce((a,b)=>a+b,0)/features.length;u.g.userData.normalization={mean,centered:features.map(x=>x-mean),phase:Math.min(2,Math.floor(p*3))};}
  const inputTone=u.mixing?rowColor:featureColor;
  u.inputs.forEach((o,i)=>{const v=sequence?1:input[i],size=Math.abs(v)<1e-9?.55:1+Math.min(2,Math.abs(v))*.3;o.scale.setScalar(1+(size-1)*entering);if(u.connections.length){o.material.color.set(detailed&&active?inputTone(i):0x6c9eb1);o.material.emissive.set(detailed?inputTone(i):0x6c9eb1);}o.material.emissiveIntensity=.08+(active&&Math.abs(v)>1e-9?entering*(1-ramp(p,.65,1)):0);});
  for(const {mesh,spark,i,j,weight} of u.connections){
   const contribution=sequence?weight:input[i]*weight,lit=Number.isFinite(contribution)&&Math.abs(contribution)>1e-9;
   mesh.material.opacity=(active?(lit?.35+.6*pulse:.08):(weight===0?.2:.8))*(1-.92*u.focusAmount);mesh.material.depthWrite=u.focusAmount===0;
   mesh.material.emissive.set(lit?color:0);mesh.material.emissiveIntensity=active&&lit?pulse*.7:0;
   const start=detailed?.4+j*timing.step:.2,end=detailed?.67+j*timing.step:.76,clock=detailed?progress:p;
   spark.visible=active&&lit&&clock>=start&&clock<=end;
   Object.assign(spark.userData,{value:contribution,inputValue:input[i],tokenRow:sequenceTiming?.row});
   if(spark.visible){const tone=detailed?inputTone(i):contribution<0?0xd6a578:color;spark.material.color.set(tone);spark.material.emissive.set(tone);spark.material.opacity=ramp(clock,start,start+.025)*(1-ramp(clock,end-.025,end));spark.scale.setScalar(.48+Math.min(2,Math.abs(contribution))*.15);spark.position.copy(mesh.geometry.parameters.path.getPoint(ramp(clock,start,end)));}
  }
  for(const {tile,face,i,j,weight} of u.weightCells){const contribution=sequence?weight:input[i]*weight,lit=active&&(detailed||Math.abs(contribution)>1e-9),highlight=detailed?(j===timing.column?focus:0):pulse,tone=detailed?accent:color;tile.material.emissive.set(lit?tone:0);tile.material.emissiveIntensity=lit?highlight*.35:0;face.material.color.set(0xffffff);if(lit)face.material.color.lerp(new T.Color(tone),highlight*.4);}
  for(const {face,j,value}of u.biasCells){const strength=active&&j===timing?.column?ramp(progress-j*timing.step,.56,.61)*(1-ramp(progress-j*timing.step,.7,.76)):0;face.material.color.set(0xffffff).lerp(new T.Color(projectionColor),strength*.5);face.userData.vocabularyBias.active=strength>0;}
  for(const {channel,gate,spark,i} of u.activationChannels){
   const from=input[i],to=output[i],isReLU=u.g.userData.activation==='relu',sigmoid=isReLU?(from>0?1:0):1/(1+Math.exp(-from)),travel=ramp(p,.16,.84),transform=ramp(p,.35,.7),value=from+(to-from)*transform,tone=from<0?0xd6a578:color;
   gate.scale.setScalar(1-(1-(.22+.78*sigmoid))*pulse);gate.material.emissive.set(tone);gate.material.emissiveIntensity=active?pulse*.85:0;
   channel.material.emissive.set(tone);channel.material.emissiveIntensity=active?pulse*.9:0;
   spark.visible=active&&p>=.16&&p<.86&&Math.abs(from)>1e-9;
   Object.assign(spark.userData,{inputValue:from,outputValue:to,value,sigmoid,phase:transform});
   if(spark.visible){spark.position.copy(channel.geometry.parameters.path.getPoint(isReLU&&from<0?Math.min(.5,travel):travel));spark.scale.setScalar(.3+Math.min(3,Math.abs(value))*.5);spark.material.color.set(tone);spark.material.emissive.set(tone);spark.material.opacity=ramp(p,.16,.23)*(1-ramp(p,isReLU&&from<0?.5:.78,isReLU&&from<0?.7:.86));}
  }
  for(const {i,left,right,leftPath,rightPath,outPath,gate,carrier} of u.pairChannels){
   if(!paired||!other){carrier.visible=false;continue;}
   const local=progress-i*timing.step,a=input[i],b=other[i],value=output[i],highlight=i===timing.column?focus:0;
   [left,right].forEach((socket,slot)=>{const v=slot?b:a,size=Math.abs(v)<1e-9?.55:1+Math.min(2,Math.abs(v))*.3;socket.scale.setScalar(1+(size-1)*entering);socket.material.emissiveIntensity=.08+highlight*.95;});
   [leftPath,rightPath,outPath].forEach(path=>{path.material.emissive.set(u.color);path.material.emissiveIntensity=highlight*.85;});
   gate.scale.setScalar(1+highlight*.22);gate.material.emissive.set(u.color);gate.material.emissiveIntensity=highlight*1.5;
   carrier.visible=active&&local>=.53&&local<.73;
   const zero=Math.abs(value)<1e-9,tone=value<0?0xc79868:u.color;
   Object.assign(carrier.userData,{operation:u.pairOperation,inputA:a,inputB:b,value,phase:ramp(local,.53,.67)});
   carrier.position.copy(outPath.geometry.parameters.path.getPoint(ramp(local,.54,.67)));
   carrier.scale.setScalar(zero?.65:.65+Math.min(3,Math.abs(value))*.55);carrier.material.wireframe=zero;
   carrier.material.color.set(tone);carrier.material.emissive.set(tone);carrier.material.opacity=ramp(local,.53,.56)*(1-ramp(local,.67,.73));
  }
  u.outputs.forEach((o,i)=>{const v=sequence?1:output[i],size=Number.isFinite(v)?Math.abs(v)<1e-9?.55:1+Math.min(2,Math.abs(v))*.18:1,gather=worked?ramp(progress,.59+i*timing.step,.68+i*timing.step):gathering,highlight=worked&&i===timing.column?focus:0;o.scale.setScalar(1+(size-1)*gather);o.material.color.set(v<0?0xc79868:u.color);if(highlight)o.material.color.lerp(new T.Color(accent),highlight*.7);o.material.emissive.set(highlight?accent:u.color);o.material.emissiveIntensity=worked?.08+highlight*.85:.08+(active?1.3*gather*(1-ramp(p,.9,1)):progress>=1?.35:0);});
 }
 function clearRMS(u){
  delete u.g.userData.rms;
  for(const row of u.rmsRows){
   row.band.material.opacity=.12;row.halo.scale.setScalar(1);row.halo.material.emissiveIntensity=0;row.hub.material.emissiveIntensity=.08;row.divisor.write('r');
   for(const f of row.features){f.label.write('x');f.square.visible=f.scaled.visible=false;f.squarePath.material.opacity=f.scalePath.material.opacity=.2;f.input.scale.setScalar(1);f.output.scale.setScalar(1);f.input.material.emissiveIntensity=f.output.material.emissiveIntensity=.08;}
  }
 }
 function clearProcessing(){for(const u of units.values()){u.focusAmount=0;u.g.userData.calculationHighlight=0;clearRMS(u);clearToken(u);clearLayerNorm(u);clearAggregation(u);clearSelection(u);delete u.g.userData.sequenceProjection;u.projectionRows.forEach(row=>row.g.visible=false);u.g.userData.processing=false;u.g.userData.activeColumn=u.g.userData.activePair=null;delete u.g.userData.normalization;u.normalizationStages.forEach(tile=>tile.material.emissiveIntensity=0);u.columnCues.forEach(({cue})=>cue.visible=false);u.outputs.forEach(o=>{o.material.emissive.set(u.color);o.material.emissiveIntensity=.08;});if(u.connections.length)u.inputs.forEach(o=>{o.material.color.set(0x6c9eb1);o.material.emissive.set(0x6c9eb1);o.material.emissiveIntensity=.08;});u.connections.forEach(c=>{c.spark.visible=false;c.mesh.material.opacity=c.weight===0?.2:.8;c.mesh.material.depthWrite=true;});u.activationChannels.forEach(({spark,gate,channel})=>{spark.visible=false;gate.scale.setScalar(1);gate.material.emissiveIntensity=0;channel.material.emissiveIntensity=0;});u.pairChannels.forEach(({left,right,leftPath,rightPath,outPath,gate,carrier})=>{carrier.visible=false;gate.scale.setScalar(1);[leftPath,rightPath,outPath,gate].forEach(o=>o.material.emissiveIntensity=0);[left,right].forEach(o=>{o.scale.setScalar(1);o.material.emissiveIntensity=.08;});});u.weightCells.forEach(({tile,face})=>{tile.material.emissiveIntensity=0;face.material.color.set(0xffffff);});u.biasCells.forEach(({face})=>face.material.color.set(0xffffff));}}
 // A single colour pulse identifies the departing row without moving its tray.
 function pulseSource(progress=1){
  const pulse=Math.pow(1-Math.min(1,Math.max(0,progress)/.65),2);
  tokens.forEach((t,i)=>{const active=t.g.userData.active;t.panel.material.color.set(active?0xe6eacb:0xbfd8d3);if(active)t.panel.material.color.lerp(new T.Color(rowColor(i)),pulse*.28);t.stripe.material.emissiveIntensity=active?.65+pulse: .1;});
 }
 function packet(values,color,{sequence=false}={}){
  const g=new T.Group(),cols=sequence?values.length/3:values.length,count=sequence?3:1;g.userData.packet=true;
  // Features run left to right, just like the input/output racks. A sequence
  // adds token rows in depth; moving the packet never transposes its tensor.
  const tray=box(g,cols*.24+.1,.045,count*.3+.06,0,0,0,0xc7d9ce);tray.userData.packetTray=true;
  values.forEach((v,i)=>{const finite=Number.isFinite(v),zero=finite&&Math.abs(v)<1e-9,c=!finite||v<0?0xd6a578:zero?0xc8d4d0:color,h=zero?.035:.08+Math.min(finite?Math.abs(v):2,2)*.24;
   const x=(i%cols-(cols-1)/2)*.24,z=(Math.floor(i/cols)-(count-1)/2)*.3;
   const pad=box(g,.2,.025,.23,x,.035,z,0xe6ece2);pad.userData.featurePad=i;const bar=box(g,.17,h,.17,x,.06+h/2,z,c,{emissive:c,emissiveIntensity:zero?.02:.4});bar.userData={feature:i,value:v};
  });return g;
 }
 function prepareProjectionPacket(g,colorFor=featureColor){
  // The same blocks leave the horizontal tray. Their new positions illustrate
  // routing into the weight rows, not a transpose of the input tensor.
  g.userData.featureRoutes=g.children.filter(o=>o.userData.feature!=null).map(bar=>{
   const i=bar.userData.feature,pad=g.children.find(o=>o.userData.featurePad===i),v=bar.userData.value;
   const labelTile=textTile(g,String(Number(v.toFixed(2))),.31,.25,bar.position.x,bar.position.y+bar.geometry.parameters.height/2+.02,bar.position.z,{size:76,color:'#fffdf4',bg:'#'+colorFor(i).toString(16).padStart(6,'0')}),label=labelTile.mesh;
   label.visible=false;label.material.transparent=true;label.material.depthWrite=false;label.userData.featureValue=i;
   return {bar,pad,label,write:labelTile.write,home:bar.position.clone(),padHome:pad.position.clone(),labelHome:label.position.clone(),tone:bar.material.color.clone()};
  });
 }
 function routeProjectionPacket(id,mesh,p,origin,anchor){
  const u=units.get(id),routes=mesh.userData.featureRoutes;if(!u||!routes)return;
  const fan=ramp(p,.12,.38),alpha=1-ramp(p,.72,.86);mesh.userData.routing=fan>0;
  mesh.children.find(o=>o.userData.packetTray).material.opacity=(1-ramp(p,.12,.3))*alpha;
  for(const {bar,pad,label,home,padHome,labelHome,tone} of routes){
   const i=bar.userData.feature,target=new T.Vector3(u.g.position.x-1.12,u.g.position.y+.87,u.g.position.z+u.inputs[i].position.z);
   for(const [object,start] of [[bar,home],[pad,padHome],[label,labelHome]]){
    object.position.copy(start);
    if(p>=.12)object.position.copy(anchor).add(start).lerp(target.clone().add(new T.Vector3(0,start.y,0)),fan).sub(origin);
   }
   bar.material.color.copy(tone).lerp(new T.Color(featureColor(i)),fan);bar.material.emissive.copy(bar.material.color);
   bar.material.opacity=pad.material.opacity=alpha;
   label.visible=p>=.12&&p<.86;label.material.opacity=ramp(p,.12,.3)*alpha;
   Object.assign(bar.userData,{mappedInput:i,mappingProgress:fan});
  }
 }
 function routeSequenceProjectionInput(id,mesh,time,origin,anchor){
  const u=units.get(id),width=u.inputs.length,timing=sequenceProjectionFrame(time,u.outputs.length,u.projectionRows.length);
  mesh.userData.routing=time>0;mesh.children.find(o=>o.userData.packetTray).material.opacity=1-ramp(time,0,.2);
  for(const {bar,pad,label,home,padHome,labelHome,tone}of mesh.userData.featureRoutes){
   const index=bar.userData.feature,row=Math.floor(index/width),i=index%width,localTime=time-row*timing.rowDuration,p=projectionFrame(localTime,u.outputs.length).phase;
   const fan=ramp(p,.04,.38),alpha=1-ramp(p,.72,.86),target=new T.Vector3(u.g.position.x-1.12,u.g.position.y+.87,u.g.position.z+u.inputs[i].position.z);
   const waiting=anchor.clone().add(new T.Vector3(-.55*ramp(time,0,.2),0,0));
   for(const [object,start]of [[bar,home],[pad,padHome],[label,labelHome]])object.position.copy(waiting).add(start).lerp(target.clone().add(new T.Vector3(0,start.y,0)),fan).sub(origin);
   bar.material.color.copy(tone).lerp(new T.Color(u.mixing?rowColor(i):featureColor(i)),fan);bar.material.emissive.copy(bar.material.color);bar.material.opacity=pad.material.opacity=alpha;
   label.material.opacity=ramp(p,.04,.3)*alpha;label.visible=label.material.opacity>0&&detailModes[detailScope];
   Object.assign(bar.userData,{sequenceProjectionInput:true,tokenRow:row,mappedInput:i,mappingProgress:fan});
  }
 }
 function routeAggregationValues(id,mesh,time){
  const u=units.get(id),origin=new T.Vector3(-1.12,.68,.55).add(u.g.position),fan=ramp(time,0,.23),alpha=1-ramp(time,.23,.33);
  mesh.position.copy(origin);mesh.userData.routing=fan>0;
  mesh.children.find(o=>o.userData.packetTray).material.opacity=(1-ramp(time,0,.12))*alpha;
  for(const {bar,pad,label,home,padHome,labelHome}of mesh.userData.featureRoutes){
   const cell=u.weightCells[bar.userData.feature],target=cell.face.position.clone().add(u.g.position);
   for(const [object,start]of [[bar,home],[pad,padHome],[label,labelHome]]){
    const end=target.clone().add(new T.Vector3(0,object===label?.22:object===bar?bar.geometry.parameters.height/2+.025:.01,0));
    object.position.copy(origin).add(start).lerp(end,fan).sub(origin);object.material.opacity=alpha;
   }
   bar.material.color.set(rowColor(cell.i));bar.material.emissive.copy(bar.material.color);label.visible=alpha>0&&detailModes[detailScope];
   Object.assign(bar.userData,{aggregationInput:true,source:cell.i,column:cell.j,mappingProgress:fan});
  }
 }
 function routeSequenceProjectionOutput(id,mesh,time,origin){
  const u=units.get(id),columns=u.outputs.length,timing=sequenceProjectionFrame(time,columns,u.projectionRows.length),gather=ramp(timing.phase,.8,.97);
  mesh.userData.gather=gather;mesh.userData.routing=gather<1;
  mesh.children.forEach((o,i)=>{
   if(o.userData.packetTray){o.material.opacity=ramp(timing.phase,.8,.94);return;}
   const index=o.userData.feature??o.userData.featurePad,row=Math.floor(index/columns),column=index%columns,home=mesh.userData.outputHomes[i],ready=row*timing.rowDuration+.67+column*timing.step;
   const alpha=ramp(time,ready,ready+.06),transfer=ramp(time,ready+.08,ready+.3);o.visible=alpha>0;o.material.opacity=alpha;
   const start=projectionOutlet(id,column).add(new T.Vector3(.28,home.y,0)),bank=u.projectionRows[row].outputs[column].getWorldPosition(new T.Vector3()).add(new T.Vector3(0,home.y,0));
   const target=origin.clone().add(home);o.position.copy(start.lerp(bank,transfer).lerp(target,gather).sub(origin));
   Object.assign(o.userData,{projectionToken:row,projectionFeature:column,transfer});
  });
 }
 function routePairedPacket(id,mesh,time,slot){
  const u=units.get(id),routes=mesh.userData.featureRoutes;if(!u||!routes)return;
  const timing=projectionFrame(time,routes.length),p=timing.phase,fan=ramp(p,.1,.36);
  // Each original horizontal tray enters through its own port. Its values fan
  // to matching sockets, then meet at that pair's + or × ring.
  const origin=new T.Vector3(-1.12,.68,slot?.55:-.55).add(u.g.position);
  mesh.position.copy(origin);mesh.userData.routing=fan>0;
  mesh.children.find(o=>o.userData.packetTray).material.opacity=1-ramp(p,.1,.3);
  for(const {bar,pad,label,home,padHome,labelHome,tone} of routes){
   const i=bar.userData.feature,channel=u.pairChannels[i],path=slot?channel.rightPath:channel.leftPath;
   const local=time-i*timing.step,meet=ramp(local,.42,.54),alpha=1-ramp(local,.5,.56);
   const target=path.geometry.parameters.path.getPoint(meet).add(u.g.position);
   for(const [object,start] of [[bar,home],[pad,padHome],[label,labelHome]])object.position.copy(origin).add(start).lerp(target.clone().add(new T.Vector3(0,start.y*(1-meet),0)),fan).sub(origin);
   bar.material.color.copy(tone).lerp(new T.Color(u.pairTones[slot]),fan);bar.material.emissive.copy(bar.material.color);bar.material.emissiveIntensity=.4+(i===timing.column?timing.focus:0);
   bar.material.opacity=alpha;pad.material.opacity=alpha*(1-ramp(local,.4,.5));
   label.material.opacity=ramp(p,.1,.28)*(1-ramp(local,.47,.51));label.visible=label.material.opacity>0;
   Object.assign(bar.userData,{[u.pairOperation==='add'?'additionInput':'multiplyInput']:slot,mappedInput:i,mappingProgress:fan,pairProgress:meet});
  }
 }
 function routeTokenInput(id,mesh,time,origin,anchor){
  if(units.get(id).tokenKind==='mask'){routeMaskInput(id,mesh,time,origin,anchor);return;}
  const u=units.get(id),timing=tokenFrame(time,u.tokenKind),lookup=u.tokenKind==='lookup',width=u.tokenWidth,fan=ramp(timing.phase,.04,.17);
  mesh.userData.routing=fan>0;
  mesh.children.find(o=>o.userData.packetTray).material.opacity=1-ramp(timing.phase,.04,.17);
  for(const {bar,pad,label,write,home,padHome,labelHome} of mesh.userData.featureRoutes){
   const index=bar.userData.feature,row=lookup?index:Math.floor(index/width),j=index%width,value=bar.userData.value,local=(time-timing.start-row*timing.step)/timing.step;
   const to=lookup?ramp(local,.02,.27):ramp(local,.02,.32),alpha=1-ramp(local,lookup?.46:.56,lookup?.6:.64);
   const rowData=u.tokenRows[row],target=lookup?new T.Vector3(-1.12,.78,rowData.z).lerp(u.lookupRows[value].address,to):rowData.features[j].input.position.clone().lerp(new T.Vector3(rowData.features[j].output.position.x,1.4,rowData.z),to);
   target.add(u.g.position);
   for(const [object,start] of [[bar,home],[pad,padHome],[label,labelHome]]){
    const end=target.clone().add(new T.Vector3(0,object===label?.22:start.y*(1-to),0));
    object.position.copy(timing.phase<.04?origin:anchor).add(start).lerp(end,fan).sub(origin);
   }
   const shown=lookup?value:value*(1+(Math.sqrt(width)-1)*to),height=lookup?.055:Math.abs(shown)<1e-9?.035:.08+Math.min(Math.abs(shown),2)*.24;
   bar.scale.y=1+(height/bar.geometry.parameters.height-1)*fan;
   const tone=lookup?rowColor(row):featureColor(j);bar.material.color.set(tone);bar.material.emissive.set(tone);bar.material.emissiveIntensity=.4+(local>=0&&local<.6?.6:0);bar.material.opacity=alpha;
   pad.material.opacity=(1-fan)*alpha;label.material.opacity=fan*alpha;label.visible=fan>0&&alpha>0&&detailModes[detailScope];
   write(lookup?'ID '+value:String(Number(shown.toFixed(2))));
   Object.assign(bar.userData,{tokenInput:true,tokenRow:row,lookupId:lookup?value:null,mappingProgress:to,displayValue:shown});
  }
 }
 function routeMaskInput(id,mesh,time,origin,anchor){
  const u=units.get(id),timing=tokenFrame(time,'mask'),fan=ramp(timing.phase,.04,.17);
  mesh.userData.routing=fan>0;mesh.children.find(o=>o.userData.packetTray).material.opacity=1-fan;
  for(const {bar,pad,label,write,home,padHome,labelHome}of mesh.userData.featureRoutes){
   const index=bar.userData.feature,row=Math.floor(index/3),f=u.tokenRows[row].features[index%3],local=(time-timing.start-row*timing.step)/timing.step;
   const advance=ramp(local,.06,.46),alpha=1-ramp(local,.46,.57),target=f.input.position.clone().lerp(new T.Vector3(f.output.position.x,1.14,u.tokenRows[row].z),advance).add(u.g.position);
   for(const [object,start]of [[bar,home],[pad,padHome],[label,labelHome]])object.position.copy(timing.phase<.04?origin:anchor).add(start).lerp(target.clone().add(new T.Vector3(0,object===label?.22:0,0)),fan).sub(origin);
   const tone=f.blocked?0xc58b65:rowColor(row);bar.material.color.set(tone);bar.material.emissive.set(tone);bar.material.emissiveIntensity=local>=0&&local<.57?1:.25;bar.material.opacity=alpha;
   pad.material.opacity=(1-fan)*alpha;label.material.opacity=fan*alpha;label.visible=fan>0&&alpha>0&&detailModes[detailScope];write(bar.userData.value===-Infinity?'−∞':String(Number(bar.userData.value.toFixed(2))));
   Object.assign(bar.userData,{maskInput:true,tokenRow:row,blocked:f.blocked,mappingProgress:advance});
  }
 }
 function routeLayerNormInput(id,mesh,time,origin,anchor,input,output){
  const u=units.get(id),count=u.lnRows.length,width=input.length/count,timing=layerNormFrame(time,count),fan=ramp(timing.phase,.04,.17);
  mesh.userData.routing=fan>0;mesh.children.find(o=>o.userData.packetTray).material.opacity=1-fan;
  const stats=u.lnRows.map(row=>layerNormStats(input.slice(row.row*width,(row.row+1)*width)));
  for(const {bar,pad,label,write,home,padHome,labelHome} of mesh.userData.featureRoutes){
   const index=bar.userData.feature,row=Math.floor(index/width),j=index%width,f=u.lnRows[row].features[j],s=stats[row],local=(time-timing.start-row*timing.step)/timing.step;
   const centering=ramp(local,.24,.43),scaling=ramp(local,.73,.89),alpha=1-ramp(local,.89,.95),centered=input[index]-s.mean*centering,shown=centered+(output[index]-centered)*scaling;
   const height=Math.abs(shown)<1e-9?.035:.08+Math.min(Math.abs(shown),2)*.24;
   const target=f.input.position.clone().lerp(f.center,centering).lerp(f.output.position.clone().add(new T.Vector3(0,.1+height/2,0)),scaling).add(u.g.position);
   for(const [object,start]of [[bar,home],[pad,padHome],[label,labelHome]]){
    const end=target.clone().add(new T.Vector3(0,object===label?.3:start.y*(1-scaling),0));
    if(object===bar&&scaling===1)end.copy(target);
    object.position.copy(timing.phase<.04?origin:anchor).add(start).lerp(end,fan).sub(origin);
   }
   const tone=shown<0?0xc79868:featureColor(j);bar.scale.y=1+(height/bar.geometry.parameters.height-1)*fan;bar.material.color.set(tone);bar.material.emissive.set(tone);bar.material.emissiveIntensity=local>=0&&local<.9?1:.3;bar.material.opacity=alpha;bar.material.wireframe=Math.abs(shown)<1e-9;
   pad.material.opacity=(1-fan)*alpha;label.material.opacity=fan*alpha;label.visible=fan>0&&alpha>0&&detailModes[detailScope];write(String(Number(shown.toFixed(3))));
   Object.assign(bar.userData,{lnInput:true,row,displayValue:shown,mean:s.mean,centered:s.centered[j],divisor:s.divisor,centering,scaling});
  }
 }
 function layerNormAnchor(id,row,time){
  const u=units.get(id),r=u.lnRows[row],local=layerNormFrame(time,u.lnRows.length).local;
  return r.meanHub.position.clone().lerp(r.scaleHub.position,ramp(local,.43,.68)).lerp(new T.Vector3(0,1.9,r.z),ramp(local,.73,.9)).add(new T.Vector3(0,.18,0)).add(u.g.position);
 }
 function routeLayerNormOutput(id,mesh,time,origin){
  const u=units.get(id),count=u.lnRows.length,width=u.outputs.length/count,timing=layerNormFrame(time,count),gather=ramp(timing.phase,.8,.97);
  mesh.userData.gather=gather;mesh.userData.routing=gather<1;
  mesh.children.forEach((o,i)=>{
   if(o.userData.packetTray){o.material.opacity=ramp(timing.phase,.8,.94);return;}
   const feature=o.userData.feature??o.userData.featurePad,row=Math.floor(feature/width),home=mesh.userData.outputHomes[i],ready=timing.start+(row+.89)*timing.step,alpha=ramp(time,ready,ready+.07);
   o.visible=alpha>0;o.material.opacity=alpha;
   const start=projectionOutlet(id,feature).add(new T.Vector3(0,home.y+.04,0)),target=origin.clone().add(home);
   o.position.copy(gather===1?home:start.lerp(target,gather).sub(origin));
  });
 }
 function tokenAnchor(id,row,input,time){
  const u=units.get(id),r=u.tokenRows[row],target=new T.Vector3(0,1.75,r.z);
  if(u.tokenKind==='lookup')target.copy(u.lookupRows[input[row]].address).lerp(new T.Vector3(0,1.75,r.z),ramp(tokenFrame(time,u.tokenKind).local,.3,.84));
  return target.add(u.g.position);
 }
 function routeTokenOutput(id,mesh,time,origin){
  const u=units.get(id),timing=tokenFrame(time,u.tokenKind),gather=ramp(timing.phase,.8,.97);
  mesh.userData.gather=gather;mesh.userData.routing=gather<1;
  mesh.children.forEach((o,i)=>{
   if(o.userData.packetTray){o.material.opacity=ramp(timing.phase,.8,.94);return;}
   const feature=o.userData.feature??o.userData.featurePad,row=Math.floor(feature/u.tokenWidth),home=mesh.userData.outputHomes[i];
   const ready=timing.start+(row+.82)*timing.step,alpha=ramp(time,ready,ready+.06);
   o.visible=alpha>0;o.material.opacity=alpha;
   const start=projectionOutlet(id,feature).add(new T.Vector3(0,home.y+.02,0)),target=origin.clone().add(home);
   o.position.copy(gather===1?home:start.lerp(target,gather).sub(origin));
  });
 }
 function projectionOutlet(id,column){return units.get(id).outputs[column].getWorldPosition(new T.Vector3());}
 function projectionRadius(id,column){const sphere=units.get(id).outputs[column];return sphere.geometry.parameters.radius*sphere.getWorldScale(new T.Vector3()).x;}
 function prepareProjectionOutput(mesh){
  mesh.userData.outputHomes=mesh.children.map(o=>o.position.clone());mesh.userData.gather=0;mesh.userData.routing=true;
 }
 function routeProjectionOutput(id,mesh,time,origin){
  const columns=units.get(id).outputs.length,timing=projectionFrame(time,columns),gather=ramp(timing.phase,.8,.96);
  mesh.userData.gather=gather;mesh.userData.routing=gather<1;
  mesh.children.forEach((o,i)=>{
   if(o.userData.packetTray){o.material.opacity=ramp(timing.phase,.8,.94);return;}
   const j=o.userData.feature??o.userData.featurePad,home=mesh.userData.outputHomes[i],alpha=ramp(time,.67+j*timing.step,.73+j*timing.step);
   o.visible=alpha>0;o.material.opacity=alpha;
   // Each result waits at its own output neuron. Once all have arrived,
   // assemble the same components in feature order on the outgoing tray.
   const start=projectionOutlet(id,j).add(new T.Vector3(.28,home.y,0)),target=origin.clone().add(home);
   if(gather===1)o.position.copy(home);else o.position.copy(start.lerp(target,gather).sub(origin));
  });
 }
 function rmsAnchor(id,row){return units.get(id).rmsRows[row].hub.getWorldPosition(new T.Vector3()).add(new T.Vector3(0,.2,0));}
 function routeRMSOutput(id,mesh,time,origin){
  const u=units.get(id),count=u.rmsRows.length,width=u.outputs.length/count,timing=rmsFrame(time,count),gather=ramp(timing.phase,.8,.97);
  mesh.userData.gather=gather;mesh.userData.routing=gather<1;
  mesh.children.forEach((o,i)=>{
   if(o.userData.packetTray){o.material.opacity=ramp(timing.phase,.8,.94);return;}
   const feature=o.userData.feature??o.userData.featurePad,row=Math.floor(feature/width),home=mesh.userData.outputHomes[i];
   const ready=timing.start+row*timing.step+.82*timing.step,alpha=ramp(time,ready,ready+.06);
   o.visible=alpha>0;o.material.opacity=alpha;
   const start=projectionOutlet(id,feature).add(new T.Vector3(0,home.y+.04,0)),target=origin.clone().add(home);
   if(gather===1)o.position.copy(home);else o.position.copy(start.lerp(target,gather).sub(origin));
  });
 }
 function fusionVisual(assembly,wires,kind,definition,weights){
  const root=new T.Group(),ghost=new T.Group(),center=new T.Vector3(0,1.2,0),ownedGeometry=new Map(),ownedMaterials=new Map();
  root.userData.fusion={kind,phase:0,count:assembly.children.length};root.add(ghost);
  const copyMaterial=m=>{if(!ownedMaterials.has(m)){const copy=m.clone();if(m.map){copy.map=m.map.clone();copy.map.needsUpdate=true;}ownedMaterials.set(m,copy);}return ownedMaterials.get(m);};
  for(const source of [assembly,wires]){
   const copy=source.clone(true);copy.visible=true;
   copy.traverse(o=>{
    if(o.geometry){if(!ownedGeometry.has(o.geometry)){const geometry=o.geometry.clone();geometry.userData={...geometry.userData,shared:false};ownedGeometry.set(o.geometry,geometry);}o.geometry=ownedGeometry.get(o.geometry);}
    if(o.material)o.material=Array.isArray(o.material)?o.material.map(copyMaterial):copyMaterial(o.material);
    o.castShadow=o.receiveShadow=false;
   });ghost.add(copy);
  }
  const packed=module(root,{id:'fusion-preview',kind,x:0,z:0,links:definition.ins.map(()=>null)},definition,weights,{register:false});packed.position.copy(center);
  const shade=group=>{const list=[];group.traverse(o=>{if(o.material)for(const material of Array.isArray(o.material)?o.material:[o.material]){list.push([material,material.opacity]);material.transparent=true;material.depthWrite=false;material.needsUpdate=true;}});return list;};
  const ghostShades=shade(ghost),packedShades=shade(packed),halo=new T.Group();halo.position.copy(center);root.add(halo);
  for(let i=0;i<3;i++){const mesh=ring(halo,1.5+i*.18,0,.2+i*.16,0,colors[kind]||0x69ab97,.022);mesh.material.transparent=true;mesh.material.depthWrite=false;mesh.material.emissive.set(colors[kind]||0x69ab97);mesh.material.emissiveIntensity=.8;}
  const sparks=assembly.children.map((node,i)=>{const spark=new T.Mesh(new T.SphereGeometry(.07,8,6),mat(featureColor(i),{emissive:featureColor(i),emissiveIntensity:1.8,transparent:true,depthWrite:false}));spark.userData.start=node.position.clone().add(new T.Vector3(0,1,0));root.add(spark);return spark;});
  const clamp=x=>Math.max(0,Math.min(1,x)),ease=x=>{x=clamp(x);return x*x*(3-2*x);},fade=(list,amount)=>list.forEach(([m,opacity])=>m.opacity=opacity*amount);
  let disposed=false;scene.add(root);
  return {root,packed,center,
   update(phase,destination=center,reduced=false){
    const gather=ease(phase/.44),reveal=ease((phase-.32)/.17),flight=ease((phase-.68)/.23);
    root.userData.fusion.phase=phase;root.userData.fusion.stage=phase<.38?'gather':phase<.68?'formed':'shelf';
    ghost.visible=!reduced&&phase<.46;ghost.scale.setScalar(1-.96*gather);ghost.position.y=.9*gather;ghost.rotation.y=.18*Math.sin(gather*Math.PI);fade(ghostShades,1-ease((phase-.23)/.21));
    packed.visible=reduced?phase<.8:reveal>0;packed.scale.setScalar(reduced?1.55:Math.max(.001,reveal*1.65*(1-.83*flight)));
    packed.position.copy(reduced?center:center.clone().lerp(destination,flight));packed.position.y+=reduced?0:Math.sin(flight*Math.PI)*.7;
    packed.rotation.y=reduced?0:(1-reveal)*-.65+Math.sin(phase*Math.PI)*.12;fade(packedShades,reduced?1:reveal*(1-ease((phase-.88)/.07)));
    halo.visible=!reduced;halo.scale.setScalar(.8+reveal*.8);halo.children.forEach((ring,i)=>{ring.material.opacity=Math.sin(clamp((phase-.25)/.44)*Math.PI)*.65;ring.rotation.z=phase*2+i*.2;});
    sparks.forEach((spark,i)=>{const travel=ease((phase-.08-i*.012)/.31);spark.visible=!reduced&&phase<.43;spark.position.copy(spark.userData.start).lerp(center,travel);spark.position.y+=Math.sin(travel*Math.PI)*1.1;spark.material.opacity=Math.sin(travel*Math.PI);});
   },
   dispose(){if(disposed)return;disposed=true;scene.remove(root);const geometry=new Set(),materials=new Set(),textures=new Set();root.traverse(o=>{if(o.geometry&&!o.geometry.userData.shared)geometry.add(o.geometry);if(o.material)for(const m of Array.isArray(o.material)?o.material:[o.material]){materials.add(m);if(m.map)textures.add(m.map);}});geometry.forEach(g=>g.dispose());textures.forEach(t=>t.dispose());materials.forEach(m=>m.dispose());}
  };
 }
 function resetUnits(){units.clear();}
 return {outputReadout,selectionFrame,selectionChoice,selectionAnchor,routeSelectionInput,routeSelectionOutput,surroundings,module,sync,fusionVisual,focusCalculations,updateDetail,detailsVisible:()=>detailModes[detailScope],pulseSource,layerNormFrame,layerNormStats,layerNormAnchor,routeLayerNormInput,routeLayerNormOutput,tokenOperation,tokenFrame,tokenAnchor,routeTokenInput,routeTokenOutput,projectionFrame,sequenceProjectionFrame,routeSequenceProjectionInput,routeSequenceProjectionOutput,routeAggregationValues,rmsFrame,rmsAnchor,routeRMSOutput,projectionColor,pairColorsFor:id=>units.get(id)?.pairTones||pairColors,processing,clearProcessing,packet,prepareProjectionPacket,routeProjectionPacket,routePairedPacket,projectionOutlet,projectionRadius,prepareProjectionOutput,routeProjectionOutput,featureColor,rowColor,rowZ,sourceOffset,resetUnits,focus(active){context.visible=!active;floor.visible=!active&&!context.userData.layout?.floating;detailScope=active?'card':'board';if(active)detailModes.card=true;}};
}
