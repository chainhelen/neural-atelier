(function(scope){
 const key='neural-atelier-language-v1',catalog=scope.AtelierEnglish||{},listeners=new Set(),missing=new Set();
 let preference='en',active=false,bindings=[];
 try{const saved=scope.localStorage?.getItem(key);if(['auto','zh','en'].includes(saved))preference=saved;}catch{}
 function recommended(){
  const languages=scope.navigator?.languages?.length?scope.navigator.languages:[scope.navigator?.language||'en'];
  for(const language of languages){if(/^zh(?:-|$)/i.test(language))return 'zh';if(/^en(?:-|$)/i.test(language))return 'en';}
  return 'en';
 }
 const language=()=>preference==='auto'?recommended():preference;
 function t(source){
  if(!active||language()==='zh'||typeof source!=='string')return source;
  if(Object.prototype.hasOwnProperty.call(catalog,source))return catalog[source];
  const chapter=source.match(/^(序章|正式关卡|网络|引擎|番外|FFN 进阶|原版基础|现代扩展) \/ (\d+)$/);
  if(chapter)return t(chapter[1])+' / '+chapter[2];
  if(/[\u3400-\u9fff]/.test(source))missing.add(source);
  return source;
 }
 function prepare(root,seen=new WeakSet()){
  if(!root||typeof root!=='object'||seen.has(root))return root;seen.add(root);
  for(const key of Object.keys(root)){
   const descriptor=Object.getOwnPropertyDescriptor(root,key);if(descriptor.get)continue;
   const value=root[key];
   if(typeof value==='string'&&/[\u3400-\u9fff]/.test(value)){
    let source=value;Object.defineProperty(root,key,{enumerable:true,configurable:true,get:()=>t(source),set:value=>source=value});
   }else if(value&&typeof value==='object')prepare(value,seen);
  }
  return root;
 }
 function bindStatic(document){
  const walk=node=>{if(node.nodeType===3){const raw=node.textContent,source=raw.trim();if(/[\u3400-\u9fff]/.test(source))bindings.push(()=>node.textContent=raw.replace(source,t(source)));return;}
   if(['SCRIPT','STYLE'].includes(node.tagName))return;
   for(const attr of ['aria-label','aria-description','title','placeholder']){const source=node.getAttribute?.(attr);if(source&&/[\u3400-\u9fff]/.test(source))bindings.push(()=>node.setAttribute(attr,t(source)));}
   for(const child of node.childNodes||[])walk(child);
  };walk(document.documentElement);
 }
 function updateDocument(){if(scope.document){scope.document.documentElement.lang=language()==='zh'?'zh-CN':'en';bindings.forEach(apply=>apply());}}
 function setPreference(value){if(!['auto','zh','en'].includes(value))return;preference=value;try{scope.localStorage?.setItem(key,value);}catch{}updateDocument();listeners.forEach(fn=>fn());}
 scope.I18n={t,prepare,language,recommended,preference:()=>preference,setPreference,missing,
  activate(){if(!active&&scope.document)bindStatic(scope.document);active=true;updateDocument();},
  onChange(fn){listeners.add(fn);return ()=>listeners.delete(fn);},
  joinWords(words){return words.join(language()==='en'?' ':'');}
 };
})(globalThis);
