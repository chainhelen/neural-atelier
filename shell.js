(function(scope){
 scope.AtelierShell={init({beforeLanguageChange=()=>{},onEnter=()=>{},onHome=()=>{}}={}){
  const $=s=>document.querySelector(s),settings=$('#settings'),welcome=$('#welcome'),audioKey='neural-atelier-audio-v1';
  let options={enabled:true,volume:.35},leaving=false,enterVersion=0;
  try{const saved=JSON.parse(scope.localStorage?.getItem(audioKey));if(typeof saved?.enabled==='boolean')options.enabled=saved.enabled;if(Number.isFinite(saved?.volume))options.volume=Math.max(0,Math.min(1,saved.volume));}catch{}
  const unavailable=()=>{$('#audio-status').hidden=false;$('#audio-status').textContent=I18n.t('这个浏览器暂时无法播放音乐。');};
  const music=scope.createAtelierMusic?.({...options,onUnavailable:unavailable});
  const persist=()=>{try{scope.localStorage?.setItem(audioKey,JSON.stringify(options));}catch{}};
  function sync(){
   $('#language-choice').value=I18n.preference();$('#language-current').textContent=I18n.language()==='zh'?I18n.t("简体中文"):'English';
   if(!$('#audio-status').hidden)unavailable();
   $('#music-enabled').setAttribute('aria-checked',String(options.enabled));$('#music-volume').value=String(Math.round(options.volume*100));$('#music-volume-value').textContent=Math.round(options.volume*100)+'%';
  }
  function open(){scope.AtelierProgressResetUI?.clear();sync();if(!settings.open)settings.showModal();}
  function close(){scope.AtelierProgressResetUI?.clear();settings.close();(welcome.open?$('#welcome-settings'):$('#settings-open')).focus({preventScroll:true});}
  function cancelEnter(){enterVersion++;leaving=false;welcome.classList.remove('is-leaving');}
  function enter(){
   if(leaving)return;leaving=true;const version=++enterVersion;music?.unlock();welcome.classList.add('is-leaving');
   const finish=()=>{if(version!==enterVersion)return;welcome.close();welcome.classList.remove('is-leaving');leaving=false;onEnter();};
   if(scope.matchMedia?.('(prefers-reduced-motion: reduce)').matches)finish();else scope.setTimeout(finish,420);
  }
  $('#settings-open').onclick=open;$('#welcome-settings').onclick=open;
  $('#settings-close').onclick=close;$('#settings-done').onclick=close;
  settings.addEventListener('cancel',e=>{e.preventDefault();close();});
  settings.addEventListener('click',e=>{if(e.target!==settings)return;const r=settings.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)close();});
  $('#language-choice').addEventListener('change',e=>{beforeLanguageChange();I18n.setPreference(e.target.value);});
  $('#music-enabled').onclick=()=>{options.enabled=!options.enabled;music?.setEnabled(options.enabled);if(options.enabled)music?.unlock();persist();sync();};
  $('#music-volume').addEventListener('input',e=>{options.volume=Math.max(0,Math.min(100,Number(e.target.value)||0))/100;music?.setVolume(options.volume);persist();sync();});
  $('#enter-workshop').onclick=enter;welcome.addEventListener('cancel',e=>{e.preventDefault();enter();});
  function showScreen(screen){
   cancelEnter();settings.close();
   if(screen!=='home'){welcome.close();return;}
   if(!welcome.open){welcome.showModal();$('#welcome-title').focus({preventScroll:true});}welcome.classList.remove('replaying');void welcome.offsetWidth;welcome.classList.add('replaying');
  }
  function home(e){e?.preventDefault();onHome();}
  $('#intro-replay').onclick=home;$('#home-open').onclick=home;
  I18n.onChange(sync);sync();
  return {music,open,close,showScreen,cancelEnter};
 }};
})(globalThis);
