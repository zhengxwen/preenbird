const sel=new Map();
const $=id=>document.getElementById(id);
let MUSIC=[];                                            // [{path,name,dir}]
let curJob=null;
const fmt=s=>{s=Math.round(s);return (s/60|0)+':'+String(s%60).padStart(2,'0')};
const clean=(s,n=72)=>{if(!s)return '';s=String(s).replace(/`{1,}\w*/g,'').replace(/[^\x20-\x7E]+/g,' ');
  s=s.replace(/(.{1,4}?)\1{3,}/g,'$1').replace(/[^\w.!?)\]]+$/,'').replace(/\s+/g,' ').trim();
  return s.length>n?s.slice(0,n)+'…':s};
function applyThemeLabel(){var t=localStorage.getItem('preen-theme')||'system';var b=$('themeBtn');
  if(b)b.textContent={system:'🖥 System',light:'☀️ Light',dark:'🌙 Dark'}[t];}
function cycleTheme(){var o=['system','light','dark'],t=localStorage.getItem('preen-theme')||'system';
  t=o[(o.indexOf(t)+1)%o.length];localStorage.setItem('preen-theme',t);
  if(t==='system')delete document.documentElement.dataset.theme;else document.documentElement.dataset.theme=t;
  applyThemeLabel();}
const cardAdj=new Map();                                 // per-card grade, keyed by stem#idx
const svgDefs=$('svgdefs');const SVGNS='http://www.w3.org/2000/svg';
function ensureFilter(fid){
  if(document.getElementById(fid))return;
  const f=document.createElementNS(SVGNS,'filter');
  f.setAttribute('id',fid);f.setAttribute('color-interpolation-filters','sRGB');
  const cm=document.createElementNS(SVGNS,'feColorMatrix');cm.setAttribute('type','matrix');
  cm.setAttribute('values','1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0');
  const cv=document.createElementNS(SVGNS,'feConvolveMatrix');
  cv.setAttribute('order','3');cv.setAttribute('preserveAlpha','true');cv.setAttribute('kernelMatrix','0 0 0 0 1 0 0 0 0');
  f.appendChild(cm);f.appendChild(cv);svgDefs.appendChild(f);
}
function updateFilter(fid,a){
  const f=document.getElementById(fid);if(!f)return;const w=a.warmth,k=a.sharpen;
  f.querySelector('feColorMatrix').setAttribute('values',(1+w)+' 0 0 0 0  0 1 0 0 0  0 0 '+(1-w)+' 0 0  0 0 0 1 0');
  f.querySelector('feConvolveMatrix').setAttribute('kernelMatrix','0 '+(-k)+' 0 '+(-k)+' '+(1+4*k)+' '+(-k)+' 0 '+(-k)+' 0');
}
function filterStr(a,fid){var s='brightness('+(1+a.brightness).toFixed(3)+') contrast('+a.contrast+') saturate('+a.saturation+')';
  if(a.sharpen||a.warmth)s+=' url(#'+fid+')';return s;}
function toggleOut(){var b=$('outbar');b.style.display=b.style.display==='flex'?'none':'flex';}
function syncAspect(){document.documentElement.style.setProperty('--mediaAR',$('aspect').value==='1080x1560'?'1080/1560':'9/16');}
let SC=null;  // scrub state: {inp,outp,mIn,mOut,D}
function mPct(t){return SC&&SC.D?Math.max(0,Math.min(100,t/SC.D*100)):0;}
function mRender(){if(!SC)return;var i=SC.mIn,o=SC.mOut;
  $('mband').style.left=mPct(i)+'%';$('mband').style.width=Math.max(0,mPct(o)-mPct(i))+'%';
  $('mhin').style.left=mPct(i)+'%';$('mhout').style.left=mPct(o)+'%';
  if(document.activeElement!==$('min'))$('min').value=i.toFixed(1);
  if(document.activeElement!==$('mout'))$('mout').value=o.toFixed(1);
  $('mlen').textContent='= '+(o-i).toFixed(1)+'s';}
function mWriteCard(){if(!SC)return;SC.inp.value=+SC.mIn.toFixed(1);SC.outp.value=+SC.mOut.toFixed(1);
  SC.inp.dispatchEvent(new Event('input'));SC.outp.dispatchEvent(new Event('input'));}
function setMIn(t){if(!SC)return;SC.mIn=Math.max(0,Math.min(t,SC.mOut-0.1));mRender();mWriteCard();}
function setMOut(t){if(!SC)return;SC.mOut=Math.min(SC.D,Math.max(t,SC.mIn+0.1));mRender();mWriteCard();}
function mHead(){if(!SC)return;var v=$('mvid');$('mhead').style.left=mPct(v.currentTime)+'%';
  $('mtime').textContent=v.currentTime.toFixed(1)+' / '+(v.duration||0).toFixed(1)+'s';}
function mTime(x){var r=$('mtrack').getBoundingClientRect();return Math.max(0,Math.min(1,(x-r.left)/r.width))*(SC?SC.D:0);}
function mToggle(){var v=$('mvid');if(v.paused){v.play();$('mplay').textContent='⏸ Pause';}else{v.pause();$('mplay').textContent='▶ Play';}}
function mWire(){var tk=$('mtrack');if(tk._wired)return;tk._wired=1;var drag=null;
  var move=function(e){if(!drag)return;var t=mTime(e.clientX);
    if(drag==='in')setMIn(t);else if(drag==='out')setMOut(t);else{$('mvid').currentTime=t;mHead();}};
  var up=function(){drag=null;document.removeEventListener('pointermove',move);document.removeEventListener('pointerup',up);};
  var down=function(e,which){drag=which;e.preventDefault();
    document.addEventListener('pointermove',move);document.addEventListener('pointerup',up);
    if(which==='seek'){$('mvid').currentTime=mTime(e.clientX);mHead();}};
  $('mhin').addEventListener('pointerdown',function(e){e.stopPropagation();down(e,'in');});
  $('mhout').addEventListener('pointerdown',function(e){e.stopPropagation();down(e,'out');});
  tk.addEventListener('pointerdown',function(e){down(e,'seek');});
  $('min').addEventListener('input',function(){var t=parseFloat(this.value);if(!isNaN(t))setMIn(t);});
  $('mout').addEventListener('input',function(){var t=parseFloat(this.value);if(!isNaN(t))setMOut(t);});}
function openScrub(stem,inp,outp){var v=$('mvid');mWire();
  SC={inp:inp,outp:outp,mIn:parseFloat(inp.value)||0,mOut:parseFloat(outp.value)||0,D:0};
  v.src='/api/source/'+stem;$('mplay').textContent='▶ Play';
  $('mhint').textContent='Loading full clip… (first open transcodes a proxy, may take a moment)';
  $('modal').style.display='flex';mRender();
  v.onloadeddata=function(){SC.D=v.duration||0;
    if(SC.mOut<=0||SC.mOut>SC.D)SC.mOut=SC.D;mRender();mHead();
    $('mhint').textContent='Drag the blue IN/OUT handles (or the band) on the timeline, type exact seconds in the in/out boxes, or use ⇤/⇥ to set from the playhead. Times are source seconds.';};
  v.ontimeupdate=mHead;}
function setInOut(which){var v=$('mvid');if(!SC||!v.duration)return;var t=+v.currentTime.toFixed(1);
  if(which==='in')setMIn(t);else setMOut(t);}
function closeModal(){var v=$('mvid');v.pause();v.removeAttribute('src');v.load();$('modal').style.display='none';SC=null;}
async function loadMusic(){try{MUSIC=(await (await fetch('/api/music')).json()).tracks||[];}catch(e){MUSIC=[];}}
function musicOptions(){let h='<option value="">no music</option>';const by={};
  MUSIC.forEach(t=>{(by[t.dir]=by[t.dir]||[]).push(t);});
  Object.keys(by).sort().forEach(d=>{h+='<optgroup label="'+(d==='.'?'(root)':d)+'">';
    by[d].forEach(t=>{h+='<option value="'+t.path.replace(/"/g,'&quot;')+'">'+t.name+'</option>';});h+='</optgroup>';});
  return h;}
function playMusic(path,btn){const a=$('aud');if(!path)return;
  if(a._btn===btn&&!a.paused){a.pause();btn.textContent='▶';return;}
  if(a._btn&&a._btn!==btn)a._btn.textContent='▶';
  a.src='/api/music_file/'+path.split('/').map(encodeURIComponent).join('/');
  a._btn=btn;btn.textContent='⏸';a.play().catch(()=>{btn.textContent='▶';});a.onended=()=>{btn.textContent='▶';};}
function cancelRender(){if(curJob){fetch('/api/render_cancel/'+curJob,{method:'POST'});var p=$('ptext');if(p)p.textContent='Cancelling…';}}
function toast(m,keep){const t=document.getElementById('toast');t.textContent=m;t.style.display='block';
  if(!keep){clearTimeout(t._t);t._t=setTimeout(()=>t.style.display='none',2600)}}
function updateN(){const n=sel.size;$('n').textContent=n;$('renderBtn').disabled=n===0;
  if($('ne'))$('ne').textContent=n;if($('exportBtn'))$('exportBtn').disabled=n===0}
async function reload(){
  const r=await fetch('/api/candidates');const d=await r.json();
  const root=document.getElementById('root');root.innerHTML='';let total=0;
  for(const g of d.groups){
    total+=g.segments.length;
    const h=document.createElement('div');h.className='group-title';
    h.textContent=`${g.video}  ·  ${g.segments.length} segments  ·  ${g.fps}fps${g.hdr?' · HDR':''}`;
    root.appendChild(h);
    const grid=document.createElement('div');grid.className='grid';root.appendChild(grid);
    for(const s of g.segments) grid.appendChild(card(g,s));
  }
  document.getElementById('count').textContent=`${total} candidates`;
  sel.clear();updateN();
}
function card(g,s){
  const key=g.stem+'#'+s.idx;
  const fid='f'+key.replace(/[^a-z0-9]/gi,'_');
  const adj={brightness:0,contrast:1,saturation:1,sharpen:0,warmth:0};
  cardAdj.set(key,adj);ensureFilter(fid);
  const el=document.createElement('div');el.className='card';
  const sc=(s.publish_score??0).toFixed(0);
  const common=s.common_lbj?'<span class="tag common">common</span>':'';
  const sp=s.species||'bird';
  const slider=(k,t,mn,mx,st,v)=>`<label><i>${t}</i><input data-k="${k}" type="range" min="${mn}" max="${mx}" step="${st}" value="${v}"><span>${(+v).toFixed(2)}</span></label>`;
  el.innerHTML=`
   <div class="media" title="click to preview">
     <img src="${s.thumb}" loading="lazy">
     <div class="play">▶</div>
     <div class="score">${sc}</div>
     ${s.species?`<span class="tag">${sp}</span>`:''}${common}
   </div>
   <div class="body">
     <div class="row"><span class="sp">${sp}</span>
       <input class="pick" type="checkbox"></div>
     <div class="meta" title="${clean(s.behavior||s.reason||'',600).replace(/"/g,'&quot;')}">${s.beauty?('★'+s.beauty+' · '):''}${clean(s.behavior||s.reason||'')}</div>
     <div class="trim">in<input type="number" class="in" value="${(s.clip_in!=null?s.clip_in:s.start).toFixed(1)}" step="0.5">
       out<input type="number" class="out" value="${(s.clip_out!=null?s.clip_out:s.end).toFixed(1)}" step="0.5">
       <span class="dur">${((s.clip_out!=null?s.clip_out:s.end)-(s.clip_in!=null?s.clip_in:s.start)).toFixed(1)}s</span>
       <span class="qmark" title="Clip in/out = start/end second in the SOURCE video. Default is a ~14s window around the highlight (sharpest frame). Detected bird range here: ${s.start.toFixed(1)}-${s.end.toFixed(1)}s. Final length ≈ (out-in) × 2 because of 0.5× slow-mo.">?</span><button class="srcbtn ghost" type="button" title="open the full source clip and set in/out by scrubbing">↔ source</button></div>
     <button class="adjtoggle ghost" type="button">🎚 Adjust</button>
     <div class="adj">
       ${slider('brightness','Bright',-0.3,0.3,0.01,0)}
       ${slider('contrast','Contrast',0.5,2,0.01,1)}
       ${slider('saturation','Saturation',0,2,0.01,1)}
       ${slider('sharpen','Sharpen',0,3,0.05,0)}
       ${slider('warmth','Warmth',-0.3,0.3,0.01,0)}
       <button class="adjreset ghost" type="button" style="font-size:11px;padding:3px 8px;align-self:flex-start">Reset</button>
     </div>
     <div class="aud"><select class="musicsel">${musicOptions()}</select><button class="mprev ghost" type="button" title="preview music">▶</button><button class="mrand ghost" type="button" title="random track">🎲</button></div>
   </div>`;
  const media=el.querySelector('.media');
  media.onclick=()=>{
    const ex=media.querySelector('video');
    if(ex){ex.paused?ex.play():ex.pause();return;}        // click toggles play/pause
    const v=document.createElement('video');
    v.src=s.preview;v.poster=s.thumb;v.loop=true;v.muted=true;v.autoplay=true;
    v.playsInline=true;v.preload='auto';v.style.filter=filterStr(adj,fid);   // no controls -> no hover dim
    media.innerHTML='';media.appendChild(v);
    const seek=document.createElement('input');seek.type='range';seek.min=0;seek.max=1000;seek.value=0;seek.className='seek';
    media.appendChild(seek);
    const tlabel=document.createElement('div');tlabel.className='time';media.appendChild(tlabel);
    const showTime=()=>{tlabel.textContent=(v.currentTime||0).toFixed(1)+' / '+(v.duration||0).toFixed(1)+'s';};
    v.addEventListener('loadedmetadata',showTime);
    v.addEventListener('timeupdate',()=>{if(!seek._drag&&v.duration)seek.value=v.currentTime/v.duration*1000;showTime();});
    seek.addEventListener('input',()=>{if(v.duration)v.currentTime=seek.value/1000*v.duration;showTime();});
    seek.addEventListener('pointerdown',e=>{e.stopPropagation();seek._drag=true;});
    seek.addEventListener('pointerup',()=>{seek._drag=false;});
    seek.addEventListener('change',()=>{seek._drag=false;});
    seek.addEventListener('click',e=>e.stopPropagation());
    const tip=document.createElement('div');tip.className='play';
    tip.style.fontSize='15px';tip.textContent='⏳ Generating…';media.appendChild(tip);
    v.addEventListener('loadeddata',()=>{tip.remove();v.play().catch(()=>{});});
    v.addEventListener('error',()=>{tip.textContent='⚠️ Preview failed';});
  };
  const panel=el.querySelector('.adj');
  const ranges=[...panel.querySelectorAll('input[type=range]')];
  const applyAdj=()=>{updateFilter(fid,adj);
    ranges.forEach(r=>{r.nextElementSibling.textContent=(+r.value).toFixed(2);});
    const v=media.querySelector('video');if(v)v.style.filter=filterStr(adj,fid);};
  ranges.forEach(r=>{r.oninput=()=>{adj[r.dataset.k]=+r.value;applyAdj();};});
  el.querySelector('.adjtoggle').onclick=()=>{panel.style.display=panel.style.display==='flex'?'none':'flex';};
  el.querySelector('.adjreset').onclick=()=>{Object.assign(adj,{brightness:0,contrast:1,saturation:1,sharpen:0,warmth:0});
    ranges.forEach(r=>{r.value=adj[r.dataset.k];});applyAdj();};
  const musicsel=el.querySelector('.musicsel'),mprev=el.querySelector('.mprev');
  mprev.onclick=()=>playMusic(musicsel.value,mprev);
  el.querySelector('.mrand').onclick=()=>{const o=[...musicsel.options].filter(x=>x.value);
    if(o.length)musicsel.value=o[Math.floor(Math.random()*o.length)].value;};
  const cb=el.querySelector('.pick'),inp=el.querySelector('.in'),outp=el.querySelector('.out');
  const dur=el.querySelector('.dur');
  const upd=()=>dur.textContent=(outp.value-inp.value).toFixed(1)+'s';
  inp.oninput=upd;outp.oninput=upd;
  el.querySelector('.srcbtn').onclick=()=>openScrub(g.stem,inp,outp);
  cb.onchange=()=>{
    if(cb.checked){sel.set(key,{stem:g.stem,seg:s.idx,inp,outp,musicsel});el.classList.add('sel')}
    else{sel.delete(key);el.classList.remove('sel')}
    updateN();
  };
  return el;
}
async function renderSel(){
  const items=[...sel.values()].map(o=>({stem:o.stem,seg:o.seg,
    start:parseFloat(o.inp.value),end:parseFloat(o.outp.value),
    adjust:cardAdj.get(o.stem+'#'+o.seg),music:o.musicsel?o.musicsel.value:''}));
  if(!items.length)return;
  const body={items,zoom:parseFloat($('zoom').value),mute:$('mute').checked,
    aspect:$('aspect').value,caption:$('caption').checked,watermark:$('watermark').checked,
    codec:$('codec').value,quality:$('quality').value,
    music_vol:+$('musicvol').value,natural_vol:+$('natvol').value};
  $('renderBtn').disabled=true;
  const t=$('toast');t.style.display='block';
  t.innerHTML='<span id="ptext">Submitting…</span> <button class="ghost" style="padding:2px 9px;font-size:12px;margin-left:8px" onclick="cancelRender()">Cancel</button>';
  const j=await (await fetch('/api/render',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
  if(!j.job){t.textContent='Render failed';$('renderBtn').disabled=false;return;}
  curJob=j.job;
  const iv=setInterval(async()=>{
    let s;try{s=await (await fetch('/api/render_status/'+j.job)).json();}catch(e){return;}
    const overall=Math.round(((s.done+(s.frac||0))/Math.max(1,s.total))*100);
    if(!s.finished){var p=$('ptext');if(p)p.textContent='Rendering '+s.done+'/'+s.total+' · '+(s.cur||'')+' '+Math.round((s.frac||0)*100)+'% · total '+overall+'%';return;}
    clearInterval(iv);$('renderBtn').disabled=false;curJob=null;
    const ok=(s.results||[]).filter(x=>x.ok);
    const links=ok.map(x=>`<a href="${x.url}" target="_blank">${x.stem} #${x.seg}</a>`).join(' · ');
    t.innerHTML='Done '+ok.length+'/'+(s.results||[]).length+'. '+links;
  },500);
}
async function exportSel(){
  const items=[...sel.values()].map(o=>({stem:o.stem,seg:o.seg,
    start:parseFloat(o.inp.value),end:parseFloat(o.outp.value)}));
  if(!items.length)return;
  const r=await fetch('/api/export',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({items,zoom:parseFloat($('zoom').value),aspect:$('aspect').value})});
  const d=await r.json();
  document.getElementById('toast').innerHTML='FCPXML exported: '+
    d.results.map(x=>x.path.split('/').pop()+' ('+x.n+')').join(' · ')+
    ' — open in Final Cut Pro, then add a 50% retime.';
  document.getElementById('toast').style.display='block';
}
// ---- Add videos: upload + server-side pick + scan ----------------------------
let SRC=[];                                              // [{path,name,root,rel,size,mtime,scanned}]
const srcSel=new Set();                                  // selected source paths
let curScan=null;
const fmtBytes=b=>{if(!b)return '0 B';const u=['B','KB','MB','GB','TB'];const i=Math.min(u.length-1,Math.floor(Math.log(b)/Math.log(1024)));return (b/Math.pow(1024,i)).toFixed(i?1:0)+' '+u[i];};
const fmtDate=ts=>{const d=new Date(ts*1000),p=n=>String(n).padStart(2,'0');return d.getFullYear()+'-'+p(d.getMonth()+1)+'-'+p(d.getDate())+' '+p(d.getHours())+':'+p(d.getMinutes());};
function openAdd(){$('addmodal').style.display='flex';loadSources();}
function closeAdd(){$('addmodal').style.display='none';}
function updateNs(){const n=srcSel.size;$('ns').textContent=n;$('scanBtn').disabled=n===0||!!curScan;}
function selectAllSrc(on){srcSel.clear();if(on)SRC.forEach(s=>srcSel.add(s.path));renderSources();}
async function loadSources(){
  const hint=$('srchint');hint.textContent='Loading…';
  let d;try{d=await (await fetch('/api/sources')).json();}catch(e){hint.textContent='Failed to list files';return;}
  SRC=d.sources||[];
  [...srcSel].forEach(p=>{if(!SRC.some(s=>s.path===p))srcSel.delete(p);});   // drop vanished
  const roots=(d.roots||[]).map(r=>r.label+': '+r.path+(r.exists?'':' (missing)')).join('   ·   ');
  hint.textContent=(SRC.length?SRC.length+' video file(s)':'No videos found')+'   ·   roots — '+roots;
  renderSources();
}
function renderSources(){
  const box=$('srclist');box.innerHTML='';
  for(const s of SRC){
    const row=document.createElement('label');row.className='srcrow'+(srcSel.has(s.path)?' on':'');
    row.innerHTML=`<input type="checkbox" ${srcSel.has(s.path)?'checked':''}>
      <span class="srcname">${s.name}</span>
      <span class="srcmeta">${fmtBytes(s.size)} · ${fmtDate(s.mtime)} · ${s.root}${s.scanned?' · <span class="scok">scanned</span>':''}</span>`;
    const cb=row.querySelector('input');
    cb.onchange=()=>{if(cb.checked)srcSel.add(s.path);else srcSel.delete(s.path);row.classList.toggle('on',cb.checked);updateNs();};
    box.appendChild(row);
  }
  updateNs();
}
async function uploadFiles(files){
  if(!files||!files.length)return;
  const fd=new FormData();[...files].forEach(f=>fd.append('files',f));
  const msg=$('uplmsg');msg.textContent='Uploading '+files.length+' file(s)…';
  try{
    const d=await (await fetch('/api/upload',{method:'POST',body:fd})).json();
    const saved=d.saved||[];msg.textContent='Uploaded: '+saved.map(s=>s.name).join(', ');
    await loadSources();
    saved.forEach(s=>srcSel.add(s.path));                // auto-select freshly uploaded
    renderSources();
  }catch(e){msg.textContent='Upload failed';}
  $('uplinput').value='';
}
async function scanSelected(){
  const paths=[...srcSel];if(!paths.length||curScan)return;
  const j=await (await fetch('/api/scan',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({paths})})).json();
  if(!j.job){$('scanprog').textContent=j.error||'Scan failed';return;}
  curScan=j.job;updateNs();$('scanCancelBtn').style.display='';
  const iv=setInterval(async()=>{
    let s;try{s=await (await fetch('/api/scan_status/'+j.job)).json();}catch(e){return;}
    if(!s.finished){
      const cur=Math.min(s.n_files||1,(s.file_idx||0)+1);
      $('scanprog').textContent=`Scanning ${cur}/${s.n_files} · ${s.cur_file||''} · ${s.stage||''} ${Math.round((s.frac||0)*100)}%`;
      return;
    }
    clearInterval(iv);curScan=null;$('scanCancelBtn').style.display='none';
    const summary=(s.results||[]).map(r=>`${r.file}: ${r.segments} seg`).join(' · ');
    $('scanprog').textContent=(s.error?'Error: '+s.error:'Done. '+summary)+(s.warning?' — '+s.warning:'');
    await loadSources();updateNs();
    await reload();                                      // new candidates flow into the main grid
  },700);
}
function cancelScan(){if(curScan){fetch('/api/scan_cancel/'+curScan,{method:'POST'});$('scanprog').textContent='Cancelling…';}}

(async () => { applyThemeLabel(); syncAspect(); await loadMusic(); reload(); })();
