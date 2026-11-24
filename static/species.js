const $=id=>document.getElementById(id);
const sender=crypto.randomUUID();
let token='',stream=null,running=false,resultVersion=-1,previewUrl=null,polling=false,serverError=false;
const canvas=document.createElement('canvas');
const ms=v=>Number.isFinite(v)?`${v.toFixed(1)} ms`:'—';
const say=(text,error=false)=>{$('message').textContent=error?text:'';$('message').hidden=!error;};
async function api(url,options={}){
  const response=await fetch(url,{...options,headers:{'X-SIR2-Token':token,...options.headers}});
  if(!response.ok){let body;try{body=await response.json();}catch{}throw Error(body?.detail||'Server nije dostupan.');}
  return response.json();
}
function stopCamera(){running=false;if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;$('camera').srcObject=null;$('camera').hidden=true;$('startCamera').hidden=false;$('stopCamera').hidden=true;resultVersion=-1;}
async function send(blob, source, started, encodingStarted, encodedAt){
  const uploadStarted=performance.now();
  const result=await api('/api/frame',{method:'POST',body:blob,headers:{'Content-Type':'image/jpeg','X-SIR2-Sender':sender,'X-SIR2-Source':source}});
  const acknowledgedAt=performance.now();
  SIR2Timing.report(token,{kind:'upload',source,sequence:result.sequence,
    capture_ms:encodingStarted-started,jpeg_encode_ms:encodedAt-encodingStarted,
    upload_round_trip_ms:acknowledgedAt-uploadStarted,capture_to_ack_ms:acknowledgedAt-started});
  return result;
}
async function photo(file){
  if(!file)return;
  stopCamera();say('Priprema fotografije…');
  try{
    const started=performance.now();
    const image=await createImageBitmap(file);const scale=Math.min(1,1280/Math.max(image.width,image.height));
    canvas.width=Math.round(image.width*scale);canvas.height=Math.round(image.height*scale);
    if(Math.min(canvas.width,canvas.height)<64)throw Error('Fotografija je premala. Najkraća stranica treba da ima bar 64 piksela.');
    canvas.getContext('2d').drawImage(image,0,0,canvas.width,canvas.height);image.close();
    const encodingStarted=performance.now();
    const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',.9));
    await send(blob,'photo',started,encodingStarted,performance.now());say('Fotografija poslata. Sačekaj rezultat…');
  }catch(error){say(error.message,true);}
}
$('file').onchange=event=>photo(event.target.files[0]);
for(const type of ['dragenter','dragover'])$('dropzone').addEventListener(type,event=>{event.preventDefault();$('dropzone').classList.add('drag');});
for(const type of ['dragleave','drop'])$('dropzone').addEventListener(type,event=>{event.preventDefault();$('dropzone').classList.remove('drag');if(type==='drop')photo(event.dataTransfer.files[0]);});
$('startCamera').onclick=async()=>{
  try{
    stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false});
    $('camera').srcObject=stream;await $('camera').play();running=true;$('camera').hidden=false;$('preview').hidden=true;$('empty').hidden=true;
    $('startCamera').hidden=true;$('stopCamera').hidden=false;say('Kamera je uključena. Usmeri je ka jednom listu.');
    capture();
  }catch(error){stopCamera();say(`Kamera nije dostupna: ${error.message}`,true);}
};
$('stopCamera').onclick=()=>{stopCamera();say('Kamera je zaustavljena. Poslednji rezultat ostaje prikazan.');};
async function capture(){
  if(!running)return;
  try{
    const started=performance.now(),video=$('camera');
    const scale=Math.min(1,640/Math.max(video.videoWidth,video.videoHeight));
    canvas.width=Math.round(video.videoWidth*scale);canvas.height=Math.round(video.videoHeight*scale);
    canvas.getContext('2d').drawImage(video,0,0,canvas.width,canvas.height);
    const encodingStarted=performance.now();
    const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',.85));
    if(running)await send(blob,'desktop',started,encodingStarted,performance.now());
  }catch(error){stopCamera();say(error.message,true);}
  if(running)setTimeout(capture,125);
}
function render(result){
  const accepted=result.accepted===true;
  $('decision').className=`decision ${accepted?'':result.status==='LOW_LIGHT'?'neutral':'unknown'}`;
  $('decision').textContent=accepted?'Podržana biljka':result.status==='LOW_LIGHT'?'Previše taman kadar':'Nepoznato / nesigurno';
  $('species').textContent=accepted?result.species_label:result.status==='LOW_LIGHT'?'Nema jasne slike':'Odbijena slika';
  if(result.status==='NO_LEAF'){$('decision').className='decision neutral';$('decision').textContent='Nema lista';$('species').textContent='List nije pronađen';}
  const hasCondition=accepted&&result.disease_prediction===true&&!!result.condition_class;
  $('conditionResult').hidden=!hasCondition;
  $('condition').textContent=hasCondition?result.condition_label:'—';
  $('conditionResult').classList.toggle('healthy',hasCondition&&result.healthy===true);
  $('conditionConfidence').textContent=hasCondition&&Number.isFinite(result.condition_confidence)?`Ocena modela: ${(result.condition_confidence*100).toFixed(1)}%`:'';
  if(result.status==='ERROR'){$('decision').className='decision neutral';$('decision').textContent='Greška pri obradi';$('species').textContent='Pokušaj ponovo';}
  $('candidate').textContent=result.candidate_label||'—';$('confidence').textContent=Number.isFinite(result.confidence)?`${(result.confidence*100).toFixed(1)}%`:'—';
  $('rejectionMethod').textContent=result.method==='bioclip_taxonomy_and_reference_knn'?'Kombinovana provera':'—';
  $('referenceScore').textContent=Number.isFinite(result.reference_distance)?`${result.reference_distance.toFixed(4)} / ${result.reference_threshold.toFixed(4)}`:'—';
  $('score').textContent=Number.isFinite(result.score)?`${result.score.toFixed(4)} / ${result.threshold.toFixed(4)}`:'—';
  $('segMs').textContent=ms(result.timings?.segmentation_ms);$('total').textContent=ms(result.processing_ms);$('bioMs').textContent=ms(result.timings?.bioclip_ms);$('diseaseMs').textContent=ms(result.timings?.disease_ms);
}
async function poll(){
  if(polling)return;polling=true;
  try{
    const pollStarted=performance.now();
    const state=await api('/api/state');token=state.token;
    const statePolledAt=performance.now();
    SIR2Timing.enabled=state.timing_log?.enabled===true;
    if(serverError&&!state.error){say('');serverError=false;}
    $('activity').textContent=state.benchmarking?'Merenje brzine…':state.busy?'Analiziram…':state.result_version?'Analiza završena':'Čekam sliku';
    $('startCamera').disabled=state.benchmarking;
    $('fps').textContent=Number.isFinite(state.processed_fps)?`${state.processed_fps.toFixed(1)} FPS`:'—';
    if(state.error){serverError=true;say(state.error,true);render({accepted:false,status:'ERROR'});$('activity').textContent='Obrada nije uspela';}
    if(state.result_version!==resultVersion&&state.result_version>0&&!state.error){
      {
        const response=await fetch(`/api/image/overlay?result_version=${state.result_version}`);
        if(response.status===204||Number(response.headers.get('X-Result-Version'))!==state.result_version)return;
        if(!response.ok)throw Error('Obrađeni kadar nije dostupan.');
        const blob=await response.blob();const old=previewUrl;previewUrl=URL.createObjectURL(blob);
        const downloadedAt=performance.now();
        $('preview').src=previewUrl;$('camera').hidden=true;$('preview').hidden=false;$('empty').hidden=true;
        if($('preview').decode)await $('preview').decode().catch(()=>{});
        if(old)URL.revokeObjectURL(old);
        render(state.result);resultVersion=state.result_version;
        SIR2Timing.report(token,{kind:'display',source:'viewer',sequence:state.processed_sequence,result_version:state.result_version,
          state_poll_ms:statePolledAt-pollStarted,overlay_download_ms:downloadedAt-statePolledAt,
          display_update_ms:performance.now()-downloadedAt});
      }
    }
  }catch(error){serverError=true;say(`Veza sa serverom: ${error.message}`,true);}finally{polling=false;setTimeout(poll,250);}
}

const frame=$('analysisWorkspace');
function expanded(){return document.fullscreenElement===frame||frame.classList.contains('expanded');}
function fullscreenLabel(){
  const active=expanded();
  $('fullscreenText').textContent=active?'Izađi iz celog ekrana':'Ceo ekran';
  $('fullscreen').setAttribute('aria-pressed',String(active));
  $('fullscreen').setAttribute('aria-label',active?'Izađi iz prikaza preko celog ekrana':'Prikaži fotografiju i rezultat preko celog ekrana');
}
function closeExpanded(){frame.classList.remove('expanded');document.body.classList.remove('preview-expanded');fullscreenLabel();}
$('fullscreen').onclick=async()=>{
  if(frame.classList.contains('expanded')){closeExpanded();return;}
  if(document.fullscreenElement===frame){
    try{await document.exitFullscreen();}catch(error){say(error.message,true);}
    return;
  }
  try{
    if(!document.fullscreenEnabled||!frame.requestFullscreen)throw Error('Fullscreen unavailable');
    await frame.requestFullscreen();
  }catch{
    frame.classList.add('expanded');document.body.classList.add('preview-expanded');
  }
  fullscreenLabel();
};
document.addEventListener('fullscreenchange',fullscreenLabel);
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&frame.classList.contains('expanded'))closeExpanded();});

window.addEventListener('pagehide',()=>{stopCamera();if(previewUrl)URL.revokeObjectURL(previewUrl);});
poll();
