const FRAME_INTERVAL_MS = 1000 / 15;
const MAX_FRAME_SIZE = 640;
const JPEG_QUALITY = 0.8;
const $ = id => document.getElementById(id),
  video = $('camera'),
  capture = document.createElement('canvas');
const sender = crypto.randomUUID();
let stream = null,
  running = false,
  token = '',
  roi = .78,
  abort = null,
  runId = 0;
let wakeLock = null;
let wholeFrame = false;
const lightSample = document.createElement('canvas');
lightSample.width = 32;
lightSample.height = 32;
const lightContext = lightSample.getContext('2d', {
  willReadFrequently: true
});
let sentFrames = 0,
  rateStart = 0,
  sendFps = 0;
async function keepScreenOn(id) {
  if (!navigator.wakeLock || document.visibilityState !== 'visible') return;
  try {
    const lock = await navigator.wakeLock.request('screen');
    if (id !== runId || !running) {
      await lock.release();
      return;
    }
    wakeLock = lock;
  } catch (_) {
    /* Screen lock support is optional. */ }
}

function guide() {
  const c = $('guide');
  c.width = c.clientWidth;
  c.height = c.clientHeight;
  const ctx = c.getContext('2d');
  if (wholeFrame) return;
  if (!video.videoWidth) return;
  const scale = Math.min(c.width / video.videoWidth, c.height / video.videoHeight),
    w = video.videoWidth * scale,
    h = video.videoHeight * scale;
  ctx.strokeStyle = '#b7edcc';
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 7]);
  ctx.strokeRect((c.width - w * roi) / 2, (c.height - h * roi) / 2, w * roi, h * roi);
}

function stop() {
  running = false;
  runId++;
  abort?.abort();
  wakeLock?.release().catch(() => {});
  wakeLock = null;
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
  video.srcObject = null;
  $('start').disabled = false;
  $('stop').disabled = true;
  $('status').textContent = '';
}
async function sendFrame(id) {
  if (!running || id !== runId) return;
  const tick = performance.now();
  try {
    if (document.hidden || video.readyState < 2 || !video.videoWidth || video.paused || stream?.getVideoTracks()[0]?.muted) {
      $('status').textContent = 'Čekam sliku kamere. Ostavite ovu stranicu otvorenu i telefon otključan.';
      setTimeout(() => sendFrame(id), 250);
      return;
    }
    const scale = Math.min(1, MAX_FRAME_SIZE / Math.max(video.videoWidth, video.videoHeight));
    capture.width = Math.round(video.videoWidth * scale);
    capture.height = Math.round(video.videoHeight * scale);
    capture.getContext('2d').drawImage(video, 0, 0, capture.width, capture.height);
    lightContext.drawImage(video, 0, 0, 32, 32);
    const pixels = lightContext.getImageData(0, 0, 32, 32).data;
    let light = 0,
      count = 0;
    for (let i = 0; i < pixels.length; i += 64) {
      light += Math.max(pixels[i], pixels[i + 1], pixels[i + 2]);
      count++;
    }
    const dark = light / count < 12;
    const encodingStarted = performance.now();
    const blob = await new Promise(resolve => capture.toBlob(resolve, 'image/jpeg', JPEG_QUALITY));
    const encodedAt = performance.now();
    if (!running || id !== runId) return;
    abort = new AbortController();
    const timeout = setTimeout(() => abort.abort(), 5000);
    let response;
    try {
      response = await fetch('/api/frame', {
        method: 'POST',
        body: blob,
        signal: abort.signal,
        headers: {
          'Content-Type': 'image/jpeg',
          'X-SIR2-Token': token,
          'X-SIR2-Sender': sender,
          'X-SIR2-Source': 'phone'
        }
      });
    } finally {
      clearTimeout(timeout);
    }
    if (!response.ok) {
      const body = await response.json();
      throw Error(body.detail || 'Prenos nije uspeo.');
    }
    const data = await response.json();
    const acknowledgedAt = performance.now();
    SIR2Timing.report(token, {
      kind: 'upload', source: 'phone', sequence: data.sequence,
      capture_ms: encodingStarted - tick,
      jpeg_encode_ms: encodedAt - encodingStarted,
      upload_round_trip_ms: acknowledgedAt - encodedAt,
      capture_to_ack_ms: acknowledgedAt - tick
    });
    if (!wholeFrame && roi !== data.settings.roi) {
      roi = data.settings.roi;
      guide();
    }
    sentFrames++;
    const now = performance.now();
    if (now - rateStart >= 1500) {
      sendFps = sentFrames * 1000 / (now - rateStart);
      sentFrames = 0;
      rateStart = now;
    }
    if (running && id === runId) $('status').textContent = dark ? 'Kamera šalje veoma tamnu sliku. Otkrijte zadnja sočiva i usmerite telefon ka osvetljenom predmetu.' : `Prenos ${sendFps.toFixed(1)} FPS · slanje ${(performance.now()-tick).toFixed(0)} ms`;
  } catch (e) {
    if (running && id === runId) {
      stop();
      $('status').textContent = `Prenos je prekinut: ${e.message}. Proverite USB i server, pa ponovo pokrenite kameru.`;
    }
    return;
  }
  if (running && id === runId) setTimeout(() => sendFrame(id), Math.max(0, FRAME_INTERVAL_MS - (performance.now() - tick)));
}
$('start').onclick = async () => {
  const id = ++runId;
  $('start').disabled = true;
  $('stop').disabled = false;
  $('status').textContent = 'Povezivanje…';
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw Error('Otvorite http://localhost:8000/phone preko USB veze u Chrome-u.');
    const response = await fetch('/api/state');
    if (!response.ok) throw Error('Server nije dostupan.');
    const state = await response.json();
    SIR2Timing.enabled = state.timing_log?.enabled === true;
    token = state.token;
    wholeFrame = state.model.input_scope === 'whole_frame' || ['species_router_bioclip2_linear', 'species_router_bioclip2_convnext'].includes(state.model.task);
    roi = state.settings.roi;
    if (id !== runId) return;
    const acquired = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: {
          ideal: 'environment'
        },
        width: {
          ideal: 1280
        },
        height: {
          ideal: 720
        }
      }
    });
    if (id !== runId) {
      acquired.getTracks().forEach(t => t.stop());
      return;
    }
    stream = acquired;
    video.srcObject = stream;
    await video.play();
    if (id !== runId) return;
    stream.getVideoTracks()[0].addEventListener('ended', () => {
      if (id === runId) {
        stop();
        $('status').textContent = 'Kamera je prekinuta. Ponovo pritisnite Pokreni kameru.';
      }
    });
    sentFrames = 0;
    rateStart = performance.now();
    sendFps = 0;
    running = true;
    $('stop').disabled = false;
    guide();
    keepScreenOn(id);
    sendFrame(id);
  } catch (e) {
    if (id === runId) {
      stop();
      $('status').textContent = `Kamera nije pokrenuta: ${e.message}`;
    }
  }
};
$('stop').onclick = stop;
window.addEventListener('pagehide', stop);
window.addEventListener('resize', guide);
document.addEventListener('visibilitychange', () => {
  if (running && document.visibilityState === 'visible') keepScreenOn(runId);
});

