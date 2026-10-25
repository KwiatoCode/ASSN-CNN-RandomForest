import { loadForest, predictForest, featureVector } from './forest.js';

const $ = (id) => document.getElementById(id);
const camera = $('camera');
const canvas = $('skeleton');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
const MP = '1.1.1';
const MODEL_URL = new URL('./models/forest.rf.gz', import.meta.url);
let forest = null, detector = null, stream = null;
let running = false, raf = 0, lastAt = 0, lastTS = -1, recent = [], lastLandmarks = null;
let viewMode = 'normalized';

const setStatus = (message, error = false) => {
  $('status').textContent = message;
  $('status').dataset.error = error ? 'true' : 'false';
};
function resetPrediction() {
  $('letter').textContent = '—';
  $('confidence').textContent = 'Czekam na dłoń';
  $('top').replaceChildren();
  recent = [];
}
function paintWhite() { ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, 400, 400); }
paintWhite();

function renderSkeleton(landmarks) {
  paintWhite();
  if (!landmarks || landmarks.length !== 21) return;
  const mirrored = $('mirror').checked;
  let points = landmarks.map(p => ({ x: mirrored ? 1 - p.x : p.x, y: p.y }));
  if (viewMode === 'normalized') {
    const w = camera.videoWidth || 640;
    const h = camera.videoHeight || 480;
    const px = points.map(p => ({ x: p.x * w, y: p.y * h }));
    const palm = Math.max(1, Math.hypot(px[9].x - px[0].x, px[9].y - px[0].y));
    const scaleY = 130 / palm;
    const scaleX = scaleY * .75;
    const xs = px.map(p => p.x), ys = px.map(p => p.y);
    const centerX = (Math.min(...xs) + Math.max(...xs)) / 2;
    const centerY = (Math.min(...ys) + Math.max(...ys)) / 2;
    points = px.map(p => ({ x: 188 + (p.x - centerX) * scaleX, y: 209 + (p.y - centerY) * scaleY }));
  } else {
    points = points.map(p => ({ x: p.x * 400, y: p.y * 400 }));
  }
  points = points.map(p => ({ x: Math.floor(p.x) + .5, y: Math.floor(p.y) + .5 }));
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#ff0000';
  ctx.lineCap = 'round';
  ctx.beginPath();
  for (const [a,b] of CONNECTIONS) { ctx.moveTo(points[a].x,points[a].y);ctx.lineTo(points[b].x,points[b].y); }
  ctx.stroke();
  for (const p of points) {
    ctx.fillStyle = '#fff'; ctx.beginPath();ctx.arc(p.x,p.y,4,0,Math.PI*2);ctx.fill();
    ctx.fillStyle = '#00ff00';ctx.beginPath();ctx.arc(p.x,p.y,2.8,0,Math.PI*2);ctx.fill();
  }
  ctx.restore();
}

function showScores(scores) {
  if ($('smooth').checked) {
    recent.push(scores);
    if (recent.length > 4) recent.shift();
    scores = scores.map((_, i) => recent.reduce((total,row) => total+row[i],0)/recent.length);
  } else { recent = []; }
  const best = [...scores].map((probability, index) => ({label:String.fromCharCode(65+index),probability}))
    .sort((a,b) => b.probability-a.probability).slice(0,3);
  const threshold = Number($('threshold').value)/100;
  $('letter').textContent = best[0].probability >= threshold ? best[0].label : '?';
  $('confidence').textContent = `Głosy lasu: ${(best[0].probability*100).toFixed(1)}% · ${forest.treeCount} drzew`;
  $('top').replaceChildren(...best.map(item => {
    const line = document.createElement('div');line.className='score';
    const name = document.createElement('strong');name.textContent=item.label;
    const pct = document.createElement('span');pct.textContent=(item.probability*100).toFixed(1)+'%';
    line.append(name,pct);
    return line;
  }));
}

async function initialize() {
  setStatus('Wczytuję Twój Random Forest (300 drzew)…');
  if (!forest) forest = await loadForest(MODEL_URL);
  setStatus('Pobieram detektor dłoni MediaPipe…');
  if (!detector) {
    const { FilesetResolver, HandLandmarker } = await import(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP}/vision_bundle.mjs`);
    const wasm = await FilesetResolver.forVisionTasks(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP}/wasm`);
    detector = await HandLandmarker.createFromOptions(wasm, {
      baseOptions: {modelAssetPath:'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',delegate:'CPU'},
      runningMode:'VIDEO',numHands:1,
      minHandDetectionConfidence:.5,minHandPresenceConfidence:.45,minTrackingConfidence:.5
    });
  }
}

async function start() {
  $('start').disabled = true;
  try {
    if (!window.isSecureContext && location.hostname !== 'localhost') throw new Error('Otwórz witrynę przez HTTPS, nie przez plik ZIP ani file://.');
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Brak obsługi kamery — użyj Safari na iPhonie.');
    await initialize();
    setStatus('Proszę o zgodę na kamerę…');
    stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:$('facing').value,width:{ideal:640},height:{ideal:480}},audio:false});
    camera.srcObject = stream;
    await camera.play();
    running = true;lastAt=0;lastTS=-1;
    $('start').hidden=true; $('stop').hidden=false;
    $('save').disabled=false;
    setStatus('Gotowe: pokaż literę dłonią');
    raf = requestAnimationFrame(frame);
  } catch(e) {
    setStatus('Nie udało się uruchomić: '+(e.message||e),true);
    $('start').disabled=false;
    stopStream();
  }
}

async function frame(now) {
  if (!running) return;
  raf = requestAnimationFrame(frame);
  if (now-lastAt<220 || camera.readyState<2) return;
  lastAt=now;
  try {
    const timestamp = Math.max(performance.now(),lastTS+1);lastTS=timestamp;
    const result=detector.detectForVideo(camera,timestamp);
    const points=result.landmarks?.[0];
    if (!points) {
      $('detection').textContent='Nie wykryto dłoni';
      lastLandmarks=null;recent=[];paintWhite();resetPrediction();return;
    }
    lastLandmarks=points;
    $('detection').textContent='Wykryto 21 punktów';
    renderSkeleton(points);
    const features=featureVector(ctx);
    const scores=predictForest(forest,features);
    showScores(scores);
  } catch(e) {stop();setStatus('Błąd analizy: '+(e.message||e),true);}
}
function stopStream() {
  if (stream) {stream.getTracks().forEach(t=>t.stop());stream=null;}
  camera.srcObject=null;
}
function stop() {
  running=false; cancelAnimationFrame(raf);stopStream();
  $('start').hidden=false;$('start').disabled=false;$('stop').hidden=true;
  $('detection').textContent='Kamera wyłączona';
  resetPrediction();
  if ($('status').dataset.error!=='true')setStatus('Zatrzymano kamerę');
}
$('start').addEventListener('click',start);
$('stop').addEventListener('click',stop);
$('facing').addEventListener('change',()=>{if(running){stop();start();}});
$('framing').addEventListener('change',()=>{viewMode=$('framing').value;recent=[];});
$('mirror').addEventListener('change',()=>{recent=[];});
$('smooth').addEventListener('change',()=>{recent=[];});
$('threshold').addEventListener('input',()=>{$('thresholdLabel').textContent=$('threshold').value+'%';});
$('save').addEventListener('click',()=>{
  if (!lastLandmarks) {setStatus('Najpierw pokaż dłoń w kadrze');return;}
  const a=document.createElement('a');a.href=canvas.toDataURL('image/png');a.download=`ASSN_szkielet_${Date.now()}.png`;a.click();
});
window.addEventListener('pagehide',stop);
