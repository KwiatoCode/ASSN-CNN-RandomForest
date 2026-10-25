const $ = id => document.getElementById(id);
const canvas = $('skeleton');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const video = $('camera');
const resized = document.createElement('canvas');
resized.width = 224;
resized.height = 224;
const resizedCtx = resized.getContext('2d', { willReadFrequently: true });
const CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
const BASE = new URL('.', import.meta.url);
const MEDIAPIPE_VERSION = '1.1.1';
const ORT_VERSION = '1.30.0';
let stream = null;
let landmarker = null;
let session = null;
let forest = null;
let running = false;
let previousTimestamp = -1;
let lastFrameAt = 0;
let history = [];
let rafId = 0;

function setStatus(message, mode = '') {
  $('status-text').textContent = message;
  $('status-dot').className = `status-dot ${mode}`;
}
function resetPredictions() {
  history = [];
  $('letter').textContent = '–';
  $('confidence').textContent = 'Brak predykcji';
  $('top3').replaceChildren();
}
function blankCanvas() { ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, 400, 400); }
blankCanvas();

function renderSkeleton(landmarks) {
  blankCanvas();
  if (!landmarks || landmarks.length !== 21) return false;
  const mirror = $('mirror').checked;
  let points = landmarks.map(p => ({ x: mirror ? 1 - p.x : p.x, y: p.y }));
  if ($('framing').value === 'normalized') {
    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;
    let px = points.map(p => ({ x: p.x * width, y: p.y * height }));
    const palm = Math.max(1, Math.hypot(px[9].x - px[0].x, px[9].y - px[0].y));
    const sx = (130 / palm) * (480 / 640);
    const sy = 130 / palm;
    const xs = px.map(p => p.x), ys = px.map(p => p.y);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    points = px.map(p => ({ x: 188 + (p.x - cx) * sx, y: 209 + (p.y - cy) * sy }));
  } else {
    points = points.map(p => ({ x: p.x * 400, y: p.y * 400 }));
  }
  points = points.map(p => ({ x: Math.floor(p.x) + .5, y: Math.floor(p.y) + .5 }));
  ctx.strokeStyle = 'rgb(255,0,0)';
  ctx.lineWidth = 2;
  ctx.lineCap = 'round';
  ctx.beginPath();
  for (const [a, b] of CONNECTIONS) { ctx.moveTo(points[a].x, points[a].y); ctx.lineTo(points[b].x, points[b].y); }
  ctx.stroke();
  for (const p of points) {
    ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(p.x,p.y,4,0,2*Math.PI); ctx.fill();
    ctx.fillStyle = 'rgb(0,255,0)'; ctx.beginPath(); ctx.arc(p.x,p.y,3,0,2*Math.PI); ctx.fill();
  }
  $('save').disabled = false;
  return true;
}

function pixelsToTensor() {
  resizedCtx.imageSmoothingEnabled = true;
  resizedCtx.imageSmoothingQuality = 'medium';
  resizedCtx.drawImage(canvas, 0, 0, 224, 224);
  const {data} = resizedCtx.getImageData(0,0,224,224);
  const area = 224 * 224;
  const values = new Float32Array(3 * area);
  for (let i=0,j=0;i<area;i++,j+=4) {
    values[i] = data[j] / 255;
    values[area+i] = data[j+1] / 255;
    values[2*area+i] = data[j+2] / 255;
  }
  return new window.ort.Tensor('float32',values,[1,3,224,224]);
}

function predictForest(embedding) {
  if (embedding.length !== forest.featureCount) throw new Error('Niezgodna liczba cech CNN i Random Forest');
  const sums = Array(forest.classes.length).fill(0);
  for (const tree of forest.trees) {
    let node = 0;
    while (tree.left[node] >= 0) {
      node = embedding[tree.feature[node]] <= tree.threshold[node] ? tree.left[node] : tree.right[node];
    }
    const probabilities = tree.probabilities[tree.leaf[node]];
    for (let c=0;c<sums.length;c++) sums[c]+=probabilities[c];
  }
  return sums.map(value => value / forest.trees.length);
}

function showPrediction(probabilities) {
  if ($('smooth').checked) {
    history.push(probabilities);
    if (history.length > 6) history.shift();
    probabilities = probabilities.map((_,i) => history.reduce((sum,row) => sum+row[i],0) / history.length);
  } else {
    history = [];
  }
  const top = probabilities.map((value,i) => ({label: forest.classes[i],value})).sort((a,b)=>b.value-a.value).slice(0,3);
  const threshold = Number($('threshold').value) / 100;
  $('letter').textContent = top[0].value >= threshold ? top[0].label : '–';
  $('confidence').textContent = top[0].value >= threshold ? `${(top[0].value*100).toFixed(1)}% pewności` : `Niepewne (${(top[0].value*100).toFixed(1)}%)`;
  $('top3').replaceChildren(...top.map(row => {
    const element = document.createElement('div'); element.className='topitem';
    const label = document.createElement('span');label.textContent=row.label;
    const percent = document.createElement('span');percent.textContent=`${(row.value*100).toFixed(1)}%`;
    element.append(label,percent); return element;
  }));
}

async function initializeModels() {
  setStatus('Pobieranie detektora dłoni MediaPipe…');
  const {FilesetResolver,HandLandmarker} = await import(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/vision_bundle.mjs`);
  const vision = await FilesetResolver.forVisionTasks(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`);
  landmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {modelAssetPath:'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', delegate:'CPU'},
    runningMode: 'VIDEO',numHands:1,minHandDetectionConfidence:.5,minHandPresenceConfidence:.5,minTrackingConfidence:.5
  });
  setStatus('Wczytywanie Twojej CNN i Random Forest…');
  if (!window.ort) throw new Error('Nie załadowano ONNX Runtime Web. Sprawdź połączenie z Internetem.');
  window.ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  window.ort.env.wasm.numThreads = 1;
  const modelURL = new URL('models/backbone.onnx', BASE);
  const forestURL = new URL('models/forest.json', BASE);
  const response = await fetch(forestURL);
  if (!response.ok) throw new Error(`Brak pliku Random Forest (${response.status}). Sprawdź publikację witryny.`);
  forest = await response.json();
  if (forest.featureCount !== 256 || forest.classes.length !== 26 || forest.trees.length !== 160) {
    throw new Error('Niewłaściwa wersja Random Forest — oczekiwano 160 drzew, 256 cech i 26 klas.');
  }
  session = await window.ort.InferenceSession.create(modelURL.toString(), {executionProviders:['wasm'],graphOptimizationLevel:'all'});
  if (!session.inputNames.length || !session.outputNames.length) throw new Error('Model ONNX nie zawiera wejścia lub wyjścia.');
}

async function runFrame(now) {
  if (!running) return;
  if (now - lastFrameAt < 360 || video.readyState < 2) {
    rafId = window.requestAnimationFrame(runFrame);
    return;
  }
  lastFrameAt = now;
  try {
    let timestamp = performance.now();
    if (timestamp <= previousTimestamp) timestamp = previousTimestamp + 1;
    previousTimestamp = timestamp;
    const t0 = performance.now();
    const result = landmarker.detectForVideo(video,timestamp);
    const landmarks = result.landmarks?.[0];
    if (!landmarks) {
      resetPredictions();blankCanvas();$('save').disabled=true;
      $('camera-badge').textContent = 'Nie wykryto dłoni';$('inference-speed').textContent='';
      return;
    }
    if (!renderSkeleton(landmarks)) return;
    const tensor = pixelsToTensor();
    const output = await session.run({[session.inputNames[0]]:tensor});
    const embedding = output[session.outputNames[0]]?.data;
    if (!embedding) throw new Error('Brak embeddingu CNN');
    const probabilities = predictForest(embedding);
    showPrediction(probabilities);
    $('camera-badge').textContent = '21 punktów dłoni ✓';
    $('inference-speed').textContent = `${Math.round(performance.now()-t0)} ms`;
  } catch(error) {
    console.error(error);
    stop();
    setStatus(`Błąd rozpoznawania: ${error.message}`, 'error');
  } finally {
    if (running) rafId = window.requestAnimationFrame(runFrame);
  }
}

async function start() {
  $('start').disabled=true;
  setStatus('Inicjalizacja…');
  try {
    if (!window.isSecureContext && location.hostname !== 'localhost') throw new Error('Kamera wymaga HTTPS. Otwórz witrynę przez https://.');
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Brak dostępu do kamery w tej przeglądarce. Użyj Safari.');
    if (!landmarker || !session || !forest) await initializeModels();
    setStatus('Prośba o zgodę na kamerę…');
    stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:{ideal:640},height:{ideal:480}},audio:false});
    video.srcObject=stream;
    await video.play();
    running=true;lastFrameAt=0;previousTimestamp=-1;
    $('camera-empty').hidden=true;$('start').disabled=true;$('stop').disabled=false;
    setStatus('Model aktywny — pokaż literę dłonią', 'working');
    $('camera-badge').textContent='Szukam dłoni…';
    runFrame(performance.now());
  } catch(error) {
    console.error(error);
    setStatus(error.message || String(error), 'error');
    $('start').disabled=false;
    if (stream) { for(const track of stream.getTracks())track.stop();stream=null; }
  }
}
function stop() {
  running = false;
  window.cancelAnimationFrame(rafId);
  if(stream){for (const track of stream.getTracks())track.stop();stream=null;}
  video.srcObject=null;
  $('camera-empty').hidden=false;
  $('camera-badge').textContent='Kamera wyłączona';
  $('start').disabled=false;$('stop').disabled=true;
  $('inference-speed').textContent='';
  resetPredictions();
  if(!$('status-dot').classList.contains('error'))setStatus('Kamera zatrzymana');
}
$('start').addEventListener('click',start);
$('stop').addEventListener('click',stop);
$('save').addEventListener('click',()=>{
  canvas.toBlob(blob=>{
    if(!blob)return;
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');a.href=url;a.download=`assn-szkielet-${Date.now()}.png`;a.click();
    setTimeout(()=>URL.revokeObjectURL(url),2500);
  },'image/png');
});
$('threshold').addEventListener('input',()=>{$('threshold-display').textContent=`${$('threshold').value}%`;});
for(const id of ['framing','mirror','smooth'])$(id).addEventListener('change',resetPredictions);
window.addEventListener('pagehide',stop);
