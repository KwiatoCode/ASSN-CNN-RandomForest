const MAGIC = 'ASNRF001';
const LEAF = 65535;
const HEADER = 24;
const RECORD = 14;

export async function loadForest(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Nie pobrano modelu: HTTP ${response.status}`);
  if (typeof DecompressionStream !== 'function') throw new Error('Wymagany iOS 16.4 lub nowszy (obsługa gzip).');
  const uncompressed = response.body.pipeThrough(new DecompressionStream('gzip'));
  const buffer = await new Response(uncompressed).arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(...bytes.slice(0,8));
  if (magic!==MAGIC)throw new Error('Niepoprawny format modelu');
  const view = new DataView(buffer);
  const treeCount = view.getUint32(8,true);
  const featureCount = view.getUint32(12,true);
  const classCount = view.getUint32(16,true);
  const total = view.getUint32(20,true);
  if (treeCount < 1 || classCount!==26 || featureCount!==2500) throw new Error('Niepoprawne parametry Random Forest');
  const labels = String.fromCharCode(...bytes.slice(HEADER,HEADER+classCount));
  if (labels!=='ABCDEFGHIJKLMNOPQRSTUVWXYZ')throw new Error('Nieznane etykiety liter');
  const starts = new Array(treeCount);
  const lengths = new Array(treeCount);
  let cursor=HEADER+classCount+treeCount*2, count=0;
  for(let i=0;i<treeCount;i++){
    const len = view.getUint16(HEADER+classCount+i*2,true);
    starts[i]=cursor;lengths[i]=len;
    count+=len;cursor+=len*RECORD;
  }
  if (count!==total || cursor!==buffer.byteLength)throw new Error('Niespójny model Random Forest');
  return {treeCount,featureCount,classCount,view,starts,lengths};
}

export function featureVector(context) {
  const pixels=context.getImageData(0,0,400,400).data;
  const features = new Float64Array(2500);
  for(let y=0;y<400;y++){
    const rowBlock=(y>>3)*50,offset=y*400*4;
    for(let x=0;x<400;x++){
      const i=offset+x*4;
      features[rowBlock+(x>>3)] += 0.299*pixels[i]+0.587*pixels[i+1]+0.114*pixels[i+2];
    }
  }
  const scale=1/(255*64);
  for(let i=0;i<features.length;i++)features[i]*=scale;
  return features;
}

export function predictForest(model,features) {
  if(features.length!==model.featureCount)throw new Error('Niepoprawne cechy obrazu');
  const votes = new Float32Array(model.classCount);
  const {view,starts,lengths,treeCount}=model;
  for(let t=0;t<treeCount;t++){
    let node=0,step=0;
    while(true){
      if (node>=lengths[t]||++step>lengths[t])throw new Error('Niezgodna struktura drzewa');
      const at=starts[t]+node*RECORD;
      const feature=view.getUint16(at,true);
      const yes=view.getUint16(at+2,true);
      if(feature===LEAF){votes[yes]+=1;break;}
      const no=view.getUint16(at+4,true);
      const threshold=view.getFloat64(at+6,true);
      node=features[feature]<=threshold?yes:no;
    }
  }
  for(let i=0;i<votes.length;i++)votes[i]/=treeCount;
  return votes;
}
