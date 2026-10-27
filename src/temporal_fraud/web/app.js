const $ = id => document.getElementById(id);
const pct = value => (value * 100).toFixed(2) + '%';
const number = value => Number(value).toLocaleString(undefined, {maximumFractionDigits: 2});
let selected = null, current = null, timer = null, requestId = 0, selectionId = 0;
async function get(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error('Evidence could not be loaded (' + response.status + ').');
  return response.json();
}
function element(tag, cls, text) {
  const el = document.createElement(tag); if (cls) el.className = cls;
  if (text !== undefined) el.textContent = text; return el;
}
async function refresh() {
  const token = ++requestId;
  $('minute-value').textContent = $('minute').value;
  const query = new URLSearchParams({model:$('model').value, minute:$('minute').value, search:$('search').value, cold:$('cold').checked});
  try {
    const cases = await get('/api/cases?' + query);
    if (token !== requestId) return;
    $('shown').textContent = cases.length; $('cases').replaceChildren();
    for (const c of cases) {
      const button = element('button', 'case' + (c.event_id === selected ? ' selected' : ''));
      button.dataset.eventId = c.event_id;button.setAttribute('role','listitem');
      const row = element('div','case-row');row.append(element('strong','', 'Event ' + c.event_id), element('strong','percent',pct(c[$('model').value + '_calibrated'])));
      button.append(row, element('div','account',c.source + ' → ' + c.target),
        element('small','',c.timestamp.slice(11,16) + ' · ' + number(c.paid) + ' ' + c.payment_currency + (c.cold ? ' · Fit-held-out account' : '')));
      button.onclick = () => choose(c.event_id);
      $('cases').append(button);
    }
    if (!cases.length) {
      $('cases').append(element('p','empty','No saved cases match these controls.'));
      selected=null;current=null;$('case-detail').hidden=true;$('empty').hidden=false;
    } else if (!cases.some(c => c.event_id === selected)) await choose(cases[0].event_id);
  } catch (error) {$('cases').textContent=error.message;}
}
async function choose(id) {
  const token=++selectionId;
  selected=id;
  const data=await get('/api/cases/'+id+'?outcomes='+$('outcomes').checked);
  if(selected!==id||token!==selectionId)return;
  current=data;$('empty').hidden=true;$('case-detail').hidden=false;
  document.querySelectorAll('.case').forEach(el=>el.classList.toggle('selected',Number(el.dataset.eventId)===id));
  $('case-title').textContent='Event '+id;$('source').textContent=data.source;$('target').textContent=data.target;
  $('case-meta').textContent=data.timestamp.replace('T',' · ')+' / '+number(data.paid)+' '+data.payment_currency+' / '+data.format;
  $('graph-score').textContent=pct(data.temporal_attention_32_calibrated);$('tree-score').textContent=pct(data.history_xgboost_calibrated);
  $('graph-bar').style.width=pct(data.temporal_attention_32_calibrated);$('tree-bar').style.width=pct(data.history_xgboost_calibrated);
  $('outcome').hidden=data.label===undefined;
  $('outcome').textContent=data.label===1?'Simulation outcome: laundering label. This is synthetic ground truth.':'Simulation outcome: legitimate label. This is synthetic ground truth.';
  $('edge-count').textContent=data.context.length+' prior interactions';drawGraph(data);
  $('context').replaceChildren();
  for(const edge of data.context){
    const row=element('tr');row.append(element('td','',edge.timestamp.replace('T',' ')),
      element('td','',edge.source+' → '+edge.target),element('td','',number(edge.paid)+' '+edge.payment_currency));$('context').append(row);
  }
}
function drawGraph(c){
  const svg=$('graph');svg.replaceChildren();const ns='http://www.w3.org/2000/svg';
  const positions=new Map([[c.source,[210,145]],[c.target,[380,145]]]);
  if(c.source===c.target)positions.set(c.source,[290,145]);
  const other=[...new Set(c.context.flatMap(e=>[e.source,e.target]))].filter(x=>!positions.has(x));
  other.forEach((node,i)=>{const a=2*Math.PI*i/Math.max(other.length,1);positions.set(node,[290+240*Math.cos(a),145+105*Math.sin(a)])});
  function shape(tag,attrs,text){const el=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);if(text!==undefined)el.textContent=text;svg.append(el);return el}
  for(const edge of c.context){const a=positions.get(edge.source),b=positions.get(edge.target);
    const line=shape('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:'#bdcdb6','stroke-width':1.2});
    const title=document.createElementNS(ns,'title');title.textContent=edge.source+' → '+edge.target+' · '+edge.timestamp;line.append(title);
  }
  const a=positions.get(c.source),b=positions.get(c.target);
  shape('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:'#367957','stroke-width':2.5,'stroke-dasharray':'5 4'});
  for(const [node,pos] of positions){const main=node===c.source||node===c.target;
    const circle=shape('circle',{cx:pos[0],cy:pos[1],r:main?13:7,fill:main?'#31694e':'#d5dfc5',stroke:main?'#fff':'#afc2a7','stroke-width':3});
    const title=document.createElementNS(ns,'title');title.textContent=node;circle.append(title);
    shape('text',{x:pos[0],y:pos[1]+(main?28:20),'text-anchor':'middle',class:main?'main-label':''},node.length>14?node.slice(0,5)+'…'+node.slice(-6):node);
  }
}
$('model').onchange=refresh;$('cold').onchange=refresh;$('minute').oninput=refresh;
let searchTimer;$('search').oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(refresh,180)};
$('outcomes').onchange=()=>{if(!$('outcomes').checked){$('outcome').hidden=true;if(current)delete current.label}if(selected!==null)choose(selected)};
$('play').onclick=()=>{
  if(timer){clearInterval(timer);timer=null;$('play').textContent='Replay from start';return}
  $('minute').value=1;refresh();$('play').textContent='Pause replay';
  timer=setInterval(()=>{if(Number($('minute').value)>=30){clearInterval(timer);timer=null;$('play').textContent='Replay from start';return}$('minute').value=Number($('minute').value)+1;refresh()},900);
};
$('export').onclick=()=>{if(!current)return;const blob=new Blob([JSON.stringify({notice:$('license').textContent,case:current},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=element('a');a.href=url;a.download='case-'+current.event_id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
(async()=>{try{const data=await get('/api/overview');if(data.mode==='Authored test fixture'){$('dataset-label').textContent='AUTHORED INTERFACE EXAMPLE';$('sample-note').textContent='Illustrative scores';$('selection-note').textContent='Three authored cases'}$('period').textContent=data.period;$('total').textContent=number(data.events_scored);$('count').textContent=data.case_count;$('latency').textContent=Math.round(data.latency_p95_ms)+' ms';$('limitations').textContent=data.selection+' '+data.limitations;$('license').textContent=data.data_notice;await refresh()}catch(error){$('period').textContent=error.message}})();
