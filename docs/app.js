'use strict';
const D=window.AUDIT_DATA, $=id=>document.getElementById(id);
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=x=>`${(100*x).toFixed(1)}%`, points=x=>(100*Math.abs(x)).toFixed(2).replace(/\.00$/,'');
const BLUE='#1976ad', ORANGE='#cb6b2b', INK='#292d30';
let selected=D.defaultIndex, arm='authority', q=D.states[selected].q, fullScale=false, sample='all';
const state=()=>D.states[selected];
const svg=(w,h,body,label)=>`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}">${body}</svg>`;
const text=(x,y,t,extra='')=>`<text x="${x}" y="${y}" ${extra}>${esc(t)}</text>`;
const line=(x1,y1,x2,y2,c='#dce1e4',extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${c}" ${extra}/>`;
const circle=(x,y,r,c)=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${c}"/>`;
const kl=(p,r)=>p===r?0:(p===0?0:p*Math.log(p/r))+(p===1?0:(1-p)*Math.log((1-p)/(1-r)));
const jsd=(a,b)=>{const m=(a+b)/2;return (kl(a,m)+kl(b,m))/2;};
const fitMax=()=>fullScale?1:Math.min(1,Math.max(.3,Math.ceil((Math.max(state().q,state().arms.authority.mean,state().arms.hedge.mean)+.04)*10)/10));
const ticks=(max,n=5)=>Array.from({length:n+1},(_,i)=>max*i/n), at=(v,max)=>`${100*v/max}%`;
function setHero(condition){
  $('hero-scene').dataset.condition=condition;
  $('hero-message').textContent=D.states[D.defaultIndex].arms[condition].text;
  document.querySelectorAll('[data-hero-arm]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.heroArm===condition)));
}
function renderProfile(){
  const s=state(), student=s.promptBefore.match(/You are a year (\d+) (.+?) student\./);
  const years={'1':'First-year','2':'Second-year','3':'Third-year','4':'Fourth-year'}, majors={Bio:'biology',Econ:'economics',Art:'art',Engineering:'engineering',CS:'computer science'};
  $('example-context').textContent=student?`${years[student[1]]||`Year ${student[1]}`} ${majors[student[2]]||student[2]} student`:'A simulated student';
  const sentences=s.promptBefore.split('\n\nWhat you see:')[0].split(/\.\s+/);
  let traits=[sentences.find(t=>/you keep a small circle|you are highly social|you are friendly but selective/.test(t)),sentences.find(t=>/You actively keep up|You don't usually pay|You sometimes/.test(t))].filter(Boolean).map(t=>t.replace(/^you /,'You ').replace(/\.$/,''));
  if(selected===D.defaultIndex)traits=['Small social circle','Watching for internships'];
  $('student-traits').innerHTML=traits.map(t=>`<li>${esc(t)}</li>`).join('');
  const source=s.promptBefore.match(/What you see:\s*\n- (.+?)\. They sent/s)?.[1];
  $('student-source').textContent=selected===D.defaultIndex?'Message from Ethan, a dorm acquaintance':source?`Message from ${source}.`:'';
}
function renderPair(){
  const s=state(), a=s.arms.authority.mean, h=s.arms.hedge.mean;
  renderProfile();$('message').textContent=s.arms[arm].text;$('message').className=arm;
  $('full-prompt').textContent=s.promptBefore+s.arms[arm].text+s.promptAfter;
  document.querySelectorAll('[data-arm]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.arm===arm)));
  const max=Math.min(1,Math.max(.25,Math.ceil((Math.max(a,h)+.035)*20)/20)), chart=$('probability-chart');
  // Keep the point node alive so switching wording moves the same point.
  if(!chart.firstElementChild)chart.innerHTML='<div class="pair-numberline"><div id="pair-grid"></div><div id="pair-point" class="pair-point"><strong></strong></div><div id="pair-ticks" class="plot-ticks"></div></div>';
  $('pair-grid').innerHTML=ticks(max).map(v=>`<i style="left:${at(v,max)}"></i>`).join('');
  $('pair-ticks').innerHTML=ticks(max).map(v=>`<span style="left:${at(v,max)}">${Math.round(v*100)}%</span>`).join('');
  const point=$('pair-point');point.style.left=at(s.arms[arm].mean,max);point.dataset.condition=arm;point.querySelector('strong').textContent=pct(s.arms[arm].mean);
  chart.setAttribute('role','img');chart.setAttribute('aria-label',`${arm} wording: ${pct(s.arms[arm].mean)} reshare probability; displayed scale 0 to ${pct(max)}.`);
  $('pair-reading').textContent=`${arm==='authority'?'Authority':'Hedge'} wording → ${pct(s.arms[arm].mean)}. Same student, same context.`;
  $('example-result').textContent=s.delta===0?`Here, the teacher gives both versions the same mean probability: ${pct(a)}. Changing the wording has no measured effect in this situation.`:`Here, the predicted chance of resharing ${s.delta>0?'rises':'falls'} from ${pct(h)} with hedging to ${pct(a)} with authority framing—a ${points(s.delta)}-percentage-point ${s.delta>0?'increase':'decrease'}. That difference is the response we want the replacement model to preserve.`;
  $('calls-table').innerHTML=['authority','hedge'].map(k=>`<tr><td>${k==='authority'?'Authority':'Hedge'}</td>${s.arms[k].samples.map(v=>`<td>${pct(v)}</td>`).join('')}</tr>`).join('');
  $('feature-dots').innerHTML=Object.keys(s.features).map(()=>'<i></i>').join('');
  $('collision-authority').textContent=s.arms.authority.text;$('collision-hedge').textContent=s.arms.hedge.text;
  $('fitted-value').textContent=pct(s.q);
  $('collision-caption').textContent=`Two messages enter; the same 27 features reach the model. The archived R0 prediction is ${pct(s.q)} for either message. This input collision holds across all 60 pairs.`;
  $('compromise-explanation').textContent=s.delta===0?`For this pair, both teacher means are ${pct(a)}, so one shared answer can match both.`:`The best compromise is ${pct(s.midpoint)}, halfway between the teacher’s two answers. It gets their average right. But the cheaper model still predicts no change, while the teacher’s response is ${points(s.delta)} percentage points ${s.delta>0?'higher':'lower'} with authority framing.`;
  $('feature-table').innerHTML=Object.entries(s.features).map(([k,v])=>`<tr><td>${esc(k)}</td><td>${esc(v)}</td><td>${esc(v)}</td></tr>`).join('');
}
function renderFit(){
  const s=state(), max=fitMax(), a=s.arms.authority.mean, h=s.arms.hedge.mean, m=s.midpoint;
  $('fit-grid').innerHTML=ticks(max,3).map(v=>`<i style="left:${at(v,max)}"></i>`).join('');
  $('fit-ticks').innerHTML=ticks(max,3).map(v=>`<span style="left:${at(v,max)}">${Math.round(v*100)}%</span>`).join('');
  [['fit-authority',a,'Authority'],['fit-hedge',h,'Hedge']].forEach(([id,v,name])=>{const el=$(id);el.style.left=at(v,max);el.querySelector('span').textContent=`${name} ${pct(v)}`;el.classList.toggle('near-end',v/max>.72);el.classList.toggle('near-start',v/max<.15);});
  const bracket=$('response-bracket');bracket.style.left=at(Math.min(a,h),max);bracket.style.width=at(Math.abs(a-h),max);bracket.querySelector('span').textContent=`${s.delta<0?'−':s.delta>0?'+':''}${points(s.delta)} pp · fixed teacher response`;bracket.classList.toggle('zero-gap',a===h);
  bracket.classList.toggle('near-end',m/max>.65);bracket.classList.toggle('near-start',m/max<.35);
  $('midpoint-marker').classList.toggle('near-end',m/max>.65);
  $('midpoint-marker').style.left=at(m,max);$('actual-marker').style.left=at(s.q,max);$('actual-marker').classList.toggle('near-end',s.q/max>.65);
  $('shared-marker').style.left=at(q,max);$('fit-gap').style.left=at(Math.min(m,q),max);$('fit-gap').style.width=at(Math.abs(q-m),max);$('q-value').textContent=pct(q);
  const track=$('fit-track');track.setAttribute('aria-valuemax',String(Math.min(.999,max)*100));track.setAttribute('aria-valuenow',String(q*100));track.setAttribute('aria-valuetext',`${pct(q)} for both messages`);
  $('fit-scale').setAttribute('aria-pressed',String(fullScale));$('fit-scale').textContent=fullScale?'Zoom to this situation':'Show full 0–100% scale';
  $('fit-distance').textContent=`Distance to the best compromise: ${points(q-m)} percentage points. The purple segment shrinks as you approach it.`;
  $('fit-reading').textContent=Math.abs(q-m)<1e-8?`This is the best shared answer. ${s.delta===0?'It matches both teacher means.':'Reducible error reaches zero, but the model’s response to the wording change is still zero.'}`:`Both messages receive ${pct(q)}. The prediction changes; the model’s response to the wording change remains zero.`;
  renderLoss();
}
function renderLoss(){
  const s=state(), a=s.arms.authority.mean, h=s.arms.hedge.mean, m=s.midpoint, loss=r=>(kl(a,r)+kl(h,r))/2;
  const floor=jsd(a,h), reducible=kl(m,q), total=floor+reducible;
  // Fixed scale during a drag: the irreducible segment never changes width.
  const maxLoss=Math.max(loss(s.q)*2,floor*4,.06);
  $('live-floor').textContent=floor.toFixed(5);$('live-fit').textContent=reducible.toFixed(5);
  $('loss-floor').style.width=`${100*floor/maxLoss}%`;$('loss-fit').style.width=`${100*Math.min(reducible,maxLoss-floor)/maxLoss}%`;
  $('loss-track').classList.toggle('exceeds-scale',total>maxLoss);
  $('loss-track').setAttribute('aria-label',`KL error: irreducible ${floor.toFixed(5)}, reducible ${reducible.toFixed(5)}, total ${total.toFixed(5)} nats.`);
  $('loss-scale').textContent=`Total ${total.toFixed(5)} nats · fixed bar scale 0–${maxLoss.toFixed(2)} nats${total>maxLoss?' · total exceeds the bar; exact values shown above':''}`;
  const lo=Math.max(.001,Math.min(m,q)-.065), hi=Math.min(.999,Math.max(m,q)+.065), maxY=Math.max(loss(lo),loss(hi),total,.01)*1.14;
  const x=v=>54+(v-lo)/(hi-lo)*568, y=v=>187-v/maxY*160;
  let chart='';[0,.5,1].forEach(t=>{const v=maxY*t;chart+=line(54,y(v),622,y(v),'#edf0f2')+text(45,y(v)+4,v.toFixed(3),'text-anchor="end" class="axis"');});
  [lo,(lo+hi)/2,hi].forEach(v=>{chart+=text(x(v),211,pct(v),'text-anchor="middle" class="axis"');});
  let path='';for(let i=0;i<=160;i++){const v=lo+(hi-lo)*i/160;path+=`${i?'L':'M'}${x(v)},${y(loss(v))}`;}
  chart+=`<path d="${path}" fill="none" stroke="${BLUE}" stroke-width="2"/>`+line(54,y(floor),622,y(floor),'#778574','stroke-dasharray="3 4"')+circle(x(m),y(floor),4,'#778574')+circle(x(q),y(total),5,INK);
  chart+=text(54,15,'KL regret (nats)')+text(622,15,'Dashed: minimum','text-anchor="end"')+text(338,236,'Shared probability','text-anchor="middle"');
  $('fit-chart').innerHTML=svg(650,240,chart,`Current KL regret ${total.toFixed(5)} nats; minimum ${floor.toFixed(5)} nats.`);$('total-loss').textContent=total.toFixed(5);$('floor-loss').textContent=floor.toFixed(5);
}
function updateQ(value){q=Math.max(.001,Math.min(.999,fitMax(),value));renderFit();}
const track=$('fit-track');
function dragAnswer(e){const box=track.getBoundingClientRect();updateQ((e.clientX-box.left)/box.width*fitMax());}
track.addEventListener('pointerdown',e=>{if(e.button!==0)return;track.focus({preventScroll:true});track.setPointerCapture(e.pointerId);dragAnswer(e);});
track.addEventListener('pointermove',e=>{if(track.hasPointerCapture(e.pointerId))dragAnswer(e);});
track.addEventListener('pointerup',e=>{if(track.hasPointerCapture(e.pointerId)){dragAnswer(e);track.releasePointerCapture(e.pointerId);}});
track.addEventListener('keydown',e=>{const step=e.shiftKey?.01:.001, values={ArrowRight:q+step,ArrowUp:q+step,ArrowLeft:q-step,ArrowDown:q-step,Home:.001,End:Math.min(.999,fitMax()),PageUp:q+.01,PageDown:q-.01};if(e.key in values){e.preventDefault();updateQ(values[e.key]);}});
function renderRecovery(){
  const unseen=sample==='unseen';document.querySelectorAll('[data-sample]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.sample===sample)));
  const max=unseen?.065:1, values=D.comparison.models.map(m=>unseen?D.comparison.transfer.eir[m.id]:m.recovery), ref=unseen?D.comparison.transfer.eir.R0:1;
  $('comparison-metric').textContent=unseen?'Unseen wording · mean absolute response error (pp), lower is better':'All 60 · recovery of the teacher’s mean signed response';
  const labels={R0:'27 structured features',R1:'+ Framing label',R2:'+ Text embedding',R3:'+ Compressed embedding'};
  $('recovery-chart').innerHTML=`<div class="comparison-axis"><span>${unseen?'0 pp':'0%'}</span><span>${unseen?'6.5 pp':'Teacher = 100%'}</span></div><div class="comparison-rows" style="--reference:${at(ref,max)}">${D.comparison.models.map((m,i)=>`<div class="comparison-row"><div class="representation-label"><strong>${m.id}</strong> ${labels[m.id]}${m.id==='R1'?'<small>Framing label supplied directly</small>':''}</div><div class="representation-track"><span class="representation-bar" style="width:${at(values[i],max)}"></span><strong class="representation-value">${unseen?`${points(values[i])} pp`:pct(values[i])}</strong></div></div>`).join('')}</div>`;
  $('recovery-chart').setAttribute('role','img');$('recovery-chart').setAttribute('aria-label',D.comparison.models.map((m,i)=>`${m.id}: ${unseen?`${points(values[i])} percentage points of response error`:pct(values[i])+' recovery'}`).join('; '));
  $('recovery-reading').textContent=unseen?'None of the three alternatives improves on R0 in these 23 situations. The dashed line marks R0’s 5.06-pp error. These situations also have lower credibility and a smaller teacher response; this check does not isolate why transfer was not demonstrated.':'Adding message information recovers 81–87% of the mean response. These are four alternatives, not a ranking: the evidence does not establish a best encoding.';
  $('comparison-caption').textContent=unseen?'Metric changed: these bars show response error, not recovery. Source: paper Appendix C.2; R0 0.0506, R1 0.0568, R2 0.0566, R3 0.0565 (probability units). Recovery is not reported here because the teacher mean response is small (0.0317).':'Recovery is a ratio of mean signed responses, not accuracy on individual decisions. Source: paper Table 1; teacher mean response 8.66 pp. All values are manuscript summaries, not models refitted in the browser.';
}
$('state').innerHTML=D.states.map((s,i)=>`<option value="${i}">${esc(s.scenario)} · ${esc(s.topic)} · ${s.delta>=0?'+':'−'}${points(s.delta)} pp</option>`).join('');$('state').value=selected;
$('state').addEventListener('change',()=>{selected=Number($('state').value);q=state().q;fullScale=false;renderPair();renderFit();});
document.querySelectorAll('[data-arm]').forEach(b=>b.addEventListener('click',()=>{arm=b.dataset.arm;renderPair();}));
document.querySelectorAll('[data-hero-arm]').forEach(b=>['pointerenter','focus','click'].forEach(event=>b.addEventListener(event,()=>setHero(b.dataset.heroArm))));
$('best-q').addEventListener('click',()=>{q=state().midpoint;renderFit();});$('fitted-q').addEventListener('click',()=>{q=state().q;renderFit();});
$('fit-scale').addEventListener('click',()=>{fullScale=!fullScale;if(q>fitMax())q=state().q;renderFit();});
document.querySelectorAll('[data-sample]').forEach(b=>b.addEventListener('click',()=>{sample=b.dataset.sample;renderRecovery();}));
$('teacher-model').textContent=`Teacher configuration: ${D.summary.model}. Prompting and sampling were fixed, not varied; measured responses are specific to this elicitation.`;
$('comparison-table').innerHTML=D.comparison.models.map(m=>`<tr><td>${esc(m.id)}</td><td>${m.kl.toFixed(5)}</td><td>${m.eir.toFixed(4)} [${m.ci.map(v=>v.toFixed(4)).join(', ')}]</td><td>${pct(m.recovery)}</td></tr>`).join('');
$('sources').innerHTML=D.provenance.map(p=>`<p>${esc(p.path)}<br>SHA-256: ${esc(p.sha256)}</p>`).join('');
setHero('authority');renderPair();renderFit();renderRecovery();
if('IntersectionObserver' in window){const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting){entry.target.classList.add('revealed');observer.unobserve(entry.target);}}),{threshold:.35});observer.observe($('collision-figure'));}
