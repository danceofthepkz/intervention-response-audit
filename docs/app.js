'use strict';
const D = window.AUDIT_DATA;
const $ = id => document.getElementById(id);
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = x => `${(100*x).toFixed(1)}%`;
const points = x => (100*Math.abs(x)).toFixed(2).replace(/\.00$/, '');
const BLUE='#1976ad', ORANGE='#cb6b2b', INK='#292d30';
let selected=D.defaultIndex, arm='authority', q=D.states[selected].q;
const state=()=>D.states[selected];
const svg=(w,h,body,label)=>`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}">${body}</svg>`;
const text=(x,y,t,extra='')=>`<text x="${x}" y="${y}" ${extra}>${esc(t)}</text>`;
const line=(x1,y1,x2,y2,color='#dce1e4',extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" ${extra}/>`;
const circle=(x,y,r,color)=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${color}"/>`;
const diamond=(x,y)=>`<path d="M${x} ${y-6}l6 6-6 6-6-6z" fill="${INK}"/>`;
const kl=(p,r)=>(p===0?0:p*Math.log(p/r))+(p===1?0:(1-p)*Math.log((1-p)/(1-r)));
const jsd=(a,b)=>{const m=(a+b)/2;return (kl(a,m)+kl(b,m))/2;};

function renderPair(){
  const s=state(), a=s.arms.authority.mean, h=s.arms.hedge.mean;
  const student=s.promptBefore.match(/You are (.+?)\./)?.[1];
  const topics={internship:'an internship lead',gossip:'campus gossip',club_event:'a club event',course_material:'course materials',policy_change:'a policy change'};
  $('example-context').textContent=student ? `In this example, the simulated student is ${student}, reading a message about ${topics[s.topic]||s.topic}.` : `This example concerns ${topics[s.topic]||s.topic}.`;
  $('message').textContent=s.arms[arm].text;
  $('message').className=arm;
  $('full-prompt').textContent=s.promptBefore+s.arms[arm].text+s.promptAfter;
  document.querySelectorAll('[data-arm]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.arm===arm)));
  const x=v=>100+245*v;
  let chart='';
  [0,.5,1].forEach(t=>{chart+=line(x(t),17,x(t),111,'#edf0f2')+text(x(t),133,`${100*t}%`,'text-anchor="middle" class="axis"');});
  [['authority',a,37,BLUE],['hedge',h,91,ORANGE]].forEach(([key,v,y,color])=>{
    chart+=text(0,y+4,key==='authority'?'Authority':'Hedge');
    chart+=`<rect x="100" y="${y-9}" width="${245*v}" height="18" fill="${color}" opacity="${arm===key?1:.35}"/>`;
    chart+=text(x(v)+9,y+4,pct(v),'class="value"');
  });
  $('probability-chart').innerHTML=svg(410,150,chart,`Teacher predictions: authority ${pct(a)}, hedge ${pct(h)}.`);
  $('pair-reading').textContent=`With ${arm==='authority'?'authority framing':'hedged wording'}, the predicted chance of resharing is ${pct(s.arms[arm].mean)}.`;
  $('example-result').textContent=s.delta===0?`Here, the teacher gives both versions the same mean probability: ${pct(a)}. Changing the wording has no measured effect in this situation.`:`Here, the predicted chance of resharing ${s.delta>0?'rises':'falls'} from ${pct(h)} with hedging to ${pct(a)} with authority framing—a ${points(s.delta)}-percentage-point ${s.delta>0?'increase':'decrease'}. That difference is the response we want the replacement model to preserve.`;
  $('calls-table').innerHTML=['authority','hedge'].map(k=>`<tr><td>${k==='authority'?'Authority':'Hedge'}</td>${s.arms[k].samples.map(v=>`<td>${pct(v)}</td>`).join('')}</tr>`).join('');
  $('feature-dots').innerHTML=Object.keys(s.features).map(()=>'<i></i>').join('');
  $('fitted-value').textContent=pct(s.q);
  $('collision-caption').textContent=`Each square stands for one unchanged input feature. The archived fitted model predicts ${pct(s.q)} for either message in this situation. These inputs are identical across all 60 message pairs.`;
  $('compromise-explanation').textContent=s.delta===0?`For this particular pair, both teacher means are ${pct(a)}, so one shared answer can match both. The representation fails where the teacher does respond to wording, not where the two teacher answers already agree.`:`The best compromise is ${pct(s.midpoint)}, halfway between the teacher’s two answers. It gets their average right. But the cheaper model still predicts no change between the messages, while the teacher’s response is ${points(s.delta)} percentage points ${s.delta>0?'higher':'lower'} with authority framing.`;
  $('feature-table').innerHTML=Object.entries(s.features).map(([k,v])=>`<tr><td>${esc(k)}</td><td>${esc(v)}</td><td>${esc(v)}</td></tr>`).join('');
}

function renderFit(){
  const s=state(), x=v=>123+389*v;
  let chart='';
  [0,.25,.5,.75,1].forEach(t=>{chart+=line(x(t),28,x(t),122,'#edf0f2')+text(x(t),145,`${100*t}%`,'text-anchor="middle" class="axis"');});
  [['Authority',s.arms.authority.mean,49,BLUE],['Hedge',s.arms.hedge.mean,100,ORANGE]].forEach(([name,v,y,color])=>{
    chart+=text(0,y+4,name)+line(123,y,512,y)+circle(x(v),y,6,color)+diamond(x(q),y)+text(545,y+4,pct(q),'class="value"');
  });
  chart+=circle(123,180,4,BLUE)+circle(134,180,4,ORANGE)+text(148,184,'Teacher’s answer')+diamond(352,180)+text(367,184,'Your shared answer');
  $('shared-answers').innerHTML=svg(660,203,chart,`Teacher: authority ${pct(s.arms.authority.mean)}, hedge ${pct(s.arms.hedge.mean)}. Shared answer: ${pct(q)} for both.`);
  $('shared-q').value=q;
  $('shared-q').setAttribute('aria-valuetext',pct(q));
  $('q-value').textContent=pct(q);
  $('fit-reading').textContent=Math.abs(q-s.midpoint)<1e-8 ? `This is the best shared answer. ${s.delta===0?'It matches both teacher means for this pair.':'The average prediction improves, but the model’s response to the wording change is still zero.'}` : `Both messages now receive ${pct(q)}. Moving the answer changes the prediction, but the difference between the model’s two answers remains zero.`;
  renderLoss();
}

function renderLoss(){
  const s=state(), a=s.arms.authority.mean, h=s.arms.hedge.mean, m=s.midpoint;
  const loss=r=>(kl(a,r)+kl(h,r))/2;
  const floor=jsd(a,h), total=loss(q);
  const lo=Math.max(.001,Math.min(m,q)-.065), hi=Math.min(.999,Math.max(m,q)+.065);
  const maxY=Math.max(loss(lo),loss(hi),total,.01)*1.14;
  const x=v=>54+(v-lo)/(hi-lo)*568, y=v=>187-v/maxY*160;
  let chart='';
  [0,.5,1].forEach(t=>{const v=maxY*t;chart+=line(54,y(v),622,y(v),'#edf0f2')+text(45,y(v)+4,v.toFixed(3),'text-anchor="end" class="axis"');});
  [lo,(lo+hi)/2,hi].forEach(v=>{chart+=text(x(v),211,pct(v),'text-anchor="middle" class="axis"');});
  let path='';for(let i=0;i<=160;i++){const v=lo+(hi-lo)*i/160;path+=`${i?'L':'M'}${x(v)},${y(loss(v))}`;}
  chart+=`<path d="${path}" fill="none" stroke="${BLUE}" stroke-width="2"/>`;
  chart+=line(54,y(floor),622,y(floor),ORANGE,'stroke-dasharray="3 4"')+circle(x(m),y(floor),4,ORANGE)+circle(x(q),y(total),5,INK);
  chart+=text(54,15,'KL regret (nats)')+text(622,15,'Orange line: minimum','text-anchor="end"')+text(338,236,'Shared probability','text-anchor="middle"');
  $('fit-chart').innerHTML=svg(650,240,chart,`Current KL regret ${total.toFixed(5)} nats; minimum ${floor.toFixed(5)} nats.`);
  $('total-loss').textContent=total.toFixed(5);
  $('floor-loss').textContent=floor.toFixed(5);
}

function renderRecovery(){
  const model=D.comparison.models.find(m=>m.id===$('model').value), x=v=>100+458*v;
  let chart='';
  [0,.5,1].forEach(t=>{chart+=line(x(t),10,x(t),86,'#edf0f2')+text(x(t),109,`${100*t}%`,'text-anchor="middle" class="axis"');});
  chart+=text(0,31,'Teacher')+`<rect x="100" y="17" width="458" height="18" fill="${BLUE}" opacity=".3"/>`+text(574,31,'100%','class="value"');
  chart+=text(0,72,model.id)+`<rect x="100" y="58" width="${458*model.recovery}" height="18" fill="${BLUE}"/>`;
  if(model.recovery===0)chart+=line(100,58,100,76,INK,'stroke-width="2"');
  chart+=text(x(model.recovery)+16,72,pct(model.recovery),'class="value"');
  $('recovery-chart').innerHTML=svg(660,132,chart,`${model.id} recovered ${pct(model.recovery)} of the teacher’s mean signed response.`);
  const descriptions={R0:'Without message information, the model’s response is exactly zero. It recovers none of the teacher’s mean response.',R1:'A framing label lets the model distinguish the two messages. The fitted model recovers 87% of the teacher’s mean response.',R2:'A numerical encoding of the message text lets the model distinguish the messages. The fitted model recovers 81.3% of the mean response.',R3:'A compressed text encoding also allows different predictions. The fitted model recovers 83.4% of the mean response.'};
  $('recovery-reading').textContent=descriptions[model.id];
}

$('state').innerHTML=D.states.map((s,i)=>`<option value="${i}">${esc(s.scenario)} · ${esc(s.topic)} · ${s.delta>=0?'+':'−'}${points(s.delta)} pp</option>`).join('');
$('state').value=selected;
$('state').addEventListener('change',()=>{selected=Number($('state').value);q=state().q;renderPair();renderFit();});
document.querySelectorAll('[data-arm]').forEach(b=>b.addEventListener('click',()=>{arm=b.dataset.arm;renderPair();}));
$('shared-q').addEventListener('input',()=>{q=Number($('shared-q').value);renderFit();});
$('best-q').addEventListener('click',()=>{q=state().midpoint;renderFit();});
$('fitted-q').addEventListener('click',()=>{q=state().q;renderFit();});
$('model').addEventListener('change',renderRecovery);
$('teacher-model').textContent=`Teacher configuration: ${D.summary.model}. Prompting and sampling were fixed, not varied; measured responses are specific to this elicitation.`;
$('comparison-table').innerHTML=D.comparison.models.map(m=>`<tr><td>${esc(m.id)}</td><td>${m.kl.toFixed(5)}</td><td>${m.eir.toFixed(4)} [${m.ci.map(v=>v.toFixed(4)).join(', ')}]</td><td>${pct(m.recovery)}</td></tr>`).join('');
$('sources').innerHTML=D.provenance.map(p=>`<p>${esc(p.path)}<br>SHA-256: ${esc(p.sha256)}</p>`).join('');
renderPair();renderFit();renderRecovery();
