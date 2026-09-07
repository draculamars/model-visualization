export function attention(Q,K,V,{causal=true,temperature=1}={}){
 const d=Q[0].length;
 if(!(temperature>0)||Q.some(r=>r.length!==d)||K.some(r=>r.length!==d)||K.length!==V.length)throw new Error('Invalid attention dimensions');
 const scores=Q.map(q=>K.map(k=>q.reduce((v,x,i)=>v+x*k[i],0)/Math.sqrt(d)));
 const masked=scores.map((r,i)=>r.map((s,j)=>causal&&j>i?-Infinity:s/temperature));
 const probabilities=masked.map(r=>{const max=Math.max(...r);const exps=r.map(x=>Math.exp(x-max));const sum=exps.reduce((a,b)=>a+b,0);return exps.map(x=>x/sum)});
 const output=probabilities.map(p=>V[0].map((_,d)=>p.reduce((acc,w,j)=>acc+w*V[j][d],0)));
 return {scores,masked,probabilities,output};
}
export const sampleMatrices=()=>({Q:[[1,.2,-.4],[.1,1,.3],[.8,.4,.2],[-.3,.6,1]],K:[[.8,.1,-.2],[.2,.9,.1],[1,.3,.4],[-.1,.5,.9]],V:[[.4,.8,.1],[.9,.2,-.3],[.2,.5,.7],[-.4,.1,1]]});
