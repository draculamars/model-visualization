import assert from 'node:assert/strict';
import {models} from '../dist/models.js';
import {architecture,buildGraph,tensorShape,tensorSize} from '../dist/graph.js';
import {attention,sampleMatrices} from '../dist/math.js';

let cases=0,maximum=0;
for(const model of models){
  for(const [label,url] of model.source){assert(label);assert.equal(new URL(url).protocol,'https:');}
  for(const level of [0,1,2,3]){
    const focuses=level===2?architecture(model).filter(n=>n.stack&&n.depth).flatMap(n=>[{module:n.id,layer:0},{module:n.id,layer:n.depth-1}]):[{}];
    for(const focus of focuses.length?focuses:[{}]){
      const graph=buildGraph(model,level,focus);
      assert(graph.nodes.length>0);
      assert.equal(new Set(graph.nodes.map(n=>n.id)).size,graph.nodes.length);
      for(const n of graph.nodes)assert([n.x,n.y,n.w,n.h].every(Number.isFinite));
      const order=new Map(graph.nodes.map((n,i)=>[n.id,i]));
      for(const edge of graph.edges){
        assert(order.has(edge.from)&&order.has(edge.to));
        if(!edge.loop)assert(order.get(edge.from)<order.get(edge.to),`${model.id}: ${edge.from} → ${edge.to}`);
      }
      maximum=Math.max(maximum,graph.nodes.length);cases++;
    }
  }
}
const get=id=>models.find(m=>m.id===id);
const hybrid=buildGraph(get('qwen35'),3);
assert.equal(hybrid.nodes.filter(n=>n.id.startsWith('language.')&&n.id.endsWith('.delta')).length,24);
assert.equal(hybrid.nodes.filter(n=>n.id.startsWith('language.')&&n.id.endsWith('.softmax')).length,8);
const pi=buildGraph(get('pi05'),3);
for(let i=0;i<18;i++){
  assert(pi.edges.some(e=>e.from===`language.${i}.krope`&&e.to===`action.${i}.prefixkv`));
  assert(pi.edges.some(e=>e.from===`language.${i}.v`&&e.to===`action.${i}.prefixkv`));
}
const groot=buildGraph(get('groot17'),3);
assert.equal(groot.nodes.filter(n=>n.id.endsWith('.context')).length,16);
assert.equal(groot.nodes.filter(n=>n.id.endsWith('.gelu')).length,32);
const deepseek=buildGraph(get('deepseek3'),3);
assert.equal(deepseek.nodes.filter(n=>n.id.endsWith('.router')).length,58);
assert.deepEqual(deepseek.map.get('language.0.up').shape,['B','T',18432]);
const {Q,K,V}=sampleMatrices();
for(const causal of [true,false]){
  const r=attention(Q,K,V,{causal});
  r.probabilities.forEach((row,i)=>{
    assert(Math.abs(row.reduce((a,b)=>a+b,0)-1)<1e-12);
    if(causal)row.forEach((v,j)=>{if(j>i)assert.equal(v,0)});
  });
  if(causal)assert.deepEqual(r.output[0],V[0]);
}
assert.deepEqual(attention([[0,0]],[[1,2],[3,4]],[[2,4],[6,8]],{causal:false}).output,[[4,6]]);
const shape=tensorShape(['B',8,'H+1','T+H+1'],{batch:2,tokens:16,horizon:50,actionDim:32},{});
assert.deepEqual(shape,[2,8,51,67]);assert.equal(tensorSize(shape).elements,54672);
assert.equal(tensorSize(['B','unknown']),null);
console.log(`${cases} graph cases, ${models.length} models, up to ${maximum} expanded nodes; family invariants, attention and shape calculations passed.`);
