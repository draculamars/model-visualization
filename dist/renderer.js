import {palette} from './models.js';
export class GraphCanvas{
 constructor(canvas,callbacks={}){
  this.canvas=canvas;this.ctx=canvas.getContext('2d');this.cb=callbacks;this.zoom=1;this.pan={x:0,y:0};this.mode='tensor';this.selected=null;this.hover=null;this.pointers=new Map();this.moved=false;this.raf=null;
  this.resize=new ResizeObserver(()=>{this.size();if(!this.fitted&&this.graph){this.fit();this.fitted=true}else this.draw()});this.resize.observe(canvas);
  canvas.addEventListener('wheel',e=>{e.preventDefault();const p=this.point(e);this.zoomAt(Math.exp(-e.deltaY*.0015),p.x,p.y)},{passive:false});
  canvas.addEventListener('pointerdown',e=>{canvas.focus({preventScroll:true});canvas.setPointerCapture(e.pointerId);const p=this.point(e);this.pointers.set(e.pointerId,p);this.down=p;this.moved=false;if(this.pointers.size===2)this.pinch=this.pinchState()});
  canvas.addEventListener('pointermove',e=>{const p=this.point(e);if(this.pointers.has(e.pointerId)){
    const last=this.pointers.get(e.pointerId);this.pointers.set(e.pointerId,p);if(this.pointers.size===2){const now=this.pinchState();if(this.pinch){this.zoomAt(now.dist/Math.max(1,this.pinch.dist),now.x,now.y);this.pan.x+=now.x-this.pinch.x;this.pan.y+=now.y-this.pinch.y}this.pinch=now;this.moved=true}
    else{if(Math.hypot(p.x-this.down.x,p.y-this.down.y)>4)this.moved=true;if(this.moved){this.pan.x+=p.x-last.x;this.pan.y+=p.y-last.y}}this.draw();
   }else{const n=this.hit(p);if(n?.id!==this.hover?.id){this.hover=n;this.cb.hover?.(n,p);this.draw()}}});
  canvas.addEventListener('pointerup',e=>{const p=this.point(e);if(!this.moved&&this.pointers.size===1){const n=this.hit(p);if(n){this.select(n.id);this.cb.select?.(n)}else if(this.minimap&&p.x>=this.minimap.x&&p.y>=this.minimap.y&&p.x<=this.minimap.x+this.minimap.w&&p.y<=this.minimap.y+this.minimap.h)this.navigateMini(p)}this.pointers.delete(e.pointerId);this.pinch=null;if(this.pointers.size){this.down=[...this.pointers.values()][0];this.moved=true}});
  canvas.addEventListener('pointercancel',e=>{this.pointers.delete(e.pointerId);this.pinch=null});
  canvas.addEventListener('pointerleave',()=>{this.hover=null;this.cb.hover?.(null);this.draw()});
  canvas.addEventListener('dblclick',e=>{const n=this.hit(this.point(e));if(n)this.cb.drill?.(n)});
  canvas.addEventListener('keydown',e=>{if(['+','=','-','f','F','ArrowDown','ArrowUp','ArrowLeft','ArrowRight','Enter'].includes(e.key)){e.preventDefault();if(e.key==='+'||e.key==='=')this.zoomAt(1.3);else if(e.key==='-')this.zoomAt(1/1.3);else if(e.key.toLowerCase()==='f')this.fit();else if(e.key==='Enter')this.cb.drill?.(this.graph?.map.get(this.selected));else this.cb.step?.(e.key==='ArrowDown'||e.key==='ArrowRight'?1:-1)}});
 }
 point(e){const r=this.canvas.getBoundingClientRect();return{x:e.clientX-r.left,y:e.clientY-r.top}}
 pinchState(){const [a,b]=[...this.pointers.values()];return{dist:Math.hypot(a.x-b.x,a.y-b.y),x:(a.x+b.x)/2,y:(a.y+b.y)/2}}
 size(){const r=this.canvas.getBoundingClientRect();this.w=r.width;this.h=r.height;this.dpr=Math.min(devicePixelRatio||1,2);this.canvas.width=Math.round(this.w*this.dpr);this.canvas.height=Math.round(this.h*this.dpr)}
 setGraph(graph,fit=true){this.graph=graph;this.hover=null;this.fitted=false;if(fit){this.fit();this.fitted=true}this.draw()}
 setMode(mode){this.mode=mode;this.draw()}
 select(id){this.selected=id;this.draw()}
 fit(){if(!this.graph||!this.w||!this.h)return;const b=this.graph.bounds;this.zoom=Math.min((this.w-110)/(b.w+60),(this.h-145)/(b.h+70),1.15);this.zoom=Math.max(.001,this.zoom);this.pan.x=(this.w-b.w*this.zoom)/2-b.minX*this.zoom;this.pan.y=(this.h-b.h*this.zoom)/2-b.minY*this.zoom+5;this.cb.zoom?.(this.zoom);this.draw()}
 focus(id){const n=this.graph?.map.get(id);if(!n)return;this.zoom=Math.max(this.zoom,Math.min(1.1,(this.w-100)/750));this.pan.x=this.w/2-(n.x+n.w/2)*this.zoom;this.pan.y=this.h/2-(n.y+n.h/2)*this.zoom;this.cb.zoom?.(this.zoom);this.draw()}
 zoomAt(factor,x=this.w/2,y=this.h/2){const z=Math.min(4,Math.max(.001,this.zoom*factor));const ratio=z/this.zoom;this.pan.x=x-(x-this.pan.x)*ratio;this.pan.y=y-(y-this.pan.y)*ratio;this.zoom=z;this.cb.zoom?.(z);this.draw()}
 hit(p){if(!this.graph)return null;const x=(p.x-this.pan.x)/this.zoom,y=(p.y-this.pan.y)/this.zoom;for(let i=this.graph.nodes.length-1;i>=0;i--){const n=this.graph.nodes[i];if(x>=n.x-5&&x<=n.x+n.w+5&&y>=n.y-15&&y<=n.y+n.h+10)return n}return null}
 draw(){if(this.raf)return;this.raf=requestAnimationFrame(()=>{this.raf=null;this.paint()})}
 path(points,fill,stroke){const c=this.ctx;c.beginPath();points.forEach((p,i)=>i?c.lineTo(...p):c.moveTo(...p));c.closePath();if(fill){c.fillStyle=fill;c.fill()}if(stroke){c.strokeStyle=stroke;c.stroke()}}
 text(text,x,y,color,size=13,max=200){const c=this.ctx;c.fillStyle=color;c.font=`${size}px "DM Sans", "PingFang SC", sans-serif`;while(c.measureText(text).width>max&&text.length>3)text=text.slice(0,-2)+'…';c.fillText(text,x,y)}
 paint(){const c=this.ctx;if(!this.w||!this.h)return;c.setTransform(this.dpr,0,0,this.dpr,0,0);c.clearRect(0,0,this.w,this.h);c.fillStyle='#10171a';c.fillRect(0,0,this.w,this.h);
  c.fillStyle='#27353b';const grid=28,ox=((this.pan.x%grid)+grid)%grid,oy=((this.pan.y%grid)+grid)%grid;for(let x=ox;x<this.w;x+=grid)for(let y=oy;y<this.h;y+=grid)c.fillRect(x,y,1,1);
  if(!this.graph)return;const z=this.zoom,px=this.pan.x,py=this.pan.y,graph=this.graph;const screen=n=>({x:n.x*z+px,y:n.y*z+py,w:n.w*z,h:n.h*z});
  const sel=graph.map.get(this.selected),neighbors=new Set(sel?[sel.id]:[]);if(sel)for(const e of graph.edges)if(e.from===sel.id||e.to===sel.id){neighbors.add(e.from);neighbors.add(e.to)}
  for(const g of graph.groups){const s=screen(g);if(s.x+s.w<0||s.y+s.h<0||s.x>this.w||s.y>this.h)continue;c.strokeStyle=g.color+'25';c.lineWidth=1;c.fillStyle=g.color+'04';c.beginPath();c.roundRect(s.x,s.y,s.w,s.h,Math.min(8,s.w/4));c.fill();c.stroke();if(z>.09)this.text(g.label,s.x+8,s.y+17,g.color+'b0',12,Math.min(s.w-16,700));}
  for(const e of graph.edges){const a=e.a,b=e.b,sa=screen(a),sb=screen(b);if(Math.max(sa.x+sa.w,sb.x+sb.w)<0||Math.min(sa.x,sb.x)>this.w||Math.max(sa.y+sa.h,sb.y+sb.h)<0||Math.min(sa.y,sb.y)>this.h)continue;
   const highlight=sel&&(e.from===sel.id||e.to===sel.id);c.strokeStyle=highlight?(e.loop?'#ecad78':'#b8dc85'):e.loop?'#98735170':e.conditional?'#849aa07a':'#58778265';c.lineWidth=highlight?1.7:Math.max(.5,Math.min(1.1,z));c.setLineDash(e.loop?[5,5]:e.conditional?[3,4]:[]);c.beginPath();
   let x1=sa.x+sa.w/2,y1=sa.y+sa.h+3,x2=sb.x+sb.w/2,y2=sb.y-4;
   if(e.residual||e.loop){const side=Math.min(sa.x,sb.x)-Math.max(13,30*z);x1=sa.x;y1=sa.y+sa.h/2;x2=sb.x;y2=sb.y+sb.h/2;c.moveTo(x1,y1);c.bezierCurveTo(side,y1,side,y2,x2,y2)}
   else if(e.betweenLayers&&Math.abs(y1-y2)<sa.h*3){c.moveTo(x1,y1);c.bezierCurveTo(x1,y1+30,x2,y2-30,x2,y2)}
   else if(Math.abs(sa.x-sb.x)>Math.abs(sa.y-sb.y)*1.2){const right=sa.x<sb.x;x1=sa.x+(right?sa.w:0);x2=sb.x+(right?-4:sb.w+4);y1=sa.y+sa.h/2;y2=sb.y+sb.h/2;c.moveTo(x1,y1);const mid=(x1+x2)/2;c.bezierCurveTo(mid,y1,mid,y2,x2,y2);e.horizontalArrow=right?1:-1}
   else{c.moveTo(x1,y1);const mid=(y1+y2)/2;c.bezierCurveTo(x1,mid,x2,mid,x2,y2);e.horizontalArrow=0}c.stroke();c.setLineDash([]);
   if(z>.13&&!e.residual&&!e.loop){c.fillStyle=c.strokeStyle;this.path(e.horizontalArrow?[[x2-5*e.horizontalArrow,y2-3],[x2-5*e.horizontalArrow,y2+3],[x2,y2]]:[[x2-3,y2-5],[x2+3,y2-5],[x2,y2]],c.fillStyle)}
  }
  for(const n of graph.nodes){const s=screen(n);if(s.x+s.w< -30||s.y+s.h< -30||s.x>this.w+30||s.y>this.h+30)continue;const active=n.id===this.selected,hover=n.id===this.hover?.id,col=n.color;const alpha=sel&&!neighbors.has(n.id)?'22':'40';c.lineWidth=active?1.8:hover?1.4:1;
   if(s.w<6||s.h<3){c.fillStyle=active?'#dcffab':col+'b0';c.fillRect(s.x,s.y,Math.max(2,s.w),Math.max(2,s.h));continue}
   if(this.mode==='tensor'){
    const lift=Math.min(13,25*z),slant=lift*1.1;
    if(active){c.shadowBlur=22;c.shadowColor=col+'40'}
    this.path([[s.x,s.y],[s.x+slant,s.y-lift],[s.x+s.w+slant,s.y-lift],[s.x+s.w,s.y]],col+'38',active?col:col+'80');
    this.path([[s.x+s.w,s.y],[s.x+s.w+slant,s.y-lift],[s.x+s.w+slant,s.y+s.h-lift],[s.x+s.w,s.y+s.h]],col+'15',active?col:col+'60');
    this.path([[s.x,s.y],[s.x+s.w,s.y],[s.x+s.w,s.y+s.h],[s.x,s.y+s.h]],active?col+'45':col+alpha,active?col:col+'80');c.shadowBlur=0;
    const cols=n.type==='input'?8:n.type==='norm'?12:16,rows=n.type==='norm'?2:4;c.strokeStyle=col+'24';c.lineWidth=.5;if(s.w>24){for(let i=1;i<cols;i++){const xx=s.x+s.w*i/cols;c.beginPath();c.moveTo(xx,s.y);c.lineTo(xx,s.y+s.h);c.stroke()}for(let i=1;i<rows;i++){const yy=s.y+s.h*i/rows;c.beginPath();c.moveTo(s.x,yy);c.lineTo(s.x+s.w,yy);c.stroke()}}
    if(z>.18||active||hover){const font=Math.max(12,Math.min(15,z*16));const maxWidth=Math.max(95,s.w+20);this.text(n.label,s.x,s.y+s.h+font+4,active?'#e0f5b9':'#c1d2db',font,maxWidth);if(z>.55){this.text(n.shape.join(' × '),s.x,s.y+s.h+font+21,'#7896a5',12,maxWidth)}}
    if(n.stack&&n.depth&&z>.15){this.text('× '+n.depth,s.x+s.w-29,s.y-8,col,12,70)}
   }else{
    c.fillStyle=active?'#26372b':'#18242b';c.strokeStyle=active?col:col+'70';c.beginPath();c.roundRect(s.x,s.y,s.w,s.h,Math.min(6,s.w/8));c.fill();c.stroke();c.fillStyle=col;c.fillRect(s.x,s.y,Math.max(2,3*z),s.h);
    if(z>.28||active){this.text(n.label,s.x+Math.max(6,10*z),s.y+Math.max(15,27*z),active?'#dcf1b9':'#c8d7df',Math.max(12,Math.min(14,z*15)),Math.max(70,s.w-14));if(z>.6)this.text(n.shape.join(' × '),s.x+10*z,s.y+51*z,'#91a4ae',12,s.w-15)}
   }
   if(n.opaque&&z>.3){c.fillStyle='#b8b0a0';c.font='12px monospace';c.fillText('◇',s.x+s.w-13,s.y+15)}
  }
  if(graph.groups.length>5)this.drawMini();else this.minimap=null;
 }
 drawMini(){const c=this.ctx,b=this.graph.bounds;const w=112,h=70,x=this.w-w-22,y=18;this.minimap={x,y,w,h};c.fillStyle='#0b1215db';c.strokeStyle='#35474f';c.lineWidth=1;c.fillRect(x-5,y-5,w+10,h+10);c.strokeRect(x-5,y-5,w+10,h+10);const sx=w/b.w,sy=h/b.h;
  for(const g of this.graph.groups){c.fillStyle=g.color+'80';c.fillRect(x+(g.x-b.minX)*sx,y+(g.y-b.minY)*sy,Math.max(2,g.w*sx),Math.max(1,g.h*sy))}
  c.strokeStyle='#d0f696';const left=Math.max(x,x+((-this.pan.x/this.zoom)-b.minX)*sx),top=Math.max(y,y+((-this.pan.y/this.zoom)-b.minY)*sy),right=Math.min(x+w,x+(((this.w-this.pan.x)/this.zoom)-b.minX)*sx),bottom=Math.min(y+h,y+(((this.h-this.pan.y)/this.zoom)-b.minY)*sy);if(right>left&&bottom>top)c.strokeRect(left,top,right-left,bottom-top);
 }
 navigateMini(p){const mini=this.minimap,b=this.graph.bounds;const xx=b.minX+(p.x-mini.x)/mini.w*b.w,yy=b.minY+(p.y-mini.y)/mini.h*b.h;this.pan.x=this.w/2-xx*this.zoom;this.pan.y=this.h/2-yy*this.zoom;this.draw()}
 destroy(){this.resize.disconnect();if(this.raf)cancelAnimationFrame(this.raf)}
}
