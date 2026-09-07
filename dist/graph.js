import {palette} from './models.js';
const B='B',T='T',H='H',A='A';
const detail={
 norm:['按最后一维计算均方根，稳定激活尺度。缩放参数由训练学习。','rms = √(mean(x²) + ε)\ny = (x / rms) ⊙ γ'],
 rope:['在成对的 Q/K 通道上旋转，写入 token 的相对位置信息。V 不进行 RoPE。','[q₂ᵢ′, q₂ᵢ₊₁′]ᵀ = R(pθᵢ)[q₂ᵢ, q₂ᵢ₊₁]ᵀ'],
 score:['每个 query 与允许访问的 key 做点积。共享 KV 头按查询头分组广播。','S = QKᵀ / √dₖ'],
 mask:['因果模型屏蔽未来 token；视觉前缀和动作块按各模型自己的可见性规则处理。','S′ᵢⱼ = Sᵢⱼ if visible(i,j)\nS′ᵢⱼ = −∞ otherwise'],
 softmax:['沿 key 轴归一化。每个 query 对所有可见 key 的注意力权重之和为 1。','Pᵢⱼ = exp(S′ᵢⱼ − maxⱼ S′ᵢⱼ)\n       / Σₖ exp(S′ᵢₖ − maxⱼ S′ᵢⱼ)'],
 mix:['以注意力概率对 V 做加权求和，再合并所有查询头。','O = PV\nO_concat = concat(O₁, …, Oₙ)'],
 residual:['保留子层输入，并加入子层的增量更新。旁路边表示这条残差路径。','y = x + F(x)'],
 flow:['从高斯噪声动作块出发，在每个采样步重复调用速度场网络。openpi 的时间从 1 积分至 0。','a₁ ~ N(0, I)\na_(t+Δt) = a_t + Δt · vθ(a_t, t, o)\nΔt < 0; t : 1 → 0'],
 unknown:['已确认模块的职责；当前可视化未核实内部逐算子配置。保持不透明，不套用其他模型参数。','内部算子 / 张量维度：待配置核实'],
};
function spec(id,label,type,row,col,opts={}){return {id,label,type,row,col,shape:[],desc:'',formula:'',...opts}}
export function architecture(m){
 const d=m.d||'D_vlm',ad=m.ad||'D_action',isPi=['pi','fast','pipaper'].includes(m.kind),s=[];
 const add=(...a)=>{const v=spec(...a);s.push(v);return v};
 const stack=(id,label,row,col,deps,role='language')=>add(id,label,role==='vision'?'vision':role==='action'?'action':'attention',row,col,{deps,role,stack:true,depth:role==='vision'?m.visionLayers:role==='action'?m.alayers:m.layers,shape:role==='action'?[B,H,ad]:[B,role==='vision'?'P':T,role==='vision'?(m.vd||'D_vision'):d],desc:role==='action'?'根据观测条件和带噪动作估计动作速度场。每次去噪重复使用同一套网络权重。':role==='vision'?'将图像 patch 表征为视觉 token，图像内 token 通常使用双向注意力。':'把输入 token 更新为上下文相关表示，按模型的可见性掩码传播信息。',formula:role==='action'?'vθ = ActionExpert(a_t, t | context)':'h_(l+1) = Block_l(h_l)',opaque:!((role==='vision'?m.visionLayers:role==='action'?m.alayers:m.layers)),...{}});
 if(m.group==='LLM'){
  add('tokens','Text tokens','input',0,1,{shape:[B,T],desc:'分词器将文本映射为整数 token ID。T 表示本次前缀长度；不在浏览器中加载真实分词器。',formula:'ids = tokenizer(text)'});
  add('embed','Token embedding','input',1,1,{deps:['tokens'],shape:[B,T,d],formula:'X = E[ids]',desc:`查表将 token ID 转换为 ${d} 维向量。词表大小为 ${m.vocab.toLocaleString()}。`});
  if(m.kind==='hybrid'){
   add('image','Image / video','input',0,0,{shape:[B,'C',3,'h','w'],desc:'图像或视频输入；分辨率与 patch 网格由预处理决定。'});
   stack('vision','Vision encoder',1,0,['image'],'vision');
   add('merge','Visual patch merger','vision',2,0,{deps:['vision'],shape:[B,'P',d],formula:'h = MLP(merge_2×2(patches))',desc:'空间合并并映射到语言宽度。此模块按语义展示，未展开所有预处理算子。',opaque:true});
   add('fusion','Multimodal sequence','input',2,1,{deps:['embed','merge'],shape:[B,T,d],formula:'X = insert(visual, text)',desc:'在对应图像 / 视频 token 位置插入视觉特征。T 是合并后总序列长度。'});
  }
  stack('language',m.kind==='dsv4'?'mHC · hybrid MoE':m.kind==='mla'?'MLA · MoE blocks':m.kind==='hybrid'?'Hybrid decoder':'Transformer decoder',3,1,[m.kind==='hybrid'?'fusion':'embed']);
  add('finalnorm','Final RMSNorm','norm',4,1,{deps:['language'],shape:[B,T,d],desc:detail.norm[0],formula:detail.norm[1]});
  add('logits','Vocabulary projection','output',5,1,{deps:['finalnorm'],shape:[B,T,m.vocab],formula:'logits = h W_vocab',desc:'把隐藏状态投影到整个词表。下一个 token 通常只读取最后一个位置的 logits。'});
  add('sample','Next-token sampling','output',6,1,{deps:['logits'],shape:[B,1],loop:'tokens',formula:'p = softmax(logits_last / τ)\ntoken ~ sample(p)',desc:'根据温度与采样规则选择下一 token，追加到序列后继续生成。循环在图中保留为边。'});
 } else if(isPi){
  const paper=m.kind==='pipaper',newer=m.id==='pi07';
  add('image',newer?'Images + history':'Camera images','input',0,0,{shape:[B,'C',3,paper?448:224,paper?448:224],desc:newer?'当前图像、最多 6 帧历史观测以及子目标图像进入历史视觉编码器。':'多相机图像先缩放和归一化；各相机共享视觉编码器权重。'});
  add('text','Language + state','input',0,1,{shape:[B,T],desc:m.id==='pi0'?'自然语言任务描述。π₀ 的状态另经线性层进入动作专家。':newer?'语言、metadata 与线性投影后的本体状态；π₀.₇ 不沿用 π₀.₆ 的状态离散化方式。':'语言任务、状态 token 与可用元数据构成文本条件。'});
  add('noise',m.kind==='fast'?'Previous action tokens':'Action noise + time','input',0,2,{shape:[B,H,A],desc:m.kind==='fast'?'自回归动作 token 前缀；推理没有连续动作去噪过程。':'高斯动作噪声与采样时间作为动作专家的输入。',formula:m.kind==='fast'?'u_<k':'a_t ∈ R^(B×H×A), t ∈ [0,1]'});
  stack('vision',newer?'MEM history encoder':'SigLIP vision encoder',1,0,['image'],'vision');
  add('embed','Token embedding','input',1,1,{deps:['text'],shape:[B,T,d],formula:'X_text = E[ids]',desc:'将语言与已编码的文本条件转换为向量。'});
  add('project','Visual projection','vision',2,0,{deps:['vision'],shape:[B,'P',d],formula:'X_img = W_img · h_vision',desc:'将视觉 token 对齐到视觉语言骨干的隐藏维度。'});
  add('prefix','Observation prefix','input',2,1,{deps:['project','embed'],shape:[B,T,d],formula:'X_prefix = concat(X_img, X_text)',desc:'组成视觉语言前缀，预先计算其各层 K/V；动作 token 读取前缀，而前缀不读取未来动作。'});
  const p=stack('language',paper?'Gemma 3 VLM':'PaliGemma · Gemma 2B',3,1,['prefix']);p.pi=true;
  if(m.kind==='fast'){
   p.deps.push('noise');
   add('actiontokens','Autoregressive action tokens','output',4,1,{deps:['language'],shape:[B,'N_fast'],desc:'语言模型逐个预测压缩后的动作 token，直到结束标记。',formula:'p(u_k | u_<k, observation)'});
   add('fastdecode','FAST decode','action',5,1,{deps:['actiontokens'],shape:[B,H,A],ops:'fast',desc:'BPE 解码、反量化，再通过逆 DCT 恢复连续动作轨迹。',formula:'a = IDCT(dequantize(BPE⁻¹(tokens)))'});
   add('actions','Robot action chunk','output',6,1,{deps:['fastdecode'],shape:[B,H,A],desc:'反归一化并映射到具体机器人的动作接口。'});
  }else{
   add('actionin','Action / time embedding','action',2,2,{deps:['noise'],shape:[B,H,ad],ops:'actionin',desc:m.id==='pi0'?'π₀ 使用动作与时间拼接后的 MLP；状态投影形成额外 token。':'动作线性投影；时间通过 MLP 编码，并调制动作专家中的 adaptive RMSNorm。',formula:'h_a = Linear(a_t)\nc_t = MLP(sinusoidal(t))'});
   const a=stack('action',paper?'Action expert · 860M':'Action expert · 300M',3,2,['actionin','language'],'action');a.pi=true;
   add('velocity','Action velocity projection','action',4,2,{deps:['action'],shape:[B,H,A],desc:'最后一层动作特征线性投影到填充动作维度，得到速度场 vθ。',formula:'vθ = W_out h_action + b_out'});
   add('integrate','Flow integration','action',5,2,{deps:['velocity','noise'],shape:[B,H,A],desc:detail.flow[0],formula:detail.flow[1],loop:'actionin'});
   add('actions','Robot action chunk','output',6,2,{deps:['integrate'],shape:[B,H,A],desc:'去噪得到连续动作块，去除 padding，反归一化并交给机器人执行器。'});
   if(m.id==='pi0')add('state','Proprioceptive state','input',1,2,{shape:[B,A],desc:'关节、夹爪等本体状态经线性映射后作为动作专家的状态 token。',to:'actionin'});
  }
 }else if(m.kind==='groot'){
  add('image','Camera observations','input',0,0,{shape:[B,'C',3,'h','w'],desc:'来自机器人相机的当前观测。'});
  add('text','Language instruction','input',0,1,{shape:[B,T],desc:'自然语言任务条件。'});
  add('state','State + embodiment ID','input',0,2,{shape:[B,'S'],desc:'本体状态和机器人类型决定状态 / 动作映射。'});
  add('vlm',m.id==='groot15'?'Eagle VLM backbone':m.id==='groot17'?'Cosmos-Reason2-2B':'Cosmos VLM backbone','vision',1,0,{deps:['image','text'],shape:[B,T,'D_vlm'],opaque:true,desc:'视觉语言骨干提取上下文语义特征；具体截取层取决于发布配置。',formula:'c_vl = VLM(images, instruction)'});
  add('stateenc','Embodiment state encoder','control',1,2,{deps:['state'],shape:[B,1,'D_action'],desc:'将本体状态映射为动作头的条件 token。',formula:'c_s = MLP_e(state)'});
  add('noise','Action noise + timestep','input',2,2,{shape:[B,H,A],desc:'动作块噪声、时间和位置编码进入动作头。'});
  if(m.id==='groot15')add('vladapter','VL attention adapter × 4','attention',2,0,{deps:['vlm'],shape:[B,T,2048],opaque:true,desc:'N1.5 配置中的 4 层视觉语言自注意力适配器；当前保留为语义模块。'});
  const a=stack('action','DiT action transformer',3,1,[m.id==='groot15'?'vladapter':'vlm','stateenc','noise'],'action');a.dit=true;
  add('decode','Embodiment action decoder','action',4,1,{deps:['action'],shape:[B,H,A],formula:'v_a = MLP_e(h_action)',desc:'从共享动作表征映射回特定机器人的动作空间。'});
  add('integrate','Flow matching sampler','action',5,1,{deps:['decode','noise'],shape:[B,H,A],loop:'action',formula:'a_(t+Δt) = a_t + Δt·vθ',desc:'按 checkpoint 的时间约定与采样器迭代更新动作。重复共享同一动作头参数。'});
  add('actions','Robot action chunk','output',6,1,{deps:['integrate'],shape:[B,H,A],desc:'连续动作块。接入 SONIC 时还需要具身动作映射，而不是直接输出电机电流。'});
 }else if(m.kind==='openvla'){
  add('image','RGB observation','input',0,0,{shape:[B,3,224,224],desc:'原始 OpenVLA 的单帧视觉输入。'});
  add('text','Language instruction','input',0,2,{shape:[B,T],desc:'自然语言任务经 Llama 分词器编码。'});
  add('dino','DINOv2 encoder','vision',1,0,{deps:['image'],shape:[B,'P','D_dino'],opaque:true,desc:'提取空间视觉表征。'});
  add('siglip','SigLIP encoder','vision',1,1,{deps:['image'],shape:[B,'P','D_siglip'],opaque:true,desc:'提取语义视觉表征，与 DINOv2 特征融合。'});
  add('fusion','Feature fusion + projector','vision',2,1,{deps:['dino','siglip'],shape:[B,'P',4096],formula:'X_img = MLP(concat(f_dino, f_siglip))',desc:'沿通道拼接两种视觉特征，再映射到语言模型空间。'});
  stack('language','Llama 2 · 7B decoder',3,1,['fusion','text']);
  add('actiontokens','Discrete action tokens','action',4,1,{deps:['language'],shape:[B,7],desc:'逐维自回归预测 7 个动作 token。',formula:'u_i = sample(p(u_i | u_<i, image, text))'});
  add('actions','Unbin + unnormalize','output',5,1,{deps:['actiontokens'],shape:[B,7],desc:'将离散 bins 映射回连续动作，按数据集统计反归一化。原版为单步动作。',formula:'a = unnormalize(bin_centers[token_ids])'});
 }else if(m.kind==='wam'){
  add('image','Observation history','input',0,0,{shape:[B,'F',3,'h','w'],desc:'历史视觉观测提供当前世界状态。'});
  add('text','Language instruction','input',0,1,{shape:[B,T],desc:'语言条件描述目标行为。'});
  add('state','Robot state','input',0,2,{shape:[B,'S'],desc:'机器人状态经过具身相关映射。'});
  add('vae','Video VAE encoder','vision',1,0,{deps:['image'],shape:[B,'F_lat','C_lat','h_lat','w_lat'],opaque:true,desc:'将视频压缩为空间与时间潜变量，降低联合生成成本。',formula:'z_obs = VAE.encode(video)'});
  add('textenc','Text encoder','attention',1,1,{deps:['text'],shape:[B,T,'D_text'],opaque:true,desc:'文本编码为扩散骨干的条件表征。'});
  add('noise','Noisy video + actions','input',2,1,{deps:['vae','state'],shape:[B,'N_joint','D'],desc:'未来视频潜变量与动作噪声进入联合生成过程。',formula:'z_t, a_t ~ noise schedule'});
  add('joint','Joint video–action DiT','attention',3,1,{deps:['noise','textenc','vae','state'],shape:[B,'N_joint','D'],opaque:true,ops:'wam',desc:'视频与动作在同一个生成模型中耦合更新；以视频预测学习物理动态。',formula:'(v_z, v_a) = fθ(z_t, a_t, t | obs, text)',loop:'noise'});
  add('videodecode','Video VAE decoder','vision',4,0,{deps:['joint'],shape:[B,'F_future',3,'h','w'],opaque:true,desc:'恢复预测未来画面。未来画面和动作由联合模型产生。'});
  add('actions','Action chunk decoder','action',4,2,{deps:['joint'],shape:[B,H,A],desc:'将动作表征映射为连续机器人动作。实际执行一段动作后采集新观测并重规划。'});
  add('future','Predicted future world','output',5,0,{deps:['videodecode'],shape:[B,'F_future',3,'h','w'],desc:'预测的视觉未来，不等于真实世界观测。'});
  add('execute','Closed-loop execution','control',5,2,{deps:['actions'],shape:[B,H,A],desc:'执行短动作段、读取新观测，继续下一个联合预测周期。'});
 }else if(m.kind==='sonic'){
  add('reference','Reference motion','input',0,0,{shape:[B,'F_ref','D_motion'],desc:'来自机器人动作、人类姿态或混合运动指令；由规划器 / 遥操作接口提供。'});
  add('proprio','Robot proprioception','input',0,2,{shape:[B,'D_obs'],desc:'关节位置、速度、机身姿态等状态按 observation config 组织。'});
  add('motionenc','Specialized motion encoders','control',1,0,{deps:['reference'],shape:[B,'D_latent'],opaque:true,desc:'不同运动输入使用对应编码器，映射到通用潜空间。',formula:'z = Encoder_modality(reference)'});
  add('normalize','Observation normalization','norm',1,2,{deps:['proprio'],shape:[B,'D_obs'],desc:'状态组织、尺度与历史拼接由所选发布版本定义。',formula:'o = normalize(observation)'});
  add('latent','Universal motion token','attention',2,0,{deps:['motionenc'],shape:[B,'D_latent'],desc:'共享运动表征使同一控制策略接收多种运动来源。',formula:'z ∈ Z_motion'});
  add('decoder','Robot control decoder','control',3,1,{deps:['latent','normalize'],shape:[B,29],opaque:true,desc:'控制解码器同时接收运动潜变量和本体状态，输出用于构造关节目标的动作。',formula:'u = Decoder(z, o)'});
  add('target','Joint target mapping','action',4,1,{deps:['decoder'],shape:[B,29],desc:'根据配置把策略动作缩放并加入默认关节位置，形成关节目标。',formula:'q_des = q_default + scale ⊙ u'});
  add('pd','Joint PD controller','control',5,1,{deps:['target','proprio'],shape:[B,29],external:true,desc:'模型外的底层关节控制环节。这里显示常见位置 PD 关系，实际 SDK 还可叠加前馈项和限幅。',formula:'τ = Kp⊙(q_des−q) + Kd⊙(q̇_des−q̇) + τ_ff'});
  add('robot','Robot dynamics','output',6,1,{deps:['pd'],shape:[B,29],external:true,desc:'执行器与机体动力学产生新状态，并反馈给控制循环。',formula:'(q, q̇)_(t+1) = dynamics(state, τ)',loop:'proprio'});
 }
 // Patch embedding and terminal norms are explicit outside repeated block stacks.
 const vis=s.find(n=>n.id==='vision'&&n.depth);
 if(vis){for(const n of s)if(n.row>=vis.row)n.row+=1;add('patches','Patch + position embedding','vision',vis.row-1,vis.col,{deps:vis.deps,shape:[B,'P',m.vd||'D_vision'],formula:m.kind==='hybrid'?'patches = Conv3D(video) + position':'patches = Conv2D(images) + position',desc:m.kind==='hybrid'?'Qwen3.5 采用时空 patch 嵌入，空间 patch 为 16、时间 patch 为 2。':'图像分块后线性映射，加入可学习位置编码；SigLIP 使用 14×14 patch。'});vis.deps=['patches'];}
 const act=s.find(n=>n.id==='action');const vel=s.find(n=>n.id==='velocity'||n.id==='decode');
 if(act&&vel){for(const n of s)if(n.row>=vel.row)n.row+=1;add('actionnorm',m.kind==='groot'?'DiT final norm + projection':'Action final RMSNorm','norm',vel.row-1,vel.col,{deps:['action'],shape:[B,H,m.kind==='groot'?1024:(m.ad||'D_action')],formula:m.kind==='groot'?'y = Linear(LayerNorm(h) ⊙ (1+s(t)) + b(t))':'y = RMSNorm(h_action)[:, −H:, :]',desc:m.kind==='groot'?'输出层先以时间条件调制 LayerNorm，再投影至动作解码器输入维度。':'对动作专家输出进行最终归一化，并只取最后 H 个动作 token。'});vel.deps=['actionnorm'];}
 for(const n of s){n.deps??=[];if(n.to){s.find(x=>x.id===n.to)?.deps.push(n.id)}}
 return s;
}

export function buildGraph(m,level=0,focus={module:'language',layer:0}){
 const nodes=[],edges=[],groups=[];let seq=0;
 function N(id,label,type,x,y,o={}){const n={id,label,type,x,y,w:200,h:76,shape:[],desc:'',formula:'',order:seq++,color:palette[type]||palette.unknown,...o};nodes.push(n);return n}
 function E(from,to,o={}){edges.push({from:typeof from==='string'?from:from.id,to:typeof to==='string'?to:to.id,...o})}
 const arch=architecture(m);
 const hasDetailedStack=arch.some(s=>s.stack&&s.depth);
 if(level===0||!hasDetailedStack){
  let positions;
  if(['pi','pipaper','fast'].includes(m.kind))positions={image:[0,0],patches:[1,0],vision:[2,0],project:[3,0],text:[0,1],embed:[1,1],prefix:[3,1],language:[4,1],noise:[0,2],state:[0,3],actionin:[1,2],action:[4,2],actionnorm:[4,3],velocity:[3,3],integrate:[2,3],actions:[1,3],actiontokens:[4,2],fastdecode:[3,3]};
  else if(m.kind==='groot')positions={image:[0,0],text:[0,1],vlm:[1,0],state:[0,2],stateenc:[1,2],noise:[1,1],vladapter:[2,0],action:[3,1],actionnorm:[3,2],decode:[2,2],integrate:[2,3],actions:[3,3]};
  else if(m.kind==='wam')positions={image:[0,0],text:[0,1],state:[0,2],vae:[1,0],textenc:[1,1],noise:[1,2],joint:[2,1],videodecode:[3,0],actions:[3,2],future:[4,0],execute:[4,2]};
  else if(m.kind==='sonic')positions={reference:[0,0],proprio:[0,2],motionenc:[1,0],normalize:[1,2],latent:[2,0],decoder:[2,1],target:[3,1],pd:[3,2],robot:[2,2]};
  else if(m.kind==='openvla')positions={image:[0,0],text:[0,2],dino:[1,0],siglip:[1,1],fusion:[2,0],language:[2,2],actiontokens:[3,2],actions:[3,1]};
  else if(m.kind==='hybrid')positions={image:[0,0],patches:[1,0],vision:[2,0],merge:[3,0],tokens:[0,2],embed:[1,2],fusion:[3,1],language:[4,1],finalnorm:[4,2],logits:[3,2],sample:[2,2]};
  else positions={tokens:[0,0],embed:[1,0],language:[2,0],finalnorm:[2,1],logits:[1,1],sample:[0,1]};
  for(const [i,s] of arch.entries()){const pos=positions[s.id]||[i%4,Math.floor(i/4)];N(s.id,s.label,s.type,pos[0]*320,pos[1]*185,{...s,w:230,h:78,module:s.id,layer:0})}
  for(const s of arch){for(const d of s.deps)E(d,s.id,{conditional:s.id==='action'&&d==='language'});if(s.loop)E(s.id,s.loop,{loop:true})}
  return finalize({modeNote:hasDetailedStack?'架构全景 · 点击模块探索':'当前为论文 / 官方模块图 · 内部配置尚未全部导入'});
 }
 function ops(s,idx,x,y){
  if(s.dit)return ditOps(s,idx,x,y);
  const id=s.id+'.'+idx+'.',role=s.role||s.ops;const isVision=role==='vision',action=role==='action';
  const d=isVision?(m.vd||'D_vision'):action?(m.ad||'D_action'):(m.d||'D');
  const ff=isVision?(m.vff||'F_vision'):action?(m.aff||'F_action'):(idx<(m.denseFirst||0)?m.ff:(m.moeff||m.ff||'F'));
  const heads=isVision?(m.vheads||'N_h'):(m.heads||'N_h'),hd=isVision?(m.vhd||'d_h'):(m.hd||'d_h');
  const kv=isVision?heads:(m.kv||'N_kv');const t=isVision?'P':action?(m.id==='pi0'?'H+1':H):T;const shape=[B,t,d];
  const local=[];const n=(k,l,ty,c,r,o={})=>{const z=N(id+k,l,ty,x+c*230,y+r*112,{module:s.id,layer:idx,shape,role, ...o});local.push(z);return z};
  const link=(...z)=>{for(let i=1;i<z.length;i++)E(z[i-1],z[i])};
  const norm=n('norm',action&&m.id!=='pi0'?'Adaptive RMSNorm':isVision?'LayerNorm':'RMSNorm','norm',1,0,{desc:isVision?'按通道减均值、除标准差，再进行可学习仿射变换。':detail.norm[0],formula:isVision?'y = (x−mean(x)) / √(var(x)+ε) ⊙ γ + β':action&&m.id!=='pi0'?'y = RMSNorm(x) ⊙ (1+s(t)) + b(t)':detail.norm[1]});
  let out,kvport,attention;
  const linear=m.kind==='hybrid'&&!isVision&&(idx+1)%4!==0;
  const mla=m.kind==='mla'&&!isVision;
  const v4=m.kind==='dsv4'&&!isVision;
  if(linear){
   const p=n('project','Q / K / V / gates','attention',1,1,{desc:'线性投影产生 Q、K、V 以及衰减和更新门；线性注意力不构建 T×T 的 softmax 矩阵。',formula:'q,k,v,g,β = projections(x)'});
   const conv=n('conv','Causal depthwise conv','attention',1,2,{desc:'局部因果卷积融合相邻 token；Qwen3.5-9B 的卷积核长度为 4。',formula:'q,k,v = SiLU(DWConv1D(q,k,v))'});
   const delta=n('delta','Gated DeltaNet recurrence','attention',1,3,{opaque:true,desc:'门控 delta 更新维护固定维度状态。以下公式是递推形式的语义表达，未展开分块并行内核。',formula:'S̄_t = α_t S_(t−1)\nS_t = S̄_t + β_t k_t(v_t−k_tᵀS̄_t)ᵀ\no_t = q_tᵀ S_t'});
   attention=n('gate','Gated output normalization','attention',1,4,{desc:'对线性注意力输出归一化并施加输出门。',formula:'o = norm(o) ⊙ SiLU(g)'});link(norm,p,conv,delta,attention);
  }else if(v4){
   const hc=n('mhcpre','mHC pre / mixing','norm',0,0,{opaque:true,desc:'4 路残差流经受流形约束的连接混合；不是普通单流相加。',formula:'h = mHC_pre(X), X ∈ R^(B×T×4×D)'});E(hc,norm);
   const q=n('q','Low-rank Q projection','attention',0,1,{desc:'Q 使用低秩投影，配置 q_lora_rank = 1536。',formula:'q = W_up RMSNorm(W_down x)'});
   const comp=n('compress',idx===60?'Sliding-window KV':idx<2||idx%2===1?'HCA · compression 128':'CSA · compression 4','attention',2,1,{opaque:true,desc:'按层使用高压缩、稀疏压缩或末层局部注意力，配置窗口为 128。压缩器内部保持语义块。',formula:'K_c,V_c = compress(K,V)'});
   const select=n('select','Sparse / compressed context','attention',1,2,{opaque:true,desc:'CSA 使用索引器选取压缩上下文；HCA 使用高压缩上下文并结合局部窗口。',formula:'context = select(K_c,V_c) ∪ local_window'});
   attention=n('attn','Compressed attention','attention',1,3,{desc:'对所选局部 / 压缩上下文计算注意力；仅显示等价数学关系。',formula:'o = softmax(q K_contextᵀ / √d) V_context'});E(norm,q);E(norm,comp);E(comp,select);E(q,attention);E(select,attention);kvport=comp;
  }else{
   let q,k,v;
   if(mla){
    const cq=n('cq','Q low-rank · 1536','attention',0,1,{shape:[B,t,1536],formula:'c_q = RMSNorm(x W_DQ)',desc:'压缩并归一化 Q 的低秩中间表示。'});
    const ckv=n('ckv','KV latent · 512','attention',2,1,{shape:[B,t,512],formula:'c_kv = RMSNorm(x W_DKV)',desc:'缓存共享的低秩 KV 潜变量，避免为每个头缓存完整 K/V。'});link(norm,cq);link(norm,ckv);
    q=n('q','Q content + RoPE','attention',0,2,{shape:[B,heads,t,192],formula:'q = concat(W_UQ c_q, q_rope)',desc:'内容维 128 与位置维 64 组成 192 维 Q。'});
    k=n('k','K content + RoPE','attention',1,2,{shape:[B,heads,t,192],formula:'k = concat(W_UK c_kv, k_rope)',desc:'内容 K 由潜变量恢复，位置 K 使用独立分支。',opaque:true});
    v=n('v','V up-projection','attention',2,2,{shape:[B,heads,t,128],formula:'v = W_UV c_kv',desc:'每个头恢复 128 维 value。'});E(cq,q);E(ckv,k);E(norm,k);E(ckv,v);kvport=ckv;
   }else{
    q=n('q','Query projection','attention',0,1,{shape:[B,heads,t,hd],formula:'Q = reshape(x W_Q)',desc:`投影为 ${heads} 个查询头，每头 ${hd} 维。`});
    k=n('k','Key projection','attention',1,1,{shape:[B,kv,t,hd],formula:'K = reshape(x W_K)',desc:`使用 ${kv} 个 KV 头，后续按查询头分组广播。`});
    v=n('v','Value projection','attention',2,1,{shape:[B,kv,t,hd],formula:'V = reshape(x W_V)',desc:'value 作为注意力加权汇聚的信息源。'});E(norm,q);E(norm,k);E(norm,v);kvport=k;
    if(m.qknorm&&!isVision){const qn=n('qnorm','Q head RMSNorm','norm',0,2,{shape:q.shape,formula:detail.norm[1],desc:'Qwen3 在每个 Q 头的最后一维执行 RMSNorm。'}),kn=n('knorm','K head RMSNorm','norm',1,2,{shape:k.shape,formula:detail.norm[1],desc:'Qwen3 对 K 同样做头内 RMSNorm。'});E(q,qn);E(k,kn);q=qn;k=kn;}
    if(!isVision){const qr=n('qrope',m.kind==='hybrid'?'Q partial M-RoPE':'Q rotary position','attention',0,3,{shape:q.shape,formula:detail.rope[1],desc:detail.rope[0]}),kr=n('krope',m.kind==='hybrid'?'K partial M-RoPE':'K rotary position','attention',1,3,{shape:k.shape,formula:detail.rope[1],desc:detail.rope[0]});E(q,qr);E(k,kr);q=qr;k=kr;}
   }
   const keyLen=action&&s.pi?(m.id==='pi0'?'T+H+1':'T+H'):t;
   const score=n('score','Q × Kᵀ / √d','attention',1,4,{shape:[B,heads,t,keyLen],desc:detail.score[0],formula:detail.score[1],lab:true});E(q,score);E(k,score);
   const mask=n('mask',isVision?'Bidirectional attention':action?'Action visibility mask':'Attention mask','attention',1,5,{shape:score.shape,formula:detail.mask[1],desc:isVision?'图像 patch 之间双向可见。':action&&s.pi?'动作 token 之间双向可见，并可访问全部观测前缀 K/V；前缀不反向读取动作。':s.pi?'观测前缀可见性按 openpi 的 block mask 组织。':detail.mask[0]});
   const soft=n('softmax','Stable softmax','attention',1,6,{shape:score.shape,desc:detail.softmax[0],formula:detail.softmax[1],lab:true});
   attention=n('mix','Attention × Value','attention',1,7,{shape:[B,t,heads,mla?128:hd],desc:detail.mix[0],formula:detail.mix[1],lab:true});link(score,mask,soft,attention);E(v,attention);
   if(action&&s.pi){const cache=n('prefixkv','Prefix K / V cache','input',2,4,{shape:[B,kv,T,hd],desc:'同层语言专家的 K/V 与动作专家的 K/V 沿序列轴拼接。两个专家宽度不同，但注意力头维度相同。',formula:'K = concat(K_prefix, K_action)\nV = concat(V_prefix, V_action)'});E(cache,score,{conditional:true});E(cache,attention,{conditional:true});}
  }
  let r=linear?5:v4?4:8;
  out=n('out','Attention output projection','attention',1,r++,{desc:'合并查询头后线性映射回残差流宽度。',formula:'y = concat(heads) W_O'});E(attention,out);
  if(m.kind==='hybrid'&&!isVision&&!linear){const gate=n('outgate','Attention output gate','attention',2,r-1,{desc:'Qwen3.5 全注意力使用独立输出门；图中以语义算子表示。',formula:'y = y ⊙ sigmoid(g)'});E(out,gate);out=gate;}
  const adaptive=action&&s.pi&&m.id!=='pi0';
  const res=n('res1',v4?'mHC post-attention':adaptive?'Gated attention residual':'Attention residual','norm',1,r++,{desc:v4?'mHC 合并子层结果与多路残差流，受连接约束。':adaptive?'adaptive RMSNorm 同时给出时间相关残差门。':detail.residual[0],formula:v4?'X′ = mHC_post(X, Attention(h))':adaptive?'y = x + gate_attn(t) ⊙ F(x)':detail.residual[1],opaque:v4});E(out,res);E(norm,res,{residual:true});
  const post=n('postnorm',isVision?'Post-attention LayerNorm':adaptive?'Adaptive FFN RMSNorm':'Post-attention RMSNorm','norm',1,r++,{desc:isVision?'在视觉 MLP 前按通道执行 LayerNorm。':detail.norm[0],formula:isVision?'y = LayerNorm(x)':adaptive?'y = RMSNorm(x) ⊙ (1+s_ff(t)) + b_ff(t)':detail.norm[1]});E(res,post);
  const moe=!!m.experts&&!isVision&&!action&&idx>=(m.denseFirst||0);
  let pre=post;
  if(moe){const router=n('router',m.kind==='dsv4'&&idx<3?'Hash expert routing':`Router · top ${m.topk} / ${m.experts}`,'mlp',1,r++,{shape:[B,t,m.experts],desc:`每个 token 激活 ${m.topk} 个路由专家。${m.shared?'另有 '+m.shared+' 个始终参与的共享专家。':''} 以下使用专家维 E 批量表示全部专家，不伪造激活结果。`,formula:m.kind==='dsv4'&&idx<3?'I = hash_route(token_id)':'s = router_score(x W_router)\nI = TopK(selection_score(s))\ny = Σ_(e∈I) g_e Expert_e(x)'});E(post,router);pre=router;}
  if(isVision){const up=n('up','MLP up projection','mlp',1,r++,{shape:[B,t,ff],formula:'u = x W_up + b_up',desc:'视觉前馈网络扩展通道宽度。'}),act=n('act','GELU activation','mlp',1,r++,{shape:up.shape,formula:'GELU(x) = x Φ(x)',desc:'逐元素平滑激活函数。'}),down=n('down','MLP down projection','mlp',1,r++,{formula:'y = GELU(u) W_down + b_down',desc:'投影回视觉隐藏维度。'});link(pre,up,act,down);pre=down;}
  else{
   const es=moe?['E',B,t,ff]:[B,t,ff],prefix=moe?'Batched expert ':'';
   const gate=n('gateproj',prefix+'gate','mlp',0,r,{shape:es,formula:'g = x W_gate',desc:moe?`包含全部 ${m.experts} 个专家的独立 gate 权重；E 为逻辑专家轴，真实执行只分发激活 token。`:'门控投影。'}),up=n('up',prefix+'up','mlp',2,r++,{shape:es,formula:'u = x W_up',desc:'与门控投影并行，形成前馈信息支路。'});E(pre,gate);E(pre,up);
   const act=n('act',s.pi?'GELU gate':'SiLU gate','mlp',0,r++,{shape:es,formula:s.pi?'g′ = GELU(g)':'g′ = g · sigmoid(g)',desc:s.pi?'openpi Gemma 前馈使用 GELU 门控。':'对门控分支逐元素施加 SiLU。'});E(gate,act);
   const mul=n('mul','Gated product','mlp',1,r++,{shape:es,formula:s.pi?'z = GELU(g) ⊙ u':'z = SiLU(g) ⊙ u',desc:'逐元素门控乘法，两个投影分支在此合并。'});E(act,mul);E(up,mul);
   const down=n('down',prefix+'down','mlp',1,r++,{shape:moe?['E',B,t,d]:shape,formula:'y_e = z_e W_down,e',desc:'投影回残差流宽度。'});E(mul,down);pre=down;
   if(moe){const combine=n('combine','Expert weighted sum','mlp',1,r++,{formula:m.shared?'y = Σ_(e∈TopK) g_e y_e + y_shared':'y = Σ_(e∈TopK) g_e y_e',desc:'仅累加被选专家的加权结果，加入共享专家输出（若有）；E 轴在此约简。'});E(down,combine);if(m.shared){const sh=n('shared','Shared expert SwiGLU','mlp',2,r-1,{desc:'所有 token 都执行共享专家。相同 gate / up / activation / multiply / down 结构以数学表达式表示。',formula:'y_shared = [SiLU(x W_g) ⊙ (x W_u)] W_d',opaque:true});E(post,sh);E(sh,combine);}pre=combine;}
  }
  const end=n('res2',v4?'mHC post-FFN':adaptive?'Gated FFN residual':'Feed-forward residual','norm',1,r++,{desc:v4?'再次通过受约束的超连接整合前馈层输出。':detail.residual[0],formula:v4?'X_next = mHC_post(X′, FFN(h′))':adaptive?'y = x′ + gate_ff(t) ⊙ FFN(x′)':detail.residual[1],opaque:v4});E(pre,end);E(res,end,{residual:true});
  const height=r*112+32;groups.push({id:s.id+'.'+idx,label:`${s.label} / layer ${String(idx+1).padStart(2,'0')}`,x:x-20,y:y-45,w:700,h:height+65,color:palette[s.type]});
  return {first:local.find(n=>n.id.endsWith('.mhcpre'))||local[0],last:end,w:720,h:height+85,kv:kvport||local[0]};
 }
 function ditOps(s,idx,x,y){
  const id=s.id+'.'+idx+'.',d=m.ad||'D_DiT',nh=m.aheads||'N_h',hd=m.ahd||'d_h',cross=idx%2===0,t='N_action';let row=0;
  const shape=[B,t,d],local=[];const n=(key,label,type,c,r,opts={})=>{const a=N(id+key,label,type,x+c*230,y+r*112,{module:s.id,layer:idx,role:'action',shape,...opts});local.push(a);return a};
  const input=n('input','Raw action tokens','input',1,row++,{desc:'动作、状态等 token 的序列。N_action 包含动作块及配置定义的条件 token，不直接等于 H。'});
  const time=n('time','Time modulation','action',2,0,{shape:[B,d],formula:'scale, shift = Linear(SiLU(time_embedding))',desc:'时间编码生成逐通道的缩放与偏置。'});
  const norm=n('norm','Adaptive LayerNorm','norm',1,row++,{formula:'h = LayerNorm(x) ⊙ (1+scale) + shift',desc:'GR00T DiT 使用 LayerNorm 时间调制，与 π 的 adaptive RMSNorm 不同。'});E(input,norm);E(time,norm);
  let context=norm;
  if(cross){context=n('context',m.id==='groot17'?(idx%4===0?'Non-image VLM context':'Image VLM context'):'VLM context','vision',2,row,{shape:[B,'N_context',2048],desc:m.id==='groot17'?'该 checkpoint 的 AlternateVLDiT 交替读取非图像与图像上下文；依据 image_mask 筛选可见 token。':'该层从视觉语言特征中构造 K/V。',formula:'c = masked VLM features'});}
  const q=n('q','Query projection','attention',0,row,{shape:[B,nh,t,hd],formula:'Q = reshape(h W_Q)',desc:'Q 始终来自当前动作隐藏状态。'}),k=n('k',cross?'Context key projection':'Self key projection','attention',1,row,{shape:[B,nh,cross?'N_context':t,hd],formula:'K = reshape(c W_K)',desc:cross?'K 来自 VLM 上下文。':'K 来自动作隐藏状态。'}),v=n('v',cross?'Context value projection':'Self value projection','attention',2,row+1,{shape:k.shape,formula:'V = reshape(c W_V)',desc:cross?'V 来自 VLM 上下文。':'V 来自动作隐藏状态。'});if(cross){context.x=x+460;context.y=y+(row-1)*112;}
  E(norm,q);E(context,k);E(context,v);row+=2;
  const score=n('score','Scaled attention scores','attention',1,row++,{shape:[B,nh,t,cross?'N_context':t],formula:'S = QKᵀ / √dₖ',desc:cross?'动作 query 与 VLM key 计算相似度。':'动作 token 之间使用双向自注意力。',lab:true});E(q,score);E(k,score);
  const mask=n('mask',cross?'Context visibility mask':'Bidirectional mask','attention',1,row++,{shape:score.shape,formula:'S′ = S + visibility_mask',desc:'仅保留此层允许访问的上下文；动作自注意力不使用语言模型的因果掩码。'});E(score,mask);
  const soft=n('softmax','Stable softmax','attention',1,row++,{shape:score.shape,formula:detail.softmax[1],desc:detail.softmax[0],lab:true});E(mask,soft);
  const mix=n('mix','Probability × Value','attention',1,row++,{formula:'o = concat(softmax(S′) V)',desc:detail.mix[0],lab:true});E(soft,mix);E(v,mix);
  const out=n('out','Attention output projection','attention',1,row++,{formula:'o = o W_O + b_O',desc:'映射回 DiT 隐藏宽度。'});E(mix,out);
  const res=n('res1','Attention residual','norm',1,row++,{formula:'x′ = x + o',desc:detail.residual[0]});E(out,res);E(input,res,{residual:true});
  const norm2=n('norm2','Feed-forward LayerNorm','norm',1,row++,{formula:'h = LayerNorm(x′)',desc:'前馈子层采用 LayerNorm。'});E(res,norm2);
  const up=n('up','MLP expansion','mlp',1,row++,{shape:[B,t,m.aff||'F_DiT'],formula:'u = h W_up + b_up',desc:'当前 DiT 实现默认前馈扩展后使用近似 GELU。'});E(norm2,up);
  const gelu=n('gelu','Approximate GELU','mlp',1,row++,{shape:up.shape,formula:'GELU(x) ≈ 0.5x(1+tanh(√(2/π)(x+0.044715x³)))',desc:'GR00T 此处是 GELU 前馈，不能套用 Llama 的 SwiGLU。'});E(up,gelu);
  const down=n('down','MLP output projection','mlp',1,row++,{formula:'f = GELU(u) W_down + b_down',desc:'回到 DiT 隐藏宽度。'});E(gelu,down);
  const end=n('res2','Feed-forward residual','norm',1,row++,{formula:'y = x′ + f',desc:'训练态 dropout 在 eval 推理模式下不生效，本图展示推理路径。'});E(down,end);E(res,end,{residual:true});
  const height=row*112+60;groups.push({id:s.id+'.'+idx,label:`DiT layer ${idx+1} · ${cross?'cross-attention':'self-attention'}`,x:x-20,y:y-40,w:720,h:height+35,color:palette.action});return{first:input,last:end,w:740,h:height+75};
 }
 function expand(s,x,y){
  if(s.stack&&s.depth){
   const count=s.depth;
   if(level===1){
    const cols=3,rows=Math.ceil(count/cols);let last,first;
    for(let i=0;i<count;i++){const col=(Math.floor(i/cols)%2===0)?i%cols:cols-1-i%cols;
     const n=N(s.id+'.'+i,s.role==='vision'?`ViT block ${i+1}`:s.role==='action'?`Action block ${i+1}`:`Decoder block ${i+1}`,s.type,x+col*240,y+Math.floor(i/cols)*120,{...s,id:s.id+'.'+i,module:s.id,layer:i,stack:false,shape:s.shape,label:s.role==='vision'?`ViT block ${i+1}`:s.role==='action'?`Action block ${i+1}`:m.kind==='hybrid'?`${(i+1)%4?'DeltaNet':'Full attention'} ${i+1}`:`Decoder block ${i+1}`,x:x+col*240,y:y+Math.floor(i/cols)*120,w:200,h:76});
     if(last)E(last,n);else first=n;last=n;
    }groups.push({id:s.id,label:`${s.label} · ${count} layers`,x:x-20,y:y-40,w:720,h:rows*120+45,color:palette[s.type]});return {first,last,w:740,h:rows*120+65};
   }
   if(level===3){
    let last,first;const cols=3;let rowY=y,maxH=0;const layerBounds=[];
    for(let i=0;i<count;i++){if(i%cols===0&&i>0){rowY+=maxH+140;maxH=0}const tile=ops(s,i,x+(i%cols)*790,rowY+40);maxH=Math.max(maxH,tile.h);if(last)E(last,tile.first,{betweenLayers:true});else first=tile.first;last=tile.last;layerBounds.push(tile)}
    return {first,last,w:cols*790,h:rowY-y+maxH+60};
   }
  }
  const n=N(s.id,s.label,s.type,x,y,{...s,w:220,h:86,module:s.id,layer:0});return {first:n,last:n,w:250,h:150};
 }
 if(level===2){
  const s=arch.find(n=>n.id===focus.module&&n.stack&&n.depth)||arch.find(n=>n.stack&&n.depth);
  if(s){const i=Math.max(0,Math.min(focus.layer||0,s.depth-1)),tile=ops(s,i,20,90);
   const entry=N('block-input','Layer input','input',250,-75,{shape:s.shape,desc:`第 ${i+1} 层输入；残差边从该输入分叉。`});E(entry,tile.first);
   // Residuals read the raw block/sub-layer input, not normalized output.
   const firstResidual=edges.find(e=>e.residual&&e.from===tile.first.id);if(firstResidual)firstResidual.from=entry.id;
   const output=N('block-output','Layer output','output',250,tile.h+130,{shape:s.shape,desc:`进入下一层；这是第 ${i+1} / ${s.depth} 层的数学算子图。`});E(tile.last,output);
   return finalize({focusModule:s.id,focusLayer:i,modeNote:'单层算子 · 重复层共享结构，各层权重独立'});
  }
 }
 const rows=Math.max(...arch.map(n=>n.row))+1,expanded=level===1||level===3;let y=0;
 const colWidths=[0,1,2].map(c=>Math.max(260,...arch.filter(n=>n.col===c).map(n=>expanded&&n.stack&&n.depth?(level===3?2400:770):280)));
 const colX=[0,colWidths[0]+90,colWidths[0]+colWidths[1]+180],ports=new Map();
 for(let row=0;row<rows;row++){let max=0;for(const s of arch.filter(n=>n.row===row)){const b=expand(s,colX[s.col],y+55);ports.set(s.id,b);max=Math.max(max,b.h)}y+=max+80}
 for(const s of arch){const p=ports.get(s.id);for(const dep of s.deps||[])if(ports.has(dep)&&!(level===3&&s.id==='action'&&dep==='language'&&m.kind==='pi'))E(ports.get(dep).last,p.first,{conditional:(s.id==='action'&&dep==='language')||s.id==='joint'});if(s.loop&&ports.has(s.loop))E(p.last,ports.get(s.loop).first,{loop:true});}
 if(level===3&&m.kind==='pi')for(let i=0;i<m.alayers;i++){const cache=nodes.find(n=>n.id===`action.${i}.prefixkv`);if(cache){E(`language.${i}.krope`,cache,{conditional:true});E(`language.${i}.v`,cache,{conditional:true})}}
 if(level===3&&m.kind==='groot')for(const n of nodes.filter(n=>n.id.endsWith('.context')))E(ports.get(m.id==='groot15'?'vladapter':'vlm').last,n,{conditional:true});
 // Correct first residual in each expanded layer to receive the preceding raw activation.
 for(const e of edges.filter(e=>e.residual&&e.from.endsWith('.norm'))){const input=edges.find(q=>q.to===e.from&&!q.residual);if(input)e.from=input.from;}
 return finalize({modeNote:level===3?'全部已建模层与算子 · 循环 / 专家采用紧凑数学表示':level===1?'网络层 · 点击层块查看内部算子':level===2?'该模型当前提供语义模块，内部细节待核实':'架构全景 · 点击模块探索'});
 function finalize(extra){
  const map=new Map(nodes.map(n=>[n.id,n]));
  for(const e of edges){e.a=map.get(e.from);e.b=map.get(e.to)}
  for(const e of edges)if(!e.a||!e.b)throw new Error(`Unknown graph endpoint: ${e.from} → ${e.to}`);
  const valid=edges;
  const degrees=new Map(nodes.map(n=>[n.id,0])),next=new Map(nodes.map(n=>[n.id,[]]));
  for(const e of valid)if(!e.loop){degrees.set(e.to,degrees.get(e.to)+1);next.get(e.from).push(e.to)}
  const queue=nodes.filter(n=>degrees.get(n.id)===0),ordered=[];
  for(let i=0;i<queue.length;i++){const n=queue[i];ordered.push(n);for(const to of next.get(n.id)){degrees.set(to,degrees.get(to)-1);if(degrees.get(to)===0)queue.push(map.get(to))}}
  if(ordered.length!==nodes.length)throw new Error('Unexpected cycle in inference graph');
  let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
  for(const n of [...nodes,...groups]){minX=Math.min(minX,n.x);minY=Math.min(minY,n.y);maxX=Math.max(maxX,n.x+n.w);maxY=Math.max(maxY,n.y+n.h)}
  return {nodes:ordered,edges:valid,groups,map,bounds:{minX,minY,maxX,maxY,w:maxX-minX,h:maxY-minY},arch,...extra};
 }
}

export function tensorShape(shape,params,m){const dict={B:params.batch,T:params.tokens,H:params.horizon,A:params.actionDim,E:m.experts,'H+1':params.horizon+1,'T+H':params.tokens+params.horizon,'T+H+1':params.tokens+params.horizon+1};return shape.map(v=>typeof v==='number'?v:dict[v]??v)}
export function tensorSize(shape,bytes=2){if(!shape.length||shape.some(v=>typeof v!=='number'))return null;const elements=shape.reduce((a,b)=>a*b,1);return {elements,bytes:elements*bytes}}
export function formatBytes(n){if(n===0)return '0 B';const units=['B','KiB','MiB','GiB','TiB'];let i=0;while(n>=1024&&i<4){n/=1024;i++}return `${n>=100?n.toFixed(0):n.toFixed(2)} ${units[i]}`}
