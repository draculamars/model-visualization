# Model Atlas

A Chinese-language model architecture explorer for LLMs, VLAs, world action models and humanoid control. Research snapshot: **2026-09-07**.

This is a dependency-free, client-side static site. The deployed product includes 18 curated model entries, overview/layer/operator/all-expanded graph modes, canvas pan and zoom, keyboard and touch navigation, node inspectors, symbolic tensor shape and storage calculations, a numerical attention laboratory, and per-model primary sources.

## Run and verify

Serve `dist/` through any static HTTP server. ES modules require HTTP rather than opening the HTML as a local file.

```sh
python -m http.server 8080 --directory dist
node tests/graph-check.mjs
```

There is no build step. `.openai/hosting.json` contains the Sites identity and static output directory. Source is persisted in the Site's Git repository.

## GitHub Pages 部署

项目无需安装依赖或编译，直接发布 `dist/`。入口脚本、样式和模块使用相对路径，兼容 GitHub Pages 的仓库子路径和独立域名。

1. 将本项目推送到目标 GitHub 仓库的 `main` 分支。
2. 在仓库 **Settings → Pages → Build and deployment → Source** 中选择 **GitHub Actions**。
3. 在 **Actions** 中运行 **Deploy Model Atlas to GitHub Pages**；之后每次推送到 `main` 都会自动验证并发布。
4. 使用成功的部署任务返回的实际网站地址。默认项目地址通常为 `https://<owner>.github.io/<repository>/`，独立域名以 Pages 设置为准。

工作流先检查 JavaScript 语法和现有模型计算图测试，再上传 `dist/`。Pull request 只执行验证。部署使用 GitHub 自动提供的令牌，无需把访问令牌放进源码。Node.js 24 仅用于验证，浏览器运行不依赖 Node.js。

源码仓库可见性和 Pages 网站访问权限是不同设置；发布前按目标受众配置。GitHub Free 支持公开仓库的 Pages，私有仓库的 Pages 可用性取决于账号套餐。部署流程参考 [GitHub 官方文档](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。

## Source structure

- `dist/models.js`: checkpoint configuration, evidence level, primary-source links, per-model limits.
- `dist/graph.js`: family-specific architecture graphs, repeated layer expansion, operator dependencies, shape evaluation.
- `dist/renderer.js`: Canvas rendering, viewport culling, tensor block / flat views, pan/zoom, keyboard/touch interaction, minimap.
- `dist/math.js`: numerically stable attention calculations independent of any model checkpoint.
- `dist/app.js`: catalogue, inspector, node and layer navigation, path playback, attention laboratory.
- `dist/style.css`: responsive workbench layout and theme.

## Coverage and interpretation

| Model family | Current detail |
| --- | --- |
| Qwen3 8B / 235B-A22B, Llama 3.1 8B | Every configured decoder layer; mathematical Q/K/V, rotary positions, masking, softmax, FFN and residual operations. MoE experts are batched on a symbolic E axis. |
| Qwen3.5 9B | 27 vision blocks; 32 decoder blocks with 24 linear-attention and 8 full-attention layers. Gated DeltaNet recurrence is a semantic operation, not an expanded parallel kernel. |
| DeepSeek V3 / R1 | 61 layers with MLA and dense-first / MoE FFNs. MLA de/rotary branches and shared experts retain explicitly labelled semantic components. |
| DeepSeek V4 Pro | 61 configured hybrid attention and MoE layers. CSA, HCA and mHC remain semantic components; not mislabeled as V3 MLA. |
| π0 / π0.5 / π0-FAST | SigLIP and Gemma stacks, separate prefix / action widths, layer-aligned prefix K/V connections, flow sampler or FAST path. Some embedding/codec operations remain mathematical modules. |
| GR00T N1.5 / N1.6 / N1.7 | Version-specific action-stack depths; AdaLayerNorm, interleaved attention and GELU FFNs. N1.7 uses a fixed configuration revision, while backbone internals remain semantic modules. |
| OpenVLA | Original Llama 2 decoder path; DINOv2 and SigLIP retained as semantic vision towers. |
| π0.6 / π0.7, DreamZero, SONIC | Official-paper and project-level module topology. Unknown / not-yet-imported internal configuration remains opaque; unsupported layer selectors are disabled. |

**“All expanded” means all currently modelled layers and operators.** It does not mean every tensor element, scalar arithmetic instruction, fused kernel, sampler time step or every physical execution of a routed expert. Repeated sampling is represented by feedback edges, experts by a batch axis. Node grids are illustrative tensor geometry and never represent activation values.

The numerical laboratory performs real arithmetic on explicitly editable **teaching matrices**. It does not load model weights, tokenize user prompts, execute the selected checkpoint or capture its runtime activations. Tensor byte estimates describe a single BF16 tensor, not a memory profiler or peak VRAM prediction. Some efficient attention implementations never materialize their full score matrix.

The distinction between code/configuration-derived dimensions and hand-authored mathematical topology appears in the interface. The project does not currently include automatic PyTorch FX / Export / ONNX graph import, model serving, training or runtime trace collection. These need dedicated import/runtime adapters before claims of exhaustive model execution would be valid.

## Adding an adapter

1. Add a named checkpoint and primary sources in `models.js`; never infer an unpublished dimension from another version.
2. Define the family's root dependencies in `architecture()` and a layer expansion when justified by source evidence.
3. Preserve unknown shapes symbolically and mark opaque modules.
4. Add family-specific invariants to `tests/graph-check.mjs`.
5. For future runtime imports, keep captured values, inference inputs, exact weight revision, dtype and execution backend separate from illustrative data.

Interaction inspiration: [Brendan Bycroft's LLM Visualization](https://bbycroft.net/llm). The implementation here is independently authored; no source code or weights were copied from that project.
