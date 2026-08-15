# 丹炉 DanLu — 里程碑路线图（v1.0）

> 单人开发估算；每里程碑的退出标准即"可以开始下一里程碑"的门槛。M3 起部分工作可与 M2 并行。

## M0 骨架（约 2 周）

**目标**：可启动的 Rust 骨架 + 空 UI + 环境体检。

- [ ] workspace 依 `docs/architecture.md` 建 10 个 crate（已完成占位骨架，`cargo check` 通过）
- [ ] `danlu-core`：Project / BaseModel / Dataset / Recipe / Run 领域模型 + 任务状态机（单元测试覆盖迁移合法性）
- [ ] `danlu-state`：SQLite 迁移 + 仓储 + run manifest 读写
- [ ] `danlu-server`：axum REST 骨架（runs CRUD）+ SSE 事件流（先用模拟事件）
- [ ] `danlu-cli`：`danlu doctor`（GPU/CUDA/磁盘/路径体检）、`danlu init`
- [ ] `danlu-app`：Tauri 2 壳 + React 空壳（丹房首页占位）
- [ ] CI：cargo fmt/clippy/test

**退出**：`danlu doctor` 全绿；UI 空壳启动并连通 SSE 模拟事件。

## M1 首炉（约 4 周）

**目标**：用 UI 完整跑通一次 **NoobAI SDXL LoRA** 训练。

- [ ] `danlu-dataset`：导入（文件夹/拖拽/zip）、缩略图（rayon 并行）、感知哈希去重、EXIF、桶分配 + 分布可视化
- [ ] 打标 v1：`BackendSdScripts` 的 WD14 ONNX 打标（compat 侧）+ 标签编辑器（批量替换/正则/标签云）
- [ ] `danlu-recipe`：丹方 schema + 校验器 + SDXL 入门预设（参数取值参考 kohya_ss 默认与社区实践）
- [ ] `danlu-engine-compat` BackendSdScripts：venv 引导、TOML 生成（参数映射表先行实现核心子集：优化器/调度器/网络/缓存/采样）、进程监督、stdout 解析、SSE 事件
- [ ] 训练控制台：火候仪表盘（进度/loss/lr/ETA）、实时采样画廊、日志流
- [ ] 药库 v1：产物落盘（safetensors + kohya 元数据 + 缩略图）、列表/重命名/删除
- [ ] 错误路径：OOM/失败摘要/一键重试

**退出**：§PRD-12 的 M1 验收全部通过（含产物可被 ComfyUI 加载）。

## M2 连烧（约 4 周）

**目标**：队列 + 续训 + Anima（sd-scripts 后端）+ Illusion 验证。

- [ ] 任务队列（持久化 SQLite）：排队/串行/失败不阻断/完成通知
- [ ] 断点续训：resume 状态落盘 + 崩溃恢复扫描 + UI 一键续丹
- [ ] 指标曲线（loss/lr）内置图表（替代 TensorBoard 依赖）；GPU/显存监控（nvml crate）
- [ ] Anima 支持：丹方族 DitAnima + BackendSdScripts 参数映射扩展（Qwen3 TE、attn 探测、NaN 防护、T-LoRA/LoKr 网络）
- [ ] Illusion 注册验证（SDXL 族第二个检查点，同一管线）
- [ ] 镜像源设置（ModelScope/hf-mirror）与系统通知

**退出**：队列连续跑完 ≥3 个任务（含失败恢复）；Anima LoRA 训练成功；Illusion 训练成功。

## M3 换骨 + Krea2（约 4–8 周）

**目标**：Rust 数据管线全面接管 + Krea 2 兼容后端。

- [ ] `danlu-dataset` 接管：图像解码/缩略图/桶/去重全 Rust（兼容引擎直接消费）
- [ ] VAE latent 缓存：candle 加载 SDXL VAE 编码器，产出与 sd-scripts 兼容的磁盘缓存（格式对齐 `cache_latents_to_disk`）
- [ ] 打标器本地化：ONNX Runtime 跑 WD14/Florence 类（`danlu-tagger` 或并入 danlu-dataset）
- [ ] Krea 2：许可调研结论落地 → `BackendAiToolkit`（YAML 生成 + run.py 调用 + krea2 arch 参数映射）
- [ ] 原生推理验证：candle SDXL 采样（训练中采样逐步切原生，见 R7）

**退出**：Krea 2 LoRA 训练成功（许可允许前提下）；兼容引擎训练全程消费 Rust 侧缓存。

## M4 原生 SDXL（R&D，约 8–12 周）

**目标**：candle 原生训练循环跑通 SDXL LoRA。

- [ ] 训练循环：时间步采样（sigmoid/lognorm 子集）、eps 损失、AdamW/Adafactor、EMA、梯度累积、梯度检查点
- [ ] 块权重与网络注入（kohya 兼容 keymaps 自实现，golden 文件单测）
- [ ] 与 BackendSdScripts 同参对比测试（loss 曲线形状 + 产物质量一致性）
- [ ] 显存策略：缓存复用、bf16、8bit 优化器评估

**退出**：同参原生/兼容产物质量一致（通过对比验收）。

## M5 原生 DiT（R&D，约 12 周+）

**目标**：Anima → Krea 2 原生训练。

- [ ] flow-matching 训练（velocity 目标 + 权重化时间步）
- [ ] Anima：Qwen3 TE 缓存 + CosmosTransformer 目标层注入
- [ ] Krea 2：SingleStreamDiT 目标层（依赖 M3 调研与 M4 框架成熟度）

**退出**：原生训练通过 PRD §12 全项目验收。

---

## 里程碑依赖关系

```text
M0 ──► M1 ──► M2 ──► M3 ──► M4 ──► M5
                    │         ▲
                    └─────────┘（M3 的 candle VAE/推理为 M4 铺垫）
```

## 范围外（明确不做，防蔓延）

- 多用户/云服务/商业化分发
- 音视频模型（wan/minimax 等）与多阶段训练（ai-toolkit 首版亦排除）
- SD WebUI / ComfyUI 功能替代（仅保证产物互操作）
- 自动打标模型的训练/微调
