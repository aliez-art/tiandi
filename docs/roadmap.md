# 天地熔炉 Tiandi Furnace — 里程碑路线图（v1.1）

> 单人开发估算；每里程碑的退出标准即"可以开始下一里程碑"的门槛。M3 起部分工作可与 M2 并行。
> 架构基调（v1.1 冻结）：**训练计算内核始终为 Python（IPC/Stdio 通信），Rust 原生训练内核不在主线**——见末尾"远期探索"。

## M0 骨架（约 2 周）

**目标**：可启动的 Rust 骨架 + 空 UI + 环境体检。✅ 核心已完成（2026-08，commit 见 git log）；Tauri 壳与 React 空壳为 M0 剩余项。

- [x] workspace 依 `docs/architecture.md` 建 10 个 crate（占位骨架 + M0 实现，`cargo check`/`clippy` 通过）
- [x] `tiandi-core`：Project / BaseModel / Dataset / Recipe / Run 领域模型 + 任务状态机（`RunState` 迁移合法性单测）+ 事件总线（`EventBus` + IPC 对齐的 `Event` 枚举）
- [x] `tiandi-state`：SQLite 迁移（`PRAGMA user_version`）+ 仓储（projects/runs/metrics/checkpoints 等）+ run manifest 原子读写
- [x] `tiandi-server`：axum REST（health/projects/runs/transition/metrics）+ SSE 事件流（`/api/runs/{id}/events`，`all` 不过滤）+ 模拟训练演示（`?simulate=1` 全状态机流转）
- [x] `tiandi-cli`：`tiandi init`（工作区 + 数据库）、`tiandi doctor`（磁盘/内存/GPU-CUDA/端口）、`tiandi server`（127.0.0.1 + 可选 --web）
- [x] `tiandi-engine`：`Trainer` trait 骨架（info/start/pause/resume/cancel/query，M1 由 compat 实现）
- [x] 质量门：cargo fmt / clippy（0 警告）/ test（31 项全绿）+ 冒烟验证（真实进程：init → server → 创建 run → 状态机推进 → metrics 入库 → SSE 事件流）
- [ ] `tiandi-app`：Tauri 2 壳 + React 空壳（丹房首页占位）——M0 剩余项，浏览器模式（`tiandi server --web`）已可先行使用

**退出（M0 核心已达成）**：`tiandi doctor` 全绿；API 冒烟通过；SSE 模拟事件连通（UI 壳落地后即可接入）。

## M1 首炉（约 4 周）

**目标**：用 UI 完整跑通一次 **NoobAI SDXL LoRA** 训练（IPC/Stdio 全链路）。

- [ ] `tiandi-dataset`：导入（文件夹/拖拽/zip）、缩略图（rayon 并行）、感知哈希去重、EXIF、桶分配 + 分布可视化
- [ ] 打标 v1：`BackendSdScripts` 的 WD14 ONNX 打标（Python 内核侧）+ 标签编辑器（批量替换/正则/标签云）
- [ ] `tiandi-recipe`：丹方 schema + 校验器 + SDXL 入门预设（参数取值参考 kohya_ss 默认与社区实践）
- [ ] `tiandi-engine-compat` BackendSdScripts：venv 引导、TOML 生成（参数映射表先行实现核心子集：优化器/调度器/网络/缓存/采样）、进程监督、**IPC/Stdio 协议（hello 握手/JSON Lines 事件/心跳/控制命令/文件监控冗余）**、SSE 事件
- [ ] 训练控制台：火候仪表盘（进度/loss/lr/ETA）、实时采样画廊、日志流
- [ ] 药库 v1：产物落盘（safetensors + kohya 元数据 + 缩略图）、列表/重命名/删除
- [ ] 错误路径：OOM/失败摘要（`fail.tail`）/一键重试

**退出**：§PRD-12 的 M1 验收全部通过（含产物可被 ComfyUI 加载）。

## M2 连烧（约 4 周）

**目标**：队列 + 续训 + Anima（sd-scripts 后端）+ Illusion 验证。

- [ ] 任务队列（持久化 SQLite）：排队/串行/失败不阻断/完成通知
- [ ] 断点续训：resume 状态落盘 + 崩溃恢复扫描 + UI 一键续丹（含内核侧优雅暂停/续训，§5.3 两段式 cancel）
- [ ] 指标曲线（loss/lr）内置图表（替代 TensorBoard 依赖）；GPU/显存监控（nvml crate）
- [ ] Anima 支持：丹方族 DitAnima + BackendSdScripts 参数映射扩展（Qwen3 TE、attn 探测、NaN 防护、T-LoRA/LoKr 网络）
- [ ] Illusion 注册验证（SDXL 族第二个检查点，同一管线）
- [ ] 镜像源设置（ModelScope/hf-mirror）与系统通知

**退出**：队列连续跑完 ≥3 个任务（含失败恢复）；Anima LoRA 训练成功；Illusion 训练成功。

## M3 拓炉 + Krea2（约 4–8 周）

**目标**：Krea 2 内核接入 + 打标完善 + 自动化。

- [ ] Krea 2：许可调研结论落地 → `BackendAiToolkit`（YAML 生成 + run.py 调用 + krea2 arch 参数映射 + 协议适配层复用）
- [ ] 打标完善：WD14/CL/Florence 多模型管理、进度流、失败不阻塞（经 Python 内核）
- [ ] 自动化 CLI：`tiandi run recipe.toml` 无人值守全流程（A 的虚拟队列思路落地）
- [ ] latent 缓存管理：Rust 侧管理内核产出的缓存目录（同数据集多任务共享、哈希校验）
- [ ] 大陆镜像与离线 tokenizer 缓存

**退出**：Krea 2 LoRA 训练成功（许可允许前提下）；CLI 无人值守跑通全流程。

---

## 里程碑依赖关系

```text
M0 ──► M1 ──► M2 ──► M3
```

## 远期探索（不排期、不承诺，进主线需另行评审）

- Rust ONNX Runtime 打标器（替代内核打标进程）
- candle 推理/训练（训练中采样切换、原生训练循环），前置验收：与 Python 内核同参产物一致性对比
- GGUF 导出评估
- Rust 侧 latent 缓存编码（candle VAE）

## 范围外（明确不做，防蔓延）

- 多用户/云服务/商业化分发
- 音视频模型（wan/minimax 等）与多阶段训练（ai-toolkit 首版亦排除）
- SD WebUI / ComfyUI 功能替代（仅保证产物互操作）
- 自动打标模型的训练/微调
