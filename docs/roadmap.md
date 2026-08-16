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
- [x] `tiandi-app`：Tauri 2 壳（单进程架构：壳内嵌 tiandi-server，数据目录 = 系统应用数据目录）+ React 空壳前端（`ui/`：Vite + React + TS，丹房首页：任务列表/点火按钮/SSE 炉火观察孔，深色琉璃主题）；浏览器模式经 vite proxy 或直连（CORS 已放行）

**退出（M0 全部达成）**：`tiandi doctor` 全绿；API 冒烟通过；SSE 模拟事件连通（Tauri 壳 + 前端已接入）；`cargo test` 31 项全绿、clippy 0 警告。

## M1 首炉（约 4 周）✅ 全部完成（2026-08）

**目标**：用 UI 完整跑通一次 **NoobAI SDXL LoRA** 训练（IPC/Stdio 全链路）。

- [x] `tiandi-dataset`：导入（文件夹扫描）、缩略图（rayon 并行 JPEG）、dHash 感知去重、EXIF、分辨率桶 + 分布统计
- [x] `tiandi-recipe`：丹方类型化 schema + 校验器 + 内置预设 4 个 + TOML 文件格式
- [x] 数据集 API + 丹方 API + 药库 API（产物列表/重命名/删除）+ 静态文件服务
- [x] 打标 v1：内核 tagger 模式（mock 打标 + WD14 真实入口）+ captions API（读/写/批量替换字符串与正则/标签统计）+ 标签编辑器 UI（图片网格/标签云/批量操作/手动编辑）
- [x] `tiandi-engine-compat`：kernel_runner.py（mock + sd-scripts）+ TOML 参数映射 + IPC/Stdio 协议（hello/JSON Lines/心跳/控制命令）+ 训练启动/取消
- [x] 训练控制台 UI：火候仪表盘（进度环/loss 曲线 SVG/采样画廊/日志流）+ 药库视图 + 任务选中联动
- [x] 错误路径：内核 fail 事件 → Failed 状态 + 失败摘要（tail）；取消走两段式

**验证**：70 测试全绿（含 2 项真实 Python 进程集成测试）+ clippy 0 警告 + 端到端冒烟（模拟训练全链路/打标/批量替换/产物管理）。

**遗留（进入 M2）**：真实 sd-scripts 内核安装（venv + torch cu128 + 内核 commit 锁定）、WD14 真实打标验证、真实基底模型注册。

## M2 连烧（约 4 周）

**目标**：队列 + 续训 + Anima（sd-scripts 后端）+ Illusion 验证。

- [x] 任务队列（持久化 SQLite）：排队/串行/失败不阻断/完成通知（scheduler 串行泵 2s 轮询自动拉起 Queued）
- [x] 断点续训：resume 状态落盘 + 崩溃恢复扫描（服务启动 Preparing/Running→Failed 可重试）+ UI 一键续丹（检测 runs/<id>/checkpoints 最新 state 目录）
- [x] 指标曲线（loss/lr）内置图表（替代 TensorBoard 依赖）；GPU/显存监控（nvidia-smi 解析 + 前端 3s 轮询）
- [x] Anima 支持：丹方族 DitAnima + BackendSdScripts 参数映射扩展（Qwen3 TE、t5 旧分词器本地化、attn_mode=torch、cache_text_encoder_outputs=false、T-LoRA 走 lora_anima）——**真实 Anima LoRA 训练已打通**（anima-base-v1.0 + qwen_3_06b_base + qwen_image_vae 全离线，run ddb1423c queued→done，产物 63.2MB 入药库）
- [x] Illusion 注册验证（SDXL 族第二个检查点，同一管线）——**真实训练打通**（illustriousXL_v01 全离线，loss 0.21→0.11，产物 81.5MB 入药库）；同步验证 NoobAI 四变体（chenkin noob / Chenkin-RF / Epsilon-11 / V-Pred-065），其中 V 预测模型新增丹方 `prediction_type` 字段（映射 sd-scripts `v_parameterization`）
- [x] 镜像源设置（HF_ENDPOINT/PIP_INDEX_URL 注入内核）与系统通知

**退出**：队列连续跑完 ≥3 个任务（含失败恢复）；Anima LoRA 训练成功；Illusion 训练成功。

## M3 拓炉 + Krea2（约 4–8 周）

**目标**：Krea 2 内核接入 + 打标完善 + 自动化。

- [x] Krea 2：许可调研结论落地（Krea 2 Community License：允许个人训练/分发 Derivative，年营收 <$1M 可商用）→ `BackendAiToolkit`（ai-toolkit 后端）全链路打通——**真实 Krea 2 LoRA 训练成功**（krea2_raw_bf16 + qwen3vl_4b + qwen_image_vae 全离线；MMDiT 430/430、VAE 194/194、tokenizer 151936 本地化；run 00873e71 queued→done，3 指标点，LoRA 54.5MB 入药库）
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
