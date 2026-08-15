# 丹炉 DanLu — 技术架构（v0.9 草案）

> 配套文档：`../PRD.md`（产品需求）、`./roadmap.md`（里程碑）、`./model-support.md`（模型矩阵）。
> 本文档描述 M0 起落地的架构骨架；M3 之后的原生引擎细节随技术验证更新。

## 1. 设计目标

1. **Rust 为核心**：领域模型、数据管线、任务调度、本地服务、桌面壳全部 Rust 实现。
2. **引擎可替换**：`Trainer` trait 隔离训练后端；兼容引擎（Python sd-scripts）保证可用性，原生引擎（candle）渐进替代。
3. **一切可恢复**：任务状态机 + 落盘 manifest，崩溃/断电后一键续丹。
4. **配置即丹方**：训练配置是一等公民（TOML 文件），可命名、继承、版本化、导出。

## 2. 分层与 crate 依赖

```
danlu-app (Tauri 壳)
    │
danlu-server (axum: REST + SSE + 静态资源)
    │
danlu-core (领域模型 / 用例 / 事件总线) ◄── danlu-state (SQLite)
    │
danlu-engine (Trainer trait + 任务状态机 + 事件协议)
    ├── danlu-engine-compat  (Python 桥)
    └── danlu-engine-native  (candle, R&D)
    │
danlu-dataset (图像管线)   danlu-recipe (丹方 schema/预设)
```

依赖方向：上层依赖下层；`danlu-engine-compat` 与 `danlu-engine-native` 互不依赖，均实现 `danlu-engine` 的 trait。

## 3. 核心领域模型（danlu-core）

```text
Project       工作区（数据根目录、默认设置）
BaseModel     基底模型注册项（family: Sdxl1 | DitAnima | DitKrea2，
              path/type/vae/text_encoders/sha256/source）
Dataset       数据集（image dir 集合 + 桶配置 + 标签集引用）
  ├── Image   单张素材（path/hash/尺寸/EXIF/桶归属/缩略图）
  └── Tags    标签或自然语言描述（编辑历史、undo/redo）
Recipe        丹方（模型族校验后的训练配置，TOML 序列化）
Run           一次炼丹任务（状态机、参数快照、产物引用）
  ├── Checkpoint  产出 LoRA / state / 采样图
  ├── Metric      指标点（step/epoch/loss/lr/…）
  └── LogEntry     结构化日志（分片落盘 + 索引）
```

持久化：领域数据入 SQLite（`danlu-state`）；大文件（模型、latent 缓存、采样图）存磁盘目录，库内只存相对路径与哈希。任务运行清单 `manifest.json` 落在任务目录，供恢复。

## 4. 任务状态机

```text
created → queued → preparing → running ⇄ paused
                                ├→ sampling（周期出图，不阻断）
                                ├→ saving（写 checkpoint）
                                └→ done | failed | canceled
failed →（一键）queued（重试）| running（续训 resume）
```

- 状态迁移由 `danlu-core` 单点驱动（单一事实来源），引擎侧事件（progress/log/sample/error）经事件总线回流。
- 崩溃恢复：启动时扫描任务目录，`preparing/running` 中带 `resume` 能力的任务进入 `failed(resumable)`，UI 提示一键续丹。

## 5. 引擎协议（danlu-engine）

```text
Rust 侧（danlu-core）                引擎侧
   │  ① 生成任务包                      │
   │     recipe.toml + dataset 引用     │
   │  ───────────────────────────────► │  ② 解析并启动（compat: accelerate
   │  ③ 事件流（SSE 转发给 UI）         │     launch sd-scripts --config_file）
   │  ◄─────────────────────────────── │  heartbeat / progress / log / sample
   │  ④ 控制指令                        │     / done / fail
   │  ───────────────────────────────► │  ⑤ pause(挂起进程)/resume/cancel(kill tree)
```

- **compat 引擎**：进程监督（job object，kill-tree）、stdout/stderr 结构化解析（进度行、loss、采样完成、错误尾部）、日志环形缓冲（参考 lora-scripts-next 1.5 万行方案）、TensorBoard event 文件直读（loss/lr 标量，参考其 train_monitor）。
- **控制面**：pause = 进程级挂起（Windows Job Object + NtSuspendProcess）；cancel = 递归杀进程树（参考 kohya_ss CommandExecutor）。P1 再做训练侧优雅暂停（写 state + 退出）。
- **版本锁定**：compat 引擎的 venv 依赖与 sd-scripts fork commit 全部锁定（参考 lora-scripts-next 的三快照做法，但改为单一份锁定的 vendored 快照 + 清单校验），`danlu doctor` 体检，升级为显式操作。

## 6. 丹方（danlu-recipe）

- 结构：`[meta]`（name/family/tags/版本）+ `[data]`（参数树），serde 反序列化到类型化结构，模型族校验器逐项检查（未知键、非法取值、族内不适用参数）。
- 分层：基础（数据集/步数/学习率/网络 dim）→ 进阶（调度器/EMA/缓存/噪声技巧）→ 专家（完整透传 compat 引擎全参数，含 optimizer_args 自由项）。
- 预设：内置每模型族「入门」预设；用户预设存 `recipes/` 目录，git 友好。
- 参数映射表：参考 kohya_ss `lora_gui.py` 的控件→TOML 键映射（其 2147–2556 行）与 lora-scripts-next 的 `sanitize/apply_anima_defaults` 逻辑，整理为 Rust 侧一张声明式映射表（crate 内常量 + 测试对照）。

## 7. 数据管线（danlu-dataset）

- 导入：递归扫描 → `image` crate 解码（异步 + rayon 并行）→ 感知哈希去重 → 缩略图（AVIF/WebP）→ EXIF 提取 → 桶分配（长宽比分桶，参考 kohya sd-scripts 的 bucket 算法与 ai-toolkit 的 bucket 实现）。
- 桶可视化：每桶样本数分布图（UI 侧渲染）。
- 打标：P0 走 compat 引擎的 Python 打标器（WD14 ONNX）；M3 起 Rust 侧 ONNX Runtime 直接推理。
- latent 缓存：P0 由 compat 引擎完成（sd-scripts cache_latents_to_disk）；M3 起 Rust 侧 candle VAE 编码产出 `.safetensors` 缓存，训练端直接消费（格式与 sd-scripts 兼容）。

## 8. 本地服务（danlu-server）

- axum：REST（领域资源 CRUD + 训练控制）+ SSE（`/api/runs/{id}/events`：progress/log/sample/metric）+ 静态资源（UI 构建产物）。
- 仅绑定 `127.0.0.1`；桌面壳（Tauri）内嵌 WebView 直连；纯浏览器模式（`danlu server --web`）等价可用。
- 认证：本地单用户，无认证；若未来开放局域网再引入 token。

## 9. 数据目录约定

```text
<workspace>/
├── models/        # 基底模型（按 family 子目录）
├── datasets/      # 数据集（图 + 标签 + latent 缓存）
├── recipes/       # 丹方 TOML
├── runs/          # 任务目录（manifest.json + logs/ + samples/ + checkpoints/）
├── vault/         # 药库（LoRA 产物 + 缩略图 + 元数据）
└── danlu.db       # SQLite
```

## 10. 与三个参考项目的对应关系（设计来源标注）

| 丹炉模块 | 主要参考 | 差异化改进 |
|---|---|---|
| 数据集约定（N_ 前缀） | kohya_ss / lora-scripts-next | Rust 侧校验 + 自动建集建议 + 桶可视化 |
| 参数面 | kohya_ss（最全） | 分层丹方 + 族校验，避免误配置 |
| 标签编辑器 | lora-scripts-next（undo/redo/批量） | 原生实现，标签云 + 正则批量 |
| 任务执行/队列 | kohya_ss CommandExecutor + lora-scripts-next 任务页 + ai-toolkit ui/cron（gpu_ids 绑定/抢占） | 持久化队列 + 状态机 + 恢复 |
| 训练中采样 | ai-toolkit（sample_now/save_now、采样画廊、first_sample 基线） | 实时画廊 + 对比 |
| 模型抽象 | ai-toolkit（ModelArch 注册表） | 收敛为 SDXL / DiT 两族 + 每族细分 |
| 环境自检 | lora-scripts-next preflight | `danlu doctor` 原生体检 |
| 大陆网络适配 | lora-scripts-next（镜像/ModelScope） | 保留为设置项 |
