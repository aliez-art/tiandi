# 天地熔炉 Tiandi Furnace — 技术架构（v1.1）

> 配套文档：`../PRD.md`（产品需求）、`./roadmap.md`（里程碑）、`./model-support.md`（模型矩阵）。
> 架构基调（ADR-001，v1.1 冻结）：**Rust 控制/数据引擎 + Python 训练计算内核，IPC/Stdio 流通信**；Rust 原生训练内核为远期探索，不在主线。

## 1. 设计目标

1. **Rust 做控制与数据引擎**：领域模型、数据管线（导入/去重/桶/标签/缩略图）、任务调度、本地服务、桌面壳全部 Rust 实现。
2. **训练计算内核为 Python**：sd-scripts / ai-toolkit 以受管子进程运行（独立 venv + 锁版本），`Trainer` trait 统一编排；放弃 PyO3 绑定（ADR-001），通信走 IPC/Stdio（理由见 PRD §8.3）。
3. **一切可恢复**：任务状态机 + 落盘 manifest，崩溃/断电后一键续丹。
4. **配置即丹方**：训练配置是一等公民（TOML 文件），可命名、继承、版本化、导出。

## 2. 分层与 crate 依赖

```
tiandi-app (Tauri 壳)
    │
tiandi-server (axum: REST + SSE + 静态资源)
    │
tiandi-core (领域模型 / 用例 / 事件总线) ◄── tiandi-state (SQLite)
    │
tiandi-engine (Trainer trait + 任务状态机 + IPC 协议)
    ├── tiandi-engine-compat  (Python 内核编排：SdScripts / AiToolkit 双后端)
    └── tiandi-engine-native  (candle，远期探索，不排期)
    │
tiandi-dataset (图像管线)   tiandi-recipe (丹方 schema/预设)
    │
Python 计算内核（受管 venv 子进程：训练 / 打标 / VAE 编码 / 训练中采样出图）
```

依赖方向：上层依赖下层；`tiandi-engine-compat` 与 `tiandi-engine-native` 互不依赖，均实现 `tiandi-engine` 的 trait。Python 内核不是 crate，是 compat 层管理的受管运行时（独立进程，IPC 通信）。

## 3. 核心领域模型（tiandi-core）

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

持久化：领域数据入 SQLite（`tiandi-state`）；大文件（模型、latent 缓存、采样图）存磁盘目录，库内只存相对路径与哈希。任务运行清单 `manifest.json` 落在任务目录，供恢复。

## 4. 任务状态机

```text
created → queued → preparing → running ⇄ paused
                                ├→ sampling（周期出图，不阻断）
                                ├→ saving（写 checkpoint）
                                └→ done | failed | canceled
failed →（一键）queued（重试）| running（续训 resume）
```

- 状态迁移由 `tiandi-core` 单点驱动（单一事实来源），引擎侧事件（progress/log/sample/error）经事件总线回流。
- 崩溃恢复：启动时扫描任务目录，`preparing/running` 中带 `resume` 能力的任务进入 `failed(resumable)`，UI 提示一键续丹。

## 5. 引擎协议（IPC/Stdio，ADR-001）

### 5.1 总体形态

```text
Rust 侧（tiandi-engine-compat）            Python 内核（受管子进程）
   │  ① 生成任务包                           │
   │     recipe.toml / YAML + 数据集引用      │
   │  ────────────────────────────────────►  │  ② 解析并启动（accelerate launch
   │  ③ 事件通道：读内核 stdout（JSON Lines） │     <内核脚本> --config_file）
   │  ◄────────────────────────────────────  │  hello / heartbeat / progress /
   │  ④ 控制通道：写内核 stdin（JSON Lines）  │  log / sample / done / fail
   │  ────────────────────────────────────►  │  pause / resume / cancel / query
   │  ⑤ 冗余通道：文件监控（notify）兜底      │  采样图 / checkpoint / state 落盘
   │  ⑥ 事件总线 → SSE 推给 UI（tiandi-core） │
```

### 5.2 事件通道（内核 stdout → Rust）

内核 stdout 输出 **JSON Lines**（每行一个 JSON 对象；人类可读日志同时双写日志文件）：

```jsonc
{"type":"hello","v":1,"backend":"sd-scripts","commit":"068bcd7","torch":"2.7.0"}
{"type":"heartbeat","ts":1755000000}
{"type":"progress","step":120,"epoch":0.4,"loss":0.1823,"lr":1.2e-4,"eta_s":3150}
{"type":"log","level":"info","msg":"saving checkpoint..."}
{"type":"sample","path":"runs/xxx/samples/e004-0123.png","prompt":"1girl, ..."}
{"type":"metric","name":"loss","step":120,"value":0.1823}
{"type":"done","code":0,"artifacts":["runs/xxx/checkpoints/xxx.safetensors"]}
{"type":"fail","code":1,"tail":"CUDA out of memory. Tried to allocate..."}
```

- 内核 stdout 中非 JSON 行（如 torch/accelerate 的杂散输出）按 `log` 事件降级处理；**结构化解析失败不阻塞**，原始日志始终落文件。
- 事件缓冲：Rust 侧环形缓冲（参考 lora-scripts-next 1.5 万行方案）供 UI 回放与失败摘要。

### 5.3 控制通道（Rust → 内核 stdin）

```jsonc
{"cmd":"pause"}     // 进程级挂起（Windows: Job Object + NtSuspendProcess）
{"cmd":"resume"}    // 恢复执行
{"cmd":"cancel"}    // 优雅请求（内核收到后写 state 退出）；超时则强杀
{"cmd":"query"}     // 请求立即补发 progress/heartbeat（UI 重连恢复）
```

- `cancel` 两段式：先发命令等 10s，未退则 taskkill /T /F 杀进程树；`pause`/`resume` 为预留命令（当前引擎返回 Unsupported，训练侧暂停列入后续里程碑）。

### 5.4 可靠性设计

- **心跳与卡死检测（已实现）**：内核（sd-scripts/ai-toolkit 模式）每 2s 发一条 `heartbeat`；Rust 侧看门狗每 5s 检查，距上次内核输出超过 30s 判定卡死（显存溢出挂起等），自动强制终止并上报失败摘要。
- **握手**：`hello` 事件在启动时上报（backend/version），后续可按锁定清单扩展比对（当前未强制）。
- **冗余通道**：采样图/checkpoint/state 由内核直接落盘；真实训练中采样图由内核侧目录监控自动上报 `sample` 事件（mock 与真实模式均已接通）；产物路径越界（`..`/绝对路径逃逸）会被拒绝入库。
- **失败语义**：非零退出码 + `fail.tail` 摘要（末 2KB 日志）入任务历史并红显于 UI 日志流；日志文件超过 5MB 自动轮转为 `.log.1`。
- **队列可靠性**：任务原子认领（`BEGIN IMMEDIATE` 事务，杜绝重复拉起）；取消先置 Canceled 再终止内核（取消不会再显示为"炸炉"）；崩溃恢复覆盖 Paused；服务优雅关停时终止全部内核进程。

### 5.5 内核版本锁定

- 每个后端一份**锁定清单**（Python 版本、torch/CUDA、内核 commit、requirements hash），`tiandi doctor` 比对体检；升级为显式操作（UI 引导），升级前后保留上一份快照可回滚。
- 参考 lora-scripts-next 的三快照教训，收敛为**单份锁定快照 + 清单校验**，避免"手工同步上游"。

## 6. 丹方（tiandi-recipe）

- 结构：`[meta]`（name/family/tags/版本）+ `[data]`（参数树），serde 反序列化到类型化结构，模型族校验器逐项检查（未知键、非法取值、族内不适用参数）。
- 分层：基础（数据集/步数/学习率/网络 dim）→ 进阶（调度器/EMA/缓存/噪声技巧）→ 专家（完整透传 compat 引擎全参数，含 optimizer_args 自由项）。
- 预设：内置每模型族「入门」预设；用户预设存 `recipes/` 目录，git 友好。
- 参数映射表：参考 kohya_ss `lora_gui.py` 的控件→TOML 键映射（其 2147–2556 行）与 lora-scripts-next 的 `sanitize/apply_anima_defaults` 逻辑，整理为 Rust 侧一张声明式映射表（crate 内常量 + 测试对照）。

## 7. 数据管线（tiandi-dataset）

- 导入：递归扫描 → `image` crate 解码（异步 + rayon 并行）→ 感知哈希去重 → 缩略图（AVIF/WebP）→ EXIF 提取 → 桶分配（长宽比分桶，参考 kohya sd-scripts 的 bucket 算法与 ai-toolkit 的 bucket 实现）。
- 桶可视化：每桶样本数分布图（UI 侧渲染）。
- 打标：由 **Python 内核**完成（WD14/CL/Florence，ONNX 推理），Rust 侧只负责任务编排与结果入库（进度/结果经 §5 协议回流）；Rust ONNX Runtime 打标列为远期探索。
- latent 缓存：由 **Python 内核**完成（sd-scripts `cache_latents_to_disk` / `cache_text_encoder_outputs_to_disk`），Rust 侧管理缓存目录的生命周期与复用（同数据集多任务共享，哈希校验）；candle VAE 编码列为远期探索。

## 8. 本地服务（tiandi-server）

- axum：REST（领域资源 CRUD + 训练控制）+ SSE（`/api/runs/{id}/events`：progress/log/sample/metric）+ 静态资源（UI 构建产物）。
- 仅绑定 `127.0.0.1`；桌面壳（Tauri）内嵌 WebView 直连；纯浏览器模式（`tiandi server --web`）等价可用。
- 认证：本地单用户，无认证；若未来开放局域网再引入 token。

## 9. 数据目录约定

```text
<workspace>/
├── models/        # 基底模型（按 family 子目录）
├── datasets/      # 数据集（图 + 标签 + latent 缓存）
├── recipes/       # 丹方 TOML
├── runs/          # 任务目录（manifest.json + logs/ + samples/ + checkpoints/）
├── vault/         # 药库（LoRA 产物 + 缩略图 + 元数据）
└── tiandi.db       # SQLite
```

## 10. 与三个参考项目的对应关系（设计来源标注）

| 熔炉模块 | 主要参考 | 差异化改进 |
|---|---|---|
| 数据集约定（N_ 前缀） | kohya_ss / lora-scripts-next | Rust 侧校验 + 自动建集建议 + 桶可视化 |
| 参数面 | kohya_ss（最全） | 分层丹方 + 族校验，避免误配置 |
| 标签编辑器 | lora-scripts-next（undo/redo/批量） | 原生实现，标签云 + 正则批量 |
| 任务执行/队列 | kohya_ss CommandExecutor + lora-scripts-next 任务页 + ai-toolkit ui/cron（gpu_ids 绑定/抢占） | 持久化队列 + 状态机 + 恢复 |
| 训练中采样 | ai-toolkit（sample_now/save_now、采样画廊、first_sample 基线） | 采样由内核完成，Rust 收集实时画廊 + 对比 |
| 计算内核 | kohya_ss / lora-scripts-next 的 accelerate launch 子进程模式 | IPC/Stdio 双通道 + 心跳 + 文件监控冗余（§5） |
| 模型抽象 | ai-toolkit（ModelArch 注册表） | 收敛为 SDXL / DiT 两族 + 每族细分 |
| 环境自检 | lora-scripts-next preflight | `tiandi doctor` 原生体检 |
| 大陆网络适配 | lora-scripts-next（镜像/ModelScope） | 保留为设置项 |
