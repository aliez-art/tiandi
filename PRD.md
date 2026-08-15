# 天地熔炉 Tiandi Furnace — 私人 LoRA 训练熔炉 · 产品需求文档（PRD）

| 项目 | 内容 |
|---|---|
| 文档版本 | v1.1（定稿：架构决策——Rust 控制/数据引擎 + Python 训练内核，IPC/Stdio 通信；项目更名「天地熔炉」） |
| 状态 | 已定稿 |
| 目标仓库 | `D:\Projects\tiandi` |
| 定位 | 单人、本机、私有使用的 LoRA 训练工具（Rust 重写） |
| 覆盖模型 | SDXL 族（Illusion、NoobAI）· DiT 族（Anima、Krea 2） |

---

## 1. 摘要（TL;DR）

把三个 Python 生态的 LoRA 训练工具 —— ai-toolkit（ostris）、kohya_ss（bmaltais）、lora-scripts-next（wochenlong）—— 合并重写为**一个用 Rust 实现核心、配全新 UI/UX 的私人炼丹工具「天地熔炉 Tiandi Furnace」**：

- **Rust 做骨**：领域模型、数据管线（图像处理、去重、分桶、缓存）、配方系统、任务状态机、本地服务、桌面 UI 全部 Rust 原生；单二进制分发、启动快、内存可控、无 Python 环境地狱。
- **引擎分层（v1.1 定稿）**：训练计算内核保持 **Python**（sd-scripts / ai-toolkit），Rust 与内核之间采用 **IPC/Stdio 流**通信（明确放弃 PyO3 直接绑定，理由见 §8.3 ADR）；训练引擎抽象为 `Trainer` 接口，多内核驱动可插拔，确保产品**从 M1 起即可用**；Rust 原生训练内核列为**远期探索**（不承诺、不排期）。
- **模型支持**：SDXL 族（Illusion、NoobAI，同属 SDXL-1.0 基线）走成熟 SDXL LoRA 管线；DiT 族（Anima、Krea 2）走 DiT 管线；Anima 已有两个参考项目实现可借鉴，Krea 2 是绿地（需前置调研）。
- **全新 UI/UX**：以「炼丹」为隐喻的产品语言（投料 → 控火 → 开炉），深色琉璃质感界面，工作流向导 + 火候仪表盘 + 实时采样画廊，中文优先。

**一句话愿景**：*把「图包 → 丹方 → 丹药（LoRA）」压缩成一个顺手的本地工作台。*

---

## 2. 背景与动机

### 2.1 三个参考项目的现状（审查摘要见附录 A）

| | ai-toolkit | kohya_ss | lora-scripts-next |
|---|---|---|---|
| 作者/定位 | Ostris，模块化训练框架 + Next.js Web UI + 自举安装器（MIT） | bmaltais，kohya sd-scripts 的 Gradio GUI（Apache-2.0，sd-scripts 同为 Apache-2.0） | wochenlong，sd-scripts 的新一代 FastAPI + 前端 WebUI（主仓库 AGPL-3.0，vendor sd-scripts Apache-2.0） |
| 技术栈 | Python + diffusers 自研训练循环，YAML 配置；419 个 .py 约 11.9 万行 + Next.js UI 约 1.9 万行 | Python + Gradio（硬锁 6.17.3），GUI 生成 TOML 配置并 `accelerate launch sd-scripts --config_file` | Python + FastAPI + vendored VuePress 前端产物（无源码，靠正则 patch）；gradio 3.44.2 仅用于旧标签编辑器 |
| 模型面 | 注册表含 sd1/sd2/sd3/sdxl/…/flux/wan21 及扩展注册 **anima、krea2**（均带完整 LoRA 训练实现）；SDXL 走 legacy StableDiffusion 路径（代码完整但 config/examples 已无 SDXL 示例） | SD1.5/2.x、SDXL、Flux、SD3、Hunyuan、Lumina、**Anima**（anima_train_network.py + Anima LLLite tab）；**Krea 2 零引用** | sd/sdxl/flux/**anima**（lora/tlora/LoKr/finetune + Fast 插件，三份 sd-scripts 快照钉 commit：068bcd7/18e62515/8f4ee8fc）；**Krea 2 零引用** |
| 强项 | 模型抽象干净（扩展注册机制）、时间步采样 10+ 策略与损失技巧最丰富、训练中采样/EMA/量化训练、Prisma 队列（gpu_ids 绑定 + 抢占） | 参数面最全（25 优化器/11 调度器/20 网络结构/块权重/缓存/噪声技巧/元数据），生态事实标准 | 现代 UX、预设、一键安装、监控聚合页（GPU/Loss/采样/日志+卡死检测）、标签编辑器 undo/redo、大陆网络适配（ModelScope/hf-mirror） |
| 弱项 | SDXL 示例缺失、diffusers 按 git 提交钉死、torchao/bitsandbytes 对 Rust 不可复用、依赖链重 | UI 陈旧（巨型函数 train_model 1490 行）、无队列（单进程互斥）、无任务历史、启动慢 | 前端无源码靠 27 处正则 patch、任务不持久化、无队列（max_concurrent=1）、SD3 文案残留、多端口服务 |

三个项目共同痛点：
1. **环境地狱**：Python + torch + CUDA 版本矩阵，安装/升级/冲突反复出现（三项目都为此写了大量 install/自举脚本）。
2. **GUI 与训练参数强耦合**：UI 是"参数表单生成器"，不是"产品"；学习曲线陡，误配置无防错。
3. **无统一工程**：数据集、配方、任务、产物之间缺乏一等公民的数据模型，难以复用与自动化（lora-scripts-next 任务重启即丢）。
4. **新模型跟进慢**：Krea 2 仅 ai-toolkit 支持；三者的前端/后端技术债（vendored 无源码前端、巨型函数、gradio 硬锁）都在快速腐化。

> **许可证说明**（重写合规）：本项目为**全新 Rust 实现**，仅以三个项目为算法与参数面的参考（clean-room 式重写，不复制其代码）；进程级调用 sd-scripts/ai-toolkit 运行时（Apache-2.0/MIT）不受传染；AGPL-3.0（lora-scripts-next 主仓库）仅约束复制其代码——不复制即不触发，文档引用其设计思路即可。

### 2.2 为什么用 Rust 重写

- **分发与运维**：单二进制 + 可选内置 Python 运行时；无 pip/venv/版本冲突；升级即换文件。
- **性能**：图像解码/哈希/去重/分桶/缩略图等数据管线用 `rayon` 全并行，比 Python 快一到两个数量级；VAE 编码前的图像预处理可在 Rust 侧完成。
- **健壮性**：类型系统承载配方校验与任务状态机；崩溃恢复、日志、诊断都是编译期约束。
- **价值定位（v1.1 定稿）**：Rust 承担**控制与数据引擎**（UI/API、任务编排、数据集、丹方、产物、监控、进程监督），训练计算保持 Python 内核；不追求"全面 Rust 化"——candle 训练循环不排期，列为远期探索（§8.4）。

### 2.3 诚实的边界（可行性声明）

Rust 生态目前**没有**成熟的 SDXL/DiT LoRA 训练库：candle/burn 提供算子与推理模型，但训练循环（时间步采样、损失调度、EMA、块权重、桶采样器、缓存管理）需自行实现，属 R&D 级工作量且收益有限（训练瓶颈在 CUDA 内核，换语言不改算力）。因此 v1.1 定稿的长期架构为：**Rust 控制/数据引擎 + Python 训练计算内核，二者以 IPC/Stdio 流通信**（放弃 PyO3 直接绑定，理由见 §8.3 ADR）。该架构"可行"的依据：三个参考项目本质都是"编排进程 + Python 训练子进程"模式（kohya_ss/lora-scripts-next 的 accelerate launch、ai-toolkit 的 UI 独立进程 + 队列），已被社区大规模验证；Rust 的价值集中在编排、数据、UX 与运维，详见 §8。

---

## 3. 产品定位与命名

### 3.1 定位

- **目标用户**：本人（单人、私人使用）；技术背景：懂提示词与模型，但不希望折腾环境与命令行。
- **使用场景**：本机单 GPU（NVIDIA + CUDA 优先），Windows 11 为主。
- **非目标**：多用户/云服务、商业分发、SD WebUI/ComfyUI 替代品、文生图工作台（仅内置"采样验证"用途的推理）。
- **设计原则**：
  1. **丹方（Recipe）优先**：一切训练参数沉淀为可命名、可继承、可版本化的配方；
  2. **渐进透明**：新手一键（预设丹方），专家全参（完整参数面板）；
  3. **本地优先**：无遥测、无账号、API 只绑 `127.0.0.1`；
  4. **任务可断可续**：崩溃后一键续丹（resume）。

### 3.2 命名

| 方案 | 理由 | 备注 |
|---|---|---|
| **天地熔炉 Tiandi Furnace**（主推） | 直接呼应"炼丹"心智；双字节好记；仓库名 `tiandi` | repo: `tiandi` |
| Crucible（坩埚） | 英文语境下的炼丹容器，意象契合 | 备选 |
| LoraForge | 直白，但缺个性 | 备选 |
| HuLu（葫芦） | 中国味足，但意象偏收纳 | 备选 |

> 本 PRD 定名 **天地熔炉 Tiandi Furnace**，仓库 `D:\Projects\tiandi`。

---

## 4. 用户故事与核心场景

- **S1 快速炼丹**：有一批参考图 → 拖入"药材库" → 自动打标 → 选基底模型（NoobAI）→ 套用"动漫人物 P0 预设" → 点火 → 训练中实时看损失曲线与采样图 → 出炉后在"药库"里看到带缩略图与元数据的 LoRA。
- **S2 精细炼丹**：手动清洗标签（标签编辑器：批量替换/排序/正则）、检查桶分布、调整块权重与学习率调度、打开 EMA 与缓存。
- **S3 DiT 炼丹**：选 Anima 基底（或 Krea 2），丹方自动切换为 DiT 族参数（预测类型、时间步采样、序列打包等），流程与 SDXL 一致。
- **S4 挂机炼丹**：队列排 3 个任务，跑完自动继续；结束后桌面通知。
- **S5 药库管理**：对比两版 LoRA 的采样图，删除/重命名/归档；导出元数据供外部工具使用。

---

## 5. 功能需求（FR）

优先级：P0 = 首个可用版本必须；P1 = 第二个版本；P2 = 后续增强。来源标注（K=kohya_ss, L=lora-scripts-next, A=ai-toolkit, N=原生新设计）。

### 5.1 项目与工作区（P0，N）

- FR-101 工作区：单一数据根目录（模型/数据集/产物/日志/数据库分目录管理），可迁移。
- FR-102 基底模型注册表：登记已下载模型（SDXL 族：Illusion、NoobAI；DiT 族：Anima、Krea 2），含路径校验、版本/来源信息、sha256。
- FR-103 模型导入辅助：支持从 HF 下载（含镜像源切换，L 的大陆网络适配思路）、本地文件夹导入、格式识别（safetensors/目录式 diffusers）。

### 5.2 数据集（药材）管理（P0，K/L/A 综合）

- FR-201 导入：文件夹、拖拽、zip；递归扫描；自动生成缩略图（Rust 侧，rayon 并行）。
- FR-202 去重与质量：像素/感知哈希去重（图像+EXIF 信息展示），尺寸过滤，模糊/损坏文件检测。
- FR-203 桶（bucket）系统：按长宽比自动分桶（K/A 的 bucket 算法），可视化分布图（L 的 UX 思路），支持手动指定目标分辨率与桶数。
- FR-204 裁剪策略：中心/随机裁剪、缩放填充选项（K 的参数面）。
- FR-205 数据集统计：样本数、标签频次、重复率、平均分辨率。

### 5.3 打标 / 标签（P0，K/L）

- FR-301 自动打标：本地模型（WD14/BLIP/Florence 类）批量打标；兼容引擎阶段调用 Python 侧工具，原生阶段接 ONNX Runtime。
- FR-302 标签编辑器：网格浏览 + 侧栏标签面板；批量替换/追加/删除、正则、按频次排序、标签云（L 的标签管理是同类最佳，重点移植）。
- FR-303 描述模式：支持"标签列表"与"自然语言描述"两种 caption 模式（A 主打自然语言，K/L 主打标签，两种都要）。

### 5.4 丹方（Recipe / 训练配置）（P0，N+L 预设思路）

- FR-401 丹方结构：模型族无关的抽象层（学习率、优化器、调度器、步数/轮数、网络结构、EMA、缓存开关、采样设置），按模型族校验合法性（如 SDXL 的块权重对 Krea 2 无意义）。进阶层需覆盖：时间步采样策略（sigmoid / lognorm_blend / flux_shift / weighted 等，参考 ai-toolkit 的 10+ 策略）、噪声与损失技巧（min-SNR / 噪声偏移 / huber / 掩码损失）、预测目标（eps / v-pred / flow-matching，DiT 族默认 flow-matching）。
- FR-402 预设丹方：内置"SDXL-NoobAI 入门 / SDXL-Illusion 入门 / Anima 入门 / Krea 2 入门"等可运行预设（参数取值参考 K/L/A 文档与社区实践），用户可另存/继承/覆盖。
- FR-403 校验器：配置错误在点火前拦截（如学习率超范围、数据集与模型族不匹配、显存估算不足）。
- FR-404 版本化：丹方存为 TOML/YAML 文件，进入 git 友好目录，支持导出分享。

### 5.5 训练引擎与任务控制（P0，K/A/L）

- FR-501 引擎抽象 `Trainer`：`prepare → start → pause → resume → cancel`；统一进度事件（epoch/step/loss/lr/eta/采样图/日志流）。
- FR-502 兼容引擎（P0 必做）：受管 Python 环境（锁版本）；双后端驱动——`BackendSdScripts`（sd-scripts：SDXL/Anima，配置文件 + accelerate launch）与 `BackendAiToolkit`（ai-toolkit：Krea 2，YAML + run.py）；进程监督 + stdout 解析 + 统一事件流；参数映射层全参数覆盖（§8.3）。
- FR-503 原生引擎（P2/R&D）：candle 实现，里程碑化见 §11。
- FR-504 任务状态机：`created → queued → preparing → running → (paused) → sampling → saving → done | failed | canceled`；失败可一键重试/续训（resume）。
- FR-505 采样出图：训练中按步数/轮数间隔出图（A/K 均有），prompt 模板可配置，出图在 UI 实时滚动展示。
- FR-506 训练中编辑：改采样 prompt、调采样频率热生效；核心训练参数锁定（防误触）。
- FR-507 日志：结构化日志（tracing）+ 分片文件，UI 内实时流式查看，可导出。

### 5.6 产物管理（药库）（P0，N）

- FR-601 LoRA 产物：safetensors + kohya 兼容元数据头（训练信息、基底模型、丹方摘要、作者=本人）；自动生成缩略图（取最后一次采样图）。
- FR-602 药库视图：卡片网格，按基底模型/日期/标签过滤；重命名、归档、删除、打开所在目录。
- FR-603 对比：同一丹方不同轮次/参数的两版 LoRA 采样图并排对比（简化版：保存各 checkpoint 采样图）。
- FR-604 导出：元数据 JSON 导出（P1）；GGUF 转换列为 P2 评估项（ai-toolkit 仓库内未内置 GGUF 转换，需外接工具链）。
- FR-605 扩展机制（P2）：模型/训练流程的插件化注册（Rust trait，参考 ai-toolkit 的 extensions + AI_TOOLKIT_MODELS 注册表），为后续新增模型架构铺路。

### 5.7 队列与自动化（P1，L/A）

- FR-701 任务队列：多任务排队、串行执行、失败不阻断后续（可配置）、完成后动作（通知/关机）。
- FR-702 自动化接口：CLI 模式（`tiandi run recipe.toml`），为后续脚本化/定时炼丹铺路（A 的虚拟队列思路）。

### 5.8 监控与系统信息（P1，L）

- FR-801 训练指标：loss/lr/epoch/step 实时曲线（内置图表，替代 TensorBoard 依赖）。
- FR-802 硬件面板：GPU 型号/显存占用/温度（nvidia-smi 解析或 NVML）、磁盘余量、训练进程状态。

### 5.9 设置（P0）

- FR-901 路径、镜像源（HF/ModelScope）、语言（中文/英文）、主题（深色/浅色）。
- FR-902 环境自检：CUDA/驱动/磁盘/内存体检（L 的 preflight 思路），一键修复指引。

---

## 6. 模型支持矩阵

### 6.1 模型族抽象

```
ModelFamily
├── Sdxl1          # SDXL-1.0 基线：Illusion / NoobAI
│   ├── text_encoders: CLIP-L + OpenCLIP-G（双编码器，输出可缓存）
│   ├── denoiser: UNet（eps / v-prediction 可选；NoobAI 类社区底模常用 v-pred）
│   ├── network: kohya LoRA / LoHa / LoKr / IA3（块权重支持）
│   ├── sampling: 标准时间步采样 + 可选零 SNR / min-SNR / 噪声偏移
│   └── resolution: 1024 基准桶，divisibility 8
└── Dit            # DiT 族
    ├── Anima      # CircleStone-Labs 2B 动漫 DiT（Cosmos 系，flow-matching）
    │   ├── text_encoders: Qwen3-0.6B（+ T5 旧分词器；kohya_ss 另支持 llm_adapter）
    │   ├── vae: Qwen-Image VAE；denoiser: CosmosTransformer3DModel（ai-toolkit 目标层）
    │   ├── network: lora_anima / tlora_anima（T-LoRA）/ LoKr（lora-scripts-next 已验证）
    │   ├── 显存: LoRA 约 8–24GB（按 attn 模式），全量微调约 24GB
    │   └── 上游: diffusers AnimaPipeline；训练参考 sorryhyun/anima_lora
    └── Krea2      # Krea 官方 DiT（Krea2Transformer2DModel, SingleStreamDiT）
        ├── text_encoders: Qwen3-VL（ai-toolkit krea2 实现）；vae: Qwen-Image VAE
        ├── 训练参考: ai-toolkit krea2 arch（完整 LoRA 实现，含 edit 模式/assistant LoRA）
        ├── 推理: diffusers Krea2Pipeline（社区已有多份转换权重）
        └── 前置调研: 官方权重许可、Qwen3-VL 许可、edit 模式配套节点需求
```

### 6.2 支持现状与结论（三项目逐项核实）

| 模型 | 家族 | ai-toolkit | kohya_ss | lora-scripts-next | 策略 |
|---|---|---|---|---|---|
| NoobAI | SDXL-1.0 | legacy 路径（无示例） | 成熟（sdxl_train_network.py） | 成熟（sdxl master 页，v-pred/rectified-flow） | **P0：兼容引擎 sd-scripts 后端首跑目标** |
| Illusion | SDXL-1.0 | 同 SDXL 路径 | 同 SDXL 路径 | 通用 sdxl 页（无专有条目） | **P0：与 NoobAI 同一 SDXL 丹方族**，仅注册表选不同检查点 |
| Anima | DiT | anima arch 完整实现（flow-match、quantize、层卸载、text_conditioner） | anima_train_network.py + Anima LLLite tab | lora/tlora/LoKr/finetune + Fast 插件（Qwen3 TE 细节最全） | **P1：兼容引擎 sd-scripts 后端**（参考 lora-scripts-next 已验证参数）；原生 M5 |
| Krea 2 | DiT | **krea2 arch 完整实现（唯一参考）** | 无 | 无 | **P1：兼容引擎 ai-toolkit 后端**（唯一现成路径）；许可调研先行；原生 M5 |

> 结论修正（相对草案 v0.9）：Krea 2 因 ai-toolkit 的完整实现从"纯绿地"降级为"有参考实现"，但**仅 ai-toolkit 一家**，因此兼容引擎需要支持**两个 Python 后端**（sd-scripts 管 SDXL/Anima，ai-toolkit 管 Krea 2），见 §8.3。
> 注：SDXL 族内部差异（Illusion 是否含 refiner/VAE 变体）不影响 LoRA 训练管线，仅影响基底模型文件选择；具体检查点由用户在注册表指定。

---

## 7. 非功能需求（NFR）

- **性能**：应用冷启动 < 3s；千图数据集的导入+缩略图 < 60s（rayon 并行）；UI 指标刷新 ≥ 2Hz；采样图从产生到展示 < 2s。
- **可靠性**：训练中断（崩溃/断电）后可续训；任务元数据与数据库事务写入；日志滚动不撑爆磁盘。
- **安全**：仅监听 127.0.0.1；无遥测；不自动上传任何数据；第三方下载仅按用户显式操作。
- **可维护性**：Rust workspace 单仓；核心库单元测试 + 引擎协议集成测试；CI（cargo test/clippy/fmt）。
- **兼容性**：Windows 11 x64 + NVIDIA CUDA 优先；macOS/Linux 列为 P2。
- **分发**：桌面版单安装包（Tauri）；可选"内置 Python 运行时"模式用于兼容引擎（首次启用时引导安装）。

---

## 8. 技术架构

### 8.1 总体分层

```
┌─────────────────────────────────────────────────────┐
│  UI 层    Tauri 2 桌面壳（可选纯浏览器模式）          │
│           React 18 + TypeScript + Vite               │
├─────────────────────────────────────────────────────┤
│  API 层   tiandi-server（axum）                       │
│           REST + SSE（进度/日志/采样图事件流）        │
├─────────────────────────────────────────────────────┤
│  应用层   tiandi-core（领域模型/用例）                │
│           项目/基底模型/数据集/丹方/任务 状态机       │
│           tiandi-state（SQLite 持久化）               │
├─────────────────────────────────────────────────────┤
│  引擎层   Trainer trait（多后端驱动）                  │
│           ├── tiandi-engine-compat（IPC/Stdio 桥）      │
│           │     ├── BackendSdScripts（sd-scripts, P0）│
│           │     └── BackendAiToolkit（ai-toolkit, P1）│
│           └── tiandi-engine-native（candle，远期探索） │
├─────────────────────────────────────────────────────┤
│  计算内核  Python 子进程（训练/打标/VAE 编码/采样出图）│
│          受管 venv + 锁版本；stdin 控制 / stdout 事件 │
├─────────────────────────────────────────────────────┤
│  基础设施 tiandi-dataset（rayon 图像管线）             │
│           tiandi-recipe（serde 校验）                  │
│           tracing / notify / 进程监督                 │
└─────────────────────────────────────────────────────┘
```

### 8.2 Cargo workspace（crates）

| crate | 职责 |
|---|---|
| `tiandi-core` | 领域模型（Project/BaseModel/Dataset/Recipe/Run/Checkpoint/Metric）、用例服务、事件总线（tokio broadcast） |
| `tiandi-state` | SQLite（rusqlite/sqlx）迁移与仓储；运行清单（manifest）落盘 |
| `tiandi-dataset` | 图像解码（image）、EXIF、哈希去重、缩略图、桶算法、数据集统计 |
| `tiandi-recipe` | 丹方 schema（serde）、校验器、内置预设、TOML 读写 |
| `tiandi-engine` | `Trainer` trait、事件协议、任务状态机 |
| `tiandi-engine-compat` | Python 桥：多后端驱动（SdScripts / AiToolkit）、venv 管理、进程监督、进度/日志解析、参数映射 |
| `tiandi-engine-native` | （远期探索）candle 训练循环：VAE 编码、时间步采样、LoRA 权重、优化器 |
| `tiandi-server` | axum REST + SSE、静态资源服务 |
| `tiandi-cli` | `tiandi` 命令行（run/import/doctor 等） |
| `tiandi-app` | Tauri 2 壳（托盘、通知、窗口） |

### 8.3 通信架构（IPC/Stdio）与兼容引擎（P0，可行性基石）

> **ADR-001（v1.1 定稿）：放弃 PyO3 直接绑定，Rust 控制/数据引擎与 Python 训练内核采用 IPC/Stdio 流通信。**
> 理由：① 进程隔离——训练内核 OOM/segfault/崩溃不拖垮 UI 与任务编排（PyO3 同进程则一损俱损）；② 版本独立——两个 Python 内核（sd-scripts / ai-toolkit）各有独立 venv、甚至不同 Python 版本，PyO3 绑定无法同时服务，IPC 则各挂各的子进程；③ 部署简单——无需链接 libpython、无需随 CPython 版本重编译；④ 可观测可复现——内核输出即日志，可脱离 UI 手动复跑同一命令排查；⑤ 三个参考项目（accelerate launch / 独立 UI 进程 + 队列）已验证该模式在 Windows 上的可靠性。

- **受管运行时**：首次使用引导创建独立 venv，依赖清单锁定（torch/CUDA 版本、sd-scripts / ai-toolkit 的 commit），`doctor` 命令体检；升级是显式动作。
- **双后端驱动**（`CompatBackend` trait）：
  - `BackendSdScripts`（P0）：SDXL / Anima。配置走 TOML + `accelerate launch <脚本> --config_file`（kohya_ss 现代用法），Anima 参数参考 lora-scripts-next 验证过的组合（Qwen3 TE、attn 模式探测、CAME+fp16→bf16 防 NaN）。
  - `BackendAiToolkit`（P1）：Krea 2（唯一现成路径）。配置走 YAML + `run.py` 调用，复用其 krea2 arch 的参数面（SingleStreamDiT 目标层、Qwen3-VL TE、edit 模式选项）。
- **IPC/Stdio 协议**（详见 `docs/architecture.md` §5）：
  - 事件通道：内核 stdout 输出 **JSON Lines**（每行一个事件：`hello` 握手 / `progress` / `log` / `sample` / `done` / `fail`），人类可读日志同时落文件双写；
  - 控制通道：Rust 写内核 stdin（JSON Lines 命令：`pause` / `resume` / `cancel` / `query`），Windows 取消兜底 = Job Object 杀进程树；
  - 冗余：采样图/checkpoint 落盘目录由 Rust 侧文件监控（notify）兜底，事件流断线不丢产物；心跳超时触发卡死检测（参考 lora-scripts-next）。
- **参数映射**：以 kohya_ss `lora_gui.py` 的控件→TOML 映射表（2147–2556 行）为主干，叠加 lora-scripts-next 的 sanitize/Anima 默认值逻辑与 ai-toolkit 的 YAML 字段面，整理为 Rust 侧声明式映射表（全参数覆盖、零隐藏、单测对照）。
- **版本锁定**：内核 venv 依赖与 Python 仓库 commit 全部锁定（参考 lora-scripts-next 三快照做法，但收敛为单份锁定快照 + 清单校验），避免上游破坏性变更。

### 8.4 远期探索：原生 Rust 训练内核（不排期、不承诺）

v1.1 起，**Rust 原生训练内核不再进入路线图**（用户决策：全面 Rust 化收益低、成本高——训练瓶颈在 CUDA 内核，换语言不改算力）。仅保留如下**可选项**供未来评估（任一都需先满足 §12 验收标准）：

1. Rust ONNX Runtime 打标器（数据引擎延伸，替代内核打标进程）；
2. candle 推理（训练中采样逐步切换，减少对内核采样实现的依赖）；
3. candle 训练循环（SDXL → DiT），以"与 Python 内核产出一致性对比测试"为前置验收。

### 8.5 数据与状态

- SQLite 存领域数据（项目/数据集/丹方/任务/指标点/日志索引）；大文件（模型、缓存 latent、采样图）存磁盘目录，库中只存路径与哈希。
- 任务运行清单（manifest.json）落盘于任务目录，崩溃后由 `tiandi-core` 恢复状态机。

---

## 9. UI / UX 设计（全新设计）

### 9.1 设计语言：熔炉隐喻

- **概念映射**：数据集=药材，打标=拣药，丹方=火候与配比，训练=控火炼丹，LoRA=丹药，队列=多炉连烧，日志=炉火观察孔。
- **视觉**：深色为主（墨色/玄青底），主色朱砂红（进度/运行），琥珀金（产物/高亮），青瓷青（成功/数据）；衬线标题 + 无衬线正文；克制的光晕与颗粒纹理，避免"游戏化"廉价感。
- **文案**：中文优先，动作动词化（投料/拣药/配丹方/点火/开炉/收丹），状态用炉火意象（文火=暂停，武火=高速训练，熄火=停止）。

### 9.2 信息架构（七个主区）

| 区 | 名称 | 内容 |
|---|---|---|
| 丹房 | Dashboard | 当前任务、GPU/磁盘概览、最近丹药、快捷入口 |
| 药材 | Dataset | 数据集列表、导入、缩略图网格、桶分布、去重报告 |
| 拣药 | Captions | 标签编辑器（网格+标签云+批量操作） |
| 丹方 | Recipes | 预设库、丹方编辑器（基础/进阶分层表单）、校验结果 |
| 炼丹 | Training | 任务队列、训练控制台（火候仪表盘、损失曲线、实时采样画廊、日志流） |
| 药库 | Vault | LoRA 产物卡片、对比、元数据、导出 |
| 炉房 | Settings | 环境体检、路径/镜像/主题、模型注册表 |

### 9.3 关键流程

1. **新建炼丹向导**（4 步）：① 选基底模型（按族分组，显示显存预估）→ ② 投料（导入数据集，实时显示样本数与桶分布）→ ③ 配丹方（选预设或手调，点火前校验清单逐项打勾）→ ④ 点火预览（确认任务卡，入队）。
2. **训练控制台**：左侧任务信息与参数摘要（只读），中部火候仪表盘（进度环 + loss/lr 双曲线 + step/epoch/ETA），右侧实时采样画廊（新图自动置顶，可放大对比），底部折叠式日志流。
3. **标签编辑器**：左侧图片网格（多选），右侧标签云（频次字号）+ 编辑面板；批量替换/正则/排序/去重标签；全程撤销。

### 9.4 组件与交互细则

- 参数表单：预设丹方为"卡片"，展开为分层表单（基础/进阶/专家三档，专家档与兼容引擎全参数一一对应）。
- 任务卡片：状态色环 + 进度 + 最近采样图缩略 + 操作（暂停/继续/取消/重试/续丹）。
- 通知：任务完成/失败/显存告警走系统通知（Tauri）。
- 响应式：桌面 1280×800 起；双屏布局（训练时控制台常驻副屏为 P2）。

---

## 10. 里程碑与路线图

| 里程碑 | 周期 | 内容 | 退出标准 |
|---|---|---|---|
| **M0 骨架** | 2 周 | workspace、core 领域模型、SQLite、axum API、Tauri 壳、空 UI 框架、CLI doctor | `tiandi doctor` 通过；UI 空壳可启动 |
| **M1 首炉** | 4 周 | 数据集导入/缩略图/去重/桶、标签编辑器、丹方系统、兼容引擎跑通 **NoobAI SDXL LoRA** 全流程、采样画廊、药库 v1 | 用 UI 完成一次 NoobAI LoRA 训练并出图（验收标准 §12） |
| **M2 连烧** | 4 周 | 任务队列、指标曲线、断点续训、Anima（sd-scripts 后端）、Illusion 验证、镜像源、系统通知 | 队列连续跑完 ≥3 个任务；Anima LoRA 训练成功 |
| **M3 拓炉 + Krea2** | 4–8 周 | **Krea 2 内核（ai-toolkit 后端）**与权重/许可调研结论落地；打标完善（WD14/CL/Florence 经 Python 内核）；自动化 CLI 全流程（`tiandi run recipe.toml`）；大陆镜像与离线缓存 | Krea 2 LoRA 训练成功（许可允许前提下）；CLI 可无人值守跑通全流程 |
| **远期探索** | 不排期 | 可选：Rust ONNX 打标器、candle 推理/训练（见 §8.4）；GGUF 导出评估 | 不阻塞主线 |

> 注：周期为单人开发估算；M3 起可与 M2 并行部分内容。**训练计算内核始终为 Python（IPC/Stdio），Rust 原生训练内核不在主线内（v1.1 决策）。**

---

## 11. 风险与可行性

| # | 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|---|
| R1 | Python 内核（sd-scripts / ai-toolkit）版本漂移与多内核维护成本 | 中 | 中 | 锁版本 + 受管 venv + doctor 体检（R4）；内核抽象 `Trainer`/`CompatBackend` 隔离差异；升级显式化 |
| R2 | **Krea 2 有参考实现（ai-toolkit）但许可/配套待定**：官方权重许可、Qwen3-VL 许可、edit 模式需外部 ComfyUI 节点 | 中 | 中 | M3 前置调研；无法训练则先支持"采样验证"（diffusers 推理），训练列 P2 |
| R3 | Anima 非商用许可与上游变动 | 低 | 低 | 私人使用合规；锁定上游 commit（lora-scripts-next 做法）；如上游失联，fork 维护 |
| R4 | Python 依赖漂移（diffusers 按 git 提交钉死 / torch / CUDA 版本） | 中 | 中 | 锁版本 + 受管 venv + doctor 体检；升级显式化；约束文件防 pip 静默降级（参考 ai-toolkit manager 教训） |
| R5 | Windows 环境差异（驱动/显存/路径/控制台） | 中 | 中 | preflight 体检清单；路径全 Unicode 处理（Rust 天然优势）；一等公民解决原项目的 Windows 补丁问题 |
| R6 | 单人开发精力分散 | 中 | 中 | 范围克制（非目标清单 §3.1）；P0 只做 SDXL 单模型首跑 |
| R7 | IPC 事件流断线/卡死导致进度失真 | 中 | 中 | JSON Lines 结构化事件 + 心跳超时卡死检测 + 采样图/checkpoint 文件监控兜底（§8.3） |
| R8 | **许可证边界**：lora-scripts-next 主仓库 AGPL-3.0 | 低 | 中 | clean-room 重写不复制其代码（仅参考设计）；进程级调用 Apache/MIT 运行时不受传染；文档中标注来源 |
| R9 | **产物互操作**：LoRA 元数据（ss_ 前缀、ss_tag_frequency、training_info）与 keymaps 须字节级兼容 | 中 | 高 | 以 kohya 元数据规范为契约，单测 golden 文件对照；否则 ComfyUI/A1111 无法加载产物 |

---

## 12. 验收标准（Definition of Done）

**M1 验收**：在 Windows 11 + NVIDIA GPU 上，从零安装（单安装包）后，用户在 UI 中完成：导入 20+ 张图 → 自动打标 → 选 NoobAI 基底 → 套入门丹方 → 点火 → 观察到进度/损失曲线/采样图 → 开炉产出带元数据与缩略图的 LoRA 文件，且该文件可被外部工具（如 ComfyUI）正常加载。

**全项目验收**：SDXL 族（Illusion/NoobAI）与 DiT 族（Anima；Krea 2 经 ai-toolkit 内核，前提是权重/许可调研通过）均可用统一丹方体系完成训练；训练全程经 IPC/Stdio 通信（Rust 编排 + Python 内核）；任务可排队、可续训；药库可管理全部产物（kohya 元数据兼容，可被 ComfyUI/A1111 加载）；`tiandi doctor` 全绿。

---

## 13. 附录

### A. 三项目代码审查摘要

> 审查基于各仓库浅克隆（`D:\Projects\_ref\`）的源码阅读（read/glob/grep），未运行代码。子代理深度审查完成于 2026-08，与 PRD v1.0 同版本。

#### A.1 ai-toolkit（MIT，v0.12.23，419 个 .py ≈ 11.9 万行 + Next.js UI ≈ 1.9 万行）

- **结构**：`run.py`（CLI，config 路径 + `-r` 失败继续）→ `toolkit/job.py` 按 `job` 键分发 → `jobs/TrainJob.py`；训练核心 `extensions_built_in/sd_trainer/`（SDTrainer 2317 行、DiffusionTrainer、UITrainer）+ `jobs/process/BaseSDTrainProcess.py`（3003 行主循环）；`toolkit/` 支撑：config_modules（全部 DTO）、data_loader + dataloader_mixins（7 个 mixin ≈ 3000 行）、buckets、optimizer/optimizers、scheduler、ema、network_mixins + lora_special + lycoris_special + models/DoRA、train_tools（SNR）、saving + metadata、samplers（flowmatch）；扩展机制 `toolkit/extension.py` 扫描 `extensions*/` 的 `AI_TOOLKIT_EXTENSIONS` / `AI_TOOLKIT_MODELS` 注册表；`manager/` 自举安装器（torch 规格映射、MinGit/FFmpeg 本地化）；`ui/` Next.js + Prisma SQLite + cron 秒级轮询队列（gpu_ids 绑定、return_to_queue 抢占）。
- **训练管线要点**：分辨率桶（保总像素 + divisibility 对齐，模型可覆写 16）；latent 内存/磁盘缓存（uint8 量化）+ text embedding 缓存 + CLIP vision 缓存（albumentations 增强与缓存互斥）；时间步采样 10+ 策略（sigmoid/lognorm_blend/flux_shift/weighted/one_step/next_sample/多步变体 + 内容/风格三次方采样）；loss 7 种（mse/mae/wavelet/pseudo_huber/pixelspace/mean_flow）；噪声技巧（noise_offset/dynamic/learnable_snr/match_noise_norm/optimal noise pairing/force_consistent_noise）；自研 LoRA 网络（lora/locon/lorm/lokr/DoRA）+ `toolkit/keymaps/*.json` 键名映射保证 kohya/A1111 互读 + `ss_` 前缀元数据；EMA（自研，含 use_feedback）；量化训练（torchao qfloat8/float8/int8/uintx/orbit/nvfp4）+ 8bit 优化器 + 层卸载；OOM 连续 3 次才 abort；resume 从 safetensors 元数据 `training_info.step/epoch` 恢复。
- **模型支持**：注册 arch 含 sd1–sdxl 及 flux 系、qwen_image、**krea2**（SingleStreamDiT + Qwen3-VL TE + Qwen-Image VAE，含 edit 模式/assistant LoRA）、**anima**（CosmosTransformer3DModel、flow-match）、wan 系等；SDXL 走 legacy `StableDiffusion`（is_xl → XL pipeline + 双 TE + SDXL keymaps），**代码完整但 config/examples 已无 SDXL 示例**。
- **移植清单**：P0 分辨率桶/多分辨率数据集、latent 磁盘缓存、flow-matching + 自研时间步采样、自研 LoRA 网络 + keymaps 兼容、min-SNR/噪声偏移/EMA/梯度累积/梯度检查点、safetensors 元数据 resume、量化 + 8bit 优化器；P1 训练中采样 + first_sample 基线、数据集管理 + 自动打标、GPU 绑定队列 + 抢占、损失曲线/样本/日志/GPU 可视化、扩展注册机制；P2 训练技巧库（guided loss/DOP/optimal noise pairing/mean_flow）、自举安装器、模型转换脚本。
- **复杂度**：数据管线 中、训练引擎 高、LoRA 网络 中、配置 低、UI 中、队列 低。
- **坑**：diffusers 按 git 提交钉死、transformers 5.5.3 激进、torchao/bitsandbytes 对 Rust 基本不可复用、gradient_accumulation 两套并存易误配、augmentations 与 latent 缓存静默互斥、krea2 edit 模式需外部 ComfyUI 节点、测试极少。

#### A.2 kohya_ss（Apache-2.0，v26.0.0；GUI 本身不实现训练）

- **结构**：`kohya_gui.py`（291 行）建 9 个顶层 tab（Dreambooth/LoRA/LECO/Anima LLLite/TI/Finetune/Utilities/Settings/About）；`kohya_gui/` 分层：特征 tab（lora_gui.py 4174 行为核心）+ 共享控件组（class_source_model 家族互斥勾选、class_basic_training、class_advanced_training 136 个控件、各模型族 class_sdxl/flux1/sd3/hunyuan/anima/lumina）+ 基础设施（class_command_executor 唯一 subprocess 出口、class_gui_config、common_gui 路径校验与 TOML 写入、localization i18n）。**sd-scripts 为 git submodule**（本检查未克隆）；训练以 `accelerate launch <sd-scripts>/<家族>_train_network.py --config_file <toml>` 方式执行（现代推荐用法，非逐参数命令行）。
- **参数面全貌**（本报告最重要输出）：优化器 25 种（AdamW8bit 默认、Adafactor、DAdaptation 系列 8 种、Prodigy、Lion 系、ScheduleFree 系、pytorch_optimizer.CAME、SGDNesterov 等 + optimizer_args 透传）；调度器 11 种；学习率（learning_rate / text_encoder_lr / t5xxl_lr / unet_lr 驱动 train_unet_only 或 train_text_encoder_only、lr_warmup %换算）；网络结构 20 种（Standard、LoCon/LoHa/LoKr/DyLoRA、LoRA-FA、LyCORIS 全套 iA3/BOFT/Diag-OFT/GLoRA、Flux1、Anima、Lumina；network_dim/alpha、dim_from_weights、network_weights 续训、conv_dim/alpha、**block weights**、25 值 block dims、LoRA+、dropout 三件套、scale_weight_norms、LyCORIS preset）；精度/显存（fp16/bf16/fp8、save_precision、full_fp16/bf16、fp8_base、highvram、gradient_checkpointing、xformers/sdpa、mem_eff_attn、blocks_to_swap、cpu_offload_checkpointing）；缓存（cache_latents[_to_disk]、cache_text_encoder_outputs[_to_disk] 按家族合并、skip_cache_check）；噪声/损失/时间步（min_snr_gamma、noise_offset_type Original/Multires、adaptive_noise_scale、ip_noise_gamma、min/max_timestep、loss_type l2/huber/smooth_l1、huber 参数、v_pred_like_loss、debiased_estimation_loss、masked_loss）；打标/正则（shuffle_caption、weighted_captions、caption_dropout、keep_tokens、max_token_length、clip_skip、color_aug、flip_aug、random_crop、prior_loss_weight、bucket 全套参数）；采样/resume/元数据（sample_every_n_steps/epochs、sample_sampler、save_every、save_last_n_*、save_state、resume、metadata_title/author/description/license/tags、tensorboard/wandb、HF 上传）；accelerate 多机多卡。
- **数据集工具**：`<repeats>_<name>` 概念子目录约定 + 同名 caption 优先；WD14（onnx/递归/tag_replacement/undesired_tags，默认 SmilingWolf/wd-convnext-tagger-v3）、BLIP/BLIP2/GIT、manual 分页打标器、group_images、dataset_balancing、dreambooth_folder_creation；**无桶可视化**（仅 Flux/SD3 show_timesteps）。
- **队列**：无。CommandExecutor 单进程互斥（运行中拒绝二次启动），Stop 用 psutil 杀进程树；stdout 环形缓冲供失败摘要。
- **模型支持**：SD1.5/2.x、SDXL（成熟）、Flux.1（含 chroma）、**Anima**（anima_train_network.py + Anima LLLite tab，LoRA 类型限 Anima/Kohya LoHa/LoKr）、SD3/Hunyuan/Lumina；**Krea 2 全库零引用**。
- **移植清单**：P0 参数面→TOML 完整映射、模型族→脚本路由、数据集约定 + 校验、WD14 打标、config TOML 生成 + 单进程执行/停止、preset 保存/加载、accelerate 等价；P1 BLIP2/GIT 打标、手工打标器、dataset_balancing、convert_model、LoRA 工具链（merge/extract/resize/verify）、tensorboard/wandb、HF 上传/resume、本地化；P2 LECO/Anima LLLite、桶可视化、队列。
- **复杂度**：数据管线 高（语义繁杂但与引擎耦合，重写收益最大）、训练引擎 极高（原生）/中（包装）、配置 低、UI 中、队列 低-中。
- **坑**：GUI 与 sd-scripts `--config_file` 键名强耦合需锁版本；gradio 6.17.3 硬锁；stop_text_encoder_training 对 LoRA 无效；lowvram 控件残留；命名不一致（learning_rate_te vs text_encoder_lr）；gui.bat 每次强制 pip 升级启动慢；测试薄弱无端到端验证。

#### A.3 lora-scripts-next（主仓库 AGPL-3.0，v2.9.1；前端名 "SD Trainer Next"）

- **结构**：`gui.py` 拉起 4 服务（主 GUI 28000 / TensorBoard 6006 / train monitor 6008 / tageditor 28001，端口自动回退）；后端 FastAPI + uvicorn（gradio 3.44.2 仅用于旧标签编辑器子模块）；`mikazuki/`：app/（application 路由 + api.py 全部 REST + proxy 反代 + config 参数记忆）、process.py + tasks.py（accelerate 命令构造、max_concurrent=1 内存任务表）、train_log_hub.py（每任务 1.5 万行环形缓冲 SSE）、dataset_editor.py（原生标签编辑器 API + undo/redo）、tagger/（WD14/CL ONNX 打标 + 进度）、schema/*.ts（schemastery 风格表单 schema）、anima_backend/ + anima_fast_backend/（可选 sorryhyun/anima_lora 插件：安装器/preflight/适配 TOML）。**前端是 vendored 的 VuePress 预编译产物**（frontend/dist，无源码，靠 scripts/patch-*.py 共 27 处正则改哈希 chunk，升级即碎）；**sd-scripts 非 submodule，是三份快照**：vendor/sd-scripts（钉 068bcd7）、scripts/dev（18e62515，Anima/Flux）、scripts/stable（8f4ee8fc，SD1.5）。
- **训练管线**：`N_` 重复子集约定 + 自动建集（suggest_num_repeat 7/5/3/1）；打标 WD14 v2/v3 + CL tagger（ONNX，china_hub 可转 ModelScope，进度流）；表单 → /api/run → fix_config_types → sanitize（去 NaN、Windows 禁 torch_compile、路径转 /）→ apply_sdxl_prediction_type（v-pred/rectified-flow）→ apply_anima_training_defaults → 写 autosave TOML → accelerate_launch.py 启动（trainer_mapping 路由到三快照脚本）；预设 `config/presets/*.toml`（[metadata] 过滤 + [data] 增量覆盖）；**无队列**（训练中再提交直接拒绝，任务不持久化重启即丢）；监控 train_monitor（tensorboard.event_accumulator 直读标量 + pynvml + 正则解析 stdout + 采样图沙箱 + 卡死检测）；采样 sample_prompts 单行语法 + sample_at_first，Anima 自动填 1024/CFG4.5/seed42/40 步与安全词条。
- **模型支持**：SD1.5（LoRA/Dreambooth/TI）、SDXL（LoRA/LoHa/LoKr + finetune，支持 v-pred 与 rectified-flow，NoobAI 类底模经此路径）、Flux（LoRA）、**Anima**（重点：Qwen3-0.6B TE + T5 旧分词器、网络 LoRA/LoKr/T-LoRA、attn flash/xformers/SDPA、8–24GB；全量微调 anima_train.py ~24GB；Fast 插件独立运行时；模型从 modelscope.cn 下载，分词器 vendored 全离线）；**Krea 2 不支持**；**Illusion 无专有条目**（走通用 sdxl 页）。
- **移植清单**：P0 config→TOML→accelerate 完整管线（trainer_mapping/sanitize/anima defaults/prediction-type）、数据集约定 + 自动建集 + 底模嗅探、任务生命周期（kill-tree/SSE/错误 tail）、训练监控聚合页、WD14/CL 打标（ONNX）；P1 schema 表单驱动（可转 JSON Schema 给 Rust 侧）、预设系统、config 导入导出校验、原生标签编辑器 undo/redo、checkpoint 管理 + 预览沙箱、大陆下载路由（ModelScope/hf-mirror 双后端 + repo 重映射）、离线 tokenizer cache；P2 Anima Fast 插件安装器、便携包、多 GPU、随机采样 prompt。
- **复杂度**：数据管线 中、训练引擎 高（仅编排则中；原生替代 PyTorch 数学栈接近不可行）、配置 低、UI 中（全新 Web UI 反而摆脱 patch 泥潭）、队列 低、监控 中。
- **坑**：三快照同步上游全手工；frontend/dist 无源码；requirements 硬 pin + onnxruntime-gpu 共存脆弱；Windows torch_compile 崩（sanitize 强删）；Anima automagic/CAME + fp16 → loss=nan（自动降 bf16）；LoKr full_matrix 高风险警告；accelerate resume 缺 step 元数据；打标模型下载失败阻塞任务；端口互抢（已有保护）；**AGPL-3.0 许可边界**。

#### A.4 三项目横向结论（直接支撑架构决策）

1. **训练内核统一采用"配置文件 + accelerate launch + stdout 事件"模式**（kohya_ss 与 lora-scripts-next 殊途同归），Rust 侧以 IPC/Stdio 编排（ADR-001），无需自研训练内核即可覆盖 SDXL/Anima；Krea 2 追加 ai-toolkit 后端。
2. **参数面以 kohya_ss 为全集基准**（25 优化器/11 调度器/20 网络结构/缓存/噪声技巧/元数据），以 lora-scripts-next 为 UX 与默认值基准，以 ai-toolkit 为时间步/损失技巧与扩展机制基准。
3. **产物互操作是硬契约**：kohya 元数据（ss_ 前缀）与 keymaps 必须字节级兼容（R9）。
4. **三个项目各自的"环境地狱/技术债"论证了 Rust 重写与全新 UI 的必要性**：gradio 硬锁、vendored 无源码前端、巨型函数、任务不持久化、无队列。

### B. 术语表

| 术语 | 含义 |
|---|---|
| 丹方 Recipe | 一组完整训练配置（含网络结构、优化器、调度、采样设置），按模型族校验 |
| 计算内核 | Python 训练运行时（sd-scripts / ai-toolkit），以 IPC/Stdio 与 Rust 引擎通信 |
| 兼容引擎 | Rust 侧对 Python 内核的编排层：`Trainer`/`CompatBackend` trait、进程监督、事件协议 |
| 远期探索 | 不排期、不承诺的候选能力（Rust ONNX 打标、candle 推理/训练、GGUF 导出） |
| 桶 bucket | 按长宽比分组的分辨率桶采样（SDXL/DiT 训练标准做法） |
| 火候 | 训练速度/状态的拟物化表达（文火=暂停，武火=运行，熄火=停止） |

---

*本文档随架构决策更新；v1.1 起架构决策冻结（ADR-001），仅功能与里程碑可迭代。*
