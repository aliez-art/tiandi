# 天地熔炉 Tiandi Furnace

> **你的私人 LoRA 训练熔炉。** 投料 · 控火 · 开炉 —— 把「图包 → 丹方 → 丹药（LoRA）」压缩成一个顺手的本地工作台。

熔炉是一个用 **Rust 重写**的私人（单人、本机）LoRA 训练工具，融合了三个成熟 Python 项目的能力与经验：

- [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) —— 模块化训练框架、时间步/损失技巧、Krea 2 / Anima 训练实现
- [bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss) —— 最全训练参数面（25 优化器 / 20 网络结构 / 缓存 / 元数据）
- [wochenlong/lora-scripts-next](https://github.com/wochenlong/lora-scripts-next) —— 现代 UX、预设、监控、大陆网络适配

**目标模型**：SDXL 族（Illusion、NoobAI）与 DiT 族（Anima、Krea 2）。

## 当前状态

| 项目 | 状态 |
|---|---|
| 产品需求文档（PRD v1.1） | ✅ 已定稿（含 ADR-001：Rust 控制/数据引擎 + Python 训练内核，IPC/Stdio） |
| 架构设计（v1.1） | ✅ `docs/architecture.md`（含 IPC/Stdio 协议细节） |
| 里程碑路线图 | ✅ `docs/roadmap.md`（M0–M3 + 远期探索） |
| 模型支持矩阵 | ✅ `docs/model-support.md` |
| **M0 核心骨架** | ✅ 完成：core / state / server / cli / engine（31 项测试全绿 + clippy 0 警告 + 冒烟通过） |
| M0 剩余：Tauri 壳 + React 空壳 | ⏳ 浏览器模式（`tiandi server --web`）已可先行使用 |

> 当前状态：**M0 核心完成，M1（SDXL 全流程）未开始**。快速体验：`tiandi init` → `tiandi server`。

## 快速开始（M0 骨架）

```bash
cargo build --release        # 构建 tiandi 二进制
tiandi init ~/tiandi-ws      # 建工作区（models/datasets/recipes/runs/vault + tiandi.db）
tiandi doctor                # 环境体检（磁盘/内存/GPU-CUDA/端口）
tiandi server --dir ~/tiandi-ws --web   # 点火（127.0.0.1:18765，自动开浏览器）

# 冒烟：创建模拟炼丹任务，观察状态机与事件流
curl -X POST 'http://127.0.0.1:18765/api/runs?simulate=1' -H 'content-type: application/json' -d '{}'
curl -N 'http://127.0.0.1:18765/api/runs/all/events'      # SSE 事件流（进度/指标/采样/状态）
```

## 仓库布局

```text
tiandi/
├── PRD.md                # 产品需求文档（v1.1）
├── docs/
│   ├── architecture.md   # 技术架构（crate 分层、领域模型、IPC/Stdio 引擎协议）
│   ├── roadmap.md        # 里程碑 M0–M3 + 远期探索
│   └── model-support.md  # 模型支持矩阵与训练要点
├── crates/
│   ├── tiandi-core        # 领域模型与用例、任务状态机
│   ├── tiandi-state       # SQLite 持久化
│   ├── tiandi-dataset     # 数据管线（图像/去重/分桶/缓存）
│   ├── tiandi-recipe      # 丹方 schema、校验、预设
│   ├── tiandi-engine      # Trainer trait、IPC 事件协议
│   ├── tiandi-engine-compat   # Python 内核编排（sd-scripts / ai-toolkit 双后端）
│   ├── tiandi-engine-native   # candle（远期探索，不排期）
│   ├── tiandi-server      # axum REST + SSE
│   ├── tiandi-cli         # tiandi run/import/doctor
│   └── tiandi-app         # Tauri 2 桌面壳
└── Cargo.toml            # workspace
```

## 核心架构决策（详见 PRD §8）

1. **Rust 控制/数据引擎 + Python 训练计算内核（ADR-001）**：训练由受管 Python 子进程（sd-scripts / ai-toolkit，独立 venv + 锁版本）承担，Rust 侧以 **IPC/Stdio 流**编排（JSON Lines 双向事件/控制通道 + 心跳 + 文件监控冗余）；放弃 PyO3 绑定（理由见 PRD §8.3）。
2. **丹方（Recipe）一等公民**：训练配置为 TOML 文件，可命名/继承/版本化/校验。
3. **任务可断可续**：持久化队列 + 状态机 + manifest，崩溃后一键续丹。
4. **产物互操作**：kohya 元数据与 keymaps 字节级兼容，产物可被 ComfyUI / A1111 加载。

## 许可说明

本项目为全新 Rust 实现（clean-room 式重写，不复制参考项目代码）。参考项目许可：ai-toolkit（MIT）、kohya_ss 与 sd-scripts（Apache-2.0）、lora-scripts-next 主仓库（AGPL-3.0，仅约束复制其代码）。进程级调用 Python 训练运行时不受传染；私人使用合规。

## 文档

- [PRD（产品需求文档）](./PRD.md)
- [技术架构](./docs/architecture.md)
- [里程碑路线图](./docs/roadmap.md)
- [模型支持矩阵](./docs/model-support.md)
