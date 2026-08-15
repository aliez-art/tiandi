# 丹炉 DanLu

> **你的私人 LoRA 训练丹炉。** 投料 · 控火 · 开炉 —— 把「图包 → 丹方 → 丹药（LoRA）」压缩成一个顺手的本地工作台。

丹炉是一个用 **Rust 重写**的私人（单人、本机）LoRA 训练工具，融合了三个成熟 Python 项目的能力与经验：

- [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) —— 模块化训练框架、时间步/损失技巧、Krea 2 / Anima 训练实现
- [bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss) —— 最全训练参数面（25 优化器 / 20 网络结构 / 缓存 / 元数据）
- [wochenlong/lora-scripts-next](https://github.com/wochenlong/lora-scripts-next) —— 现代 UX、预设、监控、大陆网络适配

**目标模型**：SDXL 族（Illusion、NoobAI）与 DiT 族（Anima、Krea 2）。

## 当前状态

| 项目 | 状态 |
|---|---|
| 产品需求文档（PRD v1.0） | ✅ 已定稿（三项目代码审查已合入） |
| 架构设计（v0.9 草案） | ✅ `docs/architecture.md` |
| 里程碑路线图 | ✅ `docs/roadmap.md` |
| 模型支持矩阵 | ✅ `docs/model-support.md` |
| Cargo workspace 骨架（10 crates） | ✅ `cargo check` 通过 |

> ⚠️ 本仓库目前是 **PRD + 骨架** 阶段，M0（可运行骨架）尚未实施。正式代码从里程碑 M0 开始落地。

## 快速开始（当前仅骨架）

```bash
cargo check --workspace   # 验证骨架可编译
```

## 仓库布局

```text
danlu/
├── PRD.md                # 产品需求文档（v1.0）
├── docs/
│   ├── architecture.md   # 技术架构（crate 分层、领域模型、引擎协议）
│   ├── roadmap.md        # 里程碑 M0–M5
│   └── model-support.md  # 模型支持矩阵与训练要点
├── crates/
│   ├── danlu-core        # 领域模型与用例、任务状态机
│   ├── danlu-state       # SQLite 持久化
│   ├── danlu-dataset     # 数据管线（图像/去重/分桶/缓存）
│   ├── danlu-recipe      # 丹方 schema、校验、预设
│   ├── danlu-engine      # Trainer trait、事件协议
│   ├── danlu-engine-compat   # 兼容引擎（sd-scripts / ai-toolkit 双后端）
│   ├── danlu-engine-native   # 原生引擎（candle，R&D）
│   ├── danlu-server      # axum REST + SSE
│   ├── danlu-cli         # danlu run/import/doctor
│   └── danlu-app         # Tauri 2 桌面壳
└── Cargo.toml            # workspace
```

## 核心架构决策（详见 PRD §8）

1. **Rust 为核心，引擎可替换**：训练引擎抽象为 `Trainer` trait；第一代「兼容引擎」以受管 Python 运行时（sd-scripts / ai-toolkit）保证立即可用，随后按里程碑用 candle 渐进原生化。
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
