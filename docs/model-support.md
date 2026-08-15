# 天地熔炉 Tiandi Furnace — 模型支持矩阵（v1.1）

> 数据来源：PRD 附录 A 的三项目代码审查 + 2026-08 公开资料（HuggingFace / diffusers 文档）。表格中的"支持"指**训练侧**支持（非仅推理）。

## 1. 目标模型

| 模型 | 家族 | 架构要点 | 文本编码器 | VAE | 预测目标 | 训练参考实现 |
|---|---|---|---|---|---|---|
| NoobAI（noobai-XL） | SDXL-1.0 | UNet（3.5B）+ 双 TE | CLIP-L + OpenCLIP-G | SDXL VAE（可换） | eps（社区 v-pred 变体） | kohya_ss `sdxl_train_network.py`；lora-scripts-next sdxl master 页 |
| Illusion（SDXL-1.0 系） | SDXL-1.0 | 同上（具体检查点由用户在注册表指定，如含 refiner/VAE 变体需选择对应 VAE 文件） | 同上 | 同上 | eps | 同上（同一 SDXL 管线） |
| Anima（CircleStone-Labs，2B 动漫 DiT） | DiT | CosmosTransformer3DModel（ai-toolkit 目标层），flow-matching | Qwen3-0.6B（+ T5 旧分词器；kohya_ss 另支持 llm_adapter） | Qwen-Image VAE | flow-matching | ai-toolkit `anima` arch；kohya_ss `anima_train_network.py`；lora-scripts-next `lora_anima`/`tlora_anima`/LoKr + `anima_train.py` 全量 + Fast 插件（sorryhyun/anima_lora） |
| Krea 2（Krea 官方） | DiT | Krea2Transformer2DModel / SingleStreamDiT | Qwen3-VL（ai-toolkit krea2 实现） | Qwen-Image VAE | flow-matching（按 diffusers 实现） | **仅 ai-toolkit `krea2` arch**（完整 LoRA 实现，含 edit 模式、assistant LoRA/ARA）；sd-scripts 生态暂无 |

## 2. 三项目支持现状核实（2026-08 代码审查）

| 能力 | ai-toolkit | kohya_ss | lora-scripts-next |
|---|---|---|---|
| SDXL LoRA | legacy 路径（代码完整，无示例配置） | ✅ 成熟（sdxl_train_network.py） | ✅ 成熟（v-pred / rectified-flow 选项） |
| Anima LoRA | ✅ anima arch（完整） | ✅ anima_train_network.py + Anima LLLite | ✅ lora / tlora / LoKr / finetune / Fast |
| Krea 2 | ✅ krea2 arch（唯一参考） | ❌ 零引用 | ❌ 零引用 |
| 其他 | flux/qwen-image/wan 等大量 | SD3/Hunyuan/Lumina/Flux | sd1.5/dreambooth/flux |

## 3. 熔炉支持策略（按里程碑）

| 里程碑 | SDXL 族（NoobAI/Illusion） | DiT 族（Anima） | DiT 族（Krea 2） |
|---|---|---|---|
| M1 | ✅ BackendSdScripts（P0 首跑） | — | — |
| M2 | ✅ 完善（队列/续训/监控） | ✅ BackendSdScripts（anima_train_network.py，参考 lora-scripts-next 参数组合） | — |
| M3 | ✅ 自动化 CLI + 缓存管理 | ✅ 同上 | ✅ BackendAiToolkit（YAML + run.py）+ 许可调研 |
| 远期探索 | — | — | candle 原生训练（不排期，见 PRD §8.4） |

> 注：训练计算内核始终为 Python（IPC/Stdio 通信，PRD ADR-001）；"远期探索"仅指 Rust 原生候选能力，不进入主线。

## 4. 各模型训练要点备忘（供丹方预设取值参考）

- **NoobAI/Illusion（SDXL）**：分辨率 1024 桶（divisibility 8）；双 TE 输出缓存（cache_text_encoder_outputs）；block weights 为 SDXL 特色参数；NoobAI 类社区底模常用 v-prediction（丹方需按检查点选择）；零 SNR / min-SNR 可选。
- **Anima**：默认采样 1024×1024、CFG 4.5、40 步（lora-scripts-next 默认值）；attn 模式 flash/xformers/SDPA 需探测（Windows 下 xformers↔SDPA 默认切换）；CAME/automagic 优化器 + fp16 易 NaN（自动降 bf16）；T-LoRA（tlora_anima）与 LoKr 已验证；全量微调约 24GB 显存；模型与分词器可完全离线（ModelScope 下载 + vendored tokenizer）。
- **Krea 2**：目标层 SingleStreamDiT（ai-toolkit 实现细节为准）；Qwen3-VL TE 与 Qwen-Image VAE 需随基底模型一起注册；edit 模式训练产物在 ComfyUI 需要 Krea2-Ostris-Edit 配套节点（未装则只训文生图模式）；**权重许可（官方 Krea org 发布 + 社区转换）与 Qwen3-VL 许可需在 M3 前置调研**。

## 5. 待办调研清单（M3 前置）

- [ ] Krea 2 官方权重许可（训练/私人使用边界）与 Qwen3-VL 许可
- [ ] Krea 2 训练侧是否已有社区实现（除 ai-toolkit 外）
- [ ] Anima 上游（circlestone-labs / sorryhyun/anima_lora）活跃度与 commit 锁定策略
- [ ] Illusion 具体检查点确认（与用户核对：哪个发行版、是否含 refiner/VAE 变体）
- [ ] （远期探索）candle 推理侧对 SDXL 的覆盖度验证——仅在原生能力进入主线评审时进行
