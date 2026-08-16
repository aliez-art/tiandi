import { useEffect, useState } from 'react'
import {
  createDataset,
  createRecipe,
  createRun,
  deleteRecipe,
  importAsset,
  listDatasets,
  listModels,
  listRecipes,
  pickDir,
  pickFile,
  registerModel,
  startRun,
  type BaseModel,
  type Dataset,
  type RecipeView,
} from '../api'

const FAMILY_LABEL: Record<string, string> = {
  sdxl1: 'SDXL 1.0（NoobAI/Illusion）',
  dit_anima: 'Anima (DiT)',
  dit_krea2: 'Krea 2 (DiT)',
}

const OPTIMIZERS: [string, string][] = [
  ['adamw8bit', 'AdamW 8bit（推荐）'],
  ['adamw', 'AdamW'],
  ['adafactor', 'Adafactor'],
  ['prodigy', 'Prodigy'],
  ['lion', 'Lion'],
  ['lion8bit', 'Lion 8bit'],
  ['dadapt_adagrad', 'D-Adaptation AdaGrad'],
  ['came', 'CAME'],
  ['sgdnesterov', 'SGD Nesterov'],
]
const SCHEDULERS: [string, string][] = [
  ['cosine', 'Cosine（推荐）'],
  ['constant', 'Constant'],
  ['constant_with_warmup', 'Constant + Warmup'],
  ['linear', 'Linear'],
  ['cosine_with_restarts', 'Cosine Restarts'],
  ['polynomial', 'Polynomial'],
  ['inverse_sqrt', 'Inverse Sqrt'],
]
const NETWORKS: [string, string][] = [
  ['lora', 'LoRA（推荐）'],
  ['locon', 'LoCon'],
  ['lokr', 'LoKr'],
  ['loha', 'LoHa'],
  ['dora', 'DoRA'],
  ['ia3', 'iA3'],
  ['tlora', 'T-LoRA'],
]
const PRECISIONS: [string, string][] = [
  ['bf16', 'bf16（推荐）'],
  ['fp16', 'fp16'],
  ['fp32', 'fp32'],
]
const PREDICTIONS: [string, string][] = [
  ['', '自动'],
  ['epsilon', 'epsilon（常规）'],
  ['v', 'v（V 预测模型）'],
]

type FieldDef = {
  key: string
  label: string
  /** 悬停在 "?" 上显示的参数说明 */
  hint: string
  /** 输入框示例值（placeholder） */
  placeholder?: string
  kind: 'num' | 'text' | 'select' | 'bool' | 'textlist'
  step?: string
  options?: [string, string][]
  /** 仅 SDXL 族显示 */
  sdxlOnly?: boolean
}

/** 训练参数表单：分区卡片式（lora-scripts-next 风格），每组内按使用频率排序；悬停 "?" 查看说明。 */
const FORM_SECTIONS: { title: string; fields: FieldDef[] }[] = [
  {
    title: '训练用模型',
    fields: [
      {
        key: 'trigger_word',
        label: '触发词',
        hint: '训练概念名（如 k2test）。推理时用它唤起 LoRA；可留空。',
        placeholder: '如 mychar, mystyle',
        kind: 'text',
      },
      {
        key: 'prediction_type',
        label: '预测类型',
        hint: 'V 预测模型（如 NoobAI vPred）必须选 v；其余用自动。',
        kind: 'select',
        options: PREDICTIONS,
      },
    ],
  },
  {
    title: '数据集设置',
    fields: [
      {
        key: 'resolution',
        label: '分辨率',
        hint: 'SDXL / Anima / Krea 2 用 1024；图小可降（如 768）。',
        placeholder: '1024',
        kind: 'num',
      },
      {
        key: 'enable_bucket',
        label: '分桶（保持纵横比）',
        hint: '按原图纵横比分桶训练，避免强制裁剪。',
        kind: 'bool',
      },
      {
        key: 'min_bucket_reso',
        label: '分桶最小分辨率',
        hint: '分桶边长的下限（小图不低于该值缩放）。',
        placeholder: '256',
        kind: 'num',
      },
      {
        key: 'max_bucket_reso',
        label: '分桶最大分辨率',
        hint: '分桶边长的上限（防止超大图爆显存）。',
        placeholder: '1024',
        kind: 'num',
      },
      {
        key: 'bucket_reso_steps',
        label: '分桶步长',
        hint: '分桶分辨率的递增步长（64 即按 64 的倍数切分）。',
        placeholder: '64',
        kind: 'num',
      },
      {
        key: 'bucket_no_upscale',
        label: '分桶不放大',
        hint: '禁止把小图放大到目标桶分辨率（避免模糊放大）。',
        kind: 'bool',
      },
      {
        key: 'batch_size',
        label: '批大小',
        hint: '1 通常足够；>1 需更大显存。',
        placeholder: '1',
        kind: 'num',
      },
      {
        key: 'reg_data_dir',
        label: '正则化数据集（可选）',
        hint: '防过拟合的辅助图片集（与训练集同主题、不带描述标签的图），用于正则化。',
        kind: 'text',
      },
      {
        key: 'prior_loss_weight',
        label: '正则化损失权重',
        hint: '正则化图片的损失权重（1.0 = 与训练图相同；调低可减少正则化影响）。',
        placeholder: '1.0',
        kind: 'num',
        step: 'any',
      },
      {
        key: 'keep_tokens',
        label: '保留开头标签数',
        hint: '打乱标签时保留前 N 个（通常是触发词/角色名）。',
        placeholder: '1',
        kind: 'num',
      },
      {
        key: 'shuffle_caption',
        label: '随机打乱标签',
        hint: '每步随机打乱逗号标签顺序（缓存 TE 输出时自动关闭）。',
        kind: 'bool',
      },
      {
        key: 'weighted_captions',
        label: '加权标签',
        hint: '支持 (tag:1.2) 形式的加权标签；与随机打乱标签不推荐同开。',
        kind: 'bool',
      },
      {
        key: 'caption_dropout_rate',
        label: '标签丢弃率',
        hint: '随机丢弃描述的比例（0.05 常规；0 = 从不丢弃）。',
        placeholder: '0.05',
        kind: 'num',
        step: 'any',
      },
      {
        key: 'caption_dropout_every_n_epochs',
        label: '标签丢弃间隔（轮）',
        hint: '每 N 轮整批丢弃一次描述标签（0 = 关闭）。',
        placeholder: '0',
        kind: 'num',
      },
      {
        key: 'caption_tag_dropout_rate',
        label: '标签词丢弃率',
        hint: '随机丢弃单个标签词的比例（0.0 = 关闭）。',
        placeholder: '0.0',
        kind: 'num',
        step: 'any',
      },
    ],
  },
  {
    title: '网络设置',
    fields: [
      {
        key: 'network_type',
        label: '网络类型',
        hint: 'LoRA 最通用；DoRA 效果更稳但更慢；LoCon/LoHa/LoKr 需配合卷积维度。',
        kind: 'select',
        options: NETWORKS,
      },
      {
        key: 'network_dim',
        label: '维度 dim（rank）',
        hint: 'LoRA 秩。角色/风格 16~64；Krea 2 建议 32；过大学过拟合。',
        placeholder: '16',
        kind: 'num',
      },
      {
        key: 'network_alpha',
        label: 'Alpha',
        hint: '缩放系数；常与 dim 相等（Krea 2 建议 alpha = rank）。',
        placeholder: '16',
        kind: 'num',
      },
      {
        key: 'conv_dim',
        label: '卷积维度（LoCon/LoHa/LoKr）',
        hint: '卷积层 rank；用 LoCon/LoHa/LoKr 时设置，常与 dim 相同。',
        placeholder: '16',
        kind: 'num',
      },
      {
        key: 'conv_alpha',
        label: '卷积 Alpha',
        hint: '卷积层 alpha；常与 conv_dim 相等。',
        placeholder: '16',
        kind: 'num',
      },
      {
        key: 'network_dropout',
        label: '网络 Dropout',
        hint: '权重随机丢弃比例（0~1）。防过拟合，0.05~0.1 可试；默认关闭。',
        placeholder: '0.05',
        kind: 'num',
        step: 'any',
      },
      {
        key: 'rank_dropout',
        label: 'Rank Dropout',
        hint: '行级丢弃（0~1）；与 network_dropout 类似，作用于 rank 维度。',
        placeholder: '0.05',
        kind: 'num',
        step: 'any',
      },
      {
        key: 'module_dropout',
        label: 'Module Dropout',
        hint: '模块级丢弃（0~1）；按模块整体丢弃，更强正则。',
        placeholder: '0.05',
        kind: 'num',
        step: 'any',
      },
      {
        key: 'network_weights',
        label: '预训练权重',
        hint: '从已训练的 LoRA 权重继续训练（.safetensors）；留空 = 从头训练。',
        placeholder: '留空 = 从头训练',
        kind: 'text',
      },
      {
        key: 'scale_weight_norms',
        label: '权重范数缩放',
        hint: '按范数归一化 LoRA 权重（1.0 关闭；>0 可提升训练稳定性）。',
        placeholder: '1.0',
        kind: 'num',
        step: 'any',
      },
      {
        key: 'network_args_custom',
        label: '网络自定义参数',
        hint: '透传给网络的额外参数，每行一个 k=v（如 conv_group_size=4）。',
        placeholder: '每行一个 k=v',
        kind: 'textlist',
      },
      {
        key: 'block_weights',
        label: 'Block 权重（SDXL）',
        hint: '25 个逗号分隔值（0/1），强化 UNet 指定层，如 0,...,1,1,1,1,1,...,1。仅 SDXL 族。',
        placeholder: '0,0,0,...,1,1,1,1,1,...,0',
        kind: 'text',
        sdxlOnly: true,
      },
    ],
  },
  {
    title: '学习率与优化器',
    fields: [
      {
        key: 'optimizer',
        label: '优化器',
        hint: 'AdamW8bit 兼容性最好；Prodigy/CAME 可自适应学习率。',
        kind: 'select',
        options: OPTIMIZERS,
      },
      {
        key: 'learning_rate',
        label: '学习率',
        hint: '常规 1e-4；概念难学可试 2e-4~5e-4；过低学不动，过高学崩。',
        placeholder: '0.0001',
        kind: 'num',
        step: 'any',
      },
      {
        key: 'unet_lr',
        label: 'UNet 学习率',
        hint: '仅 UNet 的学习率；留空 = 跟随主学习率。',
        placeholder: '0.0001',
        kind: 'num',
        step: 'any',
      },
      {
        key: 'text_encoder_lr',
        label: '文本编码器学习率',
        hint: 'TE 学习率；留空 = 跟随主学习率；缓存 TE 输出时自动为 0（SDXL 常用 5e-5）。',
        placeholder: '0.00005',
        kind: 'num',
        step: 'any',
      },
      {
        key: 'lr_scheduler',
        label: '学习率调度',
        hint: 'cosine 常规；短训用 constant_with_warmup。',
        kind: 'select',
        options: SCHEDULERS,
      },
      {
        key: 'lr_warmup_ratio',
        label: '预热比例',
        hint: '前 N% 步学习率线性升温（0.05~0.1 常见）。',
        placeholder: '0.1',
        kind: 'num',
        step: 'any',
      },
      {
        key: 'lr_scheduler_num_cycles',
        label: '调度循环数',
        hint: 'cosine_with_restarts 的循环次数（1 = 不重启）。',
        placeholder: '1',
        kind: 'num',
      },
      {
        key: 'loss_type',
        label: '损失类型',
        hint: '训练损失函数：L2 常规；L1 更稳；Huber/Smooth L1 介于两者之间。',
        kind: 'select',
        options: [
          ['', '自动（默认）'],
          ['l1', 'L1'],
          ['l2', 'L2'],
          ['huber', 'Huber'],
          ['smooth_l1', 'Smooth L1'],
        ],
      },
      {
        key: 'optimizer_args_custom',
        label: '优化器自定义参数',
        hint: '透传给优化器的额外参数，每行一个 k=v（如 weight_decay=0.01）。',
        placeholder: '每行一个 k=v',
        kind: 'textlist',
      },
    ],
  },
  {
    title: '训练相关参数',
    fields: [
      {
        key: 'train_text_encoder',
        label: '训练文本编码器（TE）',
        hint: 'LoRA 训练通常不需要（保持文本理解冻结即可）。开启会同时微调 TE——质量可能更好但显存/时间增加，且会强制关闭"缓存文本编码器输出"。Anima 的 Qwen3 TE 需要时可开启。',
        kind: 'bool',
      },
      {
        key: 'max_train_epochs',
        label: '训练轮数',
        hint: '每轮 = 全部图片过一遍；小数据集可多轮，首次建议 1 轮试跑。',
        placeholder: '10',
        kind: 'num',
      },
      {
        key: 'max_train_steps',
        label: '总步数上限',
        hint: '覆盖 epochs×图数×repeats 的自动估算；留空 = 自动。',
        placeholder: '留空 = 自动',
        kind: 'num',
      },
      {
        key: 'gradient_accumulation_steps',
        label: '梯度累积',
        hint: '等效放大批大小（2 = 每 2 步更新一次参数）。',
        placeholder: '1',
        kind: 'num',
      },
      {
        key: 'gradient_checkpointing',
        label: '梯度检查点',
        hint: '省显存、略慢；16GB 显卡建议开启。',
        kind: 'bool',
      },
      {
        key: 'max_grad_norm',
        label: '梯度裁剪',
        hint: '防梯度爆炸（1.0 常规）。',
        placeholder: '1.0',
        kind: 'num',
        step: 'any',
      },
      { key: 'seed', label: '随机种子', hint: '固定随机性，便于复现。', placeholder: '42', kind: 'num' },
      {
        key: 'clip_skip',
        label: 'CLIP 跳层',
        hint: '跳过 CLIP 最后 N 层输出（SDXL 系常用 1~2）；留空 = 默认。',
        placeholder: '留空 = 默认',
        kind: 'num',
      },
      {
        key: 'max_token_length',
        label: '最大 Token 长度',
        hint: '75/150/225/300；长描述用 225+（需同时开启加权标签）。',
        placeholder: '75',
        kind: 'num',
      },
      {
        key: 'min_timestep',
        label: '最小时间步',
        hint: '噪声范围裁剪下限（0 = 默认）；减小可加速但影响细节。',
        placeholder: '0',
        kind: 'num',
      },
      {
        key: 'max_timestep',
        label: '最大时间步',
        hint: '噪声范围裁剪上限（1000 = 默认）；减小可加速但影响结构。',
        placeholder: '1000',
        kind: 'num',
      },
      {
        key: 'zero_terminal_snr',
        label: 'Zero Terminal SNR',
        hint: '末端噪声归零，提升暗部/高光表现（需配合 v 预测或重调度）。',
        kind: 'bool',
      },
    ],
  },
  {
    title: '精度与缓存',
    fields: [
      {
        key: 'mixed_precision',
        label: '混合精度',
        hint: 'bf16 常规；老显卡用 fp16。',
        kind: 'select',
        options: PRECISIONS,
      },
      {
        key: 'full_fp16',
        label: '全程 fp16',
        hint: '整个训练流程使用 fp16（省显存；混合精度异常/老显卡时用）。',
        kind: 'bool',
      },
      {
        key: 'full_bf16',
        label: '全程 bf16',
        hint: '整个训练流程使用 bf16（Ampere+ 显卡推荐，更稳）。',
        kind: 'bool',
      },
      {
        key: 'no_half_vae',
        label: 'VAE 不降精度',
        hint: 'VAE 保持全精度（避免半精度下 VAE 输出异常）。',
        kind: 'bool',
      },
      {
        key: 'xformers',
        label: 'xformers',
        hint: '启用 xformers 注意力加速（省显存提速；部分环境需安装）。',
        kind: 'bool',
      },
      {
        key: 'lowram',
        label: '低显存模式',
        hint: '低显存模式：权重驻留 CPU、按需搬运（更慢更省显存）。',
        kind: 'bool',
      },
      {
        key: 'cache_latents',
        label: '缓存 latent',
        hint: '预编码图像 latent 到磁盘，加速训练。',
        kind: 'bool',
      },
      {
        key: 'cache_text_encoder_outputs',
        label: '缓存文本编码器输出',
        hint: 'SDXL 提速关键（省一半显存）；Anima 自动关闭（需训练 Qwen3 TE）。',
        kind: 'bool',
      },
      {
        key: 'cache_text_encoder_outputs_to_disk',
        label: '缓存 TE 输出到磁盘',
        hint: '把编码结果落盘，进一步省显存；首次较慢。',
        kind: 'bool',
      },
      {
        key: 'persistent_data_loader_workers',
        label: '常驻数据加载线程',
        hint: '保持 DataLoader 工作线程常驻（省去每轮重启开销）。',
        kind: 'bool',
      },
      {
        key: 'vae_batch_size',
        label: 'VAE 批大小',
        hint: 'VAE 编码的批大小；留空 = 自动。',
        placeholder: '留空 = 自动',
        kind: 'num',
      },
    ],
  },
  {
    title: '保存设置',
    fields: [
      {
        key: 'save_every_n_epochs',
        label: '保存间隔（轮）',
        hint: '每 N 轮存一个 LoRA 检查点（断点续训也依赖它）。',
        placeholder: '1',
        kind: 'num',
      },
      {
        key: 'save_every_n_steps',
        label: '保存间隔（步）',
        hint: '每 N 步存一个检查点；留空 = 按轮保存。',
        placeholder: '留空 = 按轮保存',
        kind: 'num',
      },
      {
        key: 'save_state',
        label: '保存优化器状态',
        hint: '额外保存优化器状态（断点续训更完整，文件更大）。',
        kind: 'bool',
      },
      {
        key: 'save_last_n_states',
        label: '保留最近状态数',
        hint: '最多保留几个状态目录（防磁盘膨胀）。',
        placeholder: '3',
        kind: 'num',
      },
      {
        key: 'save_last_n_epochs_state',
        label: '保留最近轮状态数',
        hint: '最多保留最近 N 轮的优化器状态；留空 = 全部保留。',
        placeholder: '留空 = 全部保留',
        kind: 'num',
      },
      {
        key: 'save_precision',
        label: '保存精度',
        hint: '检查点的保存精度：fp16 常规（文件小）；float 全精度更通用；bf16 更稳。',
        kind: 'select',
        options: [
          ['fp16', 'fp16'],
          ['float', 'float（fp32）'],
          ['bf16', 'bf16'],
        ],
      },
    ],
  },
  {
    title: '预览图设置',
    fields: [
      {
        key: 'sample_enabled',
        label: '训练中生成示例图',
        hint: '训练过程中按轮生成预览图（在训练控制台画廊展示）。关闭则跳过采样、训练更快。',
        kind: 'bool',
      },
      {
        key: 'sample_every_n_epochs',
        label: '示例图间隔（轮）',
        hint: '每 N 轮生成一批示例图（0 = 不生成）。训练过程可视化，强烈建议开启。',
        placeholder: '1',
        kind: 'num',
      },
      {
        key: 'sample_prompts',
        label: '示例图提示词',
        hint: '每行一个提示词；可含触发词。训练到每轮时按此出图。',
        placeholder: '每行一个提示词',
        kind: 'textlist',
      },
      {
        key: 'sample_steps',
        label: '示例图步数',
        hint: '生成示例图用的去噪步数（20~30 常见；越高越精细越慢）。',
        placeholder: '24',
        kind: 'num',
      },
      {
        key: 'guidance_scale',
        label: '示例图引导强度',
        hint: 'CFG 强度（4~7 常见；越高越贴提示词）。',
        placeholder: '5.0',
        kind: 'num',
        step: 'any',
      },
      {
        key: 'negative_prompt',
        label: '示例图负向提示词',
        hint: '示例图的负向提示词（如 lowres, bad anatomy）。',
        placeholder: 'lowres, bad anatomy',
        kind: 'text',
      },
      {
        key: 'sample_sampler',
        label: '示例图采样器',
        hint: 'euler_a 常用；示例图生成的扩散采样器。',
        placeholder: 'euler_a',
        kind: 'text',
      },
    ],
  },
  {
    title: '质量技巧（可选）',
    fields: [
      {
        key: 'min_snr_gamma',
        label: 'Min-SNR Gamma',
        hint: '5.0 常规；抑制噪声步权重失衡，留空关闭。',
        placeholder: '5.0',
        kind: 'num',
        step: 'any',
      },
      {
        key: 'noise_offset',
        label: '噪声偏移',
        hint: '0.02~0.05 提升明暗对比表现；留空关闭。',
        placeholder: '0.03',
        kind: 'num',
        step: 'any',
      },
      {
        key: 'adaptive_noise_scale',
        label: '自适应噪声偏移',
        hint: 'noise_offset 进阶版（按时间步自适应）；0.1 左右可试。',
        placeholder: '0.1',
        kind: 'num',
        step: 'any',
      },
      {
        key: 'multires_noise_iterations',
        label: '多分辨率噪声迭代',
        hint: '在多个分辨率加噪再混合，提升大图结构（6~10 常见）；0 = 关闭。',
        placeholder: '6',
        kind: 'num',
      },
      {
        key: 'multires_noise_discount',
        label: '多分辨率噪声折扣',
        hint: '多分辨率噪声的折扣系数（0.3 左右常见）。',
        placeholder: '0.3',
        kind: 'num',
        step: 'any',
      },
    ],
  },
]

/** 随 payload 保存但不属于表单字段的特殊键（底模/数据集路径等），不受可见字段过滤影响。 */
const SPECIAL_PAYLOAD_KEYS = new Set([
  'model_path',
  'vae_path',
  'te_path',
  'dataset_dir',
  'full_finetune',
  'name',
  'prediction_type',
])

/** 新建丹方表单的默认值（与后端 RecipeData 对齐）。 */
function defaultData(): Record<string, unknown> {
  return {
    learning_rate: 0.0001,
    optimizer: 'adamw8bit',
    lr_scheduler: 'cosine',
    lr_warmup_ratio: 0.1,
    network_dim: 16,
    network_alpha: 16,
    network_type: 'lora',
    max_train_epochs: 10,
    batch_size: 1,
    resolution: 1024,
    enable_bucket: true,
    min_bucket_reso: 256,
    max_bucket_reso: 1024,
    bucket_reso_steps: 64,
    bucket_no_upscale: true,
    cache_latents: true,
    cache_text_encoder_outputs: true,
    mixed_precision: 'bf16',
    gradient_checkpointing: true,
    gradient_accumulation_steps: 1,
    max_grad_norm: 1.0,
    seed: 42,
    min_snr_gamma: 5,
    shuffle_caption: true,
    weighted_captions: false,
    keep_tokens: 1,
    caption_dropout_rate: 0.05,
    save_precision: 'fp16',
    save_every_n_epochs: 1,
    sample_enabled: true,
    sample_every_n_epochs: 1,
    sample_prompts: [] as string[],
    network_args_custom: [] as string[],
    optimizer_args_custom: [] as string[],
    sample_sampler: 'euler_a',
  }
}

/** 参数名 + "?" 帮助标识（悬停显示参数说明）。 */
function FieldLabel(props: { field: FieldDef }) {
  return (
    <span className="field-name">
      {props.field.label}
      <span className="field-help" title={props.field.hint} aria-label={props.field.hint}>
        ?
      </span>
    </span>
  )
}

/** 单个参数控件。 */
function FieldInput(props: { field: FieldDef; value: unknown; onChange: (v: unknown) => void }) {
  const { field, value, onChange } = props
  switch (field.kind) {
    case 'bool':
      return (
        <label className="field chk-field" title={field.hint}>
          <FieldLabel field={field} />
          <input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />
        </label>
      )
    case 'select':
      return (
        <label className="field" title={field.hint}>
          <FieldLabel field={field} />
          <select value={String(value ?? '')} onChange={(e) => onChange(e.target.value)}>
            {field.options?.map(([v, l]) => (
              <option key={v} value={v}>
                {l}
              </option>
            ))}
          </select>
        </label>
      )
    case 'textlist':
      return (
        <label className="field wide" title={field.hint}>
          <FieldLabel field={field} />
          <textarea
            rows={2}
            placeholder={field.placeholder}
            value={Array.isArray(value) ? value.join('\n') : String(value ?? '')}
            onChange={(e) =>
              onChange(
                e.target.value
                  .split('\n')
                  .map((s) => s.trim())
                  .filter(Boolean),
              )
            }
          />
        </label>
      )
    case 'num':
      return (
        <label className="field" title={field.hint}>
          <FieldLabel field={field} />
          <input
            type="number"
            step={field.step ?? '1'}
            placeholder={field.placeholder}
            value={value === null || value === undefined || value === '' ? '' : Number(value)}
            onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))}
          />
        </label>
      )
    default:
      return (
        <label className="field" title={field.hint}>
          <FieldLabel field={field} />
          <input
            type="text"
            placeholder={field.placeholder}
            value={String(value ?? '')}
            onChange={(e) => onChange(e.target.value)}
          />
        </label>
      )
  }
}

/** 丹方页：底模/数据集选择 + 完整参数表单 + 保存/点火。
 *  `full=true` 为全量微调模式（无 LoRA 网络参数、可训练 TE）。 */
export function RecipeForm(props: { onCreated: (runId: string) => void; full?: boolean }) {
  const full = props.full ?? false
  const [recipes, setRecipes] = useState<RecipeView[]>([])
  const [models, setModels] = useState<BaseModel[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [recipeId, setRecipeId] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [family, setFamily] = useState('sdxl1')
  const [modelPath, setModelPath] = useState('')
  const [vaePath, setVaePath] = useState('')
  const [tePath, setTePath] = useState('')
  const [datasetDir, setDatasetDir] = useState('')
  const [data, setData] = useState<Record<string, unknown>>(defaultData())
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  const [msgError, setMsgError] = useState(false)
  /** 折叠卡片展开状态：默认展开前 3 组，其余收起；点击分组标题切换。 */
  const [open, setOpen] = useState<Set<string>>(new Set(['训练用模型', '数据集设置', '网络设置']))

  /** 状态消息：isError=true 渲染为红色错误样式（.status-line.error）。 */
  const showMsg = (m: string | null, isError = false) => {
    setMsg(m)
    setMsgError(isError)
  }

  const refresh = async () => {
    const [r, m, d] = await Promise.all([listRecipes(), listModels(), listDatasets()])
    setRecipes(r)
    setModels(m)
    setDatasets(d)
  }

  useEffect(() => {
    void refresh().catch(() => {})
  }, [])

  const setField = (k: string, v: unknown) =>
    setData((prev) => {
      const next = { ...prev, [k]: v }
      // 训练 TE 与 TE 输出缓存互斥：开启 TE 训练时自动关闭缓存（内核无法同时缓存与训练）
      if (k === 'train_text_encoder' && v === true) {
        next.cache_text_encoder_outputs = false
      }
      // 示例图开关 ↔ 采样间隔联动（0 = 关闭采样）
      if (k === 'sample_enabled') {
        next.sample_every_n_epochs = v === true ? 1 : 0
      } else if (k === 'sample_every_n_epochs' && typeof v === 'number') {
        next.sample_enabled = v > 0
      }
      return next
    })

  const toggleSection = (title: string) =>
    setOpen((prev) => {
      const next = new Set(prev)
      if (next.has(title)) next.delete(title)
      else next.add(title)
      return next
    })

  const basename = (p: string) => p.split(/[\\/]/).pop() ?? ''

  // ---------- 底模 / VAE / TE / 数据集选择 ----------

  const onPickModel = async () => {
    setBusy(true)
    showMsg(null)
    try {
      const path = await pickFile()
      if (path) {
        const imported = await importAsset('base_model', path)
        setModelPath(imported.path)
        // 底模同目录的配套资产（Anima/Krea 2 的 Qwen3 TE 与 VAE）自动导入并填表
        if (imported.vae_path) setVaePath(imported.vae_path)
        if (imported.te_path) setTePath(imported.te_path)
        // 已注册模型 → 同步丹方族（避免 sdxl1 丹方配 anima 底模之类的族错配）
        const known = models.find((m) => m.path === imported.path || m.path === path)
        if (known) setFamily(known.family)
        showMsg(`已选底模：${imported.path}`)
      } else {
        showMsg('未选择文件（已取消）')
      }
    } catch (e) {
      showMsg(`选择失败：${e instanceof Error ? e.message : String(e)}`, true)
    } finally {
      setBusy(false)
    }
  }

  const onPickVae = async () => {
    setBusy(true)
    showMsg(null)
    try {
      const path = await pickFile()
      if (path) {
        const imported = await importAsset('vae', path)
        setVaePath(imported.path)
        showMsg(`已选 VAE：${imported.path}`)
      } else {
        showMsg('未选择文件（已取消）')
      }
    } catch (e) {
      showMsg(`选择失败：${e instanceof Error ? e.message : String(e)}`, true)
    } finally {
      setBusy(false)
    }
  }

  const onPickTe = async () => {
    setBusy(true)
    showMsg(null)
    try {
      const path = await pickFile()
      if (path) {
        const imported = await importAsset('clip', path)
        setTePath(imported.path)
        showMsg(`已选文本编码器：${imported.path}`)
      } else {
        showMsg('未选择文件（已取消）')
      }
    } catch (e) {
      showMsg(`选择失败：${e instanceof Error ? e.message : String(e)}`, true)
    } finally {
      setBusy(false)
    }
  }

  const onPickDataset = async () => {
    setBusy(true)
    showMsg(null)
    try {
      const dir = await pickDir()
      if (dir) {
        setDatasetDir(dir)
        showMsg(`已选数据集：${dir}`)
      } else {
        showMsg('未选择目录（已取消）')
      }
    } catch (e) {
      showMsg(`选择失败：${e instanceof Error ? e.message : String(e)}`, true)
    } finally {
      setBusy(false)
    }
  }

  // 全量模式：过滤 LoRA 专属分区（训练用模型 / 网络设置）
  const hiddenSections = full ? ['训练用模型', '网络设置'] : []
  const sections = FORM_SECTIONS.filter((sec) => !hiddenSections.includes(sec.title))
  // 全量模式：精度与缓存区只显示混合精度
  const visibleFields = (sec: { title: string; fields: FieldDef[] }) => {
    if (full && sec.title === '精度与缓存') {
      return sec.fields.filter((f) => f.key === 'mixed_precision')
    }
    return sec.fields
  }

  // ---------- 保存 / 点火 ----------

  const buildPayload = () => {
    const payload = { ...data }
    const pt = payload.prediction_type
    if (pt === '' || pt === undefined || pt === null) delete payload.prediction_type
    // 全量模式：隐藏分区（训练用模型 / 网络设置）与「精度与缓存」中被过滤字段的
    // 默认值/历史残留不应随 payload 发送——仅保留当前可见字段 + 白名单特殊键
    if (full) {
      const visible = new Set<string>()
      for (const sec of sections) {
        for (const f of visibleFields(sec)) visible.add(f.key)
      }
      for (const k of Object.keys(payload)) {
        if (!visible.has(k) && !SPECIAL_PAYLOAD_KEYS.has(k)) delete payload[k]
      }
    }
    for (const k of Object.keys(payload)) {
      const v = payload[k]
      if (v === null || v === '' || (Array.isArray(v) && v.length === 0)) delete payload[k]
    }
    // 底模/数据集随丹方保存（自定义键，后端 RecipeData 忽略未知字段）
    if (modelPath) payload.model_path = modelPath
    if (vaePath) payload.vae_path = vaePath
    if (tePath) payload.te_path = tePath
    if (datasetDir) payload.dataset_dir = datasetDir
    if (full) payload.full_finetune = true
    return payload
  }

  /**
   * 保存丹方。
   * `overrideName`：点火流程传入（自动命名时表单 name 尚未更新，用显式值校验/保存）。
   */
  const onSave = async (overrideName?: string): Promise<string | null> => {
    const effName = (overrideName ?? name).trim()
    if (!effName) {
      showMsg('请填写丹方名称', true)
      return null
    }
    setBusy(true)
    showMsg(null)
    try {
      // 更新 = 先创建成功、再删旧（后端无 PUT；避免创建失败丢失旧丹方）
      const res = await createRecipe(effName, family, buildPayload())
      const newId = res.recipe.id
      if (recipeId && recipeId !== newId) {
        try {
          await deleteRecipe(recipeId)
        } catch (e) {
          // 删除旧丹方失败不阻塞（新丹方已保存成功）
          console.warn('删除旧丹方失败', e)
        }
      }
      setRecipeId(newId)
      showMsg(`丹方「${effName}」已保存`)
      await refresh()
      return newId
    } catch (e) {
      showMsg(`保存失败：${e instanceof Error ? e.message : String(e)}`, true)
      return null
    } finally {
      setBusy(false)
    }
  }

  const onFire = async () => {
    if (!modelPath || !datasetDir) {
      showMsg('请先选择底模与数据集', true)
      return
    }
    if (full && family === 'dit_krea2') {
      showMsg('Krea 2 全量训练暂不支持（ai-toolkit 后端仅 LoRA）', true)
      return
    }
    // 族匹配校验：底模族与丹方族不一致会在模型加载阶段崩溃，点火前拦截
    const knownModel = models.find((m) => m.path === modelPath)
    if (knownModel && knownModel.family !== family) {
      showMsg(
        `底模「${knownModel.name}」属于 ${knownModel.family} 族，与当前丹方族 ${family} 不匹配。请选择正确的模型族或底模`,
        true,
      )
      return
    }
    setBusy(true)
    showMsg(null)
    try {
      // 1. 丹方：无论是否已保存都先保存（表单即真理，改动必然生效）；空名自动命名
      const effName = name.trim() || basename(modelPath).replace(/\.safetensors$/i, '') + '-丹方'
      if (!name.trim()) setName(effName)
      const rid = await onSave(effName)
      if (!rid) return // 保存失败：中止点火（错误提示已由 onSave 设置）
      setBusy(true) // onSave 内部 finally 已复位 busy，这里重新锁住后续步骤
      // 2. 模型：按路径复用或自动注册
      let model = models.find((m) => m.path === modelPath)
      if (!model) {
        model = await registerModel({ name: basename(modelPath).replace(/\.safetensors$/i, ''), family, path: modelPath })
        setModels((prev) => [model as BaseModel, ...prev])
      }
      // 3. 数据集：按目录复用或自动注册
      let ds = datasets.find((d) => d.dir === datasetDir)
      if (!ds) {
        ds = await createDataset(basename(datasetDir) || '数据集', datasetDir)
        setDatasets((prev) => [ds as Dataset, ...prev])
      }
      // 4. 创建任务并点火
      const run = await createRun({ dataset_id: ds.id, recipe_id: rid, base_model_id: model.id })
      await startRun(run.id)
      showMsg('已保存丹方并点火')
      props.onCreated(run.id)
    } catch (e) {
      showMsg(`点火失败：${e instanceof Error ? e.message : String(e)}`, true)
    } finally {
      setBusy(false)
    }
  }

  // ---------- 载入已保存丹方 ----------

  const onLoadRecipe = (r: RecipeView) => {
    setRecipeId(r.id)
    setName(r.name)
    setFamily(r.family)
    const d = r.data as Record<string, unknown>
    // 单次合并默认值 + 丹方数据，并收敛联动规则（与 setField 保持一致）：
    // - train_text_encoder === true → 强制关闭 cache_text_encoder_outputs（互斥）
    // - sample_enabled 由 sample_every_n_epochs 推导（旧丹方无 sample_enabled 时兜底）
    const next = { ...defaultData(), ...d }
    if (next.train_text_encoder === true) next.cache_text_encoder_outputs = false
    next.sample_enabled = (next.sample_every_n_epochs as number) > 0
    setData(next)
    if (typeof d.model_path === 'string') setModelPath(d.model_path)
    if (typeof d.vae_path === 'string') setVaePath(d.vae_path)
    if (typeof d.te_path === 'string') setTePath(d.te_path)
    if (typeof d.dataset_dir === 'string') setDatasetDir(d.dataset_dir)
    showMsg(`已载入丹方「${r.name}」`)
  }

  const onDeleteRecipe = async (r: RecipeView) => {
    if (!window.confirm(`删除丹方「${r.name}」？`)) return
    try {
      await deleteRecipe(r.id)
      if (recipeId === r.id) {
        setRecipeId(null)
        setName('')
      }
      await refresh()
    } catch (e) {
      showMsg(`删除失败：${e instanceof Error ? e.message : String(e)}`, true)
    }
  }

  const datasetInfo = datasets.find((d) => d.dir === datasetDir)

  return (
    <div className="recipe-page">
      {/* 已保存丹方（按模式过滤） */}
      <div className="panel recipe-load">
        <div className="panel-title">
          <h2>已保存丹方{full ? '（全量微调）' : ''}</h2>
        </div>
        {recipes.filter((r) => full === ((r.data as Record<string, unknown>).full_finetune === true)).length === 0 ? (
          <p className="hint">暂无保存的{full ? '全量' : ''}丹方；在下方面板配置并点「保存丹方」。</p>
        ) : (
          <ul className="runs datasets">
            {recipes
              .filter((r) => full === ((r.data as Record<string, unknown>).full_finetune === true))
              .map((r) => (
                <li
                  key={r.id}
                  className={`run ${r.id === recipeId ? 'active' : ''}`}
                  tabIndex={0}
                  role="button"
                  onClick={() => onLoadRecipe(r)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      onLoadRecipe(r)
                    }
                  }}
                >
                  <span className="run-id">{r.name}</span>
                  <span className="run-state">{FAMILY_LABEL[r.family] ?? r.family}</span>
                  <span className="run-time">
                    {typeof r.data.learning_rate === 'number' ? `lr ${r.data.learning_rate} · ` : ''}
                    {typeof r.data.max_train_epochs === 'number' ? `${r.data.max_train_epochs} epochs` : ''}
                  </span>
                  <button
                    className="run-del"
                    title="删除丹方"
                    aria-label="删除丹方"
                    onClick={(e) => {
                      e.stopPropagation()
                      void onDeleteRecipe(r)
                    }}
                  >
                    ✕
                  </button>
                </li>
              ))}
          </ul>
        )}
      </div>

      {/* 丹方配置 */}
      <div className="panel">
        <h2>丹方配置</h2>

        {/* 底模与数据集（选择即用，保存时随丹方记录） */}
        <div className="form-sec">
          <h4>底模与数据集</h4>
          <div className="form-grid">
            <div className="field wide">
              <span>基底模型</span>
              <div className="pick-row">
                <button onClick={() => void onPickModel()} disabled={busy} className="secondary">
                  选择底模…
                </button>
                <select value={family} onChange={(e) => setFamily(e.target.value)} title="底模所属模型族">
                  <option value="sdxl1">SDXL（NoobAI/Illusion）</option>
                  <option value="dit_anima">Anima</option>
                  <option value="dit_krea2">Krea 2</option>
                </select>
              </div>
              <div className={`pick-path ${modelPath ? '' : 'dim'}`} title={modelPath}>
                {modelPath || '未选择底模（.safetensors）'}
              </div>
            </div>
            <div className="field wide">
              <span>数据集</span>
              <div className="pick-row">
                <button onClick={() => void onPickDataset()} disabled={busy} className="secondary">
                  选择数据集…
                </button>
              </div>
              <div className={`pick-path ${datasetDir ? '' : 'dim'}`} title={datasetDir}>
                {datasetDir
                  ? `${basename(datasetDir)}${datasetInfo ? `（${datasetInfo.image_count} 张）` : ''} · ${datasetDir}`
                  : '未选择数据集（图片文件夹，每图配同名 .txt 描述）'}
              </div>
              <p className="hint">
                训练次数 = 文件夹名前缀数字（如 <code>2_artstyle</code> 内图片各训练 2 次；无数字 = 1 次）
              </p>
            </div>
            <div className="field wide">
              <span>VAE（可选）</span>
              <div className="pick-row">
                <button onClick={() => void onPickVae()} disabled={busy} className="secondary" title="Anima / Krea 2 训练必需 VAE；SDXL 底模一般内嵌 VAE 可不选">
                  选择 VAE…
                </button>
                {vaePath && (
                  <button className="danger" onClick={() => setVaePath('')} title="清除 VAE">
                    清除
                  </button>
                )}
              </div>
              <div className={`pick-path ${vaePath ? '' : 'dim'}`} title={vaePath}>
                {vaePath || '未选择（Anima / Krea 2 会自动探测同目录 VAE）'}
              </div>
            </div>
            <div className="field wide">
              <span>文本编码器 / CLIP（可选）</span>
              <div className="pick-row">
                <button onClick={() => void onPickTe()} disabled={busy} className="secondary" title="选择文本编码器文件（训练 TE 时必需；不选则用底模自带/自动探测）">
                  选择文本编码器…
                </button>
                {tePath && (
                  <button className="danger" onClick={() => setTePath('')} title="清除文本编码器">
                    清除
                  </button>
                )}
              </div>
              <div className={`pick-path ${tePath ? '' : 'dim'}`} title={tePath}>
                {tePath || '未选择（用底模自带编码器；Anima 自动探测同目录 qwen3）'}
              </div>
            </div>
          </div>
        </div>

        {/* 基本 */}
        <div className="form-sec">
          <h4>丹方名称</h4>
          <div className="form-grid">
            <label className="field wide">
              <span>名称</span>
              <input value={name} onChange={(e) => setName(e.target.value)} placeholder="如 NoobAI 角色丹" />
            </label>
          </div>
        </div>

        {sections.map((sec) => {
          const isOpen = open.has(sec.title)
          return (
            <div key={sec.title} className="form-sec-card">
              <button
                type="button"
                className="form-sec-title"
                aria-expanded={isOpen}
                onClick={() => toggleSection(sec.title)}
              >
                <span className="form-sec-caret" aria-hidden="true">
                  {isOpen ? '▾' : '▸'}
                </span>
                {sec.title}
              </button>
              {isOpen && (
                <div className="form-sec-body">
                  <div className="form-grid">
                    {visibleFields(sec)
                      .filter((f) => !f.sdxlOnly || family === 'sdxl1')
                      .map((f) => (
                        <FieldInput key={f.key} field={f} value={data[f.key]} onChange={(v) => setField(f.key, v)} />
                      ))}
                  </div>
                </div>
              )}
            </div>
          )
        })}

        <div className="actions">
          <button onClick={() => void onSave()} disabled={busy} className="secondary">
            {busy ? '保存中…' : '保存丹方'}
          </button>
          <button onClick={() => void onFire()} disabled={busy} className="primary">
            {busy ? '点火中…' : '点火炼丹'}
          </button>
          {msg && <span className={`status-line${msgError ? ' error' : ''}`}>{msg}</span>}
        </div>
      </div>
    </div>
  )
}
