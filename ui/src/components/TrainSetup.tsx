import { useEffect, useState } from 'react'
import {
  createDataset,
  createRecipe,
  createRun,
  deleteRecipe,
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
  hint: string
  kind: 'num' | 'text' | 'select' | 'bool' | 'textlist'
  step?: string
  options?: [string, string][]
  /** 仅 SDXL 族显示 */
  sdxlOnly?: boolean
}

/** 完整训练参数表单（参考 kohya_ss / lora-scripts GUI 分区）。 */
const FORM_SECTIONS: { title: string; fields: FieldDef[] }[] = [
  {
    title: '概念与预测',
    fields: [
      {
        key: 'trigger_word',
        label: '触发词',
        hint: '训练概念名（如 k2test）。推理时用它唤起 LoRA；可留空。',
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
    title: '学习率与优化器',
    fields: [
      {
        key: 'learning_rate',
        label: '学习率',
        hint: '常规 1e-4；概念难学可试 2e-4~5e-4；过低学不动，过高学崩。',
        kind: 'num',
        step: 'any',
      },
      {
        key: 'text_encoder_lr',
        label: '文本编码器学习率',
        hint: 'TE 学习率；留空 = 跟随主学习率；缓存 TE 输出时自动为 0（SDXL 常用 5e-5）。',
        kind: 'num',
        step: 'any',
      },
      {
        key: 'unet_lr',
        label: 'UNet 学习率',
        hint: '仅 UNet 的学习率；留空 = 跟随主学习率。',
        kind: 'num',
        step: 'any',
      },
      { key: 'optimizer', label: '优化器', hint: 'AdamW8bit 兼容性最好。', kind: 'select', options: OPTIMIZERS },
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
        kind: 'num',
        step: 'any',
      },
    ],
  },
  {
    title: '网络结构',
    fields: [
      { key: 'network_type', label: '网络类型', hint: 'LoRA 最通用；DoRA 效果更稳但更慢。', kind: 'select', options: NETWORKS },
      {
        key: 'network_dim',
        label: '维度 dim（rank）',
        hint: 'LoRA 秩。角色/风格 16~64；Krea 2 建议 32；过大学过拟合。',
        kind: 'num',
      },
      {
        key: 'network_alpha',
        label: 'Alpha',
        hint: '缩放系数；常与 dim 相等（Krea 2 建议 alpha = rank）。',
        kind: 'num',
      },
      {
        key: 'block_weights',
        label: 'Block 权重（SDXL）',
        hint: '25 个逗号分隔值（0/1），强化 UNet 指定层，如 0,...,1,1,1,1,1,...,1。仅 SDXL 族。',
        kind: 'text',
        sdxlOnly: true,
      },
    ],
  },
  {
    title: '数据集',
    fields: [
      { key: 'batch_size', label: '批大小', hint: '1 通常足够；>1 需更大显存。', kind: 'num' },
      {
        key: 'resolution',
        label: '分辨率',
        hint: 'SDXL / Anima / Krea 2 用 1024；图小可降（如 768）。',
        kind: 'num',
      },
      {
        key: 'enable_bucket',
        label: '分桶（保持纵横比）',
        hint: '按原图纵横比分桶训练，避免强制裁剪。',
        kind: 'bool',
      },
      {
        key: 'keep_tokens',
        label: '保留开头标签数',
        hint: '打乱标签时保留前 N 个（通常是触发词/角色名）。',
        kind: 'num',
      },
      {
        key: 'shuffle_caption',
        label: '随机打乱标签',
        hint: '每步随机打乱逗号标签顺序（缓存 TE 输出时自动关闭）。',
        kind: 'bool',
      },
      {
        key: 'caption_dropout_rate',
        label: '标签丢弃率',
        hint: '随机丢弃描述的比例（0.05 常规；0 = 从不丢弃）。',
        kind: 'num',
        step: 'any',
      },
    ],
  },
  {
    title: '训练设置',
    fields: [
      {
        key: 'max_train_epochs',
        label: '训练轮数',
        hint: '每轮 = 全部图片过一遍；小数据集可多轮，首次建议 1 轮试跑。',
        kind: 'num',
      },
      {
        key: 'gradient_accumulation_steps',
        label: '梯度累积',
        hint: '等效放大批大小（2 = 每 2 步更新一次参数）。',
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
        kind: 'num',
        step: 'any',
      },
      { key: 'seed', label: '随机种子', hint: '固定随机性，便于复现。', kind: 'num' },
    ],
  },
  {
    title: '缓存与精度',
    fields: [
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
      { key: 'mixed_precision', label: '混合精度', hint: 'bf16 常规；老显卡用 fp16。', kind: 'select', options: PRECISIONS },
    ],
  },
  {
    title: '质量技巧（可选）',
    fields: [
      {
        key: 'min_snr_gamma',
        label: 'Min-SNR Gamma',
        hint: '5.0 常规；抑制噪声步权重失衡，留空关闭。',
        kind: 'num',
        step: 'any',
      },
      {
        key: 'noise_offset',
        label: '噪声偏移',
        hint: '0.02~0.05 提升明暗对比表现；留空关闭。',
        kind: 'num',
        step: 'any',
      },
    ],
  },
  {
    title: '保存与采样',
    fields: [
      {
        key: 'save_every_n_epochs',
        label: '保存间隔（轮）',
        hint: '每 N 轮存一个 LoRA 检查点（断点续训也依赖它）。',
        kind: 'num',
      },
      {
        key: 'sample_every_n_epochs',
        label: '采样间隔（轮）',
        hint: '每 N 轮生成预览图评估效果；0 = 不采样（更快）。',
        kind: 'num',
      },
      {
        key: 'sample_prompts',
        label: '采样提示词',
        hint: '每行一个提示词，训练中出图评估。',
        kind: 'textlist',
      },
      {
        key: 'sample_sampler',
        label: '采样器',
        hint: 'euler_a 常用；采样出图用的扩散采样器。',
        kind: 'text',
      },
    ],
  },
]

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
    cache_latents: true,
    cache_text_encoder_outputs: true,
    mixed_precision: 'bf16',
    gradient_checkpointing: true,
    gradient_accumulation_steps: 1,
    max_grad_norm: 1.0,
    seed: 42,
    min_snr_gamma: 5,
    shuffle_caption: true,
    keep_tokens: 1,
    caption_dropout_rate: 0.05,
    save_every_n_epochs: 1,
    sample_every_n_epochs: 0,
    sample_prompts: [] as string[],
    sample_sampler: 'euler_a',
  }
}

/** 单个参数控件。 */
function FieldInput(props: { field: FieldDef; value: unknown; onChange: (v: unknown) => void }) {
  const { field, value, onChange } = props
  const title = `${field.label}：${field.hint}`
  switch (field.kind) {
    case 'bool':
      return (
        <label className="field chk-field" title={title}>
          <span>{field.label}</span>
          <input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />
        </label>
      )
    case 'select':
      return (
        <label className="field" title={title}>
          <span>{field.label}</span>
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
        <label className="field wide" title={title}>
          <span>{field.label}</span>
          <textarea
            rows={2}
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
        <label className="field" title={title}>
          <span>{field.label}</span>
          <input
            type="number"
            step={field.step ?? '1'}
            value={value === null || value === undefined || value === '' ? '' : Number(value)}
            onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))}
          />
        </label>
      )
    default:
      return (
        <label className="field" title={title}>
          <span>{field.label}</span>
          <input type="text" value={String(value ?? '')} onChange={(e) => onChange(e.target.value)} />
        </label>
      )
  }
}

/** 丹方页（主页默认）：底模/数据集选择 + 完整参数表单 + 保存/点火。 */
export function RecipeForm(props: { onCreated: (runId: string) => void }) {
  const [recipes, setRecipes] = useState<RecipeView[]>([])
  const [models, setModels] = useState<BaseModel[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [recipeId, setRecipeId] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [family, setFamily] = useState('sdxl1')
  const [modelPath, setModelPath] = useState('')
  const [datasetDir, setDatasetDir] = useState('')
  const [data, setData] = useState<Record<string, unknown>>(defaultData())
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)

  const refresh = async () => {
    const [r, m, d] = await Promise.all([listRecipes(), listModels(), listDatasets()])
    setRecipes(r)
    setModels(m)
    setDatasets(d)
  }

  useEffect(() => {
    void refresh().catch(() => {})
  }, [])

  const setField = (k: string, v: unknown) => setData((prev) => ({ ...prev, [k]: v }))

  const basename = (p: string) => p.split(/[\\/]/).pop() ?? ''

  // ---------- 底模 / 数据集选择 ----------

  const onPickModel = async () => {
    setBusy(true)
    setMsg(null)
    try {
      const path = await pickFile()
      if (path) {
        setModelPath(path)
        setMsg(`已选底模：${path}`)
      }
    } catch (e) {
      setMsg(`选择失败：${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setBusy(false)
    }
  }

  const onPickDataset = async () => {
    setBusy(true)
    setMsg(null)
    try {
      const dir = await pickDir()
      if (dir) {
        setDatasetDir(dir)
        setMsg(`已选数据集：${dir}`)
      }
    } catch (e) {
      setMsg(`选择失败：${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setBusy(false)
    }
  }

  // ---------- 保存 / 点火 ----------

  const buildPayload = () => {
    const payload = { ...data }
    const pt = payload.prediction_type
    if (pt === '' || pt === undefined || pt === null) delete payload.prediction_type
    for (const k of Object.keys(payload)) {
      const v = payload[k]
      if (v === null || v === '' || (Array.isArray(v) && v.length === 0)) delete payload[k]
    }
    // 底模/数据集随丹方保存（自定义键，后端 RecipeData 忽略未知字段）
    if (modelPath) payload.model_path = modelPath
    if (datasetDir) payload.dataset_dir = datasetDir
    return payload
  }

  const onSave = async (): Promise<string | null> => {
    if (!name.trim()) {
      setMsg('请填写丹方名称')
      return null
    }
    setBusy(true)
    setMsg(null)
    try {
      // 更新 = 删旧建新（后端无 PUT）
      if (recipeId) {
        try {
          await deleteRecipe(recipeId)
        } catch {
          /* 忽略 */
        }
      }
      const res = await createRecipe(name.trim(), family, buildPayload())
      setRecipeId(res.recipe.id)
      setMsg(`丹方「${name.trim()}」已保存`)
      await refresh()
      return res.recipe.id
    } catch (e) {
      setMsg(`保存失败：${e instanceof Error ? e.message : String(e)}`)
      return null
    } finally {
      setBusy(false)
    }
  }

  const onFire = async () => {
    if (!modelPath || !datasetDir) {
      setMsg('请先选择底模与数据集')
      return
    }
    setBusy(true)
    setMsg(null)
    try {
      // 1. 丹方：未保存则先保存（含底模/数据集路径）
      let rid = recipeId
      if (!rid) {
        if (!name.trim()) setName(basename(modelPath).replace(/\.safetensors$/i, '') + '-丹方')
        rid = await onSave()
        if (!rid) return
      }
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
      setMsg('已入队，自动开始炼丹')
      props.onCreated(run.id)
    } catch (e) {
      setMsg(`点火失败：${e instanceof Error ? e.message : String(e)}`)
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
    setData({ ...defaultData(), ...d })
    if (typeof d.model_path === 'string') setModelPath(d.model_path)
    if (typeof d.dataset_dir === 'string') setDatasetDir(d.dataset_dir)
    setMsg(`已载入丹方「${r.name}」`)
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
      setMsg(`删除失败：${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const datasetInfo = datasets.find((d) => d.dir === datasetDir)

  return (
    <div className="recipe-page">
      {/* 已保存丹方 */}
      <div className="panel recipe-load">
        <div className="panel-title">
          <h2>已保存丹方</h2>
        </div>
        {recipes.length === 0 ? (
          <p className="hint">暂无保存的丹方；在下方面板配置并点「保存丹方」。</p>
        ) : (
          <ul className="runs datasets">
            {recipes.map((r) => (
              <li
                key={r.id}
                className={`run ${r.id === recipeId ? 'active' : ''}`}
                onClick={() => onLoadRecipe(r)}
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

        {FORM_SECTIONS.map((sec) => (
          <div key={sec.title} className="form-sec">
            <h4>{sec.title}</h4>
            <div className="form-grid">
              {sec.fields
                .filter((f) => !f.sdxlOnly || family === 'sdxl1')
                .map((f) => (
                  <FieldInput key={f.key} field={f} value={data[f.key]} onChange={(v) => setField(f.key, v)} />
                ))}
            </div>
          </div>
        ))}

        <div className="actions">
          <button onClick={() => void onSave()} disabled={busy} className="secondary">
            {busy ? '保存中…' : '保存丹方'}
          </button>
          <button onClick={() => void onFire()} disabled={busy} className="primary">
            {busy ? '点火中…' : '点火炼丹'}
          </button>
          {msg && <span className="status-line">{msg}</span>}
        </div>
      </div>
    </div>
  )
}
