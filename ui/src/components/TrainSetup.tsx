import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  createDataset,
  createRecipe,
  createRun,
  deleteDataset,
  deleteRecipe,
  listDatasets,
  listModels,
  listRecipes,
  pickDir,
  pickFile,
  registerModel,
  scanDataset,
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

/** 遮罩 + 居中弹窗骨架（portal 到 body；遮罩点击 / ✕ / Esc 均可关闭）。 */
function Dialog(props: { title: string; onClose: () => void; children: React.ReactNode; wide?: boolean }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') props.onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [props])

  return createPortal(
    <>
      {/* 遮罩与弹窗同级（遮罩 z-index 低于弹窗容器，点击遮罩关闭） */}
      <div className="overlay" onClick={props.onClose} />
      <div className="dialog-wrap">
        <div className={`dialog ${props.wide ? 'wide' : ''}`} onClick={(e) => e.stopPropagation()}>
          <div className="dialog-title">
            <h2>{props.title}</h2>
            <button className="close" onClick={props.onClose} title="关闭（Esc）">
              ✕
            </button>
          </div>
          <div className="dialog-body">{props.children}</div>
        </div>
      </div>
    </>,
    document.body,
  )
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

/** 丹方管理：我的丹方列表 + 完整参数新建表单。 */
export function RecipeManager(props: { onClose: () => void; onPick?: (r: RecipeView) => void }) {
  const [recipes, setRecipes] = useState<RecipeView[]>([])
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  // 新建表单
  const [name, setName] = useState('')
  const [family, setFamily] = useState('sdxl1')
  const [data, setData] = useState<Record<string, unknown>>(defaultData())

  const refresh = async () => {
    const r = await listRecipes()
    setRecipes(r)
  }

  useEffect(() => {
    void refresh().catch(() => {})
  }, [])

  const setField = (k: string, v: unknown) => setData((prev) => ({ ...prev, [k]: v }))

  const onCreate = async () => {
    if (!name.trim()) {
      setMsg('请填写丹方名称')
      return
    }
    setBusy(true)
    setMsg(null)
    try {
      const payload = { ...data }
      // 空值清理：null/空串/空数组不提交（用后端默认）
      const pt = payload.prediction_type
      if (pt === '' || pt === undefined || pt === null) delete payload.prediction_type
      for (const k of Object.keys(payload)) {
        const v = payload[k]
        if (v === null || v === '' || (Array.isArray(v) && v.length === 0)) delete payload[k]
      }
      await createRecipe(name.trim(), family, payload)
      setName('')
      setMsg(`丹方「${name.trim()}」已保存`)
      await refresh()
    } catch (e) {
      setMsg(`创建失败：${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setBusy(false)
    }
  }

  const onDelete = async (r: RecipeView) => {
    if (!window.confirm(`删除丹方「${r.name}」？`)) return
    try {
      await deleteRecipe(r.id)
      setRecipes((prev) => prev.filter((x) => x.id !== r.id))
    } catch (e) {
      setMsg(`删除失败：${e instanceof Error ? e.message : String(e)}`)
    }
  }

  return (
    <Dialog title="丹方管理" onClose={props.onClose} wide>
      {/* 我的丹方 */}
      <section className="recipe-sec">
        <h3>我的丹方</h3>
        {recipes.length === 0 ? (
          <p className="hint">还没有丹方，在下方填写参数并保存。</p>
        ) : (
          <ul className="runs datasets">
            {recipes.map((r) => (
              <li key={r.id} className="run">
                <span className="run-id">{r.name}</span>
                <span className="run-state">{FAMILY_LABEL[r.family] ?? r.family}</span>
                <span className="run-time">
                  dim {String(r.data.network_dim ?? '?')} · lr {String(r.data.learning_rate ?? '?')} ·{' '}
                  {String(r.data.max_train_epochs ?? '?')} epochs
                </span>
                <span className="gallery-actions">
                  {props.onPick && (
                    <button className="secondary" onClick={() => props.onPick?.(r)}>
                      选用
                    </button>
                  )}
                  <button className="danger" onClick={() => void onDelete(r)}>
                    删除
                  </button>
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* 新建丹方（完整参数表单） */}
      <section className="recipe-sec">
        <h3>新建丹方</h3>
        <div className="form-grid">
          <label className="field">
            <span>丹方名称</span>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="如 NoobAI 角色丹" />
          </label>
          <label className="field">
            <span>模型族</span>
            <select value={family} onChange={(e) => setFamily(e.target.value)}>
              <option value="sdxl1">SDXL 1.0（NoobAI/Illusion）</option>
              <option value="dit_anima">Anima (DiT)</option>
              <option value="dit_krea2">Krea 2 (DiT)</option>
            </select>
          </label>
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
          <button onClick={() => void onCreate()} disabled={busy} className="primary">
            {busy ? '保存中…' : '保存丹方'}
          </button>
          {msg && <span className="status-line">{msg}</span>}
        </div>
      </section>
    </Dialog>
  )
}

/** 数据集管理：列表 / 注册（目录选择）/ 扫描 / 删除。 */
export function DatasetManager(props: { onClose: () => void; onPick?: (d: Dataset) => void }) {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)

  const refresh = async () => {
    const list = await listDatasets()
    setDatasets(list)
    setSelected((prev) => (prev && list.some((d) => d.id === prev) ? prev : list[0]?.id ?? null))
  }

  useEffect(() => {
    void refresh().catch(() => {})
  }, [])

  const onPickDir = async () => {
    setBusy(true)
    setMsg(null)
    try {
      const path = await pickDir()
      if (!path) return
      const name = path.split(/[\\/]/).pop() ?? '数据集'
      const ds = await createDataset(name, path)
      await refresh()
      setSelected(ds.id)
      setMsg(`已注册数据集「${name}」，点「扫描」导入图像`)
    } catch (e) {
      setMsg(`注册失败：${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setBusy(false)
    }
  }

  const onScan = async () => {
    if (!selected) return
    setBusy(true)
    setMsg('扫描中…')
    try {
      const res = await scanDataset(selected)
      setMsg(
        `扫描完成：${res.report.total} 张有效 · ${res.report.duplicate_groups.length} 组重复 · ${res.report.buckets.length} 个桶`,
      )
      await refresh()
    } catch (e) {
      setMsg(`扫描失败：${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setBusy(false)
    }
  }

  const onDelete = async (d: Dataset) => {
    if (!window.confirm(`删除数据集「${d.name}」？仅移除记录，磁盘图片不受影响。`)) return
    try {
      await deleteDataset(d.id)
      await refresh()
    } catch (e) {
      setMsg(`删除失败：${e instanceof Error ? e.message : String(e)}`)
    }
  }

  return (
    <Dialog title="数据集" onClose={props.onClose}>
      <div className="actions">
        <button onClick={() => void onPickDir()} disabled={busy} className="secondary">
          注册数据集（选择文件夹）…
        </button>
        {selected && (
          <button onClick={() => void onScan()} disabled={busy} className="secondary">
            扫描（导入/去重/分桶）
          </button>
        )}
      </div>
      <ul className="runs datasets">
        {datasets.length === 0 ? (
          <p className="hint">还没有数据集，点上方按钮选择图片文件夹注册。</p>
        ) : (
          datasets.map((d) => (
            <li
              key={d.id}
              className={`run ${d.id === selected ? 'active' : ''}`}
              onClick={() => setSelected(d.id)}
            >
              <span className="run-id">{d.name}</span>
              <span className="run-state">{d.image_count} 张</span>
              <span className="run-time" title={d.dir}>
                {d.dir}
              </span>
              <span className="gallery-actions">
                {props.onPick && (
                  <button
                    className="secondary"
                    onClick={(e) => {
                      e.stopPropagation()
                      props.onPick?.(d)
                    }}
                  >
                    选用
                  </button>
                )}
                <button
                  className="danger"
                  onClick={(e) => {
                    e.stopPropagation()
                    void onDelete(d)
                  }}
                >
                  删除
                </button>
              </span>
            </li>
          ))
        )}
      </ul>
      {msg && <p className="status-line">{msg}</p>}
      <p className="hint">提示：图片文件夹内每张图配一个同名 .txt 写描述（01.png + 01.txt），训练读取此描述。</p>
    </Dialog>
  )
}

/** 扁平操作条：选底模（文件对话框）→ 选数据集 → 选丹方 → 点火。 */
export function NewRunBar(props: { onCreated: (runId: string) => void; onOpenRecipes: () => void }) {
  const [models, setModels] = useState<BaseModel[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [recipes, setRecipes] = useState<RecipeView[]>([])
  const [family, setFamily] = useState('sdxl1')
  const [modelId, setModelId] = useState('')
  const [datasetId, setDatasetId] = useState('')
  const [recipeId, setRecipeId] = useState('')
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  const [showDatasets, setShowDatasets] = useState(false)

  const refresh = async () => {
    const [m, d, r] = await Promise.all([listModels(), listDatasets(), listRecipes()])
    setModels(m)
    setDatasets(d)
    setRecipes(r)
    setModelId((prev) => prev || m[0]?.id || '')
    setDatasetId((prev) => prev || d[0]?.id || '')
    setRecipeId((prev) => prev || r[0]?.id || '')
  }

  useEffect(() => {
    void refresh().catch(() => {})
  }, [])

  const onPickModel = async () => {
    setBusy(true)
    setMsg(null)
    try {
      const path = await pickFile()
      if (!path) return // 用户取消
      const base = path.split(/[\\/]/).pop()?.replace(/\.safetensors$/i, '') ?? '模型'
      const m = await registerModel({ name: base, family, path })
      setModels((prev) => [m, ...prev])
      setModelId(m.id)
      setMsg(`已选底模：${base}`)
    } catch (e) {
      setMsg(`选择失败：${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setBusy(false)
    }
  }

  const onFire = async () => {
    if (!modelId || !datasetId || !recipeId) {
      setMsg('请先选齐底模、数据集与丹方')
      return
    }
    setBusy(true)
    setMsg(null)
    try {
      const run = await createRun({
        dataset_id: datasetId,
        recipe_id: recipeId,
        base_model_id: modelId,
      })
      await startRun(run.id)
      setMsg('已入队，自动开始炼丹')
      props.onCreated(run.id)
    } catch (e) {
      setMsg(`点火失败：${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setBusy(false)
    }
  }

  const selectedModel = models.find((m) => m.id === modelId) ?? null

  return (
    <div className="firebar">
      <div className="firebar-row">
        <button onClick={() => void onPickModel()} disabled={busy} className="secondary">
          选择底模…
        </button>
        <select value={family} onChange={(e) => setFamily(e.target.value)} title="底模所属模型族">
          <option value="sdxl1">SDXL</option>
          <option value="dit_anima">Anima</option>
          <option value="dit_krea2">Krea 2</option>
        </select>
      </div>
      <span className={`firebar-model ${selectedModel ? '' : 'dim'}`} title={selectedModel?.path ?? ''}>
        {selectedModel ? selectedModel.name : '未选择底模'}
      </span>
      <div className="firebar-row">
        <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)} title="训练数据集">
          {datasets.length === 0 ? (
            <option value="">（暂无数据集）</option>
          ) : (
            datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}（{d.image_count} 张）
              </option>
            ))
          )}
        </select>
        <button onClick={() => setShowDatasets(true)} className="secondary" title="数据集管理（注册/扫描/删除）">
          数据集
        </button>
      </div>
      <div className="firebar-row">
        <select value={recipeId} onChange={(e) => setRecipeId(e.target.value)} title="丹方">
          {recipes.length === 0 ? (
            <option value="">（暂无丹方，点「丹方」创建）</option>
          ) : (
            recipes.map((r) => (
              <option key={r.id} value={r.id}>
                {r.name}
              </option>
            ))
          )}
        </select>
        <button onClick={props.onOpenRecipes} className="secondary" title="丹方管理（新建/编辑参数）">
          丹方
        </button>
      </div>
      <button onClick={() => void onFire()} disabled={busy} className="primary firebar-fire">
        {busy ? '处理中…' : '点火炼丹'}
      </button>
      {msg && <div className="firebar-msg">{msg}</div>}
      {showDatasets && (
        <DatasetManager
          onClose={() => setShowDatasets(false)}
          onPick={(d) => {
            setDatasetId(d.id)
            setShowDatasets(false)
          }}
        />
      )}
    </div>
  )
}
