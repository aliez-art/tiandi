import { useEffect, useState } from 'react'
import {
  createRecipe,
  createRun,
  deleteRecipe,
  listDatasets,
  listModels,
  listPresets,
  listRecipes,
  startRun,
  type BaseModel,
  type Dataset,
  type PresetView,
  type RecipeView,
} from '../api'

const FAMILY_LABEL: Record<string, string> = {
  sdxl1: 'SDXL 1.0（NoobAI/Illusion）',
  dit_anima: 'Anima (DiT)',
  dit_krea2: 'Krea 2 (DiT)',
}

const OPTIMIZERS = [
  ['adamw8bit', 'AdamW 8bit'],
  ['adamw', 'AdamW'],
  ['adafactor', 'Adafactor'],
  ['prodigy', 'Prodigy'],
  ['lion', 'Lion'],
]
const SCHEDULERS = [
  ['cosine', 'Cosine'],
  ['constant', 'Constant'],
  ['linear', 'Linear'],
  ['constant_with_warmup', 'Constant + Warmup'],
  ['cosine_with_restarts', 'Cosine Restarts'],
]
const NETWORKS = [
  ['lora', 'LoRA'],
  ['locon', 'LoCon'],
  ['lokr', 'LoKr'],
  ['loha', 'LoHa'],
  ['dora', 'DoRA'],
  ['tlora', 'T-LoRA'],
]
const PRECISIONS = [
  ['bf16', 'bf16'],
  ['fp16', 'fp16'],
  ['fp32', 'fp32'],
]
const PREDICTIONS = [
  ['', '自动'],
  ['epsilon', 'epsilon（常规）'],
  ['v', 'v（V 预测模型）'],
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

function overlay(): JSX.Element {
  return <div className="overlay" />
}

/** 遮罩 + 居中弹窗骨架。 */
function Dialog(props: { title: string; onClose: () => void; children: React.ReactNode; wide?: boolean }) {
  return (
    <div className="dialog-wrap">
      {overlay()}
      <div className={`dialog ${props.wide ? 'wide' : ''}`}>
        <div className="dialog-title">
          <h2>{props.title}</h2>
          <button className="close" onClick={props.onClose} title="关闭">
            ✕
          </button>
        </div>
        <div className="dialog-body">{props.children}</div>
      </div>
    </div>
  )
}

/** 丹方管理：内置预设 + 我的丹方列表 + 新建丹方表单。 */
export function RecipeManager(props: { onClose: () => void; onPick?: (r: RecipeView) => void }) {
  const [presets, setPresets] = useState<PresetView[]>([])
  const [recipes, setRecipes] = useState<RecipeView[]>([])
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  // 新建表单
  const [name, setName] = useState('')
  const [family, setFamily] = useState('sdxl1')
  const [data, setData] = useState<Record<string, unknown>>(defaultData())
  const [advanced, setAdvanced] = useState(false)

  const refresh = async () => {
    const [p, r] = await Promise.all([listPresets(), listRecipes()])
    setPresets(p)
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
      // 预测类型空串 → 省略（自动）
      const pt = payload.prediction_type
      if (pt === '' || pt === undefined || pt === null) delete payload.prediction_type
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

  const applyPreset = (p: PresetView) => {
    setFamily(p.family)
    setData({ ...defaultData(), ...p.data })
    setMsg(`已载入预设「${p.name}」，可调整后保存`)
  }

  return (
    <Dialog title="丹方管理" onClose={props.onClose} wide>
      {/* 内置预设 */}
      <section className="recipe-sec">
        <h3>内置预设（一键套用）</h3>
        <div className="preset-grid">
          {presets.map((p) => (
            <div key={p.name} className="preset-card">
              <div className="preset-name">
                {p.name}
                <span className="family-pill">{FAMILY_LABEL[p.family] ?? p.family}</span>
              </div>
              <p className="hint">{p.description}</p>
              <button className="secondary" onClick={() => applyPreset(p)}>
                套用
              </button>
            </div>
          ))}
        </div>
      </section>

      {/* 我的丹方 */}
      <section className="recipe-sec">
        <h3>我的丹方</h3>
        {recipes.length === 0 ? (
          <p className="hint">还没有自定义丹方，从预设套用或下方新建。</p>
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

      {/* 新建丹方 */}
      <section className="recipe-sec">
        <h3>新建丹方</h3>
        <div className="form-grid">
          <label className="field">
            <span>名称</span>
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
          <label className="field">
            <span>学习率</span>
            <input
              type="number"
              step="any"
              value={Number(data.learning_rate)}
              onChange={(e) => setField('learning_rate', Number(e.target.value))}
            />
          </label>
          <label className="field">
            <span>优化器</span>
            <select value={String(data.optimizer)} onChange={(e) => setField('optimizer', e.target.value)}>
              {OPTIMIZERS.map(([v, l]) => (
                <option key={v} value={v}>
                  {l}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>调度器</span>
            <select value={String(data.lr_scheduler)} onChange={(e) => setField('lr_scheduler', e.target.value)}>
              {SCHEDULERS.map(([v, l]) => (
                <option key={v} value={v}>
                  {l}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>网络结构</span>
            <select value={String(data.network_type)} onChange={(e) => setField('network_type', e.target.value)}>
              {NETWORKS.map(([v, l]) => (
                <option key={v} value={v}>
                  {l}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>维度 dim</span>
            <input
              type="number"
              value={Number(data.network_dim)}
              onChange={(e) => setField('network_dim', Number(e.target.value))}
            />
          </label>
          <label className="field">
            <span>Alpha</span>
            <input
              type="number"
              value={Number(data.network_alpha)}
              onChange={(e) => setField('network_alpha', Number(e.target.value))}
            />
          </label>
          <label className="field">
            <span>训练轮数</span>
            <input
              type="number"
              value={Number(data.max_train_epochs)}
              onChange={(e) => setField('max_train_epochs', Number(e.target.value))}
            />
          </label>
          <label className="field">
            <span>分辨率</span>
            <input
              type="number"
              value={Number(data.resolution)}
              onChange={(e) => setField('resolution', Number(e.target.value))}
            />
          </label>
          <label className="field">
            <span>精度</span>
            <select value={String(data.mixed_precision)} onChange={(e) => setField('mixed_precision', e.target.value)}>
              {PRECISIONS.map(([v, l]) => (
                <option key={v} value={v}>
                  {l}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>预测类型</span>
            <select value={String(data.prediction_type ?? '')} onChange={(e) => setField('prediction_type', e.target.value)}>
              {PREDICTIONS.map(([v, l]) => (
                <option key={v} value={v}>
                  {l}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>触发词</span>
            <input value={String(data.trigger_word ?? '')} onChange={(e) => setField('trigger_word', e.target.value)} />
          </label>
          <label className="field chk-field">
            <span>缓存文本编码器输出（SDXL 提速）</span>
            <input
              type="checkbox"
              checked={Boolean(data.cache_text_encoder_outputs)}
              onChange={(e) => setField('cache_text_encoder_outputs', e.target.checked)}
            />
          </label>
          <label className="field chk-field">
            <span>梯度检查点（省显存）</span>
            <input
              type="checkbox"
              checked={Boolean(data.gradient_checkpointing)}
              onChange={(e) => setField('gradient_checkpointing', e.target.checked)}
            />
          </label>
          <label className="field chk-field">
            <span>随机打乱标签顺序</span>
            <input
              type="checkbox"
              checked={Boolean(data.shuffle_caption)}
              onChange={(e) => setField('shuffle_caption', e.target.checked)}
            />
          </label>
        </div>
        <button className="adv-toggle" onClick={() => setAdvanced((a) => !a)}>
          {advanced ? '收起高级参数 ▲' : '高级参数 ▼'}
        </button>
        {advanced && (
          <div className="form-grid">
            <label className="field">
              <span>预热比例</span>
              <input
                type="number"
                step="any"
                value={Number(data.lr_warmup_ratio)}
                onChange={(e) => setField('lr_warmup_ratio', Number(e.target.value))}
              />
            </label>
            <label className="field">
              <span>Min-SNR Gamma</span>
              <input
                type="number"
                step="any"
                value={Number(data.min_snr_gamma ?? 0)}
                onChange={(e) => setField('min_snr_gamma', Number(e.target.value) || null)}
              />
            </label>
            <label className="field">
              <span>保留开头标签数</span>
              <input
                type="number"
                value={Number(data.keep_tokens)}
                onChange={(e) => setField('keep_tokens', Number(e.target.value))}
              />
            </label>
            <label className="field">
              <span>标签丢弃率</span>
              <input
                type="number"
                step="any"
                value={Number(data.caption_dropout_rate)}
                onChange={(e) => setField('caption_dropout_rate', Number(e.target.value))}
              />
            </label>
            <label className="field">
              <span>保存间隔（轮）</span>
              <input
                type="number"
                value={Number(data.save_every_n_epochs)}
                onChange={(e) => setField('save_every_n_epochs', Number(e.target.value))}
              />
            </label>
            <label className="field">
              <span>批大小</span>
              <input
                type="number"
                value={Number(data.batch_size)}
                onChange={(e) => setField('batch_size', Number(e.target.value))}
              />
            </label>
          </div>
        )}
        <div className="actions">
          <button onClick={() => void onCreate()} disabled={busy}>
            {busy ? '保存中…' : '保存丹方'}
          </button>
          {msg && <span className="status-line">{msg}</span>}
        </div>
      </section>
    </Dialog>
  )
}

/** 新建炼丹：选模型 → 选数据集 → 选丹方 → 点火。 */
export function NewRunDialog(props: { onClose: () => void; onCreated: (runId: string) => void }) {
  const [models, setModels] = useState<BaseModel[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [recipes, setRecipes] = useState<RecipeView[]>([])
  const [modelId, setModelId] = useState('')
  const [datasetId, setDatasetId] = useState('')
  const [recipeId, setRecipeId] = useState('')
  const [showRecipes, setShowRecipes] = useState(false)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)

  useEffect(() => {
    void Promise.all([listModels(), listDatasets(), listRecipes()])
      .then(([m, d, r]) => {
        setModels(m)
        setDatasets(d)
        setRecipes(r)
        setModelId((prev) => prev || m[0]?.id || '')
        setDatasetId((prev) => prev || d[0]?.id || '')
        setRecipeId((prev) => prev || r[0]?.id || '')
      })
      .catch(() => {})
  }, [])

  const onFire = async () => {
    if (!modelId || !datasetId || !recipeId) {
      setMsg('请先选齐模型、数据集与丹方')
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
      setMsg('已入队，scheduler 会自动点火训练')
      props.onCreated(run.id)
    } catch (e) {
      setMsg(`点火失败：${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setBusy(false)
    }
  }

  const pickRecipe = (r: RecipeView) => {
    setRecipeId(r.id)
    setShowRecipes(false)
  }

  return (
    <Dialog title="新建炼丹" onClose={props.onClose}>
      {showRecipes ? (
        <RecipeManager onClose={() => setShowRecipes(false)} onPick={pickRecipe} />
      ) : (
        <>
          <div className="form-grid">
            <label className="field wide">
              <span>基底模型</span>
              {models.length === 0 ? (
                <p className="hint">暂无模型 → 请到「药材」页注册</p>
              ) : (
                <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
                  {models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}（{FAMILY_LABEL[m.family] ?? m.family}）
                    </option>
                  ))}
                </select>
              )}
            </label>
            <label className="field wide">
              <span>数据集</span>
              {datasets.length === 0 ? (
                <p className="hint">暂无数据集 → 请到「药材」页注册并扫描</p>
              ) : (
                <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name}（{d.image_count} 张）
                    </option>
                  ))}
                </select>
              )}
            </label>
            <label className="field wide">
              <span>丹方</span>
              {recipes.length === 0 ? (
                <p className="hint">暂无丹方 → 点下方「管理丹方」创建</p>
              ) : (
                <select value={recipeId} onChange={(e) => setRecipeId(e.target.value)}>
                  {recipes.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.name}（{FAMILY_LABEL[r.family] ?? r.family}）
                    </option>
                  ))}
                </select>
              )}
            </label>
          </div>
          <div className="actions">
            <button onClick={() => void onFire()} disabled={busy} className="primary">
              {busy ? '点火中…' : '点火炼丹'}
            </button>
            <button onClick={() => setShowRecipes(true)} className="secondary">
              管理丹方
            </button>
          </div>
          {msg && <p className="status-line">{msg}</p>}
          <p className="hint">提示：任务入队后会自动串行训练；同一时间只跑一个任务。</p>
        </>
      )}
    </Dialog>
  )
}
