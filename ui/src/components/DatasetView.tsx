import { useCallback, useEffect, useState } from 'react'
import {
  createDataset,
  listDatasets,
  listModels,
  registerModel,
  scanDataset,
  type BaseModel,
  type Dataset,
} from '../api'

const FAMILY_LABEL: Record<string, string> = {
  sdxl1: 'SDXL 1.0（NoobAI/Illusion）',
  dit_anima: 'Anima (DiT)',
  dit_krea2: 'Krea 2 (DiT)',
}

/** 药材视图：基底模型注册 + 数据集管理（专注炼丹，不含打标/标签工具）。 */
export default function DatasetView() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [models, setModels] = useState<BaseModel[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState<string | null>(null)
  // 新建数据集表单
  const [newName, setNewName] = useState('')
  const [newDir, setNewDir] = useState('')
  // 模型注册表单
  const [modelName, setModelName] = useState('')
  const [modelFamily, setModelFamily] = useState('sdxl1')
  const [modelPath, setModelPath] = useState('')

  const refreshDatasets = useCallback(async () => {
    try {
      const list = await listDatasets()
      setDatasets(list)
      setSelected((prev) => {
        if (prev && list.some((d) => d.id === prev)) return prev
        return list[0]?.id ?? null
      })
    } catch (e) {
      console.error('list datasets failed', e)
    }
  }, [])

  useEffect(() => {
    void refreshDatasets()
    void listModels().then(setModels).catch(() => {})
  }, [refreshDatasets])

  const onScan = async () => {
    if (!selected) return
    setBusy(true)
    setStatus('扫描中…')
    try {
      const res = await scanDataset(selected)
      setStatus(
        `扫描完成：${res.report.total} 张有效，${res.report.invalid} 张损坏，` +
          `${res.report.duplicate_groups.length} 组重复，${res.report.buckets.length} 个桶（${res.report.elapsed_ms}ms）`,
      )
      await refreshDatasets()
    } catch (e) {
      setStatus(`扫描失败：${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setBusy(false)
    }
  }

  const onCreate = async () => {
    if (!newName || !newDir) return
    setBusy(true)
    try {
      const ds = await createDataset(newName, newDir)
      setNewName('')
      setNewDir('')
      await refreshDatasets()
      setSelected(ds.id)
      setStatus('数据集已创建，点击「扫描」导入图像')
    } catch (e) {
      setStatus(`创建失败：${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setBusy(false)
    }
  }

  const onRegisterModel = async () => {
    if (!modelName || !modelPath) return
    setBusy(true)
    try {
      const m = await registerModel({
        name: modelName,
        family: modelFamily,
        path: modelPath,
      })
      setModelName('')
      setModelPath('')
      setModels((prev) => [m, ...prev])
      setStatus(`基底模型已注册：${m.name}`)
    } catch (e) {
      setStatus(`注册失败：${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="dataset-view">
      {/* 基底模型注册 */}
      <section className="panel">
        <div className="panel-title">
          <h2>基底模型</h2>
        </div>
        {models.length === 0 ? (
          <p className="hint">尚未注册基底模型。训练前请注册（如 NoobAI-XL 的 safetensors 路径）。</p>
        ) : (
          <ul className="runs datasets">
            {models.map((m) => (
              <li key={m.id} className="run">
                <span className="run-id">{m.name}</span>
                <span className="run-state">{FAMILY_LABEL[m.family] ?? m.family}</span>
                <span className="run-time" title={m.path ?? ''}>
                  {m.path}
                </span>
              </li>
            ))}
          </ul>
        )}
        <div className="create-row">
          <input placeholder="名称（如 NoobAI-XL）" value={modelName} onChange={(e) => setModelName(e.target.value)} />
          <select value={modelFamily} onChange={(e) => setModelFamily(e.target.value)}>
            <option value="sdxl1">SDXL 1.0（NoobAI/Illusion）</option>
            <option value="dit_anima">Anima (DiT)</option>
            <option value="dit_krea2">Krea 2 (DiT)</option>
          </select>
          <input placeholder="模型路径（safetensors/目录）" value={modelPath} onChange={(e) => setModelPath(e.target.value)} />
          <button onClick={onRegisterModel} disabled={busy} className="secondary">
            注册
          </button>
        </div>
      </section>

      {/* 数据集列表 + 操作 */}
      <section className="panel">
        <div className="panel-title">
          <h2>训练数据集</h2>
        </div>
        <ul className="runs datasets">
          {datasets.map((d) => (
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
            </li>
          ))}
        </ul>
        {selected && (
          <div className="actions">
            <button onClick={onScan} disabled={busy} className="secondary">
              扫描（导入/去重/分桶）
            </button>
          </div>
        )}
        {status && <p className="status-line">{status}</p>}
        <div className="create-row">
          <input placeholder="数据集名称" value={newName} onChange={(e) => setNewName(e.target.value)} />
          <input placeholder="图片目录（绝对路径）" value={newDir} onChange={(e) => setNewDir(e.target.value)} />
          <button onClick={onCreate} disabled={busy} className="secondary">
            注册数据集
          </button>
        </div>
        <p className="hint">
          提示：图片文件夹内每张图配一个同名 .txt 写描述（如 01.png + 01.txt），训练即读取此描述。
        </p>
      </section>
    </div>
  )
}
