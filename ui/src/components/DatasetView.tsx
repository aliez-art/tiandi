import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  assetUrl,
  batchReplaceCaptions,
  createDataset,
  listCaptions,
  listDatasets,
  runTagging,
  saveCaption,
  scanDataset,
  tagStats,
  type CaptionEntry,
  type Dataset,
  type TagStat,
} from '../api'

/** 药材视图：数据集管理 + 标签编辑器（PRD §5.2/5.3，FR-201~303）。 */
export default function DatasetView() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [captions, setCaptions] = useState<CaptionEntry[]>([])
  const [tags, setTags] = useState<TagStat[]>([])
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [editText, setEditText] = useState('')
  const [saved, setSaved] = useState(false)
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState<string | null>(null)
  // 新建数据集表单
  const [newName, setNewName] = useState('')
  const [newDir, setNewDir] = useState('')
  // 批量替换表单
  const [findText, setFindText] = useState('')
  const [replaceText, setReplaceText] = useState('')
  const [useRegex, setUseRegex] = useState(false)

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
  }, [refreshDatasets])

  const loadDataset = useCallback(async (id: string) => {
    try {
      const [caps, stats] = await Promise.all([listCaptions(id), tagStats(id)])
      setCaptions(caps)
      setTags(stats)
      setSelectedImage(caps[0]?.path ?? null)
    } catch (e) {
      console.error('load dataset failed', e)
    }
  }, [])

  useEffect(() => {
    if (selected) void loadDataset(selected)
  }, [selected, loadDataset])

  const currentCaption = useMemo(
    () => captions.find((c) => c.path === selectedImage) ?? null,
    [captions, selectedImage],
  )

  const onScan = async () => {
    if (!selected) return
    setBusy(true)
    setStatus('扫描中…')
    try {
      const res = await scanDataset(selected)
      setStatus(
        `扫描完成：${res.report.total} 张有效，${res.report.invalid} 张损坏，` +
        `${res.report.duplicate_groups.length} 组重复，${res.report.buckets.length} 个桶，` +
        `${res.report.elapsed_ms}ms`,
      )
      await refreshDatasets()
      await loadDataset(selected)
    } catch (e) {
      setStatus(`扫描失败：${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setBusy(false)
    }
  }

  const onTag = async (mode: string) => {
    if (!selected) return
    setBusy(true)
    setStatus(mode === 'wd14' ? 'WD14 打标中（需内核环境）…' : '打标中…')
    try {
      const res = await runTagging(selected, mode)
      setStatus(`打标完成：${res.tagged} 张（${res.mode}）`)
      await loadDataset(selected)
    } catch (e) {
      setStatus(`打标失败：${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setBusy(false)
    }
  }

  const onSaveCaption = async () => {
    if (!selected || !selectedImage) return
    try {
      await saveCaption(selected, selectedImage, editText)
      setSaved(true)
      window.setTimeout(() => setSaved(false), 1500)
      setCaptions((prev) =>
        prev.map((c) => (c.path === selectedImage ? { ...c, caption: editText, has_file: true } : c)),
      )
      const stats = await tagStats(selected)
      setTags(stats)
    } catch (e) {
      alert(`保存失败：${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const onBatchReplace = async () => {
    if (!selected || !findText) return
    if (!window.confirm(`将对全部 ${captions.length} 张图的 caption 应用替换规则，确定？`)) return
    setBusy(true)
    try {
      const res = await batchReplaceCaptions(selected, [
        { find: findText, replace: replaceText, regex: useRegex },
      ])
      setStatus(`批量替换完成：${res.affected} / ${res.total} 张受影响`)
      await loadDataset(selected)
    } catch (e) {
      setStatus(`批量替换失败：${e instanceof Error ? e.message : String(e)}`)
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

  const onPickImage = (path: string) => {
    setSelectedImage(path)
    const cap = captions.find((c) => c.path === path)
    setEditText(cap?.caption ?? '')
    setSaved(false)
  }

  return (
    <div className="dataset-view">
      {/* 数据集列表 + 操作 */}
      <section className="panel">
        <div className="panel-title">
          <h2>药材库（数据集）</h2>
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
              <span className="run-time">{d.dir}</span>
            </li>
          ))}
        </ul>
        {selected && (
          <div className="actions">
            <button onClick={onScan} disabled={busy} className="secondary">
              扫描（导入/去重/分桶）
            </button>
            <button onClick={() => onTag('mock')} disabled={busy} className="secondary">
              打标（mock）
            </button>
            <button onClick={() => onTag('wd14')} disabled={busy} className="secondary">
              打标（WD14）
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
      </section>

      {/* 图片网格 */}
      <section className="panel">
        <h2>拣药（图片与标签）</h2>
        {captions.length === 0 ? (
          <p className="hint">先注册并扫描数据集。</p>
        ) : (
          <ul className="img-grid">
            {captions.map((c) => (
              <li
                key={c.path}
                className={`img-cell ${c.path === selectedImage ? 'active' : ''} ${!c.has_file ? 'untagged' : ''}`}
                onClick={() => onPickImage(c.path)}
                title={c.path}
              >
                <img src={assetUrl(c.path)} alt={c.path} loading="lazy" />
                {!c.has_file && <span className="badge">未打标</span>}
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* 标签云 + 编辑器 */}
      <div className="editor-row">
        <section className="panel">
          <h2>标签云（{tags.length} 个标签）</h2>
          {tags.length === 0 ? (
            <p className="hint">打标后这里会显示标签频次。</p>
          ) : (
            <div className="tagcloud">
              {tags.slice(0, 120).map((t) => (
                <span
                  key={t.tag}
                  className="tag"
                  style={{ fontSize: `${Math.max(11, 10 + Math.log2(t.count + 1) * 3)}px` }}
                  title={`${t.count} 次`}
                  onClick={() => {
                    setFindText(t.tag)
                    setEditText((prev) => (prev ? `${prev}, ${t.tag}` : t.tag))
                  }}
                >
                  {t.tag} <em>{t.count}</em>
                </span>
              ))}
            </div>
          )}
          <div className="batch-row">
            <input
              placeholder="查找（支持正则）"
              value={findText}
              onChange={(e) => setFindText(e.target.value)}
            />
            <input placeholder="替换为" value={replaceText} onChange={(e) => setReplaceText(e.target.value)} />
            <label className="chk">
              <input type="checkbox" checked={useRegex} onChange={(e) => setUseRegex(e.target.checked)} />
              正则
            </label>
            <button onClick={onBatchReplace} disabled={busy} className="secondary">
              批量替换
            </button>
          </div>
        </section>

        <section className="panel">
          <h2>标签编辑</h2>
          {currentCaption ? (
            <>
              <div className="mono dim">{currentCaption.path}</div>
              <textarea
                value={selectedImage === currentCaption.path ? editText : currentCaption.caption}
                onChange={(e) => setEditText(e.target.value)}
                rows={6}
                placeholder="逗号分隔的标签，或自然语言描述"
              />
              <div className="actions">
                <button onClick={onSaveCaption} disabled={busy}>
                  {saved ? '✓ 已保存' : '保存标签'}
                </button>
              </div>
            </>
          ) : (
            <p className="hint">从网格选择一张图开始编辑。</p>
          )}
        </section>
      </div>
    </div>
  )
}
