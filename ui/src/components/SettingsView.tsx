import { useEffect, useState } from 'react'
import { fetchSettings, updateSettings } from '../api'

/** 炉房（设置）：镜像源等（FR-901）。 */
export default function SettingsView() {
  const [settings, setSettings] = useState<Record<string, string>>({})
  const [hf, setHf] = useState('')
  const [pip, setPip] = useState('')
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    void fetchSettings()
      .then((s) => {
        setSettings(s)
        setHf(s.hf_endpoint ?? '')
        setPip(s.pip_index ?? '')
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  const onSave = async () => {
    setSaved(false)
    setError(null)
    try {
      const next = await updateSettings({
        hf_endpoint: hf.trim(),
        pip_index: pip.trim(),
      })
      setSettings(next)
      setSaved(true)
      window.setTimeout(() => setSaved(false), 2000)
    } catch (e) {
      setError(`保存失败：${e instanceof Error ? e.message : String(e)}`)
    }
  }

  return (
    <div className="dataset-view">
      <section className="panel">
        <h2>镜像源（网络设置）</h2>
        <p className="hint">
          大陆网络适配（参考 lora-scripts-next）：训练内核下载模型/依赖时使用。
          保存后对后续任务生效（内核进程环境变量注入）。
        </p>
        <div className="create-row settings-row">
          <label>HuggingFace 镜像（HF_ENDPOINT）</label>
          <input
            placeholder="https://hf-mirror.com（留空 = 官方）"
            value={hf}
            onChange={(e) => setHf(e.target.value)}
          />
        </div>
        <div className="create-row settings-row">
          <label>pip 镜像（PIP_INDEX_URL）</label>
          <input
            placeholder="https://pypi.tuna.tsinghua.edu.cn/simple（留空 = 官方）"
            value={pip}
            onChange={(e) => setPip(e.target.value)}
          />
        </div>
        <div className="actions">
          <button onClick={onSave}>{saved ? '✓ 已保存' : '保存设置'}</button>
        </div>
        {error && <p className="status-line">{error}</p>}
      </section>

      <section className="panel">
        <h2>当前设置</h2>
        {Object.keys(settings).length === 0 ? (
          <p className="hint">暂无自定义设置（全部使用默认值）。</p>
        ) : (
          <ul className="runs datasets">
            {Object.entries(settings).map(([k, v]) => (
              <li key={k} className="run">
                <span className="run-id">{k}</span>
                <span className="run-time">{v}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}
