import { useCallback, useEffect, useRef, useState } from 'react'
import {
  API_BASE,
  createSimulatedRun,
  fetchHealth,
  listRuns,
  subscribeEvents,
  type Health,
  type Run,
} from './api'

interface EventLine {
  id: number
  type: string
  data: string
}

const STATE_LABEL: Record<string, string> = {
  created: '已创建',
  queued: '排队中',
  preparing: '准备中',
  running: '炼丹中',
  paused: '文火',
  sampling: '采样中',
  saving: '保存中',
  done: '出炉',
  failed: '炸炉',
  canceled: '已取消',
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [runs, setRuns] = useState<Run[]>([])
  const [events, setEvents] = useState<EventLine[]>([])
  const [busy, setBusy] = useState(false)
  const eventSeq = useRef(0)

  const refreshRuns = useCallback(async () => {
    try {
      setRuns(await listRuns())
    } catch (e) {
      console.error('list runs failed', e)
    }
  }, [])

  useEffect(() => {
    fetchHealth()
      .then((h) => {
        setHealth(h)
        setHealthError(null)
      })
      .catch((e: unknown) => {
        setHealth(null)
        setHealthError(e instanceof Error ? e.message : String(e))
      })
    void refreshRuns()
    const es = subscribeEvents((line) => {
      let type = 'log'
      try {
        type = (JSON.parse(line) as { type?: string }).type ?? 'log'
      } catch {
        /* 非 JSON 行按 log 处理 */
      }
      setEvents((prev) => [...prev.slice(-49), { id: ++eventSeq.current, type, data: line }])
      void refreshRuns()
    })
    return () => es.close()
  }, [refreshRuns])

  const onFire = async () => {
    setBusy(true)
    try {
      const run = await createSimulatedRun()
      console.log('simulated run', run.id)
      await refreshRuns()
    } catch (e) {
      console.error('create run failed', e)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>天地熔炉 <span className="sub">Tiandi Furnace</span></h1>
        <div className="status">
          {health ? (
            <span className="ok">● 已点火 v{health.version}{health.demo ? '（演示模式）' : ''}</span>
          ) : (
            <span className="err">● 服务未连接{healthError ? `：${healthError}` : ''}</span>
          )}
        </div>
      </header>

      <main>
        <section className="panel">
          <div className="panel-title">
            <h2>炼丹任务</h2>
            <button onClick={onFire} disabled={busy}>
              {busy ? '点火中…' : '点火（模拟炼丹）'}
            </button>
          </div>
          {runs.length === 0 ? (
            <p className="hint">还没有任务。点击「点火」创建一个模拟炼丹任务（演示状态机与事件流）。</p>
          ) : (
            <ul className="runs">
              {runs.map((r) => (
                <li key={r.id} className={`run ${r.state}`}>
                  <span className="run-id">{r.id.slice(0, 8)}</span>
                  <span className="run-state">{STATE_LABEL[r.state] ?? r.state}</span>
                  <span className="run-time">{new Date(r.created_at).toLocaleTimeString()}</span>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="panel">
          <h2>炉火观察孔（事件流）</h2>
          {events.length === 0 ? (
            <p className="hint">SSE 事件流等待中…（{API_BASE}/api/runs/all/events）</p>
          ) : (
            <ul className="events">
              {events.map((e) => (
                <li key={e.id} className={`ev ${e.type}`}>
                  <span className="ev-type">{e.type}</span>
                  <code>{e.data}</code>
                </li>
              ))}
            </ul>
          )}
        </section>
      </main>

      <footer>本地服务 {API_BASE} · 仅绑定 127.0.0.1</footer>
    </div>
  )
}
