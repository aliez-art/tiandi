import { useCallback, useEffect, useRef, useState } from 'react'
import {
  createSimulatedRun,
  discoverBase,
  fetchHealth,
  listRuns,
  subscribeEvents,
  type Health,
  type Run,
} from './api'
import Console, { type EventLine } from './components/Console'

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
  const [connecting, setConnecting] = useState(true)
  const [runs, setRuns] = useState<Run[]>([])
  const [events, setEvents] = useState<EventLine[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const eventSeq = useRef(0)

  const refreshRuns = useCallback(async () => {
    try {
      const list = await listRuns()
      setRuns(list)
      // 自动选中第一个（若未选）
      setSelected((prev) => {
        if (prev && list.some((r) => r.id === prev)) return prev
        return list[0]?.id ?? null
      })
    } catch (e) {
      console.error('list runs failed', e)
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    let es: EventSource | null = null

    // 连接循环：端口发现失败则 1.5s 后重试（覆盖服务启动竞态与端口回退）
    const connect = async () => {
      const found = await discoverBase()
      if (cancelled) return
      if (!found) {
        window.setTimeout(connect, 1500)
        return
      }
      try {
        const h = await fetchHealth()
        if (cancelled) return
        setHealth(h)
        setConnecting(false)
      } catch (e) {
        if (cancelled) return
        console.error('health check failed', e)
        window.setTimeout(connect, 1500)
        return
      }
      await refreshRuns()
      if (cancelled) return
      es = subscribeEvents((line) => {
        let type = 'log'
        try {
          type = (JSON.parse(line) as { type?: string }).type ?? 'log'
        } catch {
          /* 非 JSON 行按 log 处理 */
        }
        setEvents((prev) => [...prev.slice(-499), { id: ++eventSeq.current, type, data: line }])
        void refreshRuns()
      })
    }
    void connect()

    return () => {
      cancelled = true
      es?.close()
    }
  }, [refreshRuns])

  const onFire = async () => {
    setBusy(true)
    try {
      const run = await createSimulatedRun()
      setSelected(run.id)
      await refreshRuns()
    } catch (e) {
      console.error('create run failed', e)
    } finally {
      setBusy(false)
    }
  }

  const selectedRun = runs.find((r) => r.id === selected) ?? null

  return (
    <div className="app">
      <header className="header">
        <h1>
          天地熔炉 <span className="sub">Tiandi Furnace</span>
        </h1>
        <div className="status">
          {connecting ? (
            <span className="connecting">◌ 正在点火…</span>
          ) : health ? (
            <span className="ok">
              ● 已点火 v{health.version}
              {health.demo ? '（演示模式）' : ''}
            </span>
          ) : (
            <span className="err">● 服务未连接</span>
          )}
        </div>
      </header>

      <main className="layout">
        {/* 任务列表（丹房） */}
        <aside className="sidebar">
          <div className="panel-title">
            <h2>炼丹任务</h2>
            <button onClick={onFire} disabled={busy || connecting} title="创建模拟炼丹任务">
              {busy ? '点火中…' : '点火'}
            </button>
          </div>
          {runs.length === 0 ? (
            <p className="hint">还没有任务。点击「点火」开始。</p>
          ) : (
            <ul className="runs">
              {runs.map((r) => (
                <li
                  key={r.id}
                  className={`run ${r.state} ${r.id === selected ? 'active' : ''}`}
                  onClick={() => setSelected(r.id)}
                >
                  <span className="run-id">{r.id.slice(0, 8)}</span>
                  <span className="run-state">{STATE_LABEL[r.state] ?? r.state}</span>
                  <span className="run-time">
                    {new Date(r.created_at).toLocaleTimeString()}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </aside>

        {/* 训练控制台 / 药库 */}
        <section className="content">
          {selectedRun ? (
            <Console key={selectedRun.id} run={selectedRun} events={events} />
          ) : (
            <div className="panel">
              <p className="hint">
                选择一个任务查看训练控制台；或点击「点火」创建模拟炼丹任务（完整 IPC 链路演示）。
              </p>
            </div>
          )}
        </section>
      </main>

      <footer>本地服务 127.0.0.1 · 仅绑定本机</footer>
    </div>
  )
}
