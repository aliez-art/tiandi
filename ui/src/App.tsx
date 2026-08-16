import { useCallback, useEffect, useRef, useState } from 'react'
import {
  deleteRun,
  discoverBase,
  fetchHealth,
  fetchSystem,
  listRuns,
  subscribeEvents,
  type Health,
  type Run,
  type SystemInfo,
} from './api'
import Console, { type EventLine } from './components/Console'
import { NewRunBar, RecipeManager } from './components/TrainSetup'

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

/** 单页布局：左列 = 点火/药材/炉房；右列 = 炼丹记录 + 控制台（参考 lora-scripts/kohya 布局）。 */
export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [connecting, setConnecting] = useState(true)
  const [system, setSystem] = useState<SystemInfo | null>(null)
  const [runs, setRuns] = useState<Run[]>([])
  const [events, setEvents] = useState<EventLine[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [showRecipes, setShowRecipes] = useState(false)
  const eventSeq = useRef(0)

  // GPU 监控（3s 轮询）
  useEffect(() => {
    let cancelled = false
    const tick = async () => {
      try {
        const s = await fetchSystem()
        if (!cancelled) setSystem(s)
      } catch {
        /* 服务未就绪时静默 */
      }
      if (!cancelled) window.setTimeout(tick, 3000)
    }
    void tick()
    return () => {
      cancelled = true
    }
  }, [])

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

  const selectedRun = runs.find((r) => r.id === selected) ?? null

  const onDeleteRun = async (run: Run) => {
    if (!window.confirm(`删除任务 ${run.id.slice(0, 8)}？\n将同时删除其日志、产物与采样图（不可恢复）。`)) return
    try {
      await deleteRun(run.id)
      setSelected((prev) => (prev === run.id ? null : prev))
      await refreshRuns()
    } catch (e) {
      alert(`删除失败：${e instanceof Error ? e.message : String(e)}`)
    }
  }

  /** 清空全部已结束任务（串行删除）。 */
  const onClearFinished = async () => {
    const finished = runs.filter((r) => r.state === 'done' || r.state === 'failed' || r.state === 'canceled')
    if (finished.length === 0) return
    if (!window.confirm(`清空 ${finished.length} 条已结束的炼丹记录（含日志/产物，不可恢复）？`)) return
    for (const r of finished) {
      try {
        await deleteRun(r.id)
      } catch {
        /* 单条失败继续 */
      }
    }
    setSelected(null)
    await refreshRuns()
  }

  return (
    <div className="app">
      <header className="header">
        <h1>
          天地熔炉 <span className="sub">Tiandi Furnace</span>
        </h1>
        <div className="status">
          {system?.gpu && (
            <span className="gpu" title={system.gpu.name}>
              GPU {system.gpu.util_percent}% · {(system.gpu.mem_used_mb / 1024).toFixed(1)}/
              {(system.gpu.mem_total_mb / 1024).toFixed(0)}GB
            </span>
          )}
          {connecting ? (
            <span className="connecting">◌ 正在点火…</span>
          ) : health ? (
            <span className="ok">
              ● 已点火 v{health.version}
            </span>
          ) : (
            <span className="err">● 服务未连接</span>
          )}
        </div>
      </header>

      <main className="single-layout">
        {/* 左列：点火（底模/数据集/丹方/点火） */}
        <aside className="config-col">
          <section className="panel">
            <h2>点火</h2>
            <NewRunBar
              onCreated={(runId) => {
                setSelected(runId)
                void refreshRuns()
              }}
              onOpenRecipes={() => setShowRecipes(true)}
            />
          </section>
        </aside>

        {/* 右列：炼丹记录 + 控制台 */}
        <section className="watch-col">
          <div className="panel">
            <div className="panel-title">
              <h2>炼丹记录</h2>
              {runs.some((r) => r.state === 'done' || r.state === 'failed' || r.state === 'canceled') && (
                <button
                  className="secondary"
                  onClick={() => void onClearFinished()}
                  title="删除所有已结束（出炉/炸炉/已取消）的任务"
                >
                  清空已结束
                </button>
              )}
            </div>
            {runs.length === 0 ? (
              <p className="hint">还没有任务。在左侧选齐底模、数据集与丹方后点「点火炼丹」。</p>
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
                    <span className="run-time">{new Date(r.created_at).toLocaleTimeString()}</span>
                    <button
                      className="run-del"
                      title="删除任务（含日志/产物/采样）"
                      onClick={(e) => {
                        e.stopPropagation()
                        void onDeleteRun(r)
                      }}
                    >
                      ✕
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {selectedRun ? (
            <Console key={selectedRun.id} run={selectedRun} events={events} />
          ) : (
            <div className="panel">
              <p className="hint">从「炼丹记录」选择一个任务查看控制台（进度 / loss 曲线 / 日志 / 产物）。</p>
            </div>
          )}
        </section>
      </main>

      <footer>本地服务 127.0.0.1 · 仅绑定本机</footer>
      {showRecipes && <RecipeManager onClose={() => setShowRecipes(false)} />}
    </div>
  )
}
