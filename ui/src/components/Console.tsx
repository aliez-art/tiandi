import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  assetUrl,
  cancelRun,
  deleteCheckpoint,
  listCheckpoints,
  renameCheckpoint,
  runLogs,
  runMetrics,
  startRun,
  type Checkpoint,
  type MetricPoint,
  type Run,
} from '../api'

interface ConsoleProps {
  run: Run
  /** 全局事件流（由 App 持有），按 run 过滤 */
  events: EventLine[]
}

export interface EventLine {
  id: number
  type: string
  data: string
}

/** 日志行（error=true 渲染为红色错误行，如 fail 事件）。 */
interface LogEntry {
  text: string
  error: boolean
}

interface MetricEvent {
  step: number
  loss: number | null
  lr: number | null
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

/** 训练控制台：火候仪表盘（进度/loss/lr 曲线/采样画廊/日志流）+ 药库。 */
export default function Console({ run, events }: ConsoleProps) {
  const runEvents = useMemo(() => events.filter((e) => {
    try {
      return (JSON.parse(e.data) as { run_id?: string }).run_id === run.id
    } catch {
      return false
    }
  }), [events, run.id])

  // 实时指标/日志/采样（事件流）
  const liveMetrics = useMemo(() => {
    const out: MetricEvent[] = []
    for (const e of runEvents) {
      if (e.type !== 'metric') continue
      try {
        const v = JSON.parse(e.data) as MetricEvent
        out.push(v)
      } catch { /* 忽略 */ }
    }
    return out
  }, [runEvents])
  // 实时日志：log 事件 + fail 事件（tail 摘要渲染为红色错误行）
  const liveLogs = useMemo(() => {
    const out: LogEntry[] = []
    for (const e of runEvents) {
      if (e.type === 'log') {
        try {
          const v = JSON.parse(e.data) as { msg?: string; level?: string }
          out.push({ text: `[${v.level ?? 'info'}] ${v.msg ?? ''}`, error: false })
        } catch {
          out.push({ text: e.data, error: false })
        }
      } else if (e.type === 'fail') {
        try {
          const v = JSON.parse(e.data) as { tail?: string; code?: number }
          const tail = (v.tail ?? '').trim()
          out.push({ text: tail ? `[失败] ${tail}` : e.data, error: true })
        } catch {
          out.push({ text: e.data, error: true })
        }
      }
    }
    return out
  }, [runEvents])

  // 历史数据（切换任务时加载）
  const [historyMetrics, setHistoryMetrics] = useState<MetricPoint[]>([])
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [historyLogs, setHistoryLogs] = useState<string[]>([])
  const [latestStep, setLatestStep] = useState<number>(0)
  const [totalSteps, setTotalSteps] = useState<number | null>(null)
  const [latestLoss, setLatestLoss] = useState<number | null>(null)
  const [renaming, setRenaming] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const logBoxRef = useRef<HTMLDivElement>(null)
  /** 采样事件后刷新药库的防抖计时器（1s） */
  const checkpointTimer = useRef<number | null>(null)

  const load = useCallback(async () => {
    try {
      const [m, cps, logs] = await Promise.all([
        runMetrics(run.id),
        listCheckpoints(run.id),
        runLogs(run.id),
      ])
      setHistoryMetrics(m)
      setCheckpoints(cps)
      setHistoryLogs(logs)
    } catch (e) {
      console.error('load console failed', e)
    }
  }, [run.id])

  useEffect(() => {
    void load()
  }, [load])

  // 实时进度：最近 progress 事件
  useEffect(() => {
    let step = 0
    let total: number | null = null
    let loss: number | null = null
    for (const e of runEvents) {
      if (e.type !== 'progress') continue
      try {
        const v = JSON.parse(e.data) as { step?: number; total?: number; loss?: number }
        if (typeof v.step === 'number') step = v.step
        if (typeof v.total === 'number') total = v.total
        if (typeof v.loss === 'number') loss = v.loss
      } catch { /* 忽略 */ }
    }
    setLatestStep(step)
    setTotalSteps(total)
    setLatestLoss(loss)
    // 采样事件后防抖刷新药库（1s），避免连续 sample 事件重复拉取
    if (runEvents.some((e) => e.type === 'sample')) {
      if (checkpointTimer.current !== null) window.clearTimeout(checkpointTimer.current)
      checkpointTimer.current = window.setTimeout(() => {
        checkpointTimer.current = null
        void listCheckpoints(run.id).then(setCheckpoints).catch(() => {})
      }, 1000)
    }
    return () => {
      if (checkpointTimer.current !== null) {
        window.clearTimeout(checkpointTimer.current)
        checkpointTimer.current = null
      }
    }
  }, [runEvents, run.id])

  // 日志自动滚动
  useEffect(() => {
    if (logBoxRef.current) {
      logBoxRef.current.scrollTop = logBoxRef.current.scrollHeight
    }
  }, [liveLogs, historyLogs])

  // 合并曲线数据（历史 + 实时，按 step 去重）
  const curve = useMemo(() => {
    const map = new Map<number, MetricEvent>()
    for (const m of historyMetrics) {
      map.set(m.step, { step: m.step, loss: m.loss, lr: m.lr })
    }
    for (const m of liveMetrics) {
      map.set(m.step, m)
    }
    return [...map.values()].sort((a, b) => a.step - b.step)
  }, [historyMetrics, liveMetrics])

  const allLogs = useMemo<LogEntry[]>(
    () => [...historyLogs.map((l) => ({ text: l, error: false })), ...liveLogs],
    [historyLogs, liveLogs],
  )

  const pct = totalSteps && totalSteps > 0 ? Math.min(100, (latestStep / totalSteps) * 100) : 0

  const onDelete = async (id: string) => {
    if (!window.confirm('确定删除该产物？（记录与文件）')) return
    try {
      await deleteCheckpoint(id)
      setCheckpoints((prev) => prev.filter((c) => c.id !== id))
    } catch (e) {
      alert(`删除失败：${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const onRename = async (id: string) => {
    try {
      const updated = await renameCheckpoint(id, renameValue)
      setCheckpoints((prev) => prev.map((c) => (c.id === id ? updated : c)))
    } catch (e) {
      alert(`重命名失败：${e instanceof Error ? e.message : String(e)}`)
    }
    setRenaming(null)
    setRenameValue('')
  }

  const [opError, setOpError] = useState<string | null>(null)

  const onResume = async () => {
    setOpError(null)
    try {
      await startRun(run.id)
      window.setTimeout(() => window.location.reload(), 300)
    } catch (e) {
      setOpError(`续丹失败：${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const onCancel = async () => {
    if (!window.confirm('确定取消当前任务？')) return
    setOpError(null)
    try {
      await cancelRun(run.id)
      window.setTimeout(() => window.location.reload(), 300)
    } catch (e) {
      setOpError(`取消失败：${e instanceof Error ? e.message : String(e)}`)
    }
  }

  return (
    <div className="console">
      {/* 火候仪表盘 */}
      <section className="panel gauge">
        <div
          className="gauge-ring"
          style={{ '--pct': `${pct}%` } as React.CSSProperties}
          role="progressbar"
          aria-label="训练进度"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(pct)}
        >
          <div className="gauge-inner">
            <div className="gauge-pct">{Math.round(pct)}%</div>
            <div className="gauge-sub">{latestStep}{totalSteps ? ` / ${totalSteps}` : ''} 步</div>
          </div>
        </div>
        <div className="gauge-info">
          <div className="gauge-row">
            <span className="gauge-label">状态</span>
            <span className={`state-pill ${run.state}`}>{STATE_LABEL[run.state] ?? run.state}</span>
          </div>
          <div className="gauge-row">
            <span className="gauge-label">当前 loss</span>
            <span>{latestLoss !== null ? latestLoss.toFixed(5) : '—'}</span>
          </div>
          <div className="gauge-row">
            <span className="gauge-label">任务</span>
            <span className="mono">{run.id.slice(0, 8)}</span>
          </div>
          <div className="gauge-row">
            <span className="gauge-label">创建于</span>
            <span>{new Date(run.created_at).toLocaleString()}</span>
          </div>
          {(run.state === 'failed' || run.state === 'running' || run.state === 'queued') && (
            <div className="gauge-row actions">
              {run.state === 'failed' && (
                <button onClick={onResume}>续丹（重试/断点续训）</button>
              )}
              {(run.state === 'running' || run.state === 'queued') && (
                <button className="danger" onClick={onCancel}>熄灭（取消）</button>
              )}
            </div>
          )}
          {opError && <p className="status-line error">{opError}</p>}
        </div>
      </section>

      {/* 曲线 */}
      <section className="panel">
        <h2>损失曲线（loss / lr）</h2>
        {curve.length < 2 ? (
          <p className="hint">等待指标数据…</p>
        ) : (
          <LossCurve points={curve} />
        )}
      </section>

      {/* 采样画廊 + 药库 */}
      <section className="panel">
        <h2>采样画廊 / 药库</h2>
        {checkpoints.length === 0 ? (
          <p className="hint">暂无产物。训练中的采样出图会出现在这里。</p>
        ) : (
          <ul className="gallery">
            {checkpoints.map((cp) => (
              <li key={cp.id} className="gallery-item">
                {cp.kind === 'sample' ? (
                  <img src={assetUrl(cp.path)} alt={cp.path} loading="lazy" />
                ) : (
                  <div className="file-tile">{cp.path.split('/').pop()}</div>
                )}
                <div className="gallery-meta">
                  <code>{cp.path.split('/').pop()}</code>
                  <div className="gallery-actions">
                    {renaming === cp.id ? (
                      <>
                        <input
                          value={renameValue}
                          onChange={(e) => setRenameValue(e.target.value)}
                          onKeyDown={(e) => e.key === 'Enter' && onRename(cp.id)}
                          placeholder="新名称"
                        />
                        <button onClick={() => onRename(cp.id)}>确定</button>
                        <button onClick={() => setRenaming(null)}>取消</button>
                      </>
                    ) : (
                      <>
                        <button
                          onClick={() => {
                            setRenaming(cp.id)
                            setRenameValue(cp.path.split('/').pop() ?? '')
                          }}
                        >
                          重命名
                        </button>
                        <button className="danger" onClick={() => onDelete(cp.id)}>
                          删除
                        </button>
                      </>
                    )}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* 日志流 */}
      <section className="panel">
        <h2>炉火观察孔（日志）</h2>
        <div className="logbox" ref={logBoxRef}>
          {allLogs.length === 0 ? (
            <p className="hint">等待日志…</p>
          ) : (
            allLogs.map((l, i) => (
              <div key={i} className={`logline${l.error ? ' log-error' : ''}`}>{l.text}</div>
            ))
          )}
        </div>
      </section>
    </div>
  )
}

/** SVG 损失曲线（最近 200 点）。 */
function LossCurve({ points }: { points: MetricEvent[] }) {
  const W = 720
  const H = 180
  const PAD = 8
  const recent = points.slice(-200)

  const losses = recent.map((p) => p.loss).filter((v): v is number => v !== null)
  const maxLoss = losses.length ? Math.max(...losses) : 1
  const minLoss = losses.length ? Math.min(...losses) : 0
  const span = Math.max(maxLoss - minLoss, 1e-6)

  const lossPath = recent
    .map((p, i) => {
      if (p.loss === null) return ''
      const x = PAD + (i / Math.max(recent.length - 1, 1)) * (W - PAD * 2)
      const y = PAD + (1 - (p.loss - minLoss) / span) * (H - PAD * 2)
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="curve" role="img" aria-label="损失曲线">
      <line x1={PAD} y1={H / 2} x2={W - PAD} y2={H / 2} stroke="#33364a" strokeDasharray="4 4" />
      {lossPath && <path d={lossPath} fill="none" stroke="#c0392b" strokeWidth="2" />}
    </svg>
  )
}
