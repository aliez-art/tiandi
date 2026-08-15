// 天地熔炉 API 客户端（M0：health/runs/events + 端口发现）

/** 后端基址：桌面壳与浏览器模式均访问本地服务（CORS 已放行）。 */
export const DEFAULT_BASE = 'http://127.0.0.1:18765'

export interface Health {
  ok: boolean
  service: string
  version: string
  demo: boolean
}

export interface Run {
  id: string
  project_id: string | null
  dataset_id: string | null
  recipe_id: string | null
  state: string
  manifest_path: string | null
  created_at: string
  updated_at: string
}

export interface Checkpoint {
  id: string
  run_id: string
  kind: string
  path: string
  created_at: string
}

export interface MetricPoint {
  run_id: string
  step: number
  loss: number | null
  lr: number | null
}

/** 采样图/产物 URL（runs 静态服务）。 */
export function assetUrl(path: string): string {
  return `${base()}/${path}`
}

let apiBase: string | null = null

export function base(): string {
  return apiBase ?? DEFAULT_BASE
}

/**
 * 端口发现：依次探测 18765–18774 的 /api/health，找到可用后端即返回其基址。
 * 覆盖：内嵌服务端口回退、先后启动多个实例等场景。
 */
export async function discoverBase(): Promise<string | null> {
  for (let port = 18765; port <= 18774; port++) {
    const candidate = `http://127.0.0.1:${port}`
    try {
      const res = await fetch(`${candidate}/api/health`, {
        signal: AbortSignal.timeout(800),
      })
      if (res.ok) {
        const h = (await res.json()) as Health
        if (h.ok) {
          apiBase = candidate
          return candidate
        }
      }
    } catch {
      /* 端口未监听或超时：继续探测下一个 */
    }
  }
  return null
}

export async function fetchHealth(): Promise<Health> {
  const res = await fetch(`${base()}/api/health`)
  if (!res.ok) throw new Error(`health ${res.status}`)
  return res.json()
}

export async function listRuns(): Promise<Run[]> {
  const res = await fetch(`${base()}/api/runs`)
  if (!res.ok) throw new Error(`runs ${res.status}`)
  return res.json()
}

export async function createSimulatedRun(): Promise<Run> {
  const res = await fetch(`${base()}/api/runs?simulate=1`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: '{}',
  })
  if (!res.ok) throw new Error(`create run ${res.status}`)
  return res.json()
}

export function subscribeEvents(onEvent: (line: string) => void): EventSource {
  const es = new EventSource(`${base()}/api/runs/all/events`)
  es.onmessage = (e) => onEvent(e.data as string)
  return es
}

export async function listCheckpoints(runId: string): Promise<Checkpoint[]> {
  const res = await fetch(`${base()}/api/runs/${runId}/checkpoints`)
  if (!res.ok) throw new Error(`checkpoints ${res.status}`)
  return res.json()
}

export async function deleteCheckpoint(id: string): Promise<void> {
  const res = await fetch(`${base()}/api/checkpoints/${id}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`delete checkpoint ${res.status}`)
}

export async function renameCheckpoint(id: string, name: string): Promise<Checkpoint> {
  const res = await fetch(`${base()}/api/checkpoints/${id}/rename`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name }),
  })
  if (!res.ok) throw new Error(`rename checkpoint ${res.status}`)
  return res.json()
}

export async function runMetrics(runId: string): Promise<MetricPoint[]> {
  const res = await fetch(`${base()}/api/runs/${runId}/metrics`)
  if (!res.ok) throw new Error(`metrics ${res.status}`)
  return res.json()
}

export async function runLogs(runId: string): Promise<string[]> {
  const res = await fetch(`${base()}/api/runs/${runId}/logs`)
  if (!res.ok) throw new Error(`logs ${res.status}`)
  return res.json()
}
