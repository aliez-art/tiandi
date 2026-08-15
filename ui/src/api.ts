// 天地熔炉 API 客户端（M0 空壳：health/runs/events）

/** 后端基址：桌面壳与浏览器模式均访问本地服务（CORS 已放行） */
export const API_BASE =
  import.meta.env.DEV && import.meta.env.VITE_USE_PROXY === '1'
    ? ''
    : 'http://127.0.0.1:18765'

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

export interface Health {
  ok: boolean
  service: string
  version: string
  demo: boolean
}

export async function fetchHealth(): Promise<Health> {
  const res = await fetch(`${API_BASE}/api/health`)
  if (!res.ok) throw new Error(`health ${res.status}`)
  return res.json()
}

export async function listRuns(): Promise<Run[]> {
  const res = await fetch(`${API_BASE}/api/runs`)
  if (!res.ok) throw new Error(`runs ${res.status}`)
  return res.json()
}

export async function createSimulatedRun(): Promise<Run> {
  const res = await fetch(`${API_BASE}/api/runs?simulate=1`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: '{}',
  })
  if (!res.ok) throw new Error(`create run ${res.status}`)
  return res.json()
}

export function subscribeEvents(onEvent: (line: string) => void): EventSource {
  const es = new EventSource(`${API_BASE}/api/runs/all/events`)
  es.onmessage = (e) => onEvent(e.data as string)
  return es
}
