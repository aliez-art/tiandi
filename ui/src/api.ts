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

/** 采样图/产物 URL（runs 静态服务，/runs/ 前缀）。 */
export function assetUrl(path: string): string {
  return `${base()}/runs/${path}`
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

export interface Dataset {
  id: string
  name: string
  dir: string
  image_count: number
  created_at: string
}

export async function listDatasets(): Promise<Dataset[]> {
  const res = await fetch(`${base()}/api/datasets`)
  if (!res.ok) throw new Error(`datasets ${res.status}`)
  return res.json()
}

export interface RecipeView {
  id: string
  name: string
  family: string
  data: Record<string, unknown>
  created_at: string
}

export interface PresetView {
  name: string
  family: string
  description: string
  tags: string[]
  data: Record<string, unknown>
}

export async function listRecipes(): Promise<RecipeView[]> {
  const res = await fetch(`${base()}/api/recipes`)
  if (!res.ok) throw new Error(`recipes ${res.status}`)
  return res.json()
}

export async function listPresets(): Promise<PresetView[]> {
  const res = await fetch(`${base()}/api/recipes/presets`)
  if (!res.ok) throw new Error(`presets ${res.status}`)
  return res.json()
}

export async function createRecipe(
  name: string,
  family: string,
  data: Record<string, unknown>,
): Promise<{ recipe: RecipeView; issues: unknown[] }> {
  const res = await fetch(`${base()}/api/recipes`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name, family, data }),
  })
  if (!res.ok) {
    const err = (await res.json().catch(() => null)) as { error?: string } | null
    throw new Error(err?.error ?? `create recipe ${res.status}`)
  }
  return res.json()
}

export async function deleteRecipe(id: string): Promise<void> {
  const res = await fetch(`${base()}/api/recipes/${id}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`delete recipe ${res.status}`)
}

export async function createRun(input: {
  dataset_id: string | null
  recipe_id: string | null
  base_model_id: string | null
}): Promise<Run> {
  const res = await fetch(`${base()}/api/runs`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(input),
  })
  if (!res.ok) throw new Error(`create run ${res.status}`)
  return res.json()
}

export async function createDataset(name: string, dir: string): Promise<Dataset> {
  const res = await fetch(`${base()}/api/datasets`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name, dir }),
  })
  if (!res.ok) throw new Error(`create dataset ${res.status}`)
  return res.json()
}

export async function scanDataset(
  id: string,
  opts?: { resolution?: number },
): Promise<{ report: { total: number; invalid: number; duplicate_groups: string[][]; buckets: [string, number][]; elapsed_ms: number }; images: number }> {
  const res = await fetch(`${base()}/api/datasets/${id}/scan`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(opts ?? {}),
  })
  if (!res.ok) throw new Error(`scan dataset ${res.status}`)
  return res.json()
}

export interface BaseModel {
  id: string
  name: string
  family: string
  path: string | null
  sha256: string | null
  source: string | null
  created_at: string
}

export async function listModels(): Promise<BaseModel[]> {
  const res = await fetch(`${base()}/api/models`)
  if (!res.ok) throw new Error(`models ${res.status}`)
  return res.json()
}

export async function registerModel(input: {
  name: string
  family: string
  path: string
  source?: string
}): Promise<BaseModel> {
  const res = await fetch(`${base()}/api/models`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(input),
  })
  if (!res.ok) throw new Error(`register model ${res.status}`)
  return res.json()
}

export interface SystemInfo {
  gpu: { name: string; mem_used_mb: number; mem_total_mb: number; util_percent: number } | null
  server_time: string
}

export async function fetchSystem(): Promise<SystemInfo> {
  const res = await fetch(`${base()}/api/system`)
  if (!res.ok) throw new Error(`system ${res.status}`)
  return res.json()
}

export async function fetchSettings(): Promise<Record<string, string>> {
  const res = await fetch(`${base()}/api/settings`)
  if (!res.ok) throw new Error(`settings ${res.status}`)
  return res.json()
}

export async function updateSettings(values: Record<string, string>): Promise<Record<string, string>> {
  const res = await fetch(`${base()}/api/settings`, {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ values }),
  })
  if (!res.ok) throw new Error(`settings ${res.status}`)
  return res.json()
}

export async function startRun(runId: string): Promise<Run> {
  const res = await fetch(`${base()}/api/runs/${runId}/start`, { method: 'POST' })
  if (!res.ok) throw new Error(`start run ${res.status}`)
  return res.json()
}

export async function cancelRun(runId: string): Promise<Run> {
  const res = await fetch(`${base()}/api/runs/${runId}/cancel`, { method: 'POST' })
  if (!res.ok) throw new Error(`cancel run ${res.status}`)
  return res.json()
}

export async function deleteRun(runId: string): Promise<void> {
  const res = await fetch(`${base()}/api/runs/${runId}`, { method: 'DELETE' })
  if (!res.ok) {
    const err = (await res.json().catch(() => null)) as { error?: string } | null
    throw new Error(err?.error ?? `delete run ${res.status}`)
  }
}
