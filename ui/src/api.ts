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

/** 示例图/产物 URL（output 静态服务，/output/ 前缀；路径按段编码防特殊字符破坏 URL）。 */
export function assetUrl(path: string): string {
  const encoded = path.split('/').map((seg) => encodeURIComponent(seg)).join('/')
  return `${base()}/output/${encoded}`
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

/** 每个任务最新示例图路径（{ run_id: sample_path }）。 */
export async function fetchRunPreviews(): Promise<Record<string, string>> {
  const res = await fetch(`${base()}/api/runs/previews`)
  if (!res.ok) throw new Error(`previews ${res.status}`)
  return res.json()
}

export function subscribeEvents(
  onEvent: (line: string) => void,
  onError?: () => void,
  onOpen?: () => void,
): EventSource {
  const es = new EventSource(`${base()}/api/runs/all/events`)
  es.onmessage = (e) => onEvent(e.data as string)
  if (onError) es.onerror = onError
  if (onOpen) es.onopen = onOpen
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

export async function listRecipes(): Promise<RecipeView[]> {
  const res = await fetch(`${base()}/api/recipes`)
  if (!res.ok) throw new Error(`recipes ${res.status}`)
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

/** 弹出系统文件对话框（后端 rfd 原生对话框）；取消返回 null。5 分钟超时兜底。 */
export async function pickFile(): Promise<string | null> {
  const res = await fetch(`${base()}/api/pick-file`, {
    method: 'POST',
    signal: AbortSignal.timeout(300_000),
  })
  if (!res.ok) throw new Error(`pick file ${res.status}`)
  const j = (await res.json()) as { path: string | null }
  return j.path
}

/** 弹出系统目录对话框；取消返回 null。 */
export async function pickDir(): Promise<string | null> {
  const res = await fetch(`${base()}/api/pick-dir`, {
    method: 'POST',
    signal: AbortSignal.timeout(300_000),
  })
  if (!res.ok) throw new Error(`pick dir ${res.status}`)
  const j = (await res.json()) as { path: string | null }
  return j.path
}

/** 资产导入到工作区 models 目录（base_model / vae / clip），返回正式路径。 */
export async function importAsset(kind: 'base_model' | 'vae' | 'clip', path: string): Promise<string> {
  const res = await fetch(`${base()}/api/import-asset`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ kind, path }),
  })
  if (!res.ok) {
    const err = (await res.json().catch(() => null)) as { error?: string } | null
    throw new Error(err?.error ?? `import asset ${res.status}`)
  }
  const j = (await res.json()) as { path: string }
  return j.path
}

