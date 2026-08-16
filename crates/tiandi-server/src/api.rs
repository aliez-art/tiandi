//! REST API：健康检查、项目、炼丹任务 CRUD、状态迁移、指标、数据集、丹方、药库。

pub mod datasets;
pub mod models;
pub mod recipes;
pub mod system;
pub mod vault;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tiandi_core::{Event, Project, Run, RunState};
use tiandi_engine::Trainer as _;
use tracing::warn;

use crate::{sse, AppState};

// ---------- 错误 ----------

/// API 错误（统一 JSON 形态）。
#[derive(Debug)]
pub enum ApiError {
    NotFound(String),
    Conflict(String),
    BadRequest(String),
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, msg) = match self {
            Self::NotFound(m) => (StatusCode::NOT_FOUND, m),
            Self::Conflict(m) => (StatusCode::CONFLICT, m),
            Self::BadRequest(m) => (StatusCode::BAD_REQUEST, m),
            Self::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, m),
        };
        (status, Json(serde_json::json!({ "error": msg }))).into_response()
    }
}

impl From<tiandi_state::RepoError> for ApiError {
    fn from(e: tiandi_state::RepoError) -> Self {
        match e {
            tiandi_state::RepoError::NotFound { entity, id } => {
                Self::NotFound(format!("{entity} {id} 不存在"))
            }
            other => Self::Internal(other.to_string()),
        }
    }
}

// ---------- 路由 ----------

pub fn router(state: AppState) -> Router {
    let api_state = state.clone();
    Router::new()
        .route("/api/health", get(health))
        .route("/api/projects", get(list_projects).post(create_project))
        .route("/api/runs", get(list_runs).post(create_run))
        .route("/api/runs/previews", get(run_previews))
        .route("/api/runs/{id}", get(get_run).delete(delete_run))
        .route("/api/runs/{id}/start", post(start_run))
        .route("/api/runs/{id}/cancel", post(cancel_run))
        .route("/api/runs/{id}/transition", post(transition_run))
        .route("/api/runs/{id}/metrics", get(list_metrics))
        .route("/api/runs/{id}/events", get(sse::stream_events))
        // 注：run_id="all" 走 {id} 路由即可（handler 内已支持不过滤语义），
        // 无需单独的静态路由（静态段会让 Path 提取器失败）
        .merge(datasets::routes())
        .merge(models::routes())
        .merge(system::routes())
        .merge(recipes::routes())
        .merge(vault::routes())
        .with_state(api_state)
}

// ---------- Health ----------

#[derive(Serialize)]
struct Health {
    ok: bool,
    service: &'static str,
    version: &'static str,
    demo: bool,
}

async fn health(State(state): State<AppState>) -> Json<Health> {
    Json(Health {
        ok: true,
        service: "tiandi",
        version: env!("CARGO_PKG_VERSION"),
        demo: state.demo,
    })
}

// ---------- Projects ----------

async fn list_projects(State(state): State<AppState>) -> Result<Json<Vec<Project>>, ApiError> {
    let store = state.store.lock().await;
    Ok(Json(store.list_projects()?))
}

#[derive(Deserialize)]
struct NewProject {
    name: String,
    root_dir: String,
}

async fn create_project(
    State(state): State<AppState>,
    Json(input): Json<NewProject>,
) -> Result<(StatusCode, Json<Project>), ApiError> {
    let project = Project::new(input.name, input.root_dir);
    let store = state.store.lock().await;
    store.insert_project(&project)?;
    Ok((StatusCode::CREATED, Json(project)))
}

// ---------- Runs ----------

#[derive(Deserialize)]
struct CreateRunQuery {
    /// 创建后立即点火（走 mock 内核演示 IPC 全链路）：`?simulate=1`
    simulate: Option<u8>,
}

#[derive(Deserialize)]
struct NewRun {
    project_id: Option<String>,
    dataset_id: Option<String>,
    recipe_id: Option<String>,
    base_model_id: Option<String>,
}

async fn list_runs(State(state): State<AppState>) -> Result<Json<Vec<Run>>, ApiError> {
    let store = state.store.lock().await;
    Ok(Json(store.list_runs()?))
}

async fn create_run(
    State(state): State<AppState>,
    Query(query): Query<CreateRunQuery>,
    Json(input): Json<NewRun>,
) -> Result<(StatusCode, Json<Run>), ApiError> {
    let run = Run::new(
        input.project_id,
        input.dataset_id,
        input.recipe_id,
        input.base_model_id,
    );
    let store = state.store.lock().await;
    store.insert_run(&run)?;
    drop(store);

    if query.simulate == Some(1) {
        if state.demo {
            // 入队（scheduler 自动拉起 mock 内核，完整 IPC 链路）
            enqueue_run(&state, &run.id).await?;
        } else {
            warn!("simulate 被请求但服务未启用 demo 模式");
        }
    }
    Ok((StatusCode::CREATED, Json(run)))
}

/// `POST /api/runs/{id}/start`：入队。scheduler 串行拉起（有丹方 → sd-scripts；无 → mock）。
async fn start_run(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Run>, ApiError> {
    enqueue_run(&state, &id).await?;
    let store = state.store.lock().await;
    Ok(Json(store.get_run(&id)?))
}

/// 入队：Created/Failed → Queued（scheduler 负责拉起）。
async fn enqueue_run(state: &AppState, id: &str) -> Result<(), ApiError> {
    let store = state.store.lock().await;
    let run = store.get_run(id)?;
    if !matches!(run.state, RunState::Created | RunState::Failed) {
        return Err(ApiError::Conflict(format!(
            "任务状态 {} 不可入队（需 已创建/炸炉）",
            run.state.label()
        )));
    }
    let updated_at = chrono::Utc::now().to_rfc3339();
    store.update_run_state(id, RunState::Queued, &updated_at)?;
    state.bus.emit(Event::RunStateChanged {
        run_id: id.to_string(),
        from: run.state,
        to: RunState::Queued,
    });
    Ok(())
}

/// `POST /api/runs/{id}/cancel`：取消训练（两段式：优雅请求 → 超时 kill-tree）。
async fn cancel_run(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Run>, ApiError> {
    state
        .trainer
        .cancel(&id)
        .map_err(|e| ApiError::Conflict(format!("取消失败：{e}")))?;
    let store = state.store.lock().await;
    let run = store.get_run(&id)?;
    if !run.state.is_terminal() {
        let updated_at = chrono::Utc::now().to_rfc3339();
        store.update_run_state(&id, RunState::Canceled, &updated_at)?;
        state.bus.emit(Event::RunStateChanged {
            run_id: id.clone(),
            from: run.state,
            to: RunState::Canceled,
        });
    }
    Ok(Json(store.get_run(&id)?))
}

async fn get_run(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Run>, ApiError> {
    let store = state.store.lock().await;
    Ok(Json(store.get_run(&id)?))
}

/// `DELETE /api/runs/{id}`：删除已结束（出炉/炸炉/已取消）的任务记录与产物。
/// 运行中的任务拒绝删除。
async fn delete_run(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let store = state.store.lock().await;
    let run = store.get_run(&id)?;
    if !run.state.is_terminal() {
        return Err(ApiError::Conflict(format!(
            "任务状态 {} 未结束，不能删除（可先熄灭）",
            run.state.label()
        )));
    }
    store.delete_run(&id)?;
    drop(store);
    // 删除任务目录（配置/日志/产物/采样）
    let dir = state.trainer.runs_dir().join(&id);
    if dir.exists() {
        let _ = std::fs::remove_dir_all(&dir);
    }
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
struct TransitionRequest {
    to: RunState,
}

/// 状态迁移（演示任务状态机；非法迁移返回 409）。
async fn transition_run(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(input): Json<TransitionRequest>,
) -> Result<Json<Run>, ApiError> {
    let store = state.store.lock().await;
    let run = store.get_run(&id)?;
    if !run.state.can_transition_to(input.to) {
        return Err(ApiError::Conflict(format!(
            "非法状态迁移：{}（{}）不可迁移到 {}（{}）",
            run.state.label(),
            serde_json::to_string(&run.state).unwrap_or_default(),
            input.to.label(),
            serde_json::to_string(&input.to).unwrap_or_default()
        )));
    }
    let from = run.state;
    let updated_at = chrono::Utc::now().to_rfc3339();
    store.update_run_state(&id, input.to, &updated_at)?;
    let run = store.get_run(&id)?;
    drop(store);

    state.bus.emit(Event::RunStateChanged {
        run_id: id,
        from,
        to: run.state,
    });
    Ok(Json(run))
}

/// 任务最新示例图（炼丹记录列表缩略图）：{ run_id: sample_path }。
async fn run_previews(
    State(state): State<AppState>,
) -> Result<Json<std::collections::HashMap<String, String>>, ApiError> {
    let store = state.store.lock().await;
    let rows = store.latest_sample_per_run()?;
    Ok(Json(rows.into_iter().collect()))
}

async fn list_metrics(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<tiandi_core::MetricPoint>>, ApiError> {
    let store = state.store.lock().await;
    Ok(Json(store.list_metrics(&id)?))
}

// ---------- 测试 ----------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tiandi_core::EventBus;
    use tiandi_state::Store;
    use tower::ServiceExt;

    fn test_state() -> AppState {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_test_writer()
            .try_init();
        let store = Store::open_in_memory().unwrap();
        let st = AppState::new(
            store,
            EventBus::default(),
            std::env::temp_dir(),
            crate::default_wrapper_path(),
            true,
        );
        // 监督器（事件→状态机/入库）+ 调度器（串行拉起 Queued 任务）
        crate::supervisor::spawn(st.clone());
        crate::queue::spawn(st.clone());
        st
    }

    #[tokio::test]
    async fn health_ok() {
        let app = router(test_state());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/api/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["service"], "tiandi");
    }

    #[tokio::test]
    async fn create_and_transition_run() {
        let app = router(test_state());
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/runs")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"project_id":null,"dataset_id":null,"recipe_id":null}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::CREATED);
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let run: Run = serde_json::from_slice(&body).unwrap();
        assert_eq!(run.state, RunState::Created);

        // 合法迁移：created → queued
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/runs/{}/transition", run.id))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"to":"queued"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        // 非法迁移：created 状态机已到 queued，直接跳 running 非法（queued→running 也不合法）
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/runs/{}/transition", run.id))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"to":"running"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn run_listing_and_missing_run() {
        let app = router(test_state());
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/runs")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        let res = app
            .oneshot(
                Request::builder()
                    .uri("/api/runs/does-not-exist")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn simulate_drives_run_to_done() {
        let app = router(test_state());
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/runs?simulate=1")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::CREATED);
        let run: Run =
            serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();

        // 轮询直到 done（模拟全程约 50*250ms + 900ms ≈ 13.4s；这里给 20s 上限）
        let mut done = false;
        for _ in 0..200 {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            let res = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(format!("/api/runs/{}", run.id))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            let current: Run =
                serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes())
                    .unwrap();
            if current.state == RunState::Done {
                done = true;
                break;
            }
        }
        assert!(done, "模拟训练应最终到达 Done 状态");

        // 指标应已入库
        let res = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/runs/{}/metrics", run.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let metrics: Vec<tiandi_core::MetricPoint> =
            serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
        assert!(!metrics.is_empty(), "模拟训练应产生指标点");
    }

    #[tokio::test]
    async fn sse_stream_delivers_events() {
        let app = router(test_state());
        // 先建一个模拟 run
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/runs?simulate=1")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        let run: Run =
            serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();

        // 订阅事件流（"all" 通道），读前若干帧直到出现 progress
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/api/runs/all/events")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let mut body = res.into_body();
        let mut saw_progress = false;
        for _ in 0..60 {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            match tokio::time::timeout(std::time::Duration::from_millis(300), body.frame()).await {
                Ok(Some(Ok(frame))) => {
                    let data = frame.into_data().unwrap();
                    let text = String::from_utf8_lossy(&data);
                    if text.contains("\"type\":\"progress\"") {
                        saw_progress = true;
                        break;
                    }
                }
                Ok(Some(Err(e))) => panic!("frame error: {e}"),
                _ => continue,
            }
        }
        assert!(
            saw_progress,
            "SSE 流应推送 progress 事件（run {run_id}）",
            run_id = run.id
        );
    }
}
