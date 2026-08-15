//! REST API：健康检查、项目、炼丹任务 CRUD、状态迁移、指标、数据集、丹方。

pub mod datasets;
pub mod recipes;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tiandi_core::{Event, Project, Run, RunState};
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
    let api_state = AppState {
        store: state.store.clone(),
        bus: state.bus.clone(),
        demo: state.demo,
    };
    Router::new()
        .route("/api/health", get(health))
        .route("/api/projects", get(list_projects).post(create_project))
        .route("/api/runs", get(list_runs).post(create_run))
        .route("/api/runs/{id}", get(get_run))
        .route("/api/runs/{id}/transition", post(transition_run))
        .route("/api/runs/{id}/metrics", get(list_metrics))
        .route("/api/runs/{id}/events", get(sse::stream_events))
        // 注：run_id="all" 走 {id} 路由即可（handler 内已支持不过滤语义），
        // 无需单独的静态路由（静态段会让 Path 提取器失败）
        .merge(datasets::routes())
        .merge(recipes::routes())
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
    /// 创建后自动跑一段模拟训练（演示状态机与事件流）
    /// 创建后自动跑一段模拟训练（演示状态机与事件流）：`?simulate=1`
    simulate: Option<u8>,
}

#[derive(Deserialize)]
struct NewRun {
    project_id: Option<String>,
    dataset_id: Option<String>,
    recipe_id: Option<String>,
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
    let run = Run::new(input.project_id, input.dataset_id, input.recipe_id);
    let store = state.store.lock().await;
    store.insert_run(&run)?;
    drop(store);

    if query.simulate == Some(1) {
        if state.demo {
            spawn_simulate(state, run.id.clone());
        } else {
            warn!("simulate 被请求但服务未启用 demo 模式");
        }
    }
    Ok((StatusCode::CREATED, Json(run)))
}

async fn get_run(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Run>, ApiError> {
    let store = state.store.lock().await;
    Ok(Json(store.get_run(&id)?))
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

async fn list_metrics(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<tiandi_core::MetricPoint>>, ApiError> {
    let store = state.store.lock().await;
    Ok(Json(store.list_metrics(&id)?))
}

// ---------- 模拟训练（M0 演示） ----------

/// 模拟训练：created → queued → preparing → running → (progress/metric/sample) → saving → done。
fn spawn_simulate(state: AppState, run_id: String) {
    tokio::spawn(async move {
        let ms = |n: u64| tokio::time::sleep(std::time::Duration::from_millis(n));

        transition(&state, &run_id, RunState::Created, RunState::Queued).await;
        ms(200).await;
        transition(&state, &run_id, RunState::Queued, RunState::Preparing).await;
        ms(400).await;
        transition(&state, &run_id, RunState::Preparing, RunState::Running).await;

        const TOTAL: u64 = 50;
        for step in 1..=TOTAL {
            ms(250).await;
            let loss = 1.0 / (step as f64).sqrt() + 0.05;
            let lr = 1.0e-4;
            state.bus.emit(Event::Progress {
                run_id: run_id.clone(),
                step,
                epoch: step as f32 / TOTAL as f32,
                loss,
                lr,
                eta_s: Some((TOTAL - step) * 250 / 1000),
            });
            // 指标入库（真实训练中由 core 用例消费内核 Metric 事件落库，模拟任务直接落）
            {
                let store = state.store.lock().await;
                let _ = store.insert_metric(&tiandi_core::MetricPoint {
                    run_id: run_id.clone(),
                    step,
                    loss: Some(loss),
                    lr: Some(lr),
                });
            }
            state.bus.emit(Event::Metric {
                run_id: run_id.clone(),
                step,
                loss: Some(loss),
                lr: Some(lr),
            });
            if step % 10 == 0 {
                state.bus.emit(Event::Log {
                    run_id: run_id.clone(),
                    level: "info".into(),
                    msg: format!("采样出图 step {step}（模拟）"),
                });
                state.bus.emit(Event::Sample {
                    run_id: run_id.clone(),
                    path: format!("runs/{run_id}/samples/step-{step:04}.png"),
                });
            }
        }

        transition(&state, &run_id, RunState::Running, RunState::Saving).await;
        ms(300).await;
        transition(&state, &run_id, RunState::Saving, RunState::Done).await;
        state.bus.emit(Event::Done {
            run_id: run_id.clone(),
            code: 0,
        });
        state.bus.emit(Event::Log {
            run_id,
            level: "info".into(),
            msg: "模拟炼丹完成：出炉！".into(),
        });
    });
}

/// 状态迁移（幂等：状态不匹配或迁移非法则忽略）。
async fn transition(state: &AppState, run_id: &str, from: RunState, to: RunState) {
    let store = state.store.lock().await;
    let run = match store.get_run(run_id) {
        Ok(r) => r,
        Err(_) => return,
    };
    if run.state != from || !run.state.can_transition_to(to) {
        return;
    }
    let updated_at = chrono::Utc::now().to_rfc3339();
    if store.update_run_state(run_id, to, &updated_at).is_ok() {
        state.bus.emit(Event::RunStateChanged {
            run_id: run_id.to_string(),
            from: run.state,
            to,
        });
    }
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
        AppState::new(store, EventBus::default(), true)
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
