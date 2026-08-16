//! 药库 API（PRD FR-601~604）：产物（checkpoint）列表/重命名/删除。

use std::io::{Read, Seek, SeekFrom};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::Deserialize;
use tiandi_core::Checkpoint;

use super::ApiError;
use crate::AppState;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/checkpoints", get(list_all))
        .route("/api/runs/{run_id}/checkpoints", get(list_by_run))
        .route("/api/checkpoints/{id}", get(get_one))
        .route("/api/checkpoints/{id}", axum::routing::delete(delete_one))
        .route("/api/checkpoints/{id}/rename", post(rename_one))
        .route("/api/runs/{run_id}/logs", get(run_logs))
}

/// 训练日志（runs/<id>/logs/training.log 尾部，最多 64KB / 300 行）。
/// 防路径遍历：任务必须存在，且日志文件 canonicalize 后必须位于 runs 目录内。
async fn run_logs(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<Json<Vec<String>>, ApiError> {
    // 任务必须存在（不存在 → NotFound）
    {
        let store = state.store.lock().await;
        store.get_run(&run_id)?;
    }
    let runs_dir = state.trainer.runs_dir();
    let log_path = runs_dir.join(&run_id).join("logs").join("training.log");
    // 规范化校验：日志文件必须位于 runs 目录内（防路径遍历；文件不存在 → 尚无日志）
    let canon_runs = std::fs::canonicalize(runs_dir)
        .map_err(|e| ApiError::BadRequest(format!("运行目录不可访问：{e}")))?;
    let canon_log = match std::fs::canonicalize(&log_path) {
        Ok(p) => p,
        Err(_) => return Ok(Json(Vec::new())), // 尚无日志
    };
    if !canon_log.starts_with(&canon_runs) {
        return Err(ApiError::BadRequest("日志路径越界".into()));
    }
    // 尾部读取：最多读末尾 64KB，按行切分取最后 300 行（避免整文件载入内存）
    let mut file = std::fs::File::open(&log_path)
        .map_err(|e| ApiError::BadRequest(format!("日志读取失败：{e}")))?;
    let len = file
        .metadata()
        .map_err(|e| ApiError::BadRequest(format!("日志读取失败：{e}")))?
        .len();
    if len >= 65_536 {
        file.seek(SeekFrom::End(-(65_536i64)))
    } else {
        file.seek(SeekFrom::Start(0))
    }
    .map_err(|e| ApiError::BadRequest(format!("日志读取失败：{e}")))?;
    let mut buf = Vec::with_capacity(65_536);
    file.read_to_end(&mut buf)
        .map_err(|e| ApiError::BadRequest(format!("日志读取失败：{e}")))?;
    let text = String::from_utf8_lossy(&buf);
    let all: Vec<&str> = text.lines().collect();
    let tail: Vec<String> = all
        .iter()
        .skip(all.len().saturating_sub(300))
        .map(|s| s.to_string())
        .collect();
    Ok(Json(tail))
}

async fn list_all(State(state): State<AppState>) -> Result<Json<Vec<Checkpoint>>, ApiError> {
    let store = state.store.lock().await;
    Ok(Json(store.list_all_checkpoints()?))
}

async fn list_by_run(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<Json<Vec<Checkpoint>>, ApiError> {
    let store = state.store.lock().await;
    store.get_run(&run_id)?;
    Ok(Json(store.list_checkpoints(&run_id)?))
}

async fn get_one(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Checkpoint>, ApiError> {
    let store = state.store.lock().await;
    Ok(Json(store.get_checkpoint(&id)?))
}

/// 删除：移除记录 + 尝试删除文件（相对 output 根解析；越界路径拒绝）。
async fn delete_one(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let output_root = crate::output_root(state.trainer.runs_dir());
    let store = state.store.lock().await;
    let cp = store.get_checkpoint(&id)?;
    let abs = output_root.join(&cp.path);
    // 越界防护：规范化后必须位于 output 根内（防产物路径穿越）
    assert_within_root(&output_root, &abs)?;
    store.delete_checkpoint(&id)?;
    drop(store);

    // 文件删除失败不阻塞（记录已删）
    if abs.is_file() {
        let _ = std::fs::remove_file(&abs);
    }
    Ok(StatusCode::NO_CONTENT)
}

/// 断言绝对路径位于 output 根内（两侧 canonicalize 后 starts_with）。
/// 文件不存在（canonicalize 失败）时跳过断言（无文件可操作）。
fn assert_within_root(root: &std::path::Path, abs: &std::path::Path) -> Result<(), ApiError> {
    let canon_root = std::fs::canonicalize(root)
        .map_err(|e| ApiError::BadRequest(format!("产物目录不可访问：{e}")))?;
    if let Ok(canon_abs) = std::fs::canonicalize(abs) {
        if !canon_abs.starts_with(&canon_root) {
            return Err(ApiError::BadRequest("产物路径越界".into()));
        }
    }
    Ok(())
}

#[derive(Deserialize)]
struct RenameRequest {
    name: String,
}

/// 重命名产物文件（保留目录与扩展名），并更新记录。
async fn rename_one(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(input): Json<RenameRequest>,
) -> Result<Json<Checkpoint>, ApiError> {
    let name = input.name.trim().to_string();
    if name.is_empty() || name.contains(['/', '\\', ':']) {
        return Err(ApiError::BadRequest(
            "名称不能为空且不能包含路径分隔符".into(),
        ));
    }
    let output_root = crate::output_root(state.trainer.runs_dir());
    let store = state.store.lock().await;
    let cp = store.get_checkpoint(&id)?;
    let abs = output_root.join(&cp.path);
    // 越界防护：删除/重命名前 canonicalize 校验必须在 output 根内
    assert_within_root(&output_root, &abs)?;

    // 新路径：同目录 + 新名 + 原扩展名
    let new_abs = match abs.parent() {
        Some(parent) => {
            let ext = abs.extension().and_then(|e| e.to_str()).unwrap_or("");
            let mut file_name = name.clone();
            if !ext.is_empty() {
                file_name.push('.');
                file_name.push_str(ext);
            }
            parent.join(file_name)
        }
        None => return Err(ApiError::Internal("产物路径异常".into())),
    };
    std::fs::rename(&abs, &new_abs)
        .map_err(|e| ApiError::BadRequest(format!("重命名文件失败：{e}")))?;

    let new_path = new_abs
        .strip_prefix(&output_root)
        .unwrap_or(&new_abs)
        .to_string_lossy()
        .replace('\\', "/");
    store.update_checkpoint_path(&id, &new_path)?;
    Ok(Json(store.get_checkpoint(&id)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tiandi_core::{EventBus, Run};
    use tiandi_state::Store;
    use tower::ServiceExt;

    fn test_app(runs_dir: std::path::PathBuf) -> (axum::Router, AppState) {
        let store = Store::open_in_memory().unwrap();
        let st = AppState::new(
            store,
            EventBus::default(),
            runs_dir,
            std::env::temp_dir(),
            true,
        );
        let app = routes().with_state(st.clone());
        (app, st)
    }

    #[tokio::test]
    async fn run_logs_missing_run_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let (app, _st) = test_app(tmp.path().join("runs"));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/api/runs/does-not-exist/logs")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn run_logs_empty_when_no_log_file() {
        let tmp = tempfile::tempdir().unwrap();
        let runs_dir = tmp.path().join("runs");
        std::fs::create_dir_all(&runs_dir).unwrap();
        let (app, st) = test_app(runs_dir);
        let run = Run::new(None, None, None, None);
        {
            let s = st.store.lock().await;
            s.insert_run(&run).unwrap();
        }
        let res = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/runs/{}/logs", run.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body: Vec<String> =
            serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
        assert!(body.is_empty(), "无日志时应返回空数组");
    }

    #[tokio::test]
    async fn run_logs_returns_last_300_lines() {
        let tmp = tempfile::tempdir().unwrap();
        let runs_dir = tmp.path().join("runs");
        std::fs::create_dir_all(&runs_dir).unwrap();
        let (app, st) = test_app(runs_dir.clone());
        let run = Run::new(None, None, None, None);
        {
            let s = st.store.lock().await;
            s.insert_run(&run).unwrap();
        }
        let log_path = runs_dir.join(&run.id).join("logs/training.log");
        std::fs::create_dir_all(log_path.parent().unwrap()).unwrap();
        let lines: Vec<String> = (0..500).map(|i| format!("line-{i:04}")).collect();
        std::fs::write(&log_path, lines.join("\n") + "\n").unwrap();
        let res = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/runs/{}/logs", run.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body: Vec<String> =
            serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
        assert_eq!(body.len(), 300, "应只返回最后 300 行");
        assert_eq!(body[0], "line-0200");
        assert_eq!(body[299], "line-0499");
    }

    /// 符号链接逃逸：日志文件指向 runs 目录外 → 400（防路径遍历）。
    #[cfg(unix)]
    #[tokio::test]
    async fn run_logs_rejects_symlink_escape() {
        let tmp = tempfile::tempdir().unwrap();
        let runs_dir = tmp.path().join("runs");
        std::fs::create_dir_all(&runs_dir).unwrap();
        let (app, st) = test_app(runs_dir.clone());
        let run = Run::new(None, None, None, None);
        {
            let s = st.store.lock().await;
            s.insert_run(&run).unwrap();
            s.update_run_state(&run.id, tiandi_core::RunState::Running, "t")
                .unwrap();
        }
        let outside = tmp.path().join("secret.txt");
        std::fs::write(&outside, "secret").unwrap();
        let log_dir = runs_dir.join(&run.id).join("logs");
        std::fs::create_dir_all(&log_dir).unwrap();
        std::os::unix::fs::symlink(&outside, log_dir.join("training.log")).unwrap();
        let res = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/runs/{}/logs", run.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn delete_one_out_of_bounds_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let runs_dir = tmp.path().join("runs");
        std::fs::create_dir_all(&runs_dir).unwrap();
        // output 根 = <tmp>/output；构造根外文件，checkpoint.path 用 .. 逃逸
        let outside = tmp.path().join("escape.txt");
        std::fs::write(&outside, "secret").unwrap();
        let (app, st) = test_app(runs_dir);
        {
            let s = st.store.lock().await;
            s.insert_checkpoint(&Checkpoint {
                id: "cp1".into(),
                run_id: "r1".into(),
                kind: "lora".into(),
                path: "../escape.txt".into(),
                created_at: "t".into(),
            })
            .unwrap();
        }
        let res = app
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/api/checkpoints/cp1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        // 记录与文件都未被删除
        assert!(outside.exists(), "越界文件不得被删除");
        let s = st.store.lock().await;
        assert!(s.get_checkpoint("cp1").is_ok(), "越界记录不得被删除");
    }
}
