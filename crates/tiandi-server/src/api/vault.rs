//! 药库 API（PRD FR-601~604）：产物（checkpoint）列表/重命名/删除。

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

/// 训练日志（runs/<id>/logs/training.log 尾部，最多 300 行）。
async fn run_logs(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<Json<Vec<String>>, ApiError> {
    let log_path = state
        .trainer
        .runs_dir()
        .join(&run_id)
        .join("logs/training.log");
    let content = match std::fs::read_to_string(&log_path) {
        Ok(c) => c,
        Err(_) => return Ok(Json(Vec::new())), // 尚无日志
    };
    let lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
    let tail = if lines.len() > 300 {
        lines[lines.len() - 300..].to_vec()
    } else {
        lines
    };
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

/// 删除：移除记录 + 尝试删除文件（相对 runs 根解析）。
async fn delete_one(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let runs_dir = state.trainer.runs_dir().to_path_buf();
    let store = state.store.lock().await;
    let cp = store.get_checkpoint(&id)?;
    store.delete_checkpoint(&id)?;
    drop(store);

    // 文件删除失败不阻塞（记录已删）
    let abs = runs_dir.join(&cp.path);
    if abs.is_file() {
        let _ = std::fs::remove_file(&abs);
    }
    Ok(StatusCode::NO_CONTENT)
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
    let runs_dir = state.trainer.runs_dir().to_path_buf();
    let store = state.store.lock().await;
    let cp = store.get_checkpoint(&id)?;

    // 新路径：同目录 + 新名 + 原扩展名
    let abs = runs_dir.join(&cp.path);
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
        .strip_prefix(&runs_dir)
        .unwrap_or(&new_abs)
        .to_string_lossy()
        .replace('\\', "/");
    store.update_checkpoint_path(&id, &new_path)?;
    Ok(Json(store.get_checkpoint(&id)?))
}
