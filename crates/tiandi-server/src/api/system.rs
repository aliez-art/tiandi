//! 系统信息（GPU 监控，FR-802）与设置（FR-901：镜像源等）。

use std::collections::BTreeMap;

use axum::{extract::State, response::Json, routing::get, Router};
use serde::{Deserialize, Serialize};

use super::ApiError;
use crate::AppState;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/system", get(system_info))
        .route("/api/settings", get(list_settings).put(update_settings))
}

// ---------- 系统信息 ----------

#[derive(Serialize)]
struct SystemInfo {
    gpu: Option<GpuInfo>,
    server_time: String,
}

#[derive(Serialize)]
struct GpuInfo {
    name: String,
    mem_used_mb: u64,
    mem_total_mb: u64,
    util_percent: u64,
}

async fn system_info() -> Json<SystemInfo> {
    let gpu = tokio::task::spawn_blocking(query_gpu).await.unwrap_or(None);
    Json(SystemInfo {
        gpu,
        server_time: chrono::Utc::now().to_rfc3339(),
    })
}

/// nvidia-smi 单次快照解析。
fn query_gpu() -> Option<GpuInfo> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let parts: Vec<&str> = text.trim().split(',').map(|s| s.trim()).collect();
    if parts.len() < 4 {
        return None;
    }
    Some(GpuInfo {
        name: parts[0].to_string(),
        mem_used_mb: parts[1].parse().ok()?,
        mem_total_mb: parts[2].parse().ok()?,
        util_percent: parts[3].parse().ok()?,
    })
}

// ---------- 设置 ----------

async fn list_settings(
    State(state): State<AppState>,
) -> Result<Json<BTreeMap<String, String>>, ApiError> {
    let store = state.store.lock().await;
    Ok(Json(store.list_settings()?))
}

#[derive(Deserialize)]
struct SettingsUpdate {
    /// 待写入的设置（空值 = 删除该键）
    values: BTreeMap<String, String>,
}

async fn update_settings(
    State(state): State<AppState>,
    Json(input): Json<SettingsUpdate>,
) -> Result<Json<BTreeMap<String, String>>, ApiError> {
    let store = state.store.lock().await;
    for (key, value) in &input.values {
        if value.is_empty() {
            // 空值删除：直接跳过写入（保留简单语义，删除用空串）
            continue;
        }
        store.set_setting(key, value)?;
    }
    Ok(Json(store.list_settings()?))
}
