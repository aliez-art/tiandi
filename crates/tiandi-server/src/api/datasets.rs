//! 数据集 API：注册、扫描（导入/去重/分桶/缩略图）、统计、图像列表。

use std::path::PathBuf;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tiandi_core::Dataset;
use tiandi_dataset::{scan_dataset_dir, ScanOptions};
use tiandi_state::ImageRecord;

use super::ApiError;
use crate::AppState;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/datasets", get(list_datasets).post(create_dataset))
        .route(
            "/api/datasets/{id}",
            get(get_dataset).delete(delete_dataset),
        )
        .route("/api/datasets/{id}/scan", post(scan_dataset))
        .route("/api/datasets/{id}/images", get(list_images))
        .route("/api/datasets/{id}/buckets", get(bucket_distribution))
}

// ---------- 删除 ----------

/// `DELETE /api/datasets/{id}`：删除数据集记录与图像索引（磁盘文件不动）。
async fn delete_dataset(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let store = state.store.lock().await;
    store.delete_dataset(&id)?;
    Ok(StatusCode::NO_CONTENT)
}

// ---------- CRUD ----------

async fn list_datasets(State(state): State<AppState>) -> Result<Json<Vec<Dataset>>, ApiError> {
    let store = state.store.lock().await;
    Ok(Json(store.list_datasets()?))
}

async fn get_dataset(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Dataset>, ApiError> {
    let store = state.store.lock().await;
    Ok(Json(store.get_dataset(&id)?))
}

#[derive(Deserialize)]
struct NewDataset {
    name: String,
    /// 数据集根目录（含 `N_` 重复子集约定）
    dir: String,
}

async fn create_dataset(
    State(state): State<AppState>,
    Json(input): Json<NewDataset>,
) -> Result<(StatusCode, Json<Dataset>), ApiError> {
    let dir = PathBuf::from(&input.dir);
    if !dir.is_dir() {
        return Err(ApiError::BadRequest(format!("目录不存在：{}", input.dir)));
    }
    let dataset = Dataset::new(input.name, input.dir);
    let store = state.store.lock().await;
    store.insert_dataset(&dataset)?;
    Ok((StatusCode::CREATED, Json(dataset)))
}

// ---------- 扫描 ----------

#[derive(Deserialize)]
struct ScanRequest {
    /// 目标分辨率（默认 1024）
    resolution: Option<u32>,
    /// 桶步进（默认 64）
    bucket_steps: Option<u32>,
    /// 缩略图最长边（默认 256；0 = 不生成）
    thumb_size: Option<u32>,
    /// 重复判定阈值（默认 5）
    hash_threshold: Option<u32>,
}

/// 扫描响应：报告 + 入库图像数。
#[derive(Serialize)]
struct ScanResponse {
    report: tiandi_dataset::ScanReport,
    images: u64,
}

async fn scan_dataset(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(input): Json<ScanRequest>,
) -> Result<Json<ScanResponse>, ApiError> {
    let dataset = {
        let store = state.store.lock().await;
        store.get_dataset(&id)?
    };
    let root = PathBuf::from(&dataset.dir);
    if !root.is_dir() {
        return Err(ApiError::BadRequest(format!(
            "数据集目录不存在：{}",
            dataset.dir
        )));
    }

    let thumb_size = input.thumb_size.unwrap_or(256);
    let thumb_dir = (thumb_size > 0).then(|| root.join("thumbs"));
    if let Some(td) = &thumb_dir {
        std::fs::create_dir_all(td)
            .map_err(|e| ApiError::Internal(format!("创建缩略图目录失败：{e}")))?;
    }

    let options = ScanOptions {
        target_resolution: input.resolution.unwrap_or(1024),
        bucket_steps: input.bucket_steps.unwrap_or(64),
        bucket_min_ratio: 0.85,
        bucket_max_ratio: 1.15,
        thumb_size,
        hash_threshold: input.hash_threshold.unwrap_or(5),
        thumb_dir,
    };

    // 解码/哈希为 CPU 密集任务，走 blocking 池
    let result = tokio::task::spawn_blocking(move || scan_dataset_dir(&root, &options))
        .await
        .map_err(|e| ApiError::Internal(format!("扫描任务异常：{e}")))?
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    // 入库：重复组中保留每组第一张为主图，其余标 duplicate_of
    let mut records: Vec<ImageRecord> = Vec::with_capacity(result.entries.len());
    let mut primary_of: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for group in &result.report.duplicate_groups {
        if let Some(first) = group.first() {
            primary_of.insert(first.clone(), first.clone());
            for dup in group.iter().skip(1) {
                primary_of.insert(dup.clone(), first.clone());
            }
        }
    }
    let mut created_ids: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for entry in result.entries {
        let rec_id = uuid::Uuid::new_v4().to_string();
        created_ids.insert(entry.path.clone(), rec_id.clone());
        records.push(ImageRecord {
            id: rec_id,
            dataset_id: id.clone(),
            path: entry.path.clone(),
            width: Some(entry.width),
            height: Some(entry.height),
            dhash: Some(entry.dhash),
            bucket: entry.bucket,
            thumb: entry.thumb,
            exif: entry.exif.map(|v| v.to_string()),
            duplicate_of: None,
            created_at: chrono::Utc::now().to_rfc3339(),
        });
    }
    // 二次遍历补 duplicate_of（引用主图 id）
    for group in &result.report.duplicate_groups {
        if let Some(first) = group.first() {
            if let Some(primary_id) = created_ids.get(first) {
                for dup in group.iter().skip(1) {
                    if let Some(dup_id) = created_ids.get(dup) {
                        if let Some(rec) = records.iter_mut().find(|r| r.id == *dup_id) {
                            rec.duplicate_of = Some(primary_id.clone());
                        }
                    }
                }
            }
        }
    }

    let store = state.store.lock().await;
    store.replace_dataset_images(&id, &records)?;
    let images = store.count_dataset_images(&id)?;

    Ok(Json(ScanResponse {
        report: result.report,
        images,
    }))
}

// ---------- 查询 ----------

async fn list_images(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<ImageRecord>>, ApiError> {
    let store = state.store.lock().await;
    // 校验数据集存在
    store.get_dataset(&id)?;
    Ok(Json(store.list_dataset_images(&id)?))
}

async fn bucket_distribution(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<(String, u64)>>, ApiError> {
    let store = state.store.lock().await;
    store.get_dataset(&id)?;
    Ok(Json(store.dataset_bucket_distribution(&id)?))
}
