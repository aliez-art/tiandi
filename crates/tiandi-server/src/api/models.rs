//! 基底模型注册 API（PRD FR-102/103）。

use axum::{extract::State, http::StatusCode, response::Json, routing::get, Router};
use serde::Deserialize;
use tiandi_core::{BaseModel, ModelFamily};

use super::ApiError;
use crate::AppState;

pub fn routes() -> Router<AppState> {
    Router::new().route("/api/models", get(list_models).post(register_model))
}

async fn list_models(State(state): State<AppState>) -> Result<Json<Vec<BaseModel>>, ApiError> {
    let store = state.store.lock().await;
    Ok(Json(store.list_base_models()?))
}

#[derive(Deserialize)]
struct NewModel {
    name: String,
    family: String,
    path: String,
    source: Option<String>,
}

async fn register_model(
    State(state): State<AppState>,
    Json(input): Json<NewModel>,
) -> Result<(StatusCode, Json<BaseModel>), ApiError> {
    let family = match input.family.as_str() {
        "sdxl1" => ModelFamily::Sdxl1,
        "dit_anima" => ModelFamily::DitAnima,
        "dit_krea2" => ModelFamily::DitKrea2,
        other => {
            return Err(ApiError::BadRequest(format!(
                "未知模型族：{other}（可选 sdxl1 / dit_anima / dit_krea2）"
            )));
        }
    };
    if input.name.trim().is_empty() {
        return Err(ApiError::BadRequest("模型名称不能为空".into()));
    }
    if !std::path::Path::new(&input.path).exists() {
        return Err(ApiError::BadRequest(format!("路径不存在：{}", input.path)));
    }
    let model = BaseModel::new(
        input.name,
        family,
        Some(input.path),
        None,
        Some(input.source.unwrap_or_else(|| "manual".into())),
    );
    let store = state.store.lock().await;
    store.insert_base_model(&model)?;
    Ok((StatusCode::CREATED, Json(model)))
}
