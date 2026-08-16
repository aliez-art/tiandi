//! 丹方 API：库内 CRUD、校验（点火前拦截）、内置预设。

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tiandi_core::{ModelFamily, Recipe};
use tiandi_recipe::{
    builtin_presets, validate_recipe, IssueLevel, RecipeData, RecipeFile, RecipeIssue,
};

use super::ApiError;
use crate::AppState;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/recipes", get(list_recipes).post(create_recipe))
        .route("/api/recipes/presets", get(list_presets))
        .route("/api/recipes/validate", post(validate))
        .route("/api/recipes/{id}", axum::routing::delete(delete_recipe))
}

// ---------- 列表 / 预设 ----------

async fn list_recipes(State(state): State<AppState>) -> Result<Json<Vec<Recipe>>, ApiError> {
    let store = state.store.lock().await;
    Ok(Json(store.list_recipes()?))
}

/// 内置预设（不落库，UI 直接展示；套用 = 用户另存为自定义丹方）。
async fn list_presets() -> Json<Vec<PresetView>> {
    Json(builtin_presets().iter().map(PresetView::from).collect())
}

// ---------- 创建 ----------

#[derive(Deserialize)]
struct NewRecipe {
    name: String,
    family: String,
    data: RecipeData,
}

#[derive(Serialize)]
struct RecipeResponse {
    recipe: Recipe,
    issues: Vec<RecipeIssue>,
}

async fn create_recipe(
    State(state): State<AppState>,
    Json(input): Json<NewRecipe>,
) -> Result<(StatusCode, Json<RecipeResponse>), ApiError> {
    let family = parse_family(&input.family)?;

    // 点火前校验：有 Error 级问题拒绝入库
    let issues = validate_recipe(family, &input.data);
    if issues.iter().any(|i| i.level == IssueLevel::Error) {
        return Err(ApiError::BadRequest(format!(
            "丹方校验失败（{} 项错误）：{}",
            issues
                .iter()
                .filter(|i| i.level == IssueLevel::Error)
                .count(),
            issues
                .iter()
                .filter(|i| i.level == IssueLevel::Error)
                .map(|i| format!("{}: {}", i.field, i.message))
                .collect::<Vec<_>>()
                .join("；")
        )));
    }

    let data_json = serde_json::to_value(&input.data)
        .map_err(|e| ApiError::BadRequest(format!("丹方序列化失败：{e}")))?;
    let recipe = Recipe::new(input.name, family, data_json);
    let store = state.store.lock().await;
    store.insert_recipe(&recipe)?;
    Ok((StatusCode::CREATED, Json(RecipeResponse { recipe, issues })))
}

// ---------- 删除 ----------

async fn delete_recipe(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let store = state.store.lock().await;
    store.delete_recipe(&id)?;
    Ok(StatusCode::NO_CONTENT)
}

// ---------- 校验（不落库） ----------

async fn validate(
    State(_state): State<AppState>,
    Json(input): Json<NewRecipe>,
) -> Result<Json<ValidationResponse>, ApiError> {
    let family = parse_family(&input.family)?;
    let issues = validate_recipe(family, &input.data);
    let ok = !issues.iter().any(|i| i.level == IssueLevel::Error);
    Ok(Json(ValidationResponse { ok, issues }))
}

#[derive(Serialize)]
struct ValidationResponse {
    ok: bool,
    issues: Vec<RecipeIssue>,
}

// ---------- 视图转换 ----------

/// 预设视图（meta + 数据，family 解析为枚举）。
#[derive(Serialize)]
struct PresetView {
    name: String,
    family: ModelFamily,
    description: String,
    tags: Vec<String>,
    data: RecipeData,
}

impl From<&RecipeFile> for PresetView {
    fn from(f: &RecipeFile) -> Self {
        Self {
            name: f.meta.name.clone(),
            family: f.family().unwrap_or(ModelFamily::Sdxl1),
            description: f.meta.description.clone(),
            tags: f.meta.tags.clone(),
            data: f.data.clone(),
        }
    }
}

fn parse_family(s: &str) -> Result<ModelFamily, ApiError> {
    match s {
        "sdxl1" => Ok(ModelFamily::Sdxl1),
        "dit_anima" => Ok(ModelFamily::DitAnima),
        "dit_krea2" => Ok(ModelFamily::DitKrea2),
        other => Err(ApiError::BadRequest(format!(
            "未知模型族：{other}（可选 sdxl1 / dit_anima / dit_krea2）"
        ))),
    }
}
