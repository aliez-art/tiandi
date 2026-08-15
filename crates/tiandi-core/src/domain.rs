//! 领域实体：项目、基底模型、数据集、丹方、炼丹任务、指标、产物。
//!
//! M0 骨架以纯数据模型为主（serde 序列化，便于 SQLite/JSON 持久化与 REST 传输）；
//! 数据集/桶等复杂行为在 M1 的 `tiandi-dataset` 中实现。

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// 模型族（PRD §6.1）：决定丹方校验、内核脚本路由与采样参数。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFamily {
    /// SDXL-1.0 基线：Illusion / NoobAI
    Sdxl1,
    /// DiT 族：Anima（CircleStone-Labs 2B 动漫 DiT）
    DitAnima,
    /// DiT 族：Krea 2（Krea2Transformer2DModel / SingleStreamDiT）
    DitKrea2,
}

impl ModelFamily {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Sdxl1 => "SDXL 1.0",
            Self::DitAnima => "Anima (DiT)",
            Self::DitKrea2 => "Krea 2 (DiT)",
        }
    }

    /// 内核后端路由（PRD §8.3）：SDXL/Anima 走 sd-scripts，Krea 2 走 ai-toolkit。
    pub fn backend(&self) -> &'static str {
        match self {
            Self::Sdxl1 | Self::DitAnima => "sd-scripts",
            Self::DitKrea2 => "ai-toolkit",
        }
    }
}

fn now() -> String {
    Utc::now().to_rfc3339()
}

fn new_id() -> String {
    Uuid::new_v4().to_string()
}

/// 工作区（数据根目录 + 默认设置）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub id: String,
    pub name: String,
    pub root_dir: String,
    pub created_at: String,
}

impl Project {
    pub fn new(name: impl Into<String>, root_dir: impl Into<String>) -> Self {
        Self {
            id: new_id(),
            name: name.into(),
            root_dir: root_dir.into(),
            created_at: now(),
        }
    }
}

/// 基底模型注册项。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseModel {
    pub id: String,
    pub name: String,
    pub family: ModelFamily,
    /// 主权重路径（safetensors / diffusers 目录）
    pub path: Option<String>,
    pub sha256: Option<String>,
    pub source: Option<String>,
    pub created_at: String,
}

impl BaseModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        family: ModelFamily,
        path: Option<String>,
        sha256: Option<String>,
        source: Option<String>,
    ) -> Self {
        Self {
            id: new_id(),
            name: name.into(),
            family,
            path,
            sha256,
            source,
            created_at: now(),
        }
    }
}

/// 数据集（图像目录集合 + 桶配置 + 标签集引用；桶细节在 M1 的 tiandi-dataset）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub id: String,
    pub name: String,
    /// 数据集根目录（含 `N_` 重复子集约定，PRD §5.2 / 附录 A.2）
    pub dir: String,
    pub image_count: u64,
    pub created_at: String,
}

impl Dataset {
    pub fn new(name: impl Into<String>, dir: impl Into<String>) -> Self {
        Self {
            id: new_id(),
            name: name.into(),
            dir: dir.into(),
            image_count: 0,
            created_at: now(),
        }
    }
}

/// 丹方：模型族校验后的训练配置（`data` 为参数树，M1 起由 `tiandi-recipe` 强类型化）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recipe {
    pub id: String,
    pub name: String,
    pub family: ModelFamily,
    pub data: serde_json::Value,
    pub created_at: String,
}

impl Recipe {
    pub fn new(name: impl Into<String>, family: ModelFamily, data: serde_json::Value) -> Self {
        Self {
            id: new_id(),
            name: name.into(),
            family,
            data,
            created_at: now(),
        }
    }
}

/// 一次炼丹任务（状态机见 [`crate::state`]）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    pub id: String,
    pub project_id: Option<String>,
    pub dataset_id: Option<String>,
    pub recipe_id: Option<String>,
    /// 基底模型（v3 起；None = 用第一个注册模型或 mock）
    pub base_model_id: Option<String>,
    pub state: crate::state::RunState,
    /// 运行清单路径（`runs/<id>/manifest.json`）
    pub manifest_path: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

impl Run {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        project_id: Option<String>,
        dataset_id: Option<String>,
        recipe_id: Option<String>,
        base_model_id: Option<String>,
    ) -> Self {
        let t = now();
        Self {
            id: new_id(),
            project_id,
            dataset_id,
            recipe_id,
            base_model_id,
            state: crate::state::RunState::Created,
            manifest_path: None,
            created_at: t.clone(),
            updated_at: t,
        }
    }
}

/// 指标点（loss/lr 曲线数据）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    pub run_id: String,
    pub step: u64,
    pub loss: Option<f64>,
    pub lr: Option<f64>,
}

/// 训练产物（LoRA / state / 采样图）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub id: String,
    pub run_id: String,
    /// 产物种类：lora / state / sample
    pub kind: String,
    pub path: String,
    pub created_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entities_serialize_with_snake_case() {
        let p = Project::new("测试项目", "D:\\tiandi-ws");
        let v = serde_json::to_value(&p).unwrap();
        assert!(v.get("root_dir").is_some());
        assert!(v.get("created_at").is_some());
    }

    #[test]
    fn family_backend_routing() {
        assert_eq!(ModelFamily::Sdxl1.backend(), "sd-scripts");
        assert_eq!(ModelFamily::DitAnima.backend(), "sd-scripts");
        assert_eq!(ModelFamily::DitKrea2.backend(), "ai-toolkit");
    }

    #[test]
    fn run_defaults_to_created() {
        let r = Run::new(None, None, None, None);
        assert_eq!(r.state, crate::state::RunState::Created);
        assert!(r.updated_at >= r.created_at || r.updated_at == r.created_at);
    }

    #[test]
    fn recipe_carries_family_and_data() {
        let r = Recipe::new(
            "SDXL 入门",
            ModelFamily::Sdxl1,
            serde_json::json!({"lr": 1e-4}),
        );
        assert_eq!(r.family, ModelFamily::Sdxl1);
        assert_eq!(r.data["lr"], serde_json::json!(1e-4));
    }
}
