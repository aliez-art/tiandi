//! 运行清单（run manifest）：任务目录内 `manifest.json`，崩溃恢复的事实来源
//! （docs/architecture.md §3、§4）。

use std::path::Path;

use serde::{Deserialize, Serialize};
use tiandi_core::{Run, RunState};

/// 任务运行清单。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub run_id: String,
    pub project_id: Option<String>,
    pub dataset_id: Option<String>,
    pub recipe_id: Option<String>,
    pub state: RunState,
    pub created_at: String,
    pub updated_at: String,
    /// 参数快照（丹方展开后的完整参数树；M1 起由 tiandi-recipe 产出）
    pub params: serde_json::Value,
}

impl RunManifest {
    pub fn from_run(run: &Run, params: serde_json::Value) -> Self {
        Self {
            run_id: run.id.clone(),
            project_id: run.project_id.clone(),
            dataset_id: run.dataset_id.clone(),
            recipe_id: run.recipe_id.clone(),
            state: run.state,
            created_at: run.created_at.clone(),
            updated_at: run.updated_at.clone(),
            params,
        }
    }

    /// 原子写入（先写临时文件再 rename）。
    pub fn write_to(&self, path: &Path) -> Result<(), ManifestError> {
        let tmp = path.with_extension("json.tmp");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn read_from(path: &Path) -> Result<Self, ManifestError> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("清单 IO 错误: {0}")]
    Io(#[from] std::io::Error),
    #[error("清单 JSON 错误: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use tiandi_core::Run;

    #[test]
    fn manifest_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let run = Run::new(
            Some("p1".into()),
            Some("d1".into()),
            Some("r1".into()),
            None,
        );
        let m = RunManifest::from_run(&run, serde_json::json!({"lr": 1e-4}));
        let path = tmp.path().join("manifest.json");
        m.write_to(&path).unwrap();

        let back = RunManifest::read_from(&path).unwrap();
        assert_eq!(back.run_id, run.id);
        assert_eq!(back.state, RunState::Created);
        assert_eq!(back.params["lr"], serde_json::json!(1e-4));
    }

    #[test]
    fn manifest_write_is_atomic_no_tmp_leftover() {
        let tmp = tempfile::tempdir().unwrap();
        let run = Run::new(None, None, None, None);
        let m = RunManifest::from_run(&run, serde_json::Value::Null);
        let path = tmp.path().join("manifest.json");
        m.write_to(&path).unwrap();
        assert!(path.exists());
        assert!(!path.with_extension("json.tmp").exists());
    }

    #[test]
    fn read_missing_manifest_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let err = RunManifest::read_from(&tmp.path().join("nope.json")).unwrap_err();
        assert!(matches!(err, ManifestError::Io(_)));
    }
}
