//! 丹方文件格式：`[meta]` + `[data]` 的 TOML（PRD FR-404，git 友好、可导出分享）。

use serde::{Deserialize, Serialize};
use tiandi_core::ModelFamily;

use crate::schema::RecipeData;

/// 丹方元数据。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RecipeMeta {
    pub name: String,
    pub family: String,
    pub version: String,
    pub tags: Vec<String>,
    pub description: String,
}

impl Default for RecipeMeta {
    fn default() -> Self {
        Self {
            name: String::new(),
            family: String::new(),
            version: "1.0".into(),
            tags: Vec::new(),
            description: String::new(),
        }
    }
}

/// 丹方文件（TOML）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipeFile {
    pub meta: RecipeMeta,
    pub data: RecipeData,
}

impl RecipeFile {
    pub fn new(name: impl Into<String>, family: ModelFamily, data: RecipeData) -> Self {
        Self {
            meta: RecipeMeta {
                name: name.into(),
                family: family_to_str(family).into(),
                ..Default::default()
            },
            data,
        }
    }

    pub fn family(&self) -> Result<ModelFamily, RecipeFileError> {
        str_to_family(&self.meta.family)
    }

    /// 序列化为 TOML 字符串。
    pub fn to_toml(&self) -> Result<String, RecipeFileError> {
        toml::to_string_pretty(self).map_err(Into::into)
    }

    /// 从 TOML 字符串解析。
    pub fn from_toml(text: &str) -> Result<Self, RecipeFileError> {
        let f: Self = toml::from_str(text)?;
        // 校验 family 字段可识别
        str_to_family(&f.meta.family)?;
        Ok(f)
    }

    pub fn write_to(&self, path: &std::path::Path) -> Result<(), RecipeFileError> {
        std::fs::write(path, self.to_toml()?)?;
        Ok(())
    }

    pub fn read_from(path: &std::path::Path) -> Result<Self, RecipeFileError> {
        Self::from_toml(&std::fs::read_to_string(path)?)
    }
}

pub(crate) fn family_to_str(f: ModelFamily) -> &'static str {
    match f {
        ModelFamily::Sdxl1 => "sdxl1",
        ModelFamily::DitAnima => "dit_anima",
        ModelFamily::DitKrea2 => "dit_krea2",
    }
}

pub(crate) fn str_to_family(s: &str) -> Result<ModelFamily, RecipeFileError> {
    match s {
        "sdxl1" => Ok(ModelFamily::Sdxl1),
        "dit_anima" => Ok(ModelFamily::DitAnima),
        "dit_krea2" => Ok(ModelFamily::DitKrea2),
        other => Err(RecipeFileError::UnknownFamily(other.to_string())),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RecipeFileError {
    #[error("TOML 解析错误: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("TOML 序列化错误: {0}")]
    TomlSer(#[from] toml::ser::Error),
    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),
    #[error("未知模型族: {0}")]
    UnknownFamily(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toml_roundtrip() {
        let f = RecipeFile::new("SDXL 入门", ModelFamily::Sdxl1, RecipeData::default());
        let text = f.to_toml().unwrap();
        let back = RecipeFile::from_toml(&text).unwrap();
        assert_eq!(back.meta.name, "SDXL 入门");
        assert_eq!(back.family().unwrap(), ModelFamily::Sdxl1);
        assert_eq!(back.data.learning_rate, RecipeData::default().learning_rate);
        assert_eq!(back.data.optimizer, RecipeData::default().optimizer);
    }

    #[test]
    fn file_roundtrip_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("danfang.toml");
        let f = RecipeFile::new("测试丹方", ModelFamily::DitAnima, RecipeData::default());
        f.write_to(&path).unwrap();
        let back = RecipeFile::read_from(&path).unwrap();
        assert_eq!(back.meta.name, "测试丹方");
        assert_eq!(back.family().unwrap(), ModelFamily::DitAnima);
    }

    #[test]
    fn unknown_family_rejected() {
        let text = r#"
[meta]
name = "bad"
family = "unknown_family"
"#;
        assert!(RecipeFile::from_toml(text).is_err());
    }
}
