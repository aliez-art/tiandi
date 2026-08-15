//! 丹方：类型化训练配置（schema）、族校验、内置预设、TOML 文件格式。
//!
//! PRD FR-401~404：丹方是一等公民（TOML 文件，可命名/继承/版本化/导出）；
//! 参数面以 kohya_ss 为全集基准，M1 先实现核心子集，后续按 compat 引擎全参数扩展。

pub mod file;
pub mod preset;
pub mod schema;
pub mod validate;

pub use file::{RecipeFile, RecipeMeta};
pub use preset::builtin_presets;
pub use schema::{NetworkType, OptimizerKind, Precision, RecipeData, SchedulerKind};
pub use validate::{validate_recipe, IssueLevel, RecipeIssue};
