//! 训练引擎抽象：`Trainer` trait 与任务控制协议。
//!
//! 职责边界（docs/architecture.md §5，ADR-001）：Rust 侧编排，Python 内核负责计算。
//! `Trainer` 的具体实现：
//! - M1：`tiandi-engine-compat`（BackendSdScripts / BackendAiToolkit，IPC/Stdio 桥）
//! - 远期探索：`tiandi-engine-native`（candle，不排期）

use serde::Serialize;
use tiandi_core::RunState;

/// 引擎能力声明（IPC `hello` 事件语义：版本/能力协商）。
#[derive(Debug, Clone, Serialize)]
pub struct EngineInfo {
    pub backend: String,
    pub version: String,
    pub capabilities: Vec<String>,
}

/// 训练任务载荷：内核所需的全部路径与参数来源。
#[derive(Debug, Clone)]
pub struct TrainJob {
    pub run_id: String,
    /// 丹方文件路径（compat 后端据此展开为内核配置 TOML/YAML）
    pub recipe_path: String,
    /// 数据集目录（`N_` 重复子集约定）
    pub dataset_dir: String,
    /// 任务输出目录（runs/<run_id>）
    pub output_dir: String,
    pub params: serde_json::Value,
}

/// 引擎错误。
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("任务不存在: {0}")]
    UnknownRun(String),
    #[error("引擎未就绪: {0}")]
    NotReady(String),
    #[error("内核启动失败: {0}")]
    Spawn(String),
    #[error("内核运行错误: {0}")]
    Runtime(String),
    #[error("内核不支持该操作: {0}")]
    Unsupported(String),
}

/// 训练引擎抽象。
pub trait Trainer: Send + Sync {
    fn info(&self) -> EngineInfo;

    /// 启动训练（生成任务配置 + 拉起内核进程；事件经 EventBus 回流）。
    fn start(&self, job: TrainJob) -> Result<(), EngineError>;

    /// 暂停（文火：进程级挂起，P1 起支持训练侧优雅暂停）。
    fn pause(&self, run_id: &str) -> Result<(), EngineError>;

    /// 恢复（武火）。
    fn resume(&self, run_id: &str) -> Result<(), EngineError>;

    /// 取消（两段式：优雅请求 → 超时 kill-tree）。
    fn cancel(&self, run_id: &str) -> Result<(), EngineError>;

    /// 查询当前状态（UI 重连/心跳恢复用，IPC `query` 命令语义）。
    fn query(&self, run_id: &str) -> Result<Option<RunState>, EngineError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct StubTrainer;

    impl Trainer for StubTrainer {
        fn info(&self) -> EngineInfo {
            EngineInfo {
                backend: "stub".into(),
                version: "0.0.0".into(),
                capabilities: vec![],
            }
        }
        fn start(&self, _job: TrainJob) -> Result<(), EngineError> {
            Ok(())
        }
        fn pause(&self, _run_id: &str) -> Result<(), EngineError> {
            Ok(())
        }
        fn resume(&self, _run_id: &str) -> Result<(), EngineError> {
            Ok(())
        }
        fn cancel(&self, _run_id: &str) -> Result<(), EngineError> {
            Ok(())
        }
        fn query(&self, _run_id: &str) -> Result<Option<RunState>, EngineError> {
            Ok(None)
        }
    }

    #[test]
    fn trainer_trait_is_object_safe() {
        let t: Box<dyn Trainer> = Box::new(StubTrainer);
        assert_eq!(t.info().backend, "stub");
        t.pause("r1").unwrap();
        t.cancel("r1").unwrap();
    }
}
