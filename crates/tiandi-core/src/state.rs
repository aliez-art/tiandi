//! 任务状态机：炼丹任务的生命周期。
//!
//! 迁移合法性见 [`RunState::can_transition_to`]；状态迁移由 `tiandi-core` 单点驱动
//! （单一事实来源），引擎侧事件经事件总线回流（PRD §4、docs/architecture.md §4）。

use serde::{Deserialize, Serialize};

/// 炼丹任务状态。
///
/// 炉火意象（PRD §9）：`Running`=武火，`Paused`=文火，`Done`=出炉，`Failed`=炸炉。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    /// 已创建（任务卡落库，尚未入队）
    Created,
    /// 排队中
    Queued,
    /// 准备中（环境体检、数据集校验、缓存检查）
    Preparing,
    /// 炼丹中（武火）
    Running,
    /// 暂停（文火）
    Paused,
    /// 采样中（周期出图，不阻断训练）
    Sampling,
    /// 保存 checkpoint
    Saving,
    /// 出炉（成功）
    Done,
    /// 炸炉（失败，可重试/续丹）
    Failed,
    /// 已取消
    Canceled,
}

impl RunState {
    /// 中文标签（UI 展示用，炉火意象）。
    /// 序列化名（snake_case，与 serde 一致；数据库存储与查询用）。
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Queued => "queued",
            Self::Preparing => "preparing",
            Self::Running => "running",
            Self::Paused => "paused",
            Self::Sampling => "sampling",
            Self::Saving => "saving",
            Self::Done => "done",
            Self::Failed => "failed",
            Self::Canceled => "canceled",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Created => "已创建",
            Self::Queued => "排队中",
            Self::Preparing => "准备中",
            Self::Running => "炼丹中",
            Self::Paused => "文火",
            Self::Sampling => "采样中",
            Self::Saving => "保存中",
            Self::Done => "出炉",
            Self::Failed => "炸炉",
            Self::Canceled => "已取消",
        }
    }

    /// 从 `self` 能否合法迁移到 `to`。
    pub fn can_transition_to(&self, to: RunState) -> bool {
        self.legal_transitions().contains(&to)
    }

    /// 从 `self` 可合法迁移到的全部状态。
    pub fn legal_transitions(&self) -> Vec<RunState> {
        use RunState::*;
        match self {
            Created => vec![Queued, Failed, Canceled],
            Queued => vec![Preparing, Canceled],
            Preparing => vec![Running, Failed, Canceled],
            Running => vec![Paused, Sampling, Saving, Done, Failed, Canceled],
            Paused => vec![Running, Canceled],
            Sampling => vec![Running, Saving, Done, Failed, Canceled],
            Saving => vec![Running, Done, Failed, Canceled],
            Done | Failed | Canceled => vec![],
        }
    }

    /// 是否终态（不可再迁移）。
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done | Self::Failed | Self::Canceled)
    }
}

impl std::fmt::Display for RunState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// 非法状态迁移错误。
#[derive(Debug, thiserror::Error)]
#[error("非法状态迁移：{from:?} → {to:?}（{from} 不可迁移到 {to}）")]
pub struct RunStateError {
    pub from: RunState,
    pub to: RunState,
}

impl RunStateError {
    pub fn new(from: RunState, to: RunState) -> Self {
        Self { from, to }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_path_created_to_done() {
        let chain = [
            RunState::Created,
            RunState::Queued,
            RunState::Preparing,
            RunState::Running,
            RunState::Sampling,
            RunState::Saving,
            RunState::Done,
        ];
        for pair in chain.windows(2) {
            assert!(
                pair[0].can_transition_to(pair[1]),
                "{:?} -> {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn pause_and_resume() {
        assert!(RunState::Running.can_transition_to(RunState::Paused));
        assert!(RunState::Paused.can_transition_to(RunState::Running));
        assert!(!RunState::Paused.can_transition_to(RunState::Done));
    }

    #[test]
    fn failed_is_restartable_but_terminal() {
        assert!(RunState::Failed.is_terminal());
        // 重试走"新建排队"路径（run 复用为续丹时由用例层创建新 run），终态不可直接迁移
        assert!(!RunState::Failed.can_transition_to(RunState::Queued));
    }

    #[test]
    fn terminal_states_accept_nothing() {
        for terminal in [RunState::Done, RunState::Failed, RunState::Canceled] {
            for other in [
                RunState::Created,
                RunState::Queued,
                RunState::Preparing,
                RunState::Running,
                RunState::Paused,
                RunState::Sampling,
                RunState::Saving,
                RunState::Done,
                RunState::Failed,
                RunState::Canceled,
            ] {
                assert!(!terminal.can_transition_to(other));
            }
        }
    }

    #[test]
    fn labels_are_non_empty() {
        for s in [
            RunState::Created,
            RunState::Queued,
            RunState::Preparing,
            RunState::Running,
            RunState::Paused,
            RunState::Sampling,
            RunState::Saving,
            RunState::Done,
            RunState::Failed,
            RunState::Canceled,
        ] {
            assert!(!s.label().is_empty());
        }
    }

    #[test]
    fn serde_roundtrip_snake_case() {
        let json = serde_json::to_string(&RunState::Running).unwrap();
        assert_eq!(json, "\"running\"");
        assert_eq!(
            serde_json::from_str::<RunState>(&json).unwrap(),
            RunState::Running
        );
    }
}
