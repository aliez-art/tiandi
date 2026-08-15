//! 事件总线与引擎事件协议（docs/architecture.md §5）。
//!
//! 事件形态与 IPC 协议（内核 stdout JSON Lines）对齐：`type` 字段 + snake_case。

use serde::Serialize;
use tokio::sync::broadcast;

use crate::state::RunState;

/// 引擎/内核事件（经事件总线回流，再由 `tiandi-server` 转 SSE 推给 UI）。
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// 内核握手（IPC §5.2）：版本/能力校验；run_id 归属用于状态机驱动
    Hello {
        run_id: String,
        backend: String,
        version: String,
    },
    /// 训练进度
    Progress {
        run_id: String,
        step: u64,
        epoch: f32,
        loss: f64,
        lr: f64,
        eta_s: Option<u64>,
    },
    /// 日志行
    Log {
        run_id: String,
        level: String,
        msg: String,
    },
    /// 采样出图
    Sample { run_id: String, path: String },
    /// 指标点（loss/lr 曲线）
    Metric {
        run_id: String,
        step: u64,
        loss: Option<f64>,
        lr: Option<f64>,
    },
    /// 任务状态迁移
    RunStateChanged {
        run_id: String,
        from: RunState,
        to: RunState,
    },
    /// 成功结束
    Done { run_id: String, code: u32 },
    /// 失败（含日志尾部摘要）
    Fail {
        run_id: String,
        code: u32,
        tail: String,
    },
}

/// 进程内事件总线（tokio broadcast）。
#[derive(Debug, Clone)]
pub struct EventBus {
    tx: broadcast::Sender<Event>,
}

impl EventBus {
    /// `capacity`：环形缓冲事件数（UI 回放/失败摘要用，参考 lora-scripts-next 方案）。
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.tx.subscribe()
    }

    pub fn emit(&self, event: Event) {
        // 无订阅者时丢弃（broadcast send 失败仅因无接收者，非错误）
        let _ = self.tx.send(event);
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_serializes_with_type_tag() {
        let ev = Event::Progress {
            run_id: "r1".into(),
            step: 42,
            epoch: 0.5,
            loss: 0.123,
            lr: 1e-4,
            eta_s: Some(900),
        };
        let v: serde_json::Value = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["type"], "progress");
        assert_eq!(v["run_id"], "r1");
        assert_eq!(v["step"], 42);
    }

    #[tokio::test]
    async fn bus_delivers_to_subscribers() {
        let bus = EventBus::default();
        let mut rx = bus.subscribe();
        bus.emit(Event::Done {
            run_id: "r1".into(),
            code: 0,
        });
        let ev = rx.recv().await.unwrap();
        match ev {
            Event::Done { code, .. } => assert_eq!(code, 0),
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[tokio::test]
    async fn bus_emits_without_subscribers_are_dropped() {
        let bus = EventBus::default();
        bus.emit(Event::Done {
            run_id: "r1".into(),
            code: 0,
        }); // 不应 panic
    }
}
