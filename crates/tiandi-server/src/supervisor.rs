//! 任务监督器：订阅 EventBus，将内核事件同步到 SQLite 状态机与指标。
//!
//! 单一事实来源（docs/architecture.md §4）：只有 supervisor 与显式 API 迁移
//! 修改 run 状态；内核事件经 IPC 到达后在此落地。

use std::sync::Arc;

use tiandi_core::{Event, RunState};
use tokio::sync::Mutex;
use tracing::warn;

use crate::AppState;

/// 启动监督器任务（随服务生命周期运行）。
pub fn spawn(state: AppState) {
    let mut rx = state.bus.subscribe();
    let store = state.store.clone();
    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(ev) => handle_event(&store, &state.bus, ev).await,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });
}

/// 事件 → 状态机/指标。
async fn handle_event(
    store: &Arc<Mutex<tiandi_state::Store>>,
    bus: &tiandi_core::EventBus,
    ev: Event,
) {
    let run_id = match ev_run_id(&ev) {
        Some(id) => id.to_string(),
        None => return, // hello/全局事件不驱动任务
    };
    let mut store = store.lock().await;
    let run = match store.get_run(&run_id) {
        Ok(r) => r,
        Err(_) => return,
    };

    match &ev {
        Event::Hello { .. } => {
            // 内核握手：准备中
            advance(&mut store, bus, &run_id, run.state, RunState::Preparing).await;
        }
        Event::Progress { .. } => {
            // 首个进度 → 炼丹中
            if matches!(run.state, RunState::Preparing | RunState::Queued) {
                advance(&mut store, bus, &run_id, run.state, RunState::Running).await;
            }
        }
        Event::Metric { step, loss, lr, .. } => {
            let _ = store.insert_metric(&tiandi_core::MetricPoint {
                run_id: run_id.clone(),
                step: *step,
                loss: *loss,
                lr: *lr,
            });
        }
        Event::Sample { path, .. } => {
            let _ = store.insert_checkpoint(&tiandi_core::Checkpoint {
                id: uuid::Uuid::new_v4().to_string(),
                run_id: run_id.clone(),
                kind: "sample".into(),
                path: path.clone(),
                created_at: chrono::Utc::now().to_rfc3339(),
            });
        }
        Event::Done { .. } => {
            if !run.state.is_terminal() {
                advance(&mut store, bus, &run_id, run.state, RunState::Done).await;
            }
        }
        Event::Fail { tail, .. } if !run.state.is_terminal() => {
            warn!("任务 {run_id} 失败：{tail}");
            advance(&mut store, bus, &run_id, run.state, RunState::Failed).await;
        }
        _ => {}
    }
}

/// 推进状态机：若非法（如 Queued→Running 中间态缺失），沿合法路径补跳。
async fn advance(
    store: &mut tokio::sync::MutexGuard<'_, tiandi_state::Store>,
    bus: &tiandi_core::EventBus,
    run_id: &str,
    from: RunState,
    to: RunState,
) {
    // 合法路径兜底：Queued → Preparing → Running
    let path: Vec<RunState> = if from == RunState::Queued && to == RunState::Running {
        vec![RunState::Preparing, RunState::Running]
    } else if from.can_transition_to(to) {
        vec![to]
    } else {
        vec![]
    };
    for target in path {
        let current = match store.get_run(run_id) {
            Ok(r) => r.state,
            Err(_) => return,
        };
        if !current.can_transition_to(target) {
            continue;
        }
        let updated_at = chrono::Utc::now().to_rfc3339();
        if store.update_run_state(run_id, target, &updated_at).is_ok() {
            bus.emit(Event::RunStateChanged {
                run_id: run_id.to_string(),
                from: current,
                to: target,
            });
        }
    }
}

fn ev_run_id(ev: &Event) -> Option<&str> {
    match ev {
        Event::Hello { run_id, .. }
        | Event::Progress { run_id, .. }
        | Event::Log { run_id, .. }
        | Event::Sample { run_id, .. }
        | Event::Metric { run_id, .. }
        | Event::RunStateChanged { run_id, .. }
        | Event::Done { run_id, .. }
        | Event::Fail { run_id, .. } => Some(run_id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tiandi_core::{EventBus, Run};
    use tiandi_state::Store;

    fn state() -> AppState {
        let store = Store::open_in_memory().unwrap();
        let bus = EventBus::default();
        AppState::new(store, bus, std::env::temp_dir(), std::env::temp_dir(), true)
    }

    #[tokio::test]
    async fn hello_and_progress_drive_state_machine() {
        let st = state();
        let run = Run::new(None, None, None);
        {
            let s = st.store.lock().await;
            s.insert_run(&run).unwrap();
            s.update_run_state(&run.id, RunState::Queued, "t").unwrap();
        }
        // hello → Preparing
        handle_event(
            &st.store,
            &st.bus,
            Event::Hello {
                run_id: run.id.clone(),
                backend: "test".into(),
                version: "0".into(),
            },
        )
        .await;
        {
            let s = st.store.lock().await;
            assert_eq!(s.get_run(&run.id).unwrap().state, RunState::Preparing);
        }
        // progress → Running（跨过 Queued 缺失链）
        handle_event(
            &st.store,
            &st.bus,
            Event::Progress {
                run_id: run.id.clone(),
                step: 1,
                epoch: 0.0,
                loss: 1.0,
                lr: 1e-4,
                eta_s: None,
            },
        )
        .await;
        {
            let s = st.store.lock().await;
            assert_eq!(s.get_run(&run.id).unwrap().state, RunState::Running);
        }
    }

    #[tokio::test]
    async fn metric_and_done_are_persisted() {
        let st = state();
        let run = Run::new(None, None, None);
        {
            let s = st.store.lock().await;
            s.insert_run(&run).unwrap();
            s.update_run_state(&run.id, RunState::Running, "t").unwrap();
        }
        handle_event(
            &st.store,
            &st.bus,
            Event::Metric {
                run_id: run.id.clone(),
                step: 3,
                loss: Some(0.5),
                lr: Some(1e-4),
            },
        )
        .await;
        handle_event(
            &st.store,
            &st.bus,
            Event::Done {
                run_id: run.id.clone(),
                code: 0,
            },
        )
        .await;
        {
            let s = st.store.lock().await;
            assert_eq!(s.list_metrics(&run.id).unwrap().len(), 1);
            assert_eq!(s.get_run(&run.id).unwrap().state, RunState::Done);
        }
    }
}
