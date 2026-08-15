//! SSE 事件流：订阅进程内事件总线，按 run_id 过滤后推给 UI。
//!
//! 与 IPC 协议（内核 stdout JSON Lines）同构：`data:` 载荷为 `{"type": ...}` JSON。

use std::convert::Infallible;

use axum::{
    extract::{Path, State},
    response::sse::{Event as SseEvent, KeepAlive, Sse},
};
use futures::stream::{Stream, StreamExt};
use tiandi_core::Event;
use tokio_stream::wrappers::BroadcastStream;

use crate::AppState;

/// 取事件所属 run_id（无 run 归属的事件返回 None）。
pub fn event_run_id(ev: &Event) -> Option<&str> {
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

/// `GET /api/runs/{run_id}/events`（`run_id = "all"` 时不过滤）。
pub async fn stream_events(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    let rx = state.bus.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(move |res| {
        let ev = match res {
            Ok(ev) => ev,
            Err(_) => return futures::future::ready(None), // 滞后订阅者被挤掉：直接跳过
        };
        if run_id != "all" && event_run_id(&ev) != Some(run_id.as_str()) {
            return futures::future::ready(None);
        }
        let data = serde_json::to_string(&ev).unwrap_or_else(|_| "{}".into());
        futures::future::ready(Some(Ok(SseEvent::default().data(data))))
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}
