//! 任务监督器：订阅 EventBus，将内核事件同步到 SQLite 状态机、指标、
//! 采样占位图与日志文件。
//!
//! 单一事实来源（docs/architecture.md §4）：只有 supervisor 与显式 API 迁移
//! 修改 run 状态；内核事件经 IPC 到达后在此落地。

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tiandi_core::{Event, RunState};
use tokio::sync::Mutex;
use tracing::warn;

use crate::AppState;

/// 启动监督器任务（随服务生命周期运行）。
pub fn spawn(state: AppState) {
    let mut rx = state.bus.subscribe();
    let store = state.store.clone();
    let runs_dir = state.trainer.runs_dir().to_path_buf();
    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(ev) => handle_event(&store, &state.bus, &runs_dir, ev).await,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });
}

/// 事件 → 状态机/指标/文件。
async fn handle_event(
    store: &Arc<Mutex<tiandi_state::Store>>,
    bus: &tiandi_core::EventBus,
    runs_dir: &Path,
    ev: Event,
) {
    let run_id = match ev_run_id(&ev) {
        Some(id) => id.to_string(),
        None => return,
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
            // 真实内核未出图前（mock 不产图），生成渐变占位图供画廊展示
            let abs = abs_sample_path(runs_dir, path);
            if let Some(abs) = &abs {
                if !abs.exists() {
                    let _ = gen_placeholder_sample(abs, sample_step_hint(path));
                }
            }
            let rel = rel_runs_path(runs_dir, &abs.unwrap_or_else(|| PathBuf::from(path)));
            let _ = store.insert_checkpoint(&tiandi_core::Checkpoint {
                id: uuid::Uuid::new_v4().to_string(),
                run_id: run_id.clone(),
                kind: "sample".into(),
                path: rel,
                created_at: chrono::Utc::now().to_rfc3339(),
            });
        }
        Event::Log { msg, level, .. } => {
            // 日志落盘（runs/<id>/logs/training.log）
            let log_path = runs_dir.join(&run_id).join("logs/training.log");
            if let Some(parent) = log_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path)
            {
                let ts = chrono::Local::now().format("%H:%M:%S");
                let _ = writeln!(f, "[{ts}] [{level}] {msg}");
            }
        }
        Event::Done { .. } => {
            if !run.state.is_terminal() {
                advance(&mut store, bus, &run_id, run.state, RunState::Done).await;
            }
        }
        Event::Fail { tail, .. }
            if !run.state.is_terminal() => {
                warn!("任务 {run_id} 失败：{tail}");
                advance(&mut store, bus, &run_id, run.state, RunState::Failed).await;
            }
        _ => {}
    }
}

/// 采样图绝对路径（内核可能给绝对或相对 runs 根路径）。
fn abs_sample_path(runs_dir: &Path, path: &str) -> Option<PathBuf> {
    let p = PathBuf::from(path);
    if p.is_absolute() {
        Some(p)
    } else {
        Some(runs_dir.join(p))
    }
}

/// 相对 runs 根的路径（入库/URL 用）。
fn rel_runs_path(runs_dir: &Path, abs: &Path) -> String {
    abs.strip_prefix(runs_dir)
        .unwrap_or(abs)
        .to_string_lossy()
        .replace('\\', "/")
}

/// 从采样文件名提取 step 序号（"mock-step-0010.png" → 10）。
fn sample_step_hint(path: &str) -> u32 {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.rsplit('-').next())
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0)
}

/// 生成渐变占位采样图（512×512 PNG，色相随 step 变化）。
fn gen_placeholder_sample(path: &Path, step: u32) -> std::io::Result<()> {
    use image::RgbImage;
    const SIZE: u32 = 512;
    let hue = (step as f32 * 0.05).fract();
    let mut img = RgbImage::new(SIZE, SIZE);
    for (x, _y, p) in img.enumerate_pixels_mut() {
        let t = x as f32 / SIZE as f32;
        // 简单 HSL → RGB（饱和度 0.7，亮度 0.5）
        let r = hue_to_rgb(hue + 1.0 / 3.0, t);
        let g = hue_to_rgb(hue, t);
        let b = hue_to_rgb(hue - 1.0 / 3.0, t);
        *p = image::Rgb([r, g, b]);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    img.save(path).map_err(std::io::Error::other)
}

fn hue_to_rgb(h: f32, t: f32) -> u8 {
    let h = h.rem_euclid(1.0);
    let v = 0.5 + 0.35 * t; // 水平渐变亮度
    let c = 0.7 * v;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let r = match (h * 6.0) as u32 {
        0 => c,
        1 => x,
        2 => 0.0,
        3 => 0.0,
        4 => x,
        _ => c,
    };
    let m = v - c;
    ((r + m) * 255.0) as u8
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
            &std::env::temp_dir(),
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
            &std::env::temp_dir(),
            Event::Progress {
                run_id: run.id.clone(),
                step: 1,
                total: Some(10),
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
            &std::env::temp_dir(),
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
            &std::env::temp_dir(),
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

    #[tokio::test]
    async fn sample_event_creates_placeholder_and_log_writes_file() {
        let tmp = tempfile::tempdir().unwrap();
        let runs_dir = tmp.path().join("runs");
        let st = state();
        let run = Run::new(None, None, None);
        {
            let s = st.store.lock().await;
            s.insert_run(&run).unwrap();
            s.update_run_state(&run.id, RunState::Running, "t").unwrap();
        }
        let sample_abs = runs_dir.join(&run.id).join("samples/mock-step-0010.png");
        handle_event(
            &st.store,
            &st.bus,
            &runs_dir,
            Event::Sample {
                run_id: run.id.clone(),
                path: sample_abs.to_string_lossy().into_owned(),
            },
        )
        .await;
        // 占位图已生成
        assert!(sample_abs.exists(), "占位采样图应已生成");

        // 日志落盘
        handle_event(
            &st.store,
            &st.bus,
            &runs_dir,
            Event::Log {
                run_id: run.id.clone(),
                level: "info".into(),
                msg: "测试日志".into(),
            },
        )
        .await;
        let log_path = runs_dir.join(&run.id).join("logs/training.log");
        let content = std::fs::read_to_string(&log_path).unwrap();
        assert!(content.contains("测试日志"), "日志应落盘：{content}");

        // checkpoint 入库（相对路径）
        let cps = st.store.lock().await.list_checkpoints(&run.id).unwrap();
        assert_eq!(cps.len(), 1);
        assert!(cps[0].path.ends_with("samples/mock-step-0010.png"));
        assert!(
            !cps[0].path.contains('\\'),
            "路径应为正斜杠：{}",
            cps[0].path
        );
    }
}
