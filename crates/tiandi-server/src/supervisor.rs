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
    // 规范化产物根（相对/含符号链接路径统一；越界校验依赖两侧一致）
    let output_root = std::fs::canonicalize(crate::output_root(&runs_dir))
        .unwrap_or_else(|_| crate::output_root(&runs_dir));
    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(ev) => handle_event(&store, &state.bus, &runs_dir, &output_root, ev).await,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });
}

/// 事件 → 状态机/指标/文件。
///
/// 持锁原则：锁内只做 DB 操作（状态迁移/指标/checkpoint 入库）；
/// 磁盘 IO（日志追加、占位图生成、产物扫描）一律在锁外执行，
/// 避免长持锁阻塞 API 侧对 store 的访问。
async fn handle_event(
    store: &Arc<Mutex<tiandi_state::Store>>,
    bus: &tiandi_core::EventBus,
    runs_dir: &Path,
    output_root: &Path,
    ev: Event,
) {
    let run_id = match ev_run_id(&ev) {
        Some(id) => id.to_string(),
        None => return,
    };
    // 锁内快速读取 run（事件归属校验：未知任务的事件一律忽略）
    let run = {
        let store = store.lock().await;
        match store.get_run(&run_id) {
            Ok(r) => r,
            Err(_) => return,
        }
    };
    // 规范化产物根：starts_with / strip_prefix 两侧必须一致（Windows canonicalize
    // 会带 \\?\ 前缀）；目录未落盘时沿父目录宽松解析，不可解析时退回原路径
    let output_root = canonicalize_loose(output_root).unwrap_or_else(|| output_root.to_path_buf());

    match &ev {
        Event::Hello { .. } => {
            // 内核握手：准备中（claim 已置 Preparing，此处幂等 no-op）
            let mut store = store.lock().await;
            advance(&mut store, bus, &run_id, run.state, RunState::Preparing).await;
        }
        Event::Progress { .. } => {
            // 首个进度 → 炼丹中
            if matches!(run.state, RunState::Preparing | RunState::Queued) {
                let mut store = store.lock().await;
                advance(&mut store, bus, &run_id, run.state, RunState::Running).await;
            }
        }
        Event::Metric { step, loss, lr, .. } => {
            let store = store.lock().await;
            let _ = store.insert_metric(&tiandi_core::MetricPoint {
                run_id: run_id.clone(),
                step: *step,
                loss: *loss,
                lr: *lr,
            });
        }
        Event::Sample { path, .. } => {
            // 采样图：路径越界防护（绝对路径规范化后须在 output 根内；
            // 相对路径禁止 `..` 逃逸），越界 → 跳过（abs_output_path 内已 warn）
            let Some(abs) = abs_output_path(&output_root, path) else {
                return;
            };
            // 磁盘 IO（占位图生成）锁外执行
            if !abs.exists() {
                let _ = gen_placeholder_sample(&abs, sample_step_hint(path));
            }
            let Some(rel) = rel_output_path(&output_root, &abs) else {
                warn!("采样路径越界，跳过入库：{path}");
                return;
            };
            let store = store.lock().await;
            let _ = store.insert_checkpoint(&tiandi_core::Checkpoint {
                id: uuid::Uuid::new_v4().to_string(),
                run_id: run_id.clone(),
                kind: "sample".into(),
                path: rel,
                created_at: chrono::Utc::now().to_rfc3339(),
            });
        }
        Event::Log { msg, level, .. } => {
            // 日志落盘（锁外 IO，>5MB 先旋转）
            append_training_log(runs_dir, &run_id, level, msg);
        }
        Event::Done { .. } => {
            if !run.state.is_terminal() {
                // 产物扫描（锁外 IO，返回文件列表），锁内幂等入库
                let files = scan_lora_artifacts(&output_root, &run_id);
                if !files.is_empty() {
                    let store = store.lock().await;
                    let existing: std::collections::HashSet<String> = store
                        .list_checkpoints(&run_id)
                        .unwrap_or_default()
                        .into_iter()
                        .map(|c| c.path)
                        .collect();
                    for path in files {
                        let Some(rel) = rel_output_path(&output_root, &path) else {
                            continue;
                        };
                        if existing.contains(&rel) {
                            continue;
                        }
                        let _ = store.insert_checkpoint(&tiandi_core::Checkpoint {
                            id: uuid::Uuid::new_v4().to_string(),
                            run_id: run_id.clone(),
                            kind: "lora".into(),
                            path: rel,
                            created_at: chrono::Utc::now().to_rfc3339(),
                        });
                    }
                }
                let mut store = store.lock().await;
                advance(&mut store, bus, &run_id, run.state, RunState::Done).await;
            }
        }
        Event::Fail { tail, .. } if !run.state.is_terminal() => {
            warn!("任务 {run_id} 失败：{tail}");
            let mut store = store.lock().await;
            advance(&mut store, bus, &run_id, run.state, RunState::Failed).await;
        }
        _ => {}
    }
}

/// 产物绝对路径校验（防内核给出的路径逃逸 output 根）。
///
/// - 绝对路径：规范化（文件未落盘时沿父目录宽松解析）后必须位于 output 根内，
///   返回规范化后的路径（保证与根路径两侧一致，strip_prefix 才可靠）；
/// - 相对路径：禁止父目录/根/前缀组件（`..` 出现在任意位置均拒绝），按 output 根解析；
/// - 不满足：返回 `None`（调用方跳过，越界处已 warn）。
fn abs_output_path(output_root: &Path, path: &str) -> Option<PathBuf> {
    use std::path::Component;
    let p = PathBuf::from(path);
    if p.is_absolute() {
        let canon = canonicalize_loose(&p)?;
        let root = std::fs::canonicalize(output_root).unwrap_or_else(|_| output_root.to_path_buf());
        if !canon.starts_with(&root) {
            warn!("产物路径越界，已跳过：{path}");
            return None;
        }
        Some(canon)
    } else {
        if p.components().any(|c| {
            matches!(
                c,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        }) {
            warn!("产物路径越界，已跳过：{path}");
            return None;
        }
        Some(output_root.join(p))
    }
}

/// 宽松规范化：路径存在则直接 canonicalize；否则沿父目录向上找到最近的
/// 已存在祖先并拼接剩余部分（占位图等尚未落盘的路径需要）。
fn canonicalize_loose(p: &Path) -> Option<PathBuf> {
    if let Ok(c) = std::fs::canonicalize(p) {
        return Some(c);
    }
    let mut ancestor = p;
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    loop {
        let parent = ancestor.parent()?;
        if parent == ancestor {
            return None; // 已到根仍不存在（理论上不可能：根必存在）
        }
        tail.push(ancestor.file_name()?.to_os_string());
        ancestor = parent;
        if let Ok(c) = std::fs::canonicalize(ancestor) {
            let mut out = c;
            for seg in tail.iter().rev() {
                out.push(seg);
            }
            return Some(out);
        }
    }
}

/// 相对 output 根的路径（入库/URL 用）；不在根内 → `None`（调用方跳过）。
fn rel_output_path(output_root: &Path, abs: &Path) -> Option<String> {
    abs.strip_prefix(output_root)
        .ok()
        .map(|p| p.to_string_lossy().replace('\\', "/"))
}

/// 日志落盘（runs/<id>/logs/training.log）：超过 5MB 先旋转为 training.log.1。
/// 锁外 IO：仅写文件，不触碰数据库。
fn append_training_log(runs_dir: &Path, run_id: &str, level: &str, msg: &str) {
    const MAX_LOG_BYTES: u64 = 5 * 1024 * 1024;
    let log_path = runs_dir.join(run_id).join("logs").join("training.log");
    if let Some(parent) = log_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    // 旋转：超过 5MB → training.log → training.log.1（先删旧的 .1）
    if std::fs::metadata(&log_path).is_ok_and(|m| m.len() > MAX_LOG_BYTES) {
        let rotated = log_path.with_extension("log.1");
        let _ = std::fs::remove_file(&rotated);
        let _ = std::fs::rename(&log_path, &rotated);
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

/// 扫描产物根下 runs 的 checkpoints（*.safetensors，递归；ai-toolkit
/// 后端产物在 <name>/ 子目录）。纯磁盘 IO（锁外执行），返回文件列表；
/// 入库由调用方在锁内幂等完成。产物路径相对 output 根。
fn scan_lora_artifacts(output_root: &Path, run_id: &str) -> Vec<PathBuf> {
    let root = match std::fs::canonicalize(output_root) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    let ckpts = root.join(run_id);
    let mut files: Vec<PathBuf> = Vec::new();
    let mut stack = vec![ckpts];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // 跳过样本子目录（示例图是 kind=sample 由事件入库）
                if path.file_name().and_then(|n| n.to_str()) == Some("samples") {
                    continue;
                }
                stack.push(path);
            } else if path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
            {
                files.push(path);
            }
        }
    }
    files
}

/// 推进状态机：若非法（如 Preparing→Done 中间态缺失），沿合法路径补跳。
async fn advance(
    store: &mut tokio::sync::MutexGuard<'_, tiandi_state::Store>,
    bus: &tiandi_core::EventBus,
    run_id: &str,
    from: RunState,
    to: RunState,
) {
    // 沿状态机合法边找一条 from→to 的路径（BFS；状态空间极小）
    let path = find_state_path(from, to);
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

/// BFS 找合法状态路径（from == to → 空）。
fn find_state_path(from: RunState, to: RunState) -> Vec<RunState> {
    use std::collections::{HashMap, VecDeque};
    if from == to {
        return vec![];
    }
    let mut prev: HashMap<RunState, RunState> = HashMap::new();
    let mut queue = VecDeque::from([from]);
    while let Some(cur) = queue.pop_front() {
        for next in cur.legal_transitions() {
            if prev.contains_key(&next) {
                continue;
            }
            prev.insert(next, cur);
            if next == to {
                // 回溯路径
                let mut path = vec![to];
                let mut cur = to;
                while let Some(p) = prev.get(&cur) {
                    path.push(*p);
                    cur = *p;
                    if *p == from {
                        break;
                    }
                }
                path.reverse();
                return path;
            }
            queue.push_back(next);
        }
    }
    vec![]
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
        let run = Run::new(None, None, None, None);
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
        let run = Run::new(None, None, None, None);
        {
            let s = st.store.lock().await;
            s.insert_run(&run).unwrap();
            s.update_run_state(&run.id, RunState::Running, "t").unwrap();
        }
        handle_event(
            &st.store,
            &st.bus,
            &std::env::temp_dir(),
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
        let run = Run::new(None, None, None, None);
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

    #[tokio::test]
    async fn sample_out_of_bounds_relative_path_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let runs_dir = tmp.path().join("runs");
        std::fs::create_dir_all(&runs_dir).unwrap();
        let st = state();
        let run = Run::new(None, None, None, None);
        {
            let s = st.store.lock().await;
            s.insert_run(&run).unwrap();
            s.update_run_state(&run.id, RunState::Running, "t").unwrap();
        }
        handle_event(
            &st.store,
            &st.bus,
            &runs_dir,
            &runs_dir,
            Event::Sample {
                run_id: run.id.clone(),
                path: "../escape.png".into(),
            },
        )
        .await;
        // 越界路径被跳过：不入库、不落盘
        let cps = st.store.lock().await.list_checkpoints(&run.id).unwrap();
        assert!(cps.is_empty(), "越界相对路径不应入库");
        assert!(
            !tmp.path().join("escape.png").exists(),
            "不应在根外生成文件"
        );
    }

    #[tokio::test]
    async fn sample_absolute_out_of_bounds_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let runs_dir = tmp.path().join("runs");
        std::fs::create_dir_all(&runs_dir).unwrap();
        let outside = tmp.path().join("outside.png");
        std::fs::write(&outside, b"x").unwrap();
        let st = state();
        let run = Run::new(None, None, None, None);
        {
            let s = st.store.lock().await;
            s.insert_run(&run).unwrap();
            s.update_run_state(&run.id, RunState::Running, "t").unwrap();
        }
        handle_event(
            &st.store,
            &st.bus,
            &runs_dir,
            &runs_dir,
            Event::Sample {
                run_id: run.id.clone(),
                path: outside.to_string_lossy().into_owned(),
            },
        )
        .await;
        let cps = st.store.lock().await.list_checkpoints(&run.id).unwrap();
        assert!(cps.is_empty(), "越界绝对路径不应入库");
    }

    #[tokio::test]
    async fn done_scans_lora_artifacts() {
        let tmp = tempfile::tempdir().unwrap();
        let runs_dir = tmp.path().join("runs");
        let st = state();
        let run = Run::new(None, None, None, None);
        {
            let s = st.store.lock().await;
            s.insert_run(&run).unwrap();
            s.update_run_state(&run.id, RunState::Running, "t").unwrap();
        }
        // 造产物：output_root/<id>/checkpoints/xxx.safetensors（此处 output_root = runs_dir）
        let out = runs_dir.join(&run.id).join("checkpoints");
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(out.join("lora-0001.safetensors"), b"x").unwrap();
        handle_event(
            &st.store,
            &st.bus,
            &runs_dir,
            &runs_dir,
            Event::Done {
                run_id: run.id.clone(),
                code: 0,
            },
        )
        .await;
        {
            let s = st.store.lock().await;
            assert_eq!(s.get_run(&run.id).unwrap().state, RunState::Done);
            let cps = s.list_checkpoints(&run.id).unwrap();
            assert_eq!(cps.len(), 1, "Done 应扫描并入库 LoRA 产物");
            assert_eq!(cps[0].kind, "lora");
            assert!(cps[0].path.ends_with("lora-0001.safetensors"));
        }
    }

    #[tokio::test]
    async fn log_rotates_when_over_5mb() {
        let tmp = tempfile::tempdir().unwrap();
        let runs_dir = tmp.path().join("runs");
        let st = state();
        let run = Run::new(None, None, None, None);
        {
            let s = st.store.lock().await;
            s.insert_run(&run).unwrap();
            s.update_run_state(&run.id, RunState::Running, "t").unwrap();
        }
        let log_path = runs_dir.join(&run.id).join("logs/training.log");
        std::fs::create_dir_all(log_path.parent().unwrap()).unwrap();
        // 写入超过 5MB 的旧日志
        let big = vec![b'x'; 5 * 1024 * 1024 + 1];
        std::fs::write(&log_path, &big).unwrap();
        handle_event(
            &st.store,
            &st.bus,
            &runs_dir,
            &runs_dir,
            Event::Log {
                run_id: run.id.clone(),
                level: "info".into(),
                msg: "旋转后的新日志".into(),
            },
        )
        .await;
        // 旧日志旋转为 .1，新日志写入 training.log
        assert!(
            log_path.with_extension("log.1").exists(),
            "应旋转出 training.log.1"
        );
        let content = std::fs::read_to_string(&log_path).unwrap();
        assert!(content.contains("旋转后的新日志"));
        let rotated = std::fs::read_to_string(log_path.with_extension("log.1")).unwrap();
        assert_eq!(rotated.len(), big.len(), "旧日志应完整保留在 .1");
    }
}
