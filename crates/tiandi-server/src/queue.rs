//! 任务队列调度器（PRD FR-701）：串行拉起 Queued 任务。
//!
//! 规则：同一时刻最多一个训练任务（GPU 单卡）；Queued 且无运行中任务 →
//! 组装 TrainJob 并拉起内核。崩溃恢复在服务启动时执行。

use std::time::Duration;

use tiandi_core::{Event, RunState};
use tiandi_engine::Trainer as _;
use tracing::{info, warn};

use crate::AppState;

/// 启动调度器（随服务生命周期运行；每 2s 轮询）。
pub fn spawn(state: AppState) {
    tokio::spawn(async move {
        // 崩溃恢复（仅启动时一次）：Preparing/Running 的中断任务 → Failed(可重试)
        recover_interrupted(&state).await;
        loop {
            try_pump(&state).await;
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    });
}

/// 串行泵：无运行中任务时，原子认领最早的 Queued 任务并拉起内核。
async fn try_pump(state: &AppState) {
    // 事务内原子认领：无运行中任务 → 最早的 Queued 置 Preparing 并返回该 Run；
    // 有运行中任务或无排队任务 → Ok(None)。取代"先查 has_running_run + list_queued_runs
    // 再 start"的两步逻辑，杜绝并发认领竞态（claim 返回的 Run 状态已是 Preparing，
    // supervisor 收到 hello 后再置 Preparing 是幂等 no-op）。
    let next = {
        let mut store = state.store.lock().await;
        match store.claim_next_queued() {
            Ok(r) => r,
            Err(e) => {
                warn!("队列认领失败：{e}");
                return;
            }
        }
    };
    let Some(run) = next else { return };

    // 组装训练任务（无 recipe → mock 内核）
    let job = match build_job(state, &run).await {
        Ok(j) => j,
        Err(e) => {
            fail_run(state, &run.id, &format!("任务组装失败：{e}")).await;
            return;
        }
    };

    info!(
        "队列调度：拉起任务 {}（{}）",
        &run.id[..8],
        run.state.label()
    );
    if let Err(e) = state.trainer.start(job) {
        fail_run(state, &run.id, &format!("内核启动失败：{e}")).await;
    }
}

/// 组装 TrainJob（复用 start API 的逻辑）。
pub async fn build_job(
    state: &AppState,
    run: &tiandi_core::Run,
) -> Result<tiandi_engine::TrainJob, String> {
    let (recipe_id, dataset_id, base_model_id) = (
        run.recipe_id.clone(),
        run.dataset_id.clone(),
        run.base_model_id.clone(),
    );

    let mut params = serde_json::Value::Null;
    let mut family = tiandi_core::ModelFamily::Sdxl1;
    let mut base_model = None;
    let mut dataset_dir = String::new();
    let mut output_name = None;

    if let Some(rid) = &recipe_id {
        let store = state.store.lock().await;
        let recipe = store.get_recipe(rid).map_err(|e| e.to_string())?;
        params = recipe.data;
        family = recipe.family;
        output_name = Some(recipe.name);
    } else {
        // 无丹方 → mock 内核（协议联调/UI 演示）
    }
    if let Some(did) = &dataset_id {
        let store = state.store.lock().await;
        let ds = store.get_dataset(did).map_err(|e| e.to_string())?;
        dataset_dir = ds.dir;
    }
    // 适配"直接含图目录"：sd-scripts 的 train_data_dir 要求"父目录包含图片子文件夹"
    // （DreamBooth 布局）。若用户选择的目录直接包含图片（无含图子文件夹），
    // 生成训练镜像 `<runs>/<run_id>/dataset/<N>_data/`（硬链接原图与同名 .txt），
    // 让两种目录结构都能训练（图片仍在用户目录，不移动/不复制大文件）。
    // 训练次数 repeats 取**文件夹名前缀数字**（如 2_artstyle → 2），与
    // sd-scripts 的 `N_` 约定一致；无前缀默认 1。
    let runs_dir = state.trainer.runs_dir();
    let repeats = repeats_from_dir(&dataset_dir);
    dataset_dir = prepare_dataset_dir(&dataset_dir, &run.id, runs_dir, repeats);
    if let Some(mid) = &base_model_id {
        let store = state.store.lock().await;
        let m = store.get_base_model(mid).map_err(|e| e.to_string())?;
        base_model = m.path;
    } else {
        // 兜底：第一个注册模型
        let store = state.store.lock().await;
        if let Ok(list) = store.list_base_models() {
            if let Some(m) = list.first() {
                base_model = m.path.clone();
            }
        }
    }

    let runs_dir = state.trainer.runs_dir();
    // 产物目录：<workspace>/output/<run_id>（示例图与每轮 LoRA 集中存放）
    let output_dir = crate::output_root(runs_dir).join(&run.id);
    // 断点续训：checkpoints 下最新 *.state 目录（sd-scripts save_state 产物）
    let resume_dir = detect_resume_dir(&output_dir);
    // 设置注入：镜像源等（HF_ENDPOINT 等环境变量传给内核）
    let mut env: Vec<(String, String)> = Vec::new();
    {
        let store = state.store.lock().await;
        let settings = store.list_settings().unwrap_or_default();
        for (k, v) in settings {
            if k == "hf_endpoint" {
                env.push(("HF_ENDPOINT".into(), v));
            } else if k == "pip_index" {
                env.push(("PIP_INDEX_URL".into(), v));
            }
        }
    }
    Ok(tiandi_engine::TrainJob {
        run_id: run.id.clone(),
        recipe_path: recipe_id.unwrap_or_default(),
        dataset_dir,
        output_dir: output_dir.to_string_lossy().into_owned(),
        params,
        family,
        base_model,
        output_name,
        repeats,
        resume_dir,
        env,
    })
}

/// 训练次数：取文件夹名前缀数字（kohya `N_` 约定，如 `2_artstyle` → 2，
/// `10_cat` → 10）；无数字前缀默认 1。仅对"直接含图目录"生效——
/// 子文件夹结构下由 sd-scripts 按各子文件夹名自行解析，此处返回 1。
fn repeats_from_dir(dataset_dir: &str) -> u64 {
    std::path::Path::new(dataset_dir)
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.split(['_', '-', ' ']).next())
        .and_then(|n| n.parse::<u64>().ok())
        .unwrap_or(1)
        .max(1)
}

/// 探测最新 sd-scripts state 目录（`<name>-<step>.state`）。
/// 时间戳取目录内最新文件 mtime（Windows 目录 mtime 更新不可靠）。
fn detect_resume_dir(run_dir: &std::path::Path) -> Option<String> {
    let ckpts = run_dir.join("checkpoints");
    let entries = std::fs::read_dir(&ckpts).ok()?;
    let mut best: Option<(std::time::SystemTime, String)> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if !path.is_dir() || !name.ends_with(".state") {
            continue;
        }
        if let Some(mtime) = dir_latest_mtime(&path) {
            if best.as_ref().is_none_or(|(t, _)| mtime > *t) {
                best = Some((mtime, path.to_string_lossy().into_owned()));
            }
        }
    }
    best.map(|(_, p)| p)
}

/// 目录内最新文件 mtime（递归）。
fn dir_latest_mtime(dir: &std::path::Path) -> Option<std::time::SystemTime> {
    let mut latest: Option<std::time::SystemTime> = None;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let meta = entry.metadata().ok()?;
        let t = if meta.is_dir() {
            dir_latest_mtime(&entry.path())?
        } else {
            meta.modified().ok()?
        };
        if latest.is_none_or(|l| t > l) {
            latest = Some(t);
        }
    }
    latest
}

/// 任务失败落库 + 事件。
pub async fn fail_run(state: &AppState, run_id: &str, reason: &str) {
    let store = state.store.lock().await;
    let run = match store.get_run(run_id) {
        Ok(r) => r,
        Err(_) => return,
    };
    if run.state.is_terminal() {
        return;
    }
    let updated_at = chrono::Utc::now().to_rfc3339();
    if store
        .update_run_state(run_id, RunState::Failed, &updated_at)
        .is_ok()
    {
        state.bus.emit(Event::RunStateChanged {
            run_id: run_id.to_string(),
            from: run.state,
            to: RunState::Failed,
        });
        state.bus.emit(Event::Fail {
            run_id: run_id.to_string(),
            code: 1,
            tail: reason.to_string(),
        });
        warn!("任务 {run_id} 入队失败：{reason}");
    }
}

/// 崩溃恢复：启动时将中断态（Preparing/Running/Paused/Sampling/Saving）任务置 Failed（可一键重试）。
async fn recover_interrupted(state: &AppState) {
    let store = state.store.lock().await;
    let runs = match store.list_runs() {
        Ok(r) => r,
        Err(_) => return,
    };
    for run in runs {
        if matches!(
            run.state,
            RunState::Preparing
                | RunState::Running
                | RunState::Paused
                | RunState::Sampling
                | RunState::Saving
        ) {
            let updated_at = chrono::Utc::now().to_rfc3339();
            if store
                .update_run_state(&run.id, RunState::Failed, &updated_at)
                .is_ok()
            {
                info!(
                    "崩溃恢复：任务 {} 在 {} 状态中断，已标记为炸炉（可重试）",
                    &run.id[..8],
                    run.state.label()
                );
            }
        }
    }
}

/// sd-scripts 的 `train_data_dir` 要求"父目录包含图片子文件夹"（DreamBooth 布局），
/// 且子文件夹名必须带 `N_` 重复前缀（无前缀的子文件夹会被 sd-scripts 直接忽略）。
///
/// 用户选择的目录不满足该约定时，统一生成训练镜像：
/// `<runs>/<run_id>/dataset/<N>_<名>/`（硬链接图片与同名 .txt；跨卷回退复制）：
/// - **直接含图**的目录 → 镜像子文件夹 `<N>_data`（`N` = 目录名前缀数字，默认 1）
/// - **含图但无 `N_` 前缀**的子文件夹 → 自动补前缀 `1_<原名>`（如 `tag` → `1_tag`）
/// - 已有 `N_` 前缀的子文件夹（如 `10_cat`）→ 原样镜像，重复次数不变
///
/// 返回镜像根（即 train_data_dir 指向的父目录）；目录不可用或无图时原样返回。
fn prepare_dataset_dir(
    dataset_dir: &str,
    run_id: &str,
    runs_dir: &std::path::Path,
    repeats: u64,
) -> String {
    const IMAGE_EXTS: [&str; 5] = [".jpg", ".jpeg", ".png", ".webp", ".bmp"];
    let src = std::path::Path::new(dataset_dir);
    if !src.is_dir() {
        return dataset_dir.to_string(); // 目录缺失：交给内核报错，语义清晰
    }

    let is_image = |name: &std::ffi::OsStr| -> bool {
        let lower = name.to_string_lossy().to_lowercase();
        IMAGE_EXTS.iter().any(|e| lower.ends_with(e))
    };
    let dir_contains_images = |d: &std::path::Path| -> bool {
        std::fs::read_dir(d)
            .map(|it| {
                it.flatten()
                    .any(|e| e.path().is_file() && is_image(&e.file_name()))
            })
            .unwrap_or(false)
    };
    // 本目录是否直接含图；含图子文件夹列表（保序、去重）
    let entries = std::fs::read_dir(src)
        .map(|it| {
            it.flatten()
                .filter_map(|e| e.file_type().ok().map(|t| (e.path(), t)))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let direct_images = dir_contains_images(src);
    let nested: Vec<std::path::PathBuf> = entries
        .iter()
        .filter(|(p, t)| t.is_dir() && dir_contains_images(p))
        .map(|(p, _)| p.clone())
        .collect();
    if !direct_images && nested.is_empty() {
        return dataset_dir.to_string(); // 无图：交给内核报错，语义清晰
    }

    // 生成镜像结构（train_data_dir = 镜像根，子文件夹带 N_ 前缀）
    let target_root = runs_dir.join(run_id).join("dataset");
    if direct_images {
        let group_name = format!("{repeats}_data");
        link_image_files(src, &target_root.join(&group_name), &is_image);
    }
    for sub in nested {
        let name = sub.file_name().and_then(|n| n.to_str()).unwrap_or("data");
        let group_name = normalize_group_name(name);
        link_image_files(&sub, &target_root.join(&group_name), &is_image);
    }
    target_root.to_string_lossy().into_owned()
}

/// 把目录内的图片与同名 .txt（caption）硬链接（跨卷回退复制）到目标子文件夹；
/// 幂等：目标已有同名文件则跳过。
fn link_image_files(
    src_dir: &std::path::Path,
    target_group: &std::path::Path,
    is_image: &dyn Fn(&std::ffi::OsStr) -> bool,
) {
    if target_group.join("01.png").exists() || target_group.join("01.jpg").exists() {
        return; // 已生成过（幂等）
    }
    let _ = std::fs::create_dir_all(target_group);
    let entries = std::fs::read_dir(src_dir)
        .map(|it| it.flatten().collect::<Vec<_>>())
        .unwrap_or_default();
    for entry in entries {
        let path = entry.path();
        let name = entry.file_name();
        if !path.is_file() {
            continue;
        }
        // 图片 + 同名 .txt（caption）一起纳入；其余文件忽略
        let is_txt_of_image = {
            let s = name.to_string_lossy();
            let stem = path.file_stem().and_then(|x| x.to_str()).unwrap_or("");
            s.ends_with(".txt")
                && std::fs::read_dir(src_dir)
                    .map(|it| {
                        it.flatten().any(|e| {
                            e.path().is_file()
                                && e.path().file_stem().and_then(|x| x.to_str()) == Some(stem)
                                && is_image(&e.file_name())
                        })
                    })
                    .unwrap_or(false)
        };
        if !(is_image(&name) || is_txt_of_image) {
            continue;
        }
        let dst = target_group.join(&name);
        if dst.exists() {
            continue;
        }
        if std::fs::hard_link(&path, &dst).is_err() {
            let _ = std::fs::copy(&path, &dst); // 跨卷回退复制
        }
    }
}

/// 子文件夹名规范化：已有 `N_` 前缀（数字开头 + `_`/`-` 分隔）原样保留；
/// 无前缀时补 `1_`（sd-scripts 忽略无 repeats 前缀的子文件夹）。
fn normalize_group_name(name: &str) -> String {
    let first = name.split(['_', '-', ' ']).next().unwrap_or("");
    if first.parse::<u64>().is_ok() {
        name.to_string() // 已带 N_ 前缀（或纯数字名，kohya 可解析）
    } else {
        format!("1_{name}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tiandi_core::{EventBus, Run};
    use tiandi_state::Store;

    /// 测试状态：wrapper 指向不存在的脚本（内核启动必然快速失败/退出，
    /// 避免真实 mock 内核在单测中长时间运行）；不 spawn supervisor。
    fn test_state(runs_dir: &std::path::Path) -> AppState {
        let store = Store::open_in_memory().unwrap();
        AppState::new(
            store,
            EventBus::default(),
            runs_dir.to_path_buf(),
            std::env::temp_dir().join("no-such-kernel_runner.py"),
            true,
        )
    }

    async fn insert_queued(state: &AppState, n: usize) -> Vec<Run> {
        let mut runs = Vec::new();
        let s = state.store.lock().await;
        for _ in 0..n {
            let r = Run::new(None, None, None, None);
            s.insert_run(&r).unwrap();
            s.update_run_state(&r.id, RunState::Queued, "t").unwrap();
            runs.push(r);
        }
        runs
    }

    #[tokio::test]
    async fn try_pump_blocks_when_another_run_in_progress() {
        let tmp = tempfile::tempdir().unwrap();
        let st = test_state(tmp.path());
        let running = Run::new(None, None, None, None);
        let queued = Run::new(None, None, None, None);
        {
            let s = st.store.lock().await;
            s.insert_run(&running).unwrap();
            s.insert_run(&queued).unwrap();
            s.update_run_state(&running.id, RunState::Running, "t")
                .unwrap();
            s.update_run_state(&queued.id, RunState::Queued, "t")
                .unwrap();
        }
        // 有运行中任务 → 不得认领排队任务（原子守卫）
        try_pump(&st).await;
        let s = st.store.lock().await;
        assert_eq!(s.get_run(&queued.id).unwrap().state, RunState::Queued);
        assert_eq!(s.get_run(&running.id).unwrap().state, RunState::Running);
    }

    #[tokio::test]
    async fn try_pump_claims_earliest_queued() {
        let tmp = tempfile::tempdir().unwrap();
        let st = test_state(tmp.path());
        let runs = insert_queued(&st, 2).await;
        // 无运行中任务 → 认领最早的 Queued（置 Preparing），并尝试拉起内核
        try_pump(&st).await;
        let s = st.store.lock().await;
        let first = s.get_run(&runs[0].id).unwrap();
        // 已离开 Queued：Preparing（认领成功，内核已启动）或 Failed（本机无 Python 启动失败）
        assert!(
            matches!(
                first.state,
                RunState::Preparing | RunState::Running | RunState::Failed
            ),
            "最早任务应被认领：{}",
            first.state.label()
        );
        // 第二个任务保持 Queued（单次认领只取最早一个）
        assert_eq!(s.get_run(&runs[1].id).unwrap().state, RunState::Queued);
        drop(s);
        // 清理可能残留的内核进程
        st.trainer.kill_all();
    }

    #[tokio::test]
    async fn try_pump_noop_with_empty_queue() {
        let tmp = tempfile::tempdir().unwrap();
        let st = test_state(tmp.path());
        try_pump(&st).await; // 空队列：不应 panic / 不应认领任何任务
        let s = st.store.lock().await;
        assert!(s.list_runs().unwrap().is_empty());
    }

    #[test]
    fn resume_detects_latest_state_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run1");
        let ckpts = run_dir.join("checkpoints");
        std::fs::create_dir_all(&ckpts).unwrap();
        // 两个 state 目录 + 一个普通文件
        std::fs::create_dir(ckpts.join("lora-000001.state")).unwrap();
        std::fs::create_dir(ckpts.join("lora-000010.state")).unwrap();
        std::fs::write(ckpts.join("lora.safetensors"), b"x").unwrap();
        // 让第二个更新的 mtime 生效（mtime 精度问题：强制设置）
        let newer = ckpts.join("lora-000010.state");
        std::fs::write(newer.join("placeholder"), b"y").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));

        let found = detect_resume_dir(&run_dir).unwrap();
        assert!(
            found.ends_with("lora-000010.state"),
            "应探测到最新的 state 目录：{found}"
        );
    }

    #[test]
    fn resume_none_when_no_state() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run2");
        std::fs::create_dir_all(run_dir.join("checkpoints")).unwrap();
        assert!(detect_resume_dir(&run_dir).is_none());
    }

    #[test]
    fn prepare_dataset_dir_mirrors_flat_image_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let flat = tmp.path().join("2_风格"); // 前缀数字 2 = 训练 2 次
        std::fs::create_dir_all(&flat).unwrap();
        std::fs::write(flat.join("01.png"), b"png").unwrap();
        std::fs::write(flat.join("01.txt"), b"1girl").unwrap();
        std::fs::write(flat.join("notes.md"), b"x").unwrap(); // 非图片忽略
        let runs = tmp.path().join("runs");
        let out = super::prepare_dataset_dir(flat.to_str().unwrap(), "run-1", &runs, 2);
        // 镜像结构：train_data_dir = runs/run-1/dataset（父目录），
        // 图片在子文件夹 2_data/ 内（N_ 前缀 = 训练次数）
        assert!(std::path::Path::new(&out).join("2_data/01.png").is_file());
        assert!(std::path::Path::new(&out).join("2_data/01.txt").is_file());
        assert!(!std::path::Path::new(&out).join("2_data/notes.md").exists());
        // 幂等：再次调用不报错且内容一致
        let out2 = super::prepare_dataset_dir(flat.to_str().unwrap(), "run-1", &runs, 2);
        assert_eq!(out, out2);
    }

    #[test]
    fn prepare_dataset_dir_prefixes_nested_groups() {
        let tmp = tempfile::tempdir().unwrap();
        let nested = tmp.path().join("ds");
        // 无前缀子文件夹（tag）→ 自动补 1_；有前缀（10_cat）→ 原样
        std::fs::create_dir_all(nested.join("tag")).unwrap();
        std::fs::create_dir_all(nested.join("10_cat")).unwrap();
        std::fs::write(nested.join("tag/01.png"), b"png").unwrap();
        std::fs::write(nested.join("10_cat/a.png"), b"png").unwrap();
        let runs = tmp.path().join("runs");
        let out = super::prepare_dataset_dir(nested.to_str().unwrap(), "run-2", &runs, 1);
        let root = std::path::Path::new(&out);
        assert!(
            root.join("1_tag/01.png").is_file(),
            "无前缀子文件夹应补 1_ 前缀"
        );
        assert!(
            root.join("10_cat/a.png").is_file(),
            "有前缀子文件夹原样镜像"
        );
        // 幂等
        let out2 = super::prepare_dataset_dir(nested.to_str().unwrap(), "run-2", &runs, 1);
        assert_eq!(out, out2);
    }

    #[test]
    fn prepare_dataset_dir_missing_dir_passthrough() {
        let out = super::prepare_dataset_dir(
            "Z:\\不存在的目录",
            "run-3",
            &std::path::PathBuf::from("Z:\\x"),
            1,
        );
        assert_eq!(out, "Z:\\不存在的目录");
    }

    #[test]
    fn repeats_from_dir_name_prefix() {
        assert_eq!(super::repeats_from_dir("D:\\数据\\2_artstyle"), 2);
        assert_eq!(super::repeats_from_dir("D:\\数据\\10_cat"), 10);
        assert_eq!(super::repeats_from_dir("D:\\数据\\tag"), 1);
        assert_eq!(super::repeats_from_dir("D:\\数据\\0_abc"), 1); // 0 → 兜底 1
        assert_eq!(super::repeats_from_dir(""), 1);
    }

    #[test]
    fn normalize_group_name_rules() {
        assert_eq!(super::normalize_group_name("tag"), "1_tag");
        assert_eq!(super::normalize_group_name("10_cat"), "10_cat");
        assert_eq!(super::normalize_group_name("2-artstyle"), "2-artstyle");
        assert_eq!(super::normalize_group_name("我的图"), "1_我的图");
    }
}
