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

/// 串行泵：无运行中任务时，拉起最早的 Queued 任务。
async fn try_pump(state: &AppState) {
    let next = {
        let store = state.store.lock().await;
        let running = match store.has_running_run() {
            Ok(r) => r,
            Err(e) => {
                warn!("队列检查失败：{e}");
                return;
            }
        };
        if running {
            return;
        }
        match store.list_queued_runs() {
            Ok(list) => list.into_iter().next(),
            Err(e) => {
                warn!("读取队列失败：{e}");
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
    Ok(tiandi_engine::TrainJob {
        run_id: run.id.clone(),
        recipe_path: recipe_id.unwrap_or_default(),
        dataset_dir,
        output_dir: runs_dir.join(&run.id).to_string_lossy().into_owned(),
        params,
        family,
        base_model,
        output_name,
    })
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

/// 崩溃恢复：启动时将中断态（Preparing/Running）任务置 Failed（可一键重试）。
async fn recover_interrupted(state: &AppState) {
    let store = state.store.lock().await;
    let runs = match store.list_runs() {
        Ok(r) => r,
        Err(_) => return,
    };
    for run in runs {
        if matches!(
            run.state,
            RunState::Preparing | RunState::Running | RunState::Sampling | RunState::Saving
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
