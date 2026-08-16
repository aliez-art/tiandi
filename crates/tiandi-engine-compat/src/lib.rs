//! Python 内核编排：sd-scripts / ai-toolkit 双后端（M1 实现 SdScripts + mock 模式）。
//!
//! 架构（docs/architecture.md §5，ADR-001）：
//! Rust 侧 ←JSON Lines(stdout)→ kernel_runner.py ←subprocess→ sd-scripts（或 mock）。
//! 事件经 [`kernel::publish_event`] 挂到 EventBus，由 server 层转 SSE 并驱动任务状态机。

pub mod kernel;
pub mod toml_map;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tiandi_core::EventBus;
use tiandi_engine::{EngineError, EngineInfo, TrainJob, Trainer};
use tiandi_recipe::RecipeData;

use crate::kernel::{
    publish_event, spawn_kernel, KernelEnv, KernelHandle, KernelLaunch, KernelMode,
};
use crate::toml_map::{build_sdscripts_toml, TrainPaths};

/// SdScripts 后端训练器：驱动 Python 内核（真实 sd-scripts 或 mock）。
pub struct SdScriptsTrainer {
    bus: EventBus,
    env: KernelEnv,
    /// 运行目录（runs/），任务子目录 = run_id
    runs_dir: PathBuf,
    /// kernel_runner.py 路径
    wrapper: PathBuf,
    /// 运行中的内核句柄（run_id → handle）
    handles: Arc<Mutex<HashMap<String, KernelHandle>>>,
}

impl SdScriptsTrainer {
    pub fn new(bus: EventBus, runs_dir: PathBuf, wrapper: PathBuf) -> Self {
        // 优先读工作区 kernel.json（venv python + sd-scripts）；回退系统 Python
        let env = KernelEnv::detect_for(runs_dir.parent());
        Self {
            bus,
            env,
            runs_dir,
            wrapper,
            handles: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// 当前是否具备运行内核的条件（Python 可用）。
    pub fn ready(&self) -> bool {
        self.env.ready()
    }

    pub fn env_message(&self) -> Option<&str> {
        self.env.message.as_deref()
    }

    /// 内核环境（python/sd-scripts；供打标等共享内核的路径使用）。
    pub fn kernel_env(&self) -> &KernelEnv {
        &self.env
    }

    /// 运行目录（runs/）。
    pub fn runs_dir(&self) -> &std::path::Path {
        &self.runs_dir
    }

    /// 本地 tokenizer 目录（<workspace>/tokenizers/{clip-l,clip-g}，存在才返回）。
    fn local_tokenizers(&self) -> (Option<String>, Option<String>) {
        let Some(sd_scripts) = &self.env.sd_scripts else {
            return (None, None);
        };
        // sd_scripts = <ws>/.kernel/sd-scripts → workspace = 上两级
        let ws = sd_scripts.parent().and_then(|p| p.parent());
        let Some(ws) = ws else { return (None, None) };
        let clip_l = ws.join("tokenizers/clip-l");
        let clip_g = ws.join("tokenizers/clip-g");
        (
            clip_l
                .is_dir()
                .then(|| clip_l.to_string_lossy().into_owned()),
            clip_g
                .is_dir()
                .then(|| clip_g.to_string_lossy().into_owned()),
        )
    }

    fn kernel_launch(&self, job: &TrainJob, mode: KernelMode) -> Result<KernelLaunch, EngineError> {
        let python =
            self.env.python.clone().ok_or_else(|| {
                EngineError::NotReady(self.env.message.clone().unwrap_or_default())
            })?;

        // 任务目录
        let run_dir = self.runs_dir.join(&job.run_id);
        let logs_dir = run_dir.join("logs");
        let samples_dir = run_dir.join("samples");
        let checkpoints_dir = run_dir.join("checkpoints");
        for d in [&run_dir, &logs_dir, &samples_dir, &checkpoints_dir] {
            std::fs::create_dir_all(d)
                .map_err(|e| EngineError::Spawn(format!("创建任务目录失败：{e}")))?;
        }

        // 任务配置：mock 写占位；sdscripts 由丹方生成（与 kohya 键名对齐）
        let config_path = run_dir.join("train_config.toml");
        if mode == KernelMode::Mock {
            std::fs::write(&config_path, "# mock 任务配置（占位）\n")
                .map_err(|e| EngineError::Spawn(format!("写 mock 配置失败：{e}")))?;
        } else {
            let recipe: RecipeData = serde_json::from_value(job.params.clone())
                .map_err(|e| EngineError::Spawn(format!("丹方参数解析失败：{e}")))?;
            // 本地 tokenizer（离线化：<workspace>/tokenizers/{clip-l,clip-g}）
            let (tokenizer, tokenizer2) = self.local_tokenizers();
            // Anima 家族：qwen3 TE / VAE 与基底模型同目录探测；T5 分词器用内核自带
            let (anima_qwen3, anima_vae, anima_t5_tokenizer) =
                if job.family == tiandi_core::ModelFamily::DitAnima {
                    let base_dir = job
                        .base_model
                        .as_ref()
                        .map(std::path::Path::new)
                        .and_then(|p| p.parent())
                        .map(|p| p.to_path_buf())
                        .unwrap_or_default();
                    let qwen3 = base_dir.join("qwen_3_06b_base.safetensors");
                    let vae = base_dir.join("qwen_image_vae.safetensors");
                    let t5 = self
                        .env
                        .sd_scripts
                        .as_ref()
                        .map(|s| s.join("configs/t5_old"))
                        .filter(|p| p.is_dir());
                    (
                        qwen3.exists().then(|| qwen3.to_string_lossy().into_owned()),
                        vae.exists().then(|| vae.to_string_lossy().into_owned()),
                        t5.map(|p| p.to_string_lossy().into_owned()),
                    )
                } else {
                    (None, None, None)
                };
            let paths = TrainPaths {
                base_model: job.base_model.clone().unwrap_or_default(),
                dataset_dir: job.dataset_dir.clone(),
                output_dir: run_dir.join("checkpoints").to_string_lossy().into_owned(),
                output_name: job
                    .output_name
                    .clone()
                    .unwrap_or_else(|| job.run_id.clone())
                    .to_string(),
                logging_dir: logs_dir.to_string_lossy().into_owned(),
                resume: job.resume_dir.clone(),
                tokenizer,
                tokenizer2,
                anima_qwen3,
                anima_vae,
                anima_t5_tokenizer,
                anima_qwen3_tokenizer: None,
            };
            let toml = build_sdscripts_toml(&recipe, job.family, &paths);
            std::fs::write(&config_path, toml)
                .map_err(|e| EngineError::Spawn(format!("写训练配置失败：{e}")))?;
        }

        // 环境变量：run_id 供事件归属；sample 目录供 mock 出图；设置注入（镜像源等）；
        // 训练脚本按模型族指定（在 sd-scripts 目录下）
        // 训练内核强制离线（HF 缓存已种入 tokenizer；模型/数据集本地化，网络抖动不阻断训练）
        let mut env = vec![
            ("TIANDI_RUN_ID".to_string(), job.run_id.clone()),
            (
                "TIANDI_SAMPLE_DIR".to_string(),
                samples_dir.to_string_lossy().into_owned(),
            ),
            ("HF_HUB_OFFLINE".to_string(), "1".into()),
            ("TRANSFORMERS_OFFLINE".to_string(), "1".into()),
        ];
        env.extend(job.env.iter().cloned());
        if mode == KernelMode::SdScripts {
            let script = match job.family {
                tiandi_core::ModelFamily::Sdxl1 => "sdxl_train_network.py",
                tiandi_core::ModelFamily::DitAnima => "anima_train_network.py",
                tiandi_core::ModelFamily::DitKrea2 => "krea2_train_network.py",
            };
            env.push(("TIANDI_TRAIN_SCRIPT".into(), script.into()));
        }
        if mode == KernelMode::Mock {
            env.push(("TIANDI_MOCK_TOTAL".into(), "60".into()));
            env.push(("TIANDI_MOCK_INTERVAL".into(), "0.15".into()));
        }

        // cwd：sd-scripts 目录（训练脚本相对解析）；未知时用任务目录
        let cwd = self.env.sd_scripts.clone().unwrap_or(run_dir.clone());

        Ok(KernelLaunch {
            python,
            wrapper: self.wrapper.clone(),
            config_path,
            mode,
            env,
            cwd,
        })
    }

    fn start_kernel(&self, job: &TrainJob, mode: KernelMode) -> Result<(), EngineError> {
        let launch = self.kernel_launch(job, mode)?;
        tracing::info!(
            "start_kernel: run={} python={} mode={}",
            job.run_id,
            launch.python.display(),
            mode.as_str()
        );
        let run_id = job.run_id.clone();
        let bus = self.bus.clone();
        let handles = self.handles.clone();

        let handle = spawn_kernel(&launch, move |value| {
            publish_event(&bus, &value, &run_id);
            // 终态事件后释放句柄
            if let Some(t) = value.get("type").and_then(|v| v.as_str()) {
                if matches!(t, "done" | "fail") {
                    handles.lock().unwrap().remove(&run_id);
                }
            }
        })
        .map_err(|e| EngineError::Spawn(e.to_string()))?;

        self.handles
            .lock()
            .unwrap()
            .insert(job.run_id.clone(), handle);
        Ok(())
    }
}

impl Trainer for SdScriptsTrainer {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            backend: "sd-scripts (kernel_runner)".into(),
            version: "0.1.0".into(),
            capabilities: vec!["ipc-jsonlines-v1".into(), "mock".into(), "sdxl-toml".into()],
        }
    }

    fn start(&self, job: TrainJob) -> Result<(), EngineError> {
        // 有丹方 → 真实 sd-scripts 模式；无丹方 → mock（协议联调/UI 演示）
        let mode = if job.params.is_null() || job.recipe_path.is_empty() {
            KernelMode::Mock
        } else {
            KernelMode::SdScripts
        };
        self.start_kernel(&job, mode)
    }

    fn pause(&self, run_id: &str) -> Result<(), EngineError> {
        let _ = run_id;
        // M2：训练侧优雅暂停（写 state + 退出）。当前不支持。
        Err(EngineError::Unsupported(
            "暂停将在 M2 落地（当前支持取消）".into(),
        ))
    }

    fn resume(&self, run_id: &str) -> Result<(), EngineError> {
        let _ = run_id;
        Err(EngineError::Unsupported(
            "恢复将在 M2 落地（当前支持取消）".into(),
        ))
    }

    fn cancel(&self, run_id: &str) -> Result<(), EngineError> {
        let mut handles = self.handles.lock().unwrap();
        if let Some(h) = handles.get_mut(run_id) {
            // Trainer trait 为同步接口；用一次性 current_thread runtime 阻塞取消
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| EngineError::Runtime(e.to_string()))?;
            rt.block_on(h.cancel())
                .map_err(|e| EngineError::Runtime(e.to_string()))?;
            handles.remove(run_id);
            Ok(())
        } else {
            Err(EngineError::UnknownRun(run_id.into()))
        }
    }

    fn query(&self, run_id: &str) -> Result<Option<tiandi_core::RunState>, EngineError> {
        // 状态由 server 层从 SQLite 查询（M1 简化：Trainer 不持有存储）
        let _ = run_id;
        Ok(None)
    }
}
