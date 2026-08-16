//! Python 内核编排：sd-scripts / ai-toolkit 双后端（M1 实现 SdScripts + mock 模式）。
//!
//! 架构（docs/architecture.md §5，ADR-001）：
//! Rust 侧 ←JSON Lines(stdout)→ kernel_runner.py ←subprocess→ sd-scripts（或 mock）。
//! 事件经 [`kernel::publish_event`] 挂到 EventBus，由 server 层转 SSE 并驱动任务状态机。

pub mod kernel;
pub mod toml_map;
pub mod yaml_map;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tiandi_core::EventBus;
use tiandi_engine::{EngineError, EngineInfo, TrainJob, Trainer};
use tiandi_recipe::RecipeData;

use crate::kernel::{
    publish_event, spawn_kernel, KernelEnv, KernelHandle, KernelLaunch, KernelMode,
};
use crate::toml_map::{build_sdscripts_toml, build_sdscripts_toml_full, TrainPaths};
use crate::yaml_map::{build_aitk_yaml, AitkPaths};

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

    /// 服务关停：终止所有运行中的内核进程树并清空句柄表。
    /// 供 tiandi-server 关停流程调用（进程内一致性优先，中毒锁继续用）。
    pub fn kill_all(&self) {
        let mut handles = self.handles.lock().unwrap_or_else(|e| e.into_inner());
        for (run_id, mut h) in handles.drain() {
            h.kill();
            tracing::info!("kill_all: 已终止内核 run={run_id}");
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
        // aitk 模式用 ai-toolkit 独立 venv；其余用 sd-scripts venv（或系统 Python）
        let python = if mode == KernelMode::Aitk {
            self.env.aitk.clone().map(|a| a.python).ok_or_else(|| {
                EngineError::NotReady(
                    "ai-toolkit 内核未安装（tiandi kernel install --backend aitk）".into(),
                )
            })?
        } else {
            self.env.python.clone().ok_or_else(|| {
                EngineError::NotReady(self.env.message.clone().unwrap_or_default())
            })?
        };

        // 任务目录：日志/配置留在 runs/<id>；产物（LoRA/示例图）在 job.output_dir（output/<id>）
        let run_dir = self.runs_dir.join(&job.run_id);
        let logs_dir = run_dir.join("logs");
        let samples_dir = std::path::PathBuf::from(&job.output_dir).join("samples");
        let checkpoints_dir = std::path::PathBuf::from(&job.output_dir).join("checkpoints");
        for d in [&run_dir, &logs_dir, &samples_dir, &checkpoints_dir] {
            std::fs::create_dir_all(d)
                .map_err(|e| EngineError::Spawn(format!("创建任务目录失败：{e}")))?;
        }

        // 任务配置：mock 写占位；sdscripts 由丹方生成（与 kohya 键名对齐）；
        // aitk 由丹方生成 YAML（Krea 2 等 DiT 模型）
        let config_path = run_dir.join(if mode == KernelMode::Aitk {
            "train_config.yaml"
        } else {
            "train_config.toml"
        });
        if mode == KernelMode::Mock {
            std::fs::write(&config_path, "# mock 任务配置（占位）\n")
                .map_err(|e| EngineError::Spawn(format!("写 mock 配置失败：{e}")))?;
        } else if mode == KernelMode::Aitk {
            let recipe: RecipeData = serde_json::from_value(job.params.clone())
                .map_err(|e| EngineError::Spawn(format!("丹方参数解析失败：{e}")))?;
            // 全量训练暂不支持 ai-toolkit（Krea 2）
            if job
                .params
                .get("full_finetune")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                return Err(EngineError::Spawn(
                    "Krea 2 全量训练暂不支持（ai-toolkit 后端仅 LoRA）".into(),
                ));
            }
            let aitk = self.env.aitk.clone().ok_or_else(|| {
                EngineError::NotReady(
                    "ai-toolkit 内核未安装（tiandi kernel install --backend aitk）".into(),
                )
            })?;
            let krea2 = aitk.krea2.clone().ok_or_else(|| {
                EngineError::NotReady(
                    "Krea 2 资产未就绪（tiandi kernel prepare-krea2 <底模目录>）".into(),
                )
            })?;
            // 总步数 = epochs × 图片数 × 重复次数 ÷ batch_size（ai-toolkit 按步训练，
            // 每步消费 batch_size 张图；除后兜底 ≥1 步）
            let repeats = recipe.num_repeats.unwrap_or(1).max(1) as u64;
            let batch_size = recipe.batch_size.max(1) as u64;
            let steps = ((dataset_image_count(job.dataset_dir.as_str()) as u64
                * recipe.max_train_epochs as u64
                * repeats)
                / batch_size)
                .max(1);
            let paths = AitkPaths {
                base_model: job.base_model.clone().unwrap_or_default(),
                dataset_dir: job.dataset_dir.clone(),
                training_folder: std::path::PathBuf::from(&job.output_dir)
                    .join("checkpoints")
                    .to_string_lossy()
                    .into_owned(),
                output_name: job
                    .output_name
                    .clone()
                    .unwrap_or_else(|| job.run_id.clone())
                    .to_string(),
                text_encoder: krea2.text_encoder.to_string_lossy().into_owned(),
                vae_root: krea2.vae_root.to_string_lossy().into_owned(),
                steps,
            };
            let yaml = build_aitk_yaml(&recipe, &paths);
            std::fs::write(&config_path, yaml)
                .map_err(|e| EngineError::Spawn(format!("写训练配置失败：{e}")))?;
        } else {
            let recipe: RecipeData = serde_json::from_value(job.params.clone())
                .map_err(|e| EngineError::Spawn(format!("丹方参数解析失败：{e}")))?;
            // 丹方可选指定 VAE / TE（data.vae_path / data.te_path 自定义键）
            let vae_override = job
                .params
                .get("vae_path")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string());
            let te_override = job
                .params
                .get("te_path")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string());
            // 本地 tokenizer（离线化：<workspace>/tokenizers/{clip-l,clip-g}）
            let (tokenizer, tokenizer2) = self.local_tokenizers();
            // Anima 家族：qwen3 TE / VAE 与基底模型同目录探测；T5 分词器用内核自带
            // （丹方指定 VAE/TE 时优先）
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
                        te_override.clone().or_else(|| {
                            qwen3.exists().then(|| qwen3.to_string_lossy().into_owned())
                        }),
                        vae_override
                            .clone()
                            .or_else(|| vae.exists().then(|| vae.to_string_lossy().into_owned())),
                        t5.map(|p| p.to_string_lossy().into_owned()),
                    )
                } else {
                    (None, None, None)
                };
            let paths = TrainPaths {
                base_model: job.base_model.clone().unwrap_or_default(),
                dataset_dir: job.dataset_dir.clone(),
                output_dir: std::path::PathBuf::from(&job.output_dir)
                    .join("checkpoints")
                    .to_string_lossy()
                    .into_owned(),
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
            // 全量微调（full_finetune=true）：无 LoRA 网络段，输出完整 checkpoint
            let full = job
                .params
                .get("full_finetune")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let train_te = job
                .params
                .get("train_text_encoder")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let toml = if full {
                build_sdscripts_toml_full(&recipe, job.family, &paths, train_te)
            } else {
                build_sdscripts_toml(&recipe, job.family, &paths)
            };
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
            // 全量微调 → sdxl_train.py / anima_train.py；LoRA → *_train_network.py
            let full = job
                .params
                .get("full_finetune")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let script = match (job.family, full) {
                (tiandi_core::ModelFamily::Sdxl1, true) => "sdxl_train.py",
                (tiandi_core::ModelFamily::DitAnima, true) => "anima_train.py",
                (tiandi_core::ModelFamily::Sdxl1, false) => "sdxl_train_network.py",
                (tiandi_core::ModelFamily::DitAnima, false) => "anima_train_network.py",
                (tiandi_core::ModelFamily::DitKrea2, _) => "krea2_train_network.py",
            };
            env.push(("TIANDI_TRAIN_SCRIPT".into(), script.into()));
        }
        if mode == KernelMode::Mock {
            env.push(("TIANDI_MOCK_TOTAL".into(), "60".into()));
            env.push(("TIANDI_MOCK_INTERVAL".into(), "0.15".into()));
        }

        // cwd：sd-scripts 目录（训练脚本相对解析）；aitk 用 ai-toolkit 仓库；
        // 未知时用任务目录
        let cwd = if mode == KernelMode::Aitk {
            self.env
                .aitk
                .clone()
                .ok_or_else(|| EngineError::NotReady("ai-toolkit 内核未安装".into()))?
                .repo
        } else {
            self.env.sd_scripts.clone().unwrap_or(run_dir.clone())
        };

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
            // 终态事件后释放句柄（中毒的锁继续用：进程内一致即可）
            if let Some(t) = value.get("type").and_then(|v| v.as_str()) {
                if matches!(t, "done" | "fail") {
                    handles
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .remove(&run_id);
                }
            }
        })
        .map_err(|e| EngineError::Spawn(e.to_string()))?;

        self.handles
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(job.run_id.clone(), handle);
        Ok(())
    }
}

/// 扫描数据集目录图片数（ai-toolkit 总步数换算：steps = epochs × 图片数）。
fn dataset_image_count(dataset_dir: &str) -> usize {
    const EXTS: [&str; 5] = [".jpg", ".jpeg", ".png", ".webp", ".bmp"];
    let mut count = 0;
    let mut stack = vec![PathBuf::from(dataset_dir)];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_lowercase();
            if name == "thumbs" || name == ".cache" {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
            } else if EXTS.iter().any(|e| name.ends_with(e)) {
                count += 1;
            }
        }
    }
    count.max(1)
}

/// 运行时资产路径（kernel_runner.py 等随 crate 分发的文件）。
///
/// 依次尝试：当前可执行文件所在目录 → `<exe目录>/assets` → 编译期
/// `CARGO_MANIFEST_DIR/assets`（开发/测试兜底），返回第一个存在的。
/// 分发后（cargo install / 打包）资产与可执行文件同目录部署，不再依赖源码树。
pub fn asset_path(name: &str) -> std::path::PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let direct = dir.join(name);
            if direct.exists() {
                return direct;
            }
            let in_assets = dir.join("assets").join(name);
            if in_assets.exists() {
                return in_assets;
            }
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join(name)
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
        // 无丹方 → mock（协议联调/UI 演示）；Krea 2 族 → ai-toolkit；其余 → sd-scripts
        let mode = if job.params.is_null() || job.recipe_path.is_empty() {
            KernelMode::Mock
        } else if job.family == tiandi_core::ModelFamily::DitKrea2 {
            KernelMode::Aitk
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
        let mut handles = self.handles.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = handles.get_mut(run_id) {
            // Trainer trait 为同步接口；用一次性 current_thread runtime 阻塞取消
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| EngineError::Runtime(e.to_string()))?;
            let result = rt.block_on(h.cancel());
            // 无论成功/超时都移除句柄：超时已强杀、句柄已失效，滞留会泄漏（kill_all 兜不住）
            handles.remove(run_id);
            result.map_err(|e| EngineError::Runtime(e.to_string()))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_path_resolves_wrapper_script() {
        // 测试环境（current_exe 无资产）→ 回退 CARGO_MANIFEST_DIR/assets，应存在
        let p = asset_path("kernel_runner.py");
        assert!(
            p.exists(),
            "kernel_runner.py 应可解析到实际存在的路径：{}",
            p.display()
        );
        assert_eq!(
            p.file_name().and_then(|n| n.to_str()),
            Some("kernel_runner.py")
        );
    }

    #[test]
    fn steps_estimate_divides_by_batch_size() {
        // 纯函数换算核对：epochs × 图数 × repeats ÷ batch_size，且 ≥1
        let count = 4u64; // 4 张图
        let epochs = 10u32;
        let repeats = 2u64;
        let batch_size = 4u64;
        let steps = ((count * epochs as u64 * repeats) / batch_size).max(1);
        assert_eq!(steps, 20); // 4×10×2/4
                               // batch_size 大于总量时兜底 1 步
        let steps2 = ((count * epochs as u64 * repeats) / 1000).max(1);
        assert_eq!(steps2, 1);
    }
}
