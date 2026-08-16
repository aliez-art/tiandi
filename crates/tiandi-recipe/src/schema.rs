//! 丹方参数 schema（M1 核心子集；全参数映射随 compat 引擎扩展）。

use serde::{Deserialize, Serialize};
use tiandi_core::ModelFamily;

/// 优化器（kohya_ss 25 种中的常用子集；序列化名与 sd-scripts 参数一致）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerKind {
    #[serde(rename = "adamw")]
    AdamW,
    #[serde(rename = "adamw8bit")]
    AdamW8Bit,
    #[serde(rename = "adafactor")]
    Adafactor,
    #[serde(rename = "prodigy")]
    Prodigy,
    #[serde(rename = "lion")]
    Lion,
    #[serde(rename = "lion8bit")]
    Lion8Bit,
    #[serde(rename = "dadapt_adagrad")]
    DAdaptAdaGrad,
    #[serde(rename = "came")]
    CAME,
    #[serde(rename = "sgdnesterov")]
    SgdNesterov,
}

impl OptimizerKind {
    pub fn label(&self) -> &'static str {
        match self {
            Self::AdamW => "AdamW",
            Self::AdamW8Bit => "AdamW8bit",
            Self::Adafactor => "Adafactor",
            Self::Prodigy => "Prodigy",
            Self::Lion => "Lion",
            Self::Lion8Bit => "Lion8bit",
            Self::DAdaptAdaGrad => "DAdaptAdaGrad",
            Self::CAME => "CAME (pytorch_optimizer)",
            Self::SgdNesterov => "SGDNesterov",
        }
    }
}

/// 学习率调度器（kohya_ss 11 种常用子集；序列化名与 sd-scripts 参数一致）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchedulerKind {
    #[serde(rename = "constant")]
    Constant,
    #[serde(rename = "constant_with_warmup")]
    ConstantWithWarmup,
    #[serde(rename = "cosine")]
    Cosine,
    #[serde(rename = "cosine_with_restarts")]
    CosineWithRestarts,
    #[serde(rename = "linear")]
    Linear,
    #[serde(rename = "polynomial")]
    Polynomial,
    #[serde(rename = "inverse_sqrt")]
    InverseSqrt,
}

impl SchedulerKind {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::ConstantWithWarmup => "constant_with_warmup",
            Self::Cosine => "cosine",
            Self::CosineWithRestarts => "cosine_with_restarts",
            Self::Linear => "linear",
            Self::Polynomial => "polynomial",
            Self::InverseSqrt => "inverse_sqrt",
        }
    }
}

/// LoRA 网络结构（kohya_ss 20 种中的常用子集；序列化名与 sd-scripts 参数一致）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkType {
    #[serde(rename = "lora")]
    Lora,
    #[serde(rename = "locon")]
    Locon,
    #[serde(rename = "lokr")]
    Lokr,
    #[serde(rename = "loha")]
    LoHa,
    #[serde(rename = "ia3")]
    Ia3,
    #[serde(rename = "dora")]
    DoRa,
    #[serde(rename = "tlora")]
    Tlora,
}

impl NetworkType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Lora => "LoRA",
            Self::Locon => "LoCon",
            Self::Lokr => "LoKr",
            Self::LoHa => "LoHa",
            Self::Ia3 => "iA3",
            Self::DoRa => "DoRA",
            Self::Tlora => "T-LoRA",
        }
    }
}

/// 混合精度。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Precision {
    #[serde(rename = "fp16")]
    Fp16,
    #[serde(rename = "bf16")]
    Bf16,
    #[serde(rename = "fp32")]
    Fp32,
}

/// 预测目标（模型训练时使用的去噪目标；SDXL 族映射为 sd-scripts `v_parameterization`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionType {
    #[serde(rename = "epsilon")]
    Epsilon,
    #[serde(rename = "v")]
    V,
    #[serde(rename = "sample")]
    Sample,
}

impl PredictionType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Epsilon => "epsilon（常规）",
            Self::V => "v（速度场）",
            Self::Sample => "sample（x0）",
        }
    }
}

/// 丹方数据（M1 核心子集）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RecipeData {
    // ---- 基础 ----
    pub learning_rate: f64,
    pub text_encoder_lr: Option<f64>,
    pub unet_lr: Option<f64>,
    pub optimizer: OptimizerKind,
    pub lr_scheduler: SchedulerKind,
    /// 预热步数（占总步数比例，0~1）
    pub lr_warmup_ratio: f64,
    pub network_dim: u32,
    pub network_alpha: u32,
    pub network_type: NetworkType,
    pub max_train_epochs: u32,
    pub batch_size: u32,
    pub resolution: u32,
    pub enable_bucket: bool,
    // ---- 缓存与精度 ----
    pub cache_latents: bool,
    pub cache_text_encoder_outputs: bool,
    pub mixed_precision: Precision,
    pub gradient_checkpointing: bool,
    pub gradient_accumulation_steps: u32,
    pub max_grad_norm: f64,
    pub seed: u64,
    // ---- 质量技巧 ----
    pub min_snr_gamma: Option<f64>,
    pub noise_offset: Option<f64>,
    pub shuffle_caption: bool,
    pub keep_tokens: u32,
    pub caption_dropout_rate: Option<f64>,
    // ---- 保存与采样 ----
    pub save_every_n_epochs: u32,
    pub sample_every_n_epochs: u32,
    pub sample_prompts: Vec<String>,
    pub sample_sampler: String,
    // ---- 高级（M3：全参数扩展，参考 kohya_ss GUI）----
    /// 网络 dropout（LoRA 权重随机丢弃比例）
    pub network_dropout: Option<f64>,
    /// rank dropout（行级丢弃）
    pub rank_dropout: Option<f64>,
    /// module dropout（模块级丢弃）
    pub module_dropout: Option<f64>,
    /// 卷积网络维度（LoCon/LoHa/LoKr 用）
    pub conv_dim: Option<u32>,
    /// 卷积网络 alpha
    pub conv_alpha: Option<u32>,
    /// 数据集重复次数（每张图重复 N 次，等效放大步数）
    pub num_repeats: Option<u32>,
    /// 总步数上限（覆盖 epochs×图数×repeats 的估算）
    pub max_train_steps: Option<u32>,
    /// 每 N 步保存检查点（0 = 关闭）
    pub save_every_n_steps: Option<u32>,
    /// 保存优化器状态（断点续训更完整，文件更大）
    pub save_state: bool,
    /// 保留最近 N 个状态目录
    pub save_last_n_states: Option<u32>,
    /// zero terminal SNR（末端噪声归零，提升暗部表现）
    pub zero_terminal_snr: bool,
    /// 自适应噪声偏移（noise_offset 的进阶版）
    pub adaptive_noise_scale: Option<f64>,
    /// 多分辨率噪声迭代（提升大图结构）
    pub multires_noise_iterations: Option<u32>,
    /// 多分辨率噪声折扣
    pub multires_noise_discount: Option<f64>,
    /// 最小时间步（噪声范围裁剪，0 = 默认）
    pub min_timestep: Option<u32>,
    /// 最大时间步
    pub max_timestep: Option<u32>,
    /// CLIP 跳层（SDXL 系常用 1~2）
    pub clip_skip: Option<u32>,
    /// 最大 token 长度（75/150/225/300）
    pub max_token_length: Option<u32>,
    /// 采样步数（示例图生成用）
    pub sample_steps: Option<u32>,
    /// 采样引导强度（示例图生成用）
    pub guidance_scale: Option<f64>,
    /// 采样负向提示词
    pub negative_prompt: Option<String>,
    /// 缓存 TE 输出到磁盘（省显存，首次慢）
    pub cache_text_encoder_outputs_to_disk: bool,
    // ---- SDXL 族专属 ----
    /// block weights（"0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1" 25 值）
    pub block_weights: Option<String>,
    /// 训练触发词/概念名（加入 caption）
    pub trigger_word: Option<String>,
    /// 预测目标（None = 随模型/后端默认；v 预测模型需显式指定 "v"）
    pub prediction_type: Option<PredictionType>,
    // ---- 前端附属字段（TrainSetup.tsx 等把路径/开关存入 recipe.data）----
    // 这些字段仅被 schema 识别以保持向前兼容（旧丹方无这些键也能解析）；
    // 路径实际由 TrainJob 提供，toml_map/yaml_map 不消费它们。
    /// 基底模型路径（前端存储；服务端以 TrainJob.base_model 为准）
    pub model_path: Option<String>,
    /// VAE 路径（前端存储；服务端以 TrainJob/内核探测为准）
    pub vae_path: Option<String>,
    /// 文本编码器路径（前端存储；服务端以 TrainJob/内核探测为准）
    pub te_path: Option<String>,
    /// 数据集目录（前端存储；服务端以 TrainJob.dataset_dir 为准）
    pub dataset_dir: Option<String>,
    /// 全量微调标记（前端存储；服务端以 TrainJob.params.full_finetune 为准）
    pub full_finetune: Option<bool>,
}

impl Default for RecipeData {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            text_encoder_lr: Some(5e-5),
            unet_lr: None,
            optimizer: OptimizerKind::AdamW8Bit,
            lr_scheduler: SchedulerKind::Cosine,
            lr_warmup_ratio: 0.1,
            network_dim: 16,
            network_alpha: 16,
            network_type: NetworkType::Lora,
            max_train_epochs: 10,
            batch_size: 1,
            resolution: 1024,
            enable_bucket: true,
            cache_latents: true,
            cache_text_encoder_outputs: true,
            mixed_precision: Precision::Bf16,
            gradient_checkpointing: true,
            gradient_accumulation_steps: 1,
            max_grad_norm: 1.0,
            seed: 42,
            min_snr_gamma: Some(5.0),
            noise_offset: None,
            shuffle_caption: true,
            keep_tokens: 1,
            caption_dropout_rate: Some(0.05),
            save_every_n_epochs: 1,
            sample_every_n_epochs: 5,
            sample_prompts: vec!["1girl, masterpiece, best quality".to_string()],
            sample_sampler: "euler_a".to_string(),
            network_dropout: None,
            rank_dropout: None,
            module_dropout: None,
            conv_dim: None,
            conv_alpha: None,
            num_repeats: None,
            max_train_steps: None,
            save_every_n_steps: None,
            save_state: false,
            save_last_n_states: None,
            zero_terminal_snr: false,
            adaptive_noise_scale: None,
            multires_noise_iterations: None,
            multires_noise_discount: None,
            min_timestep: None,
            max_timestep: None,
            clip_skip: None,
            max_token_length: None,
            sample_steps: None,
            guidance_scale: None,
            negative_prompt: None,
            cache_text_encoder_outputs_to_disk: false,
            block_weights: None,
            trigger_word: None,
            prediction_type: None,
            model_path: None,
            vae_path: None,
            te_path: None,
            dataset_dir: None,
            full_finetune: None,
        }
    }
}

impl RecipeData {
    /// 是否为 SDXL 族适用（block_weights 等字段的族约束在 validate 中检查）。
    pub fn family_supported(family: ModelFamily) -> bool {
        matches!(
            family,
            ModelFamily::Sdxl1 | ModelFamily::DitAnima | ModelFamily::DitKrea2
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid_sdxl_starting_point() {
        let d = RecipeData::default();
        assert_eq!(d.resolution, 1024);
        assert_eq!(d.optimizer, OptimizerKind::AdamW8Bit);
        assert!(d.learning_rate > 0.0);
    }

    #[test]
    fn serde_roundtrip_with_defaults_fill() {
        // 缺少字段时由 serde(default) 填充
        let json = r#"{"learning_rate": 2e-4}"#;
        let d: RecipeData = serde_json::from_str(json).unwrap();
        assert_eq!(d.learning_rate, 2e-4);
        assert_eq!(d.network_dim, 16);
        assert_eq!(d.optimizer, OptimizerKind::AdamW8Bit);
    }

    #[test]
    fn enums_are_compat_named() {
        assert_eq!(
            serde_json::to_string(&OptimizerKind::AdamW8Bit).unwrap(),
            "\"adamw8bit\""
        );
        assert_eq!(
            serde_json::to_string(&SchedulerKind::CosineWithRestarts).unwrap(),
            "\"cosine_with_restarts\""
        );
        assert_eq!(
            serde_json::to_string(&NetworkType::DoRa).unwrap(),
            "\"dora\""
        );
        assert_eq!(serde_json::to_string(&Precision::Bf16).unwrap(), "\"bf16\"");
    }

    #[test]
    fn custom_ui_fields_are_optional_and_roundtrip() {
        // 旧丹方（无这些键）仍可解析，字段为 None
        let json = r#"{"learning_rate": 2e-4}"#;
        let d: RecipeData = serde_json::from_str(json).unwrap();
        assert_eq!(d.model_path, None);
        assert_eq!(d.vae_path, None);
        assert_eq!(d.te_path, None);
        assert_eq!(d.dataset_dir, None);
        assert_eq!(d.full_finetune, None);

        // 新丹方：前端存入的附属字段被识别
        let json = r#"{
            "learning_rate": 2e-4,
            "model_path": "D:/models/base.safetensors",
            "vae_path": "D:/models/vae.safetensors",
            "te_path": "D:/models/te",
            "dataset_dir": "D:/ds",
            "full_finetune": true
        }"#;
        let d: RecipeData = serde_json::from_str(json).unwrap();
        assert_eq!(d.model_path.as_deref(), Some("D:/models/base.safetensors"));
        assert_eq!(d.vae_path.as_deref(), Some("D:/models/vae.safetensors"));
        assert_eq!(d.te_path.as_deref(), Some("D:/models/te"));
        assert_eq!(d.dataset_dir.as_deref(), Some("D:/ds"));
        assert_eq!(d.full_finetune, Some(true));

        // 序列化往返保留这些字段
        let roundtrip: RecipeData =
            serde_json::from_value(serde_json::to_value(&d).unwrap()).unwrap();
        assert_eq!(roundtrip.full_finetune, Some(true));
        assert_eq!(
            roundtrip.model_path.as_deref(),
            Some("D:/models/base.safetensors")
        );
    }
}
