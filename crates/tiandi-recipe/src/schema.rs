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
    // ---- SDXL 族专属 ----
    /// block weights（"0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1" 25 值）
    pub block_weights: Option<String>,
    /// 训练触发词/概念名（加入 caption）
    pub trigger_word: Option<String>,
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
            block_weights: None,
            trigger_word: None,
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
}
