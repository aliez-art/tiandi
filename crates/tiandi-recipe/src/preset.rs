//! 内置预设丹方（PRD FR-402）：每模型族「入门」即可运行。
//!
//! 参数取值参考 kohya_ss 默认与社区实践（NoobAI/Illustrious 系训练指南），
//! 供 UI 一键套用；用户可另存/继承/覆盖。

use tiandi_core::ModelFamily;

use crate::file::RecipeFile;
use crate::schema::{NetworkType, OptimizerKind, Precision, RecipeData, SchedulerKind};

/// 内置预设列表（含各模型族入门丹方）。
pub fn builtin_presets() -> Vec<RecipeFile> {
    vec![
        sdxl_noobai_entry(),
        sdxl_advanced(),
        anima_entry(),
        krea2_entry(),
    ]
}

/// SDXL 入门（NoobAI / Illusion 通用）。
fn sdxl_noobai_entry() -> RecipeFile {
    let data = RecipeData {
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
        sample_prompts: vec!["1girl, masterpiece, best quality, detailed".to_string()],
        sample_sampler: "euler_a".to_string(),
        block_weights: None,
        trigger_word: None,
        prediction_type: None,
        ..RecipeData::default()
    };
    RecipeFile::new("SDXL 入门（NoobAI/Illusion）", ModelFamily::Sdxl1, data)
}

/// SDXL 进阶（低学习率长训 + block weights 强化 UNet 中段）。
fn sdxl_advanced() -> RecipeFile {
    let data = RecipeData {
        learning_rate: 8e-5,
        text_encoder_lr: Some(2e-5),
        unet_lr: None,
        optimizer: OptimizerKind::Prodigy,
        lr_scheduler: SchedulerKind::ConstantWithWarmup,
        lr_warmup_ratio: 0.05,
        network_dim: 32,
        network_alpha: 16,
        network_type: NetworkType::Locon,
        max_train_epochs: 20,
        batch_size: 1,
        resolution: 1024,
        enable_bucket: true,
        cache_latents: true,
        cache_text_encoder_outputs: true,
        mixed_precision: Precision::Bf16,
        gradient_checkpointing: true,
        gradient_accumulation_steps: 2,
        max_grad_norm: 1.0,
        seed: 42,
        min_snr_gamma: Some(5.0),
        noise_offset: Some(0.02),
        shuffle_caption: true,
        keep_tokens: 1,
        caption_dropout_rate: Some(0.1),
        save_every_n_epochs: 1,
        sample_every_n_epochs: 2,
        sample_prompts: vec![
            "1girl, masterpiece, best quality, detailed".to_string(),
            "1boy, masterpiece, best quality".to_string(),
        ],
        sample_sampler: "euler_a".to_string(),
        block_weights: Some("0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1".into()),
        trigger_word: None,
        prediction_type: None,
        ..RecipeData::default()
    };
    RecipeFile::new(
        "SDXL 进阶（低 LR + LoCon + 块权重）",
        ModelFamily::Sdxl1,
        data,
    )
}

/// Anima 入门（M2 随 BackendSdScripts Anima 路径细化；默认值参考 lora-scripts-next）。
fn anima_entry() -> RecipeFile {
    let data = RecipeData {
        learning_rate: 1e-4,
        text_encoder_lr: None,
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
        min_snr_gamma: None,
        noise_offset: None,
        shuffle_caption: true,
        keep_tokens: 1,
        caption_dropout_rate: Some(0.05),
        save_every_n_epochs: 1,
        sample_every_n_epochs: 5,
        sample_prompts: vec!["1girl, masterpiece, best quality, detailed".to_string()],
        sample_sampler: "euler".to_string(),
        block_weights: None,
        trigger_word: None,
        prediction_type: None,
        ..RecipeData::default()
    };
    RecipeFile::new("Anima 入门（DiT）", ModelFamily::DitAnima, data)
}

/// Krea 2 入门（M3 随 BackendAiToolkit 落地；占位可运行参数）。
fn krea2_entry() -> RecipeFile {
    let data = RecipeData {
        learning_rate: 5e-4,
        text_encoder_lr: None,
        unet_lr: None,
        optimizer: OptimizerKind::AdamW,
        lr_scheduler: SchedulerKind::ConstantWithWarmup,
        lr_warmup_ratio: 0.05,
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
        min_snr_gamma: None,
        noise_offset: None,
        shuffle_caption: true,
        keep_tokens: 1,
        caption_dropout_rate: Some(0.05),
        save_every_n_epochs: 1,
        sample_every_n_epochs: 5,
        sample_prompts: vec!["a beautiful scene, masterpiece".to_string()],
        sample_sampler: "euler".to_string(),
        block_weights: None,
        trigger_word: None,
        prediction_type: None,
        ..RecipeData::default()
    };
    RecipeFile::new("Krea 2 入门（DiT）", ModelFamily::DitKrea2, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::validate_recipe;

    #[test]
    fn all_presets_validate_without_errors() {
        for preset in builtin_presets() {
            let family = preset.family().unwrap();
            let issues = validate_recipe(family, &preset.data);
            assert!(
                issues
                    .iter()
                    .all(|i| i.level == crate::validate::IssueLevel::Warning),
                "预设 {} 存在错误：{issues:?}",
                preset.meta.name
            );
        }
    }

    #[test]
    fn presets_have_distinct_families() {
        let families: Vec<_> = builtin_presets()
            .iter()
            .map(|p| p.family().unwrap())
            .collect();
        assert!(families.contains(&ModelFamily::Sdxl1));
        assert!(families.contains(&ModelFamily::DitAnima));
        assert!(families.contains(&ModelFamily::DitKrea2));
    }
}

