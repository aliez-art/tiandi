//! 丹方校验器：范围检查 + 模型族适用性（PRD FR-403，点火前拦截）。

use serde::Serialize;
use tiandi_core::ModelFamily;

use crate::schema::{NetworkType, RecipeData};

/// 问题级别。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueLevel {
    /// 阻断点火
    Error,
    /// 可带警告继续
    Warning,
}

/// 单条校验结果。
#[derive(Debug, Clone, Serialize)]
pub struct RecipeIssue {
    pub level: IssueLevel,
    pub field: String,
    pub message: String,
}

/// 校验丹方（家族 + 数据）。
pub fn validate_recipe(family: ModelFamily, data: &RecipeData) -> Vec<RecipeIssue> {
    let mut issues = Vec::new();
    let mut issue = |level: IssueLevel, field: &str, message: String| {
        issues.push(RecipeIssue {
            level,
            field: field.into(),
            message,
        });
    };

    // ---- 学习率 ----
    if !(data.learning_rate > 0.0 && data.learning_rate <= 1e-1) {
        issue(
            IssueLevel::Error,
            "learning_rate",
            format!("学习率 {} 超出合理范围 (0, 0.1]", data.learning_rate),
        );
    }
    if let Some(te) = data.text_encoder_lr {
        if !(te > 0.0 && te <= 1e-1) {
            issue(
                IssueLevel::Error,
                "text_encoder_lr",
                format!("文本编码器学习率 {te} 超出合理范围 (0, 0.1]"),
            );
        }
    }
    if let Some(u) = data.unet_lr {
        if !(u > 0.0 && u <= 1e-1) {
            issue(
                IssueLevel::Error,
                "unet_lr",
                format!("UNet 学习率 {u} 超出合理范围 (0, 0.1]"),
            );
        }
    }
    // text_encoder_lr 与 unet_lr 同时设置时二者应协调（社区惯例：TE ≤ UNet）
    if let (Some(te), Some(u)) = (data.text_encoder_lr, data.unet_lr) {
        if te > u * 2.0 {
            issue(
                IssueLevel::Warning,
                "text_encoder_lr",
                format!("文本编码器学习率 {te} 显著高于 UNet {u}，易导致过拟合（建议 TE ≤ UNet）"),
            );
        }
    }

    // ---- 网络 ----
    if data.network_dim == 0 || data.network_dim > 256 {
        issue(
            IssueLevel::Error,
            "network_dim",
            format!("network_dim {} 超出范围 [1, 256]", data.network_dim),
        );
    }
    if data.network_alpha == 0 || data.network_alpha > 512 {
        issue(
            IssueLevel::Error,
            "network_alpha",
            format!("network_alpha {} 超出范围 [1, 512]", data.network_alpha),
        );
    }
    // 网络 dropout 类：比例必须在 [0, 1]
    for (field, value) in [
        ("network_dropout", data.network_dropout),
        ("rank_dropout", data.rank_dropout),
        ("module_dropout", data.module_dropout),
    ] {
        if let Some(v) = value {
            if !(0.0..=1.0).contains(&v) {
                issue(
                    IssueLevel::Error,
                    field,
                    format!("{field} {v} 超出范围 [0, 1]"),
                );
            }
        }
    }
    // block_weights：SDXL 25 值逗号分隔数字（与 schema 注释一致）
    if let Some(bw) = &data.block_weights {
        let parts: Vec<&str> = bw.split(',').map(str::trim).collect();
        let parseable = parts.len() == 25 && parts.iter().all(|p| p.parse::<f64>().is_ok());
        if !parseable {
            issue(
                IssueLevel::Error,
                "block_weights",
                format!(
                    "block_weights 必须是 25 个逗号分隔数字（当前 {} 段，解析失败）：{bw}",
                    parts.len()
                ),
            );
        }
    }
    // max_token_length：sd-scripts 仅支持 75/150/225/300
    if let Some(v) = data.max_token_length {
        if ![75, 150, 225, 300].contains(&v) {
            issue(
                IssueLevel::Error,
                "max_token_length",
                format!("max_token_length {v} 必须为 75/150/225/300 之一"),
            );
        }
    }
    // 族约束：block_weights 仅 SDXL 族
    if data.block_weights.is_some() && family != ModelFamily::Sdxl1 {
        issue(
            IssueLevel::Error,
            "block_weights",
            format!("block_weights 仅适用于 SDXL 族（当前 {}）", family.label()),
        );
    }
    // Krea 2：T-LoRA/LoHa 等网络未在 ai-toolkit krea2 路径验证
    if family == ModelFamily::DitKrea2
        && matches!(
            data.network_type,
            NetworkType::Tlora | NetworkType::LoHa | NetworkType::Ia3
        )
    {
        issue(
            IssueLevel::Warning,
            "network_type",
            format!(
                "{} 网络在 Krea 2（ai-toolkit 内核）路径未验证，建议使用 lora/locon/lokr",
                data.network_type.label()
            ),
        );
    }

    // ---- 训练规模 ----
    if data.max_train_epochs == 0 || data.max_train_epochs > 1000 {
        issue(
            IssueLevel::Error,
            "max_train_epochs",
            format!("轮数 {} 超出范围 [1, 1000]", data.max_train_epochs),
        );
    }
    if data.batch_size == 0 || data.batch_size > 64 {
        issue(
            IssueLevel::Error,
            "batch_size",
            format!("batch_size {} 超出范围 [1, 64]", data.batch_size),
        );
    }
    if data.gradient_accumulation_steps == 0 || data.gradient_accumulation_steps > 64 {
        issue(
            IssueLevel::Error,
            "gradient_accumulation_steps",
            format!(
                "梯度累积 {} 超出范围 [1, 64]",
                data.gradient_accumulation_steps
            ),
        );
    }
    if data.resolution < 256 || data.resolution > 4096 {
        issue(
            IssueLevel::Error,
            "resolution",
            format!("分辨率 {} 超出范围 [256, 4096]", data.resolution),
        );
    }
    // 数据集重复次数 ≥ 1
    if let Some(v) = data.num_repeats {
        if v < 1 {
            issue(
                IssueLevel::Error,
                "num_repeats",
                format!("num_repeats {v} 必须 ≥ 1"),
            );
        }
    }
    // 时间步裁剪：min ≤ max
    if let (Some(min), Some(max)) = (data.min_timestep, data.max_timestep) {
        if min > max {
            issue(
                IssueLevel::Error,
                "min_timestep",
                format!("min_timestep {min} 大于 max_timestep {max}"),
            );
        }
    }
    // 保存步数 > 0（0 表示关闭，但显式 Some(0) 视为配置错误）
    if let Some(v) = data.save_every_n_steps {
        if v == 0 {
            issue(
                IssueLevel::Error,
                "save_every_n_steps",
                "save_every_n_steps 必须 > 0（0 表示关闭，请省略该字段）".into(),
            );
        }
    }

    // ---- 质量技巧数值范围 ----
    if let Some(v) = data.min_snr_gamma {
        if v <= 0.0 {
            issue(
                IssueLevel::Error,
                "min_snr_gamma",
                format!("min_snr_gamma {v} 必须 > 0"),
            );
        }
    }
    if let Some(v) = data.adaptive_noise_scale {
        if v <= 0.0 {
            issue(
                IssueLevel::Error,
                "adaptive_noise_scale",
                format!("adaptive_noise_scale {v} 必须 > 0"),
            );
        }
    }
    if data.max_grad_norm <= 0.0 {
        issue(
            IssueLevel::Error,
            "max_grad_norm",
            format!("max_grad_norm {} 必须 > 0", data.max_grad_norm),
        );
    }
    if let Some(v) = data.caption_dropout_rate {
        if !(0.0..=1.0).contains(&v) {
            issue(
                IssueLevel::Error,
                "caption_dropout_rate",
                format!("caption_dropout_rate {v} 超出范围 [0, 1]"),
            );
        }
    }
    // ---- 数据集扩展（lora-scripts-next 参数面） ----
    if let Some(v) = data.caption_tag_dropout_rate {
        if !(0.0..=1.0).contains(&v) {
            issue(
                IssueLevel::Error,
                "caption_tag_dropout_rate",
                format!("caption_tag_dropout_rate {v} 超出范围 [0, 1]"),
            );
        }
    }
    if let Some(v) = data.prior_loss_weight {
        if v <= 0.0 {
            issue(
                IssueLevel::Error,
                "prior_loss_weight",
                format!("prior_loss_weight {v} 必须 > 0"),
            );
        }
    }
    if let (Some(min), Some(max)) = (data.min_bucket_reso, data.max_bucket_reso) {
        if min >= max {
            issue(
                IssueLevel::Error,
                "min_bucket_reso",
                format!("min_bucket_reso {min} 必须小于 max_bucket_reso {max}"),
            );
        }
    }
    for (k, v) in [
        ("min_bucket_reso", data.min_bucket_reso),
        ("max_bucket_reso", data.max_bucket_reso),
        ("bucket_reso_steps", data.bucket_reso_steps),
    ] {
        if let Some(v) = v {
            if v == 0 || v > 4096 {
                issue(IssueLevel::Error, k, format!("{k} {v} 超出范围 [1, 4096]"));
            }
        }
    }
    // weighted_captions 与 shuffle_caption 不推荐同开（sd-scripts 规则）
    if data.weighted_captions == Some(true) && data.shuffle_caption {
        issue(
            IssueLevel::Warning,
            "weighted_captions",
            "加权 tag 与随机打乱 tag 不推荐同开（打乱会破坏加权语法）".into(),
        );
    }
    // ---- 网络扩展 ----
    if let Some(v) = data.scale_weight_norms {
        if v <= 0.0 {
            issue(
                IssueLevel::Error,
                "scale_weight_norms",
                format!("scale_weight_norms {v} 必须 > 0"),
            );
        }
    }
    // ---- 优化扩展 ----
    if let Some(v) = &data.loss_type {
        if !["l1", "l2", "huber", "smooth_l1"].contains(&v.as_str()) {
            issue(
                IssueLevel::Error,
                "loss_type",
                format!("loss_type {v} 非法（可选 l1/l2/huber/smooth_l1）"),
            );
        }
    }
    if let Some(v) = data.lr_scheduler_num_cycles {
        if v == 0 {
            issue(
                IssueLevel::Error,
                "lr_scheduler_num_cycles",
                "lr_scheduler_num_cycles 必须 ≥ 1".into(),
            );
        }
    }
    // ---- 保存扩展 ----
    if let Some(v) = &data.save_precision {
        if !["fp16", "float", "bf16"].contains(&v.as_str()) {
            issue(
                IssueLevel::Error,
                "save_precision",
                format!("save_precision {v} 非法（可选 fp16/float/bf16）"),
            );
        }
    }
    // ---- 精度互斥 ----
    if data.full_fp16 == Some(true) && data.full_bf16 == Some(true) {
        issue(
            IssueLevel::Error,
            "full_fp16",
            "full_fp16 与 full_bf16 不能同时开启".into(),
        );
    }

    // ---- 调度 ----
    if !(0.0..=0.5).contains(&data.lr_warmup_ratio) {
        issue(
            IssueLevel::Warning,
            "lr_warmup_ratio",
            format!("预热比例 {} 建议在 [0, 0.5]", data.lr_warmup_ratio),
        );
    }

    // ---- 保存/采样 ----
    if data.save_every_n_epochs == 0 {
        issue(
            IssueLevel::Error,
            "save_every_n_epochs",
            "保存间隔必须 ≥ 1".into(),
        );
    }
    if data.sample_every_n_epochs > 0 && data.sample_prompts.is_empty() {
        issue(
            IssueLevel::Warning,
            "sample_prompts",
            "启用了周期采样但没有采样提示词，将无图可出".into(),
        );
    }

    // ---- 精度 ----
    if data.mixed_precision == crate::schema::Precision::Fp16 && family == ModelFamily::DitAnima {
        issue(
            IssueLevel::Warning,
            "mixed_precision",
            "Anima 训练用 fp16 易出现 NaN（社区经验），建议 bf16（参考 lora-scripts-next）".into(),
        );
    }

    // ---- 缓存与 caption 增强冲突（sd-scripts 规则） ----
    if data.cache_text_encoder_outputs {
        if data.shuffle_caption {
            issue(
                IssueLevel::Warning,
                "shuffle_caption",
                "缓存文本编码器输出时 shuffle_caption 无效（sd-scripts 会禁用），如需随机打乱请关闭缓存".into(),
            );
        }
        if data.caption_dropout_rate.is_some() {
            issue(
                IssueLevel::Warning,
                "caption_dropout_rate",
                "缓存文本编码器输出时 caption_dropout_rate 无效（sd-scripts 会禁用）".into(),
            );
        }
    }

    issues
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_recipe_passes_for_sdxl() {
        let data = RecipeData::default();
        let issues = validate_recipe(ModelFamily::Sdxl1, &data);
        assert!(
            issues.iter().all(|i| i.level != IssueLevel::Error),
            "默认丹方不应有错误：{issues:?}"
        );
    }

    #[test]
    fn bad_lr_is_error() {
        let data = RecipeData {
            learning_rate: 5.0,
            ..RecipeData::default()
        };
        let issues = validate_recipe(ModelFamily::Sdxl1, &data);
        assert!(issues
            .iter()
            .any(|i| i.field == "learning_rate" && i.level == IssueLevel::Error));
    }

    #[test]
    fn block_weights_rejected_outside_sdxl() {
        let data = RecipeData {
            block_weights: Some("1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1".into()),
            ..RecipeData::default()
        };
        let issues = validate_recipe(ModelFamily::DitKrea2, &data);
        assert!(issues.iter().any(|i| i.field == "block_weights"));
        // SDXL 允许
        let issues = validate_recipe(ModelFamily::Sdxl1, &data);
        assert!(!issues.iter().any(|i| i.field == "block_weights"));
    }

    #[test]
    fn anima_fp16_warns() {
        let data = RecipeData::default();
        let issues = validate_recipe(ModelFamily::DitAnima, &data);
        // 默认 bf16 → 无警告；切 fp16 才有
        assert!(!issues.iter().any(|i| i.field == "mixed_precision"));
        let mut d = data;
        d.mixed_precision = crate::schema::Precision::Fp16;
        let issues = validate_recipe(ModelFamily::DitAnima, &d);
        assert!(issues.iter().any(|i| i.field == "mixed_precision"));
    }

    #[test]
    fn caption_dropout_rate_out_of_range_is_error() {
        let data = RecipeData {
            caption_dropout_rate: Some(1.5),
            ..RecipeData::default()
        };
        let issues = validate_recipe(ModelFamily::Sdxl1, &data);
        assert!(issues
            .iter()
            .any(|i| { i.field == "caption_dropout_rate" && i.level == IssueLevel::Error }));
    }

    #[test]
    fn min_timestep_above_max_is_error() {
        let data = RecipeData {
            min_timestep: Some(900),
            max_timestep: Some(100),
            ..RecipeData::default()
        };
        let issues = validate_recipe(ModelFamily::Sdxl1, &data);
        assert!(issues
            .iter()
            .any(|i| { i.field == "min_timestep" && i.level == IssueLevel::Error }));
    }

    #[test]
    fn block_weights_wrong_arity_is_error() {
        let data = RecipeData {
            block_weights: Some("1,2".into()),
            ..RecipeData::default()
        };
        let issues = validate_recipe(ModelFamily::Sdxl1, &data);
        assert!(issues
            .iter()
            .any(|i| { i.field == "block_weights" && i.level == IssueLevel::Error }));
        // 25 值合法（SDXL 允许）
        let ok = RecipeData {
            block_weights: Some("1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1".into()),
            ..RecipeData::default()
        };
        let issues = validate_recipe(ModelFamily::Sdxl1, &ok);
        assert!(!issues.iter().any(|i| i.field == "block_weights"));
    }

    #[test]
    fn num_repeats_zero_is_error() {
        let data = RecipeData {
            num_repeats: Some(0),
            ..RecipeData::default()
        };
        let issues = validate_recipe(ModelFamily::Sdxl1, &data);
        assert!(issues
            .iter()
            .any(|i| i.field == "num_repeats" && i.level == IssueLevel::Error));
    }

    #[test]
    fn dropout_rates_out_of_range_are_errors() {
        let data = RecipeData {
            network_dropout: Some(1.2),
            rank_dropout: Some(-0.1),
            module_dropout: Some(0.5),
            ..RecipeData::default()
        };
        let issues = validate_recipe(ModelFamily::Sdxl1, &data);
        assert!(issues
            .iter()
            .any(|i| i.field == "network_dropout" && i.level == IssueLevel::Error));
        assert!(issues
            .iter()
            .any(|i| i.field == "rank_dropout" && i.level == IssueLevel::Error));
        assert!(!issues.iter().any(|i| i.field == "module_dropout"));
    }

    #[test]
    fn max_token_length_restricted_values() {
        let data = RecipeData {
            max_token_length: Some(200),
            ..RecipeData::default()
        };
        let issues = validate_recipe(ModelFamily::Sdxl1, &data);
        assert!(issues
            .iter()
            .any(|i| i.field == "max_token_length" && i.level == IssueLevel::Error));
        let ok = RecipeData {
            max_token_length: Some(225),
            ..RecipeData::default()
        };
        let issues = validate_recipe(ModelFamily::Sdxl1, &ok);
        assert!(!issues.iter().any(|i| i.field == "max_token_length"));
    }

    #[test]
    fn positive_quality_numbers_checked() {
        let data = RecipeData {
            min_snr_gamma: Some(-1.0),
            adaptive_noise_scale: Some(0.0),
            max_grad_norm: -0.5,
            ..RecipeData::default()
        };
        let issues = validate_recipe(ModelFamily::Sdxl1, &data);
        assert!(issues
            .iter()
            .any(|i| i.field == "min_snr_gamma" && i.level == IssueLevel::Error));
        assert!(issues
            .iter()
            .any(|i| i.field == "adaptive_noise_scale" && i.level == IssueLevel::Error));
        assert!(issues
            .iter()
            .any(|i| i.field == "max_grad_norm" && i.level == IssueLevel::Error));
    }

    #[test]
    fn save_every_n_steps_zero_is_error() {
        let data = RecipeData {
            save_every_n_steps: Some(0),
            ..RecipeData::default()
        };
        let issues = validate_recipe(ModelFamily::Sdxl1, &data);
        assert!(issues
            .iter()
            .any(|i| i.field == "save_every_n_steps" && i.level == IssueLevel::Error));
    }
}
