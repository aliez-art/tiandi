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
}
