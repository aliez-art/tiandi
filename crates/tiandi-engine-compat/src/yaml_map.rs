//! 丹方 → ai-toolkit YAML 参数映射（M3：Krea 2 / DitKrea2 后端）。
//!
//! 目标结构（config/examples/train_lora_wan21_14b_24gb.yaml 同构）：
//! `job: extension` + `config.process[0].type: sd_trainer`。
//! Krea 2 arch 关键参数（extensions_built_in/diffusion_models/krea2/krea2.py）：
//! - arch = "krea2"，flow-matching（noise_scheduler=flowmatch + Linear timesteps）
//! - 文本编码器 Qwen3-VL-4B（model_kwargs.text_encoder_path，本地目录格式）
//! - VAE = Qwen-Image（model_kwargs.vae_path，指向含 vae/ 子目录的父目录）
//! - 16GB 显存：quantize(qfloat8) + quantize_te + low_vram + unload_text_encoder

use tiandi_recipe::RecipeData;

/// ai-toolkit 内核运行所需路径。
pub struct AitkPaths {
    /// Krea 2 MMDiT 权重（本地 .safetensors 或目录）
    pub base_model: String,
    /// 数据集目录（folder_path）
    pub dataset_dir: String,
    /// training_folder：ai-toolkit 在此下建 `<name>/` 子目录并保存产物
    pub training_folder: String,
    /// 任务/产物名（config.name + 文件名）
    pub output_name: String,
    /// Qwen3-VL-4B transformers 目录（含 config.json/model.safetensors/tokenizer）
    pub text_encoder: String,
    /// Qwen-Image VAE 父目录（其下有 vae/ 子目录，diffusers from_pretrained 约定）
    pub vae_root: String,
    /// 总训练步数（= epochs × 数据集图片数）
    pub steps: u64,
}

/// 生成 ai-toolkit 训练 YAML 字符串。
pub fn build_aitk_yaml(recipe: &RecipeData, paths: &AitkPaths) -> String {
    let mut s = String::new();
    s.push_str("job: extension\n");
    s.push_str("config:\n");
    s.push_str(&format!("  name: \"{}\"\n", paths.output_name));
    s.push_str("  process:\n");
    s.push_str("    - type: 'sd_trainer'\n");
    s.push_str(&format!(
        "      training_folder: \"{}\"\n",
        paths.training_folder.replace('\\', "/")
    ));
    s.push_str("      device: cuda:0\n");
    if let Some(tw) = &recipe.trigger_word {
        s.push_str(&format!("      trigger_word: \"{tw}\"\n"));
    }
    // ---- network（Krea 2 仅 linear target；conv 刻意不开放）----
    s.push_str("      network:\n");
    s.push_str("        type: 'lora'\n");
    s.push_str(&format!("        linear: {}\n", recipe.network_dim));
    s.push_str(&format!("        linear_alpha: {}\n", recipe.network_alpha));
    // ---- save（save_every=总步数 → 训练结束前最后存一次；结束自动保存兜底）----
    s.push_str("      save:\n");
    s.push_str("        dtype: bf16\n");
    s.push_str(&format!("        save_every: {}\n", paths.steps.max(1)));
    s.push_str("        max_step_saves_to_keep: 2\n");
    // ---- datasets ----
    s.push_str("      datasets:\n");
    s.push_str(&format!(
        "        - folder_path: \"{}\"\n",
        paths.dataset_dir.replace('\\', "/")
    ));
    s.push_str("          caption_ext: \"txt\"\n");
    s.push_str(&format!(
        "          caption_dropout_rate: {}\n",
        recipe.caption_dropout_rate.unwrap_or(0.0)
    ));
    s.push_str(&format!(
        "          shuffle_tokens: {}\n",
        recipe.shuffle_caption
    ));
    s.push_str("          cache_latents_to_disk: true\n");
    // 分辨率桶：Krea 2 训练以 1K 为上限（RunComfy 指南 512+768+1024）
    s.push_str(&format!("          resolution: [{}]\n", recipe.resolution));
    // ---- train ----
    s.push_str("      train:\n");
    s.push_str(&format!("        batch_size: {}\n", recipe.batch_size));
    s.push_str(&format!("        steps: {}\n", paths.steps.max(1)));
    s.push_str(&format!(
        "        gradient_accumulation: {}\n",
        recipe.gradient_accumulation_steps
    ));
    s.push_str("        train_unet: true\n");
    s.push_str("        train_text_encoder: false\n");
    s.push_str(&format!(
        "        gradient_checkpointing: {}\n",
        recipe.gradient_checkpointing
    ));
    s.push_str("        noise_scheduler: 'flowmatch'\n");
    s.push_str("        timestep_type: 'linear'\n");
    s.push_str("        optimizer: 'adamw8bit'\n");
    s.push_str(&format!("        lr: {}\n", recipe.learning_rate));
    s.push_str(&format!(
        "        max_grad_norm: {}\n",
        recipe.max_grad_norm
    ));
    s.push_str("        dtype: bf16\n");
    s.push_str("        ema_config:\n");
    s.push_str("          use_ema: true\n");
    s.push_str("          ema_decay: 0.99\n");
    // 16GB 显存：编码触发词后卸载 TE（等价于缓存 TE 输出）
    s.push_str("        unload_text_encoder: true\n");
    // ---- model ----
    s.push_str("      model:\n");
    s.push_str(&format!(
        "        name_or_path: \"{}\"\n",
        paths.base_model.replace('\\', "/")
    ));
    s.push_str("        arch: 'krea2'\n");
    s.push_str("        quantize: true\n");
    s.push_str("        quantize_te: true\n");
    s.push_str("        low_vram: true\n");
    s.push_str("        model_kwargs:\n");
    s.push_str(&format!(
        "          text_encoder_path: \"{}\"\n",
        paths.text_encoder.replace('\\', "/")
    ));
    s.push_str(&format!(
        "          vae_path: \"{}\"\n",
        paths.vae_root.replace('\\', "/")
    ));
    s.push_str("          max_text_length: 512\n");
    // ---- sampling：冒烟/链路验证不采样（出图由外部工作流负责）----
    s.push_str("      sample:\n");
    s.push_str("        disable_sampling: true\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use tiandi_recipe::RecipeData;

    fn paths() -> AitkPaths {
        AitkPaths {
            base_model: r"D:\models\krea2_raw_bf16.safetensors".into(),
            dataset_dir: r"D:\ds".into(),
            training_folder: r"D:\runs\r1\checkpoints".into(),
            output_name: "krea2-lora".into(),
            text_encoder: r"D:\kernel\qwen3vl_4b".into(),
            vae_root: r"D:\kernel\krea2".into(),
            steps: 12,
        }
    }

    #[test]
    fn yaml_contains_aitk_keys() {
        let recipe = RecipeData {
            trigger_word: Some("k2test".into()),
            ..RecipeData::default()
        };
        let yaml = build_aitk_yaml(&recipe, &paths());
        for key in [
            "job: extension",
            "type: 'sd_trainer'",
            "arch: 'krea2'",
            "noise_scheduler: 'flowmatch'",
            "timestep_type: 'linear'",
            "optimizer: 'adamw8bit'",
            "quantize: true",
            "quantize_te: true",
            "low_vram: true",
            "unload_text_encoder: true",
            "steps: 12",
            "trigger_word: \"k2test\"",
            "linear: 16",
            "save_every: 12",
        ] {
            assert!(yaml.contains(key), "YAML 缺少 {key}:\n{yaml}");
        }
        // Windows 路径正斜杠
        assert!(!yaml.contains("D:\\"), "YAML 不应有反斜杠：\n{yaml}");
    }

    #[test]
    fn krea2_yaml_roundtrip_parseable() {
        let recipe = RecipeData::default();
        let yaml = build_aitk_yaml(&recipe, &paths());
        // 基本结构自检（Rust 无 yaml 解析依赖：校验关键层级缩进）
        assert!(yaml.contains("  process:\n    - type:"));
        assert!(yaml.contains("      model:\n        name_or_path:"));
        assert!(yaml.contains("      train:\n        batch_size:"));
    }
}
