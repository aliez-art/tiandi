//! 丹方 → ai-toolkit YAML 参数映射（M3：Krea 2 / DitKrea2 后端）。
//!
//! 目标结构（config/examples/train_lora_wan21_14b_24gb.yaml 同构）：
//! `job: extension` + `config.process[0].type: sd_trainer`。
//! Krea 2 arch 关键参数（extensions_built_in/diffusion_models/krea2/krea2.py）：
//! - arch = "krea2"，flow-matching（noise_scheduler=flowmatch + Linear timesteps）
//! - 文本编码器 Qwen3-VL-4B（model_kwargs.text_encoder_path，本地目录格式）
//! - VAE = Qwen-Image（model_kwargs.vae_path，指向含 vae/ 子目录的父目录）
//! - 16GB 显存：quantize(qfloat8) + quantize_te + low_vram + unload_text_encoder
//!
//! 注意：tiandi-recipe 新增的数据集/网络/优化/保存扩展字段（reg_data_dir、
//! prior_loss_weight、min_bucket_reso/max_bucket_reso、bucket_*、weighted_captions、
//! caption_dropout_*、network_weights、scale_weight_norms、network_args_custom、
//! loss_type、lr_scheduler_num_cycles、optimizer_args_custom、save_precision、
//! save_last_n_epochs_state、full_fp16/full_bf16、no_half_vae、xformers、lowram、
//! persistent_data_loader_workers、vae_batch_size）均为 sd-scripts（kohya）专用键，
//! Krea 2 / ai-toolkit 无对应概念，此处不映射；如需新增请先核实 ai-toolkit
//! config_modules.py 是否存在对应键。

use tiandi_recipe::OptimizerKind;
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
///
/// `repeats`：每张图训练次数（来自数据集目录名前缀数字，如 2_artstyle → 2；
/// UI 不再单独设置该参数）。
pub fn build_aitk_yaml(recipe: &RecipeData, paths: &AitkPaths, repeats: u64) -> String {
    let mut s = String::new();
    s.push_str("job: extension\n");
    s.push_str("config:\n");
    s.push_str(&format!("  name: {}\n", yaml_str(&paths.output_name)));
    s.push_str("  process:\n");
    s.push_str("    - type: 'sd_trainer'\n");
    s.push_str(&format!(
        "      training_folder: {}\n",
        yaml_str(&paths.training_folder.replace('\\', "/"))
    ));
    s.push_str("      device: cuda:0\n");
    if let Some(tw) = &recipe.trigger_word {
        s.push_str(&format!("      trigger_word: {}\n", yaml_str(tw)));
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
        "        - folder_path: {}\n",
        yaml_str(&paths.dataset_dir.replace('\\', "/"))
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
    // 缓存 latents 到磁盘：消费丹方 cache_latents 字段（ai-toolkit 数据集级键，
    // config_modules.py:1005 DatasetConfig.cache_latents_to_disk）
    s.push_str(&format!(
        "          cache_latents_to_disk: {}\n",
        recipe.cache_latents
    ));
    // 分辨率桶：Krea 2 训练以 1K 为上限（RunComfy 指南 512+768+1024）
    s.push_str(&format!("          resolution: [{}]\n", recipe.resolution));
    if repeats > 1 {
        s.push_str(&format!("          num_repeats: {repeats}\n"));
    }
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
    // optimizer/lr_scheduler 消费丹方字段（ai-toolkit TrainConfig 键，
    // config_modules.py:387/389；不支持的优化器回退 adamw8bit 并注释）
    let (optimizer, opt_fallback) = aitk_optimizer(recipe.optimizer);
    if opt_fallback {
        s.push_str("        # 注意：丹方优化器在 ai-toolkit 无对应实现，回退 adamw8bit\n");
    }
    s.push_str(&format!("        optimizer: '{}'\n", optimizer));
    s.push_str(&format!(
        "        lr_scheduler: '{}'\n",
        recipe.lr_scheduler.label()
    ));
    s.push_str(&format!("        lr: {}\n", recipe.learning_rate));
    s.push_str(&format!(
        "        max_grad_norm: {}\n",
        recipe.max_grad_norm
    ));
    s.push_str("        dtype: bf16\n");
    s.push_str("        ema_config:\n");
    s.push_str("          use_ema: true\n");
    s.push_str("          ema_decay: 0.99\n");
    // 无采样提示词时禁用采样：disable_sampling 是 train 级键
    // （实测 flux_train_ui.py:199/210 写 process[0]["train"]["disable_sampling"]，
    //  SDTrainer.py:143 读 self.train_config.disable_sampling；放 sample: 段无效）
    if recipe.sample_prompts.is_empty() {
        s.push_str("        disable_sampling: true\n");
    }
    // 16GB 显存：编码触发词后卸载 TE（等价于缓存 TE 输出）
    s.push_str("        unload_text_encoder: true\n");
    // ---- model ----
    s.push_str("      model:\n");
    s.push_str(&format!(
        "        name_or_path: {}\n",
        yaml_str(&paths.base_model.replace('\\', "/"))
    ));
    s.push_str("        arch: 'krea2'\n");
    s.push_str("        quantize: true\n");
    s.push_str("        quantize_te: true\n");
    s.push_str("        low_vram: true\n");
    s.push_str("        model_kwargs:\n");
    s.push_str(&format!(
        "          text_encoder_path: {}\n",
        yaml_str(&paths.text_encoder.replace('\\', "/"))
    ));
    s.push_str(&format!(
        "          vae_path: {}\n",
        yaml_str(&paths.vae_root.replace('\\', "/"))
    ));
    s.push_str("          max_text_length: 512\n");
    // ---- sampling：示例图（每 sample_every_n_epochs 轮出一批）----
    if !recipe.sample_prompts.is_empty() {
        s.push_str("      sample:\n");
        // 采样器消费丹方 sample_sampler（无则用 Krea 2 flow-match 默认）
        let sampler = if recipe.sample_sampler.is_empty() {
            "flowmatch"
        } else {
            &recipe.sample_sampler
        };
        s.push_str(&format!("        sampler: {}\n", yaml_str(sampler)));
        // 训练步数占比换算：sample_every 按轮 → 每 epochs/steps 换算为步
        let steps = paths.steps.max(1);
        let sample_every = if recipe.sample_every_n_epochs > 0 {
            (steps as f64 / recipe.max_train_epochs.max(1) as f64
                * recipe.sample_every_n_epochs as f64)
                .ceil()
                .max(1.0) as u64
        } else {
            0
        };
        if sample_every > 0 {
            s.push_str(&format!("        sample_every: {sample_every}\n"));
            s.push_str("        sample_start_step: 0\n");
        }
        if let Some(v) = recipe.sample_steps {
            s.push_str(&format!("        sample_steps: {v}\n"));
        } else {
            s.push_str("        sample_steps: 30\n");
        }
        if let Some(v) = recipe.guidance_scale {
            s.push_str(&format!("        guidance_scale: {v}\n"));
        } else {
            s.push_str("        guidance_scale: 4\n");
        }
        s.push_str("        width: 1024\n");
        s.push_str("        height: 1024\n");
        if let Some(v) = &recipe.negative_prompt {
            s.push_str(&format!("        neg: {}\n", yaml_str(v)));
        } else {
            s.push_str("        neg: \"\"\n");
        }
        s.push_str("        prompts:\n");
        for p in &recipe.sample_prompts {
            s.push_str(&format!("          - {}\n", yaml_str(p)));
        }
        // 采样种子消费丹方 seed（SampleConfig 键，config_modules.py:87）
        s.push_str(&format!("        seed: {}\n", recipe.seed));
        s.push_str("        walk_seed: false\n");
    }
    s
}

/// RecipeData 优化器 → ai-toolkit 优化器字符串。
///
/// 返回 `(名称, 是否回退)`：ai-toolkit 无对应实现的枚举（CAME/SGDNesterov）回退
/// adamw8bit（原硬编码默认），调用方输出注释提示。
/// 对照 toolkit/optimizer.py 实测支持的名称。
fn aitk_optimizer(o: OptimizerKind) -> (&'static str, bool) {
    match o {
        OptimizerKind::AdamW => ("adamw", false),
        OptimizerKind::AdamW8Bit => ("adamw8bit", false),
        OptimizerKind::Adafactor => ("adafactor", false),
        OptimizerKind::Prodigy => ("prodigy", false),
        OptimizerKind::Lion => ("lion", false),
        OptimizerKind::Lion8Bit => ("lion8bit", false),
        // ai-toolkit 用 dadaptation（DAdaptAdam）；DAdaptAdaGrad 无对应，走 D-Adapt 家族
        OptimizerKind::DAdaptAdaGrad => ("dadaptation", true),
        OptimizerKind::CAME => ("adamw8bit", true),
        OptimizerKind::SgdNesterov => ("adamw8bit", true),
    }
}

/// YAML 双引号字符串：用 JSON 字符串序列化输出，天然是合法 YAML 双引号字符串
/// （引号/反斜杠/换行均被正确转义）。
fn yaml_str(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string())
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
        let yaml = build_aitk_yaml(&recipe, &paths(), 1);
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
        let yaml = build_aitk_yaml(&recipe, &paths(), 1);
        // 基本结构自检（Rust 无 yaml 解析依赖：校验关键层级缩进）
        assert!(yaml.contains("  process:\n    - type:"));
        assert!(yaml.contains("      model:\n        name_or_path:"));
        assert!(yaml.contains("      train:\n        batch_size:"));
    }

    #[test]
    fn disable_sampling_is_train_level_when_no_prompts() {
        let recipe = RecipeData {
            sample_prompts: vec![],
            ..RecipeData::default()
        };
        let yaml = build_aitk_yaml(&recipe, &paths(), 1);
        // disable_sampling 是 train 级键（ai-toolkit SDTrainer 读 self.train_config.disable_sampling）
        assert!(
            yaml.contains("      train:\n        batch_size:")
                && yaml.contains("        disable_sampling: true\n"),
            "disable_sampling 应在 train 段：\n{yaml}"
        );
        // 不应再出现在 sample: 段，且无采样时不输出 sample 段
        assert!(
            !yaml.contains("sample:"),
            "无提示词不应输出 sample 段：\n{yaml}"
        );
        assert!(
            !yaml.contains("sample:\n        disable_sampling"),
            "{yaml}"
        );
    }

    #[test]
    fn krea2_consumes_recipe_fields() {
        let recipe = RecipeData {
            optimizer: tiandi_recipe::OptimizerKind::Prodigy,
            sample_sampler: "euler_a".into(),
            seed: 12345,
            cache_latents: false,
            ..RecipeData::default()
        };
        let yaml = build_aitk_yaml(&recipe, &paths(), 1);
        assert!(yaml.contains("optimizer: 'prodigy'"), "{yaml}");
        assert!(yaml.contains("sampler: \"euler_a\""), "{yaml}");
        assert!(yaml.contains("seed: 12345"), "{yaml}");
        assert!(yaml.contains("cache_latents_to_disk: false"), "{yaml}");
        // 默认缓存打开
        let yaml = build_aitk_yaml(&RecipeData::default(), &paths(), 1);
        assert!(yaml.contains("cache_latents_to_disk: true"), "{yaml}");
        // 有采样提示词时 train 段不输出 disable_sampling
        assert!(!yaml.contains("disable_sampling"), "{yaml}");
    }

    #[test]
    fn unsupported_optimizer_falls_back_with_comment() {
        let recipe = RecipeData {
            optimizer: tiandi_recipe::OptimizerKind::CAME,
            ..RecipeData::default()
        };
        let yaml = build_aitk_yaml(&recipe, &paths(), 1);
        assert!(yaml.contains("optimizer: 'adamw8bit'"), "{yaml}");
        assert!(yaml.contains("回退 adamw8bit"), "{yaml}");
    }

    #[test]
    fn yaml_prompts_are_json_escaped() {
        let recipe = RecipeData {
            sample_prompts: vec!["a \"quoted\" prompt".into(), "back\\slash".into()],
            negative_prompt: Some("bad \"x\"".into()),
            ..RecipeData::default()
        };
        let yaml = build_aitk_yaml(&recipe, &paths(), 1);
        // serde_json::to_string 输出即合法 YAML 双引号字符串
        assert!(
            yaml.contains(r#"          - "a \"quoted\" prompt""#),
            "{yaml}"
        );
        assert!(yaml.contains(r#"          - "back\\slash""#), "{yaml}");
        assert!(yaml.contains(r#"        neg: "bad \"x\"""#), "{yaml}");
    }
}
