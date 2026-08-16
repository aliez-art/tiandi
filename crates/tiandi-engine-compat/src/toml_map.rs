//! 丹方 → sd-scripts TOML 参数映射（M1 核心子集）。
//!
//! 键名与 kohya sd-scripts `--config_file` 严格对齐（参考 kohya_ss lora_gui.py
//! 的控件→TOML 映射表）；后续按 compat 全参数扩展。

use tiandi_core::ModelFamily;
use tiandi_recipe::RecipeData;

/// 生成 sd-scripts 训练配置 TOML 字符串。
///
/// `paths`：内核运行所需的路径集合。
pub fn build_sdscripts_toml(
    recipe: &RecipeData,
    family: ModelFamily,
    paths: &TrainPaths,
) -> String {
    let mut t = String::new();

    // [model]
    t.push_str("[model]\n");
    t.push_str(&format!(
        "pretrained_model_name_or_path = \"{}\"\n",
        paths.base_model.replace('\\', "/")
    ));
    // tokenizer 是 sd-scripts 顶层（model 相关）参数，放 [model] 段
    if let Some(tok) = &paths.tokenizer {
        t.push_str(&format!("tokenizer = \"{}\"\n", tok.replace('\\', "/")));
    }
    if let Some(tok2) = &paths.tokenizer2 {
        t.push_str(&format!("tokenizer2 = \"{}\"\n", tok2.replace('\\', "/")));
    }
    // Anima 家族：Qwen3 TE / VAE / 分词器（模型相关顶层参数）
    if family == ModelFamily::DitAnima {
        if let Some(q) = &paths.anima_qwen3 {
            t.push_str(&format!("qwen3 = \"{}\"\n", q.replace('\\', "/")));
        }
        if let Some(v) = &paths.anima_vae {
            t.push_str(&format!("vae = \"{}\"\n", v.replace('\\', "/")));
        }
        if let Some(t5) = &paths.anima_t5_tokenizer {
            t.push_str(&format!(
                "t5_tokenizer_path = \"{}\"\n",
                t5.replace('\\', "/")
            ));
        }
        if let Some(qt) = &paths.anima_qwen3_tokenizer {
            t.push_str(&format!(
                "qwen3_tokenizer_path = \"{}\"\n",
                qt.replace('\\', "/")
            ));
        }
    }
    t.push('\n');

    // [network]
    t.push_str("[network]\n");
    // Anima 家族网络：lora_anima / tlora_anima（kohya 生态）
    let network_module = if family == ModelFamily::DitAnima {
        match recipe.network_type {
            tiandi_recipe::NetworkType::Tlora => "tlora_anima",
            _ => "lora_anima",
        }
    } else {
        network_module(recipe.network_type)
    };
    t.push_str(&format!(
        "network_module = \"networks.{}\"\n",
        network_module
    ));
    t.push_str(&format!("network_dim = {}\n", recipe.network_dim));
    t.push_str(&format!("network_alpha = {}\n", recipe.network_alpha));
    // 高级网络参数（LoRA dropout / 卷积维度）
    if let Some(v) = recipe.network_dropout {
        t.push_str(&format!("network_dropout = {v}\n"));
    }
    if let Some(v) = recipe.rank_dropout {
        t.push_str(&format!("rank_dropout = {v}\n"));
    }
    if let Some(v) = recipe.module_dropout {
        t.push_str(&format!("module_dropout = {v}\n"));
    }
    if let Some(v) = recipe.conv_dim {
        t.push_str(&format!("conv_dim = {v}\n"));
    }
    if let Some(v) = recipe.conv_alpha {
        t.push_str(&format!("conv_alpha = {v}\n"));
    }
    if let Some(bw) = &recipe.block_weights {
        t.push_str(&format!("down_lr_weight = \"{}\"\n", bw));
    }
    // Anima：attn 模式（torch 最稳，Windows 下 xformers 可选）
    if family == ModelFamily::DitAnima {
        t.push_str("attn_mode = \"torch\"\n");
        t.push_str("split_attn = true\n");
    }
    t.push('\n');

    // [optimizer]
    t.push_str("[optimizer]\n");
    t.push_str(&format!(
        "optimizer_type = \"{}\"\n",
        optimizer_type(recipe.optimizer)
    ));
    t.push('\n');

    // [training]
    t.push_str("[training]\n");
    t.push_str(&format!("learning_rate = {}\n", recipe.learning_rate));
    // sd-scripts 规则：缓存 TE 输出时只能训练 UNet（TE 冻结）
    if recipe.cache_text_encoder_outputs {
        t.push_str("text_encoder_lr = 0\n");
        t.push_str("network_train_unet_only = true\n");
    } else if let Some(te) = recipe.text_encoder_lr {
        t.push_str(&format!("text_encoder_lr = {te}\n"));
    }
    if let Some(u) = recipe.unet_lr {
        t.push_str(&format!("unet_lr = {u}\n"));
    }
    t.push_str(&format!(
        "lr_scheduler = \"{}\"\n",
        recipe.lr_scheduler.label()
    ));
    t.push_str(&format!("lr_warmup_steps = {}\n", lr_warmup_steps(recipe)));
    t.push_str(&format!("max_train_epochs = {}\n", recipe.max_train_epochs));
    t.push_str(&format!("seed = {}\n", recipe.seed));
    t.push_str(&format!(
        "mixed_precision = \"{}\"\n",
        precision(recipe.mixed_precision)
    ));
    t.push_str(&format!(
        "gradient_checkpointing = {}\n",
        recipe.gradient_checkpointing
    ));
    t.push_str(&format!(
        "gradient_accumulation_steps = {}\n",
        recipe.gradient_accumulation_steps
    ));
    t.push_str(&format!("max_grad_norm = {}\n", recipe.max_grad_norm));
    // sd-scripts 规则：缓存 TE 输出时禁用 caption 增强（shuffle/dropout）
    if recipe.cache_text_encoder_outputs {
        t.push_str("shuffle_caption = false\n");
        t.push_str(&format!("keep_tokens = {}\n", recipe.keep_tokens));
    } else {
        t.push_str(&format!("shuffle_caption = {}\n", recipe.shuffle_caption));
        t.push_str(&format!("keep_tokens = {}\n", recipe.keep_tokens));
        if let Some(cd) = recipe.caption_dropout_rate {
            t.push_str(&format!("caption_dropout_rate = {cd}\n"));
        }
    }
    if let Some(snr) = recipe.min_snr_gamma {
        t.push_str(&format!("min_snr_gamma = {snr}\n"));
    }
    if let Some(no) = recipe.noise_offset {
        t.push_str(&format!("noise_offset = {no}\n"));
    }
    if let Some(v) = recipe.adaptive_noise_scale {
        t.push_str(&format!("adaptive_noise_scale = {v}\n"));
    }
    if let Some(v) = recipe.multires_noise_iterations {
        t.push_str(&format!("multires_noise_iterations = {v}\n"));
        if let Some(d) = recipe.multires_noise_discount {
            t.push_str(&format!("multires_noise_discount = {d}\n"));
        }
    }
    if recipe.zero_terminal_snr {
        t.push_str("zero_terminal_snr = true\n");
    }
    if let Some(v) = recipe.min_timestep {
        t.push_str(&format!("min_timestep = {v}\n"));
    }
    if let Some(v) = recipe.max_timestep {
        t.push_str(&format!("max_timestep = {v}\n"));
    }
    if let Some(v) = recipe.clip_skip {
        t.push_str(&format!("clip_skip = {v}\n"));
    }
    if let Some(v) = recipe.max_token_length {
        t.push_str(&format!("max_token_length = {v}\n"));
    }
    t.push_str(&format!("cache_latents = {}\n", recipe.cache_latents));
    // Anima 家族：不缓存 TE 输出（需训练 Qwen3 TE）；其余按丹方
    let cache_te = recipe.cache_text_encoder_outputs && family != ModelFamily::DitAnima;
    t.push_str(&format!("cache_text_encoder_outputs = {cache_te}\n"));
    if recipe.cache_text_encoder_outputs_to_disk && cache_te {
        t.push_str("cache_text_encoder_outputs_to_disk = true\n");
    }
    t.push_str(&format!(
        "save_every_n_epochs = {}\n",
        recipe.save_every_n_epochs
    ));
    if let Some(v) = recipe.save_every_n_steps {
        if v > 0 {
            t.push_str(&format!("save_every_n_steps = {v}\n"));
        }
    }
    if recipe.save_state {
        t.push_str("save_state = true\n");
        if let Some(n) = recipe.save_last_n_states {
            t.push_str(&format!("save_last_n_states = {n}\n"));
        }
    }
    t.push_str(&format!(
        "sample_every_n_epochs = {}\n",
        recipe.sample_every_n_epochs
    ));
    if let Some(v) = recipe.max_train_steps {
        t.push_str(&format!("max_train_steps = {v}\n"));
    }
    t.push_str(&format!(
        "output_dir = \"{}\"\n",
        paths.output_dir.replace('\\', "/")
    ));
    t.push_str(&format!("output_name = \"{}\"\n", paths.output_name));
    t.push_str("save_model_as = \"safetensors\"\n");
    if let Some(resume) = &paths.resume {
        t.push_str(&format!("resume = \"{}\"\n", resume.replace('\\', "/")));
    }
    if let Some(tw) = &recipe.trigger_word {
        t.push_str(&format!("trigger_word = \"{tw}\"\n"));
    }
    // 预测目标：SDXL 族映射为 sd-scripts v_parameterization（v 预测模型必须开启）
    if family == ModelFamily::Sdxl1 {
        match recipe.prediction_type {
            Some(tiandi_recipe::PredictionType::V) => {
                t.push_str("v_parameterization = true\n");
            }
            Some(tiandi_recipe::PredictionType::Epsilon) => {
                t.push_str("v_parameterization = false\n");
            }
            Some(tiandi_recipe::PredictionType::Sample) => {
                t.push_str("# prediction_type=sample：sd-scripts SDXL 路径不支持，按 epsilon 处理\n");
            }
            None => {}
        }
    }
    // DiT 族：预测目标与时间步（M2 随 Anima 路径细化；Krea 2 走 ai-toolkit 不在此）
    if family == ModelFamily::DitAnima {
        // anima_train_network.py 需要 qwen3 文本编码器路径等，M2 补充
        t.push_str("# anima 专用参数（M2 落地）\n");
    }
    t.push('\n');

    // [dataset]
    t.push_str("[dataset]\n");
    t.push_str(&format!(
        "train_data_dir = \"{}\"\n",
        paths.dataset_dir.replace('\\', "/")
    ));
    // sd-scripts 的 resolution 参数为字符串（"1024" 或 "1024,768" 多分辨率）
    t.push_str(&format!("resolution = \"{}\"\n", recipe.resolution));
    t.push_str(&format!("enable_bucket = {}\n", recipe.enable_bucket));
    if let Some(v) = recipe.num_repeats {
        if v > 1 {
            t.push_str(&format!("num_repeats = {v}\n"));
        }
    }
    t.push('\n');

    // [sampling]
    if !recipe.sample_prompts.is_empty() {
        t.push_str("[sampling]\n");
        t.push_str("sample_prompts = \"\"\"\n");
        for p in &recipe.sample_prompts {
            t.push_str(p);
            t.push('\n');
        }
        t.push_str("\"\"\"\n");
        t.push_str(&format!("sample_sampler = \"{}\"\n", recipe.sample_sampler));
        if let Some(v) = recipe.sample_steps {
            t.push_str(&format!("sample_steps = {v}\n"));
        }
        if let Some(v) = recipe.guidance_scale {
            t.push_str(&format!("guidance_scale = {v}\n"));
        }
        if let Some(v) = &recipe.negative_prompt {
            t.push_str(&format!("negative_prompt = \"{v}\"\n"));
        }
        t.push('\n');
    }

    // [logging]
    t.push_str("[logging]\n");
    t.push_str("log_with = \"tensorboard\"\n");
    t.push_str(&format!(
        "logging_dir = \"{}\"\n",
        paths.logging_dir.replace('\\', "/")
    ));
    t.push('\n');

    t
}

/// 内核运行所需路径。
pub struct TrainPaths {
    pub base_model: String,
    pub dataset_dir: String,
    pub output_dir: String,
    pub output_name: String,
    pub logging_dir: String,
    /// 断点续训（sd-scripts state 目录，可选）
    pub resume: Option<String>,
    /// 本地 CLIP-L tokenizer 目录（可选；离线化，避免运行时下载）
    pub tokenizer: Option<String>,
    /// 本地 CLIP-G tokenizer2 目录（可选；SDXL 双编码器）
    pub tokenizer2: Option<String>,
    /// Anima：Qwen3 文本编码器路径
    pub anima_qwen3: Option<String>,
    /// Anima：VAE 路径
    pub anima_vae: Option<String>,
    /// Anima：T5 旧分词器目录
    pub anima_t5_tokenizer: Option<String>,
    /// Anima：Qwen3 分词器目录
    pub anima_qwen3_tokenizer: Option<String>,
}

/// 预热步数：按总步数比例换算（M1 估算：epochs × 1000 步/轮基准）。
pub fn lr_warmup_steps(recipe: &RecipeData) -> u64 {
    let est_total = recipe.max_train_epochs as f64 * 1000.0;
    (est_total * recipe.lr_warmup_ratio).round() as u64
}

fn network_module(t: tiandi_recipe::NetworkType) -> &'static str {
    match t {
        tiandi_recipe::NetworkType::Lora => "lora",
        tiandi_recipe::NetworkType::Locon => "locon",
        tiandi_recipe::NetworkType::Lokr => "lokr",
        tiandi_recipe::NetworkType::LoHa => "loha",
        tiandi_recipe::NetworkType::Ia3 => "ia3",
        tiandi_recipe::NetworkType::DoRa => "dora",
        tiandi_recipe::NetworkType::Tlora => "tlora",
    }
}

fn optimizer_type(o: tiandi_recipe::OptimizerKind) -> &'static str {
    match o {
        tiandi_recipe::OptimizerKind::AdamW => "AdamW",
        tiandi_recipe::OptimizerKind::AdamW8Bit => "AdamW8bit",
        tiandi_recipe::OptimizerKind::Adafactor => "Adafactor",
        tiandi_recipe::OptimizerKind::Prodigy => "Prodigy",
        tiandi_recipe::OptimizerKind::Lion => "Lion",
        tiandi_recipe::OptimizerKind::Lion8Bit => "Lion8bit",
        tiandi_recipe::OptimizerKind::DAdaptAdaGrad => "DAdaptAdaGrad",
        tiandi_recipe::OptimizerKind::CAME => "CAME",
        tiandi_recipe::OptimizerKind::SgdNesterov => "SGDNesterov",
    }
}

fn precision(p: tiandi_recipe::Precision) -> &'static str {
    match p {
        tiandi_recipe::Precision::Fp16 => "fp16",
        tiandi_recipe::Precision::Bf16 => "bf16",
        tiandi_recipe::Precision::Fp32 => "no",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths() -> TrainPaths {
        TrainPaths {
            base_model: r"D:\models\noobai.safetensors".into(),
            dataset_dir: r"D:\ds".into(),
            output_dir: r"D:\runs\r1".into(),
            output_name: "noobai-lora".into(),
            logging_dir: r"D:\runs\r1\logs".into(),
            resume: None,
            tokenizer: None,
            tokenizer2: None,
            anima_qwen3: None,
            anima_vae: None,
            anima_t5_tokenizer: None,
            anima_qwen3_tokenizer: None,
        }
    }

    #[test]
    fn resume_is_emitted_when_present() {
        let recipe = RecipeData::default();
        let mut paths = paths();
        paths.resume = Some(r"D:\runs\r1\checkpoints\noobai-lora-000010.state".into());
        let toml = build_sdscripts_toml(&recipe, ModelFamily::Sdxl1, &paths);
        assert!(
            toml.contains("resume = \"D:/runs/r1/checkpoints/noobai-lora-000010.state\""),
            "{toml}"
        );
    }

    #[test]
    fn toml_contains_kohya_keys() {
        let recipe = RecipeData::default();
        let toml = build_sdscripts_toml(&recipe, ModelFamily::Sdxl1, &paths());
        // 关键键名与 kohya sd-scripts 对齐
        for key in [
            "[model]",
            "[network]",
            "[optimizer]",
            "[training]",
            "[dataset]",
            "[logging]",
            "pretrained_model_name_or_path",
            "network_module = \"networks.lora\"",
            "optimizer_type = \"AdamW8bit\"",
            "learning_rate = 0.0001",
            "lr_scheduler = \"cosine\"",
            "mixed_precision = \"bf16\"",
            "cache_latents = true",
            "cache_text_encoder_outputs = true",
            "enable_bucket = true",
            "min_snr_gamma = 5",
            "save_model_as = \"safetensors\"",
            "train_data_dir = \"D:/ds\"",
        ] {
            assert!(toml.contains(key), "TOML 缺少 {key}:\n{toml}");
        }
    }

    #[test]
    fn windows_paths_are_forward_slashed() {
        let recipe = RecipeData::default();
        let toml = build_sdscripts_toml(&recipe, ModelFamily::Sdxl1, &paths());
        assert!(!toml.contains("D:\\"), "TOML 中不应有反斜杠路径");
    }

    #[test]
    fn warmup_steps_derived_from_ratio() {
        let recipe = RecipeData {
            lr_warmup_ratio: 0.1,
            max_train_epochs: 10,
            ..RecipeData::default()
        };
        assert_eq!(lr_warmup_steps(&recipe), 1000);
    }

    #[test]
    fn trigger_word_included() {
        let recipe = RecipeData {
            trigger_word: Some("zhongzi".into()),
            ..RecipeData::default()
        };
        let toml = build_sdscripts_toml(&recipe, ModelFamily::Sdxl1, &paths());
        assert!(toml.contains("trigger_word = \"zhongzi\""));
    }

    #[test]
    fn v_prediction_emits_v_parameterization_sdxl_only() {
        let recipe = RecipeData {
            prediction_type: Some(tiandi_recipe::PredictionType::V),
            ..RecipeData::default()
        };
        let toml = build_sdscripts_toml(&recipe, ModelFamily::Sdxl1, &paths());
        assert!(toml.contains("v_parameterization = true"), "{toml}");
        // Anima 家族不输出该参数（其预测目标由 anima 训练脚本自理）
        let anima_toml = build_sdscripts_toml(&recipe, ModelFamily::DitAnima, &paths());
        assert!(!anima_toml.contains("v_parameterization"), "{anima_toml}");
    }

    #[test]
    fn epsilon_prediction_emits_explicit_false() {
        let recipe = RecipeData {
            prediction_type: Some(tiandi_recipe::PredictionType::Epsilon),
            ..RecipeData::default()
        };
        let toml = build_sdscripts_toml(&recipe, ModelFamily::Sdxl1, &paths());
        assert!(toml.contains("v_parameterization = false"), "{toml}");
    }
}
