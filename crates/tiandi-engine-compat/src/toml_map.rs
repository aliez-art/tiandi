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
        "pretrained_model_name_or_path = {}\n",
        toml_quote(&paths.base_model.replace('\\', "/"))
    ));
    // tokenizer 是 sd-scripts 顶层（model 相关）参数，放 [model] 段
    if let Some(tok) = &paths.tokenizer {
        t.push_str(&format!(
            "tokenizer = {}\n",
            toml_quote(&tok.replace('\\', "/"))
        ));
    }
    if let Some(tok2) = &paths.tokenizer2 {
        t.push_str(&format!(
            "tokenizer2 = {}\n",
            toml_quote(&tok2.replace('\\', "/"))
        ));
    }
    // Anima 家族：Qwen3 TE / VAE / 分词器（模型相关顶层参数）
    if family == ModelFamily::DitAnima {
        if let Some(q) = &paths.anima_qwen3 {
            t.push_str(&format!("qwen3 = {}\n", toml_quote(&q.replace('\\', "/"))));
        }
        if let Some(v) = &paths.anima_vae {
            t.push_str(&format!("vae = {}\n", toml_quote(&v.replace('\\', "/"))));
        }
        if let Some(t5) = &paths.anima_t5_tokenizer {
            t.push_str(&format!(
                "t5_tokenizer_path = {}\n",
                toml_quote(&t5.replace('\\', "/"))
            ));
        }
        if let Some(qt) = &paths.anima_qwen3_tokenizer {
            t.push_str(&format!(
                "qwen3_tokenizer_path = {}\n",
                toml_quote(&qt.replace('\\', "/"))
            ));
        }
    }
    t.push('\n');

    // [network]
    t.push_str("[network]\n");
    // Anima 家族网络：lora_anima（实测 068bcd7 networks/ 无 tlora_anima.py，
    // T-LoRA 与 LoRA 统一走 networks.lora_anima，kohya 生态 Anima 文档同此）
    let network_module = if family == ModelFamily::DitAnima {
        "lora_anima"
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
        t.push_str(&format!("down_lr_weight = {}\n", toml_quote(bw)));
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
    // train_batch_size：--config_file 解析为 argparse 选项名（train_util.py:4013，
    // 实测 068bcd7 无 --batch_size 选项；dataset 段的 batch_size 是 dataset_config 专用键）
    t.push_str(&format!("train_batch_size = {}\n", recipe.batch_size));
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
    // warmup 仅对需要它的调度器有效：constant 等调度器输出后会直接报错
    // （sd-scripts: "SchedulerType.CONSTANT does not require num_warmup_steps"）
    if recipe.lr_scheduler != tiandi_recipe::SchedulerKind::Constant {
        t.push_str(&format!(
            "lr_warmup_steps = {}\n",
            lr_warmup_steps(recipe, dataset_image_count(&paths.dataset_dir))
        ));
    }
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
    // 训练文本编码器：LoRA 默认关闭；开启后与 TE 输出缓存互斥（缓存自动关闭，
    // sd-scripts 无法在缓存 TE 输出的同时训练 TE）。Anima 家族支持训练 Qwen3 TE，
    // 且 Anima 默认不缓存 TE 输出（正好配合 TE 训练场景）。
    let train_te = recipe.train_text_encoder.unwrap_or(false);
    t.push_str(&format!("train_text_encoder = {train_te}\n"));
    t.push_str(&format!("cache_latents = {}\n", recipe.cache_latents));
    // TE 输出缓存：训练 TE 时强制关闭；Anima 家族不缓存（Qwen3 TE 需训练）
    let cache_te =
        recipe.cache_text_encoder_outputs && !train_te && family != ModelFamily::DitAnima;
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
    // 0 或未启用时不输出（sd-scripts 对 sample_every_n_epochs<=0 会打警告）
    if recipe.sample_every_n_epochs > 0 {
        t.push_str(&format!(
            "sample_every_n_epochs = {}\n",
            recipe.sample_every_n_epochs
        ));
    }
    if let Some(v) = recipe.max_train_steps {
        t.push_str(&format!("max_train_steps = {v}\n"));
    }
    t.push_str(&format!(
        "output_dir = {}\n",
        toml_quote(&paths.output_dir.replace('\\', "/"))
    ));
    t.push_str(&format!(
        "output_name = {}\n",
        toml_quote(&paths.output_name)
    ));
    t.push_str("save_model_as = \"safetensors\"\n");
    if let Some(resume) = &paths.resume {
        t.push_str(&format!(
            "resume = {}\n",
            toml_quote(&resume.replace('\\', "/"))
        ));
    }
    if let Some(tw) = &recipe.trigger_word {
        t.push_str(&format!("trigger_word = {}\n", toml_quote(tw)));
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
                t.push_str(
                    "# prediction_type=sample：sd-scripts SDXL 路径不支持，按 epsilon 处理\n",
                );
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
        "train_data_dir = {}\n",
        toml_quote(&paths.dataset_dir.replace('\\', "/"))
    ));
    // 描述文件扩展名：本项目约定图片旁同名 .txt（sd-scripts 默认 .caption，
    // 不设置会把 .txt 描述全部当不存在）
    t.push_str("caption_extension = \".txt\"\n");
    // sd-scripts 的 resolution 参数为字符串（"1024" 或 "1024,768" 多分辨率）
    t.push_str(&format!(
        "resolution = {}\n",
        toml_quote(&recipe.resolution.to_string())
    ));
    t.push_str(&format!("enable_bucket = {}\n", recipe.enable_bucket));
    // 训练次数不输出：sd-scripts 老格式由数据集子文件夹名 `N_` 前缀控制
    // （直接含图目录由 Rust 侧生成 `<N>_data` 镜像），num_repeats 参数已从 UI 移除。
    t.push('\n');

    // [sampling]
    // 采样开关由 sample_prompts_file（lib.rs 生成的提示词文件）驱动：
    // kohya 的 sample_prompts 参数必须是**文件路径**（字符串会被 isfile 检查拒绝）
    if let Some(file) = &paths.sample_prompts_file {
        t.push_str("[sampling]\n");
        t.push_str(&format!(
            "sample_prompts = {}\n",
            toml_quote(&file.replace('\\', "/"))
        ));
        t.push_str(&format!(
            "sample_sampler = {}\n",
            toml_quote(&recipe.sample_sampler)
        ));
        if let Some(v) = recipe.sample_steps {
            t.push_str(&format!("sample_steps = {v}\n"));
        }
        if let Some(v) = recipe.guidance_scale {
            t.push_str(&format!("guidance_scale = {v}\n"));
        }
        if let Some(v) = &recipe.negative_prompt {
            t.push_str(&format!("negative_prompt = {}\n", toml_quote(v)));
        }
        t.push('\n');
    }

    // [logging]
    t.push_str("[logging]\n");
    t.push_str("log_with = \"tensorboard\"\n");
    t.push_str(&format!(
        "logging_dir = {}\n",
        toml_quote(&paths.logging_dir.replace('\\', "/"))
    ));
    t.push('\n');

    t
}

/// 生成 sd-scripts 全量微调（full fine-tune）配置 TOML：无 LoRA 网络段，
/// 输出完整模型 checkpoint；train_text_encoder 可选。
pub fn build_sdscripts_toml_full(
    recipe: &RecipeData,
    family: ModelFamily,
    paths: &TrainPaths,
    train_text_encoder: bool,
) -> String {
    let mut t = String::new();

    // [model]
    t.push_str("[model]\n");
    t.push_str(&format!(
        "pretrained_model_name_or_path = {}\n",
        toml_quote(&paths.base_model.replace('\\', "/"))
    ));
    if let Some(tok) = &paths.tokenizer {
        t.push_str(&format!(
            "tokenizer = {}\n",
            toml_quote(&tok.replace('\\', "/"))
        ));
    }
    if let Some(tok2) = &paths.tokenizer2 {
        t.push_str(&format!(
            "tokenizer2 = {}\n",
            toml_quote(&tok2.replace('\\', "/"))
        ));
    }
    if family == ModelFamily::DitAnima {
        if let Some(q) = &paths.anima_qwen3 {
            t.push_str(&format!("qwen3 = {}\n", toml_quote(&q.replace('\\', "/"))));
        }
        if let Some(v) = &paths.anima_vae {
            t.push_str(&format!("vae = {}\n", toml_quote(&v.replace('\\', "/"))));
        }
        if let Some(t5) = &paths.anima_t5_tokenizer {
            t.push_str(&format!(
                "t5_tokenizer_path = {}\n",
                toml_quote(&t5.replace('\\', "/"))
            ));
        }
        if let Some(qt) = &paths.anima_qwen3_tokenizer {
            t.push_str(&format!(
                "qwen3_tokenizer_path = {}\n",
                toml_quote(&qt.replace('\\', "/"))
            ));
        }
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
    // train_batch_size：--config_file 解析为 argparse 选项名（train_util.py:4013，
    // 实测 068bcd7 无 --batch_size 选项；dataset 段的 batch_size 是 dataset_config 专用键）
    t.push_str(&format!("train_batch_size = {}\n", recipe.batch_size));
    if train_text_encoder {
        t.push_str("train_text_encoder = true\n");
        if let Some(te) = recipe.text_encoder_lr {
            t.push_str(&format!("text_encoder_lr = {te}\n"));
        }
    } else {
        t.push_str("train_text_encoder = false\n");
        t.push_str("text_encoder_lr = 0\n");
    }
    if let Some(u) = recipe.unet_lr {
        t.push_str(&format!("unet_lr = {u}\n"));
    }
    t.push_str(&format!(
        "lr_scheduler = \"{}\"\n",
        recipe.lr_scheduler.label()
    ));
    // warmup 仅对需要它的调度器有效：constant 等调度器输出后会直接报错
    // （sd-scripts: "SchedulerType.CONSTANT does not require num_warmup_steps"）
    if recipe.lr_scheduler != tiandi_recipe::SchedulerKind::Constant {
        t.push_str(&format!(
            "lr_warmup_steps = {}\n",
            lr_warmup_steps(recipe, dataset_image_count(&paths.dataset_dir))
        ));
    }
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
    t.push_str(&format!("shuffle_caption = {}\n", recipe.shuffle_caption));
    t.push_str(&format!("keep_tokens = {}\n", recipe.keep_tokens));
    if let Some(cd) = recipe.caption_dropout_rate {
        t.push_str(&format!("caption_dropout_rate = {cd}\n"));
    }
    if let Some(snr) = recipe.min_snr_gamma {
        t.push_str(&format!("min_snr_gamma = {snr}\n"));
    }
    if let Some(no) = recipe.noise_offset {
        t.push_str(&format!("noise_offset = {no}\n"));
    }
    // 全量训练：latent 缓存可用（TE 不训时）
    t.push_str(&format!(
        "cache_latents = {}\n",
        recipe.cache_latents && !train_text_encoder
    ));
    t.push_str(&format!(
        "save_every_n_epochs = {}\n",
        recipe.save_every_n_epochs
    ));
    if let Some(v) = recipe.max_train_steps {
        t.push_str(&format!("max_train_steps = {v}\n"));
    }
    t.push_str(&format!(
        "output_dir = {}\n",
        toml_quote(&paths.output_dir.replace('\\', "/"))
    ));
    t.push_str(&format!(
        "output_name = {}\n",
        toml_quote(&paths.output_name)
    ));
    t.push_str("save_model_as = \"safetensors\"\n");
    if let Some(resume) = &paths.resume {
        t.push_str(&format!(
            "resume = {}\n",
            toml_quote(&resume.replace('\\', "/"))
        ));
    }
    if family == ModelFamily::Sdxl1 {
        match recipe.prediction_type {
            Some(tiandi_recipe::PredictionType::V) => {
                t.push_str("v_parameterization = true\n");
            }
            Some(tiandi_recipe::PredictionType::Epsilon) => {
                t.push_str("v_parameterization = false\n");
            }
            _ => {}
        }
    }
    t.push('\n');

    // [dataset]
    t.push_str("[dataset]\n");
    t.push_str(&format!(
        "train_data_dir = {}\n",
        toml_quote(&paths.dataset_dir.replace('\\', "/"))
    ));
    // 描述文件扩展名：本项目约定图片旁同名 .txt（sd-scripts 默认 .caption）
    t.push_str("caption_extension = \".txt\"\n");
    t.push_str(&format!(
        "resolution = {}\n",
        toml_quote(&recipe.resolution.to_string())
    ));
    t.push_str(&format!("enable_bucket = {}\n", recipe.enable_bucket));
    // 训练次数不输出：sd-scripts 老格式由数据集子文件夹名 `N_` 前缀控制
    // （直接含图目录由 Rust 侧生成 `<N>_data` 镜像），num_repeats 参数已从 UI 移除。
    t.push('\n');

    // [sampling]
    // 采样开关由 sample_prompts_file（lib.rs 生成的提示词文件）驱动：
    // kohya 的 sample_prompts 参数必须是**文件路径**（字符串会被 isfile 检查拒绝）
    if let Some(file) = &paths.sample_prompts_file {
        t.push_str("[sampling]\n");
        t.push_str(&format!(
            "sample_prompts = {}\n",
            toml_quote(&file.replace('\\', "/"))
        ));
        t.push_str(&format!(
            "sample_sampler = {}\n",
            toml_quote(&recipe.sample_sampler)
        ));
        if let Some(v) = recipe.sample_steps {
            t.push_str(&format!("sample_steps = {v}\n"));
        }
        t.push('\n');
    }

    // [logging]
    t.push_str("[logging]\n");
    t.push_str("log_with = \"tensorboard\"\n");
    t.push_str(&format!(
        "logging_dir = {}\n",
        toml_quote(&paths.logging_dir.replace('\\', "/"))
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
    /// 示例图提示词文件路径（kohya sample_prompts 需文件路径；None = 不采样）
    pub sample_prompts_file: Option<String>,
}

/// 预热步数：按总步数比例换算。
///
/// `dataset_images`：数据集图片数（Some(图片数) 时按真实规模估算总步数：
/// `图片数 × epochs × num_repeats / batch_size`；None = 拿不到规模信息，
/// 保留旧兜底估算 epochs × 1000 步/轮基准）。
pub fn lr_warmup_steps(recipe: &RecipeData, dataset_images: Option<u64>) -> u64 {
    let est_total = match dataset_images {
        Some(images) => {
            let images = images.max(1) as f64;
            let repeats = recipe.num_repeats.unwrap_or(1).max(1) as f64;
            let batch = recipe.batch_size.max(1) as f64;
            // 每轮步数 = 图片数 × 重复次数 / batch_size
            let per_epoch = images * repeats / batch;
            per_epoch * recipe.max_train_epochs as f64
        }
        None => recipe.max_train_epochs as f64 * 1000.0,
    };
    (est_total * recipe.lr_warmup_ratio).round() as u64
}

/// 扫描数据集目录图片数（与 lib.rs dataset_image_count 规则一致；
/// 目录不可读或无图 → None，调用方回退旧估算）。
fn dataset_image_count(dataset_dir: &str) -> Option<u64> {
    const EXTS: [&str; 5] = [".jpg", ".jpeg", ".png", ".webp", ".bmp"];
    let mut count = 0u64;
    let mut stack = vec![std::path::PathBuf::from(dataset_dir)];
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
    (count > 0).then_some(count)
}

/// TOML 基本字符串：双引号包裹，转义 `\` 与 `"`。
fn toml_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            c => out.push(c),
        }
    }
    out.push('"');
    out
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
            sample_prompts_file: Some(r"D:\runs\r1\sample_prompts.txt".into()),
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
            "caption_extension = \".txt\"",
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
        // 拿不到数据集规模 → 旧兜底估算（epochs × 1000）
        assert_eq!(lr_warmup_steps(&recipe, None), 1000);
    }

    #[test]
    fn warmup_steps_use_dataset_image_count_when_known() {
        let recipe = RecipeData {
            lr_warmup_ratio: 0.1,
            max_train_epochs: 10,
            num_repeats: Some(2),
            batch_size: 4,
            ..RecipeData::default()
        };
        // 总步数 = 100 图 × 10 轮 × 2 重复 / 4 batch = 500；预热 10% = 50
        assert_eq!(lr_warmup_steps(&recipe, Some(100)), 50);
        // batch_size 兜底 ≥ 1（0 时按 1 算，避免除零）
        let zero_batch = RecipeData {
            lr_warmup_ratio: 0.1,
            max_train_epochs: 1,
            batch_size: 0,
            ..RecipeData::default()
        };
        assert_eq!(lr_warmup_steps(&zero_batch, Some(100)), 10);
    }

    #[test]
    fn dataset_image_count_scans_recursively() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::create_dir_all(dir.path().join("thumbs")).unwrap();
        std::fs::write(dir.path().join("a.jpg"), b"x").unwrap();
        std::fs::write(dir.path().join("b.PNG"), b"x").unwrap();
        std::fs::write(dir.path().join("sub/c.webp"), b"x").unwrap();
        std::fs::write(dir.path().join("note.txt"), b"x").unwrap();
        std::fs::write(dir.path().join("thumbs/d.jpg"), b"x").unwrap();
        assert_eq!(dataset_image_count(dir.path().to_str().unwrap()), Some(3));
        // 目录不存在 → None（回退旧估算）
        assert_eq!(dataset_image_count("Z:/no/such/dir"), None);
    }

    #[test]
    fn train_batch_size_emitted_in_training() {
        let recipe = RecipeData {
            batch_size: 4,
            ..RecipeData::default()
        };
        let toml = build_sdscripts_toml(&recipe, ModelFamily::Sdxl1, &paths());
        assert!(toml.contains("train_batch_size = 4"), "{toml}");
        let full = build_sdscripts_toml_full(&recipe, ModelFamily::Sdxl1, &paths(), false);
        assert!(full.contains("train_batch_size = 4"), "{full}");
    }

    #[test]
    fn anima_tlora_maps_to_lora_anima() {
        let recipe = RecipeData {
            network_type: tiandi_recipe::NetworkType::Tlora,
            ..RecipeData::default()
        };
        let toml = build_sdscripts_toml(&recipe, ModelFamily::DitAnima, &paths());
        assert!(
            toml.contains("network_module = \"networks.lora_anima\""),
            "{toml}"
        );
        assert!(!toml.contains("tlora_anima"), "{toml}");
        // Anima + Lora 同样走 lora_anima
        let lora = RecipeData {
            network_type: tiandi_recipe::NetworkType::Lora,
            ..RecipeData::default()
        };
        let toml = build_sdscripts_toml(&lora, ModelFamily::DitAnima, &paths());
        assert!(toml.contains("network_module = \"networks.lora_anima\""));
    }

    #[test]
    fn toml_strings_are_escaped() {
        // 含引号/反斜杠的路径与提示词必须被转义（可被 toml 重新解析）
        let recipe = RecipeData {
            trigger_word: Some("say \"hi\" \\ ok".into()),
            sample_prompts: vec!["a \"quote\" prompt".into(), "back\\slash".into()],
            negative_prompt: Some("bad \"x\"".into()),
            ..RecipeData::default()
        };
        let toml = build_sdscripts_toml(&recipe, ModelFamily::Sdxl1, &paths());
        assert!(
            toml.contains(r#"trigger_word = "say \"hi\" \\ ok""#),
            "{toml}"
        );
        // sample_prompts 现在指向提示词文件（kohya 要求文件路径），转义后输出
        assert!(
            toml.contains(r#"sample_prompts = "D:/runs/r1/sample_prompts.txt""#),
            "{toml}"
        );
        assert!(toml.contains(r#"negative_prompt = "bad \"x\"""#), "{toml}");
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
    fn sampling_switch_and_prompt_fallback() {
        // 有提示词文件（lib.rs 生成）→ 输出 [sampling] 段并引用文件路径
        let toml = build_sdscripts_toml(&RecipeData::default(), ModelFamily::Sdxl1, &paths());
        assert!(toml.contains("[sampling]"), "{toml}");
        assert!(
            toml.contains(r#"sample_prompts = "D:/runs/r1/sample_prompts.txt""#),
            "{toml}"
        );
        // 无提示词文件（采样关闭）→ 不输出 [sampling] 段
        let no_sample = TrainPaths {
            sample_prompts_file: None,
            ..paths()
        };
        let toml2 = build_sdscripts_toml(&RecipeData::default(), ModelFamily::Sdxl1, &no_sample);
        assert!(!toml2.contains("[sampling]"), "{toml2}");
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

    #[test]
    fn train_text_encoder_emitted_and_disables_te_cache() {
        // 默认（关闭）：train_text_encoder = false，TE 缓存按丹方
        let default_toml =
            build_sdscripts_toml(&RecipeData::default(), ModelFamily::Sdxl1, &paths());
        assert!(
            default_toml.contains("train_text_encoder = false"),
            "{default_toml}"
        );
        assert!(
            default_toml.contains("cache_text_encoder_outputs = true"),
            "{default_toml}"
        );
        // 开启：train_text_encoder = true 且 TE 缓存强制关闭
        let recipe = RecipeData {
            train_text_encoder: Some(true),
            ..RecipeData::default()
        };
        let toml = build_sdscripts_toml(&recipe, ModelFamily::Sdxl1, &paths());
        assert!(toml.contains("train_text_encoder = true"), "{toml}");
        assert!(
            toml.contains("cache_text_encoder_outputs = false"),
            "训练 TE 时缓存必须关闭：{toml}"
        );
        // Anima：支持训练 TE（不限制），缓存保持关闭（Qwen3 TE 需训练）
        let anima_toml = build_sdscripts_toml(&recipe, ModelFamily::DitAnima, &paths());
        assert!(
            anima_toml.contains("train_text_encoder = true"),
            "{anima_toml}"
        );
    }
}
