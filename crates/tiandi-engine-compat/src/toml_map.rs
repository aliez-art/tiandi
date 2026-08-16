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
    // 网络扩展（续训权重 / 最大范数正则 / 自定义 network_args）
    push_network_ext(&mut t, recipe);
    t.push('\n');

    // [optimizer]
    t.push_str("[optimizer]\n");
    t.push_str(&format!(
        "optimizer_type = \"{}\"\n",
        optimizer_type(recipe.optimizer)
    ));
    // 自定义 optimizer_args（k=v 列表，见 custom_args_toml）
    push_optimizer_ext(&mut t, recipe);
    t.push('\n');

    // [training]
    t.push_str("[training]\n");
    t.push_str(&format!("learning_rate = {}\n", recipe.learning_rate));
    // train_batch_size：--config_file 解析为 argparse 选项名（train_util.py:4013，
    // 实测 068bcd7 无 --batch_size 选项；dataset 段的 batch_size 是 dataset_config 专用键）
    t.push_str(&format!("train_batch_size = {}\n", recipe.batch_size));
    // TE 训练联动（train_util.py:3984 附近 / train_network.py:1765,1810）：
    // train_text_encoder=Some(true) → 训练 UNet+TE（network_train_unet_only=false），
    // text_encoder_lr 用丹方值（未指定则不输出、跟随主学习率）；
    // 未开启 → 维持现状：缓存 TE 输出时冻结 TE（text_encoder_lr=0 +
    // network_train_unet_only=true），否则用丹方 text_encoder_lr。
    let train_te = recipe.train_text_encoder.unwrap_or(false);
    if train_te {
        t.push_str("network_train_unet_only = false\n");
        if let Some(te) = recipe.text_encoder_lr {
            t.push_str(&format!("text_encoder_lr = {te}\n"));
        }
    } else if recipe.cache_text_encoder_outputs {
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
            lr_warmup_steps(
                recipe,
                dataset_image_count(&paths.dataset_dir),
                paths.repeats
            )
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
    push_training_ext(&mut t, recipe);
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

    // [saving]
    t.push_str("[saving]\n");
    push_saving_ext(&mut t, recipe);
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
    // 数据集扩展（正则化目录 / arb 桶 / 加权 tag / caption 丢弃 / 加载器与 VAE 批量）
    push_dataset_ext(&mut t, recipe);
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

    // 全量微调无网络段；仅当配置了网络扩展参数（续训 LoRA 权重等）时输出
    // （sdxl_train.py / anima_train.py 无这些选项，属无害冗余；键名与
    // train_network.py setup_parser 对齐）
    if recipe.network_weights.is_some()
        || recipe.scale_weight_norms.is_some()
        || recipe
            .network_args_custom
            .iter()
            .any(|s| !s.trim().is_empty())
    {
        t.push_str("[network]\n");
        push_network_ext(&mut t, recipe);
        t.push('\n');
    }

    // [optimizer]
    t.push_str("[optimizer]\n");
    t.push_str(&format!(
        "optimizer_type = \"{}\"\n",
        optimizer_type(recipe.optimizer)
    ));
    // 自定义 optimizer_args（k=v 列表，见 custom_args_toml）
    push_optimizer_ext(&mut t, recipe);
    t.push('\n');

    // [training]
    t.push_str("[training]\n");
    t.push_str(&format!("learning_rate = {}\n", recipe.learning_rate));
    // train_batch_size：--config_file 解析为 argparse 选项名（train_util.py:4013，
    // 实测 068bcd7 无 --batch_size 选项；dataset 段的 batch_size 是 dataset_config 专用键）
    t.push_str(&format!("train_batch_size = {}\n", recipe.batch_size));
    // TE 训练联动（与 LoRA 版一致）：开启时 network_train_unet_only=false、
    // text_encoder_lr 用丹方值（未指定则不输出、跟随主学习率）；未开启维持现状
    // （text_encoder_lr=0）。train_text_encoder 参数与丹方字段同源，取并集兜底。
    let train_te = recipe.train_text_encoder.unwrap_or(false) || train_text_encoder;
    if train_te {
        t.push_str("train_text_encoder = true\n");
        t.push_str("network_train_unet_only = false\n");
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
            lr_warmup_steps(
                recipe,
                dataset_image_count(&paths.dataset_dir),
                paths.repeats
            )
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
    push_training_ext(&mut t, recipe);
    if let Some(snr) = recipe.min_snr_gamma {
        t.push_str(&format!("min_snr_gamma = {snr}\n"));
    }
    if let Some(no) = recipe.noise_offset {
        t.push_str(&format!("noise_offset = {no}\n"));
    }
    // 全量训练：latent 缓存可用（TE 不训时）；训练 TE 时显式关掉 TE 输出缓存
    // （与 LoRA 版联动一致，避免"以为在缓存实则没缓存"的隐性差异）
    t.push_str(&format!(
        "cache_latents = {}\n",
        recipe.cache_latents && !train_te
    ));
    if train_te {
        t.push_str("cache_text_encoder_outputs = false\n");
    }
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

    // [saving]
    t.push_str("[saving]\n");
    push_saving_ext(&mut t, recipe);
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
    // 数据集扩展（正则化目录 / arb 桶 / 加权 tag / caption 丢弃 / 加载器与 VAE 批量）
    push_dataset_ext(&mut t, recipe);
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

/// 内核运行所需路径。
pub struct TrainPaths {
    pub base_model: String,
    pub dataset_dir: String,
    pub output_dir: String,
    pub output_name: String,
    pub logging_dir: String,
    /// 每张图训练次数（数据集目录名前缀数字，如 `2_artstyle` → 2；
    /// 用于步数/预热估算，与镜像目录一致）
    pub repeats: u64,
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
/// `图片数 × epochs × repeats / batch_size`；None = 拿不到规模信息，
/// 保留旧兜底估算 epochs × 1000 步/轮基准）。
/// `repeats`：每张图训练次数（来自数据集目录名前缀数字，经 TrainPaths 传入；
/// 丹方里的 num_repeats 已从 UI 移除、恒 None，不能再用它估算）。
pub fn lr_warmup_steps(recipe: &RecipeData, dataset_images: Option<u64>, repeats: u64) -> u64 {
    let est_total = match dataset_images {
        Some(images) => {
            let images = images.max(1) as f64;
            let repeats = repeats.max(1) as f64;
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

/// TOML 基本字符串：双引号包裹，转义 `\`、`"` 与控制字符（换行/制表符/其他
/// 控制字符 → `\n`/`\t`/`\uXXXX`）——否则会生成非法 TOML，内核配置解析直接失败。
fn toml_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c == '\u{7f}' => {
                out.push_str(&format!("\\u{:04X}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// 输出 `key = value`（None 时不输出）。bool 输出 true/false，与 sd-scripts
/// argparse store_true 布尔选项的 TOML 写法一致。
fn push_opt<T: std::fmt::Display>(t: &mut String, key: &str, value: Option<T>) {
    if let Some(v) = value {
        t.push_str(&format!("{key} = {v}\n"));
    }
}

/// 自定义 nargs="*" 参数行（network_args / optimizer_args）→ TOML 数组。
///
/// 每行 trim、空行过滤、含 `=` 的行原样保留。必须输出**数组**而非逗号拼接的
/// 字符串：kohya 消费端按列表迭代（train_network.py:663 `for net_arg in
/// args.network_args`、train_util.py:4992 `for arg in args.optimizer_args`，
/// 实测 068bcd7），字符串会被按字符迭代导致 split("=") 崩溃。
fn custom_args_toml(lines: &[String]) -> Option<String> {
    let items: Vec<&str> = lines
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    if items.is_empty() {
        return None;
    }
    Some(format!(
        "[{}]",
        items
            .iter()
            .map(|s| toml_quote(s))
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// [dataset] 段扩展键（键名与 train_util.py add_dataset_arguments 对齐）。
fn push_dataset_ext(t: &mut String, recipe: &RecipeData) {
    if let Some(v) = &recipe.reg_data_dir {
        t.push_str(&format!(
            "reg_data_dir = {}\n",
            toml_quote(&v.replace('\\', "/"))
        ));
    }
    push_opt(t, "min_bucket_reso", recipe.min_bucket_reso);
    push_opt(t, "max_bucket_reso", recipe.max_bucket_reso);
    push_opt(t, "bucket_reso_steps", recipe.bucket_reso_steps);
    push_opt(t, "bucket_no_upscale", recipe.bucket_no_upscale);
    push_opt(t, "weighted_captions", recipe.weighted_captions);
    push_opt(
        t,
        "caption_dropout_every_n_epochs",
        recipe.caption_dropout_every_n_epochs,
    );
    push_opt(
        t,
        "caption_tag_dropout_rate",
        recipe.caption_tag_dropout_rate,
    );
    push_opt(
        t,
        "persistent_data_loader_workers",
        recipe.persistent_data_loader_workers,
    );
    push_opt(t, "vae_batch_size", recipe.vae_batch_size);
}

/// [network] 段扩展键（键名与 train_network.py setup_parser 对齐；不输出段头）。
fn push_network_ext(t: &mut String, recipe: &RecipeData) {
    if let Some(v) = &recipe.network_weights {
        t.push_str(&format!(
            "network_weights = {}\n",
            toml_quote(&v.replace('\\', "/"))
        ));
    }
    push_opt(t, "scale_weight_norms", recipe.scale_weight_norms);
    if let Some(args) = custom_args_toml(&recipe.network_args_custom) {
        t.push_str(&format!("network_args = {args}\n"));
    }
}

/// [optimizer] 段扩展键（键名与 train_util.py add_optimizer_arguments 对齐）。
fn push_optimizer_ext(t: &mut String, recipe: &RecipeData) {
    if let Some(args) = custom_args_toml(&recipe.optimizer_args_custom) {
        t.push_str(&format!("optimizer_args = {args}\n"));
    }
}

/// [training] 段扩展键（键名与 train_util.py add_training_arguments /
/// add_optimizer_arguments（lr_scheduler_num_cycles）对齐）。
fn push_training_ext(t: &mut String, recipe: &RecipeData) {
    push_opt(t, "prior_loss_weight", recipe.prior_loss_weight);
    if let Some(lt) = &recipe.loss_type {
        t.push_str(&format!("loss_type = {}\n", toml_quote(lt)));
    }
    push_opt(t, "lr_scheduler_num_cycles", recipe.lr_scheduler_num_cycles);
    push_opt(t, "full_fp16", recipe.full_fp16);
    push_opt(t, "full_bf16", recipe.full_bf16);
    push_opt(t, "no_half_vae", recipe.no_half_vae);
    push_opt(t, "xformers", recipe.xformers);
    push_opt(t, "lowram", recipe.lowram);
}

/// [saving] 段：保存格式与状态保留（save_model_as + 扩展键）。
///
/// 段名仅为组织用途：sd-scripts read_config_from_file（train_util.py:4882）
/// 会把所有段扁平化合并为 argparse Namespace，键名必须与 argparse 选项一致。
fn push_saving_ext(t: &mut String, recipe: &RecipeData) {
    t.push_str("save_model_as = \"safetensors\"\n");
    if let Some(sp) = &recipe.save_precision {
        t.push_str(&format!("save_precision = {}\n", toml_quote(sp)));
    }
    push_opt(
        t,
        "save_last_n_epochs_state",
        recipe.save_last_n_epochs_state,
    );
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
            repeats: 1,
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
        assert_eq!(lr_warmup_steps(&recipe, None, 1), 1000);
    }

    #[test]
    fn warmup_steps_use_dataset_image_count_when_known() {
        let recipe = RecipeData {
            lr_warmup_ratio: 0.1,
            max_train_epochs: 10,
            batch_size: 4,
            ..RecipeData::default()
        };
        // 总步数 = 100 图 × 10 轮 × 2 重复 / 4 batch = 500；预热 10% = 50
        // repeats 走显式参数（来自数据集目录名前缀数字，丹方 num_repeats 已废弃）
        assert_eq!(lr_warmup_steps(&recipe, Some(100), 2), 50);
        // batch_size 兜底 ≥ 1（0 时按 1 算，避免除零）
        let zero_batch = RecipeData {
            lr_warmup_ratio: 0.1,
            max_train_epochs: 1,
            batch_size: 0,
            ..RecipeData::default()
        };
        assert_eq!(lr_warmup_steps(&zero_batch, Some(100), 1), 10);
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

    #[test]
    fn dataset_ext_keys_emitted_in_both_builders() {
        let recipe = RecipeData {
            reg_data_dir: Some(r"D:\reg".into()),
            min_bucket_reso: Some(256),
            max_bucket_reso: Some(1024),
            bucket_reso_steps: Some(64),
            bucket_no_upscale: Some(true),
            weighted_captions: Some(true),
            caption_dropout_every_n_epochs: Some(5),
            caption_tag_dropout_rate: Some(0.1),
            persistent_data_loader_workers: Some(true),
            vae_batch_size: Some(8),
            ..RecipeData::default()
        };
        let toml = build_sdscripts_toml(&recipe, ModelFamily::Sdxl1, &paths());
        for key in [
            "reg_data_dir = \"D:/reg\"",
            "min_bucket_reso = 256",
            "max_bucket_reso = 1024",
            "bucket_reso_steps = 64",
            "bucket_no_upscale = true",
            "weighted_captions = true",
            "caption_dropout_every_n_epochs = 5",
            "caption_tag_dropout_rate = 0.1",
            "persistent_data_loader_workers = true",
            "vae_batch_size = 8",
        ] {
            assert!(toml.contains(key), "TOML 缺少 {key}:\n{toml}");
        }
        // Windows 路径正斜杠
        assert!(!toml.contains("D:\\"), "TOML 不应有反斜杠：\n{toml}");
        // 全量版同样输出
        let full = build_sdscripts_toml_full(&recipe, ModelFamily::Sdxl1, &paths(), false);
        for key in [
            "reg_data_dir = \"D:/reg\"",
            "min_bucket_reso = 256",
            "bucket_no_upscale = true",
            "vae_batch_size = 8",
        ] {
            assert!(full.contains(key), "full TOML 缺少 {key}:\n{full}");
        }
    }

    #[test]
    fn custom_args_join_as_toml_array() {
        let recipe = RecipeData {
            // 含前后空白（trim）、空行（过滤）、含 `=` 原样保留
            network_args_custom: vec![
                " conv_dim=32 ".into(),
                "conv_alpha=16".into(),
                "".into(),
                "   ".into(),
                "train_llm_adapter=True".into(),
            ],
            optimizer_args_custom: vec!["lr=1e-5".into(), "weight_decay=0.1".into(), "  ".into()],
            ..RecipeData::default()
        };
        let toml = build_sdscripts_toml(&recipe, ModelFamily::Sdxl1, &paths());
        assert!(
            toml.contains(
                r#"network_args = ["conv_dim=32", "conv_alpha=16", "train_llm_adapter=True"]"#
            ),
            "{toml}"
        );
        assert!(
            toml.contains(r#"optimizer_args = ["lr=1e-5", "weight_decay=0.1"]"#),
            "{toml}"
        );
        // 全量版同样输出
        let full = build_sdscripts_toml_full(&recipe, ModelFamily::Sdxl1, &paths(), false);
        assert!(
            full.contains(
                r#"network_args = ["conv_dim=32", "conv_alpha=16", "train_llm_adapter=True"]"#
            ),
            "{full}"
        );
        assert!(
            full.contains(r#"optimizer_args = ["lr=1e-5", "weight_decay=0.1"]"#),
            "{full}"
        );
    }

    #[test]
    fn training_and_saving_ext_keys_emitted_in_both_builders() {
        let recipe = RecipeData {
            prior_loss_weight: Some(0.5),
            loss_type: Some("huber".into()),
            lr_scheduler_num_cycles: Some(3),
            full_fp16: Some(true),
            full_bf16: Some(false),
            no_half_vae: Some(true),
            xformers: Some(true),
            lowram: Some(true),
            save_precision: Some("bf16".into()),
            save_last_n_epochs_state: Some(3),
            ..RecipeData::default()
        };
        let toml = build_sdscripts_toml(&recipe, ModelFamily::Sdxl1, &paths());
        for key in [
            "prior_loss_weight = 0.5",
            "loss_type = \"huber\"",
            "lr_scheduler_num_cycles = 3",
            "full_fp16 = true",
            "full_bf16 = false",
            "no_half_vae = true",
            "xformers = true",
            "lowram = true",
            "[saving]",
            "save_model_as = \"safetensors\"",
            "save_precision = \"bf16\"",
            "save_last_n_epochs_state = 3",
        ] {
            assert!(toml.contains(key), "TOML 缺少 {key}:\n{toml}");
        }
        let full = build_sdscripts_toml_full(&recipe, ModelFamily::Sdxl1, &paths(), false);
        for key in [
            "prior_loss_weight = 0.5",
            "loss_type = \"huber\"",
            "save_precision = \"bf16\"",
            "save_last_n_epochs_state = 3",
            "[saving]",
        ] {
            assert!(full.contains(key), "full TOML 缺少 {key}:\n{full}");
        }
    }

    #[test]
    fn network_ext_keys_emitted_in_lora_and_full() {
        let recipe = RecipeData {
            network_weights: Some(r"D:\weights\base-lora.safetensors".into()),
            scale_weight_norms: Some(1.0),
            network_args_custom: vec!["conv_dim=32".into()],
            ..RecipeData::default()
        };
        let toml = build_sdscripts_toml(&recipe, ModelFamily::Sdxl1, &paths());
        assert!(
            toml.contains("network_weights = \"D:/weights/base-lora.safetensors\""),
            "{toml}"
        );
        assert!(toml.contains("scale_weight_norms = 1"), "{toml}");
        // 全量版：配置了网络扩展时才输出 [network] 段
        let full = build_sdscripts_toml_full(&recipe, ModelFamily::Sdxl1, &paths(), false);
        assert!(full.contains("[network]"), "{full}");
        assert!(
            full.contains("network_weights = \"D:/weights/base-lora.safetensors\""),
            "{full}"
        );
        let full_default =
            build_sdscripts_toml_full(&RecipeData::default(), ModelFamily::Sdxl1, &paths(), false);
        assert!(!full_default.contains("[network]"), "{full_default}");
    }

    #[test]
    fn te_training_linkage_uses_recipe_lr_and_unet_only_false() {
        // 开启 TE 训练：network_train_unet_only=false，text_encoder_lr 用丹方值
        let recipe = RecipeData {
            train_text_encoder: Some(true),
            text_encoder_lr: Some(2e-5),
            ..RecipeData::default()
        };
        let toml = build_sdscripts_toml(&recipe, ModelFamily::Sdxl1, &paths());
        assert!(toml.contains("network_train_unet_only = false"), "{toml}");
        assert!(toml.contains("text_encoder_lr = 0.00002"), "{toml}");
        assert!(!toml.contains("network_train_unet_only = true"), "{toml}");
        // 未指定 TE lr → 不输出 text_encoder_lr（跟随主学习率）
        let recipe2 = RecipeData {
            train_text_encoder: Some(true),
            text_encoder_lr: None,
            ..RecipeData::default()
        };
        let toml2 = build_sdscripts_toml(&recipe2, ModelFamily::Sdxl1, &paths());
        assert!(toml2.contains("network_train_unet_only = false"), "{toml2}");
        assert!(
            !toml2.contains("text_encoder_lr"),
            "未指定 TE lr 不应输出 text_encoder_lr：\n{toml2}"
        );
        // 未开启 → 保持现状：缓存 TE 输出时冻结 TE
        let toml3 = build_sdscripts_toml(&RecipeData::default(), ModelFamily::Sdxl1, &paths());
        assert!(toml3.contains("network_train_unet_only = true"), "{toml3}");
        assert!(toml3.contains("text_encoder_lr = 0"), "{toml3}");
        // 全量版联动：开启时输出 network_train_unet_only=false + 丹方 TE lr
        let full = build_sdscripts_toml_full(&recipe, ModelFamily::Sdxl1, &paths(), true);
        assert!(full.contains("network_train_unet_only = false"), "{full}");
        assert!(full.contains("text_encoder_lr = 0.00002"), "{full}");
        // 全量版未开启 → 维持现状（text_encoder_lr=0，无 network_train_unet_only）
        let full_off =
            build_sdscripts_toml_full(&RecipeData::default(), ModelFamily::Sdxl1, &paths(), false);
        assert!(full_off.contains("text_encoder_lr = 0"), "{full_off}");
        assert!(!full_off.contains("network_train_unet_only"), "{full_off}");
    }

    #[test]
    fn none_fields_not_emitted() {
        // 默认丹方（全部新字段 None/空）不应输出任何扩展键
        let toml = build_sdscripts_toml(&RecipeData::default(), ModelFamily::Sdxl1, &paths());
        for key in [
            "reg_data_dir",
            "prior_loss_weight",
            "min_bucket_reso",
            "max_bucket_reso",
            "bucket_reso_steps",
            "bucket_no_upscale",
            "weighted_captions",
            "caption_dropout_every_n_epochs",
            "caption_tag_dropout_rate",
            "network_weights",
            "scale_weight_norms",
            "network_args",
            "loss_type",
            "lr_scheduler_num_cycles",
            "optimizer_args",
            "save_precision",
            "save_last_n_epochs_state",
            "full_fp16",
            "full_bf16",
            "no_half_vae",
            "xformers",
            "lowram",
            "persistent_data_loader_workers",
            "vae_batch_size",
        ] {
            assert!(!toml.contains(key), "默认丹方不应输出 {key}:\n{toml}");
        }
        let full =
            build_sdscripts_toml_full(&RecipeData::default(), ModelFamily::Sdxl1, &paths(), false);
        for key in [
            "reg_data_dir",
            "min_bucket_reso",
            "network_args",
            "optimizer_args",
            "save_precision",
            "save_last_n_epochs_state",
        ] {
            assert!(!full.contains(key), "full 默认丹方不应输出 {key}:\n{full}");
        }
    }

    #[test]
    fn full_sampling_emits_steps_scale_and_negative_prompt() {
        let recipe = RecipeData {
            sample_steps: Some(28),
            guidance_scale: Some(6.5),
            negative_prompt: Some("bad quality".into()),
            ..RecipeData::default()
        };
        let full = build_sdscripts_toml_full(&recipe, ModelFamily::Sdxl1, &paths(), false);
        assert!(full.contains("sample_steps = 28"), "{full}");
        assert!(full.contains("guidance_scale = 6.5"), "{full}");
        assert!(full.contains("negative_prompt = \"bad quality\""), "{full}");
    }
}
