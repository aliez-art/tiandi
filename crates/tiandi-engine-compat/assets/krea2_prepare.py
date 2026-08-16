#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""天地熔炉 · Krea 2 训练资产本地化（krea2_prepare.py）

离线把用户提供的 Krea 2 单文件模型转换为训练内核可用形态：

输入（--model-dir，通常为 测试底模/krea2）：
  - krea2_raw_bf16.safetensors         Krea 2 MMDiT 主权重（25GB，原样使用）
  - qwen3vl_4b_bf16.safetensors        Qwen3-VL-4B 文本编码器（8.4GB，原样硬链接）
  - qwen_image_vae.safetensors         Qwen-Image VAE（242MB，重命名映射）

输出（--out-root，通常为 <workspace>/.kernel-ws/models/krea2/）：
  - mmdit 原路径（kernel.json 记录）
  - qwen3vl_4b/  transformers 目录（config.json + model.safetensors + tokenizer*）
  - vae/          diffusers 目录（config.json + diffusion_pytorch_model.safetensors）

tokenizer 来源：Qwen3 系列共用词表（151936），从 sd-scripts configs/qwen3_06b 复用
（含 <|im_start|>/<|im_end|> 等 special tokens），并补齐 Qwen3 的 271 个
<|reserved_special_token_N|>，使总词表 == embed_tokens 行数。

用法：
    python krea2_prepare.py --model-dir <dir> --out-root <dir> [--tokenizer-dir <dir>]

无第三方依赖（仅标准库 + safetensors + torch 可选）。
"""

import argparse
import json
import os
import struct
import sys

# ---------------------------------------------------------------- VAE 映射

LATENTS_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517,
                1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497,
                0.2503, -0.2921]
LATENTS_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
               3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]

RES_MAP = {
    "norm1.gamma": "residual.0.gamma",
    "conv1.weight": "residual.2.weight",
    "conv1.bias": "residual.2.bias",
    "norm2.gamma": "residual.3.gamma",
    "conv2.weight": "residual.6.weight",
    "conv2.bias": "residual.6.bias",
    "conv_shortcut.weight": "shortcut.weight",
    "conv_shortcut.bias": "shortcut.bias",
}

import re  # noqa: E402

_UP_SRC = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11], 3: [12, 13, 14]}


def _vae_src_for(target):
    m = re.match(r"encoder\.down_blocks\.(\d+)\.(.*)$", target)
    if m:
        return f"encoder.downsamples.{int(m.group(1))}.{RES_MAP.get(m.group(2), m.group(2))}"
    m = re.match(r"encoder\.mid_block\.resnets\.(\d+)\.(.*)$", target)
    if m:
        j = 0 if int(m.group(1)) == 0 else 2
        return f"encoder.middle.{j}.{RES_MAP.get(m.group(2), m.group(2))}"
    m = re.match(r"encoder\.mid_block\.attentions\.0\.(.*)$", target)
    if m:
        return f"encoder.middle.1.{m.group(1)}"
    m = re.match(r"decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.*)$", target)
    if m:
        src_i = _UP_SRC[int(m.group(1))][int(m.group(2))]
        return f"decoder.upsamples.{src_i}.{RES_MAP.get(m.group(3), m.group(3))}"
    m = re.match(r"decoder\.up_blocks\.(\d+)\.upsamplers\.0\.(.*)$", target)
    if m:
        return f"decoder.upsamples.{_UP_SRC[int(m.group(1))][3]}.{m.group(2)}"
    m = re.match(r"decoder\.mid_block\.resnets\.(\d+)\.(.*)$", target)
    if m:
        j = 0 if int(m.group(1)) == 0 else 2
        return f"decoder.middle.{j}.{RES_MAP.get(m.group(2), m.group(2))}"
    m = re.match(r"decoder\.mid_block\.attentions\.0\.(.*)$", target)
    if m:
        return f"decoder.middle.1.{m.group(1)}"
    direct = {
        "encoder.conv_in.weight": "encoder.conv1.weight",
        "encoder.conv_in.bias": "encoder.conv1.bias",
        "encoder.norm_out.gamma": "encoder.head.0.gamma",
        "encoder.conv_out.weight": "encoder.head.2.weight",
        "encoder.conv_out.bias": "encoder.head.2.bias",
        "quant_conv.weight": "conv1.weight",
        "quant_conv.bias": "conv1.bias",
        "post_quant_conv.weight": "conv2.weight",
        "post_quant_conv.bias": "conv2.bias",
        "decoder.conv_in.weight": "decoder.conv1.weight",
        "decoder.conv_in.bias": "decoder.conv1.bias",
        "decoder.norm_out.gamma": "decoder.head.0.gamma",
        "decoder.conv_out.weight": "decoder.head.2.weight",
        "decoder.conv_out.bias": "decoder.head.2.bias",
    }
    return direct.get(target)


def convert_vae(vae_src: str, out_dir: str) -> None:
    """原始 VAE → diffusers AutoencoderKLQwenImage 目录格式（194/194 映射）。"""
    from safetensors.torch import load_file, save_file

    # 目标 key 表：与 diffusers 0.36 AutoencoderKLQwenImage 默认实例化一致
    with open(os.path.join(os.path.dirname(__file__), "krea2_vae_targets.json"), encoding="utf-8") as f:
        targets = json.load(f)
    sd = load_file(vae_src)
    mapping = {}
    for t in targets:
        s = _vae_src_for(t)
        if s is None or s not in sd:
            raise RuntimeError(f"VAE 映射失败: {t} <- {s}")
        mapping[t] = s
    unused = set(sd) - set(mapping.values())
    if unused:
        raise RuntimeError(f"VAE 未消费 key: {sorted(unused)[:10]}")
    out = {t: sd[s] for t, s in mapping.items()}
    os.makedirs(out_dir, exist_ok=True)
    cfg = {
        "_class_name": "AutoencoderKLQwenImage",
        "attn_scales": [],
        "base_dim": 96,
        "dim_mult": [1, 2, 4, 4],
        "dropout": 0.0,
        "latents_mean": LATENTS_MEAN,
        "latents_std": LATENTS_STD,
        "num_res_blocks": 2,
        "temperal_downsample": [False, True, True],
        "z_dim": 16,
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    save_file(out, os.path.join(out_dir, "diffusion_pytorch_model.safetensors"))
    print(f"  VAE -> {out_dir} ({len(out)} tensors)")


# ---------------------------------------------------------------- TE 构建

def build_te(te_src: str, tok_dir: str, out_dir: str) -> None:
    """单文件 Qwen3-VL-4B → transformers 目录格式（config + 硬链接权重 + tokenizer）。"""
    text_config = {
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": True,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 9728,
        "max_position_embeddings": 32768,
        "mlp_bias": False,
        "model_type": "qwen3",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "pad_token_id": 151643,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "vocab_size": 151936,
    }
    vision_config = {
        "depth": 24,
        "embed_dim": 1024,
        "fullatt_block_indexes": [5, 11, 17],
        "hidden_size": 1024,
        "image_size": 2304,
        "intermediate_size": 4096,
        "model_type": "qwen3_vision",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "rope_theta": 10000.0,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "window_size": 14,
        "deepstack_visual_indexes": [5, 11, 17],
    }
    config = {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "model_type": "qwen3_vl",
        "pad_token_id": 151643,
        "text_config": text_config,
        "vision_config": vision_config,
        "torch_dtype": "bfloat16",
        "transformers_version": "5.5.3",
        "tie_word_embeddings": True,
    }
    tokenizer_config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "bos_token": "<|endoftext|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|endoftext|>",
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "tokenizer_class": "Qwen2TokenizerFast",
        "unk_token": "<|endoftext|>",
    }
    os.makedirs(out_dir, exist_ok=True)
    # 权重：原样硬链接（8.4GB 不复制）
    dst_model = os.path.join(out_dir, "model.safetensors")
    if not os.path.exists(dst_model):
        os.link(te_src, dst_model)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    # tokenizer：Qwen3 词表 + 补齐 reserved tokens 至 151936
    import shutil
    for name in ["vocab.json", "merges.txt", "tokenizer.json"]:
        src = os.path.join(tok_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, name))
    tok_json_path = os.path.join(out_dir, "tokenizer.json")
    vocab_path = os.path.join(out_dir, "vocab.json")
    with open(tok_json_path, encoding="utf-8") as f:
        tj = json.load(f)
    with open(vocab_path, encoding="utf-8") as f:
        v = json.load(f)
    added = tj["added_tokens"]
    base = len(tj["model"]["vocab"]) + len(added)
    if base < 151936:
        max_id = max(t["id"] for t in added)
        n = 151936 - base
        for i in range(max_id + 1, 151936):
            name = f"<|reserved_special_token_{i - max_id - 1}|>"
            added.append({"id": i, "content": name, "single_word": False,
                          "lstrip": False, "rstrip": False, "normalized": False,
                          "special": True})
            v[name] = i
        tj["added_tokens"] = added
        with open(tok_json_path, "w", encoding="utf-8") as f:
            json.dump(tj, f, ensure_ascii=False)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(v, f, ensure_ascii=False)
        print(f"  tokenizer 补齐 {n} 个 reserved tokens（共 {base + n}）")
    print(f"  TE -> {out_dir}")


# ---------------------------------------------------------------- 入口

def find_in(model_dir: str, needles: list) -> str:
    for root, dirs, files in os.walk(model_dir):
        dirs[:] = [d for d in dirs if d not in ("thumbs", ".cache")]
        for f in files:
            if not f.endswith(".safetensors"):
                continue
            low = f.lower()
            if any(n in low for n in needles):
                return os.path.join(root, f)
    raise FileNotFoundError(f"在 {model_dir} 未找到包含 {needles} 的 safetensors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Krea 2 模型目录（含三个单文件）")
    ap.add_argument("--out-root", required=True, help="输出根（<workspace>/.kernel-ws/models/krea2）")
    ap.add_argument("--tokenizer-dir", default=None, help="Qwen3 tokenizer 目录（默认 sd-scripts configs/qwen3_06b）")
    args = ap.parse_args()

    mmdit = find_in(args.model_dir, ["krea2", "raw"])
    te = find_in(args.model_dir, ["qwen3vl_4b"])
    vae = find_in(args.model_dir, ["qwen_image_vae"])
    print(f"MMDiT: {mmdit}")
    print(f"TE:    {te}")
    print(f"VAE:   {vae}")

    out_root = args.out_root
    convert_vae(vae, os.path.join(out_root, "vae"))
    tok_dir = args.tokenizer_dir
    if tok_dir is None:
        # 默认：<out_root 上两级>/.kernel/sd-scripts/configs/qwen3_06b
        ws = os.path.dirname(os.path.dirname(out_root))
        tok_dir = os.path.join(ws, ".kernel", "sd-scripts", "configs", "qwen3_06b")
    build_te(te, tok_dir, os.path.join(out_root, "qwen3vl_4b"))

    # 输出 JSON 清单（kernel.json 的 krea2 字段）
    manifest = {
        "mmdit": mmdit.replace("\\", "/"),
        "text_encoder": os.path.join(out_root, "qwen3vl_4b").replace("\\", "/"),
        "vae_root": out_root.replace("\\", "/"),
    }
    print(json.dumps(manifest, ensure_ascii=False))
    print("Krea 2 资产准备完成 OK")


if __name__ == "__main__":
    main()
