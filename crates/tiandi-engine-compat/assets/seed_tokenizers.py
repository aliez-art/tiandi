# -*- coding: utf-8 -*-
"""将本地 tokenizer 目录种入 HF 缓存（离线化）。

sd-scripts 068bcd7 硬编码 TOKENIZER1_PATH/TOKENIZER2_PATH（HF repo id），
from_pretrained 会先查 HF 缓存：把文件按缓存结构放好即可完全离线。
"""
import hashlib
import os
import shutil
import sys

from huggingface_hub import constants
from huggingface_hub.file_download import repo_folder_name


def seed_cache(repo_id: str, local_dir: str) -> None:
    cache_dir = constants.HF_HUB_CACHE
    repo_folder = os.path.join(cache_dir, repo_folder_name(repo_id=repo_id, repo_type="model"))
    blobs = os.path.join(repo_folder, "blobs")
    snapshots = os.path.join(repo_folder, "snapshots")
    refs = os.path.join(repo_folder, "refs")
    commit = "0" * 40
    os.makedirs(blobs, exist_ok=True)
    os.makedirs(snapshots, exist_ok=True)
    os.makedirs(refs, exist_ok=True)
    snap = os.path.join(snapshots, commit)
    os.makedirs(snap, exist_ok=True)
    for name in os.listdir(local_dir):
        src = os.path.join(local_dir, name)
        if not os.path.isfile(src):
            continue
        with open(src, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        blob = os.path.join(blobs, sha)
        if not os.path.exists(blob):
            shutil.copy(src, blob)
        target = os.path.join(snap, name)
        if not os.path.exists(target):
            try:
                os.link(blob, target)
            except OSError:
                shutil.copy(blob, target)
    with open(os.path.join(refs, "main"), "w") as fh:
        fh.write(commit)
    print(f"seeded {repo_id} -> {repo_folder}")


if __name__ == "__main__":
    seed_cache("openai/clip-vit-large-patch14", sys.argv[1])
    seed_cache("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", sys.argv[2])
