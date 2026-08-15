#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""天地熔炉 · Python 内核适配层（kernel_runner.py）

职责（docs/architecture.md §5，ADR-001）：
- Rust 侧 ←JSON Lines(stdout)→ 本脚本 ←subprocess→ sd-scripts 训练脚本（或 mock）
- 事件通道：stdout 输出 JSON Lines（hello/progress/metric/log/sample/done/fail）
- 控制通道：stdin 接收 JSON Lines 命令（cancel）
- 原始训练日志转发到 stderr（Rust 侧落文件），进度行解析后发结构化事件

用法：
    python kernel_runner.py <任务配置.toml> [--mock] [--train-script sdxl_train_network.py]

环境变量：
    TIANDI_MOCK_TOTAL    mock 总步数（默认 60）
    TIANDI_MOCK_INTERVAL mock 步间隔毫秒（默认 200）
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time

# ---------------------------------------------------------------- 事件通道

def emit(event: dict) -> None:
    """向 stdout 输出一个 JSON Lines 事件。

    用 UTF-8 字节直写管道：不经过 locale 编码器（Windows 管道默认 GBK，
    某些字符/缓冲组合会触发 OSError 22），Rust 侧按 UTF-8 解析。
    """
    line = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
    sys.stdout.buffer.write(line)
    sys.stdout.buffer.flush()


def log(msg: str, level: str = "info") -> None:
    emit({"type": "log", "level": level, "msg": msg})


# ---------------------------------------------------------------- mock 模式

def run_mock(config_path: str) -> None:
    """模拟训练：不依赖 torch/sd-scripts，用于协议联调与 UI 演示。"""
    # 控制通道：stdin 收到 cancel 立即退出（mock 无子进程可杀）
    def control_loop():
        for line in sys.stdin:
            try:
                msg = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if msg.get("cmd") == "cancel":
                os._exit(0)

    threading.Thread(target=control_loop, daemon=True).start()

    total = int(os.environ.get("TIANDI_MOCK_TOTAL", "60"))
    interval = float(os.environ.get("TIANDI_MOCK_INTERVAL", "0.2"))
    log(f"mock 训练开始：{total} 步 × {interval * 1000:.0f}ms（配置 {config_path}）")
    for step in range(1, total + 1):
        time.sleep(interval)
        loss = 1.0 / (step ** 0.5) + 0.05
        lr = 1e-4
        emit({
            "type": "progress",
            "run_id": os.environ.get("TIANDI_RUN_ID", ""),
            "step": step,
            "total": total,
            "epoch": step / total,
            "loss": round(loss, 6),
            "lr": lr,
            "eta_s": int((total - step) * interval),
        })
        emit({
            "type": "metric",
            "run_id": os.environ.get("TIANDI_RUN_ID", ""),
            "step": step,
            "loss": round(loss, 6),
            "lr": lr,
        })
        if step % 10 == 0:
            sample_dir = os.environ.get("TIANDI_SAMPLE_DIR", "samples")
            path = os.path.join(sample_dir, f"mock-step-{step:04d}.png")
            emit({"type": "sample", "run_id": os.environ.get("TIANDI_RUN_ID", ""), "path": path})
            log(f"采样出图（mock）step {step}")
    emit({"type": "done", "run_id": os.environ.get("TIANDI_RUN_ID", ""), "code": 0})


# ---------------------------------------------------------------- sd-scripts 模式

def run_sdscripts(config_path: str) -> None:
    """启动 accelerate launch <train_script> --config_file <toml>，解析 kohya 进度行。"""
    train_script = os.environ.get("TIANDI_TRAIN_SCRIPT", "sdxl_train_network.py")
    cmd = ["accelerate", "launch", train_script, "--config_file", config_path]
    log(f"启动内核：{' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        emit({"type": "fail", "code": 127, "tail": "accelerate 未找到：请先完成内核环境安装（tiandi kernel install）"})
        return

    # 控制通道：stdin 读取 Rust 命令
    def control_loop():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                cmd_msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if cmd_msg.get("cmd") == "cancel":
                log("收到取消指令，正在终止内核...")
                kill_tree(proc)

    threading.Thread(target=control_loop, daemon=True).start()

    tail_buf = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        # 原始日志转发到 stderr（Rust 侧统一落文件）
        print(line, file=sys.stderr, flush=True)
        tail_buf.append(line)
        if len(tail_buf) > 50:
            tail_buf.pop(0)
        progress = parse_kohya_progress(line)
        if progress:
            emit({"type": "progress", **progress})
        elif "saving checkpoint" in line.lower():
            log(line)

    code = proc.wait()
    if code == 0:
        emit({"type": "done", "code": 0})
    else:
        emit({"type": "fail", "code": code, "tail": "\n".join(tail_buf[-15:])})


def parse_kohya_progress(line: str):
    """解析 kohya/sd-scripts 进度行：
    steps: 12%|█▏     | 12/120 [00:10<01:30, 1.20it/s, loss=0.123]
    steps: 100%|█████| 120/120 [02:00<00:00, 1.00it/s]
    """
    if "steps:" not in line or "|" not in line:
        return None
    try:
        head, rest = line.split("|", 2)[1], line.split("|", 2)[2]
        cur, total = [int(x) for x in head.split("/")[:2]]
    except (ValueError, IndexError):
        return None
    event = {"step": cur, "epoch": 0.0}
    if total > 0:
        event["epoch"] = round(cur / total, 4)
    loss = None
    lr = None
    if "loss=" in rest:
        try:
            loss = float(rest.split("loss=")[1].split(",")[0].split("]")[0].strip())
        except (ValueError, IndexError):
            pass
    if loss is not None:
        event["loss"] = round(loss, 6)
        event["lr"] = 0.0  # sd-scripts 不逐行输出 lr；由采样/日志补充
    return event


# ---------------------------------------------------------------- 工具

def kill_tree(proc: subprocess.Popen) -> None:
    """终止进程树（Windows 用 taskkill /T，POSIX 用 SIGTERM）。"""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass


def main() -> None:
    if len(sys.argv) < 2:
        emit({"type": "fail", "code": 2, "tail": "用法：kernel_runner.py <config.toml> [--mock]"})
        sys.exit(2)
    config_path = sys.argv[1]
    mode = "mock" if "--mock" in sys.argv else os.environ.get("TIANDI_KERNEL_MODE", "sdscripts")
    run_id = os.environ.get("TIANDI_RUN_ID", "")
    emit({"type": "hello", "run_id": run_id, "backend": "kernel-runner", "version": "0.1.0", "mode": mode})
    log(f"内核适配层启动（mode={mode}，python={sys.version.split()[0]}）")
    try:
        if mode == "mock":
            run_mock(config_path)
        else:
            run_sdscripts(config_path)
    except Exception as exc:  # noqa: BLE001
        emit({"type": "fail", "code": 1, "tail": str(exc)})
        sys.exit(1)


if __name__ == "__main__":
    main()
