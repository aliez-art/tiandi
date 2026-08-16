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
import re
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


# ---------------------------------------------------------------- 心跳与采样监控

def heartbeat_loop(stop_event: threading.Event) -> None:
    """心跳：每 2s 发一次 heartbeat 事件。

    Rust 侧看门狗以 30s 无 stdout 输出判卡死；长静默阶段（checkpoint 落盘、
    采样等）靠心跳续命。stdout 直写与 emit() 一致（GIL 下 buffer.write+flush
    原子性可接受）。
    """
    while not stop_event.wait(2.0):
        emit({"type": "heartbeat"})


def sample_watcher(stop_event: threading.Event, sample_dir: str, run_id: str) -> None:
    """采样目录监控：新出现的 *.png/*.jpg 发 sample 事件（真实训练的出图通道）。

    Rust 侧已注入 TIANDI_SAMPLE_DIR（绝对路径），mock 模式也是绝对路径事件，
    保持一致：sample 事件 path 用绝对路径（supervisor 对绝对路径直接使用）。
    """
    if not sample_dir:
        return
    seen = set()
    try:
        os.makedirs(sample_dir, exist_ok=True)
        seen.update(os.listdir(sample_dir))
    except OSError:
        pass
    while True:
        scan_samples(sample_dir, seen, run_id)
        if stop_event.wait(2.0):
            # 终态前最后扫一遍，尽量补齐训练末尾落盘的采样
            scan_samples(sample_dir, seen, run_id)
            return


def scan_samples(sample_dir: str, seen: set, run_id: str) -> None:
    """列出采样目录中尚未上报的图片文件并逐张发 sample 事件。"""
    try:
        if not os.path.isdir(sample_dir):
            return
        for name in sorted(os.listdir(sample_dir)):
            if name in seen:
                continue
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            seen.add(name)
            path = os.path.join(sample_dir, name)
            emit({"type": "sample", "run_id": run_id, "path": path})
            log(f"采样出图：{path}")
    except OSError:
        pass


# ---------------------------------------------------------------- sd-scripts 模式

def run_sdscripts(config_path: str) -> None:
    """启动 accelerate launch <train_script> --config_file <toml>，解析 kohya 进度行。"""
    train_script = os.environ.get("TIANDI_TRAIN_SCRIPT", "sdxl_train_network.py")
    # accelerate 在 venv/Scripts 下（venv python 未激活时不在 PATH）
    scripts_dir = os.path.dirname(sys.executable)
    accelerate_exe = os.path.join(scripts_dir, "accelerate.exe" if os.name == "nt" else "accelerate")
    accelerate = accelerate_exe if os.path.exists(accelerate_exe) else "accelerate"
    cmd = [accelerate, "launch", train_script, "--config_file", config_path]
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
            # 关键：训练子进程不继承控制管道（Rust 侧 stdin 保持打开时，
            # 脚本内任何 stdin 读取都会永久阻塞；DEVNULL 保证读到 EOF）
            stdin=subprocess.DEVNULL,
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

    # 心跳（≥2s）与采样目录监控：Rust 侧看门狗/采样画廊的输入
    stop_event = threading.Event()
    threading.Thread(target=heartbeat_loop, args=(stop_event,), daemon=True).start()
    threading.Thread(
        target=sample_watcher,
        args=(stop_event, os.environ.get("TIANDI_SAMPLE_DIR", ""), os.environ.get("TIANDI_RUN_ID", "")),
        daemon=True,
    ).start()

    # 终态事件兜底：无论子进程如何退出，确保 done/fail 必然发出
    code = None
    tail_buf = []
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            # 原始日志转发到 stderr：UTF-8 字节直写（locale GBK 编码器对
            # tqdm 乱码符号会抛 OSError 22，errors=replace 兜底）
            sys.stderr.buffer.write((line + "\n").encode("utf-8", errors="replace"))
            sys.stderr.buffer.flush()
            tail_buf.append(line)
            if len(tail_buf) > 50:
                tail_buf.pop(0)
            progress = parse_kohya_progress(line)
            if progress:
                run_id = os.environ.get("TIANDI_RUN_ID", "")
                emit({"type": "progress", **progress})
                if "loss" in progress:
                    emit({
                        "type": "metric",
                        "run_id": run_id,
                        "step": progress.get("step", 0),
                        "loss": progress.get("loss"),
                        "lr": progress.get("lr"),
                    })
            elif "saving checkpoint" in line.lower():
                log(line)
        code = proc.wait()
    except Exception as exc:  # noqa: BLE001
        emit({"type": "fail", "code": 1, "tail": str(exc)})
        return
    finally:
        stop_event.set()
        if code is None:
            if proc.poll() is None:
                kill_tree(proc)
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
        # tqdm 行以 "|" 分隔：前缀、进度条、统计段（统计段形如 "1/30 [00:35<17:04, 35.34s/it]"）
        parts = line.split("|")
        seg = parts[2].strip().split()[0]  # "1/30"
        cur, total = [int(x) for x in seg.split("/")[:2]]
    except (ValueError, IndexError):
        return None
    event = {"step": cur, "total": total, "epoch": 0.0}
    if total > 0:
        event["epoch"] = round(cur / total, 4)
    loss = None
    lr = None
    if "loss=" in line:
        try:
            loss = float(line.split("loss=")[1].split(",")[0].split("]")[0].strip())
        except (ValueError, IndexError):
            pass
    if loss is not None:
        event["loss"] = round(loss, 6)
        event["lr"] = 0.0  # sd-scripts 不逐行输出 lr；由采样/日志补充
    return event


# ---------------------------------------------------------------- ai-toolkit 模式（Krea 2 / FLUX 等）

def parse_aitk_progress(line: str):
    """解析 ai-toolkit tqdm 进度行（stderr，非 tty 逐行输出）：
    "  12%|██        | 12/200 [00:30<08:20, 2.66it/s, lr: 1.0e-4 loss: 1.23e-1]"
    或 (postfix 多键) "loss: 1.2e-1 lr: 1.0e-4"
    tqdm 行以 "|" 分隔：desc+百分比、进度条本体、统计段（含 cur/total）。
    """
    if "|" not in line or "/" not in line:
        return None
    try:
        parts = line.split("|")
        seg = parts[2].strip().split()[0]  # "12/200"
        cur, total = [int(x) for x in seg.split("/")[:2]]
    except (ValueError, IndexError):
        return None
    event = {"step": cur, "total": total, "epoch": 0.0}
    if total > 0:
        event["epoch"] = round(cur / total, 4)
    # postfix：`lr: 1.0e-4 loss: 1.23e-1`（冒号分隔；kohya 用 loss= 已由 kohya 解析器处理）
    m = re.search(r"loss:\s*([0-9.eE+-]+|nan)", line)
    if m and m.group(1) != "nan":
        try:
            event["loss"] = round(float(m.group(1)), 6)
        except ValueError:
            pass
    m = re.search(r"lr:\s*([0-9.eE+-]+)", line)
    if m:
        try:
            event["lr"] = float(m.group(1))
        except ValueError:
            pass
    return event


def run_aitk(config_path: str) -> None:
    """启动 ai-toolkit：python run.py <config.yaml>（cwd=ai-toolkit 仓库）。

    事件：tqdm 行 → progress/metric；stdout 原文转发 stderr 并解析保存日志。
    """
    cmd = [sys.executable, "run.py", config_path]
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
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        emit({"type": "fail", "code": 127, "tail": "ai-toolkit 内核未找到：请先完成安装（tiandi kernel install --backend aitk）"})
        return

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

    # 心跳（≥2s）与采样目录监控：Rust 侧看门狗/采样画廊的输入
    stop_event = threading.Event()
    threading.Thread(target=heartbeat_loop, args=(stop_event,), daemon=True).start()
    threading.Thread(
        target=sample_watcher,
        args=(stop_event, os.environ.get("TIANDI_SAMPLE_DIR", ""), os.environ.get("TIANDI_RUN_ID", "")),
        daemon=True,
    ).start()

    code = None
    tail_buf = []
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            sys.stderr.buffer.write((line + "\n").encode("utf-8", errors="replace"))
            sys.stderr.buffer.flush()
            tail_buf.append(line)
            if len(tail_buf) > 50:
                tail_buf.pop(0)
            progress = parse_aitk_progress(line)
            if progress:
                run_id = os.environ.get("TIANDI_RUN_ID", "")
                emit({"type": "progress", **progress})
                if "loss" in progress:
                    emit({
                        "type": "metric",
                        "run_id": run_id,
                        "step": progress.get("step", 0),
                        "loss": progress.get("loss"),
                        "lr": progress.get("lr", 0.0),
                    })
            elif "Saved checkpoint" in line or "saving" in line.lower():
                log(line)
        code = proc.wait()
    except Exception as exc:  # noqa: BLE001
        emit({"type": "fail", "code": 1, "tail": str(exc)})
        return
    finally:
        stop_event.set()
        if code is None:
            if proc.poll() is None:
                kill_tree(proc)
            code = proc.wait()
    if code == 0:
        emit({"type": "done", "code": 0})
    else:
        emit({"type": "fail", "code": code, "tail": "\n".join(tail_buf[-15:])})


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
        elif mode == "aitk":
            run_aitk(config_path)
        else:
            run_sdscripts(config_path)
    except Exception as exc:  # noqa: BLE001
        emit({"type": "fail", "code": 1, "tail": str(exc)})
        sys.exit(1)


if __name__ == "__main__":
    main()

