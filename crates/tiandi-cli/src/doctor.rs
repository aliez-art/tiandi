//! `tiandi doctor`：环境体检（lora-scripts-next preflight 的 Rust 版，PRD FR-902）。

use std::process::Command;

use sysinfo::{Disks, System};

pub fn run() {
    let mut all_ok = true;
    let mut warn = |ok: bool, msg: String| {
        all_ok &= ok;
        println!("{} {msg}", if ok { "✓" } else { "✗" });
    };

    println!("== 天地熔炉 环境体检 ==");

    // 1. 工作区
    let cwd = std::env::current_dir().unwrap_or_default();
    let db = cwd.join("tiandi.db");
    let is_workspace = db.exists()
        && ["models", "datasets", "recipes", "runs", "vault"]
            .iter()
            .all(|d| cwd.join(d).is_dir());
    warn(
        is_workspace,
        format!(
            "工作区：{}（{}）",
            cwd.display(),
            if is_workspace {
                "结构完整"
            } else {
                "未初始化，请运行 tiandi init"
            }
        ),
    );

    // 2. 磁盘
    let disks = Disks::new_with_refreshed_list();
    let mut disk_ok = true;
    let mut disk_lines = Vec::new();
    for d in disks.list() {
        let avail_gb = d.available_space() as f64 / (1024.0 * 1024.0 * 1024.0);
        let total_gb = d.total_space() as f64 / (1024.0 * 1024.0 * 1024.0);
        let low = avail_gb < 20.0;
        disk_ok &= !low;
        disk_lines.push(format!(
            "  - {} 可用 {:.1} GB / 共 {:.1} GB{}",
            d.mount_point().display(),
            avail_gb,
            total_gb,
            if low { "（⚠ 余量偏低）" } else { "" }
        ));
    }
    warn(disk_ok, "磁盘空间：".to_string());
    for l in disk_lines {
        println!("{l}");
    }

    // 3. 内存
    let mut sys = System::new();
    sys.refresh_memory();
    let mem_ok = sys.available_memory() > 8 * 1024 * 1024 * 1024;
    warn(
        mem_ok,
        format!(
            "内存：可用 {:.1} GB / 共 {:.1} GB",
            sys.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0),
            sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0)
        ),
    );

    // 4. GPU / CUDA
    let gpu = nvidia_smi();
    match &gpu {
        Some(info) => warn(true, format!("GPU（CUDA）：{info}")),
        None => {
            warn(
                false,
                "GPU（CUDA）：未检测到 nvidia-smi —— 训练需要 NVIDIA GPU + 驱动（训练内核安装时可重检）".to_string(),
            );
        }
    }

    // 5. 训练内核（venv + torch + sd-scripts）
    let kernel_dir = cwd.join(".kernel");
    let kernel_json = kernel_dir.join("kernel.json");
    let kernel_status = if kernel_json.exists() {
        match std::fs::read_to_string(&kernel_json) {
            Ok(text) => match serde_json::from_str::<serde_json::Value>(&text) {
                Ok(v) => {
                    let torch = v.get("torch").and_then(|t| t.as_str()).unwrap_or("?");
                    let commit = v.get("commit").and_then(|c| c.as_str()).unwrap_or("?");
                    let venv_py = v.get("python").and_then(|p| p.as_str()).unwrap_or("");
                    // 探测 CUDA 可用性（短超时）
                    let cuda = if !venv_py.is_empty() {
                        std::process::Command::new(venv_py)
                            .args(["-c", "import torch; print(torch.cuda.is_available())"])
                            .output()
                            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                            .unwrap_or_else(|_| "未知".into())
                    } else {
                        "未知".into()
                    };
                    format!("已安装（torch {torch}，commit {commit}，cuda={cuda}）")
                }
                Err(_) => "已安装（清单损坏）".into(),
            },
            Err(_) => "已安装（清单不可读）".into(),
        }
    } else {
        "未安装（运行 `tiandi kernel install` 引导：venv + torch cu128 + sd-scripts）".into()
    };
    let kernel_ok = kernel_status.starts_with("已安装") && kernel_status.contains("cuda=True");
    warn(kernel_ok, format!("训练内核：{kernel_status}"));

    // 6. 端口
    let port_free = check_port(18765);
    warn(
        port_free,
        format!(
            "端口 18765：{}",
            if port_free {
                "空闲"
            } else {
                "被占用（tiandi server 可换 --port）"
            }
        ),
    );

    println!();
    if all_ok {
        println!("✓ 体检全部通过，可以开炉炼丹。");
    } else {
        println!("✗ 体检存在未通过项（见上）。训练必需项：GPU/CUDA；其余项可带警告继续。");
    }
}

/// 查询 nvidia-smi 摘要（名称/显存/驱动）。
fn nvidia_smi() -> Option<String> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used,driver_version",
            "--format=csv,noheader",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() {
        return None;
    }
    Some(text.replace('\n', " | "))
}

/// 检查端口是否可绑定（通过尝试监听判断）。
fn check_port(port: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", port)).is_ok()
}
