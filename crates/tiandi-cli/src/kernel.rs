//! `tiandi kernel install`：真实训练内核引导。
//!
//! 步骤（docs/architecture.md §5.5 版本锁定）：
//! 1. 定位系统 Python（3.12+）
//! 2. 创建 `<workspace>/.kernel/venv`
//! 3. 安装 torch/torchvision（CUDA 12.8 wheels，Blackwell 必需）
//! 4. 克隆 sd-scripts 并锁定 commit（068bcd7，与 lora-scripts-next 验证一致）
//! 5. 安装 sd-scripts 依赖
//! 6. 写入内核清单 kernel.json（doctor 与引擎读取）

use std::path::{Path, PathBuf};
use std::process::Command;
/// sd-scripts 锁定 commit（与 lora-scripts-next vendor 快照一致，社区验证）。
pub const SD_SCRIPTS_COMMIT: &str = "068bcd7";
pub const SD_SCRIPTS_REPO: &str = "https://github.com/kohya-ss/sd-scripts.git";

pub fn cmd_kernel_install(workspace: &Path, torch_index: &str) {
    let kernel_dir = workspace.join(".kernel");
    let venv = kernel_dir.join("venv");
    let sd_scripts = kernel_dir.join("sd-scripts");
    let kernel_json = kernel_dir.join("kernel.json");

    if kernel_json.exists() {
        println!(
            "✓ 内核已安装（{}）。如需重装：删除 .kernel 目录后重试。",
            kernel_json.display()
        );
        return;
    }

    println!("== 天地熔炉 · 训练内核安装 ==");
    println!("工作区：{}", workspace.display());
    println!("内核目录：{}", kernel_dir.display());

    // 1. 定位 Python
    let python = locate_python();
    println!("\n[1/6] 使用 Python：{}", python.display());
    run(&[python.to_str().unwrap(), "--version"], None).expect("Python 不可用");

    // 2. venv
    println!("\n[2/6] 创建虚拟环境…");
    std::fs::create_dir_all(&kernel_dir).expect("创建 .kernel 目录");
    run(
        &[
            python.to_str().unwrap(),
            "-m",
            "venv",
            venv.to_str().unwrap(),
        ],
        None,
    )
    .expect("创建 venv 失败");
    let venv_python = venv_python_path(&venv);
    run(
        &[
            venv_python.to_str().unwrap(),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
        ],
        None,
    )
    .expect("升级 pip 失败");

    // 3. torch cu128
    println!(
        "\n[3/6] 安装 torch/torchvision（{}，约 3GB，请耐心等待）…",
        torch_index
    );
    run(
        &[
            venv_python.to_str().unwrap(),
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "--index-url",
            torch_index,
        ],
        None,
    )
    .expect("torch 安装失败");

    // 4. sd-scripts（锁 commit）
    println!(
        "\n[4/6] 克隆 sd-scripts（锁定 commit {}）…",
        SD_SCRIPTS_COMMIT
    );
    if !sd_scripts.exists() {
        run(
            &[
                "git",
                "clone",
                "--depth",
                "200",
                SD_SCRIPTS_REPO,
                sd_scripts.to_str().unwrap(),
            ],
            None,
        )
        .expect("克隆 sd-scripts 失败");
    }
    run(
        &[
            "git",
            "-C",
            sd_scripts.to_str().unwrap(),
            "checkout",
            SD_SCRIPTS_COMMIT,
        ],
        None,
    )
    .expect("锁定 sd-scripts commit 失败");

    // 5. 依赖（cwd = sd-scripts，`-e .` 相对依赖从该目录解析）
    println!("\n[5/6] 安装 sd-scripts 依赖…");
    let req = sd_scripts.join("requirements.txt");
    if !req.exists() {
        println!("⚠ 未找到 requirements.txt，跳过依赖安装（后续可手动 pip install -r）");
    } else {
        run_in(
            sd_scripts.as_path(),
            &[
                venv_python.to_str().unwrap(),
                "-m",
                "pip",
                "install",
                "-r",
                req.to_str().unwrap(),
            ],
            None,
        )
        .expect("依赖安装失败");
    }

    // 6. accelerate 默认配置（首次 launch 会进交互向导卡死，预生成）
    println!("\n[6/7] 生成 accelerate 默认配置…");
    let accelerate_exe = if cfg!(windows) {
        venv.join("Scripts/accelerate.exe")
    } else {
        venv.join("bin/accelerate")
    };
    if accelerate_exe.exists() {
        let _ = run(
            &[accelerate_exe.to_str().unwrap(), "config", "default"],
            None,
        );
    }

    // 7. 内核清单
    println!("\n[7/7] 写入内核清单…");
    let torch_ver = Command::new(&venv_python)
        .args(["-c", "import torch; print(torch.__version__)"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into());
    let json = serde_json::json!({
        "python": venv_python.to_string_lossy(),
        "sd_scripts": sd_scripts.to_string_lossy(),
        "commit": SD_SCRIPTS_COMMIT,
        "torch": torch_ver,
        "created_at": chrono::Utc::now().to_rfc3339(),
    });
    std::fs::write(&kernel_json, serde_json::to_string_pretty(&json).unwrap())
        .expect("写 kernel.json 失败");

    println!("\n✓ 内核安装完成！");
    println!("  torch: {torch_ver}");
    println!("  sd-scripts: {}", sd_scripts.display());
    println!("  清单: {}", kernel_json.display());
    println!("下一步：注册基底模型（tiandi models add 或 UI），然后创建炼丹任务。");
}

/// ai-toolkit 后端（M3：Krea 2 等 DiT 模型）独立 venv + 仓库安装。
pub const AI_TOOLKIT_REPO: &str = "https://github.com/ostris/ai-toolkit.git";

pub fn cmd_kernel_install_aitk(workspace: &Path, torch_index: &str) {
    let kernel_dir = workspace.join(".kernel-aitk");
    let venv = kernel_dir.join("venv");
    let repo = kernel_dir.join("ai-toolkit");
    let kernel_json = workspace.join(".kernel/kernel.json");

    if !kernel_json.exists() {
        eprintln!(
            "✗ 请先安装 sd-scripts 内核（tiandi kernel install），ai-toolkit 清单会合并到同一 kernel.json"
        );
        std::process::exit(1);
    }

    println!("== 天地熔炉 · ai-toolkit 内核安装（Krea 2 后端）==");
    println!("venv：{}", venv.display());
    println!("仓库：{}", repo.display());

    // 1. venv（与 sd-scripts 内核隔离，transformers 5.x 互不干扰；已存在则复用）
    let python = locate_python();
    println!("\n[1/5] 创建虚拟环境…");
    let venv_python = venv_python_path(&venv);
    if venv_python.exists() {
        println!("  venv 已存在（复用）：{}", venv.display());
    } else {
        std::fs::create_dir_all(&kernel_dir).expect("创建 .kernel-aitk 目录");
        run(
            &[python.to_str().unwrap(), "-m", "venv", venv.to_str().unwrap()],
            None,
        )
        .expect("创建 venv 失败");
    }
    run(
        &[
            venv_python.to_str().unwrap(),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
        ],
        None,
    )
    .expect("升级 pip 失败");

    // 2. torch cu128（torchaudio 为 ai-toolkit 必需：toolkit/config_modules import）
    println!("\n[2/5] 安装 torch/torchvision/torchaudio（{torch_index}，约 3GB）…");
    run(
        &[
            venv_python.to_str().unwrap(),
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            torch_index,
        ],
        None,
    )
    .expect("torch 安装失败");

    // 3. 克隆 ai-toolkit
    println!("\n[3/5] 克隆 ai-toolkit…");
    if !repo.exists() {
        run(&["git", "clone", "--depth", "1", AI_TOOLKIT_REPO, repo.to_str().unwrap()], None)
            .expect("克隆 ai-toolkit 失败");
    }

    // 4. 依赖（requirements.txt 内含 git+diffusers 固定 commit，需联网）
    println!("\n[4/5] 安装 ai-toolkit 依赖（含 diffusers git commit，请耐心等待）…");
    let req = repo.join("requirements.txt");
    if !req.exists() {
        eprintln!("✗ 未找到 ai-toolkit requirements.txt");
        std::process::exit(1);
    }
    run(
        &[
            venv_python.to_str().unwrap(),
            "-m",
            "pip",
            "install",
            "-r",
            req.to_str().unwrap(),
        ],
        None,
    )
    .expect("依赖安装失败");

    // 5. 合并写 kernel.json（保留 sd-scripts 字段）
    println!("\n[5/5] 合并内核清单…");
    let torch_ver = Command::new(&venv_python)
        .args(["-c", "import torch; print(torch.__version__)"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into());
    let mut json: serde_json::Value = std::fs::read_to_string(&kernel_json)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    json["ai_toolkit"] = serde_json::json!({
        "python": venv_python.to_string_lossy(),
        "repo": repo.to_string_lossy(),
        "torch": torch_ver,
    });
    std::fs::write(&kernel_json, serde_json::to_string_pretty(&json).unwrap())
        .expect("写 kernel.json 失败");

    println!("\n✓ ai-toolkit 内核安装完成！");
    println!("下一步：tiandi kernel prepare-krea2 <底模目录>（离线转换 TE/VAE 资产）。");
}

/// Krea 2 资产准备：单文件模型 → TE/VAE 本地化目录 + 写 kernel.json krea2 字段。
pub fn cmd_prepare_krea2(workspace: &Path, model_dir: &Path) {
    let kernel_json = workspace.join(".kernel/kernel.json");
    let json_text = std::fs::read_to_string(&kernel_json)
        .expect("kernel.json 不存在（请先 tiandi kernel install）");
    let mut json: serde_json::Value =
        serde_json::from_str(&json_text).expect("kernel.json 解析失败");
    if json.get("ai_toolkit").is_none() {
        eprintln!("✗ kernel.json 无 ai_toolkit 字段（请先 tiandi kernel install-aitk）");
        std::process::exit(1);
    }
    let aitk_python = json["ai_toolkit"]["python"]
        .as_str()
        .expect("ai_toolkit.python 缺失")
        .to_string();
    let sd_scripts = json.get("sd_scripts").and_then(|s| s.as_str()).map(String::from);

    println!("== 天地熔炉 · Krea 2 资产准备 ==");
    println!("模型目录：{}", model_dir.display());
    // 资产输出：<workspace>/models/krea2（workspace 即数据目录，如 .kernel-ws）
    let out_root = workspace.join("models/krea2");

    // krea2_prepare.py 与 kernel_runner.py 同目录（assets 随 crate 分发）
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("tiandi-engine-compat/assets/krea2_prepare.py");
    let mut args: Vec<String> = vec![
        script.to_str().unwrap().to_string(),
        "--model-dir".into(),
        model_dir.to_str().unwrap().to_string(),
        "--out-root".into(),
        out_root.to_str().unwrap().to_string(),
    ];
    if let Some(sd) = &sd_scripts {
        args.push("--tokenizer-dir".into());
        args.push(format!("{}\\configs\\qwen3_06b", sd));
    }
    // 输出 JSON 清单（最后一行 stdout）
    let output = Command::new(&aitk_python)
        .args(&args)
        .output()
        .expect("krea2_prepare.py 执行失败");
    println!("{}", String::from_utf8_lossy(&output.stdout));
    if !output.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        eprintln!("✗ Krea 2 资产准备失败");
        std::process::exit(1);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let manifest_line = stdout
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))
        .expect("未找到 krea2 清单输出");
    let krea2: serde_json::Value =
        serde_json::from_str(manifest_line).expect("krea2 清单解析失败");

    json["ai_toolkit"]["krea2"] = krea2;
    std::fs::write(&kernel_json, serde_json::to_string_pretty(&json).unwrap())
        .expect("写 kernel.json 失败");
    println!("✓ Krea 2 资产就绪，kernel.json 已更新（ai_toolkit.krea2）");
    println!("下一步：注册 Krea 2 模型（tiandi models add --family dit_krea2）后即可开炉。");
}

/// 定位系统 Python。
fn locate_python() -> PathBuf {    // 1. PATH
    if Command::new("python")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        return PathBuf::from("python");
    }
    // 2. LOCALAPPDATA（python.org 默认）
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        for ver in ["Python312", "Python313", "Python311"] {
            let p = Path::new(&local).join(format!("Programs/Python/{ver}/python.exe"));
            if p.exists() {
                return p;
            }
        }
    }
    eprintln!("✗ 未检测到 Python 3.12+。请先安装：https://www.python.org/downloads/");
    std::process::exit(1);
}

fn venv_python_path(venv: &Path) -> PathBuf {
    if cfg!(windows) {
        venv.join("Scripts/python.exe")
    } else {
        venv.join("bin/python")
    }
}

fn run(args: &[&str], label: Option<&str>) -> Result<(), ()> {
    run_in(&std::env::current_dir().unwrap_or_default(), args, label)
}

/// 在指定目录执行命令（`-e .` 之类的相对依赖从该目录解析）。
fn run_in(dir: &Path, args: &[&str], label: Option<&str>) -> Result<(), ()> {
    if let Some(l) = label {
        println!("  {l}");
    }
    let mut cmd = Command::new(args[0]);
    cmd.args(&args[1..]).current_dir(dir);
    let mut child = cmd
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .map_err(|e| {
            eprintln!("✗ 命令启动失败：{e}");
        })?;
    let status = child.wait().map_err(|e| {
        eprintln!("✗ 命令执行失败：{e}");
    })?;
    if !status.success() {
        eprintln!("✗ 命令退出码：{}", status.code().unwrap_or(-1));
        return Err(());
    }
    Ok(())
}
