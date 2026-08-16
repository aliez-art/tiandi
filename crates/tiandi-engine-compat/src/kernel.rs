//! Python 内核进程监督：spawn wrapper → 解析 JSON Lines 事件 → EventBus；
//! stdin 控制通道（cancel）；Windows kill-tree。

use std::path::{Path, PathBuf};
use std::process::Stdio;

use tiandi_core::{Event, EventBus};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;

/// 内核句柄：持有子进程与写端，可取消。
pub struct KernelHandle {
    child: tokio::process::Child,
    stdin: Option<tokio::process::ChildStdin>,
}

impl KernelHandle {
    /// 发送控制命令（JSON Lines over stdin）。
    pub async fn send_command(&mut self, cmd: &str) -> Result<(), KernelError> {
        let stdin = self.stdin.as_mut().ok_or(KernelError::StdinClosed)?;
        stdin.write_all(format!("{cmd}\n").as_bytes()).await?;
        stdin.flush().await?;
        Ok(())
    }

    /// 请求取消：先发命令等内核优雅退出（5s），超时则强杀进程树。
    pub async fn cancel(&mut self) -> Result<(), KernelError> {
        if self.send_command(r#"{"cmd":"cancel"}"#).await.is_err() {
            return Ok(()); // stdin 已关闭（内核已退出）
        }
        match tokio::time::timeout(std::time::Duration::from_secs(5), self.child.wait()).await {
            Ok(_) => Ok(()),
            Err(_) => {
                self.kill();
                Err(KernelError::CancelTimeout)
            }
        }
    }

    /// 等待内核进程退出（返回退出状态）。
    pub async fn child_wait(&mut self) -> Option<std::process::ExitStatus> {
        self.child.wait().await.ok()
    }

    /// 强制终止进程树。
    pub fn kill(&mut self) {
        if let Some(id) = self.child.id() {
            let _ = kill_tree(id);
        }
    }
}

/// 终止进程树（Windows taskkill /T /F；POSIX kill -TERM）。
pub fn kill_tree(pid: u32) -> std::io::Result<()> {
    let mut cmd = std::process::Command::new(if cfg!(windows) { "taskkill" } else { "kill" });
    let status = if cfg!(windows) {
        cmd.args(["/PID", &pid.to_string(), "/T", "/F"])
    } else {
        cmd.args(["-TERM", &pid.to_string()])
    }
    .stdout(std::process::Stdio::null())
    .stderr(std::process::Stdio::null())
    .status();
    status.map(|_| ())
}

/// 内核启动配置。
pub struct KernelLaunch {
    pub python: PathBuf,
    pub wrapper: PathBuf,
    pub config_path: PathBuf,
    pub mode: KernelMode,
    pub env: Vec<(String, String)>,
    pub cwd: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelMode {
    Mock,
    SdScripts,
    Aitk,
    Tagger,
}

impl KernelMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Mock => "mock",
            Self::SdScripts => "sdscripts",
            Self::Aitk => "aitk",
            Self::Tagger => "tagger",
        }
    }
}

/// 启动内核进程并接管事件流。
///
/// `on_event` 在每个解析出的 JSON 事件上调用（由调用方转 EventBus）。
/// 返回句柄；stdout/stderr 读取在后台任务进行。
pub fn spawn_kernel(
    launch: &KernelLaunch,
    on_event: impl Fn(serde_json::Value) + Send + 'static,
) -> Result<KernelHandle, KernelError> {
    let mut cmd = Command::new(&launch.python);
    cmd.arg(&launch.wrapper)
        .arg(&launch.config_path)
        .current_dir(&launch.cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("TIANDI_KERNEL_MODE", launch.mode.as_str());
    for (k, v) in &launch.env {
        cmd.env(k, v);
    }

    let mut child: tokio::process::Child = cmd.spawn().map_err(KernelError::Spawn)?;
    let stdout = child.stdout.take().ok_or(KernelError::StdoutClosed)?;
    let stderr = child.stderr.take().ok_or(KernelError::StderrClosed)?;
    let stdin = child.stdin.take();

    let handle = KernelHandle { child, stdin };

    // stdout → 事件（逐行 JSON）；EOF 兜底：进程退出但无终态事件（崩溃/被杀）→ 发 Fail
    // 注意：Windows 下内核子进程可能输出 GBK 字节（tqdm 等），用字节级读取 + lossy 解码，
    // 避免 lines() 遇非法 UTF-8 提前退出导致管道不再被读（wrapper 写阻塞）
    let terminated = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let term_flag = terminated.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);
        let mut buf: Vec<u8> = Vec::with_capacity(1024);
        loop {
            buf.clear();
            match reader.read_until(b'\n', &mut buf).await {
                Ok(0) => break,
                Ok(_) => {
                    let line = String::from_utf8_lossy(&buf).trim().to_string();
                    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
                        if let Some(t) = value.get("type").and_then(|v| v.as_str()) {
                            if matches!(t, "done" | "fail") {
                                term_flag.store(true, std::sync::atomic::Ordering::SeqCst);
                            }
                        }
                        on_event(value);
                    }
                }
                Err(_) => break,
            }
        }
        if !term_flag.load(std::sync::atomic::Ordering::SeqCst) {
            // 内核进程已退出但未发终态事件（崩溃/被外部终止）
            on_event(serde_json::json!({
                "type": "fail",
                "code": 1,
                "tail": "内核进程意外退出（未收到 done/fail 事件）"
            }));
        }
    });

    // stderr → 训练日志转发（字节级 + lossy：内核输出可能含 GBK 字节）
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut buf: Vec<u8> = Vec::with_capacity(1024);
        loop {
            buf.clear();
            match reader.read_until(b'\n', &mut buf).await {
                Ok(0) => break,
                Ok(_) => {
                    let line = String::from_utf8_lossy(&buf);
                    tracing::info!("[kernel] {}", line.trim_end());
                }
                Err(_) => break,
            }
        }
    });

    Ok(handle)
}

/// 内核事件 → 领域事件（run_id 归属由调用方注入）。
pub fn event_from_kernel(value: &serde_json::Value, run_id: &str) -> Option<Event> {
    let kind = value.get("type")?.as_str()?;
    match kind {
        "hello" => Some(Event::Hello {
            run_id: run_id.to_string(),
            backend: value.get("backend")?.as_str()?.to_string(),
            version: value.get("version")?.as_str()?.to_string(),
        }),
        "progress" => Some(Event::Progress {
            run_id: run_id.to_string(),
            step: value.get("step").and_then(|v| v.as_u64()).unwrap_or(0),
            total: value.get("total").and_then(|v| v.as_u64()),
            epoch: value.get("epoch").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            loss: value.get("loss").and_then(|v| v.as_f64()).unwrap_or(0.0),
            lr: value.get("lr").and_then(|v| v.as_f64()).unwrap_or(0.0),
            eta_s: value.get("eta_s").and_then(|v| v.as_u64()),
        }),
        "metric" => Some(Event::Metric {
            run_id: run_id.to_string(),
            step: value.get("step").and_then(|v| v.as_u64()).unwrap_or(0),
            loss: value.get("loss").and_then(|v| v.as_f64()),
            lr: value.get("lr").and_then(|v| v.as_f64()),
        }),
        "log" => Some(Event::Log {
            run_id: run_id.to_string(),
            level: value
                .get("level")
                .and_then(|v| v.as_str())
                .unwrap_or("info")
                .to_string(),
            msg: value
                .get("msg")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
        }),
        "sample" => Some(Event::Sample {
            run_id: run_id.to_string(),
            path: value
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
        }),
        "done" => Some(Event::Done {
            run_id: run_id.to_string(),
            code: value.get("code").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        }),
        "fail" => Some(Event::Fail {
            run_id: run_id.to_string(),
            code: value.get("code").and_then(|v| v.as_u64()).unwrap_or(1) as u32,
            tail: value
                .get("tail")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
        }),
        _ => None,
    }
}

/// 解析内核事件并挂到事件总线（附 run_id 归属）。
pub fn publish_event(bus: &EventBus, value: &serde_json::Value, run_id: &str) {
    if let Some(ev) = event_from_kernel(value, run_id) {
        bus.emit(ev);
    }
}

/// 内核环境引导：优先读取工作区 kernel.json（venv python + sd-scripts 路径），
/// 未安装内核时回退系统 Python（mock/打标用）。
#[derive(Debug, Clone)]
pub struct KernelEnv {
    pub python: Option<PathBuf>,
    pub sd_scripts: Option<PathBuf>,
    /// ai-toolkit 后端（M3：Krea 2 等 DiT 模型；kernel.json `ai_toolkit` 字段）
    pub aitk: Option<AitkEnv>,
    pub message: Option<String>,
}

/// ai-toolkit 内核环境（独立 venv，与 sd-scripts 隔离避免 transformers 版本冲突）。
#[derive(Debug, Clone)]
pub struct AitkEnv {
    pub python: PathBuf,
    pub repo: PathBuf,
    /// Krea 2 资产（prepare 命令生成并写入 kernel.json）
    pub krea2: Option<Krea2Assets>,
}

/// Krea 2 训练资产路径（本地化：MMDiT 单文件 + Qwen3-VL TE 目录 + Qwen-Image VAE 目录）。
#[derive(Debug, Clone)]
pub struct Krea2Assets {
    pub mmdit: PathBuf,
    pub text_encoder: PathBuf,
    pub vae_root: PathBuf,
}

impl KernelEnv {
    /// `workspace`：工作区根（含 `.kernel/kernel.json`；None = 不检查内核清单）。
    pub fn detect_for(workspace: Option<&Path>) -> Self {
        if let Some(ws) = workspace {
            if let Some((python, sd_scripts, aitk)) = read_kernel_manifest(ws) {
                return Self {
                    python: Some(python),
                    sd_scripts,
                    aitk,
                    message: None,
                };
            }
        }
        if let Some(p) = find_python() {
            Self {
                python: Some(p),
                sd_scripts: None,
                aitk: None,
                message: None,
            }
        } else {
            Self {
                python: None,
                sd_scripts: None,
                aitk: None,
                message: Some(
                    "未检测到 Python。请安装 Python 3.12+（https://www.python.org/downloads/），\
                     或运行 `tiandi doctor` 查看指引"
                        .into(),
                ),
            }
        }
    }

    /// 兼容旧调用（无工作区上下文）。
    pub fn detect() -> Self {
        Self::detect_for(None)
    }

    /// 是否具备运行内核的条件。
    pub fn ready(&self) -> bool {
        self.python.is_some()
    }
}

/// 读取 `<workspace>/.kernel/kernel.json`（kernel install 产物）。
fn read_kernel_manifest(workspace: &Path) -> Option<(PathBuf, Option<PathBuf>, Option<AitkEnv>)> {
    let path = workspace.join(".kernel/kernel.json");
    let text = std::fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let python = v.get("python")?.as_str()?;
    let sd_scripts = v.get("sd_scripts").and_then(|s| s.as_str());
    let p = PathBuf::from(python);
    if !p.exists() {
        return None;
    }
    // ai_toolkit 字段（可选）
    let aitk = v.get("ai_toolkit").and_then(|a| {
        let py = a.get("python")?.as_str()?;
        let repo = a.get("repo")?.as_str()?;
        let krea2 = a.get("krea2").and_then(|k| {
            let mmdit = k.get("mmdit").and_then(|x| x.as_str()).map(PathBuf::from)?;
            let te = k.get("text_encoder").and_then(|x| x.as_str()).map(PathBuf::from)?;
            let vae = k.get("vae_root").and_then(|x| x.as_str()).map(PathBuf::from)?;
            Some(Krea2Assets { mmdit, text_encoder: te, vae_root: vae })
        });
        Some(AitkEnv {
            python: PathBuf::from(py),
            repo: PathBuf::from(repo),
            krea2,
        })
    });
    Some((p, sd_scripts.map(PathBuf::from), aitk))
}

fn find_python() -> Option<PathBuf> {
    // 1. PATH 中的 python（venv 激活或已装）
    if std::process::Command::new("python")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        return Some(PathBuf::from("python"));
    }
    // 2. Windows 用户安装路径（python.org 默认）
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        for ver in ["Python312", "Python313", "Python311"] {
            let p = Path::new(&local).join(format!("Programs/Python/{ver}/python.exe"));
            if p.exists() {
                return Some(p);
            }
        }
    }
    None
}

#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error("内核进程启动失败: {0}")]
    Spawn(std::io::Error),
    #[error("stdout 通道不可用")]
    StdoutClosed,
    #[error("stderr 通道不可用")]
    StderrClosed,
    #[error("stdin 通道不可用（内核已退出）")]
    StdinClosed,
    #[error("控制命令发送失败: {0}")]
    Io(#[from] std::io::Error),
    #[error("取消超时（内核未在 5s 内退出，已强杀）")]
    CancelTimeout,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_kernel_events() {
        let progress = serde_json::json!({
            "type": "progress", "step": 12, "epoch": 0.2,
            "loss": 0.31, "lr": 1e-4, "eta_s": 90
        });
        match event_from_kernel(&progress, "r1").unwrap() {
            Event::Progress {
                run_id, step, loss, ..
            } => {
                assert_eq!(run_id, "r1");
                assert_eq!(step, 12);
                assert_eq!(loss, 0.31);
            }
            other => panic!("unexpected {other:?}"),
        }

        let done = serde_json::json!({"type": "done", "code": 0});
        assert!(matches!(
            event_from_kernel(&done, "r1").unwrap(),
            Event::Done { code: 0, .. }
        ));

        let fail = serde_json::json!({"type": "fail", "code": 1, "tail": "OOM"});
        assert!(matches!(
            event_from_kernel(&fail, "r1").unwrap(),
            Event::Fail { code: 1, .. }
        ));

        // 未知类型 → None
        assert!(event_from_kernel(&serde_json::json!({"type": "x"}), "r1").is_none());
    }

    #[test]
    fn env_detection_does_not_panic() {
        let env = KernelEnv::detect();
        // 不 panic 即可；ready 与否取决于机器
        let _ = env.ready();
    }
}
