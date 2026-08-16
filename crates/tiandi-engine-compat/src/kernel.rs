//! Python 内核进程监督：spawn wrapper → 解析 JSON Lines 事件 → EventBus；
//! stdin 控制通道（cancel）；卡死看门狗（30s 无输出判卡死并杀进程树）；
//! kill-tree（Windows taskkill /T /F；POSIX 进程组 TERM）。

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

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

    /// 请求取消：先发命令等内核优雅退出（10s），超时则强杀进程树。
    pub async fn cancel(&mut self) -> Result<(), KernelError> {
        if self.send_command(r#"{"cmd":"cancel"}"#).await.is_err() {
            return Ok(()); // stdin 已关闭（内核已退出）
        }
        match tokio::time::timeout(std::time::Duration::from_secs(10), self.child.wait()).await {
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

/// 终止进程树。
///
/// Windows：`taskkill /PID <pid> /T /F`（强制、含子进程树）。
/// POSIX：先尝试 `kill -TERM -<pgid>`（负 pid = 进程组，一次覆盖
/// wrapper/accelerate/python 训练子进程）；拿不到 pgid 时回退单进程 `kill -TERM <pid>`。
/// wrapper 由 [`spawn_kernel`] 以 `process_group(0)` 启动（pid == pgid，独立进程组），
/// 因此组杀不会波及服务进程自身。
pub fn kill_tree(pid: u32) -> std::io::Result<()> {
    let status = if cfg!(windows) {
        std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
    } else {
        #[cfg(unix)]
        {
            if let Some(pgid) = process_group_id(pid).filter(|p| *p > 0) {
                std::process::Command::new("kill")
                    .args(["-TERM", &format!("-{pgid}")])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status()
            } else {
                std::process::Command::new("kill")
                    .args(["-TERM", &pid.to_string()])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status()
            }
        }
        #[cfg(not(unix))]
        {
            unreachable!("kill_tree 非 Windows 分支仅用于 POSIX")
        }
    };
    status.map(|_| ())
}

/// 读取进程组 id（POSIX：`ps -o pgid= -p <pid>`；读取/解析失败返回 None）。
#[cfg(unix)]
fn process_group_id(pid: u32) -> Option<i32> {
    let out = std::process::Command::new("ps")
        .args(["-o", "pgid=", "-p", &pid.to_string()])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<i32>()
        .ok()
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
}

impl KernelMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Mock => "mock",
            Self::SdScripts => "sdscripts",
            Self::Aitk => "aitk",
        }
    }
}

/// 启动内核进程并接管事件流。
///
/// `on_event` 在每个解析出的 JSON 事件上调用（由调用方转 EventBus）；
/// 心跳等未知事件类型会被 [`event_from_kernel`] 静默丢弃。
/// 返回句柄；stdout/stderr 读取在后台任务进行。
///
/// 监督职责：
/// - POSIX 下 wrapper 以独立进程组启动（`process_group(0)`），保证
///   [`kill_tree`] 的组杀只波及 wrapper 及其训练子进程；
/// - 卡死看门狗：每 5s 检查一次最近 stdout 输出时间，超过 30s 无输出
///   即杀进程树并发 fail 事件（架构 §5.4 心跳/卡死检测承诺）。
pub fn spawn_kernel(
    launch: &KernelLaunch,
    on_event: impl Fn(serde_json::Value) + Send + Sync + 'static,
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
    // POSIX：wrapper 独立进程组（pid == pgid）→ kill_tree 组杀安全（不波及服务进程）
    #[cfg(unix)]
    cmd.process_group(0);

    let mut child: tokio::process::Child = cmd.spawn().map_err(KernelError::Spawn)?;
    let stdout = child.stdout.take().ok_or(KernelError::StdoutClosed)?;
    let stderr = child.stderr.take().ok_or(KernelError::StderrClosed)?;
    let stdin = child.stdin.take();

    // 看门狗需要的共享状态（在 child 移入句柄前取 pid）
    let watchdog_pid = child.id();
    let handle = KernelHandle { child, stdin };

    let on_event = Arc::new(on_event);

    // 终态标记：done/fail 已发（或看门狗已发 fail）→ EOF 兜底 fail 不得重复发
    let terminated = Arc::new(AtomicBool::new(false));
    // 最近一次 stdout 输出的时间（任意一行都会刷新；心跳/进度均计入）
    let last_output = Arc::new(Mutex::new(std::time::Instant::now()));
    // stdout 读取任务结束（进程退出 ≈ 管道 EOF）
    let reader_done = Arc::new(AtomicBool::new(false));

    // stdout → 事件（逐行 JSON）；EOF 兜底：进程退出但无终态事件（崩溃/被杀）→ 发 Fail
    // 注意：Windows 下内核子进程可能输出 GBK 字节（tqdm 等），用字节级读取 + lossy 解码，
    // 避免 lines() 遇非法 UTF-8 提前退出导致管道不再被读（wrapper 写阻塞）
    let term_flag = terminated.clone();
    let reader_last = last_output.clone();
    let reader_done_flag = reader_done.clone();
    let reader_on_event = on_event.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);
        let mut buf: Vec<u8> = Vec::with_capacity(1024);
        loop {
            buf.clear();
            match reader.read_until(b'\n', &mut buf).await {
                Ok(0) => break,
                Ok(_) => {
                    *reader_last.lock().unwrap_or_else(|e| e.into_inner()) =
                        std::time::Instant::now();
                    let line = String::from_utf8_lossy(&buf).trim().to_string();
                    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
                        if let Some(t) = value.get("type").and_then(|v| v.as_str()) {
                            if matches!(t, "done" | "fail") {
                                term_flag.store(true, Ordering::SeqCst);
                            }
                        }
                        reader_on_event(value);
                    }
                }
                Err(_) => break,
            }
        }
        reader_done_flag.store(true, Ordering::SeqCst);
        if !term_flag.load(Ordering::SeqCst) {
            // 内核进程已退出但未发终态事件（崩溃/被外部终止）
            reader_on_event(serde_json::json!({
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

    // 卡死看门狗：每 5s 检查；距上次 stdout 输出 > 30s → 杀进程树 + 发 fail。
    // 进程退出（reader_done）或已收到终态事件（term_flag）后看门狗自行结束。
    let watchdog_last = last_output.clone();
    let watchdog_reader_done = reader_done.clone();
    let watchdog_term = terminated.clone();
    let watchdog_on_event = on_event.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            if watchdog_reader_done.load(Ordering::SeqCst) || watchdog_term.load(Ordering::SeqCst) {
                break;
            }
            let last = *watchdog_last.lock().unwrap_or_else(|e| e.into_inner());
            if last.elapsed() > std::time::Duration::from_secs(30) {
                // 先置终态标记：随后 stdout EOF 的兜底 fail 不得重复发
                watchdog_term.store(true, Ordering::SeqCst);
                if let Some(pid) = watchdog_pid {
                    if let Err(e) = kill_tree(pid) {
                        tracing::warn!("看门狗杀进程树失败 pid={pid}: {e}");
                    }
                }
                watchdog_on_event(serde_json::json!({
                    "type": "fail",
                    "code": 1,
                    "tail": "训练卡死检测：超过 30 秒无输出，已强制终止"
                }));
                break;
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
            let te = k
                .get("text_encoder")
                .and_then(|x| x.as_str())
                .map(PathBuf::from)?;
            let vae = k
                .get("vae_root")
                .and_then(|x| x.as_str())
                .map(PathBuf::from)?;
            Some(Krea2Assets {
                mmdit,
                text_encoder: te,
                vae_root: vae,
            })
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
    #[error("取消超时（内核未在 10s 内退出，已强杀）")]
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

    /// kill_tree 应能终止真实子进程（Windows：taskkill /T /F）。
    #[cfg(windows)]
    #[test]
    fn kill_tree_terminates_process() {
        let mut child = std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command", "Start-Sleep -Seconds 60"])
            .spawn()
            .expect("spawn powershell");
        let pid = child.id();
        kill_tree(pid).expect("kill_tree 应成功");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if child.try_wait().ok().flatten().is_some() {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "kill_tree 未能在 5s 内终止进程"
            );
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }

    /// kill_tree 的 POSIX 分支：进程组 TERM 应终止以 process_group(0) 启动的整棵进程树。
    #[cfg(unix)]
    #[test]
    fn kill_tree_terminates_process_group() {
        use std::os::unix::process::CommandExt as _;
        let mut child = std::process::Command::new("sh")
            .arg("-c")
            .arg("sleep 60")
            .process_group(0)
            .spawn()
            .expect("spawn sh");
        let pid = child.id();
        // 组杀不应波及其他进程（负 pid = 进程组，pid == pgid）
        kill_tree(pid).expect("kill_tree 应成功");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if child.try_wait().ok().flatten().is_some() {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "kill_tree 未能在 5s 内终止进程组"
            );
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}
