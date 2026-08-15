//! 命令行入口：`tiandi init | doctor | server`。
//!
//! M0 骨架：`doctor` 环境体检（lora-scripts-next preflight 思路的 Rust 版）、
//! `init` 建工作区、`server` 点火本地服务。

mod doctor;

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use tiandi_server::{serve, AppState, ServerConfig};

#[derive(Parser)]
#[command(
    name = "tiandi",
    version,
    about = "天地熔炉 — 私人 LoRA 训练熔炉",
    long_about = "天地熔炉 Tiandi Furnace：Rust 控制/数据引擎 + Python 训练内核（IPC/Stdio）的私人 LoRA 训练工具。"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// 初始化工作区（models/datasets/recipes/runs/vault + tiandi.db）
    Init {
        /// 工作区根目录（默认当前目录）
        #[arg(default_value = ".")]
        dir: PathBuf,
        /// 是否同时注册为项目
        #[arg(long)]
        name: Option<String>,
    },
    /// 环境体检：路径/磁盘/CUDA/端口
    Doctor,
    /// 点火本地服务（仅绑定 127.0.0.1）
    Server {
        /// 数据目录（默认当前目录）
        #[arg(long, default_value = ".")]
        dir: PathBuf,
        #[arg(long, default_value_t = 18765)]
        port: u16,
        /// 禁用模拟训练演示
        #[arg(long)]
        no_demo: bool,
        /// 启动后自动打开浏览器
        #[arg(long)]
        web: bool,
    },
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("tiandi=info,tower_http=info")),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Init { dir, name } => cmd_init(&dir, name),
        Command::Doctor => doctor::run(),
        Command::Server {
            dir,
            port,
            no_demo,
            web,
        } => cmd_server(&dir, port, !no_demo, web).await,
    }
}

fn cmd_init(dir: &std::path::Path, name: Option<String>) {
    let root = if dir.as_os_str().is_empty() || dir == std::path::Path::new(".") {
        std::env::current_dir().expect("读取当前目录")
    } else {
        dir.to_path_buf()
    };
    if let Err(e) = tiandi_state::ensure_workspace_layout(&root) {
        eprintln!("✗ 创建工作区失败：{e}");
        std::process::exit(1);
    }
    let db_path = root.join("tiandi.db");
    if !db_path.exists() {
        match tiandi_state::open(&db_path) {
            Ok(_) => tracing::info!("数据库已创建：{}", db_path.display()),
            Err(e) => {
                eprintln!("✗ 初始化数据库失败：{e}");
                std::process::exit(1);
            }
        }
    }
    tracing::info!("✓ 工作区就绪：{}", root.display());
    for sub in ["models", "datasets", "recipes", "runs", "vault"] {
        tracing::info!("  - {}", root.join(sub).display());
    }
    if let Some(name) = name {
        tracing::info!("项目注册（M1 实现）：{name}");
    }
    tracing::info!("下一步：tiandi doctor 体检，或 tiandi server 点火");
}

async fn cmd_server(dir: &std::path::Path, port: u16, demo: bool, web: bool) {
    let root = if dir.as_os_str().is_empty() || dir == std::path::Path::new(".") {
        std::env::current_dir().expect("读取当前目录")
    } else {
        dir.to_path_buf()
    };
    if let Err(e) = tiandi_state::ensure_workspace_layout(&root) {
        eprintln!("✗ 工作区不完整：{e}（先运行 tiandi init）");
        std::process::exit(1);
    }
    let store = match tiandi_state::Store::open(&root.join("tiandi.db")) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("✗ 打开数据库失败：{e}");
            std::process::exit(1);
        }
    };
    let state = AppState::new(
        store,
        tiandi_core::EventBus::default(),
        root.join("runs"),
        tiandi_server::default_wrapper_path(),
        demo,
    );
    let config = ServerConfig {
        host: "127.0.0.1".into(),
        port,
        demo,
    };
    // 端口回退在 serve 内部处理；浏览器模式先探测实际端口再打开
    if web {
        let probe_port = match std::net::TcpListener::bind(("127.0.0.1", port)) {
            Ok(_) => port, // 空闲：serve 将用它（此处仅探测，随即释放）
            Err(_) => port + 1,
        };
        let url = format!("http://127.0.0.1:{probe_port}");
        if let Err(e) = webbrowser::open(&url) {
            tracing::warn!("打开浏览器失败：{e}；请手动访问 {url}");
        }
    }
    match serve(state, config).await {
        Ok(actual) => tracing::info!("服务已在 http://127.0.0.1:{actual} 提供"),
        Err(e) => {
            eprintln!("✗ 服务异常退出：{e}");
            std::process::exit(1);
        }
    }
}
