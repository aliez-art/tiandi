//! 本地服务：axum REST + SSE 事件流。
//!
//! M0 骨架：健康检查、项目/任务 CRUD、任务状态机流转、SSE 事件流。
//! M1：训练启动链路（POST /api/runs/{id}/start → compat 引擎 → IPC 事件 → supervisor 状态同步）。
//! 仅绑定 `127.0.0.1`（PRD §7 安全要求）。

pub mod api;
pub mod sse;
pub mod supervisor;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use tiandi_core::EventBus;
use tiandi_engine_compat::SdScriptsTrainer;
use tiandi_state::Store;
use tokio::sync::Mutex;

/// 全局应用状态。
#[derive(Clone)]
pub struct AppState {
    /// SQLite 连接（单人本地工具，单连接 + 互斥；rusqlite Connection 非 Sync）
    pub store: Arc<Mutex<Store>>,
    /// 进程内事件总线（SSE 订阅源 + supervisor 状态同步）
    pub bus: EventBus,
    /// 训练内核编排器（SdScriptsBackend + mock）
    pub trainer: Arc<SdScriptsTrainer>,
    /// 是否启用模拟训练演示（POST /api/runs?simulate=1 走 mock 内核）
    pub demo: bool,
}

impl AppState {
    pub fn new(
        store: Store,
        bus: EventBus,
        runs_dir: PathBuf,
        wrapper: PathBuf,
        demo: bool,
    ) -> Self {
        let trainer = Arc::new(SdScriptsTrainer::new(bus.clone(), runs_dir, wrapper));
        Self {
            store: Arc::new(Mutex::new(store)),
            bus,
            trainer,
            demo,
        }
    }
}

/// 构建完整路由。
pub fn build_router(state: AppState) -> Router {
    // 本地单用户工具：WebView（tauri://localhost）与纯浏览器模式均需跨源访问，
    // 且服务只绑 127.0.0.1，permissive CORS 无风险（PRD §7 安全要求）。
    api::router(state).layer(tower_http::cors::CorsLayer::permissive())
}

/// 服务器配置。
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    /// 是否启用模拟训练演示（POST /api/runs 后自动推进状态机）
    pub demo: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 18765,
            demo: true,
        }
    }
}

impl ServerConfig {
    pub fn addr(&self) -> SocketAddr {
        format!("{}:{}", self.host, self.port)
            .parse()
            .expect("合法地址")
    }
    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }
}

/// 内核适配层默认路径（kernel_runner.py，随 crate 分发）。
pub fn default_wrapper_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tiandi-engine-compat/assets/kernel_runner.py")
}

/// 启动服务（阻塞直到 Ctrl-C）。
///
/// 端口占用时自动向后回退尝试（最多 10 个端口），返回实际监听端口；
/// 全部失败返回最后一次的绑定错误。
pub async fn serve(state: AppState, config: ServerConfig) -> Result<u16, std::io::Error> {
    // 任务监督器：内核事件 → 状态机/指标（先于请求服务启动）
    supervisor::spawn(state.clone());
    let mut last_err = None;
    for offset in 0..10 {
        let port = config.port + offset;
        let addr = format!("{}:{}", config.host, port)
            .parse::<SocketAddr>()
            .expect("合法地址");
        match tokio::net::TcpListener::bind(addr).await {
            Ok(listener) => {
                let actual = listener.local_addr()?;
                if actual.port() != config.port {
                    tracing::warn!(
                        "端口 {port_0} 被占用，已回退到 {}",
                        actual.port(),
                        port_0 = config.port
                    );
                }
                tracing::info!("天地熔炉已点火（server listening on {actual}）");
                axum::serve(listener, build_router(state))
                    .with_graceful_shutdown(shutdown_signal())
                    .await
                    .map_err(std::io::Error::other)?;
                return Ok(actual.port());
            }
            Err(e) => {
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| std::io::Error::other("端口绑定失败")))
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.expect("注册 Ctrl-C 处理器");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("注册 SIGTERM 处理器")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("收到停止信号，熄火...");
}
