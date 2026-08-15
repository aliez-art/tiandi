//! 本地服务：axum REST + SSE 事件流。
//!
//! M0 骨架：健康检查、项目/任务 CRUD、任务状态机流转、SSE 事件流（含模拟训练演示）。
//! 仅绑定 `127.0.0.1`（PRD §7 安全要求）。

pub mod api;
pub mod sse;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use tiandi_core::EventBus;
use tiandi_state::Store;
use tokio::sync::Mutex;

/// 全局应用状态。
#[derive(Clone)]
pub struct AppState {
    /// SQLite 连接（单人本地工具，单连接 + 互斥；rusqlite Connection 非 Sync）
    pub store: Arc<Mutex<Store>>,
    /// 进程内事件总线（SSE 订阅源）
    pub bus: EventBus,
    /// 是否启用模拟训练演示（POST /api/runs?simulate=1 自动推进状态机）
    pub demo: bool,
}

impl AppState {
    pub fn new(store: Store, bus: EventBus, demo: bool) -> Self {
        Self {
            store: Arc::new(Mutex::new(store)),
            bus,
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

/// 启动服务（阻塞直到 Ctrl-C）。
pub async fn serve(state: AppState, config: ServerConfig) -> Result<(), std::io::Error> {
    let listener = tokio::net::TcpListener::bind(config.addr()).await?;
    let addr = listener.local_addr()?;
    tracing::info!("天地熔炉已点火（server listening on {addr}）");
    axum::serve(listener, build_router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(std::io::Error::other)
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
