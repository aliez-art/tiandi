//! 本地服务：axum REST + SSE 事件流。
//!
//! M0 骨架：健康检查、项目/任务 CRUD、任务状态机流转、SSE 事件流。
//! M1：训练启动链路（POST /api/runs/{id}/start → compat 引擎 → IPC 事件 → supervisor 状态同步）。
//! 仅绑定 `127.0.0.1`（PRD §7 安全要求）。

pub mod api;
pub mod queue;
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
    // 路由：/api/* → API；/runs/* → runs 静态（采样图/产物）；其余 → UI（SPA，若存在）。
    let runs_dir = state.trainer.runs_dir().to_path_buf();
    let serve_runs =
        tower_http::services::ServeDir::new(&runs_dir).append_index_html_on_directories(false);
    let mut router = api::router(state)
        .layer(tower_http::cors::CorsLayer::permissive())
        .nest_service("/runs", serve_runs.clone())
        // 未知 /api/* 路径：404（不落入 SPA fallback）
        .route(
            "/api/{*rest}",
            axum::routing::get(|| async { axum::http::StatusCode::NOT_FOUND })
                .post(|| async { axum::http::StatusCode::NOT_FOUND }),
        );
    match ui_dist_dir() {
        // 一键启动模式：服务 UI 构建产物（SPA fallback 到 index.html）
        Some(ui) => {
            let spa = tower_http::services::ServeDir::new(&ui)
                .append_index_html_on_directories(true)
                .fallback(tower_http::services::ServeFile::new(ui.join("index.html")));
            router = router.fallback_service(spa);
        }
        // 无 UI 产物：保持旧行为（runs 静态兜底）
        None => {
            router = router.fallback_service(serve_runs);
        }
    }
    router
}

/// UI 构建产物目录（`<repo>/ui/dist`）；可用 `TIANDI_UI_DIR` 覆盖。
///
/// 从编译路径推导：`crates/tiandi-server` → 上两级 = 仓库根 → `ui/dist`。
/// 发布安装（cargo install 等）后该路径不存在，返回 None（仅 API + runs 静态）。
fn ui_dist_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("TIANDI_UI_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()?
        .parent()?
        .join("ui/dist");
    p.is_dir().then_some(p)
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
    // 任务监督器（内核事件 → 状态机/指标）+ 队列调度器（串行拉起）
    supervisor::spawn(state.clone());
    queue::spawn(state.clone());
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
