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

use axum::{
    extract::Request,
    http::{
        header::{self, HeaderValue},
        StatusCode,
    },
    middleware::{self, Next},
    response::{IntoResponse, Response},
    Router,
};
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
    // 且服务只绑 127.0.0.1。permissive CORS 仅为本地跨源提供 ACAO 响应头，
    // 真正的访问控制由最外层 `guard_local_access` 中间件完成（Host/Origin 白名单）。
    // 路由：/api/* → API；/output/* → 产物静态（示例图/LoRA）；其余 → UI（SPA，若存在）。
    let runs_dir = state.trainer.runs_dir().to_path_buf();
    let output_root = output_root(&runs_dir);
    let serve_output =
        tower_http::services::ServeDir::new(&output_root).append_index_html_on_directories(false);
    let mut router = api::router(state)
        .nest_service("/output", serve_output.clone())
        // 未知 /api/* 路径：404（不落入 SPA fallback）
        .route(
            "/api/{*rest}",
            axum::routing::get(|| async { StatusCode::NOT_FOUND })
                .post(|| async { StatusCode::NOT_FOUND }),
        );
    match ui_dist_dir() {
        // 一键启动模式：服务 UI 构建产物（SPA fallback 到 index.html）
        Some(ui) => {
            let spa = tower_http::services::ServeDir::new(&ui)
                .append_index_html_on_directories(true)
                .fallback(tower_http::services::ServeFile::new(ui.join("index.html")));
            router = router.fallback_service(spa);
        }
        // 无 UI 产物：保持旧行为（output 静态兜底）
        None => {
            router = router.fallback_service(serve_output);
        }
    }
    // 中间件（后加的在外层）：guard 校验 Host/Origin 并附加安全响应头；
    // CORS 放行在其内（本地跨源仍需 ACAO 头，恶意来源在 guard 即被拒）
    router
        .layer(tower_http::cors::CorsLayer::permissive())
        .layer(middleware::from_fn(guard_local_access))
}

/// 产物根目录（runs 的兄弟目录 `<workspace>/output`）：示例图与每轮 LoRA 集中存放。
pub fn output_root(runs_dir: &std::path::Path) -> std::path::PathBuf {
    runs_dir.parent().unwrap_or(runs_dir).join("output")
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
    tiandi_engine_compat::asset_path("kernel_runner.py")
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
    for offset in 0..10u16 {
        let port = config.port.saturating_add(offset);
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
                let trainer = state.trainer.clone();
                axum::serve(listener, build_router(state))
                    .with_graceful_shutdown(shutdown_signal())
                    .await
                    .map_err(std::io::Error::other)?;
                // 优雅关停完成：终止全部内核进程（防止孤儿内核继续训练/写产物）
                tracing::info!("关停：终止全部内核进程");
                trainer.kill_all();
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

// ---------- 本地访问守卫（Host/Origin 白名单 + 安全响应头） ----------

/// 本地访问守卫中间件：
/// - `Host` 必须为 localhost / 127.0.0.1（可带端口，忽略大小写）→ 防 DNS rebinding；
///   无 `Host` 头的 HTTP/1.0 请求放行；
/// - `Origin` 缺失（同源/非浏览器）放行；存在时必须为本地来源
///   （`tauri://localhost`、`http://localhost[:port]`、`http://127.0.0.1[:port]`、
///   `http://tauri.localhost[:port]`），其余一律 403（防任意网页跨源读写本地服务）；
/// - 同时为所有响应附加 `X-Content-Type-Options: nosniff`。
async fn guard_local_access(request: Request, next: Next) -> Response {
    // Host：必须为 localhost / 127.0.0.1（可带端口，忽略大小写）→ 防 DNS rebinding；
    // 无 Host 头（HTTP/1.0）放行
    if let Some(h) = request.headers().get(header::HOST) {
        let Ok(host) = h.to_str() else {
            return forbidden();
        };
        if !host_allowed(host) {
            return forbidden();
        }
    }
    // Origin：缺失（同源/非浏览器）放行；存在时必须为本地来源，其余 403
    if let Some(o) = request.headers().get(header::ORIGIN) {
        let Ok(origin) = o.to_str() else {
            return forbidden();
        };
        if !origin_allowed(origin) {
            return forbidden();
        }
    }
    let mut response = next.run(request).await;
    response.headers_mut().insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    response
}

fn forbidden() -> Response {
    (StatusCode::FORBIDDEN, "forbidden").into_response()
}

/// `Host` 头是否为本机回环地址（localhost / 127.0.0.1，可带端口，忽略大小写）。
fn host_allowed(host: &str) -> bool {
    let hostname = host.trim().split(':').next().unwrap_or(host);
    hostname.eq_ignore_ascii_case("localhost") || hostname.eq_ignore_ascii_case("127.0.0.1")
}

/// `Origin` 头是否为本机来源（scheme://host[:port] 形态）。
fn origin_allowed(origin: &str) -> bool {
    let Some((scheme, authority)) = origin.trim().split_once("://") else {
        return false;
    };
    let host = authority.split(':').next().unwrap_or(authority);
    match scheme.to_ascii_lowercase().as_str() {
        // Tauri WebView（Windows/macOS）
        "tauri" => host.eq_ignore_ascii_case("localhost"),
        // 纯浏览器模式 / Linux WebView（WebKitGTK）
        "http" => {
            host.eq_ignore_ascii_case("localhost")
                || host.eq_ignore_ascii_case("127.0.0.1")
                || host.eq_ignore_ascii_case("tauri.localhost")
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request as HttpRequest;
    use http_body_util::BodyExt;
    use tiandi_core::EventBus;
    use tiandi_state::Store;
    use tower::ServiceExt;

    fn test_router() -> Router {
        let store = Store::open_in_memory().unwrap();
        let st = AppState::new(
            store,
            EventBus::default(),
            std::env::temp_dir(),
            default_wrapper_path(),
            true,
        );
        build_router(st)
    }

    async fn get(app: &Router, headers: &[(&str, &str)]) -> axum::response::Response {
        let mut builder = HttpRequest::builder().uri("/api/health");
        for (k, v) in headers {
            builder = builder.header(*k, *v);
        }
        app.clone()
            .oneshot(builder.body(Body::empty()).unwrap())
            .await
            .unwrap()
    }

    #[test]
    fn host_allowlist() {
        assert!(host_allowed("localhost"));
        assert!(host_allowed("LOCALHOST"));
        assert!(host_allowed("localhost:18765"));
        assert!(host_allowed("127.0.0.1"));
        assert!(host_allowed("127.0.0.1:8080"));
        assert!(!host_allowed("evil.example.com"));
        assert!(!host_allowed("localhost.evil.com:8080"));
        assert!(!host_allowed(""));
        assert!(!host_allowed("0.0.0.0:18765"));
    }

    #[test]
    fn origin_allowlist() {
        assert!(origin_allowed("tauri://localhost"));
        assert!(origin_allowed("http://localhost"));
        assert!(origin_allowed("http://localhost:5173"));
        assert!(origin_allowed("http://127.0.0.1:18765"));
        assert!(origin_allowed("http://tauri.localhost"));
        assert!(origin_allowed("http://tauri.localhost:1420"));
        assert!(!origin_allowed("https://evil.example.com"));
        assert!(!origin_allowed("http://evil.example.com"));
        assert!(!origin_allowed("file:///etc/passwd"));
        assert!(!origin_allowed("tauri://evil.com"));
        assert!(!origin_allowed("garbage"));
        assert!(!origin_allowed(""));
    }

    #[tokio::test]
    async fn malicious_origin_rejected() {
        let app = test_router();
        let res = get(&app, &[("origin", "https://evil.example.com")]).await;
        assert_eq!(res.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn malicious_host_rejected() {
        let app = test_router();
        let res = get(&app, &[("host", "evil.example.com")]).await;
        assert_eq!(res.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn localhost_origin_and_host_allowed() {
        let app = test_router();
        let res = get(
            &app,
            &[
                ("host", "127.0.0.1:18765"),
                ("origin", "http://localhost:5173"),
            ],
        )
        .await;
        assert_eq!(res.status(), StatusCode::OK);
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["ok"], true);
    }

    #[tokio::test]
    async fn tauri_origin_allowed() {
        let app = test_router();
        let res = get(&app, &[("origin", "tauri://localhost")]).await;
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn no_origin_request_allowed() {
        // 同源/非浏览器请求无 Origin → 放行
        let app = test_router();
        let res = get(&app, &[]).await;
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn responses_carry_nosniff_header() {
        let app = test_router();
        let res = get(&app, &[]).await;
        assert_eq!(
            res.headers()
                .get("x-content-type-options")
                .and_then(|h| h.to_str().ok()),
            Some("nosniff")
        );
    }
}
