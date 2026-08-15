//! 天地熔炉桌面壳：内嵌启动本地服务（127.0.0.1:18765），WebView 加载前端。
//!
//! 单进程架构（docs/architecture.md §2）：Tauri 壳 = tiandi-server + WebView；
//! 数据目录 = 系统应用数据目录（tiandi.db + models/datasets/recipes/runs/vault）。
//! 端口被占用时自动回退（18765–18774），前端通过 /api/health 探测实际端口。

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::time::Duration;

use tauri::Manager;
use tiandi_core::EventBus;
use tiandi_server::{serve, AppState, ServerConfig};
use tiandi_state::Store;

const PORT: u16 = 18765;

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let data_dir = app.path().app_data_dir().expect("获取应用数据目录");
            if let Err(e) = tiandi_state::ensure_workspace_layout(&data_dir) {
                eprintln!("工作区初始化失败：{e}");
            }
            let store = match Store::open(&data_dir.join("tiandi.db")) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("打开数据库失败：{e}");
                    std::process::exit(1);
                }
            };
            let state = AppState::new(
                store,
                EventBus::default(),
                data_dir.join("runs"),
                tiandi_server::default_wrapper_path(),
                true,
            );

            // 端口预选：被占用时向后回退（serve 内还有兜底重试）
            let port = pick_free_port(PORT);
            let config = ServerConfig {
                host: "127.0.0.1".into(),
                port,
                demo: true,
            };

            std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().expect("创建 tokio runtime");
                rt.block_on(async move {
                    match serve(state, config).await {
                        Ok(_) => tracing::info!("本地服务正常退出"),
                        Err(e) => eprintln!("✗ 本地服务启动失败：{e}"),
                    }
                });
            });

            // 等待端口就绪再返回（窗口创建时前端首连必成功，消除竞态）
            for _ in 0..50 {
                if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("天地熔炉启动失败");
}

/// 从 `start` 起找第一个可绑定端口（立即释放，供内嵌 server 使用）。
fn pick_free_port(start: u16) -> u16 {
    for port in start..start + 10 {
        if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
    }
    start
}
