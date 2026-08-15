//! 天地熔炉桌面壳：内嵌启动本地服务（127.0.0.1:18765），WebView 加载前端。
//!
//! 单进程架构（docs/architecture.md §2）：Tauri 壳 = tiandi-server + WebView；
//! 数据目录 = 系统应用数据目录（tiandi.db + models/datasets/recipes/runs/vault）。

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

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
            let state = AppState::new(store, EventBus::default(), true);
            let config = ServerConfig {
                host: "127.0.0.1".into(),
                port: PORT,
                demo: true,
            };

            std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().expect("创建 tokio runtime");
                rt.block_on(async move {
                    match serve(state, config).await {
                        Ok(()) => tracing::info!("本地服务正常退出"),
                        Err(e) => {
                            eprintln!("✗ 本地服务启动失败（{e}）。若端口 {PORT} 被占用，请先关闭已运行的实例，或改用浏览器模式（tiandi server）");
                        }
                    }
                });
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("天地熔炉启动失败");
}
