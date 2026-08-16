//! 系统信息（GPU 监控，FR-802）与设置（FR-901：镜像源等）。

use std::collections::BTreeMap;

use axum::{
    extract::State,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};

use super::ApiError;
use crate::AppState;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/system", get(system_info))
        .route("/api/settings", get(list_settings).put(update_settings))
        .route("/api/pick-file", post(pick_file))
        .route("/api/pick-dir", post(pick_dir))
        .route("/api/import-asset", post(import_asset))
}

// ---------- 资产导入（models 三目录：base_model / vae / clip） ----------

#[derive(Deserialize)]
struct ImportAsset {
    /// base_model | vae | clip
    kind: String,
    /// 源文件路径（.safetensors 等）
    path: String,
}

#[derive(Serialize)]
struct ImportResult {
    /// 导入后的正式路径（<workspace>/models/<kind>/<name>）
    path: String,
}

/// 把用户选择的文件收纳进工作区 models 目录（硬链接优先，跨盘回退复制），
/// 与 ComfyUI 的 models 目录风格一致。
async fn import_asset(
    State(state): State<AppState>,
    Json(input): Json<ImportAsset>,
) -> Result<Json<ImportResult>, ApiError> {
    let dir_name = match input.kind.as_str() {
        "base_model" => "base_model",
        "vae" => "vae",
        "clip" => "clip",
        other => {
            return Err(ApiError::BadRequest(format!(
                "未知资产类型：{other}（base_model / vae / clip）"
            )))
        }
    };
    let src = std::path::PathBuf::from(&input.path);
    if !src.is_file() {
        return Err(ApiError::BadRequest(format!(
            "源文件不存在：{}",
            input.path
        )));
    }
    let file_name = src
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| ApiError::BadRequest("无法解析文件名".into()))?;
    let models_root = crate::output_root(state.trainer.runs_dir())
        .parent()
        .unwrap_or_else(|| state.trainer.runs_dir())
        .join("models");
    let target_dir = models_root.join(dir_name);
    std::fs::create_dir_all(&target_dir)
        .map_err(|e| ApiError::Internal(format!("创建目录失败：{e}")))?;

    // 目标路径（同名冲突自动加序号）
    let mut target = target_dir.join(file_name);
    let mut n = 1;
    while target.exists() {
        let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("asset");
        let ext = src.extension().and_then(|e| e.to_str()).unwrap_or("");
        target = target_dir.join(format!("{stem}_{n}.{ext}"));
        n += 1;
    }
    // 同盘硬链接（即时）；跨盘复制
    let ok = std::fs::hard_link(&src, &target).is_ok();
    if !ok {
        std::fs::copy(&src, &target)
            .map_err(|e| ApiError::Internal(format!("复制文件失败：{e}")))?;
    }
    Ok(Json(ImportResult {
        path: target.to_string_lossy().into_owned(),
    }))
}

// ---------- 本地文件选择（rfd 原生对话框） ----------

/// Windows：隐藏 owner 窗口，保证原生对话框置顶显示（无需点任务栏）。
#[cfg(windows)]
mod dialog_owner {
    use std::num::NonZeroIsize;

    use raw_window_handle::{
        DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawWindowHandle,
        Win32WindowHandle, WindowHandle,
    };
    use windows_sys::Win32::Foundation::HWND;
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::UI::WindowsAndMessaging::*;

    pub struct WinHandle(pub HWND);
    // 对话框在 blocking 线程使用；句柄仅用于 rfd 调用期间
    unsafe impl Send for WinHandle {}

    impl HasWindowHandle for WinHandle {
        fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
            let h = Win32WindowHandle::new(
                NonZeroIsize::new(self.0 as isize).ok_or(HandleError::Unavailable)?,
            );
            Ok(unsafe { WindowHandle::borrow_raw(RawWindowHandle::Win32(h)) })
        }
    }

    impl HasDisplayHandle for WinHandle {
        fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
            Ok(DisplayHandle::windows())
        }
    }

    unsafe extern "system" fn wnd_proc(
        hwnd: HWND,
        msg: u32,
        wparam: usize,
        lparam: isize,
    ) -> isize {
        DefWindowProcW(hwnd, msg, wparam, lparam)
    }

    /// 创建隐藏窗口（对话框 owner）。
    pub fn create() -> Option<(WinHandle, HWND)> {
        unsafe {
            let hinstance = GetModuleHandleW(std::ptr::null());
            let class: Vec<u16> = "TiandiDialogOwner\0".encode_utf16().collect();
            let wc = WNDCLASSW {
                style: 0,
                lpfnWndProc: Some(wnd_proc),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: hinstance,
                hIcon: std::ptr::null_mut(),
                hCursor: std::ptr::null_mut(),
                hbrBackground: std::ptr::null_mut(),
                lpszMenuName: std::ptr::null(),
                lpszClassName: class.as_ptr(),
            };
            RegisterClassW(&wc);
            let hwnd = CreateWindowExW(
                0,
                class.as_ptr(),
                class.as_ptr(),
                0, // 隐藏
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                0,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                hinstance,
                std::ptr::null_mut(),
            );
            if hwnd.is_null() {
                return None;
            }
            Some((WinHandle(hwnd), hwnd))
        }
    }

    pub fn destroy(hwnd: HWND) {
        unsafe {
            DestroyWindow(hwnd);
        }
    }
}

#[derive(Serialize)]
struct PickResult {
    path: Option<String>,
}

/// 弹出系统文件选择框（阻塞调用，放 blocking 池）；取消返回 path=null。
async fn pick_file() -> Json<PickResult> {
    let picked = tokio::task::spawn_blocking(|| {
        #[cfg(windows)]
        let owner = dialog_owner::create();
        let mut dlg = rfd::FileDialog::new()
            .set_title("选择基底模型")
            .add_filter("模型文件", &["safetensors"]);
        #[cfg(windows)]
        if let Some((h, _)) = &owner {
            dlg = dlg.set_parent(h);
        }
        let picked = dlg.pick_file();
        #[cfg(windows)]
        if let Some((_, hwnd)) = owner {
            dialog_owner::destroy(hwnd);
        }
        picked
    })
    .await
    .ok()
    .flatten();
    Json(PickResult {
        path: picked.map(|p| p.to_string_lossy().into_owned()),
    })
}

/// 弹出系统目录选择框（数据集目录）。
async fn pick_dir() -> Json<PickResult> {
    let picked = tokio::task::spawn_blocking(|| {
        #[cfg(windows)]
        let owner = dialog_owner::create();
        let mut dlg = rfd::FileDialog::new().set_title("选择数据集目录");
        #[cfg(windows)]
        if let Some((h, _)) = &owner {
            dlg = dlg.set_parent(h);
        }
        let picked = dlg.pick_folder();
        #[cfg(windows)]
        if let Some((_, hwnd)) = owner {
            dialog_owner::destroy(hwnd);
        }
        picked
    })
    .await
    .ok()
    .flatten();
    Json(PickResult {
        path: picked.map(|p| p.to_string_lossy().into_owned()),
    })
}

// ---------- 系统信息 ----------

#[derive(Serialize)]
struct SystemInfo {
    gpu: Option<GpuInfo>,
    server_time: String,
}

#[derive(Serialize)]
struct GpuInfo {
    name: String,
    mem_used_mb: u64,
    mem_total_mb: u64,
    util_percent: u64,
}

async fn system_info() -> Json<SystemInfo> {
    let gpu = tokio::task::spawn_blocking(query_gpu).await.unwrap_or(None);
    Json(SystemInfo {
        gpu,
        server_time: chrono::Utc::now().to_rfc3339(),
    })
}

/// nvidia-smi 单次快照解析。
fn query_gpu() -> Option<GpuInfo> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let parts: Vec<&str> = text.trim().split(',').map(|s| s.trim()).collect();
    if parts.len() < 4 {
        return None;
    }
    Some(GpuInfo {
        name: parts[0].to_string(),
        mem_used_mb: parts[1].parse().ok()?,
        mem_total_mb: parts[2].parse().ok()?,
        util_percent: parts[3].parse().ok()?,
    })
}

// ---------- 设置 ----------

async fn list_settings(
    State(state): State<AppState>,
) -> Result<Json<BTreeMap<String, String>>, ApiError> {
    let store = state.store.lock().await;
    Ok(Json(store.list_settings()?))
}

#[derive(Deserialize)]
struct SettingsUpdate {
    /// 待写入的设置（空值 = 删除该键）
    values: BTreeMap<String, String>,
}

async fn update_settings(
    State(state): State<AppState>,
    Json(input): Json<SettingsUpdate>,
) -> Result<Json<BTreeMap<String, String>>, ApiError> {
    let store = state.store.lock().await;
    for (key, value) in &input.values {
        if value.is_empty() {
            // 空值 = 删除该键（实现注释承诺的语义）
            store.delete_setting(key)?;
        } else {
            store.set_setting(key, value)?;
        }
    }
    Ok(Json(store.list_settings()?))
}
