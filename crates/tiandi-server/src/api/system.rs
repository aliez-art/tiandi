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
    /// base_model 导入时顺带导入的同目录 VAE（Anima/Krea 2 场景），无则 null
    vae_path: Option<String>,
    /// base_model 导入时顺带导入的同目录文本编码器（Qwen3 系），无则 null
    te_path: Option<String>,
}

/// 把用户选择的文件收纳进工作区 models 目录（硬链接优先，跨盘回退复制）。
///
/// - **幂等**：目标（含 `_N` 序号变体）已存在且与源文件同尺寸同修改时间时，
///   直接复用现有路径，不复制、不累加序号（修复重复选择导致
///   `xxx1.safetensors`、`xxx2.safetensors` 无限累积的问题）；
/// - **配套资产顺带导入**：导入底模（base_model）时，若源目录存在
///   Qwen3 系文本编码器（qwen_3_06b_base 等）与 VAE（qwen_image_vae 等），
///   一并导入到 models/clip 与 models/vae 并返回路径（Anima/Krea 2 训练必需，
///   避免"选完底模还要手动找 TE/VAE"）。
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
    let models_root = crate::output_root(state.trainer.runs_dir())
        .parent()
        .unwrap_or_else(|| state.trainer.runs_dir())
        .join("models");
    let path = import_one(&models_root.join(dir_name), &src)?;

    // base_model：顺带导入同目录配套资产（Qwen3 TE / VAE）
    let (vae_path, te_path) = if input.kind == "base_model" {
        if let Some(dir) = src.parent() {
            let vae = find_sibling_safetensors(dir, &["qwen_image_vae", "image_vae"]);
            let te = find_sibling_safetensors(dir, &["qwen_3_06b_base", "qwen3vl_4b", "qwen_3"]);
            let vae_path = vae
                .as_ref()
                .map(|p| import_one(&models_root.join("vae"), p))
                .transpose()?;
            let te_path = te
                .as_ref()
                .map(|p| import_one(&models_root.join("clip"), p))
                .transpose()?;
            (vae_path, te_path)
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };
    Ok(Json(ImportResult {
        path,
        vae_path,
        te_path,
    }))
}

/// 单文件幂等导入：目标已存在且与源文件相同（尺寸 + 修改时间）→ 复用；
/// 否则取第一个不存在的 `_N` 序号名；同盘硬链接、跨盘复制。
fn import_one(target_dir: &std::path::Path, src: &std::path::Path) -> Result<String, ApiError> {
    let file_name = src
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| ApiError::BadRequest("无法解析文件名".into()))?;
    let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("asset");
    let ext = src.extension().and_then(|e| e.to_str()).unwrap_or("");
    std::fs::create_dir_all(target_dir)
        .map_err(|e| ApiError::Internal(format!("创建目录失败：{e}")))?;

    let same_as_src = |p: &std::path::Path| -> bool {
        p.is_file()
            && std::fs::metadata(p)
                .and_then(|m1| {
                    std::fs::metadata(src).map(|m2| {
                        m1.len() == m2.len()
                            && match (m1.modified(), m2.modified()) {
                                // 两侧 mtime 都拿得到才比较（否则 None==None 会退化成
                                // 仅按尺寸判定，误判不同文件为同一文件）
                                (Ok(a), Ok(b)) => a == b,
                                _ => false,
                            }
                    })
                })
                .unwrap_or(false)
    };
    // 已存在同源文件 → 直接复用（幂等，不累加序号）
    let mut target = target_dir.join(file_name);
    if target.exists() {
        if same_as_src(&target) {
            return Ok(target.to_string_lossy().into_owned());
        }
        let mut found: Option<std::path::PathBuf> = None;
        for n in 1..1000 {
            let cand = target_dir.join(format!("{stem}_{n}.{ext}"));
            if !cand.exists() {
                found = Some(cand);
                break;
            }
            if same_as_src(&cand) {
                return Ok(cand.to_string_lossy().into_owned());
            }
        }
        target = found.ok_or_else(|| {
            ApiError::Conflict(format!(
                "同名文件 {file_name} 及其 999 个序号副本均已存在且都不是本次导入的文件，\
                 无法继续累加序号。请清理目标目录中的旧副本后再试"
            ))
        })?;
    }
    let ok = std::fs::hard_link(src, &target).is_ok();
    if !ok {
        std::fs::copy(src, &target)
            .map_err(|e| ApiError::Internal(format!("复制文件失败：{e}")))?;
    }
    Ok(target.to_string_lossy().into_owned())
}

/// 目录中按文件名子串（不区分大小写）找第一个匹配的 .safetensors。
fn find_sibling_safetensors(dir: &std::path::Path, needles: &[&str]) -> Option<std::path::PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_lowercase();
        if name.ends_with(".safetensors")
            && needles.iter().any(|n| name.contains(&n.to_lowercase()))
        {
            return Some(path);
        }
    }
    None
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
                0, // 先隐藏创建
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
            // 显示但不激活（SW_SHOWNA）+ 置顶位序：让 IFileDialog 以本窗口为 owner
            // 弹出时能正常前台激活（避免对话框在后台/任务栏闪烁而用户看不到）。
            ShowWindow(hwnd, SW_SHOWNA);
            SetWindowPos(
                hwnd,
                HWND_TOP,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
            );
            Some((WinHandle(hwnd), hwnd))
        }
    }

    pub fn destroy(hwnd: HWND) {
        unsafe {
            DestroyWindow(hwnd);
        }
    }

    /// 提升本进程前台激活机会：允许任意进程把前台让给我们（配合 SetForegroundWindow），
    /// 避免模态对话框在后台弹出（用户只看到任务栏闪烁）。
    pub fn prepare_foreground(hwnd: HWND) {
        unsafe {
            AllowSetForegroundWindow(ASFW_ANY);
            SetForegroundWindow(hwnd);
        }
    }
}

#[derive(Serialize)]
struct PickResult {
    path: Option<String>,
}

/// 弹出系统文件选择框（阻塞调用，放 blocking 池；5 分钟超时兜底）。
async fn pick_file() -> Result<Json<PickResult>, ApiError> {
    let picked = pick_with_timeout("选择基底模型", true).await?;
    Ok(Json(PickResult { path: picked }))
}

/// 弹出系统目录选择框（数据集目录）。
async fn pick_dir() -> Result<Json<PickResult>, ApiError> {
    let picked = pick_with_timeout("选择数据集目录", false).await?;
    Ok(Json(PickResult { path: picked }))
}

/// rfd 原生对话框统一入口：Windows 下带隐藏 owner 窗口 + 前台激活；
/// 5 分钟超时兜底（对话框异常挂起时不让 HTTP 请求永久阻塞）。
async fn pick_with_timeout(title: &'static str, file: bool) -> Result<Option<String>, ApiError> {
    // owner 窗口就绪通道：超时兜底时向窗口投递 WM_CLOSE 主动关掉对话框，
    // 让阻塞线程尽快退出（此前超时只放弃 await，线程与隐藏窗口会无限滞留）。
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<isize>();
    let handle = tokio::task::spawn_blocking(move || {
        #[cfg(windows)]
        let owner = dialog_owner::create();
        #[cfg(windows)]
        if let Some((_, hwnd)) = &owner {
            let _ = ready_tx.send(*hwnd as isize);
        } else {
            drop(ready_tx);
        }
        #[cfg(not(windows))]
        drop(ready_tx);
        let mut dlg = rfd::FileDialog::new().set_title(title);
        if file {
            dlg = dlg.add_filter("模型文件", &["safetensors"]);
        }
        #[cfg(windows)]
        if let Some((h, hwnd)) = &owner {
            dialog_owner::prepare_foreground(*hwnd);
            dlg = dlg.set_parent(h);
        }
        let picked = if file {
            dlg.pick_file()
        } else {
            dlg.pick_folder()
        };
        #[cfg(windows)]
        if let Some((_, hwnd)) = owner {
            dialog_owner::destroy(hwnd);
        }
        picked
    });
    match tokio::time::timeout(std::time::Duration::from_secs(300), handle).await {
        Ok(inner) => {
            let picked = inner.ok().flatten();
            Ok(picked.map(|p| p.to_string_lossy().into_owned()))
        }
        Err(_) => {
            // 超时：向 owner 投递 WM_CLOSE 关闭对话框（若线程仍在等待用户操作）
            if let Ok(hwnd) = ready_rx.await {
                #[cfg(windows)]
                unsafe {
                    windows_sys::Win32::UI::WindowsAndMessaging::PostMessageW(
                        hwnd as windows_sys::Win32::Foundation::HWND,
                        windows_sys::Win32::UI::WindowsAndMessaging::WM_CLOSE,
                        0,
                        0,
                    );
                }
            }
            Err(ApiError::Internal(
                "文件选择超时（对话框 5 分钟无响应，已尝试自动关闭）".into(),
            ))
        }
    }
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
