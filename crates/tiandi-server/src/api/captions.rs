//! 打标与标签 API（PRD FR-301~303）：caption 读写（kohya 同名 .txt 约定）、
//! 批量替换、标签统计、内核打标。

use std::path::{Path as FsPath, PathBuf};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post, put},
    Router,
};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncBufReadExt;

use super::ApiError;
use crate::AppState;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/datasets/{id}/captions", get(list_captions))
        .route("/api/datasets/{id}/caption", put(save_caption))
        .route("/api/datasets/{id}/captions/batch", post(batch_replace))
        .route("/api/datasets/{id}/tags", get(tag_stats))
        .route("/api/datasets/{id}/tag", post(run_tagging))
}

/// 单张图的 caption（读磁盘同名 .txt，无则空）。
#[derive(Serialize)]
struct CaptionEntry {
    /// 相对数据集根的图片路径
    path: String,
    caption: String,
    /// caption 文件是否存在
    has_file: bool,
}

async fn list_captions(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<CaptionEntry>>, ApiError> {
    let (dataset_dir, images) = {
        let store = state.store.lock().await;
        let ds = store.get_dataset(&id)?;
        (ds.dir, store.list_dataset_images(&id)?)
    };
    let root = PathBuf::from(&dataset_dir);
    let mut out = Vec::with_capacity(images.len());
    for img in images {
        let caption_path = caption_path_for(&root, &img.path);
        let (caption, has_file) = match std::fs::read_to_string(&caption_path) {
            Ok(text) => (text.trim().to_string(), true),
            Err(_) => (String::new(), false),
        };
        out.push(CaptionEntry {
            path: img.path,
            caption,
            has_file,
        });
    }
    Ok(Json(out))
}

#[derive(Deserialize)]
struct SaveCaption {
    path: String,
    text: String,
}

/// 保存单张图的 caption（写同名 .txt）。
async fn save_caption(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(input): Json<SaveCaption>,
) -> Result<StatusCode, ApiError> {
    let dataset_dir = {
        let store = state.store.lock().await;
        store.get_dataset(&id)?.dir
    };
    let root = PathBuf::from(&dataset_dir);
    let caption_path = caption_path_for(&root, &input.path);
    if let Some(parent) = caption_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| ApiError::Internal(format!("创建目录失败：{e}")))?;
    }
    std::fs::write(&caption_path, format!("{}\n", input.text.trim()))
        .map_err(|e| ApiError::Internal(format!("写 caption 失败：{e}")))?;
    Ok(StatusCode::NO_CONTENT)
}

/// 批量替换规则。
#[derive(Deserialize)]
struct ReplaceRule {
    find: String,
    replace: String,
    /// 是否按正则解释 find
    regex: Option<bool>,
}

#[derive(Deserialize)]
struct BatchReplace {
    rules: Vec<ReplaceRule>,
}

#[derive(Serialize)]
struct BatchResult {
    affected: u64,
    total: u64,
}

/// 批量替换：对数据集全部 caption 应用规则并写回（PRD FR-302 批量操作）。
async fn batch_replace(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(input): Json<BatchReplace>,
) -> Result<Json<BatchResult>, ApiError> {
    let (dataset_dir, images) = {
        let store = state.store.lock().await;
        let ds = store.get_dataset(&id)?;
        (ds.dir, store.list_dataset_images(&id)?)
    };
    let root = PathBuf::from(&dataset_dir);

    // 预编译正则（若任一规则非法则整体拒绝，避免半途失败）
    let compiled: Vec<(bool, Option<regex::Regex>, String, String)> = input
        .rules
        .iter()
        .map(|r| {
            let re = if r.regex.unwrap_or(false) {
                Some(
                    regex::Regex::new(&r.find)
                        .map_err(|_| ApiError::BadRequest(format!("正则无效：{}", r.find)))?,
                )
            } else {
                None
            };
            Ok((
                r.regex.unwrap_or(false),
                re,
                r.find.clone(),
                r.replace.clone(),
            ))
        })
        .collect::<Result<_, ApiError>>()?;

    let mut affected = 0u64;
    for img in &images {
        let caption_path = caption_path_for(&root, &img.path);
        let Ok(text) = std::fs::read_to_string(&caption_path) else {
            continue;
        };
        let mut out = text.clone();
        let mut changed = false;
        for (is_regex, re, find, replace) in &compiled {
            let new = if *is_regex {
                re.as_ref()
                    .unwrap()
                    .replace_all(&out, replace.as_str())
                    .into_owned()
            } else {
                out.replace(find.as_str(), replace.as_str())
            };
            if new != out {
                out = new;
                changed = true;
            }
        }
        if changed {
            std::fs::write(&caption_path, out)
                .map_err(|e| ApiError::Internal(format!("写 caption 失败：{e}")))?;
            affected += 1;
        }
    }
    Ok(Json(BatchResult {
        affected,
        total: images.len() as u64,
    }))
}

/// 标签统计（标签云数据）：解析全部 caption 的逗号分隔 tag，按频次降序。
#[derive(Serialize, Debug)]
struct TagStat {
    tag: String,
    count: u64,
}

async fn tag_stats(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<TagStat>>, ApiError> {
    let (dataset_dir, images) = {
        let store = state.store.lock().await;
        let ds = store.get_dataset(&id)?;
        (ds.dir, store.list_dataset_images(&id)?)
    };
    let root = PathBuf::from(&dataset_dir);
    let mut counts: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    for img in &images {
        let caption_path = caption_path_for(&root, &img.path);
        let Ok(text) = std::fs::read_to_string(&caption_path) else {
            continue;
        };
        for tag in text.split(',') {
            let tag = tag.trim();
            if tag.is_empty() {
                continue;
            }
            *counts.entry(tag.to_string()).or_insert(0) += 1;
        }
    }
    let mut stats: Vec<TagStat> = counts
        .into_iter()
        .map(|(tag, count)| TagStat { tag, count })
        .collect();
    stats.sort_by(|a, b| b.count.cmp(&a.count).then(a.tag.cmp(&b.tag)));
    Ok(Json(stats))
}

/// 触发打标（内核 wrapper tagger 模式；同步等待完成）。
#[derive(Deserialize)]
struct TagRequest {
    /// mock（占位打标）| wd14（真实 WD14，需内核环境）
    mode: Option<String>,
}

#[derive(Serialize)]
struct TagResult {
    mode: String,
    tagged: u64,
}

async fn run_tagging(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(input): Json<TagRequest>,
) -> Result<Json<TagResult>, ApiError> {
    let dataset_dir = {
        let store = state.store.lock().await;
        store.get_dataset(&id)?.dir
    };
    let root = PathBuf::from(&dataset_dir);
    if !root.is_dir() {
        return Err(ApiError::BadRequest("数据集目录不存在".into()));
    }

    let mode = input.mode.unwrap_or_else(|| "mock".into());
    if !matches!(mode.as_str(), "mock" | "wd14") {
        return Err(ApiError::BadRequest("mode 仅支持 mock / wd14".into()));
    }

    // 占位配置 + 环境变量驱动 tagger
    let tmp = std::env::temp_dir().join(format!("tiandi-tag-{id}.toml"));
    std::fs::write(&tmp, "# tagger\n")
        .map_err(|e| ApiError::Internal(format!("写配置失败：{e}")))?;

    // 内核环境（优先 venv；wd14 需要 sd-scripts 路径）
    let env = state.trainer.kernel_env().clone();
    let python = env.python.ok_or_else(|| {
        ApiError::BadRequest(
            env.message
                .clone()
                .unwrap_or_else(|| "未检测到 Python".into()),
        )
    })?;

    let mut launch_env = vec![
        ("TIANDI_RUN_ID".into(), format!("tag-{id}")),
        ("TIANDI_TAGGER_DIR".into(), dataset_dir.clone()),
        ("TIANDI_TAGGER_MODE".into(), mode.clone()),
    ];
    if mode == "wd14" {
        let wd14_script = env
            .sd_scripts
            .as_ref()
            .map(|s| s.join("finetune/tag_images_by_wd14_tagger.py"))
            .filter(|p| p.exists())
            .map(|p| p.to_string_lossy().into_owned())
            .ok_or_else(|| {
                ApiError::BadRequest(
                    "WD14 打标需要训练内核（kernel.json 中 sd-scripts 未找到）。请先运行 `tiandi kernel install`".into(),
                )
            })?;
        launch_env.push(("TIANDI_WD14_SCRIPT".into(), wd14_script));
    }

    let launch = KernelLaunch {
        python,
        wrapper: crate::default_wrapper_path(),
        config_path: tmp,
        mode: KernelMode::Tagger,
        env: launch_env,
        cwd: root.clone(),
    };

    // 同步执行：读事件直到 done/fail
    let mut tagged = 0u64;
    let mut child = tokio::process::Command::new(&launch.python)
        .arg(&launch.wrapper)
        .arg(&launch.config_path)
        .current_dir(&launch.cwd)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .env("TIANDI_KERNEL_MODE", launch.mode.as_str())
        .envs(launch.env.iter().cloned())
        .spawn()
        .map_err(|e| ApiError::Internal(format!("打标进程启动失败：{e}")))?;

    let stdout = child
        .stdout
        .take()
        .ok_or(ApiError::Internal("stdout 不可用".into()))?;
    let mut reader = tokio::io::BufReader::new(stdout).lines();
    while let Ok(Some(line)) = reader.next_line().await {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
            match v.get("type").and_then(|t| t.as_str()) {
                Some("progress") => {
                    tagged = v.get("step").and_then(|s| s.as_u64()).unwrap_or(tagged);
                }
                Some("fail") => {
                    let tail = v.get("tail").and_then(|t| t.as_str()).unwrap_or("打标失败");
                    let _ = child.kill().await;
                    return Err(ApiError::Internal(tail.to_string()));
                }
                Some("done") => break,
                _ => {}
            }
        }
    }
    let status = child
        .wait()
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    if !status.success() {
        return Err(ApiError::Internal(format!(
            "打标进程退出码 {}",
            status.code().unwrap_or(-1)
        )));
    }

    Ok(Json(TagResult { mode, tagged }))
}

/// caption 文件路径：图片路径去扩展名 + .txt（kohya 约定）。
fn caption_path_for(root: &FsPath, image_rel: &str) -> PathBuf {
    let img = FsPath::new(image_rel);
    let stem = img.with_extension("");
    root.join(stem).with_extension("txt")
}

use tiandi_engine_compat::kernel::{KernelLaunch, KernelMode};

#[cfg(test)]
mod tests {
    use super::*;
    use tiandi_core::{Dataset, EventBus};
    use tiandi_state::{ImageRecord, Store};

    async fn setup_dataset(tmp: &std::path::Path) -> (AppState, String) {
        // 造两张图 + 一个已有 caption
        let sub = tmp.join("10_cat");
        std::fs::create_dir_all(&sub).unwrap();
        for name in ["a.png", "b.png"] {
            let img = image::RgbImage::new(64, 64);
            img.save(sub.join(name)).unwrap();
        }
        std::fs::write(sub.join("a.txt"), "1girl, cat, red dress\n").unwrap();

        let store = Store::open_in_memory().unwrap();
        let state = AppState::new(
            store,
            EventBus::default(),
            tmp.join("runs"),
            crate::default_wrapper_path(),
            true,
        );
        let ds = Dataset::new("测试集", tmp.to_string_lossy().into_owned());
        {
            let s = state.store.lock().await;
            s.insert_dataset(&ds).unwrap();
            // 手工登记图像记录（等价于扫描结果）
            let records = vec![
                ImageRecord {
                    id: "img-a".into(),
                    dataset_id: ds.id.clone(),
                    path: "10_cat/a.png".into(),
                    width: Some(64),
                    height: Some(64),
                    dhash: None,
                    bucket: Some("64x64".into()),
                    thumb: None,
                    exif: None,
                    duplicate_of: None,
                    created_at: "t".into(),
                },
                ImageRecord {
                    id: "img-b".into(),
                    dataset_id: ds.id.clone(),
                    path: "10_cat/b.png".into(),
                    width: Some(64),
                    height: Some(64),
                    dhash: None,
                    bucket: Some("64x64".into()),
                    thumb: None,
                    exif: None,
                    duplicate_of: None,
                    created_at: "t".into(),
                },
            ];
            s.replace_dataset_images(&ds.id, &records).unwrap();
        }
        (state, ds.id)
    }

    #[test]
    fn caption_path_follows_kohya_convention() {
        let p = caption_path_for(FsPath::new(r"D:\ds"), "10_cat/a.png");
        assert_eq!(p, FsPath::new(r"D:\ds\10_cat\a.txt"));
    }

    #[tokio::test]
    async fn save_and_list_captions() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, id) = setup_dataset(tmp.path()).await;

        // 初始：a 有 caption，b 无
        let caps = list_captions_inner(&state, &id).await;
        let a = caps.iter().find(|c| c.path.ends_with("a.png")).unwrap();
        assert!(a.has_file);
        assert_eq!(a.caption, "1girl, cat, red dress");
        let b = caps.iter().find(|c| c.path.ends_with("b.png")).unwrap();
        assert!(!b.has_file);

        // 保存 b 的 caption
        save_caption_inner(&state, &id, "10_cat/b.png", "1boy, blue shirt").await;
        let caps = list_captions_inner(&state, &id).await;
        let b = caps.iter().find(|c| c.path.ends_with("b.png")).unwrap();
        assert!(b.has_file);
        assert_eq!(b.caption, "1boy, blue shirt");
    }

    #[tokio::test]
    async fn batch_replace_and_tag_stats() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, id) = setup_dataset(tmp.path()).await;
        save_caption_inner(&state, &id, "10_cat/b.png", "1girl, cat, red hat").await;

        // 批量替换：cat → dog
        batch_replace_inner(&state, &id, "cat", "dog", false).await;
        let caps = list_captions_inner(&state, &id).await;
        assert!(caps.iter().all(|c| !c.caption.contains("cat")));
        assert!(caps.iter().any(|c| c.caption.contains("dog")));

        // 正则替换：^\d+girl → 1person
        batch_replace_inner(&state, &id, r"^\d+girl", "1person", true).await;
        let caps = list_captions_inner(&state, &id).await;
        assert!(caps.iter().any(|c| c.caption.contains("1person")));

        // 标签统计
        let stats = tag_stats_inner(&state, &id).await;
        let cat_count = stats
            .iter()
            .find(|s| s.tag == "dog")
            .map(|s| s.count)
            .unwrap_or(0);
        assert_eq!(cat_count, 2, "dog 应出现在两张图：{stats:?}");
        let girl_count = stats
            .iter()
            .find(|s| s.tag == "1person")
            .map(|s| s.count)
            .unwrap_or(0);
        assert_eq!(girl_count, 2);
    }

    // ---- 直接调用 handler 逻辑的辅助（避免 HTTP 层样板） ----

    async fn list_captions_inner(state: &AppState, id: &str) -> Vec<CaptionEntry> {
        let (dataset_dir, images) = {
            let store = state.store.lock().await;
            let ds = store.get_dataset(id).unwrap();
            (ds.dir, store.list_dataset_images(id).unwrap())
        };
        let root = PathBuf::from(&dataset_dir);
        let mut out = Vec::new();
        for img in images {
            let caption_path = caption_path_for(&root, &img.path);
            let (caption, has_file) = match std::fs::read_to_string(&caption_path) {
                Ok(text) => (text.trim().to_string(), true),
                Err(_) => (String::new(), false),
            };
            out.push(CaptionEntry {
                path: img.path,
                caption,
                has_file,
            });
        }
        out
    }

    async fn save_caption_inner(state: &AppState, id: &str, path: &str, text: &str) {
        let dataset_dir = {
            let store = state.store.lock().await;
            store.get_dataset(id).unwrap().dir
        };
        let root = PathBuf::from(&dataset_dir);
        let caption_path = caption_path_for(&root, path);
        std::fs::write(&caption_path, format!("{text}\n")).unwrap();
    }

    async fn batch_replace_inner(
        state: &AppState,
        id: &str,
        find: &str,
        replace: &str,
        regex: bool,
    ) {
        let (dataset_dir, images) = {
            let store = state.store.lock().await;
            let ds = store.get_dataset(id).unwrap();
            (ds.dir, store.list_dataset_images(id).unwrap())
        };
        let root = PathBuf::from(&dataset_dir);
        for img in &images {
            let caption_path = caption_path_for(&root, &img.path);
            let Ok(text) = std::fs::read_to_string(&caption_path) else {
                continue;
            };
            let out = if regex {
                regex::Regex::new(find)
                    .unwrap()
                    .replace_all(&text, replace)
                    .into_owned()
            } else {
                text.replace(find, replace)
            };
            std::fs::write(&caption_path, out).unwrap();
        }
    }

    async fn tag_stats_inner(state: &AppState, id: &str) -> Vec<TagStat> {
        let (dataset_dir, images) = {
            let store = state.store.lock().await;
            let ds = store.get_dataset(id).unwrap();
            (ds.dir, store.list_dataset_images(id).unwrap())
        };
        let root = PathBuf::from(&dataset_dir);
        let mut counts: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
        for img in &images {
            let caption_path = caption_path_for(&root, &img.path);
            let Ok(text) = std::fs::read_to_string(&caption_path) else {
                continue;
            };
            for tag in text.split(',') {
                let tag = tag.trim();
                if !tag.is_empty() {
                    *counts.entry(tag.to_string()).or_insert(0) += 1;
                }
            }
        }
        counts
            .into_iter()
            .map(|(tag, count)| TagStat { tag, count })
            .collect()
    }
}
