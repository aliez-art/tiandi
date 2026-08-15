//! 数据集扫描主流程：收集 → 并行处理 → 去重 → 分桶 → 统计。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use image::GenericImageView;
use rayon::prelude::*;
use serde::Serialize;

use crate::bucket::{assign_buckets, candidate_buckets, BucketInfo};
use crate::hash::{dhash64, find_duplicates};
use crate::thumb::make_thumbnail;
use crate::ImageEntry;

/// 支持的图片扩展名。
const IMAGE_EXTS: [&str; 7] = ["jpg", "jpeg", "png", "webp", "bmp", "gif", "tiff"];

/// 并行处理中的中间结果：路径 / 宽 / 高 / dHash / 缩略图相对路径 / EXIF。
type ProcessedItem = (
    PathBuf,
    u32,
    u32,
    u64,
    Option<String>,
    Option<serde_json::Value>,
);
/// 有效条目（相对路径化后）。
type ValidItem = (
    String,
    u32,
    u32,
    u64,
    Option<String>,
    Option<serde_json::Value>,
);

/// 扫描选项。
#[derive(Debug, Clone)]
pub struct ScanOptions {
    /// 目标分辨率（SDXL 默认 1024）
    pub target_resolution: u32,
    /// 桶步进（kohya bucket_reso_steps）
    pub bucket_steps: u32,
    /// 桶面积约束（相对 target² 的上下限）
    pub bucket_min_ratio: f64,
    pub bucket_max_ratio: f64,
    /// 缩略图最长边
    pub thumb_size: u32,
    /// 重复判定汉明距离阈值
    pub hash_threshold: u32,
    /// 缩略图输出目录（None = 不生成）
    pub thumb_dir: Option<PathBuf>,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            target_resolution: 1024,
            bucket_steps: 64,
            bucket_min_ratio: 0.85,
            bucket_max_ratio: 1.15,
            thumb_size: 256,
            hash_threshold: 5,
            thumb_dir: None,
        }
    }
}

/// 扫描报告。
#[derive(Debug, Clone, Serialize)]
pub struct ScanReport {
    /// 有效图像数
    pub total: u64,
    /// 无法解码/损坏的文件数
    pub invalid: u64,
    /// 重复组（每组为文件路径列表）
    pub duplicate_groups: Vec<Vec<String>>,
    /// 桶分布（label -> 数量，按数量降序）
    pub buckets: Vec<(String, u64)>,
    pub avg_width: f64,
    pub avg_height: f64,
    /// 处理耗时（秒）
    pub elapsed_ms: u64,
}

/// 扫描结果：条目 + 报告。
#[derive(Debug, Clone, Serialize)]
pub struct ScanResult {
    pub entries: Vec<ImageEntry>,
    pub report: ScanReport,
}

/// 扫描数据集目录。
///
/// - 递归收集图片（`N_` 前缀子目录约定内任意层级）
/// - rayon 并行：解码尺寸、dHash、缩略图、EXIF
/// - 近重复分组（dHash 汉明距离）
/// - 分辨率桶分配
pub fn scan_dataset_dir(root: &Path, options: &ScanOptions) -> Result<ScanResult, ScanError> {
    let started = std::time::Instant::now();

    // 1. 收集候选文件
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in walkdir::WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let ext = entry
            .path()
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase());
        if ext.is_some_and(|e| IMAGE_EXTS.contains(&e.as_str())) {
            files.push(entry.into_path());
        }
    }

    // 准备缩略图目录
    if let Some(td) = &options.thumb_dir {
        std::fs::create_dir_all(td).map_err(|e| ScanError(format!("创建缩略图目录失败：{e}")))?;
    }

    // 2. 并行处理：解码/哈希/缩略图/EXIF
    let thumb_dir = options.thumb_dir.clone();
    let thumb_size = options.thumb_size;
    let processed: Vec<Option<ProcessedItem>> = files
        .par_iter()
        .map(|path| {
            let decoded = match image::open(path) {
                Ok(img) => img,
                Err(_) => return None, // 损坏/无法解码
            };
            let (w, h) = decoded.dimensions();
            let hash = dhash64(&decoded);
            let thumb = thumb_dir.as_ref().and_then(|td| {
                let name = format!("{:016x}.jpg", hash);
                let dest = td.join(&name);
                if !dest.exists() {
                    make_thumbnail(path, &dest, thumb_size).ok()?;
                }
                Some(format!("thumbs/{name}"))
            });
            let exif = read_exif_summary(path);
            Some((path.clone(), w, h, hash, thumb, exif))
        })
        .collect();

    // 3. 拆分有效/无效，相对路径化
    let mut valid: Vec<ValidItem> = Vec::new();
    for item in processed.into_iter().flatten() {
        let (path, w, h, hash, thumb, exif) = item;
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        valid.push((rel, w, h, hash, thumb, exif));
    }
    let invalid = files.len() as u64 - valid.len() as u64;

    // 4. 去重（按路径排序保证确定性：重复组内先出现的为主）
    valid.sort_by(|a, b| a.0.cmp(&b.0));
    let hash_list: Vec<(String, u64)> = valid.iter().map(|v| (v.0.clone(), v.3)).collect();
    let groups = find_duplicates(&hash_list, options.hash_threshold);
    let duplicate_groups: Vec<Vec<String>> = groups
        .iter()
        .map(|g| g.iter().map(|(p, _)| p.clone()).collect())
        .collect();

    // 5. 桶分配
    let candidates = candidate_buckets(
        options.target_resolution,
        options.bucket_steps,
        options.bucket_min_ratio,
        options.bucket_max_ratio,
    );
    let dims: Vec<(u32, u32)> = valid.iter().map(|v| (v.1, v.2)).collect();
    let assigned: Vec<Option<BucketInfo>> = assign_buckets(&dims, &candidates);

    let entries: Vec<ImageEntry> = valid
        .into_iter()
        .zip(assigned)
        .map(|(v, b)| ImageEntry {
            path: v.0,
            width: v.1,
            height: v.2,
            dhash: format!("{:016x}", v.3),
            bucket: b.map(|b| b.label()),
            thumb: v.4,
            exif: v.5,
        })
        .collect();

    // 6. 统计
    let total = entries.len() as u64;
    let mut bucket_map: BTreeMap<String, u64> = BTreeMap::new();
    for e in &entries {
        if let Some(b) = &e.bucket {
            *bucket_map.entry(b.clone()).or_insert(0) += 1;
        }
    }
    let mut buckets: Vec<(String, u64)> = bucket_map.into_iter().collect();
    buckets.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    let sum_w: u64 = entries.iter().map(|e| e.width as u64).sum();
    let sum_h: u64 = entries.iter().map(|e| e.height as u64).sum();
    let avg_width = if total > 0 {
        sum_w as f64 / total as f64
    } else {
        0.0
    };
    let avg_height = if total > 0 {
        sum_h as f64 / total as f64
    } else {
        0.0
    };

    Ok(ScanResult {
        entries,
        report: ScanReport {
            total,
            invalid,
            duplicate_groups,
            buckets,
            avg_width,
            avg_height,
            elapsed_ms: started.elapsed().as_millis() as u64,
        },
    })
}

/// EXIF 摘要（Orientation / DateTimeOriginal / 相机）。
fn read_exif_summary(path: &Path) -> Option<serde_json::Value> {
    use exif::{In, Reader, Tag};
    let file = std::fs::File::open(path).ok()?;
    let mut bufreader = std::io::BufReader::new(file);
    let exif = Reader::new().read_from_container(&mut bufreader).ok()?;
    let mut map = serde_json::Map::new();
    for tag in [
        Tag::Orientation,
        Tag::DateTimeOriginal,
        Tag::Make,
        Tag::Model,
        Tag::PixelXDimension,
        Tag::PixelYDimension,
    ] {
        if let Some(v) = exif.get_field(tag, In::PRIMARY) {
            map.insert(
                tag.to_string(),
                serde_json::Value::String(v.display_value().to_string()),
            );
        }
    }
    if map.is_empty() {
        None
    } else {
        Some(serde_json::Value::Object(map))
    }
}

#[derive(Debug, thiserror::Error)]
#[error("数据集扫描错误: {0}")]
pub struct ScanError(String);

impl ScanError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn make_file(dir: &Path, name: &str, w: u32, h: u32, pattern: u8) {
        let mut img = RgbImage::new(w, h);
        for (x, y, p) in img.enumerate_pixels_mut() {
            let v = match pattern {
                // 0: 垂直上升（左暗右亮，dHash 有上升沿）
                0 => {
                    if x >= w / 2 {
                        220
                    } else {
                        20
                    }
                }
                // 1: 粗棋盘格（32px）
                _ => {
                    if (x / 32 + y / 32) % 2 == 0 {
                        220
                    } else {
                        20
                    }
                }
            };
            *p = Rgb([v, v / 2, v / 3]);
        }
        img.save(dir.join(name)).unwrap();
    }

    #[test]
    fn scan_collects_counts_buckets_and_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("ds");
        let sub = root.join("10_cat");
        std::fs::create_dir_all(&sub).unwrap();
        // 1 张正方形 + 2 张近重复宽图 + 1 张损坏
        make_file(&sub, "a.png", 1024, 1024, 30);
        make_file(&sub, "b.png", 1536, 768, 90);
        make_file(&sub, "b-copy.png", 1536, 768, 90);
        std::fs::write(root.join("broken.png"), b"not an image").unwrap();

        let opts = ScanOptions {
            target_resolution: 1024,
            bucket_steps: 64,
            bucket_min_ratio: 0.85,
            bucket_max_ratio: 1.15,
            thumb_size: 256,
            hash_threshold: 5,
            thumb_dir: Some(root.join("thumbs")),
        };
        let result = scan_dataset_dir(&root, &opts).unwrap();

        assert_eq!(result.report.total, 3);
        assert_eq!(result.report.invalid, 1);
        assert_eq!(result.report.duplicate_groups.len(), 1);
        assert_eq!(result.report.duplicate_groups[0].len(), 2);

        // 桶：正方形 → 1024x1024；宽图 → 横向桶
        let square = result
            .entries
            .iter()
            .find(|e| e.path.ends_with("a.png"))
            .unwrap();
        assert_eq!(square.bucket.as_deref(), Some("1024x1024"));
        let wide = result
            .entries
            .iter()
            .find(|e| e.path.ends_with("b.png"))
            .unwrap();
        let wb = wide.bucket.as_deref().unwrap();
        assert!(
            wb.starts_with("1536x768")
                || wb.starts_with("1472x768")
                || wb.split('x').next().unwrap().parse::<u32>().unwrap() > 1024,
            "宽图桶 {wb}"
        );

        // 缩略图已生成
        assert!(square.thumb.is_some());
        assert!(root.join("thumbs").is_dir());

        // 桶分布汇总
        let total_in_buckets: u64 = result.report.buckets.iter().map(|(_, n)| n).sum();
        assert_eq!(total_in_buckets, 3);
    }

    #[test]
    fn empty_dir_scan_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let result = scan_dataset_dir(dir.path(), &ScanOptions::default()).unwrap();
        assert_eq!(result.report.total, 0);
        assert_eq!(result.report.invalid, 0);
        assert!(result.report.duplicate_groups.is_empty());
        assert!(result.report.buckets.is_empty());
    }
}
