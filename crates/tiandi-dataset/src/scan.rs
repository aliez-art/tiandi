//! 数据集扫描主流程：收集 → 并行处理 → 去重 → 分桶 → 统计。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use image::GenericImageView;
use rayon::prelude::*;
use serde::Serialize;

use crate::bucket::{assign_buckets, candidate_buckets, BucketInfo};
use crate::hash::{dhash64, find_duplicates};
use crate::thumb::make_thumbnail_from_image;
use crate::ImageEntry;

/// 支持的图片扩展名。
const IMAGE_EXTS: [&str; 7] = ["jpg", "jpeg", "png", "webp", "bmp", "gif", "tiff"];

/// 解码前尺寸上限（防解压炸弹 / 解码内存峰值）：任一边超过或像素总数超过即判无效。
const MAX_DIM: u32 = 16_384;
const MAX_PIXELS: u64 = 64_000_000;

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
    /// 重复判定汉明距离阈值（0..=64）
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
    /// 无法解码/损坏/超限的文件数
    pub invalid: u64,
    /// 目录遍历失败（权限/IO）的条目数
    pub scan_errors: u64,
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
/// - rayon 并行：解码一次（应用 EXIF Orientation）→ 尺寸、dHash、缩略图、EXIF
/// - 解码前用头部尺寸检查防超限（防解压炸弹）
/// - 近重复分组（dHash 汉明距离，并查集传递闭包）
/// - 分辨率桶分配
pub fn scan_dataset_dir(root: &Path, options: &ScanOptions) -> Result<ScanResult, ScanError> {
    let started = std::time::Instant::now();

    // 0. 参数边界校验
    if options.hash_threshold > 64 {
        return Err(ScanError(format!(
            "hash_threshold 超出范围（0..=64）：{}",
            options.hash_threshold
        )));
    }
    if options.target_resolution < 64 {
        return Err(ScanError(format!(
            "target_resolution 必须 ≥ 64（当前 {}）",
            options.target_resolution
        )));
    }
    if !root.is_dir() {
        return Err(ScanError(format!(
            "数据集目录不存在或不是目录：{}",
            root.display()
        )));
    }

    // 1. 收集候选文件（遍历错误计入 scan_errors，不再静默丢弃）
    let mut files: Vec<PathBuf> = Vec::new();
    let mut scan_errors: u64 = 0;
    for entry in walkdir::WalkDir::new(root).follow_links(false) {
        match entry {
            Ok(e) => {
                if !e.file_type().is_file() {
                    continue;
                }
                let ext = e
                    .path()
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.to_ascii_lowercase());
                if ext.is_some_and(|e| IMAGE_EXTS.contains(&e.as_str())) {
                    files.push(e.into_path());
                }
            }
            Err(_) => scan_errors += 1,
        }
    }

    // 准备缩略图目录
    if let Some(td) = &options.thumb_dir {
        std::fs::create_dir_all(td).map_err(|e| ScanError(format!("创建缩略图目录失败：{e}")))?;
    }

    // 2. 并行处理：单次解码（应用 EXIF Orientation）→ 尺寸/哈希/缩略图/EXIF
    let thumb_dir = options.thumb_dir.clone();
    let thumb_size = options.thumb_size;
    let processed: Vec<Option<ProcessedItem>> = files
        .par_iter()
        .map(|path| {
            // 廉价头部检查（仅读文件头，不解码）：超限文件计入 invalid 并跳过
            let (w, h) = match image::image_dimensions(path) {
                Ok(d) => d,
                Err(_) => return None, // 损坏/无法解码
            };
            if w > MAX_DIM || h > MAX_DIM || (w as u64) * (h as u64) > MAX_PIXELS {
                return None; // 超限（防解压炸弹/内存峰值）
            }
            let (decoded, _orientation) = match decode_with_orientation(path) {
                Some(v) => v,
                None => return None,
            };
            // 宽高基于应用 Orientation 后的实际图像（手机竖拍图为竖版）
            let (w, h) = decoded.dimensions();
            let hash = dhash64(&decoded);
            let thumb = thumb_dir.as_ref().and_then(|td| {
                let name = format!("{:016x}.jpg", hash);
                let dest = td.join(&name);
                // 已存在则跳过（避免重复劳动）；写入本身是原子的（见 thumb.rs）
                if !dest.exists() {
                    make_thumbnail_from_image(&decoded, &dest, thumb_size).ok()?;
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
            scan_errors,
            duplicate_groups,
            buckets,
            avg_width,
            avg_height,
            elapsed_ms: started.elapsed().as_millis() as u64,
        },
    })
}

/// 解码并应用 EXIF Orientation。
///
/// Orientation（显示所需变换）：1=原样、2=水平镜像、3=180°、4=垂直镜像、
/// 5=水平镜像+270°CW、6=90°CW、7=水平镜像+90°CW、8=270°CW（90°CCW）。
fn decode_with_orientation(path: &Path) -> Option<(image::DynamicImage, u8)> {
    let img = image::open(path).ok()?;
    let orientation = read_orientation(path);
    Some((apply_orientation(img, orientation), orientation))
}

fn apply_orientation(img: image::DynamicImage, orientation: u8) -> image::DynamicImage {
    match orientation {
        2 => image::DynamicImage::ImageRgba8(image::imageops::flip_horizontal(&img)),
        3 => img.rotate180(),
        4 => image::DynamicImage::ImageRgba8(image::imageops::flip_vertical(&img)),
        5 => image::DynamicImage::ImageRgba8(image::imageops::flip_horizontal(&img)).rotate270(),
        6 => img.rotate90(),
        7 => image::DynamicImage::ImageRgba8(image::imageops::flip_horizontal(&img)).rotate90(),
        8 => img.rotate270(),
        _ => img,
    }
}

/// 读取 EXIF Orientation 字段（1..=8，缺省按 1 处理）。
fn read_orientation(path: &Path) -> u8 {
    use exif::{In, Reader, Tag};
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return 1,
    };
    let mut bufreader = std::io::BufReader::new(file);
    let exif = match Reader::new().read_from_container(&mut bufreader) {
        Ok(e) => e,
        Err(_) => return 1,
    };
    exif.get_field(Tag::Orientation, In::PRIMARY)
        .and_then(|v| v.value.get_uint(0))
        .filter(|v| (1..=8).contains(v))
        .map(|v| v as u8)
        .unwrap_or(1)
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

    /// 构造带 EXIF Orientation 的 JPEG（横图 w×h + APP1(Exif) 段，垂直上升图案）。
    fn jpeg_with_orientation(w: u32, h: u32, orientation: u16) -> Vec<u8> {
        let mut img = RgbImage::new(w, h);
        for (x, _y, p) in img.enumerate_pixels_mut() {
            let v = if x >= w / 2 { 220 } else { 20 }; // 垂直上升（左暗右亮）
            *p = Rgb([v, v / 2, v / 3]);
        }
        let mut jpeg = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut jpeg),
            image::ImageFormat::Jpeg,
        )
        .unwrap();
        inject_exif_orientation(&jpeg, orientation)
    }

    /// 在 SOI 后插入 APP1(Exif) 段：Exif\0\0 + TIFF(LE) + IFD0（仅 Orientation 条目）。
    fn inject_exif_orientation(jpeg: &[u8], orientation: u16) -> Vec<u8> {
        assert!(jpeg.starts_with(&[0xFF, 0xD8]), "应以 SOI 开头");
        let mut payload: Vec<u8> = Vec::new();
        payload.extend_from_slice(b"Exif\0\0");
        payload.extend_from_slice(&[0x49, 0x49, 0x2A, 0x00]); // "II*\0"
        payload.extend_from_slice(&8u32.to_le_bytes()); // IFD0 偏移
        payload.extend_from_slice(&1u16.to_le_bytes()); // IFD0 条目数
        payload.extend_from_slice(&0x0112u16.to_le_bytes()); // Orientation tag
        payload.extend_from_slice(&3u16.to_le_bytes()); // SHORT
        payload.extend_from_slice(&1u32.to_le_bytes()); // count
        payload.extend_from_slice(&(orientation as u32).to_le_bytes()); // 值（4 字节槽）
        payload.extend_from_slice(&0u32.to_le_bytes()); // 下一个 IFD 偏移
        let mut out = Vec::with_capacity(jpeg.len() + payload.len() + 4);
        out.extend_from_slice(&jpeg[..2]); // SOI
        out.extend_from_slice(&[0xFF, 0xE1]); // APP1
        out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(&payload);
        out.extend_from_slice(&jpeg[2..]);
        out
    }

    /// 构造仅含头部（IHDR + 空 IDAT + IEND）的 PNG，用于验证超大尺寸上限。
    /// `image::image_dimensions` 只读头部，无需真实像素数据。
    fn png_with_dimensions(w: u32, h: u32) -> Vec<u8> {
        fn crc32(data: &[u8]) -> u32 {
            let mut crc: u32 = 0xFFFF_FFFF;
            for &b in data {
                crc ^= b as u32;
                for _ in 0..8 {
                    let mask = (crc & 1).wrapping_neg();
                    crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
                }
            }
            !crc
        }
        fn chunk(typ: &[u8; 4], data: &[u8]) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(&(data.len() as u32).to_be_bytes());
            out.extend_from_slice(typ);
            out.extend_from_slice(data);
            let mut crc_input = Vec::with_capacity(4 + data.len());
            crc_input.extend_from_slice(typ);
            crc_input.extend_from_slice(data);
            out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
            out
        }
        let mut out = Vec::new();
        out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&w.to_be_bytes());
        ihdr.extend_from_slice(&h.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit RGB, deflate, adaptive, 无隔行
        out.extend_from_slice(&chunk(b"IHDR", &ihdr));
        // 空 deflate 流作为占位 IDAT（image_dimensions 不解压）
        out.extend_from_slice(&chunk(b"IDAT", &[0x78, 0x01, 0x01, 0x00, 0x00, 0xFF, 0xFF]));
        out.extend_from_slice(&chunk(b"IEND", &[]));
        out
    }

    fn parse_bucket(label: &str) -> (u32, u32) {
        let mut it = label.split('x');
        (
            it.next().unwrap().parse().unwrap(),
            it.next().unwrap().parse().unwrap(),
        )
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
        assert_eq!(result.report.scan_errors, 0);
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
        assert_eq!(result.report.scan_errors, 0);
        assert!(result.report.duplicate_groups.is_empty());
        assert!(result.report.buckets.is_empty());
    }

    #[test]
    fn parameter_bounds_are_validated() {
        let dir = tempfile::tempdir().unwrap();
        let opts = ScanOptions {
            hash_threshold: 65,
            ..ScanOptions::default()
        };
        assert!(
            scan_dataset_dir(dir.path(), &opts).is_err(),
            "hash_threshold > 64 应报错"
        );
        let opts = ScanOptions {
            target_resolution: 0,
            ..ScanOptions::default()
        };
        assert!(
            scan_dataset_dir(dir.path(), &opts).is_err(),
            "target_resolution=0 应报错"
        );
        let opts = ScanOptions {
            target_resolution: 63,
            ..ScanOptions::default()
        };
        assert!(
            scan_dataset_dir(dir.path(), &opts).is_err(),
            "target_resolution<64 应报错"
        );
        let opts = ScanOptions {
            target_resolution: 64,
            ..ScanOptions::default()
        };
        assert!(
            scan_dataset_dir(dir.path(), &opts).is_ok(),
            "target_resolution=64 应合法"
        );
    }

    #[test]
    fn missing_root_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("does-not-exist");
        assert!(
            scan_dataset_dir(&missing, &ScanOptions::default()).is_err(),
            "根目录不存在应返回 Err 而非静默空结果"
        );
        // 根路径是文件也应报错
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "x").unwrap();
        assert!(
            scan_dataset_dir(&file, &ScanOptions::default()).is_err(),
            "根路径不是目录应报错"
        );
    }

    #[test]
    fn oversized_images_counted_as_invalid() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("ds");
        std::fs::create_dir_all(&root).unwrap();
        // 宽超 16384（头部即可读出，无需真实像素）
        let too_wide = root.join("too-wide.png");
        std::fs::write(&too_wide, png_with_dimensions(20000, 100)).unwrap();
        assert_eq!(
            image::image_dimensions(&too_wide).unwrap(),
            (20000, 100),
            "构造的 PNG 头部应可读（否则该用例测不到上限逻辑）"
        );
        // 像素总数超 6400 万（10000×7000 = 7000 万）
        std::fs::write(
            root.join("too-many-pixels.png"),
            png_with_dimensions(10000, 7000),
        )
        .unwrap();
        // 正常图
        make_file(&root, "ok.png", 512, 512, 0);
        let result = scan_dataset_dir(&root, &ScanOptions::default()).unwrap();
        assert_eq!(result.report.total, 1);
        assert_eq!(result.report.invalid, 2, "两张超限图应计入 invalid");
        assert!(result.entries.iter().all(|e| e.path == "ok.png"));
    }

    #[test]
    fn exif_orientation_6_yields_portrait_entry_bucket_and_thumb() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("ds");
        std::fs::create_dir_all(&root).unwrap();
        // 横图 1536×768 + EXIF Orientation=6（显示需顺时针旋转 90° → 竖版 768×1536）
        let jpeg = jpeg_with_orientation(1536, 768, 6);
        let path = root.join("phone.jpg");
        std::fs::write(&path, &jpeg).unwrap();

        // decode_with_orientation 直接验证：旋转方向 + 像素分布（右半亮区应转到下半）
        let (img, ori) = decode_with_orientation(&path).unwrap();
        assert_eq!(ori, 6);
        assert_eq!(img.dimensions(), (768, 1536), "Orientation 6 应旋转为竖版");
        let p_top = img.get_pixel(0, 100)[0];
        let p_bottom = img.get_pixel(0, 900)[0];
        assert!(
            p_top < 100 && p_bottom > 100,
            "旋转后应上暗下亮（原右半亮区转到底部）：{p_top}/{p_bottom}"
        );

        let opts = ScanOptions {
            thumb_dir: Some(root.join("thumbs")),
            ..ScanOptions::default()
        };
        let result = scan_dataset_dir(&root, &opts).unwrap();
        assert_eq!(result.report.total, 1);
        assert_eq!(result.report.invalid, 0);
        let e = &result.entries[0];
        assert_eq!((e.width, e.height), (768, 1536));
        let bucket = e.bucket.as_deref().unwrap();
        let (bw, bh) = parse_bucket(bucket);
        assert!(bh > bw, "竖拍图应分到纵向桶：{bucket}");

        let thumb_path = root.join(e.thumb.as_deref().unwrap());
        let thumb = image::open(&thumb_path).unwrap();
        assert!(
            thumb.height() > thumb.width(),
            "缩略图应为竖版（{}x{}）",
            thumb.width(),
            thumb.height()
        );
    }
}
