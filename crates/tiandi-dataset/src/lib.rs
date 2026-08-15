//! 数据管线：图像扫描、dHash 去重、缩略图、分辨率桶、统计（rayon 并行）。
//!
//! 对应 PRD FR-201~205、FR-203 桶系统；算法参考 kohya sd-scripts 的 bucket 与
//! ai-toolkit `buckets.py`（保面积 + 对齐），实现为 Rust 原生（docs/architecture.md §7）。

pub mod bucket;
pub mod hash;
pub mod scan;
pub mod thumb;

pub use bucket::{assign_buckets, candidate_buckets, BucketInfo};
pub use hash::dhash64;
pub use scan::{scan_dataset_dir, ScanOptions, ScanReport, ScanResult};
pub use thumb::make_thumbnail;

use serde::Serialize;

/// 单张图像的处理结果（扫描产物，供入库与 UI 展示）。
#[derive(Debug, Clone, Serialize)]
pub struct ImageEntry {
    /// 相对数据集根的路径（正斜杠）
    pub path: String,
    pub width: u32,
    pub height: u32,
    /// 64 位 dHash（hex 字符串，去重键）
    pub dhash: String,
    /// 分配的桶（"1024x1024" 格式）
    pub bucket: Option<String>,
    /// 缩略图相对路径（thumb 目录下）
    pub thumb: Option<String>,
    /// EXIF 摘要（JSON）
    pub exif: Option<serde_json::Value>,
}
