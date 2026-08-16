//! 缩略图：保持纵横比缩放到目标尺寸（rayon 并行调用方负责）。
//!
//! 写入为原子操作：先写同目录唯一临时文件，成功后 `rename` 到目标，
//! 避免并行写同一哈希时出现文件撕裂（TOCTOU：先检查后写入非原子）。

use std::path::{Path, PathBuf};

use image::GenericImageView;

/// 生成缩略图并写入 `dest`（JPEG，质量 80）。
/// `max_size` 为最长边。
pub fn make_thumbnail(src: &Path, dest: &Path, max_size: u32) -> Result<(), ThumbError> {
    let img = image::open(src).map_err(ThumbError::Decode)?;
    make_thumbnail_from_image(&img, dest, max_size)
}

/// 基于已解码图像生成缩略图并原子写入 `dest`。
/// 扫描流程复用同一次解码结果（应用 EXIF Orientation 后），避免每文件二次解码。
pub fn make_thumbnail_from_image(
    img: &image::DynamicImage,
    dest: &Path,
    max_size: u32,
) -> Result<(), ThumbError> {
    let (w, h) = img.dimensions();
    let (tw, th) = if w >= h {
        let th = (h as u64 * max_size as u64 / w as u64).max(1) as u32;
        (max_size, th)
    } else {
        let tw = (w as u64 * max_size as u64 / h as u64).max(1) as u32;
        (tw, max_size)
    };
    let thumb = img.thumbnail(tw, th);

    // 原子写入：先写同目录唯一临时文件，成功后 rename 到目标；
    // 任一步失败都清理临时文件，避免残留。
    let tmp = temp_path_for(dest);
    if let Err(e) = thumb.save_with_format(&tmp, image::ImageFormat::Jpeg) {
        let _ = std::fs::remove_file(&tmp);
        return Err(ThumbError::Save(e));
    }
    if let Err(e) = std::fs::rename(&tmp, dest) {
        let _ = std::fs::remove_file(&tmp);
        return Err(ThumbError::Rename(e));
    }
    Ok(())
}

/// 同目录唯一临时文件名（进程 id + 线程 id，避免多线程写同一哈希时互相覆盖；
/// 同目录保证 `rename` 不跨文件系统，是原子操作）。
fn temp_path_for(dest: &Path) -> PathBuf {
    let name = dest.file_name().and_then(|n| n.to_str()).unwrap_or("thumb");
    dest.with_file_name(format!(
        ".{name}.tmp-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ))
}

#[derive(Debug, thiserror::Error)]
pub enum ThumbError {
    #[error("图像解码失败: {0}")]
    Decode(image::ImageError),
    #[error("缩略图保存失败: {0}")]
    Save(image::ImageError),
    #[error("缩略图写入失败: {0}")]
    Rename(std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn thumbnail_resizes_keeping_aspect() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src.png");
        let img = RgbImage::from_pixel(800, 400, Rgb([10, 20, 30]));
        img.save(&src).unwrap();
        let dest = dir.path().join("thumb.jpg");
        make_thumbnail(&src, &dest, 256).unwrap();
        let thumb = image::open(&dest).unwrap();
        let (w, h) = thumb.dimensions();
        assert_eq!(w, 256);
        assert_eq!(h, 128);
    }

    #[test]
    fn thumbnail_write_is_atomic_and_leaves_no_temp() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src.png");
        let img = RgbImage::from_pixel(800, 400, Rgb([10, 20, 30]));
        img.save(&src).unwrap();
        let dest = dir.path().join("thumb.jpg");

        make_thumbnail(&src, &dest, 256).unwrap();
        assert_eq!(image::open(&dest).unwrap().dimensions(), (256, 128));
        // 同目录不应残留临时文件
        let leftovers: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".tmp-"))
            .collect();
        assert!(leftovers.is_empty(), "不应残留临时文件：{leftovers:?}");

        // 覆盖写入（重复生成同一哈希）同样原子
        make_thumbnail(&src, &dest, 128).unwrap();
        assert_eq!(image::open(&dest).unwrap().dimensions(), (128, 64));
        let leftovers: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".tmp-"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "覆盖写入后不应残留临时文件：{leftovers:?}"
        );
    }
}
