//! 缩略图：保持纵横比缩放到目标尺寸（rayon 并行调用方负责）。

use std::path::Path;

use image::GenericImageView;

/// 生成缩略图并写入 `dest`（JPEG，质量 80）。
/// `max_size` 为最长边。
pub fn make_thumbnail(src: &Path, dest: &Path, max_size: u32) -> Result<(), ThumbError> {
    let img = image::open(src).map_err(ThumbError::Decode)?;
    let (w, h) = img.dimensions();
    let (tw, th) = if w >= h {
        let th = (h as u64 * max_size as u64 / w as u64).max(1) as u32;
        (max_size, th)
    } else {
        let tw = (w as u64 * max_size as u64 / h as u64).max(1) as u32;
        (tw, max_size)
    };
    let thumb = img.thumbnail(tw, th);
    thumb
        .save_with_format(dest, image::ImageFormat::Jpeg)
        .map_err(ThumbError::Save)?;
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum ThumbError {
    #[error("图像解码失败: {0}")]
    Decode(image::ImageError),
    #[error("缩略图保存失败: {0}")]
    Save(image::ImageError),
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
}
