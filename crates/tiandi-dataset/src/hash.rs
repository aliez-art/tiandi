//! 感知哈希（dHash）：8×8 灰度差分哈希，64 位。
//!
//! 对近重复图（缩放/轻微压缩/裁剪）鲁棒；汉明距离 ≤ 阈值判重。
//! 注意：dHash 编码"右侧比左侧亮"（上升沿），纯下降/均匀图案会产生全零哈希。

/// 计算图像文件的 64 位 dHash（hex 字符串）。
pub fn dhash64(img: &image::DynamicImage) -> u64 {
    use image::GenericImageView;
    // 9×8 灰度（8 个横向差分 × 8 行）
    let gray = img
        .grayscale()
        .resize_exact(9, 8, image::imageops::FilterType::Triangle);
    let mut hash: u64 = 0;
    for y in 0..8 {
        for x in 0..8 {
            let left = gray.get_pixel(x, y)[0];
            let right = gray.get_pixel(x + 1, y)[0];
            hash <<= 1;
            if right > left {
                hash |= 1;
            }
        }
    }
    hash
}

/// 汉明距离。
pub fn hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// 按 16 位前缀分桶的近重复检测：
/// 仅同前缀（高 16 位相同）的图像才可能距离 ≤ 阈值，避免 O(n²)。
pub fn find_duplicates(entries: &[(String, u64)], threshold: u32) -> Vec<Vec<(String, u64)>> {
    use std::collections::HashMap;
    let mut buckets: HashMap<u16, Vec<(String, u64)>> = HashMap::new();
    for (path, h) in entries {
        buckets
            .entry((*h >> 48) as u16)
            .or_default()
            .push((path.clone(), *h));
    }
    let mut groups: Vec<Vec<(String, u64)>> = Vec::new();
    for group in buckets.values() {
        // 桶内 O(k²)（同前缀桶通常很小）
        for i in 0..group.len() {
            let (path_i, h_i) = &group[i];
            let mut matched: Vec<(String, u64)> = vec![(path_i.clone(), *h_i)];
            for (path_j, h_j) in group.iter().skip(i + 1) {
                if hamming(*h_i, *h_j) <= threshold {
                    matched.push((path_j.clone(), *h_j));
                }
            }
            if matched.len() > 1 {
                // 去重：避免同一张在多个组里重复出现（保留已分组的不再参与）
                if groups.iter().any(|g| g.iter().any(|(p, _)| p == path_i)) {
                    continue;
                }
                groups.push(matched);
            }
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GenericImageView, RgbImage};

    /// 粗图案。图案须含"左暗右亮"的上升沿（dHash 只编码上升沿）。
    fn make_image(w: u32, h: u32, pattern: u8) -> image::DynamicImage {
        let mut img = RgbImage::new(w, h);
        for (x, y, p) in img.enumerate_pixels_mut() {
            let v = match pattern {
                // 0: 垂直上升（左暗右亮）→ 行内上升沿
                0 => {
                    if x >= w / 2 {
                        220
                    } else {
                        20
                    }
                }
                // 1: 粗棋盘格（32px）→ 多次上升/下降
                1 => {
                    if (x / 32 + y / 32) % 2 == 0 {
                        220
                    } else {
                        20
                    }
                }
                // 2: 水平分割（上亮下暗）→ 行内均匀 → 全零哈希
                2 => {
                    if y < h / 2 {
                        220
                    } else {
                        20
                    }
                }
                // 3: 对角上升（左上暗、右下亮）
                _ => {
                    if x * h > y * w {
                        220
                    } else {
                        20
                    }
                }
            };
            *p = image::Rgb([v, v / 2, v / 3]);
        }
        image::DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn identical_images_have_same_hash() {
        let a = dhash64(&make_image(512, 512, 0));
        let b = dhash64(&make_image(512, 512, 0));
        assert_eq!(a, b);
        assert_eq!(hamming(a, b), 0);
    }

    #[test]
    fn different_patterns_differ() {
        let a = dhash64(&make_image(512, 512, 0)); // 垂直上升
        let b = dhash64(&make_image(512, 512, 1)); // 棋盘
        let c = dhash64(&make_image(512, 512, 2)); // 水平（全零）
        let d = dhash64(&make_image(512, 512, 3)); // 对角
        assert_ne!(a, 0, "垂直上升图哈希不应为全零");
        assert!(hamming(a, b) > 5, "a-b 应不同（距离 {}）", hamming(a, b));
        assert!(hamming(a, c) > 5, "a-c 应不同（距离 {}）", hamming(a, c));
        assert!(hamming(a, d) > 5, "a-d 应不同（距离 {}）", hamming(a, d));
    }

    #[test]
    fn scaled_copy_is_near_duplicate() {
        let orig = make_image(1024, 768, 0);
        let scaled = orig.resize_exact(512, 384, image::imageops::FilterType::Triangle);
        let a = dhash64(&orig);
        let b = dhash64(&scaled);
        assert!(
            hamming(a, b) <= 5,
            "缩放副本应在阈值内（距离 {})",
            hamming(a, b)
        );
    }

    #[test]
    fn find_duplicates_groups_copies() {
        let entries = vec![
            ("a.png".to_string(), dhash64(&make_image(512, 512, 0))),
            ("a-copy.png".to_string(), dhash64(&make_image(512, 512, 0))),
            ("b.png".to_string(), dhash64(&make_image(512, 512, 2))),
        ];
        let groups = find_duplicates(&entries, 5);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 2);
    }

    #[test]
    fn grayscale_probe_sanity() {
        // 验证缩放后确实存在上升沿（防回归：图案被抹平导致全零哈希）
        let gray = make_image(512, 512, 0).grayscale().resize_exact(
            9,
            8,
            image::imageops::FilterType::Triangle,
        );
        let row: Vec<u8> = (0..9).map(|x| gray.get_pixel(x, 0)[0]).collect();
        assert!(
            row.windows(2).any(|w| w[1] > w[0]),
            "行内应存在上升沿：{row:?}"
        );
    }
}
