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

/// 近重复检测：多索引候选 + 并查集聚类。
///
/// - **全零哈希**（纯色/横幅图）不参与判重，直接视为唯一；
/// - **候选生成**：把 64 位哈希切成 4 个 16 位块，建立 4 张 `HashMap<u16, Vec<usize>>`
///   （块值 → 条目位置）。对每个条目，候选 = 每块"块值 + 16 个单比特翻转变体"
///   对应的全部条目（共 4×17 次查表），去重后仅对候选计算汉明距离。
///   鸽笼原理保证零漏检：两哈希距离 ≤ 5（乃至 ≤ 7）时，4 块中必有一块差异 ≤ 1 比特，
///   该对必然落入某次查表结果；而差异 ≥ 8 的配对本身已远超默认阈值。
/// - **聚类**：并查集（union-find）把"距离 ≤ 阈值"的两两关系合并为传递闭包
///   （链式 A~B~C 同组），输出每个非平凡集合；组内成员按路径排序、
///   组按首元素路径排序，保证输出确定性。
pub fn find_duplicates(entries: &[(String, u64)], threshold: u32) -> Vec<Vec<(String, u64)>> {
    use std::collections::HashMap;

    // 全零哈希条目不参与判重（纯色/横幅图直接视为唯一）
    let indexed: Vec<&(String, u64)> = entries.iter().filter(|(_, h)| *h != 0).collect();
    if indexed.is_empty() {
        return Vec::new();
    }

    // 4 个 16 位块索引：块值 → 条目位置（indexed 内下标）
    let mut block_maps: [HashMap<u16, Vec<usize>>; 4] = std::array::from_fn(|_| HashMap::new());
    for (pos, (_, h)) in indexed.iter().enumerate() {
        for (b, map) in block_maps.iter_mut().enumerate() {
            map.entry(block_value(*h, b)).or_default().push(pos);
        }
    }

    let mut uf = UnionFind::new(indexed.len());
    for (pos, (_, h_i)) in indexed.iter().enumerate() {
        let mut candidates: Vec<usize> = Vec::new();
        for (b, map) in block_maps.iter().enumerate() {
            let v = block_value(*h_i, b);
            for variant in single_bit_variants(v) {
                if let Some(list) = map.get(&variant) {
                    candidates.extend_from_slice(list);
                }
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        for &j in candidates.iter().filter(|&&j| j > pos) {
            if hamming(*h_i, indexed[j].1) <= threshold {
                uf.union(pos, j);
            }
        }
    }

    // 输出每个非平凡集合（成员按路径排序；集合按首元素路径排序，保证确定性）
    let mut by_root: HashMap<usize, Vec<(String, u64)>> = HashMap::new();
    for (pos, (path, h)) in indexed.iter().enumerate() {
        by_root
            .entry(uf.find(pos))
            .or_default()
            .push((path.clone(), *h));
    }
    let mut groups: Vec<Vec<(String, u64)>> =
        by_root.into_values().filter(|g| g.len() > 1).collect();
    for g in &mut groups {
        g.sort_by(|a, b| a.0.cmp(&b.0));
    }
    groups.sort_by(|a, b| a[0].0.cmp(&b[0].0));
    groups
}

/// 取出 64 位哈希的第 `b` 个 16 位块（b ∈ 0..4，从低位起）。
fn block_value(h: u64, b: usize) -> u16 {
    ((h >> (16 * b)) & 0xFFFF) as u16
}

/// 块值自身 + 16 个单比特翻转变体（共 17 个）。
fn single_bit_variants(v: u16) -> [u16; 17] {
    let mut out = [0u16; 17];
    out[0] = v;
    for k in 0..16 {
        out[k + 1] = v ^ (1 << k);
    }
    out
}

/// 并查集（路径压缩 + 按秩合并）。
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
    }
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
                3 => {
                    if x * h > y * w {
                        220
                    } else {
                        20
                    }
                }
                // 4/5: 纯色（无上升沿 → 全零哈希）
                4 => 60,
                5 => 200,
                _ => 0,
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

    #[test]
    fn all_zero_hashes_are_not_grouped() {
        // 两张不同纯色图（dHash 全零）+ 一张正常图：全零条目直接视为唯一，不判重
        let solid_a = dhash64(&make_image(64, 64, 4));
        let solid_b = dhash64(&make_image(64, 64, 5));
        assert_eq!(solid_a, 0, "纯色图 dHash 应为全零");
        assert_eq!(solid_b, 0, "纯色图 dHash 应为全零");
        let normal = dhash64(&make_image(64, 64, 0));
        assert_ne!(normal, 0);
        let entries = vec![
            ("solid-a.png".to_string(), solid_a),
            ("solid-b.png".to_string(), solid_b),
            ("normal.png".to_string(), normal),
        ];
        let groups = find_duplicates(&entries, 5);
        assert!(groups.is_empty(), "全零哈希不应参与判重：{groups:?}");
    }

    #[test]
    fn cross_prefix_boundary_near_duplicate_detected() {
        // 高 16 位差 1 比特、低位完全相同：旧"同前缀"分桶会漏检，多索引候选必须检出
        let h1 = 0x8000_0000_0000_0001u64;
        let h2 = 0x0000_0000_0000_0001u64;
        assert_eq!(hamming(h1, h2), 1, "两哈希应仅差 1 比特");
        let entries = vec![("a.png".to_string(), h1), ("b.png".to_string(), h2)];
        let groups = find_duplicates(&entries, 5);
        assert_eq!(groups.len(), 1, "跨前缀近重复应检出：{groups:?}");
        assert_eq!(groups[0].len(), 2);
    }

    #[test]
    fn chained_near_duplicates_form_transitive_group() {
        // 链式 A~B~C：A 与 C 距离 > 阈值，但经 B 传递后应同组（并查集传递闭包）
        let a = 0x0000_0000_0000_0001u64;
        let b = a ^ (1 << 5);
        let c = b ^ (1 << 6) ^ (1 << 7) ^ (1 << 8) ^ (1 << 9) ^ (1 << 10);
        assert!(hamming(a, b) <= 5, "A-B 距离 {}", hamming(a, b));
        assert!(hamming(b, c) <= 5, "B-C 距离 {}", hamming(b, c));
        assert!(hamming(a, c) > 5, "A-C 距离应超阈值：{}", hamming(a, c));
        let entries = vec![
            ("a.png".to_string(), a),
            ("b.png".to_string(), b),
            ("c.png".to_string(), c),
        ];
        let groups = find_duplicates(&entries, 5);
        assert_eq!(groups.len(), 1, "链式 A~B~C 应合并为一组：{groups:?}");
        assert_eq!(groups[0].len(), 3);
    }
}
