//! 分辨率桶：按长宽比分组，保持总面积 ≈ 目标分辨率²（kohya sd-scripts 风格）。

use serde::Serialize;

/// 桶描述。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BucketInfo {
    pub width: u32,
    pub height: u32,
}

impl BucketInfo {
    pub fn label(&self) -> String {
        format!("{}x{}", self.width, self.height)
    }
}

/// 生成候选桶集合：围绕目标分辨率、按 `steps` 步进调整宽高，
/// 面积约束在 [res² × min_ratio, res² × max_ratio]（默认 0.85 / 1.15）。
pub fn candidate_buckets(
    target: u32,
    steps: u32,
    min_ratio: f64,
    max_ratio: f64,
) -> Vec<BucketInfo> {
    use std::collections::BTreeSet;
    let mut set: BTreeSet<(u32, u32)> = BTreeSet::new();
    let area = target as f64 * target as f64;
    let steps = steps.max(8);

    // 从正方形出发，宽逐步增加（高相应减小），直到面积比例越界
    // 注意：不归一化 (w,h) —— 宽图桶保持 width>height，纵向桶保持 width<height
    let mut w = target;
    loop {
        let h = (((area / w as f64) / steps as f64).round() as u32 * steps).max(steps);
        let ratio = (w as f64 * h as f64) / area;
        if !(min_ratio..=max_ratio).contains(&ratio) {
            break;
        }
        set.insert((w, h));
        w += steps;
        if w > target.saturating_mul(4) {
            break; // 保险丝：极端纵横比不再枚举
        }
    }

    // 宽逐步减小（高增大）
    let mut w = target;
    loop {
        let h = (((area / w as f64) / steps as f64).round() as u32 * steps).max(steps);
        let ratio = (w as f64 * h as f64) / area;
        if !(min_ratio..=max_ratio).contains(&ratio) {
            break;
        }
        set.insert((w, h));
        if w <= steps {
            break;
        }
        w = w.saturating_sub(steps);
    }

    set.into_iter()
        .map(|(w, h)| BucketInfo {
            width: w,
            height: h,
        })
        .collect()
}

/// 为每张图分配面积损失最小的桶。
/// 成本 = 面积对数差 + 0.15 × 纵横比对数差（偏向保持原纵横比）。
pub fn assign_buckets(images: &[(u32, u32)], candidates: &[BucketInfo]) -> Vec<Option<BucketInfo>> {
    images
        .iter()
        .map(|(w, h)| {
            let area_log = ((*w as f64) * (*h as f64)).ln();
            let aspect_log = ((*w as f64) / (*h as f64)).ln();
            let mut best: Option<(f64, BucketInfo)> = None;
            for c in candidates {
                let c_area_log = ((c.width as f64) * (c.height as f64)).ln();
                let c_aspect_log = ((c.width as f64) / (c.height as f64)).ln();
                let cost = (c_area_log - area_log).abs() + 0.15 * (c_aspect_log - aspect_log).abs();
                if best.as_ref().is_none_or(|(bc, _)| cost < *bc) {
                    best = Some((cost, *c));
                }
            }
            best.map(|(_, b)| b)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidates_include_square_and_wide() {
        let buckets = candidate_buckets(1024, 64, 0.85, 1.15);
        assert!(buckets.contains(&BucketInfo {
            width: 1024,
            height: 1024
        }));
        // 存在横向桶与纵向桶
        assert!(buckets.iter().any(|b| b.width > b.height), "应包含横向桶");
        assert!(buckets.iter().any(|b| b.height > b.width), "应包含纵向桶");
        // 面积约束
        let area = 1024f64 * 1024f64;
        for b in &buckets {
            let a = (b.width as f64) * (b.height as f64);
            assert!(
                a >= area * 0.85 - 1e-6 && a <= area * 1.15 + 1e-6,
                "桶 {}x{} 面积越界 {a}",
                b.width,
                b.height
            );
        }
    }

    #[test]
    fn square_image_gets_square_bucket() {
        let candidates = candidate_buckets(1024, 64, 0.85, 1.15);
        let assigned = assign_buckets(&[(1024, 1024)], &candidates);
        assert_eq!(assigned[0].map(|b| b.label()), Some("1024x1024".into()));
    }

    #[test]
    fn wide_image_gets_wide_bucket() {
        let candidates = candidate_buckets(1024, 64, 0.85, 1.15);
        let assigned = assign_buckets(&[(1536, 768)], &candidates);
        let b = assigned[0].unwrap();
        assert!(b.width > b.height, "宽图应分到横向桶（{}）", b.label());
    }

    #[test]
    fn tall_image_gets_tall_bucket() {
        let candidates = candidate_buckets(1024, 64, 0.85, 1.15);
        let assigned = assign_buckets(&[(768, 1536)], &candidates);
        let b = assigned[0].unwrap();
        assert!(b.height > b.width, "高图应分到纵向桶（{}）", b.label());
    }
}
