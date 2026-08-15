//! `tiandi models`：基底模型注册（add/list）。

use std::path::Path;

use tiandi_core::ModelFamily;
use tiandi_state::Store;

pub fn cmd_add(workspace: &Path, name: &str, family: &str, path: &str) {
    let family = match family {
        "sdxl1" => ModelFamily::Sdxl1,
        "dit_anima" => ModelFamily::DitAnima,
        "dit_krea2" => ModelFamily::DitKrea2,
        other => {
            eprintln!("✗ 未知模型族：{other}（可选 sdxl1 / dit_anima / dit_krea2）");
            std::process::exit(1);
        }
    };
    let p = Path::new(path);
    if !p.exists() {
        eprintln!("✗ 路径不存在：{path}");
        std::process::exit(1);
    }
    let db = workspace.join("tiandi.db");
    let store = match Store::open(&db) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("✗ 打开数据库失败：{e}（先运行 tiandi init）");
            std::process::exit(1);
        }
    };
    let model = tiandi_core::BaseModel::new(
        name,
        family,
        Some(path.to_string()),
        None,
        Some("cli".into()),
    );
    if let Err(e) = store.insert_base_model(&model) {
        eprintln!("✗ 注册失败：{e}");
        std::process::exit(1);
    }
    println!("✓ 已注册基底模型：{name} [{}] → {path}", family.label());
}

pub fn cmd_list(workspace: &Path) {
    let db = workspace.join("tiandi.db");
    let store = match Store::open(&db) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("✗ 打开数据库失败：{e}");
            std::process::exit(1);
        }
    };
    match store.list_base_models() {
        Ok(list) if list.is_empty() => {
            println!("（暂无注册模型）用 `tiandi models add --name NoobAI-XL --family sdxl1 --path <文件>` 注册");
        }
        Ok(list) => {
            for m in list {
                println!(
                    "  {} [{}]  {}",
                    m.name,
                    m.family.label(),
                    m.path.unwrap_or_default()
                );
            }
        }
        Err(e) => {
            eprintln!("✗ 查询失败：{e}");
            std::process::exit(1);
        }
    }
}
