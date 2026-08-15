//! SQLite 持久化：迁移、仓储、运行清单（run manifest）。
//!
//! 领域数据入 SQLite；大文件（模型/latent 缓存/采样图）存磁盘目录，库内只存路径与哈希
//! （docs/architecture.md §3）。任务运行清单 `manifest.json` 落盘于任务目录，供崩溃恢复。

pub mod manifest;
pub mod repos;

pub use manifest::{ManifestError, RunManifest};
pub use repos::{ImageRecord, RepoError, Store};

use rusqlite::Connection;

/// 当前 schema 版本（`PRAGMA user_version`）。
const SCHEMA_VERSION: i64 = 3;

/// 打开（或创建）数据库并执行迁移。
pub fn open(path: &std::path::Path) -> Result<Connection, rusqlite::Error> {
    let conn = Connection::open(path)?;
    migrate(&conn)?;
    Ok(conn)
}

/// 打开内存数据库（测试用）。
pub fn open_in_memory() -> Result<Connection, rusqlite::Error> {
    let conn = Connection::open_in_memory()?;
    migrate(&conn)?;
    Ok(conn)
}

/// 迁移：按 `PRAGMA user_version` 增量执行。
pub fn migrate(conn: &Connection) -> Result<(), rusqlite::Error> {
    let current: i64 = conn.pragma_query_value(None, "user_version", |row| row.get(0))?;
    if current < 1 {
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS projects (
                id         TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                root_dir   TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS base_models (
                id         TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                family     TEXT NOT NULL,
                path       TEXT,
                sha256     TEXT,
                source     TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS datasets (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                dir         TEXT NOT NULL,
                image_count INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS recipes (
                id         TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                family     TEXT NOT NULL,
                data       TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                id            TEXT PRIMARY KEY,
                project_id    TEXT,
                dataset_id    TEXT,
                recipe_id     TEXT,
                state         TEXT NOT NULL,
                manifest_path TEXT,
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_runs_state ON runs(state);
            CREATE TABLE IF NOT EXISTS metrics (
                run_id TEXT NOT NULL,
                step   INTEGER NOT NULL,
                loss   REAL,
                lr     REAL,
                PRIMARY KEY (run_id, step)
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                id         TEXT PRIMARY KEY,
                run_id     TEXT NOT NULL,
                kind       TEXT NOT NULL,
                path       TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            "#,
        )?;
        conn.pragma_update(None, "user_version", 1)?;
    }
    if current < 2 {
        // v2（M1）：数据集图像索引
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS image_files (
                id           TEXT PRIMARY KEY,
                dataset_id   TEXT NOT NULL,
                path         TEXT NOT NULL,
                width        INTEGER,
                height       INTEGER,
                dhash        TEXT,
                bucket       TEXT,
                thumb        TEXT,
                exif         TEXT,
                duplicate_of TEXT,
                created_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_image_files_dataset ON image_files(dataset_id);
            "#,
        )?;
        conn.pragma_update(None, "user_version", 2)?;
    }
    if current < 3 {
        // v3（M2）：任务关联基底模型
        conn.execute_batch(
            r#"
            ALTER TABLE runs ADD COLUMN base_model_id TEXT;
            "#,
        )?;
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
    }
    Ok(())
}

/// 建工作区目录骨架（models/datasets/recipes/runs/vault）。
pub fn ensure_workspace_layout(root: &std::path::Path) -> std::io::Result<()> {
    for dir in ["models", "datasets", "recipes", "runs", "vault"] {
        std::fs::create_dir_all(root.join(dir))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrate_is_idempotent() {
        let conn = open_in_memory().unwrap();
        // 再跑一次迁移不报错、版本不前进
        migrate(&conn).unwrap();
        let v: i64 = conn
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .unwrap();
        assert_eq!(v, SCHEMA_VERSION);
    }

    #[test]
    fn schema_tables_exist() {
        let conn = open_in_memory().unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(count >= 6, "expected at least 6 tables, got {count}");
    }

    #[test]
    fn workspace_layout_creates_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        ensure_workspace_layout(tmp.path()).unwrap();
        for dir in ["models", "datasets", "recipes", "runs", "vault"] {
            assert!(tmp.path().join(dir).is_dir(), "missing {dir}");
        }
    }
}
