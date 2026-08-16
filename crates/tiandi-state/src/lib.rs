//! SQLite 持久化：迁移、仓储、运行清单（run manifest）。
//!
//! 领域数据入 SQLite；大文件（模型/latent 缓存/采样图）存磁盘目录，库内只存路径与哈希
//! （docs/architecture.md §3）。任务运行清单 `manifest.json` 落盘于任务目录，供崩溃恢复。

pub mod manifest;
pub mod repos;

pub use manifest::{ManifestError, RunManifest};
pub use repos::{ImageRecord, RepoError, Store};

use rusqlite::Connection;

/// 当前 schema 版本（`PRAGMA user_version`；仅测试断言使用——
/// 迁移块各自写字面量版本号，升版须新增对应块，见 [`migrate`]）。
#[cfg(test)]
const SCHEMA_VERSION: i64 = 4;

/// v1（M0）：基础表。
const V1_SQL: &str = r#"
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
"#;

/// v2（M1）：数据集图像索引。
const V2_SQL: &str = r#"
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
"#;

/// v3（M2）：任务关联基底模型。`ALTER TABLE … ADD COLUMN` 不可重入，
/// 执行前必须经 [`column_exists`] 确认列不存在。
const V3_SQL: &str = "ALTER TABLE runs ADD COLUMN base_model_id TEXT;";

/// v4（M2）：设置（镜像源等）。
const V4_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"#;

/// 幂等补索引（不推进版本；对既有 v4 库同样生效）。
const IDX_SQL: &str = r#"
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON checkpoints(run_id);
"#;

/// 打开（或创建）数据库并执行迁移。
pub fn open(path: &std::path::Path) -> Result<Connection, rusqlite::Error> {
    let conn = Connection::open(path)?;
    conn.busy_timeout(std::time::Duration::from_secs(5))?;
    // WAL：读写并发不互斥、崩溃更安全；须在迁移前设置（内存库无需）。
    conn.pragma_update(None, "journal_mode", "WAL")?;
    migrate(&conn)?;
    Ok(conn)
}

/// 打开内存数据库（测试用；内存库无需 WAL，busy_timeout 同样有益）。
pub fn open_in_memory() -> Result<Connection, rusqlite::Error> {
    let conn = Connection::open_in_memory()?;
    conn.busy_timeout(std::time::Duration::from_secs(5))?;
    migrate(&conn)?;
    Ok(conn)
}

/// 检查表中是否存在某列（`PRAGMA table_info` 遍历）。
fn column_exists(conn: &Connection, table: &str, column: &str) -> rusqlite::Result<bool> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        // PRAGMA table_info 列：cid(0), name(1), type(2), notnull(3), dflt_value(4), pk(5)
        let name: String = row.get(1)?;
        if name == column {
            return Ok(true);
        }
    }
    Ok(false)
}

/// 迁移：按 `PRAGMA user_version` 增量执行。
///
/// 每个版本块在单条 `execute_batch` 内以
/// `BEGIN IMMEDIATE; … DDL …; PRAGMA user_version = N; COMMIT;` 完成，
/// DDL 与版本号同事务提交，杜绝“DDL 已执行但 user_version 未更新”的半迁移崩溃窗口。
pub fn migrate(conn: &Connection) -> Result<(), rusqlite::Error> {
    let current: i64 = conn.pragma_query_value(None, "user_version", |row| row.get(0))?;
    if current < 1 {
        conn.execute_batch(&format!(
            "BEGIN IMMEDIATE; {V1_SQL} PRAGMA user_version = 1; COMMIT;"
        ))?;
    }
    if current < 2 {
        conn.execute_batch(&format!(
            "BEGIN IMMEDIATE; {V2_SQL} PRAGMA user_version = 2; COMMIT;"
        ))?;
    }
    if current < 3 {
        // ALTER 不可重入：仅当列不存在时才执行（覆盖 v3 崩溃窗口：ALTER 已生效但版本未提交）。
        let mut sql = String::from("BEGIN IMMEDIATE; ");
        if !column_exists(conn, "runs", "base_model_id")? {
            sql.push_str(V3_SQL);
            sql.push(' ');
        }
        sql.push_str("PRAGMA user_version = 3; COMMIT;");
        conn.execute_batch(&sql)?;
    }
    if current < 4 {
        conn.execute_batch(&format!(
            "BEGIN IMMEDIATE; {V4_SQL} PRAGMA user_version = 4; COMMIT;"
        ))?;
    }
    // 幂等步骤：补齐索引（旧 v4 库也生效，不推进版本）。
    conn.execute_batch(IDX_SQL)?;
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
    fn migrate_recovers_from_v3_crash_window() {
        let conn = Connection::open_in_memory().unwrap();
        // 模拟 v3 崩溃窗口：v1/v2 已提交（user_version=2），
        // v3 的 ALTER 已执行但 user_version 未提交（旧代码在两者之间崩溃的现场）。
        conn.execute_batch(&format!(
            "BEGIN; {V1_SQL} PRAGMA user_version = 1; COMMIT;
             BEGIN; {V2_SQL} PRAGMA user_version = 2; COMMIT;
             {V3_SQL}"
        ))
        .unwrap();
        // 前置条件成立：列已存在、版本停在 2
        assert!(column_exists(&conn, "runs", "base_model_id").unwrap());
        let v: i64 = conn
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .unwrap();
        assert_eq!(v, 2);

        // 重启后 migrate 必须成功（不再报 duplicate column name）且版本前进
        migrate(&conn).unwrap();
        let v: i64 = conn
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .unwrap();
        assert_eq!(v, SCHEMA_VERSION);
        // 后续迁移幂等
        migrate(&conn).unwrap();
    }

    #[test]
    fn column_exists_detects_columns() {
        let conn = open_in_memory().unwrap();
        assert!(column_exists(&conn, "runs", "base_model_id").unwrap());
        assert!(!column_exists(&conn, "runs", "no_such_column").unwrap());
        // 不存在的表视为不含任何列
        assert!(!column_exists(&conn, "no_such_table", "id").unwrap());
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
