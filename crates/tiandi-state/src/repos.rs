//! 仓储层：对 SQLite 的薄封装（单人本地工具，单连接 + 互斥即可）。
//!
//! 领域实体 ↔ 行映射集中在各 repo 方法内；`Store` 持有连接并提供事务边界。

use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use serde::{Deserialize, Serialize};
use tiandi_core::{
    BaseModel, Checkpoint, Dataset, MetricPoint, ModelFamily, Project, Recipe, Run, RunState,
};

/// 数据集图像记录（扫描产物入库）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRecord {
    pub id: String,
    pub dataset_id: String,
    pub path: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub dhash: Option<String>,
    pub bucket: Option<String>,
    pub thumb: Option<String>,
    pub exif: Option<String>,
    /// 重复组主图 id（本图是重复项）
    pub duplicate_of: Option<String>,
    pub created_at: String,
}

/// 仓储错误。
#[derive(Debug, thiserror::Error)]
pub enum RepoError {
    #[error("数据库错误: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("未找到: {entity} {id}")]
    NotFound { entity: &'static str, id: String },
    #[error("非法家族值: {0}")]
    BadFamily(String),
}

/// 解析 RunState 存储值：兼容新格式（纯 snake_case）与旧格式（JSON 带引号）。
fn parse_run_state(s: &str) -> RunState {
    if s.starts_with('"') {
        serde_json::from_str(s).unwrap_or(RunState::Created)
    } else {
        serde_json::from_str(&format!("\"{s}\"")).unwrap_or(RunState::Created)
    }
}

/// 运行中状态集合（含旧格式带引号变体），供 `has_running_run` / `claim_next_queued` 复用。
const RUNNING_STATES_SQL: &str = r#"'preparing','running','sampling','saving','paused','"preparing"','"running"','"sampling"','"saving"','"paused"'"#;

/// 排队状态集合（含旧格式带引号变体）。
const QUEUED_STATES_SQL: &str = r#"'queued','"queued"'"#;

/// 数据存储（SQLite 连接 + 各仓储方法）。
pub struct Store {
    conn: Connection,
}

impl Store {
    pub fn new(conn: Connection) -> Self {
        Self { conn }
    }

    /// 打开/创建文件数据库并迁移。
    pub fn open(path: &std::path::Path) -> Result<Self, RepoError> {
        Ok(Self::new(crate::open(path)?))
    }

    /// 内存数据库（测试用）。
    pub fn open_in_memory() -> Result<Self, RepoError> {
        Ok(Self::new(crate::open_in_memory()?))
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    fn family_from_row(s: &str) -> Result<ModelFamily, RepoError> {
        match s {
            "sdxl1" => Ok(ModelFamily::Sdxl1),
            "dit_anima" => Ok(ModelFamily::DitAnima),
            "dit_krea2" => Ok(ModelFamily::DitKrea2),
            other => Err(RepoError::BadFamily(other.into())),
        }
    }

    fn family_to_str(f: ModelFamily) -> &'static str {
        match f {
            ModelFamily::Sdxl1 => "sdxl1",
            ModelFamily::DitAnima => "dit_anima",
            ModelFamily::DitKrea2 => "dit_krea2",
        }
    }

    // ---- Project ----

    pub fn insert_project(&self, p: &Project) -> Result<(), RepoError> {
        self.conn.execute(
            "INSERT INTO projects (id, name, root_dir, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![p.id, p.name, p.root_dir, p.created_at],
        )?;
        Ok(())
    }

    pub fn get_project(&self, id: &str) -> Result<Project, RepoError> {
        self.conn
            .query_row(
                "SELECT id, name, root_dir, created_at FROM projects WHERE id = ?1",
                [id],
                |row| {
                    Ok(Project {
                        id: row.get(0)?,
                        name: row.get(1)?,
                        root_dir: row.get(2)?,
                        created_at: row.get(3)?,
                    })
                },
            )
            .optional()?
            .ok_or_else(|| RepoError::NotFound {
                entity: "project",
                id: id.into(),
            })
    }

    pub fn list_projects(&self) -> Result<Vec<Project>, RepoError> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, name, root_dir, created_at FROM projects ORDER BY created_at")?;
        let rows = stmt.query_map([], |row| {
            Ok(Project {
                id: row.get(0)?,
                name: row.get(1)?,
                root_dir: row.get(2)?,
                created_at: row.get(3)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    // ---- BaseModel ----

    pub fn insert_base_model(&self, m: &BaseModel) -> Result<(), RepoError> {
        self.conn.execute(
            "INSERT INTO base_models (id, name, family, path, sha256, source, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                m.id,
                m.name,
                Self::family_to_str(m.family),
                m.path,
                m.sha256,
                m.source,
                m.created_at
            ],
        )?;
        Ok(())
    }

    pub fn get_base_model(&self, id: &str) -> Result<BaseModel, RepoError> {
        let row = self
            .conn
            .query_row(
                "SELECT id, name, family, path, sha256, source, created_at FROM base_models WHERE id = ?1",
                [id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, Option<String>>(4)?,
                        row.get::<_, Option<String>>(5)?,
                        row.get::<_, String>(6)?,
                    ))
                },
            )
            .optional()?
            .ok_or_else(|| RepoError::NotFound { entity: "base_model", id: id.into() })?;
        let (id, name, family, path, sha256, source, created_at) = row;
        Ok(BaseModel {
            id,
            name,
            family: Self::family_from_row(&family)?,
            path,
            sha256,
            source,
            created_at,
        })
    }

    pub fn list_base_models(&self) -> Result<Vec<BaseModel>, RepoError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, family, path, sha256, source, created_at FROM base_models ORDER BY created_at",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, String>(6)?,
            ))
        })?;
        let mut out = Vec::new();
        for r in rows {
            let (id, name, family, path, sha256, source, created_at) = r?;
            out.push(BaseModel {
                id,
                name,
                family: Self::family_from_row(&family)?,
                path,
                sha256,
                source,
                created_at,
            });
        }
        Ok(out)
    }

    // ---- Dataset ----

    pub fn insert_dataset(&self, d: &Dataset) -> Result<(), RepoError> {
        self.conn.execute(
            "INSERT INTO datasets (id, name, dir, image_count, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![d.id, d.name, d.dir, d.image_count, d.created_at],
        )?;
        Ok(())
    }

    pub fn list_datasets(&self) -> Result<Vec<Dataset>, RepoError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, dir, image_count, created_at FROM datasets ORDER BY created_at",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(Dataset {
                id: row.get(0)?,
                name: row.get(1)?,
                dir: row.get(2)?,
                image_count: row.get(3)?,
                created_at: row.get(4)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn get_dataset(&self, id: &str) -> Result<Dataset, RepoError> {
        self.conn
            .query_row(
                "SELECT id, name, dir, image_count, created_at FROM datasets WHERE id = ?1",
                [id],
                |row| {
                    Ok(Dataset {
                        id: row.get(0)?,
                        name: row.get(1)?,
                        dir: row.get(2)?,
                        image_count: row.get(3)?,
                        created_at: row.get(4)?,
                    })
                },
            )
            .optional()?
            .ok_or_else(|| RepoError::NotFound {
                entity: "dataset",
                id: id.into(),
            })
    }

    /// 删除数据集记录及图像索引（不存在则 NotFound；事务内执行，出错整体回滚）。
    pub fn delete_dataset(&self, id: &str) -> Result<(), RepoError> {
        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        tx.execute("DELETE FROM image_files WHERE dataset_id = ?1", [id])?;
        let n = tx.execute("DELETE FROM datasets WHERE id = ?1", [id])?;
        if n == 0 {
            return Err(RepoError::NotFound {
                entity: "dataset",
                id: id.into(),
            });
        }
        tx.commit()?;
        Ok(())
    }

    // ---- Recipe ----

    pub fn insert_recipe(&self, r: &Recipe) -> Result<(), RepoError> {
        let data = serde_json::to_string(&r.data)
            .map_err(|e| RepoError::Sql(rusqlite::Error::ToSqlConversionFailure(Box::new(e))))?;
        self.conn.execute(
            "INSERT INTO recipes (id, name, family, data, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                r.id,
                r.name,
                Self::family_to_str(r.family),
                data,
                r.created_at
            ],
        )?;
        Ok(())
    }

    pub fn get_recipe(&self, id: &str) -> Result<Recipe, RepoError> {
        let row = self
            .conn
            .query_row(
                "SELECT id, name, family, data, created_at FROM recipes WHERE id = ?1",
                [id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                    ))
                },
            )
            .optional()?
            .ok_or_else(|| RepoError::NotFound {
                entity: "recipe",
                id: id.into(),
            })?;
        let (id, name, family, data, created_at) = row;
        Ok(Recipe {
            id,
            name,
            family: Self::family_from_row(&family)?,
            data: serde_json::from_str(&data).map_err(|e| {
                RepoError::Sql(rusqlite::Error::FromSqlConversionFailure(
                    3,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                ))
            })?,
            created_at,
        })
    }

    pub fn list_recipes(&self) -> Result<Vec<Recipe>, RepoError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, family, data, created_at FROM recipes ORDER BY created_at",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
            ))
        })?;
        let mut out = Vec::new();
        for r in rows {
            let (id, name, family, data, created_at) = r?;
            out.push(Recipe {
                id,
                name,
                family: Self::family_from_row(&family)?,
                data: serde_json::from_str(&data).map_err(|e| {
                    RepoError::Sql(rusqlite::Error::FromSqlConversionFailure(
                        3,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    ))
                })?,
                created_at,
            });
        }
        Ok(out)
    }

    /// 删除丹方（不存在则 NotFound；事务内执行，出错整体回滚）。
    pub fn delete_recipe(&self, id: &str) -> Result<(), RepoError> {
        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        let n = tx.execute("DELETE FROM recipes WHERE id = ?1", [id])?;
        if n == 0 {
            return Err(RepoError::NotFound {
                entity: "recipe",
                id: id.into(),
            });
        }
        tx.commit()?;
        Ok(())
    }

    /// 删除任务记录及关联指标/产物索引（不存在则 NotFound；事务内执行，出错整体回滚）。
    pub fn delete_run(&self, id: &str) -> Result<(), RepoError> {
        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        tx.execute("DELETE FROM metrics WHERE run_id = ?1", [id])?;
        tx.execute("DELETE FROM checkpoints WHERE run_id = ?1", [id])?;
        let n = tx.execute("DELETE FROM runs WHERE id = ?1", [id])?;
        if n == 0 {
            return Err(RepoError::NotFound {
                entity: "run",
                id: id.into(),
            });
        }
        tx.commit()?;
        Ok(())
    }

    // ---- Run ----

    pub fn insert_run(&self, r: &Run) -> Result<(), RepoError> {
        self.conn.execute(
            "INSERT INTO runs (id, project_id, dataset_id, recipe_id, base_model_id, state, manifest_path, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                r.id,
                r.project_id,
                r.dataset_id,
                r.recipe_id,
                r.base_model_id,
                r.state.as_str(),
                r.manifest_path,
                r.created_at,
                r.updated_at
            ],
        )?;
        Ok(())
    }

    pub fn get_run(&self, id: &str) -> Result<Run, RepoError> {
        self.conn
            .query_row(
                "SELECT id, project_id, dataset_id, recipe_id, base_model_id, state, manifest_path, created_at, updated_at
                 FROM runs WHERE id = ?1",
                [id],
                |row| {
                    let state: String = row.get(5)?;
                    Ok(Run {
                        id: row.get(0)?,
                        project_id: row.get(1)?,
                        dataset_id: row.get(2)?,
                        recipe_id: row.get(3)?,
                        base_model_id: row.get(4)?,
                        state: parse_run_state(&state),
                        manifest_path: row.get(6)?,
                        created_at: row.get(7)?,
                        updated_at: row.get(8)?,
                    })
                },
            )
            .optional()?
            .ok_or_else(|| RepoError::NotFound { entity: "run", id: id.into() })
    }

    pub fn list_runs(&self) -> Result<Vec<Run>, RepoError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, project_id, dataset_id, recipe_id, base_model_id, state, manifest_path, created_at, updated_at
             FROM runs ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            let state: String = row.get(5)?;
            Ok(Run {
                id: row.get(0)?,
                project_id: row.get(1)?,
                dataset_id: row.get(2)?,
                recipe_id: row.get(3)?,
                base_model_id: row.get(4)?,
                state: parse_run_state(&state),
                manifest_path: row.get(6)?,
                created_at: row.get(7)?,
                updated_at: row.get(8)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// 排队中的任务（按创建时间升序；兼容旧格式带引号状态值）。
    pub fn list_queued_runs(&self) -> Result<Vec<Run>, RepoError> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT id, project_id, dataset_id, recipe_id, base_model_id, state, manifest_path, created_at, updated_at
             FROM runs WHERE state IN ({QUEUED_STATES_SQL}) ORDER BY created_at ASC"
        ))?;
        let rows = stmt.query_map([], |row| {
            let state: String = row.get(5)?;
            Ok(Run {
                id: row.get(0)?,
                project_id: row.get(1)?,
                dataset_id: row.get(2)?,
                recipe_id: row.get(3)?,
                base_model_id: row.get(4)?,
                state: parse_run_state(&state),
                manifest_path: row.get(6)?,
                created_at: row.get(7)?,
                updated_at: row.get(8)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// 是否有任务正在运行（Preparing/Running/Sampling/Saving/Paused；兼容旧格式带引号状态值）。
    pub fn has_running_run(&self) -> Result<bool, RepoError> {
        let n: i64 = self.conn.query_row(
            &format!("SELECT count(*) FROM runs WHERE state IN ({RUNNING_STATES_SQL})"),
            [],
            |row| row.get(0),
        )?;
        Ok(n > 0)
    }

    /// 原子认领最早的 Queued 任务（BEGIN IMMEDIATE 事务内：无运行中任务 → 选中 → 置 Preparing）。
    /// 返回被认领的 Run；无排队任务或有运行中任务时返回 Ok(None)。
    pub fn claim_next_queued(&mut self) -> Result<Option<Run>, RepoError> {
        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        // 同一事务内先判定是否有运行中任务，避免并发认领竞态
        let running: i64 = tx.query_row(
            &format!("SELECT count(*) FROM runs WHERE state IN ({RUNNING_STATES_SQL})"),
            [],
            |row| row.get(0),
        )?;
        if running > 0 {
            tx.commit()?;
            return Ok(None);
        }
        let row = tx
            .query_row(
                &format!(
                    "SELECT id, project_id, dataset_id, recipe_id, base_model_id, state, manifest_path, created_at, updated_at
                     FROM runs WHERE state IN ({QUEUED_STATES_SQL}) ORDER BY created_at ASC LIMIT 1"
                ),
                [],
                |row| {
                    let state: String = row.get(5)?;
                    Ok(Run {
                        id: row.get(0)?,
                        project_id: row.get(1)?,
                        dataset_id: row.get(2)?,
                        recipe_id: row.get(3)?,
                        base_model_id: row.get(4)?,
                        state: parse_run_state(&state),
                        manifest_path: row.get(6)?,
                        created_at: row.get(7)?,
                        updated_at: row.get(8)?,
                    })
                },
            )
            .optional()?;
        let Some(run) = row else {
            tx.commit()?;
            return Ok(None);
        };
        // 事务内置 Preparing 并写回 updated_at（沿用认领前原值，保持时间戳格式一致），随后整体提交
        let n = tx.execute(
            "UPDATE runs SET state = 'preparing', updated_at = ?1 WHERE id = ?2",
            params![run.updated_at, run.id],
        )?;
        debug_assert_eq!(n, 1, "被认领的行应恰好更新一行");
        tx.commit()?;
        Ok(Some(Run {
            state: RunState::Preparing,
            ..run
        }))
    }

    /// 更新任务状态与 updated_at（事务内）。
    pub fn update_run_state(
        &self,
        id: &str,
        state: RunState,
        updated_at: &str,
    ) -> Result<(), RepoError> {
        let n = self.conn.execute(
            "UPDATE runs SET state = ?1, updated_at = ?2 WHERE id = ?3",
            params![state.as_str(), updated_at, id],
        )?;
        if n == 0 {
            return Err(RepoError::NotFound {
                entity: "run",
                id: id.into(),
            });
        }
        Ok(())
    }

    // ---- Metrics / Checkpoints ----

    pub fn insert_metric(&self, m: &MetricPoint) -> Result<(), RepoError> {
        self.conn.execute(
            "INSERT OR REPLACE INTO metrics (run_id, step, loss, lr) VALUES (?1, ?2, ?3, ?4)",
            params![m.run_id, m.step as i64, m.loss, m.lr],
        )?;
        Ok(())
    }

    pub fn list_metrics(&self, run_id: &str) -> Result<Vec<MetricPoint>, RepoError> {
        let mut stmt = self.conn.prepare(
            "SELECT run_id, step, loss, lr FROM metrics WHERE run_id = ?1 ORDER BY step",
        )?;
        let rows = stmt.query_map([run_id], |row| {
            Ok(MetricPoint {
                run_id: row.get(0)?,
                step: row.get::<_, i64>(1)? as u64,
                loss: row.get(2)?,
                lr: row.get(3)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn insert_checkpoint(&self, c: &Checkpoint) -> Result<(), RepoError> {
        self.conn.execute(
            "INSERT INTO checkpoints (id, run_id, kind, path, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![c.id, c.run_id, c.kind, c.path, c.created_at],
        )?;
        Ok(())
    }

    pub fn get_checkpoint(&self, id: &str) -> Result<Checkpoint, RepoError> {
        self.conn
            .query_row(
                "SELECT id, run_id, kind, path, created_at FROM checkpoints WHERE id = ?1",
                [id],
                |row| {
                    Ok(Checkpoint {
                        id: row.get(0)?,
                        run_id: row.get(1)?,
                        kind: row.get(2)?,
                        path: row.get(3)?,
                        created_at: row.get(4)?,
                    })
                },
            )
            .optional()?
            .ok_or_else(|| RepoError::NotFound {
                entity: "checkpoint",
                id: id.into(),
            })
    }

    pub fn delete_checkpoint(&self, id: &str) -> Result<(), RepoError> {
        let n = self
            .conn
            .execute("DELETE FROM checkpoints WHERE id = ?1", [id])?;
        if n == 0 {
            return Err(RepoError::NotFound {
                entity: "checkpoint",
                id: id.into(),
            });
        }
        Ok(())
    }

    pub fn update_checkpoint_path(&self, id: &str, path: &str) -> Result<(), RepoError> {
        let n = self.conn.execute(
            "UPDATE checkpoints SET path = ?1 WHERE id = ?2",
            params![path, id],
        )?;
        if n == 0 {
            return Err(RepoError::NotFound {
                entity: "checkpoint",
                id: id.into(),
            });
        }
        Ok(())
    }

    pub fn list_all_checkpoints(&self) -> Result<Vec<Checkpoint>, RepoError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, run_id, kind, path, created_at FROM checkpoints ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(Checkpoint {
                id: row.get(0)?,
                run_id: row.get(1)?,
                kind: row.get(2)?,
                path: row.get(3)?,
                created_at: row.get(4)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn list_checkpoints(&self, run_id: &str) -> Result<Vec<Checkpoint>, RepoError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, run_id, kind, path, created_at FROM checkpoints WHERE run_id = ?1 ORDER BY created_at",
        )?;
        let rows = stmt.query_map([run_id], |row| {
            Ok(Checkpoint {
                id: row.get(0)?,
                run_id: row.get(1)?,
                kind: row.get(2)?,
                path: row.get(3)?,
                created_at: row.get(4)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// 每个任务最新的示例图（kind='sample'），炼丹记录列表缩略图用。
    pub fn latest_sample_per_run(&self) -> Result<Vec<(String, String)>, RepoError> {
        let mut stmt = self.conn.prepare(
            "SELECT c.run_id, c.path FROM checkpoints c
             WHERE c.kind = 'sample'
               AND c.created_at = (
                   SELECT MAX(created_at) FROM checkpoints
                   WHERE run_id = c.run_id AND kind = 'sample'
               )",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    // ---- Settings（设置） ----

    pub fn get_setting(&self, key: &str) -> Result<Option<String>, RepoError> {
        Ok(self
            .conn
            .query_row("SELECT value FROM settings WHERE key = ?1", [key], |row| {
                row.get(0)
            })
            .optional()?)
    }

    pub fn set_setting(&self, key: &str, value: &str) -> Result<(), RepoError> {
        self.conn.execute(
            "INSERT INTO settings (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    /// 删除设置项（不存在也视为成功；供“空值即删除”语义使用）。
    pub fn delete_setting(&self, key: &str) -> Result<(), RepoError> {
        self.conn
            .execute("DELETE FROM settings WHERE key = ?1", [key])?;
        Ok(())
    }

    pub fn list_settings(&self) -> Result<std::collections::BTreeMap<String, String>, RepoError> {
        let mut stmt = self
            .conn
            .prepare("SELECT key, value FROM settings ORDER BY key")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    // ---- ImageRecord（数据集图像） ----

    /// 批量写入扫描结果（先清空该数据集旧记录）。
    pub fn replace_dataset_images(
        &self,
        dataset_id: &str,
        records: &[ImageRecord],
    ) -> Result<(), RepoError> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "DELETE FROM image_files WHERE dataset_id = ?1",
            [dataset_id],
        )?;
        for r in records {
            tx.execute(
                "INSERT INTO image_files
                    (id, dataset_id, path, width, height, dhash, bucket, thumb, exif, duplicate_of, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    r.id,
                    r.dataset_id,
                    r.path,
                    r.width.map(|w| w as i64),
                    r.height.map(|h| h as i64),
                    r.dhash,
                    r.bucket,
                    r.thumb,
                    r.exif,
                    r.duplicate_of,
                    r.created_at
                ],
            )?;
        }
        // 同步数据集计数
        tx.execute(
            "UPDATE datasets SET image_count = (SELECT count(*) FROM image_files WHERE dataset_id = ?1) WHERE id = ?1",
            [dataset_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn list_dataset_images(&self, dataset_id: &str) -> Result<Vec<ImageRecord>, RepoError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, dataset_id, path, width, height, dhash, bucket, thumb, exif, duplicate_of, created_at
             FROM image_files WHERE dataset_id = ?1 ORDER BY path",
        )?;
        let rows = stmt.query_map([dataset_id], image_from_row)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn count_dataset_images(&self, dataset_id: &str) -> Result<u64, RepoError> {
        let n: i64 = self.conn.query_row(
            "SELECT count(*) FROM image_files WHERE dataset_id = ?1",
            [dataset_id],
            |row| row.get(0),
        )?;
        Ok(n as u64)
    }

    /// 桶分布（label -> 数量，按数量降序）。
    pub fn dataset_bucket_distribution(
        &self,
        dataset_id: &str,
    ) -> Result<Vec<(String, u64)>, RepoError> {
        let mut stmt = self.conn.prepare(
            "SELECT bucket, count(*) FROM image_files
             WHERE dataset_id = ?1 AND bucket IS NOT NULL
             GROUP BY bucket ORDER BY count(*) DESC",
        )?;
        let rows = stmt.query_map([dataset_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as u64))
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }
}

fn image_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ImageRecord> {
    Ok(ImageRecord {
        id: row.get(0)?,
        dataset_id: row.get(1)?,
        path: row.get(2)?,
        width: row.get::<_, Option<i64>>(3)?.map(|v| v as u32),
        height: row.get::<_, Option<i64>>(4)?.map(|v| v as u32),
        dhash: row.get(5)?,
        bucket: row.get(6)?,
        thumb: row.get(7)?,
        exif: row.get(8)?,
        duplicate_of: row.get(9)?,
        created_at: row.get(10)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tiandi_core::{Project, Run};

    fn store() -> Store {
        Store::open_in_memory().unwrap()
    }

    #[test]
    fn project_crud() {
        let s = store();
        let p = Project::new("测试", "D:\\ws");
        s.insert_project(&p).unwrap();
        assert_eq!(s.get_project(&p.id).unwrap().name, "测试");
        assert_eq!(s.list_projects().unwrap().len(), 1);
        let err = s.get_project("nope").unwrap_err();
        assert!(matches!(err, RepoError::NotFound { .. }));
    }

    #[test]
    fn run_lifecycle_via_store() {
        let s = store();
        let r = Run::new(None, None, None, None);
        s.insert_run(&r).unwrap();
        let got = s.get_run(&r.id).unwrap();
        assert_eq!(got.state, RunState::Created);

        s.update_run_state(&r.id, RunState::Queued, "t").unwrap();
        assert_eq!(s.get_run(&r.id).unwrap().state, RunState::Queued);
        assert_eq!(s.list_runs().unwrap().len(), 1);
    }

    #[test]
    fn base_model_family_roundtrip() {
        let s = store();
        let m = BaseModel::new(
            "NoobAI",
            ModelFamily::Sdxl1,
            Some("x.safetensors".into()),
            None,
            None,
        );
        s.insert_base_model(&m).unwrap();
        let list = s.list_base_models().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].family, ModelFamily::Sdxl1);
    }

    #[test]
    fn recipe_json_roundtrip() {
        let s = store();
        let r = Recipe::new(
            "入门",
            ModelFamily::DitAnima,
            serde_json::json!({"lr": 2e-4}),
        );
        s.insert_recipe(&r).unwrap();
        let list = s.list_recipes().unwrap();
        assert_eq!(list[0].family, ModelFamily::DitAnima);
        assert_eq!(list[0].data["lr"], serde_json::json!(2e-4));
    }

    #[test]
    fn metrics_upsert() {
        let s = store();
        let m1 = MetricPoint {
            run_id: "r1".into(),
            step: 1,
            loss: Some(0.5),
            lr: None,
        };
        let m2 = MetricPoint {
            run_id: "r1".into(),
            step: 1,
            loss: Some(0.3),
            lr: Some(1e-4),
        };
        s.insert_metric(&m1).unwrap();
        s.insert_metric(&m2).unwrap(); // upsert 同一 step
        let list = s.list_metrics("r1").unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].loss, Some(0.3));
    }

    #[test]
    fn checkpoint_crud() {
        let s = store();
        let c = Checkpoint {
            id: "c1".into(),
            run_id: "r1".into(),
            kind: "lora".into(),
            path: "runs/r1/lora.safetensors".into(),
            created_at: "t".into(),
        };
        s.insert_checkpoint(&c).unwrap();
        assert_eq!(s.list_checkpoints("r1").unwrap().len(), 1);
    }

    #[test]
    fn dataset_images_replace_and_distribute() {
        let s = store();
        let d = Dataset::new("测试集", "D:\\ds");
        s.insert_dataset(&d).unwrap();

        let mk = |path: &str, bucket: Option<&str>| ImageRecord {
            id: uuid::Uuid::new_v4().to_string(),
            dataset_id: d.id.clone(),
            path: path.into(),
            width: Some(1024),
            height: Some(1024),
            dhash: Some("abc".into()),
            bucket: bucket.map(String::from),
            thumb: None,
            exif: None,
            duplicate_of: None,
            created_at: "t".into(),
        };
        let records = vec![
            mk("a.png", Some("1024x1024")),
            mk("b.png", Some("1024x1024")),
            mk("c.png", Some("1024x768")),
        ];
        s.replace_dataset_images(&d.id, &records).unwrap();
        assert_eq!(s.count_dataset_images(&d.id).unwrap(), 3);
        assert_eq!(s.list_dataset_images(&d.id).unwrap().len(), 3);

        let dist = s.dataset_bucket_distribution(&d.id).unwrap();
        assert_eq!(
            dist,
            vec![("1024x1024".to_string(), 2), ("1024x768".to_string(), 1)]
        );

        // 再次替换会清空旧记录
        s.replace_dataset_images(&d.id, &[]).unwrap();
        assert_eq!(s.count_dataset_images(&d.id).unwrap(), 0);
        assert_eq!(s.list_dataset_images(&d.id).unwrap().len(), 0);
    }

    #[test]
    fn legacy_quoted_state_visible_to_queue_queries() {
        let s = store();
        // 旧格式：状态以带引号 JSON 字符串落库（parse_run_state 兼容，查询须匹配两种格式）
        s.conn()
            .execute(
                "INSERT INTO runs (id, project_id, dataset_id, recipe_id, base_model_id, state, manifest_path, created_at, updated_at)
                 VALUES ('legacy1', NULL, NULL, NULL, NULL, '\"queued\"', NULL, 't1', 't1')",
                [],
            )
            .unwrap();
        let r = Run::new(None, None, None, None);
        s.insert_run(&r).unwrap();
        s.update_run_state(&r.id, RunState::Queued, "t2").unwrap();

        assert_eq!(s.list_queued_runs().unwrap().len(), 2);
        assert!(!s.has_running_run().unwrap());

        // 旧格式 running 变体也应被 has_running_run 识别
        s.conn()
            .execute(
                "UPDATE runs SET state = '\"running\"' WHERE id = 'legacy1'",
                [],
            )
            .unwrap();
        assert!(s.has_running_run().unwrap());
        assert_eq!(s.list_queued_runs().unwrap().len(), 1);
    }

    #[test]
    fn claim_next_queued_picks_earliest_and_blocks_when_running() {
        let mut s = store();
        let mut r1 = Run::new(None, None, None, None);
        let mut r2 = Run::new(None, None, None, None);
        r1.created_at = "2024-01-01T00:00:00+00:00".into();
        r1.updated_at = r1.created_at.clone();
        r2.created_at = "2024-01-02T00:00:00+00:00".into();
        r2.updated_at = r2.created_at.clone();
        s.insert_run(&r1).unwrap();
        s.insert_run(&r2).unwrap();
        s.update_run_state(&r1.id, RunState::Queued, "2024-01-03T00:00:00+00:00")
            .unwrap();
        s.update_run_state(&r2.id, RunState::Queued, "2024-01-04T00:00:00+00:00")
            .unwrap();

        // 最早创建的先认领，且状态置 Preparing
        let claimed = s.claim_next_queued().unwrap().unwrap();
        assert_eq!(claimed.id, r1.id);
        assert_eq!(claimed.state, RunState::Preparing);
        assert_eq!(s.get_run(&r1.id).unwrap().state, RunState::Preparing);

        // 有运行中任务时不再认领，排队任务保持原状
        s.update_run_state(&r1.id, RunState::Running, "2024-01-05T00:00:00+00:00")
            .unwrap();
        assert!(s.claim_next_queued().unwrap().is_none());
        assert_eq!(s.get_run(&r2.id).unwrap().state, RunState::Queued);

        // 运行中任务结束后（Done），可继续认领下一个排队任务
        s.update_run_state(&r1.id, RunState::Done, "2024-01-06T00:00:00+00:00")
            .unwrap();
        let claimed2 = s.claim_next_queued().unwrap().unwrap();
        assert_eq!(claimed2.id, r2.id);
        assert_eq!(claimed2.state, RunState::Preparing);

        // 全部认领完后无排队任务，返回 None
        assert!(s.claim_next_queued().unwrap().is_none());
    }

    #[test]
    fn claim_next_queued_accepts_legacy_quoted_state() {
        let mut s = store();
        s.conn()
            .execute(
                "INSERT INTO runs (id, project_id, dataset_id, recipe_id, base_model_id, state, manifest_path, created_at, updated_at)
                 VALUES ('legacy1', NULL, NULL, NULL, NULL, '\"queued\"', NULL, 't1', 't1')",
                [],
            )
            .unwrap();
        let claimed = s.claim_next_queued().unwrap().unwrap();
        assert_eq!(claimed.id, "legacy1");
        assert_eq!(claimed.state, RunState::Preparing);
        assert_eq!(s.get_run("legacy1").unwrap().state, RunState::Preparing);
    }

    #[test]
    fn delete_dataset_recipe_run_not_found_keeps_data() {
        let s = store();
        let d = Dataset::new("测试集", "D:\\ds");
        s.insert_dataset(&d).unwrap();
        let err = s.delete_dataset("nope").unwrap_err();
        assert!(matches!(err, RepoError::NotFound { .. }));
        assert_eq!(s.list_datasets().unwrap().len(), 1);

        let r = Recipe::new("入门", ModelFamily::DitAnima, serde_json::json!({}));
        s.insert_recipe(&r).unwrap();
        let err = s.delete_recipe("nope").unwrap_err();
        assert!(matches!(err, RepoError::NotFound { .. }));
        assert_eq!(s.list_recipes().unwrap().len(), 1);

        let run = Run::new(None, None, None, None);
        s.insert_run(&run).unwrap();
        let err = s.delete_run("nope").unwrap_err();
        assert!(matches!(err, RepoError::NotFound { .. }));
        assert_eq!(s.list_runs().unwrap().len(), 1);

        // 正常删除成功（含关联 metrics/checkpoints 一并清理）
        let m = MetricPoint {
            run_id: run.id.clone(),
            step: 1,
            loss: Some(0.5),
            lr: None,
        };
        s.insert_metric(&m).unwrap();
        s.delete_run(&run.id).unwrap();
        assert!(s.list_runs().unwrap().is_empty());
        assert!(s.list_metrics(&run.id).unwrap().is_empty());
        s.delete_dataset(&d.id).unwrap();
        s.delete_recipe(&r.id).unwrap();
        assert!(s.list_datasets().unwrap().is_empty());
        assert!(s.list_recipes().unwrap().is_empty());
    }

    #[test]
    fn settings_delete() {
        let s = store();
        s.set_setting("k", "v").unwrap();
        assert_eq!(s.get_setting("k").unwrap().as_deref(), Some("v"));
        s.delete_setting("k").unwrap();
        assert_eq!(s.get_setting("k").unwrap(), None);
        // 0 行也算成功
        s.delete_setting("missing").unwrap();
    }
}
