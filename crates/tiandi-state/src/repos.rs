//! 仓储层：对 SQLite 的薄封装（单人本地工具，单连接 + 互斥即可）。
//!
//! 领域实体 ↔ 行映射集中在各 repo 方法内；`Store` 持有连接并提供事务边界。

use rusqlite::{params, Connection, OptionalExtension};
use tiandi_core::{
    BaseModel, Checkpoint, Dataset, MetricPoint, ModelFamily, Project, Recipe, Run, RunState,
};

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

    // ---- Run ----

    pub fn insert_run(&self, r: &Run) -> Result<(), RepoError> {
        self.conn.execute(
            "INSERT INTO runs (id, project_id, dataset_id, recipe_id, state, manifest_path, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                r.id,
                r.project_id,
                r.dataset_id,
                r.recipe_id,
                serde_json::to_string(&r.state).expect("RunState 可序列化"),
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
                "SELECT id, project_id, dataset_id, recipe_id, state, manifest_path, created_at, updated_at
                 FROM runs WHERE id = ?1",
                [id],
                |row| {
                    let state: String = row.get(4)?;
                    Ok(Run {
                        id: row.get(0)?,
                        project_id: row.get(1)?,
                        dataset_id: row.get(2)?,
                        recipe_id: row.get(3)?,
                        state: serde_json::from_str(&state).expect("RunState 可反序列化"),
                        manifest_path: row.get(5)?,
                        created_at: row.get(6)?,
                        updated_at: row.get(7)?,
                    })
                },
            )
            .optional()?
            .ok_or_else(|| RepoError::NotFound { entity: "run", id: id.into() })
    }

    pub fn list_runs(&self) -> Result<Vec<Run>, RepoError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, project_id, dataset_id, recipe_id, state, manifest_path, created_at, updated_at
             FROM runs ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            let state: String = row.get(4)?;
            Ok(Run {
                id: row.get(0)?,
                project_id: row.get(1)?,
                dataset_id: row.get(2)?,
                recipe_id: row.get(3)?,
                state: serde_json::from_str(&state).expect("RunState 可反序列化"),
                manifest_path: row.get(5)?,
                created_at: row.get(6)?,
                updated_at: row.get(7)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
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
            params![
                serde_json::to_string(&state).expect("RunState 可序列化"),
                updated_at,
                id
            ],
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
        let r = Run::new(None, None, None);
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
}
