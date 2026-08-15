//! 领域模型与用例：项目/基底模型/数据集/丹方/任务状态机、事件总线。
//!
//! M0 骨架：实体（[`domain`]）、任务状态机（[`state`]）、事件总线（[`events`]）。
//! 用例服务（创建炼丹任务、队列调度等）在 M1 随数据管线一起落地。

pub mod domain;
pub mod events;
pub mod state;

pub use domain::{BaseModel, Checkpoint, Dataset, MetricPoint, ModelFamily, Project, Recipe, Run};
pub use events::{Event, EventBus};
pub use state::{RunState, RunStateError};
