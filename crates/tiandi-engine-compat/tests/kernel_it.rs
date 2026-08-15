//! 集成测试：真实 Python 进程跑 kernel_runner.py（mock 模式），验证 IPC 事件链路。
//!
//! 需要本机 Python 3.11+；无 Python 时测试自动跳过。

use std::path::PathBuf;
use std::time::Duration;

use tiandi_core::Event;
use tiandi_core::EventBus;
use tiandi_engine_compat::kernel::{spawn_kernel, KernelLaunch, KernelMode};

fn python() -> Option<PathBuf> {
    let env = tiandi_engine_compat::kernel::KernelEnv::detect();
    env.python
}

fn wrapper() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/kernel_runner.py")
}

#[tokio::test]
async fn mock_kernel_emits_full_event_stream() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init();
    let Some(python) = python() else {
        eprintln!("SKIP: 未检测到 Python");
        return;
    };

    let bus = EventBus::default();
    let mut rx = bus.subscribe();
    let tmp = tempfile::tempdir().unwrap();
    let config = tmp.path().join("mock.toml");
    std::fs::write(&config, "# mock\n").unwrap();

    let launch = KernelLaunch {
        python,
        wrapper: wrapper(),
        config_path: config,
        mode: KernelMode::Mock,
        env: vec![
            ("TIANDI_RUN_ID".into(), "itest-run".into()),
            ("TIANDI_MOCK_TOTAL".into(), "6".into()),
            ("TIANDI_MOCK_INTERVAL".into(), "0.05".into()),
        ],
        cwd: tmp.path().to_path_buf(),
    };

    let mut handle = spawn_kernel(&launch, move |v| {
        tiandi_engine_compat::kernel::publish_event(&bus, &v, "itest-run");
    })
    .expect("spawn mock kernel");

    let mut saw_hello = false;
    let mut saw_progress = false;
    let mut saw_metric = false;
    let mut saw_done = false;

    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    while tokio::time::Instant::now() < deadline {
        let ev = match tokio::time::timeout(Duration::from_millis(500), rx.recv()).await {
            Ok(Ok(ev)) => ev,
            Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => continue,
            Ok(Err(e)) => {
                eprintln!("[it] recv 错误：{e:?} → break");
                break;
            }
            Err(_) => {
                eprintln!("[it] recv 超时 → break");
                break;
            }
        };
        match &ev {
            Event::Hello { .. } => {
                eprintln!("[it] hello 收到");
                saw_hello = true;
            }
            Event::Progress { step, .. } => {
                if *step == 1 {
                    eprintln!("[it] progress 收到");
                    saw_progress = true;
                }
            }
            Event::Metric { .. } => {
                eprintln!("[it] metric 收到");
                saw_metric = true;
            }
            Event::Done { code, .. } => {
                eprintln!("[it] done 收到 code={code}");
                assert_eq!(*code, 0);
                saw_done = true;
            }
            other => eprintln!("[it] 其他事件：{other:?}"),
        }
        if saw_hello && saw_progress && saw_metric && saw_done {
            break;
        }
    }

    // 内核应已自行退出（mock 走完即 done）
    let _ = tokio::time::timeout(Duration::from_secs(3), handle.child_wait()).await;

    assert!(saw_hello, "应收到 hello 握手事件");
    assert!(saw_progress, "应收到 progress 事件");
    assert!(saw_metric, "应收到 metric 事件");
    assert!(saw_done, "应收到 done 事件");
}

#[tokio::test]
async fn cancel_kills_mock_kernel() {
    let Some(python) = python() else {
        eprintln!("SKIP: 未检测到 Python");
        return;
    };

    let bus = EventBus::default();
    let tmp = tempfile::tempdir().unwrap();
    let config = tmp.path().join("mock.toml");
    std::fs::write(&config, "# mock\n").unwrap();

    let launch = KernelLaunch {
        python,
        wrapper: wrapper(),
        config_path: config,
        mode: KernelMode::Mock,
        env: vec![
            ("TIANDI_RUN_ID".into(), "cancel-run".into()),
            ("TIANDI_MOCK_TOTAL".into(), "600".into()), // 长任务
            ("TIANDI_MOCK_INTERVAL".into(), "0.1".into()),
        ],
        cwd: tmp.path().to_path_buf(),
    };

    let mut handle = spawn_kernel(&launch, move |_v| {
        let _ = bus;
    })
    .expect("spawn mock kernel");

    // 等内核起来（hello 已发）再取消
    tokio::time::sleep(Duration::from_millis(300)).await;
    handle.cancel().await.expect("cancel 应成功");
    // 取消后句柄已退出：再 cancel 不再报错（幂等）
    let _ = handle.cancel().await;
}
