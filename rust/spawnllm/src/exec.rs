use std::io::Write;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use tempfile::{Builder, NamedTempFile};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};

use spawnllm_core::wire::{ExecPlan, FileId, ReadResultFrom};

use crate::run::{Attempt, AttemptKind, resolve_kind};
use crate::spec::RunSpec;

pub(crate) async fn exec_attempt(
    plan: &ExecPlan,
    spec: &RunSpec,
    provider: &str,
    isolated_dir: Option<&Path>,
    wants_value: bool,
) -> std::io::Result<Attempt> {
    let mut temp_files: Vec<NamedTempFile> = Vec::new();
    let mut schema_path = None;
    let mut result_path = None;
    let mut stdout_path = None;
    for file in &plan.files {
        let mut handle = Builder::new().suffix(&file.suffix).tempfile()?;
        if let Some(content) = &file.content {
            handle.write_all(content.as_bytes())?;
            handle.flush()?;
        }
        let path = handle.path().to_path_buf();
        match file.id {
            FileId::Schema => schema_path = Some(path),
            FileId::Result => result_path = Some(path),
            FileId::Stdout => stdout_path = Some(path),
        }
        temp_files.push(handle);
    }

    let argv: Vec<String> = plan
        .argv
        .iter()
        .map(|arg| match arg.as_str() {
            "${file:schema}" => schema_path
                .as_ref()
                .expect("schema file materialized")
                .to_string_lossy()
                .into_owned(),
            "${file:result}" => result_path
                .as_ref()
                .expect("result file materialized")
                .to_string_lossy()
                .into_owned(),
            other => other.to_owned(),
        })
        .collect();

    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..]);
    cmd.stdin(Stdio::piped());
    cmd.stderr(Stdio::piped());
    for (key, value) in &plan.env {
        let value = match isolated_dir {
            Some(dir) => value.replace("${isolated_config_dir}", &dir.to_string_lossy()),
            None => value.clone(),
        };
        cmd.env(key, value);
    }
    if let Some(env) = &spec.env {
        for (key, value) in env {
            cmd.env(key, value);
        }
    }
    if let Some(cwd) = &spec.cwd {
        cmd.current_dir(cwd);
    }
    if plan.stdout_to_file {
        let path = stdout_path.as_ref().expect("stdout file materialized");
        cmd.stdout(Stdio::from(std::fs::File::create(path)?));
    } else {
        cmd.stdout(Stdio::piped());
    }
    cmd.kill_on_drop(true);

    let mut child = ChildGuard::new(cmd.spawn()?);
    let mut stdin = child.child_mut().stdin.take().expect("stdin piped");
    let piped_stdout = child.child_mut().stdout.take();
    let mut stderr_handle = child.child_mut().stderr.take().expect("stderr piped");
    let write_in = async move {
        stdin.write_all(plan.stdin.as_bytes()).await?;
        stdin.shutdown().await.ok();
        Ok::<(), std::io::Error>(())
    };
    let read_out = async move {
        let mut buf = Vec::new();
        if let Some(mut handle) = piped_stdout {
            handle.read_to_end(&mut buf).await?;
        }
        Ok::<Vec<u8>, std::io::Error>(buf)
    };
    let read_err = async move {
        let mut buf = Vec::new();
        stderr_handle.read_to_end(&mut buf).await?;
        Ok::<Vec<u8>, std::io::Error>(buf)
    };
    let wait = child.child_mut().wait();
    let combined = async move {
        let (_, stdout, stderr, status) = tokio::try_join!(write_in, read_out, read_err, wait)?;
        Ok::<_, std::io::Error>((stdout, stderr, status))
    };

    let outcome = tokio::time::timeout(spec.timeout, combined).await;
    child.reap().await;
    match outcome {
        Err(_) => Ok(Attempt {
            output: String::new(),
            kind: AttemptKind::Timeout {
                duration: spec.timeout,
            },
        }),
        Ok(result) => {
            let (stdout_bytes, stderr_bytes, status) = result?;
            let returncode = status.code().map_or(-1, i64::from);
            let stderr = String::from_utf8_lossy(&stderr_bytes).into_owned();
            let raw = read_raw(
                plan,
                stdout_path.as_deref(),
                result_path.as_deref(),
                &stdout_bytes,
            )?;
            drop(temp_files);
            let kind = resolve_kind(provider, &raw, returncode, &stderr, wants_value);
            Ok(Attempt { output: raw, kind })
        }
    }
}

struct ChildGuard {
    child: Option<Child>,
}

impl ChildGuard {
    fn new(child: Child) -> Self {
        Self { child: Some(child) }
    }

    fn child_mut(&mut self) -> &mut Child {
        self.child.as_mut().expect("child is present")
    }

    async fn reap(&mut self) {
        if let Some(mut child) = self.child.take() {
            reap(&mut child).await;
        }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            // Cancellation cannot await cleanup, so transfer the child to a live runtime task.
            tokio::spawn(async move { reap(&mut child).await });
        }
    }
}

fn read_raw(
    plan: &ExecPlan,
    stdout_path: Option<&Path>,
    result_path: Option<&Path>,
    piped: &[u8],
) -> std::io::Result<String> {
    let bytes = if plan.stdout_to_file {
        std::fs::read(stdout_path.expect("stdout file materialized"))?
    } else if matches!(plan.read_result_from, ReadResultFrom::FileResult) {
        std::fs::read(result_path.expect("result file materialized"))?
    } else {
        return Ok(String::from_utf8_lossy(piped).into_owned());
    };
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

async fn reap(child: &mut Child) {
    if matches!(child.try_wait(), Ok(Some(_))) {
        return;
    }
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        // SIGTERM first, then SIGKILL after a grace period.
        unsafe {
            libc::kill(pid as libc::pid_t, libc::SIGTERM);
        }
        if matches!(
            tokio::time::timeout(Duration::from_secs(2), child.wait()).await,
            Ok(Ok(_))
        ) {
            return;
        }
    }
    let _ = child.start_kill();
    let _ = child.wait().await;
}
