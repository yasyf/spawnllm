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

    let mut child = cmd.spawn()?;

    {
        let mut stdin = child.stdin.take().expect("stdin piped");
        stdin.write_all(plan.stdin.as_bytes()).await?;
        stdin.shutdown().await.ok();
    }

    let piped_stdout = child.stdout.take();
    let mut stderr_handle = child.stderr.take().expect("stderr piped");
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
    let wait = child.wait();
    let combined = async { tokio::join!(read_out, read_err, wait) };

    let outcome = tokio::time::timeout(spec.timeout, combined).await;
    match outcome {
        Err(_) => {
            reap(&mut child).await;
            Ok(Attempt {
                output: String::new(),
                kind: AttemptKind::Timeout {
                    duration: spec.timeout,
                },
            })
        }
        Ok((stdout_res, stderr_res, status_res)) => {
            let stdout_bytes = stdout_res?;
            let stderr_bytes = stderr_res?;
            let status = status_res?;
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
        if tokio::time::timeout(Duration::from_secs(2), child.wait())
            .await
            .is_ok()
        {
            return;
        }
    }
    let _ = child.start_kill();
    let _ = child.wait().await;
}
