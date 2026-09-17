use std::io;

use serde::{Deserialize, Serialize};
use serde_json::ser::Formatter;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use unicode_normalization::UnicodeNormalization;

use crate::{OpError, OpResult, from_input, unimplemented};

#[derive(Debug, Deserialize)]
struct IsolationSourcesInput {
    host: IsolationHost,
}

#[derive(Debug, Deserialize)]
struct IsolationHost {
    platform: String,
    home: String,
    claude_config_dir_env: Option<String>,
    #[serde(default)]
    claude_securestorage_config_dir_env: Option<String>,
    #[serde(default)]
    claude_code_custom_oauth_url_env: Option<String>,
}

#[derive(Debug, Serialize)]
struct IsolationSources {
    account_path: String,
    credentials_path: String,
    keychain_service: Option<String>,
}

#[derive(Debug, Deserialize)]
struct IsolationSeedInput {
    account_json: Option<String>,
    credentials_json: Option<String>,
}

#[derive(Debug, Serialize)]
struct IsolationSeed {
    files: Vec<SeedFile>,
}

#[derive(Debug, Serialize)]
struct SeedFile {
    name: &'static str,
    content: String,
    mode: &'static str,
}

struct PythonFormatter;

impl Formatter for PythonFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(b": ")
    }
}

fn is_set(value: &Option<String>) -> bool {
    value.as_deref().is_some_and(|value| !value.is_empty())
}

fn config_dir_digest(config_dir_env: &str) -> String {
    let digest = format!(
        "{:x}",
        Sha256::digest(config_dir_env.nfc().collect::<String>().as_bytes())
    );
    format!("-{}", &digest[..8])
}

// Mirrors Claude Code 2.1.274's lb()/mI(): a defined CLAUDE_SECURESTORAGE_CONFIG_DIR
// overrides CLAUDE_CONFIG_DIR for the storage dir and the digest of the NFC value as set.
fn isolation_sources(input: IsolationSourcesInput) -> IsolationSources {
    let host = input.host;
    let default_home = format!("{}/.claude", host.home);
    let (account_path, config_home) = match &host.claude_config_dir_env {
        Some(config_dir_env) if !config_dir_env.is_empty() => {
            let config_home = config_dir_env.trim_end_matches('/');
            (
                format!("{config_home}/.claude.json"),
                config_home.to_owned(),
            )
        }
        _ => (format!("{}/.claude.json", host.home), default_home.clone()),
    };
    let (credentials_home, hashed_dir) = match &host.claude_securestorage_config_dir_env {
        Some(dir) if dir.is_empty() => (default_home, None),
        Some(dir) => (dir.trim_end_matches('/').to_owned(), Some(dir.as_str())),
        None => (
            config_home,
            host.claude_config_dir_env
                .as_deref()
                .filter(|dir| !dir.is_empty()),
        ),
    };
    let oauth_file_suffix = if is_set(&host.claude_code_custom_oauth_url_env) {
        "-custom-oauth"
    } else {
        ""
    };
    let keychain_service = (host.platform == "darwin").then(|| {
        let digest = hashed_dir.map(config_dir_digest).unwrap_or_default();
        format!("Claude Code{oauth_file_suffix}-credentials{digest}")
    });
    IsolationSources {
        account_path,
        credentials_path: format!("{credentials_home}/.credentials.json"),
        keychain_service,
    }
}

fn python_json(value: &impl Serialize) -> Result<String, serde_json::Error> {
    let mut output = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut output, PythonFormatter);
    value.serialize(&mut serializer)?;
    Ok(String::from_utf8(output).expect("JSON serializer emits UTF-8"))
}

fn isolation_seed(input: IsolationSeedInput) -> Result<IsolationSeed, serde_json::Error> {
    let mut files = Vec::new();
    if let Some(account_json) = input.account_json {
        let mut account = serde_json::from_str::<Map<String, Value>>(&account_json)?;
        account.remove("mcpServers");
        files.push(SeedFile {
            name: ".claude.json",
            content: python_json(&account)?,
            mode: "0644",
        });
    }
    if let Some(credentials_json) = input.credentials_json {
        files.push(SeedFile {
            name: ".credentials.json",
            content: credentials_json,
            mode: "0600",
        });
    }
    Ok(IsolationSeed { files })
}

pub(crate) fn dispatch(op: &str, input: Value) -> OpResult {
    match op {
        "claude_isolation_sources" => {
            let input = from_input::<IsolationSourcesInput>(input)?;
            serde_json::to_value(isolation_sources(input)).map_err(OpError::internal)
        }
        "claude_isolation_seed" => {
            let input = from_input::<IsolationSeedInput>(input)?;
            let seed = isolation_seed(input).map_err(OpError::internal)?;
            serde_json::to_value(seed).map_err(OpError::internal)
        }
        other => Err(unimplemented(other)),
    }
}
