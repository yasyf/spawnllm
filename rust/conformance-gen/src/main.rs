use std::process::ExitCode;

use conformance_gen::{Mode, run};

fn main() -> ExitCode {
    let mode = if std::env::args().skip(1).any(|arg| arg == "--write") {
        Mode::Write
    } else {
        Mode::Diff
    };
    if run(mode) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
