use std::process::Command;

#[test]
fn status_switch_covers_every_enumerator() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = root.join("tests/c/status_switch.cpp");
    let output = std::env::temp_dir().join("rgsaddle-status-switch");
    let compile = Command::new("g++")
        .args([
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wswitch",
            "-Werror",
            "-I",
        ])
        .arg(root.join("include"))
        .arg("-I")
        .arg(root.join("bind"))
        .arg(&source)
        .arg("-o")
        .arg(&output)
        .output()
        .expect("g++ is required to compile the status switch");
    assert!(
        compile.status.success(),
        "status switch failed to compile:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = Command::new(&output)
        .output()
        .expect("status switch executable");
    assert!(
        run.status.success(),
        "status switch returned {}",
        run.status
    );
}
