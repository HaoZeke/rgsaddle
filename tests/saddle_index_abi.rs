//! Compiles and runs tests/c/saddle_index_abi.c against the built cdylib.

use std::path::PathBuf;
use std::process::Command;

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/cdylib.rs"
));

fn cc() -> Option<String> {
    [
        std::env::var("CC").ok(),
        Some("cc".into()),
        Some("gcc".into()),
    ]
    .into_iter()
    .flatten()
    .find(|c| Command::new(c).arg("--version").output().is_ok())
}

#[test]
#[cfg_attr(not(feature = "capi"), ignore)]
fn saddle_index_c_abi() {
    let Some(cc) = cc() else {
        eprintln!("no C compiler; skipping");
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out =
        std::env::temp_dir().join(format!("rgsaddle_saddle_index_abi_{}", std::process::id()));
    let libdir = cdylib_dir();

    let status = Command::new(&cc)
        .arg(root.join("tests/c/saddle_index_abi.c"))
        .arg("-I")
        .arg(root.join("include"))
        .arg("-L")
        .arg(&libdir)
        .arg("-lrgsaddle")
        .arg("-lm")
        .arg("-o")
        .arg(&out)
        .status()
        .expect("compile the saddle-index C ABI test");
    assert!(status.success(), "saddle-index C ABI test failed to build");

    let run = Command::new(&out)
        .env("LD_LIBRARY_PATH", &libdir)
        .output()
        .expect("run the saddle-index C ABI test");
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "saddle-index C ABI test failed:\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("RGSADDLE_SADDLE_INDEX_ABI_OK"),
        "stdout: {stdout}"
    );
    let _ = std::fs::remove_file(&out);
}
