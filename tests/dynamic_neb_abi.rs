//! Compiles and runs tests/c/dynamic_neb_abi.c against the built cdylib.

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
fn dynamic_neb_c_abi() {
    let Some(cc) = cc() else {
        eprintln!("no C compiler; skipping");
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = std::env::temp_dir().join(format!("rgsaddle_dynamic_neb_abi_{}", std::process::id()));
    let libdir = cdylib_dir();
    let status = Command::new(&cc)
        .arg(root.join("tests/c/dynamic_neb_abi.c"))
        .arg("-I")
        .arg(root.join("include"))
        .arg("-I")
        .arg(root.join("tests/c"))
        .arg("-L")
        .arg(&libdir)
        .arg("-lrgsaddle")
        .arg("-lm")
        .arg("-o")
        .arg(&out)
        .status()
        .expect("compile dynamic neb abi");
    assert!(status.success(), "C compile failed");
    let run = Command::new(&out)
        .env("LD_LIBRARY_PATH", &libdir)
        .output()
        .expect("run dynamic neb abi");
    assert!(
        run.status.success(),
        "dynamic neb abi failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(String::from_utf8_lossy(&run.stdout).contains("RGSADDLE_DYNAMIC_NEB_ABI_OK"));
}
