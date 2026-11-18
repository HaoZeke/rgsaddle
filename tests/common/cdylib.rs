/// Directory that holds `librgsaddle.so`.
///
/// The final cdylib stays in the profile directory (`target/debug`).
/// Cargo 1.100 places each unit under `build/<package>/<hash>/out`.
/// The test binary and the cdylib are sibling units there, so the
/// library is not an ancestor of the executable. Older Cargo kept the
/// binary in `deps` and the library one level up, or the library
/// inside `deps`.
fn cdylib_dir() -> std::path::PathBuf {
    let mut dir = std::env::current_exe().expect("test executable");
    dir.pop();
    let start = dir.clone();
    loop {
        if dir.join("librgsaddle.so").is_file() {
            return dir;
        }
        let deps = dir.join("deps");
        if deps.join("librgsaddle.so").is_file() {
            return deps;
        }
        if let Some(found) = sibling_cdylib(&dir) {
            return found;
        }
        if !dir.pop() {
            break;
        }
    }
    panic!(
        "build the C ABI library before testing (searched from {})",
        start.display()
    );
}

/// Cargo 1.100 keeps the cdylib in a sibling `build/<package>/<hash>/out`.
fn sibling_cdylib(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    if dir.file_name().is_none_or(|name| name != "out") {
        return None;
    }
    let pkg = dir.parent()?.parent()?;
    let entries = std::fs::read_dir(pkg).ok()?;
    for entry in entries.flatten() {
        let candidate = entry.path().join("out").join("librgsaddle.so");
        if candidate.is_file() {
            return candidate.parent().map(std::path::Path::to_path_buf);
        }
    }
    None
}
