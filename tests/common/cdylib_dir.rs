/// Directory that holds `librgsaddle.so`.
///
/// Cargo 1.100 puts each unit under
/// `target/<profile>/build/<pkg>/<hash>/out`. The cdylib is a sibling
/// unit of the test binary, and may also be hard-linked into
/// `target/<profile>`. Earlier cargo put the binary in
/// `target/<profile>/deps` and the cdylib next to that directory.
/// Walk up from the executable until one of those layouts matches.
fn cdylib_dir() -> std::path::PathBuf {
    let mut dir = std::env::current_exe().expect("test executable");
    dir.pop();
    let start = dir.clone();
    for _ in 0..8 {
        if dir.join("librgsaddle.so").is_file() {
            return dir;
        }
        let deps = dir.join("deps");
        if deps.join("librgsaddle.so").is_file() {
            return deps;
        }
        if dir.file_name().is_some_and(|name| name == "out") {
            if let Some(pkg) = dir.parent().and_then(|hash| hash.parent()) {
                if let Ok(entries) = std::fs::read_dir(pkg) {
                    for entry in entries.flatten() {
                        let candidate = entry.path().join("out").join("librgsaddle.so");
                        if candidate.is_file() {
                            return candidate
                                .parent()
                                .map(std::path::Path::to_path_buf)
                                .unwrap_or(start);
                        }
                    }
                }
            }
        }
        if !dir.pop() {
            break;
        }
    }
    start
}
