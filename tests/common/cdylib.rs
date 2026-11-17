/// Directory that holds `librgsaddle.so`.
///
/// The final cdylib stays in the profile directory (`target/debug`).
/// Cargo 1.100 places the test binary under `build/<package>/<hash>/out`
/// instead of `deps`, so the library is an ancestor of the executable.
/// Older Cargo kept the binary in `deps` and the library one level up,
/// or the library inside `deps`.
///
/// A `[profile.test]` opt-level that differs from dev builds the cdylib
/// as its own unit. Cargo then leaves `librgsaddle.so` in that unit's
/// `out` directory, a sibling of the test binary's hash, not an ancestor.
fn cdylib_dir() -> std::path::PathBuf {
    let mut dir = std::env::current_exe().expect("test executable");
    dir.pop();
    let start = dir.clone();
    let mut cursor = dir;
    loop {
        if cursor.join("librgsaddle.so").is_file() {
            return cursor;
        }
        let deps = cursor.join("deps");
        if deps.join("librgsaddle.so").is_file() {
            return deps;
        }
        if !cursor.pop() {
            break;
        }
    }
    if let Some(found) = newest_sibling_cdylib(&start) {
        return found;
    }
    panic!(
        "build the C ABI library before testing (searched from {})",
        start.display()
    );
}

/// Newest `librgsaddle.so` under `build/<pkg>/<hash>/out` next to `start`.
fn newest_sibling_cdylib(start: &std::path::Path) -> Option<std::path::PathBuf> {
    let hash_dir = start.parent()?;
    let pkg_dir = hash_dir.parent()?;
    if pkg_dir.file_name().and_then(|name| name.to_str()) != Some("rgsaddle") {
        return None;
    }
    let entries = std::fs::read_dir(pkg_dir).ok()?;
    let mut best: Option<(std::time::SystemTime, std::path::PathBuf)> = None;
    for entry in entries.flatten() {
        let out = entry.path().join("out");
        let library = out.join("librgsaddle.so");
        if !library.is_file() {
            continue;
        }
        let Ok(modified) = library.metadata().and_then(|meta| meta.modified()) else {
            continue;
        };
        let replace = match &best {
            Some((previous, _)) => modified > *previous,
            None => true,
        };
        if replace {
            best = Some((modified, out));
        }
    }
    best.map(|(_, dir)| dir)
}
