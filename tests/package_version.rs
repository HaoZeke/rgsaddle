//! Cargo.toml, pyproject.toml, and meson.build publish the same version
//! string.

use std::path::PathBuf;

fn section_version(text: &str, header: &str) -> String {
    let mut in_section = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            in_section = trimmed == header;
            continue;
        }
        if !in_section || trimmed.starts_with('#') {
            continue;
        }
        let Some(rest) = trimmed.strip_prefix("version") else {
            continue;
        };
        let rest = rest.trim_start();
        let Some(rest) = rest.strip_prefix('=') else {
            continue;
        };
        return rest.trim().trim_matches('"').to_string();
    }
    panic!("no version under {header}");
}

fn meson_project_version(text: &str) -> String {
    let start = text.find("project(").expect("meson.build calls project()");
    let call = &text[start..];
    let call = &call[..call.find(')').expect("project() closes")];
    for line in call.lines() {
        let Some(rest) = line.trim().strip_prefix("version:") else {
            continue;
        };
        return rest.trim().trim_end_matches(',').trim_matches('\'').to_string();
    }
    panic!("no version: in project()");
}

#[test]
fn cargo_and_pyproject_versions_agree() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let cargo = std::fs::read_to_string(root.join("Cargo.toml")).expect("read Cargo.toml");
    let py = std::fs::read_to_string(root.join("pyproject.toml")).expect("read pyproject.toml");
    let cargo_version = section_version(&cargo, "[package]");
    let py_version = section_version(&py, "[project]");
    assert_eq!(cargo_version, py_version);
    assert_eq!(cargo_version, env!("CARGO_PKG_VERSION"));
}

#[test]
fn cargo_and_meson_versions_agree() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let meson = std::fs::read_to_string(root.join("meson.build")).expect("read meson.build");
    assert_eq!(meson_project_version(&meson), env!("CARGO_PKG_VERSION"));
}
