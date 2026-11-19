//! The readable files named on a command line or dropped on the app, folders walked recursively in name order.

use std::path::{Path, PathBuf};

use crate::extract;

/// Every readable file under `path` (a folder, walked recursively in name order), or `path` itself if it is a file.
pub fn files_in(path: &Path) -> Result<Vec<PathBuf>, String> {
    fn walk(path: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(path)
            .map_err(|e| format!("{}: {e}", path.display()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .collect();
        entries.sort();
        for entry in entries {
            if entry.is_dir() {
                walk(&entry, out)?;
            } else if extract::readable(&entry) {
                out.push(entry);
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    if path.is_file() {
        out.push(path.to_owned());
    } else if path.is_dir() {
        walk(path, &mut out)?;
    } else {
        return Err(format!("{}: no such file or folder", path.display()));
    }
    Ok(out)
}

/// Every readable file named: files as given (whatever their kind), `-` for standard input, folders walked.
pub fn collect(inputs: &[String]) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for input in inputs {
        let path = Path::new(input);
        if input == "-" || path.is_file() {
            out.push(input.clone());
        } else if path.is_dir() {
            out.extend(files_in(path)?.into_iter().map(|file| file.display().to_string()));
        } else {
            return Err(format!("{input}: no such file or folder"));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folders_walk_in_name_order_and_skip_unreadable_kinds() {
        let root = std::env::temp_dir().join(format!("litmus_walk_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("b")).unwrap();
        for name in ["z.txt", "a.md", "skip.png", "b/c.html", "b/d.docx"] {
            std::fs::write(root.join(name), b"x").unwrap();
        }
        let names: Vec<String> = files_in(&root).unwrap().iter().map(|p| p.strip_prefix(&root).unwrap().display().to_string().replace('\\', "/")).collect();
        assert_eq!(names, ["a.md", "b/c.html", "b/d.docx", "z.txt"]);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
