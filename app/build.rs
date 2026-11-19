//! On Windows (MSVC), draw the mark into an icon resource and link it into `litmus-app.exe`, so Explorer and the
//! taskbar show it on the file itself (`src/icon_resource.rs`). Other targets need nothing here.

#[allow(dead_code)]
#[path = "src/brand.rs"]
mod brand;
#[allow(dead_code)]
#[path = "src/icon_resource.rs"]
mod icon_resource;
#[allow(dead_code)]
#[path = "src/png.rs"]
mod png;

fn main() {
    for file in ["build.rs", "src/brand.rs", "src/icon_resource.rs", "src/png.rs"] {
        println!("cargo:rerun-if-changed={file}");
    }
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if os != "windows" || env != "msvc" {
        return;
    }
    let images = icon_resource::images(brand::icon_rgba, |size, rgba| png::encode(size as u32, size as u32, rgba));
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR")).join("litmus_icon.res");
    std::fs::write(&out, icon_resource::res(&images)).expect("the icon resource is written");
    println!("cargo:rustc-link-arg-bin=litmus-app={}", out.display());
}
