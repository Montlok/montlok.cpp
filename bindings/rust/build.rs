//! Compiles the montlok runtime from this checkout (a git dependency checks
//! out the whole repository, so the C++ sources are two directories up).

use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let root = manifest.join("../..").canonicalize().expect("the montlok.cpp checkout");
    let core = root.join("src/core");
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .include(root.join("include"))
        .include(&core)
        .include(root.join("microstructure"))
        .opt_level(3)
        .flag_if_supported("-fno-math-errno")
        .flag_if_supported("-ffp-contract=fast")
        .flag_if_supported("-fvisibility=hidden")
        .warnings(false);
    for name in ["archive", "c_api", "json", "plugin", "runtime", "sequential", "trees"] {
        build.file(core.join(format!("{name}.cpp")));
    }
    build.file(root.join("microstructure/bindings.cpp"));
    build.compile("montlok");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=dl");
    }
    for dir in ["src", "include", "microstructure"] {
        println!("cargo:rerun-if-changed={}", root.join(dir).display());
    }
}
