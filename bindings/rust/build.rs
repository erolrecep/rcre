fn main() {
    println!("cargo:rustc-link-search=native=../../lib");
    println!("cargo:rustc-link-lib=rcre");
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=AVFoundation");
        println!("cargo:rustc-link-lib=framework=CoreVideo");
        println!("cargo:rustc-link-lib=framework=CoreMedia");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
}
