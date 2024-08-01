fn main() {
    tonic_build::configure()
        .out_dir("src/")
        .compile(&["proto/infer.proto"], &["proto"])
        .unwrap();
}