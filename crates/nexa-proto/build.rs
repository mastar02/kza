use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_root = PathBuf::from("proto");
    let protos = [
        proto_root.join("nexa/v1/common.proto"),
        proto_root.join("nexa/v1/audio.proto"),
        proto_root.join("nexa/v1/ml.proto"),
        proto_root.join("nexa/v1/memory.proto"),
        proto_root.join("nexa/v1/control.proto"),
    ];

    for proto in &protos {
        println!("cargo:rerun-if-changed={}", proto.display());
    }

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&protos, &[proto_root])?;

    Ok(())
}
