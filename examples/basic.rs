use gguf_rs::GGUFModel;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get model path from command line
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).ok_or("Please provide a GGUF file path")?;

    // Load the GGUF model
    let model = GGUFModel::load(model_path)?;

    // Print basic info
    println!("Successfully loaded GGUF v{} model", model.header.version);
    println!("Tensors: {}", model.tensors.len());
    println!("Alignment: {}", model.header.alignment);

    // Print some metadata
    if let Some(arch) = model.header.metadata.get("general.architecture") {
        println!("Architecture: {}", arch.as_string().unwrap_or("unknown"));
    }

    if let Some(ctx_len) = model.header.metadata.get("llama.context_length") {
        println!("Context length: {}", ctx_len.as_uint32().unwrap_or(0));
    }

    // Print first 5 tensors
    println!("\nFirst 5 tensors:");
    for tensor in model.tensors.iter().take(5) {
        println!(
            "- {}: {:?} (type: {})",
            tensor.name, tensor.dimensions, tensor.type_
        );
    }

    Ok(())
}
