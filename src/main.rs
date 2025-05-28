use std::env;
use std::path::Path;

use ggml_qwen3::{GgmlFile, Qwen3Model};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get model path from command line arguments
    let args: Vec<String> = env::args().collect();
    let model_path = if args.len() > 1 {
        &args[1]
    } else {
        eprintln!("Usage: {} <path-to-gguf-model>", args[0]);
        return Ok(());
    };

    println!("Loading model from: {}", model_path);
    
    // Open the GGUF file
    let ggml = GgmlFile::open(model_path)?;
    
    // Print tensor count
    let tensors: Vec<_> = ggml.tensors().collect();
    println!("Model contains {} tensors", tensors.len());
    
    // Print first few tensors
    println!("\nFirst 5 tensors:");
    for (i, tensor) in tensors.iter().take(5).enumerate() {
        println!("  {}: {} {:?} {:?}", i, tensor.name, tensor.shape, tensor.dtype);
    }
    
    // Try to parse as Qwen3 model
    println!("\nAttempting to parse as Qwen3 model...");
    match Qwen3Model::from_ggml(&ggml) {
        Ok(model) => {
            println!("Successfully parsed Qwen3 model!");
            println!("Model configuration:");
            println!("  Hidden size: {}", model.config.hidden_size);
            println!("  Intermediate size: {}", model.config.intermediate_size);
            println!("  Attention heads: {}", model.config.num_attention_heads);
            println!("  Hidden layers: {}", model.config.num_hidden_layers);
            println!("  Vocabulary size: {}", model.config.vocab_size);
            
            println!("\nModel structure:");
            println!("  Embedding tensor: {} {:?}", model.embed_tokens.name, model.embed_tokens.shape);
            println!("  Number of layers: {}", model.layers.len());
            
            // Print details of first layer
            if !model.layers.is_empty() {
                let layer = &model.layers[0];
                println!("\nFirst layer details:");
                println!("  Input layernorm: {} {:?}", layer.input_layernorm.name, layer.input_layernorm.shape);
                println!("  Q projection: {} {:?}", layer.attn.q_proj.name, layer.attn.q_proj.shape);
                println!("  K projection: {} {:?}", layer.attn.k_proj.name, layer.attn.k_proj.shape);
                println!("  V projection: {} {:?}", layer.attn.v_proj.name, layer.attn.v_proj.shape);
                println!("  O projection: {} {:?}", layer.attn.o_proj.name, layer.attn.o_proj.shape);
                println!("  Gate projection: {} {:?}", layer.mlp.gate_proj.name, layer.mlp.gate_proj.shape);
                println!("  Up projection: {} {:?}", layer.mlp.up_proj.name, layer.mlp.up_proj.shape);
                println!("  Down projection: {} {:?}", layer.mlp.down_proj.name, layer.mlp.down_proj.shape);
            }
            
            // Demonstrate tensor data access (zero-copy)
            if let Some(tensor) = ggml.get_tensor("model.embed_tokens.weight") {
                let data = ggml.tensor_data(tensor);
                println!("\nEmbed tokens data (first 16 bytes): {:?}", &data[..16.min(data.len())]);
            }
        },
        Err(err) => {
            println!("Failed to parse as Qwen3 model: {:?}", err);
        }
    }
    
    Ok(())
}
