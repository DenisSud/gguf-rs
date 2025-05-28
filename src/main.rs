use std::env;
use std::path::Path; // This import is unused, will keep the warning for now

use ggml_qwen3::{GgmlFile, Qwen3Model}; // Removed GgmlTensor

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

            // Print Embedding layer details
            println!("  Embedding tensor:");
            // Assuming model.embed_tokens directly holds struct with name, shape, and dtype
            println!("    Name: {}", model.embed_tokens.name);
            println!("    Shape: {:?}", model.embed_tokens.shape);
            println!("    Dtype: {:?}", model.embed_tokens.dtype);


            println!("\n  Transformer Layers:");
            // Print details for each transformer layer
            for (i, layer) in model.layers.iter().enumerate() {
                println!("  Layer {}:", i);
                // Assuming layer.input_layernorm holds struct with name, shape, and dtype
                println!("    Input layernorm:");
                println!("      Name: {}", layer.input_layernorm.name);
                println!("      Shape: {:?}", layer.input_layernorm.shape);
                println!("      Dtype: {:?}", layer.input_layernorm.dtype);

                println!("    Attention:");
                // Assuming layer.attn.q_proj etc. hold structs with name, shape, and dtype
                println!("      Q projection:");
                println!("        Name: {}", layer.attn.q_proj.name);
                println!("        Shape: {:?}", layer.attn.q_proj.shape);
                println!("        Dtype: {:?}", layer.attn.q_proj.dtype);

                println!("      K projection:");
                println!("        Name: {}", layer.attn.k_proj.name);
                println!("        Shape: {:?}", layer.attn.k_proj.shape);
                println!("        Dtype: {:?}", layer.attn.k_proj.dtype);

                println!("      V projection:");
                println!("        Name: {}", layer.attn.v_proj.name);
                println!("        Shape: {:?}", layer.attn.v_proj.shape);
                println!("        Dtype: {:?}", layer.attn.v_proj.dtype);

                println!("      O projection:");
                println!("        Name: {}", layer.attn.o_proj.name);
                println!("        Shape: {:?}", layer.attn.o_proj.shape);
                println!("        Dtype: {:?}", layer.attn.o_proj.dtype);


                println!("    MLP:");
                // Assuming layer.mlp.gate_proj etc. hold structs with name, shape, and dtype
                println!("      Gate projection:");
                println!("        Name: {}", layer.mlp.gate_proj.name);
                println!("        Shape: {:?}", layer.mlp.gate_proj.shape);
                println!("        Dtype: {:?}", layer.mlp.gate_proj.dtype);

                println!("      Up projection:");
                println!("        Name: {}", layer.mlp.up_proj.name);
                println!("        Shape: {:?}", layer.mlp.up_proj.shape);
                println!("        Dtype: {:?}", layer.mlp.up_proj.dtype);

                println!("      Down projection:");
                println!("        Name: {}", layer.mlp.down_proj.name);
                println!("        Shape: {:?}", layer.mlp.down_proj.shape);
                println!("        Dtype: {:?}", layer.mlp.down_proj.dtype);

            }

        },
        Err(err) => {
            println!("Failed to parse as Qwen3 model: {:?}", err);
        }
    }

    Ok(())
}
