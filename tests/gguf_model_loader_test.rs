//! Integration test for loading an actual GGUF model file
//!
//! This test verifies that we can parse metadata, extract model configuration,
//! and organize tensors into proper model structures.

use std::fs::File;
use std::io::BufReader;

// Explicitly declare the crate dependency used for testing
extern crate gguf_rs;
use gguf_rs::{
    GGUF_MAGIC, GgufHeader, GgufReader, ModelBuilder, TensorLoader, extract_model_config,
};

const MODEL_PATH: &str = "tests/data/Qwen3-0.6B-F16.gguf";

#[test]
fn load_qwen_model() -> Result<(), Box<dyn std::error::Error>> {
    // Open model file using buffered I/O
    let file = File::open(MODEL_PATH)?;
    let mut reader = BufReader::new(file);

    // Verify GGUF header format
    let header = GgufHeader::parse(&mut reader)?;
    assert_eq!(header.magic, GGUF_MAGIC);
    assert!(
        header.is_version_supported(),
        "Unsupported GGUF version: {}",
        header.version
    );

    // Load metadata
    let metadata = GgufReader::read_metadata(&mut reader, header.n_kv)?;

    // Extract model configuration from metadata
    let config = extract_model_config(&metadata)?;

    // Verify crucial model parameters
    assert_eq!(&config.architecture, "qwen3");
    assert_eq!(config.block_count, 28); // Verify actual value based on model
    assert_eq!(config.context_length, 40960); // Verify actual value based on model

    // Load tensor information
    let tensor_infos = TensorLoader::read_tensor_info(&mut reader, header.n_tensors)?;
    assert_eq!(tensor_infos.len() as u64, header.n_tensors);

    // Find tensor data section
    let tensor_data_start = TensorLoader::get_tensor_data_start(&mut reader)?;

    // Load all model tensors
    let tensors = TensorLoader::load_all_tensors(&mut reader, &tensor_infos, tensor_data_start)?;

    // Build model structure
    let model = ModelBuilder::new(tensors, config).build()?;

    // Validate model structure
    assert_eq!(model.vocab_size(), 151936);
    assert_eq!(model.model_dim(), 1024);
    assert_eq!(model.num_layers(), 28); // Should match block_count

    // Verify tensor organization
    assert_eq!(
        model.embeddings.token_embeddings.info.dims,
        vec![1024, 151936] // [embedding_dim, vocab_size]
    );

    // Validate attention layer dimensions
    let block0 = model.get_block(0).unwrap();
    assert_eq!(block0.attention.model_dim(), 1024);
    assert_eq!(block0.attention.attention_dim(), 2048); // For multi-head models this would differ

    Ok(())
}
