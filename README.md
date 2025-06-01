# gguf-interface

[![crates.io](https://img.shields.io/crates/v/gguf-interface)](https://crates.io/crates/gguf-interface)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A comprehensive Rust library for parsing, analyzing, and working with GGUF files. The crate provides a type-safe interface to extract model configuration, metadata, and tensor data from GGUF format files used in LLMs.

## Features

- **Header Parsing**: Read and validate GGUF file headers
- **Metadata Extraction**: Extract key-value metadata from model files
- **Model Configuration**: Convert GGUF metadata into structured model configurations
- **Tensor Loading**: Load tensor data with automatic data type conversion
- **Model Organization**: Organize tensors into structured model layers (embedding, attention, feed-forward)
- **Cross-platform**: Works on Linux, macOS, and Windows

## Supported Tensor Types
Current implementation supports:
- `F32` (FP32)
- `F16` (FP16)
- `I8`, `I16`, `I32`, `I64`
- `F64`

*Quantized tensor types are recognized but currently not loaded (contributions welcome)*

## Quick Start

Add to your `Cargo.toml`:
```toml
[dependencies]
gguf-interface = "0.1"
```

Basic usage example:
```rust
use gguf_interface::*;
use std::fs::File;

fn main() -> Result<(), GgufError> {
    let mut file = File::open("model.gguf")?;

    // Parse file header
    let header = GgufHeader::parse(&mut file)?;

    // Read metadata
    let metadata = GgufReader::read_metadata(&mut file, header.n_kv)?;

    // Extract model configuration
    let config = extract_model_config(&metadata)?;

    // Read tensor information
    let tensor_infos = TensorLoader::read_tensor_info(&mut file, header.n_tensors)?;

    // Get tensor data start position
    let tensor_data_start = TensorLoader::get_tensor_data_start(&mut file)?;

    // Load all tensors
    let tensors = TensorLoader::load_all_tensors(&mut file, &tensor_infos, tensor_data_start)?;

    // Build structured model
    let model = ModelBuilder::new(tensors, config).build()?;

    println!("Loaded {} layers", model.num_layers());
    Ok(())
}
```

## Key Components

### Project Structure
```terminal
gguf-interface/
├── src/
│   ├── config.rs       // Model configuration extraction
│   ├── metadata.rs     // GGUF format parsing and types
│   ├── model.rs        // Model layer organization
│   ├── tensors.rs      // Tensor loading functionality
│   └── lib.rs          // Public API
└── README.md
```

### Core Modules
1. **Metadata Parsing**
   - Parses GGUF headers and key-value metadata
   - Supports all GGUF value types including arrays
   - Error handling for malformed files

2. **Model Configuration**
```rs gguf-interface/src/config.rs#L13-27
pub fn extract_model_config(metadata: &HashMap<String, Value>) -> Result<ModelConfig> {
    let architecture = /* ... */;
    let arch_prefix = &architecture;
    let block_count = get_u32_field(metadata, &format!("{}.block_count", arch_prefix))?;
    let context_length = get_u32_field(metadata, &format!("{}.context_length", arch_prefix))?;
    // ...
```
*Extracts model parameters like layers, dimensions, and attention heads*

3. **Tensor Loading**
   - Reads tensor information and binary data
   - Automatic FP16 → FP32 conversion
   - Memory-efficient loading
```rs gguf-interface/src/tensors.rs#L98-112
pub fn load_tensor<R: Read + Seek>(
    reader: &mut R,
    tensor_info: &TensorInfo,
    tensor_data_start: u64,
) -> Result<Tensor> {
    // ...
    let mut data = vec![0u8; byte_size as usize];
    reader.read_exact(&mut data)?;
    // ...
}
```

4. **Model Organization**
```rs gguf-interface/src/model.rs#L197-233
fn build_transformer_block(&mut self, layer_idx: usize) -> Result<TransformerBlock> {
    let prefix = format!("blk.{}", layer_idx);
    let attention = AttentionLayer {
        query_weights: self.take_tensor(&format!("{}.attn_q.weight", prefix))?,
        // ...
    };
    // ...
}
```
*Organizes tensors into structured layers (TransformerBlock, AttentionLayer, etc.)*

## Supported Models

The crate has been tested with these architectures:
- Qwen and adapted Llama-style architectures
- Models that follow the GGUF metadata standard

*Note: This crate is in active development. More model architectures will be added.*

## Roadmap

- [ ] Full quantized tensor support
- [ ] Async loading
- [ ] Memory mapping
- [ ] Enhanced error messages
- [ ] Validation against reference models

## Contributing

Contributions are welcome! Please open issues or pull requests for:
- Supporting additional GGUF features
- Adding new model architectures
- Performance improvements

## License

MIT License - see [LICENSE](LICENSE) for details.

---

This README provides a comprehensive overview of your crate that will be useful for potential users on crates.io. It includes installation instructions, a usage example, project structure details, and clear explanations of the core functionality.
