# gguf-rs (WIP ⚠️)

**A simple Rust parser for GGUF files**
*Work in Progress - Not Production Ready*

## About

This is an experimental Rust implementation for reading GGUF (GPT-Generated Unified Format) files.
Use this only for learning/testing purposes - not recommended for production use yet.

## Basic Usage

1. Add to your `Cargo.toml`:
```toml
[dependencies]
gguf-rs = { git = "https://github.com/yourusername/gguf-rs" }
```

2. Load a GGUF file:
```rust
use gguf_rs::GGUFModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = GGUFModel::load("your-model.gguf")?;

    // Basic info
    println!("Version: {}", model.header.version);
    println!("Tensors: {}", model.tensors.len());

    // Access metadata
    if let Some(arch) = model.header.metadata.get("general.architecture") {
        println!("Architecture: {}", arch.as_string().unwrap_or("unknown"));
    }

    Ok(())
}
```

## Current Features

- Read GGUF file headers
- Extract basic metadata
- List tensor information
- Simple error handling

## Contributing

This is a learning project - feedback and simple fixes welcome!
Open an issue first before sending PRs.

## License

MIT
