# gguf-rs - Rust GGUF File Parser

A lightweight Rust library for loading, parsing and managing GGUF (GGML Universal Format) files written completely in Rust, with an accompanying CLI tool for inspecting GGUF files.

## Features

## Features

__Core functionality__
- [x] Parse GGUF file headers (magic number, version, tensor/kv counts)
- [x] Extract metadata key-value pairs with various data types
- [x] Basic tensor parsing
- [ ] No automatic model assembly or tensor processing for inference

- [x] Basic file validation
- [x] Simple CLI interface for file inspection

## Installation

```bash
git clone https://github.com/yourusername/gguf-rs
cd gguf-rs
cargo install --path .
```

## CLI Usage

```bash
gguf-rs <command> <file> [--verbose]

Commands:
  info      - Show basic file information
  metadata  - Display all metadata key-value pairs
  query     - Query specific metadata key (query <file> <key>)
  validate  - Validate GGUF file format
```

### Examples

Show file info:
```bash
gguf-rs info model.gguf
```

View metadata:
```bash
gguf-rs metadata model.gguf --verbose
```

Query specific key:
```bash
gguf-rs query model.gguf "tokenizer.ggml.model"
```

## Library Usage

Add to your `Cargo.toml`:
```toml
gguf-rs = { path = "./gguf-interface" }
```

Parse a file:
```rust
use std::fs::File;
use gguf_rs::{GgufHeader, GgufReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::open("model.gguf")?;
    let header = GgufHeader::parse(&mut file)?;
    let metadata = GgufReader::read_metadata(&mut file, header.n_kv)?;

    // Access metadata
    if let Some(Value::String(name)) = metadata.get("general.name") {
        println!("Model name: {}", name);
    }

    Ok(())
}
```

## Supported Features

- All standard GGUF value types
- GGUF versions 1-3
- Array metadata values

## License

Dual-licensed under MIT or Apache 2.0
