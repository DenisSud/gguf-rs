use std::fs::File;
use std::path::PathBuf;
use std::process;

use gguf_rs::metadata::{GgufHeader, GgufReader, Value, GGUF_MAGIC};

/// Command line interface for GGUF file inspection
#[derive(Debug)]
struct Args {
    command: Command,
    file_path: PathBuf,
    verbose: bool,
}

#[derive(Debug)]
enum Command {
    Info,
    Metadata,
    Query(String),
    Validate,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let args: Vec<String> = std::env::args().collect();

        if args.len() < 3 {
            return Err(format!(
                "Usage: {} <command> <file> [options]\n\n\
                Commands:\n  \
                info      - Show basic file information\n  \
                metadata  - Display all metadata key-value pairs\n  \
                query     - Query specific metadata key (use: query <file> <key>)\n  \
                validate  - Validate GGUF file format\n\n\
                Options:\n  \
                --verbose - Show detailed output",
                args[0]
            ));
        }

        let command = match args[1].as_str() {
            "info" => Command::Info,
            "metadata" => Command::Metadata,
            "query" => {
                if args.len() < 4 {
                    return Err("Query command requires a key argument".to_string());
                }
                Command::Query(args[3].clone())
            }
            "validate" => Command::Validate,
            _ => return Err(format!("Unknown command: {}", args[1])),
        };

        let file_path = PathBuf::from(&args[2]);
        let verbose = args.iter().any(|arg| arg == "--verbose" || arg == "-v");

        Ok(Args {
            command,
            file_path,
            verbose,
        })
    }
}

fn main() {
    let args = match Args::parse() {
        Ok(args) => args,
        Err(err) => {
            eprintln!("Error: {}", err);
            process::exit(1);
        }
    };

    if let Err(err) = run(args) {
        eprintln!("Error: {}", err);
        process::exit(1);
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    // Check if file exists
    if !args.file_path.exists() {
        return Err(format!("File not found: {}", args.file_path.display()).into());
    }

    // Open and parse the file
    let mut file = File::open(&args.file_path)?;
    let header = GgufHeader::parse(&mut file)?;

    // Check version compatibility
    if !header.is_version_supported() {
        println!("⚠️  Warning: GGUF version {} may not be fully supported", header.version);
    }

    match args.command {
        Command::Info => show_info(&header, &args.file_path, args.verbose)?,
        Command::Metadata => show_metadata(&mut file, &header, args.verbose)?,
        Command::Query(key) => query_metadata(&mut file, &header, &key)?,
        Command::Validate => validate_file(&header, &args.file_path)?,
    }

    Ok(())
}

fn show_info(header: &GgufHeader, file_path: &PathBuf, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("📄 GGUF File Information");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("File: {}", file_path.display());
    println!("Magic: 0x{:08X} ({})", header.magic,
        std::str::from_utf8(&header.magic.to_le_bytes()).unwrap_or("invalid"));
    println!("Version: {}", header.version);
    println!("Tensors: {}", header.n_tensors);
    println!("Metadata entries: {}", header.n_kv);

    if verbose {
        let file_size = std::fs::metadata(file_path)?.len();
        println!("File size: {} bytes ({:.2} MB)", file_size, file_size as f64 / 1_048_576.0);
        println!("Version supported: {}", header.is_version_supported());

        // Calculate approximate header size
        let header_size = 4 + 4 + 8 + 8; // magic + version + n_tensors + n_kv
        println!("Header size: {} bytes", header_size);
    }

    Ok(())
}

fn show_metadata(file: &mut File, header: &GgufHeader, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    let metadata = GgufReader::read_metadata(file, header.n_kv)?;

    // Keys to exclude from output
    let excluded_keys = [
        "tokenizer.ggml.merges",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.token_type"
    ];

    // Filter out excluded keys
    let filtered_metadata: std::collections::HashMap<_, _> = metadata
        .iter()
        .filter(|(key, _)| !excluded_keys.contains(&key.as_str()))
        .collect();

    let excluded_count = metadata.len() - filtered_metadata.len();

    println!("🔍 GGUF Metadata ({} entries", filtered_metadata.len());
    if excluded_count > 0 {
        println!(", {} tokenizer entries hidden)", excluded_count);
    } else {
        println!(")");
    }
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Sort keys for consistent output
    let mut sorted_keys: Vec<_> = filtered_metadata.keys().collect();
    sorted_keys.sort();

    for key in sorted_keys {
        let value = filtered_metadata[key];
        print!("{}: ", key);
        print_value(value, verbose);
        println!();
    }

    Ok(())
}

fn query_metadata(file: &mut File, header: &GgufHeader, query_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    let metadata = GgufReader::read_metadata(file, header.n_kv)?;

    match metadata.get(query_key) {
        Some(value) => {
            println!("🔎 Query Result for '{}'", query_key);
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            print_value(value, true);
            println!();
        }
        None => {
            println!("❌ Key '{}' not found in metadata", query_key);
            println!("\nAvailable keys:");
            let mut keys: Vec<_> = metadata.keys().collect();
            keys.sort();
            for key in keys.iter().take(10) {
                println!("  - {}", key);
            }
            if keys.len() > 10 {
                println!("  ... and {} more", keys.len() - 10);
            }
        }
    }

    Ok(())
}

fn validate_file(header: &GgufHeader, file_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("✅ GGUF File Validation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("File: {}", file_path.display());

    // Basic validation checks
    let mut issues = Vec::new();
    let mut warnings = Vec::new();

    // Check magic number
    if header.magic == GGUF_MAGIC {
        println!("✓ Magic number is valid");
    } else {
        issues.push("Invalid magic number");
    }

    // Check version
    if header.is_version_supported() {
        println!("✓ Version {} is supported", header.version);
    } else {
        warnings.push(format!("Version {} may not be fully supported", header.version));
    }

    // Check reasonable bounds
    if header.n_tensors > 0 {
        println!("✓ Contains {} tensors", header.n_tensors);
    } else {
        warnings.push("File contains no tensors".to_string());
    }

    if header.n_kv > 0 {
        println!("✓ Contains {} metadata entries", header.n_kv);
    } else {
        warnings.push("File contains no metadata".to_string());
    }

    // Check file size reasonableness
    let file_size = std::fs::metadata(file_path)?.len();
    if file_size < 32 {
        issues.push("File too small to be a valid GGUF file");
    } else {
        println!("✓ File size appears reasonable ({} bytes)", file_size);
    }

    // Print warnings
    if !warnings.is_empty() {
        println!("\n⚠️  Warnings:");
        for warning in warnings {
            println!("  - {}", warning);
        }
    }

    // Print issues
    if !issues.is_empty() {
        println!("\n❌ Issues found:");
        for issue in issues {
            println!("  - {}", issue);
        }
        return Err("File validation failed".into());
    }

    println!("\n🎉 File appears to be a valid GGUF file!");
    Ok(())
}

fn print_value(value: &Value, verbose: bool) {
    match value {
        Value::String(s) => {
            if verbose || s.len() <= 50 {
                println!("\"{}\"", s);
            } else {
                println!("\"{}...\" (truncated, {} chars)", &s[..47], s.len());
            }
        }
        Value::Bool(b) => println!("{}", b),
        Value::Uint8(n) => println!("{}", n),
        Value::Int8(n) => println!("{}", n),
        Value::Uint16(n) => println!("{}", n),
        Value::Int16(n) => println!("{}", n),
        Value::Uint32(n) => println!("{}", n),
        Value::Int32(n) => println!("{}", n),
        Value::Uint64(n) => println!("{}", n),
        Value::Int64(n) => println!("{}", n),
        Value::Float32(f) => println!("{}", f),
        Value::Float64(f) => println!("{}", f),
        Value::Array(element_type, elements) => {
            if verbose || elements.len() <= 5 {
                print!("[");
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 { print!(", "); }
                    match elem {
                        Value::String(s) => print!("\"{}\"", s),
                        Value::Float32(f) => print!("{}", f),
                        Value::Float64(f) => print!("{}", f),
                        _ => print!("{:?}", elem),
                    }
                }
                println!("] ({:?}, {} elements)", element_type, elements.len());
            } else {
                println!("[...] ({:?}, {} elements)", element_type, elements.len());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_args_parsing() {
        // Test would require mocking std::env::args,
        // which is complex in Rust. In practice, you'd use a crate like `clap`
        // for more robust argument parsing.
    }
}
