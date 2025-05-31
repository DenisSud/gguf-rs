use std::fs::File;
use std::path::PathBuf;
use std::process;

// Add TensorInfo to imports
use gguf_rs::metadata::{GgufHeader, GgufReader, Value, GGUF_MAGIC};
use gguf_rs::TensorLoader;

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
    // Add new commands
    Params,
    Tensors,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let args: Vec<String> = std::env::args().collect();

        // Update usage message with new commands
        if args.len() < 3 {
            return Err(format!(
                "Usage: {} <command> <file> [options]\n\n\
                Commands:\n  \
                info      - Show basic file information\n  \
                metadata  - Display all metadata key-value pairs\n  \
                query <key> - Query specific metadata key\n  \
                validate  - Validate GGUF file format\n  \
                params    - Calculate and show total number of parameters\n  \
                tensors   - List tensors, their labels, and shapes\n\n\
                Options:\n  \
                --verbose - Show detailed output",
                args[0]
            ));
        }

        let command = match args[1].as_str() {
            "info" => Command::Info,
            "metadata" => Command::Metadata,
            "query" => {
                // Check for the key argument
                if args.len() < 4 {
                    return Err("Query command requires a key argument (Usage: query <file> <key>)".to_string());
                }
                Command::Query(args[3].clone())
            }
            "validate" => Command::Validate,
            // Add parsing for new commands
            "params" => Command::Params,
            "tensors" => Command::Tensors,
            _ => return Err(format!("Unknown command: {}", args[1])),
        };

        // File path is always the second argument after the program name and command
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
            // Exit with a non-zero status code to indicate an error
            process::exit(1);
        }
    };

    if let Err(err) = run(args) {
        eprintln!("Error: {}", err);
        // Exit with a non-zero status code to indicate an error
        process::exit(1);
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    // Check if file exists
    if !args.file_path.exists() {
        return Err(format!("File not found: {}", args.file_path.display()).into());
    }

    // Open the file
    let mut file = File::open(&args.file_path)?;

    // Read the header - required for all commands
    let header = GgufHeader::parse(&mut file)?;

    // Check version compatibility (warn unless validation is the command)
    if !header.is_version_supported() {
        println!("⚠️  Warning: GGUF version {} may not be fully supported", header.version);
    }

    match args.command {
        Command::Info => show_info(&header, &args.file_path, args.verbose)?,
        Command::Metadata => show_metadata(&mut file, &header, args.verbose)?,
        Command::Query(key) => query_metadata(&mut file, &header, &key)?,
        Command::Validate => validate_file(&header, &args.file_path)?,
        // Add new command handlers
        Command::Params => {
            let params = calculate_params(&mut file, &header)?;
            println!("🔢 Total Parameters: {}", params);
        }
        Command::Tensors => show_tensors(&mut file, &header)?,
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
        println!("File size: {} bytes ({:.2} MB)", file_size, file_size as f64 / 1_048_1_048_576.0); // Corrected MB calculation
        println!("Version supported: {}", header.is_version_supported());

        // Calculate approximate header size
        let header_size = 4 + 4 + 8 + 8; // magic + version + n_tensors + n_kv
        println!("Header size: {} bytes", header_size);
    }

    Ok(())
}

fn show_metadata(file: &mut File, header: &GgufHeader, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    // GgufReader::read_metadata reads the metadata block and positions the file cursor afterwards
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
        println!(); // Add newline after printing value
    }

    Ok(())
}

fn query_metadata(file: &mut File, header: &GgufHeader, query_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    // GgufReader::read_metadata reads the metadata block and positions the file cursor afterwards
    let metadata = GgufReader::read_metadata(file, header.n_kv)?;

    match metadata.get(query_key) {
        Some(value) => {
            println!("🔎 Query Result for '{}'", query_key);
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            print_value(value, true); // Always print full value for a query
            println!(); // Add newline after printing value
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
            // The original code didn't return an error here, just printed a message.
            // Keeping it consistent.
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
        // This is typically a warning for a model file
        warnings.push("File contains no tensors".to_string());
    }

    if header.n_kv > 0 {
        println!("✓ Contains {} metadata entries", header.n_kv);
    } else {
        // This is typically a warning for a model file
        warnings.push("File contains no metadata".to_string());
    }

    // Check file size reasonableness
    let file_size = std::fs::metadata(file_path)?.len();
    // Minimum size is header (32 bytes) + minimal metadata. 32 is too small.
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
        // Return an error Box<dyn std::error::Error>
        return Err("File validation failed".into());
    }

    println!("\n🎉 File appears to be a valid GGUF file!");
    Ok(())
}

// print_value utility function kept as-is from original
fn print_value(value: &Value, verbose: bool) {
    match value {
        Value::String(s) => {
            if verbose || s.len() <= 50 {
                print!("\"{}\"", s);
            } else {
                print!("\"{}...\" (truncated, {} chars)", &s[..47], s.len());
            }
        }
        Value::Bool(b) => print!("{}", b),
        Value::Uint8(n) => print!("{}", n),
        Value::Int8(n) => print!("{}", n),
        Value::Uint16(n) => print!("{}", n),
        Value::Int16(n) => print!("{}", n),
        Value::Uint32(n) => print!("{}", n),
        Value::Int32(n) => print!("{}", n),
        Value::Uint64(n) => print!("{}", n),
        Value::Int64(n) => print!("{}", n),
        Value::Float32(f) => print!("{}", f),
        Value::Float64(f) => print!("{}", f),
        Value::Array(element_type, elements) => {
            if verbose || elements.len() <= 5 {
                print!("[");
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 { print!(", "); }
                    // Original code printed specific types nicely, others with debug
                     match elem {
                        Value::String(s) => print!("\"{}\"", s),
                        Value::Float32(f) => print!("{}", f),
                        Value::Float64(f) => print!("{}", f),
                        _ => print!("{:?}", elem), // Use debug for other value types in arrays
                    }
                }
                print!("] ({:?}, {} elements)", element_type, elements.len());
            } else {
                print!("[...] ({:?}, {} elements)", element_type, elements.len());
            }
        }
    }
}

// --- New functions for the prompt ---

/// Calculates the total number of parameters across all tensors.
/// Reads tensor information from the file to get shapes and computes the total.
fn calculate_params(file: &mut File, header: &GgufHeader) -> Result<u64, Box<dyn std::error::Error>> {
    let _metadata = GgufReader::read_metadata(file, header.n_kv)?;

    // Change to use TensorLoader instead of GgufReader
    let tensor_infos = TensorLoader::read_tensor_info(file, header.n_tensors)?;  // <-- FIXED HERE

    let mut total_params: u64 = 0;
    for info in tensor_infos {
        let tensor_params = info.dims.iter().product::<u64>();
        total_params += tensor_params;
    }
    Ok(total_params)
}
/// Lists all tensors, their labels, and shapes.
/// Reads tensor information from the file and prints it.
fn show_tensors(file: &mut File, header: &GgufHeader) -> Result<(), Box<dyn std::error::Error>> {
    println!("📜 Tensors ({} entries)", header.n_tensors);
    println!("━━━━━━━━━━━━━━━━━━━━");

    if header.n_tensors == 0 {
        println!("No tensors found in this file.");
        return Ok(());
    }

    // To read tensor information, the file cursor must be positioned after the header and metadata.
    // GgufHeader::parse left the cursor after the header.
    // GgufReader::read_metadata reads the metadata and positions the cursor after the metadata.
    let _metadata = GgufReader::read_metadata(file, header.n_kv)?;

    // GgufReader::read_tensor_infos reads the tensor information block and positions the cursor after it.
    let tensor_infos = TensorLoader::read_tensor_info(file, header.n_tensors)?;

    // Find max name length for alignment (optional, but nice)
    let max_name_len = tensor_infos.iter().map(|info| info.name.len()).max().unwrap_or(0);

    for (i, info) in tensor_infos.iter().enumerate() {
        // Format dims nicely, e.g., "[dim1, dim2, dim3]" or "dim1 x dim2 x dim3"
        let dimensions_str = info.dims.iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x"); // Join dims with 'x'

        println!("{:>4}: {:<name_width$} [{}]",
                 i,
                 info.name,
                 dimensions_str,
                 name_width = max_name_len // Use exact max_name_len for alignment
        );
    }

    Ok(())
}
