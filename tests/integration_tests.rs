use std::fs::File;
use std::path::PathBuf;
use gguf_rs::metadata::GgufHeader;
use gguf_rs::tensors::TensorLoader; // Assuming TensorLoader is public

#[test]
fn test_load_tensors_from_real_file() {
    // NOTE: You must place your GGUF file in `gguf-interface/tests/data/`
    let file_path = PathBuf::from("tests/data/Qwen3-0.6B-F16.gguf");

    if !file_path.exists() {
        // Skip the test if the file isn't present, or panic - your choice
        eprintln!("Skipping test_load_tensors_from_real_file: File not found at {}", file_path.display());
        return;
    }

    let mut file = File::open(&file_path).expect("Failed to open test file");

    // Parse header
    let header = GgufHeader::parse(&mut file).expect("Failed to parse header");
    println!("Successfully parsed header: {:?}", header);

    // Read metadata (optional, but good to skip past it)
    // You might want to verify some metadata here too
    let metadata = gguf_rs::metadata::GgufReader::read_metadata(&mut file, header.n_kv)
        .expect("Failed to read metadata");
    println!("Read {} metadata entries", metadata.len());

    // Read tensor info
    let tensor_infos = TensorLoader::read_tensor_info(&mut file, header.n_tensors)
        .expect("Failed to read tensor info");
    println!("Read {} tensor infos", tensor_infos.len());
    assert_eq!(tensor_infos.len() as u64, header.n_tensors);

    // Get tensor data start position
    let tensor_data_start = TensorLoader::get_tensor_data_start(&mut file)
        .expect("Failed to get tensor data start");
    println!("Tensor data starts at byte offset: {}", tensor_data_start);

    // Load all supported tensors
    let loaded_tensors = TensorLoader::load_all_tensors(&mut file, &tensor_infos, tensor_data_start)
        .expect("Failed to load all tensors");
    println!("Loaded {} supported tensors", loaded_tensors.len());

    // Add assertions based on what you expect from your Qwen3-0.6B-F16.gguf file
    // For example, check if specific tensors exist and have expected shapes/types:
    // assert!(loaded_tensors.contains_key("model.embed_tokens.weight"));
    // let embed_tensor = loaded_tensors.get("model.embed_tokens.weight").unwrap();
    // assert_eq!(embed_tensor.info.tensor_type, gguf_rs::metadata::TensorType::F16); // Or whatever type it is
    // assert_eq!(embed_tensor.info.dims, vec![32000, 4096]); // Or whatever dimensions it has

    // You could also sample some data points and check their values if you know them
    // let embed_data_f32 = embed_tensor.as_f32_vec().expect("Failed to convert embed tensor to f32");
    // println!("First 10 elements of embed_tokens.weight: {:?}", &embed_data_f32[0..10]);

    println!("Successfully processed tensors from real file.");
}
