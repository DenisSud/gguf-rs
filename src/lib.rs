use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use thiserror::Error;

/// Error types for GGML operations
#[derive(Error, Debug)]
pub enum GgmlError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    
    #[error("Invalid GGUF magic value")]
    InvalidMagic,
    
    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),
    
    #[error("Invalid tensor name: {0}")]
    InvalidTensorName(String),
    
    #[error("Tensor not found: {0}")]
    TensorNotFound(String),
    
    #[error("Invalid tensor metadata")]
    InvalidTensorMetadata,
    
    #[error("Invalid model configuration")]
    InvalidModelConfig,
    
    #[error("Unsupported tensor type: {0}")]
    UnsupportedTensorType(u32),
    
    #[error("Unsupported quantization: {0}")]
    UnsupportedQuantization(u32),
}

/// Tensor data types supported in GGUF format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q8_0,
    // Add other types as needed
}

impl TensorType {
    fn from_u32(value: u32) -> Result<Self, GgmlError> {
        match value {
            0 => Ok(TensorType::F32),
            1 => Ok(TensorType::F16),
            2 => Ok(TensorType::Q4_0),
            3 => Ok(TensorType::Q4_1),
            4 => Ok(TensorType::Q8_0),
            _ => {
                eprintln!("Unsupported tensor type value: {}", value);
                Err(GgmlError::UnsupportedTensorType(value))
            },
        }
    }
    
    /// Get the size in bytes for a single element of this type
    pub fn element_size(&self) -> usize {
        match self {
            TensorType::F32 => 4,
            TensorType::F16 => 2,
            TensorType::Q4_0 => 4, // 4 bits per value, but packed
            TensorType::Q4_1 => 4, // 4 bits per value, but packed
            TensorType::Q8_0 => 8, // 8 bits per value, but packed
        }
    }
}

/// Quantization information for tensors
#[derive(Debug, Clone)]
pub struct QuantizationInfo {
    pub scale: Option<Vec<f32>>,
    pub offset: Option<Vec<f32>>,
}

/// Lightweight tensor metadata - no data loaded
#[derive(Debug, Clone)]
pub struct TensorRef {
    pub name: String,
    pub shape: Vec<u32>,
    pub dtype: TensorType,
    pub quantization: QuantizationInfo,
    // Internal file offset and size
    offset: u64,
    size: usize,
}

/// GGUF file format constants
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little endian
const GGUF_VERSION_V2: u32 = 2;
const GGUF_VERSION_V3: u32 = 3;

/// Main file handle - memory-mapped, metadata parsed
pub struct GgmlFile {
    path: PathBuf,
    mmap: Mmap,
    tensors: Vec<TensorRef>,
    tensor_map: std::collections::HashMap<String, usize>,
}

impl GgmlFile {
    /// Open a GGUF file and parse its metadata
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgmlError> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        
        // Memory map the file for zero-copy access
        let mmap = unsafe { Mmap::map(&file)? };
        
        // Parse file header and tensor metadata
        let mut tensors = Vec::new();
        let mut tensor_map = std::collections::HashMap::new();
        
        // Parse header and tensors
        let mut cursor = io::Cursor::new(&mmap[..]);
        
        // Check magic number
        let magic = cursor.read_u32::<LittleEndian>()?;
        eprintln!("GGUF Magic: 0x{:08x}", magic);
        if magic != GGUF_MAGIC {
            eprintln!("Invalid magic number: 0x{:08x}, expected: 0x{:08x}", magic, GGUF_MAGIC);
            return Err(GgmlError::InvalidMagic);
        }
        
        // Check version
        let version = cursor.read_u32::<LittleEndian>()?;
        eprintln!("GGUF Version: {}", version);
        if version != GGUF_VERSION_V2 && version != GGUF_VERSION_V3 {
            eprintln!("Unsupported version: {}", version);
            return Err(GgmlError::UnsupportedVersion(version));
        }
        
        // For debugging, dump the first 128 bytes of the file
        let current_pos = cursor.position();
        cursor.seek(SeekFrom::Start(0))?;
        let mut header_bytes = vec![0u8; 128.min(mmap.len())];
        cursor.read_exact(&mut header_bytes)?;
        eprintln!("First 128 bytes of file (hex): {:02x?}", header_bytes);
        cursor.seek(SeekFrom::Start(current_pos))?;
        
        // Parse metadata count
        let metadata_count = cursor.read_u64::<LittleEndian>()?;
        eprintln!("Metadata count: {}", metadata_count);
        
        // Parse metadata - completely rewritten for GGUF v3 compatibility based on spec
        // In a full implementation, we would store all metadata here
        let mut alignment: u32 = 32; // Default alignment
        
        // Define GGUF metadata value types based on the spec
        const GGUF_METADATA_VALUE_TYPE_UINT8: u32 = 0;
        const GGUF_METADATA_VALUE_TYPE_INT8: u32 = 1;
        const GGUF_METADATA_VALUE_TYPE_UINT16: u32 = 2;
        const GGUF_METADATA_VALUE_TYPE_INT16: u32 = 3;
        const GGUF_METADATA_VALUE_TYPE_UINT32: u32 = 4;
        const GGUF_METADATA_VALUE_TYPE_INT32: u32 = 5;
        const GGUF_METADATA_VALUE_TYPE_FLOAT32: u32 = 6;
        const GGUF_METADATA_VALUE_TYPE_BOOL: u32 = 7;
        const GGUF_METADATA_VALUE_TYPE_STRING: u32 = 8;
        const GGUF_METADATA_VALUE_TYPE_ARRAY: u32 = 9;
        const GGUF_METADATA_VALUE_TYPE_UINT64: u32 = 10;
        const GGUF_METADATA_VALUE_TYPE_INT64: u32 = 11;
        const GGUF_METADATA_VALUE_TYPE_FLOAT64: u32 = 12;
        
        for i in 0..metadata_count {
            let current_pos = cursor.position();
            eprintln!("Parsing metadata {} at position {}", i, current_pos);
            
            // Read key length and key with bounds checking
            let key_len = match cursor.read_u64::<LittleEndian>() {
                Ok(len) => {
                    eprintln!("  Key length: {}", len);
                    // Sanity check for key length - no reasonable key should be larger than 1024 bytes
                    if len > 1024 {
                        eprintln!("  ERROR: Key length {} is unreasonably large, likely a parsing error", len);
                        eprintln!("  Current cursor position: {}", cursor.position());
                        // Dump some raw bytes around the current position for debugging
                        let pos = cursor.position();
                        if pos >= 8 {
                            cursor.seek(SeekFrom::Start(pos - 8))?;
                            let mut debug_bytes = [0u8; 16];
                            cursor.read_exact(&mut debug_bytes)?;
                            eprintln!("  Raw bytes around position (hex): {:02x?}", debug_bytes);
                            cursor.seek(SeekFrom::Start(pos))?;
                        }
                        return Err(GgmlError::InvalidModelConfig);
                    }
                    len
                },
                Err(e) => {
                    eprintln!("  Error reading key length: {:?}", e);
                    return Err(GgmlError::Io(e));
                }
            };
            
            // Read key for alignment check
            let mut key_bytes = vec![0u8; key_len as usize];
            match cursor.read_exact(&mut key_bytes) {
                Ok(_) => {},
                Err(e) => {
                    eprintln!("  Error reading key bytes: {:?}", e);
                    return Err(GgmlError::Io(e));
                }
            };
            
            let key = String::from_utf8_lossy(&key_bytes);
            eprintln!("  Metadata key: {}", key);
            
            // Check if this is the alignment metadata
            let is_alignment = key == "general.alignment";
            
            // Read value type according to GGUF v3 spec
            let value_type = match cursor.read_u32::<LittleEndian>() {
                Ok(vt) => {
                    eprintln!("  Value type: {}", vt);
                    vt
                },
                Err(e) => {
                    eprintln!("  Error reading value type: {:?}", e);
                    return Err(GgmlError::Io(e));
                }
            };
            
            // Handle different value types according to the GGUF v3 spec
            match value_type {
                GGUF_METADATA_VALUE_TYPE_UINT8 => {
                    eprintln!("  Value is UINT8");
                    match cursor.read_u8() {
                        Ok(val) => {
                            eprintln!("  UINT8 value: {}", val);
                        },
                        Err(e) => {
                            eprintln!("  Error reading UINT8: {:?}", e);
                            return Err(GgmlError::Io(e));
                        }
                    }
                },
                GGUF_METADATA_VALUE_TYPE_INT8 => {
                    eprintln!("  Value is INT8");
                    match cursor.read_i8() {
                        Ok(val) => {
                            eprintln!("  INT8 value: {}", val);
                        },
                        Err(e) => {
                            eprintln!("  Error reading INT8: {:?}", e);
                            return Err(GgmlError::Io(e));
                        }
                    }
                },
                GGUF_METADATA_VALUE_TYPE_UINT16 => {
                    eprintln!("  Value is UINT16");
                    match cursor.read_u16::<LittleEndian>() {
                        Ok(val) => {
                            eprintln!("  UINT16 value: {}", val);
                        },
                        Err(e) => {
                            eprintln!("  Error reading UINT16: {:?}", e);
                            return Err(GgmlError::Io(e));
                        }
                    }
                },
                GGUF_METADATA_VALUE_TYPE_INT16 => {
                    eprintln!("  Value is INT16");
                    match cursor.read_i16::<LittleEndian>() {
                        Ok(val) => {
                            eprintln!("  INT16 value: {}", val);
                        },
                        Err(e) => {
                            eprintln!("  Error reading INT16: {:?}", e);
                            return Err(GgmlError::Io(e));
                        }
                    }
                },
                GGUF_METADATA_VALUE_TYPE_UINT32 => {
                    eprintln!("  Value is UINT32");
                    match cursor.read_u32::<LittleEndian>() {
                        Ok(val) => {
                            eprintln!("  UINT32 value: {}", val);
                            if is_alignment {
                                alignment = val;
                                eprintln!("  Updated alignment to: {}", alignment);
                            }
                        },
                        Err(e) => {
                            eprintln!("  Error reading UINT32: {:?}", e);
                            return Err(GgmlError::Io(e));
                        }
                    }
                },
                GGUF_METADATA_VALUE_TYPE_INT32 => {
                    eprintln!("  Value is INT32");
                    match cursor.read_i32::<LittleEndian>() {
                        Ok(val) => {
                            eprintln!("  INT32 value: {}", val);
                        },
                        Err(e) => {
                            eprintln!("  Error reading INT32: {:?}", e);
                            return Err(GgmlError::Io(e));
                        }
                    }
                },
                GGUF_METADATA_VALUE_TYPE_FLOAT32 => {
                    eprintln!("  Value is FLOAT32");
                    match cursor.read_f32::<LittleEndian>() {
                        Ok(val) => {
                            eprintln!("  FLOAT32 value: {}", val);
                        },
                        Err(e) => {
                            eprintln!("  Error reading FLOAT32: {:?}", e);
                            return Err(GgmlError::Io(e));
                        }
                    }
                },
                GGUF_METADATA_VALUE_TYPE_BOOL => {
                    eprintln!("  Value is BOOL");
                    match cursor.read_u8() {
                        Ok(val) => {
                            let bool_val = val != 0;
                            eprintln!("  BOOL value: {}", bool_val);
                        },
                        Err(e) => {
                            eprintln!("  Error reading BOOL: {:?}", e);
                            return Err(GgmlError::Io(e));
                        }
                    }
                },
                GGUF_METADATA_VALUE_TYPE_STRING => {
                    eprintln!("  Value is STRING");
                    // Read string length
                    let str_len = match cursor.read_u64::<LittleEndian>() {
                        Ok(len) => {
                            eprintln!("  String length: {}", len);
                            // Sanity check for string length
                            if len > 1024 * 1024 { // 1MB max string length
                                eprintln!("  ERROR: String length {} is unreasonably large", len);
                                return Err(GgmlError::InvalidModelConfig);
                            }
                            len
                        },
                        Err(e) => {
                            eprintln!("  Error reading string length: {:?}", e);
                            return Err(GgmlError::Io(e));
                        }
                    };
                    
                    // Read string value
                    let mut str_bytes = vec![0u8; str_len as usize];
                    match cursor.read_exact(&mut str_bytes) {
                        Ok(_) => {},
                        Err(e) => {
                            eprintln!("  Error reading string value: {:?}", e);
                            return Err(GgmlError::Io(e));
                        }
                    };
                    
                    let str_value = String::from_utf8_lossy(&str_bytes);
                    eprintln!("  String value: {}", str_value);
                    
                    // If this is the alignment metadata, update the alignment value
                    if is_alignment && str_value.parse::<u32>().is_ok() {
                        alignment = str_value.parse::<u32>().unwrap();
                        eprintln!("  Updated alignment to: {}", alignment);
                    }
                },
                GGUF_METADATA_VALUE_TYPE_ARRAY => {
                    eprintln!("  Value is ARRAY");
                    // Read array type
                    let array_type = match cursor.read_u32::<LittleEndian>() {
                        Ok(at) => {
                            eprintln!("  Array type: {}", at);
                            at
                        },
                        Err(e) => {
                            eprintln!("  Error reading array type: {:?}", e);
                            return Err(GgmlError::Io(e));
                        }
                    };
                    
                    // Read array length
                    let array_len = match cursor.read_u64::<LittleEndian>() {
                        Ok(len) => {
                            eprintln!("  Array length: {}", len);
                            len
                        },
                        Err(e) => {
                            eprintln!("  Error reading array length: {:?}", e);
                            return Err(GgmlError::Io(e));
                        }
                    };
                    
                    // Skip array elements based on type
                    match array_type {
                        GGUF_METADATA_VALUE_TYPE_UINT8 | GGUF_METADATA_VALUE_TYPE_INT8 | GGUF_METADATA_VALUE_TYPE_BOOL => {
                            eprintln!("  Array elements are 1-byte values");
                            match cursor.seek(SeekFrom::Current(array_len as i64)) {
                                Ok(_) => {},
                                Err(e) => {
                                    eprintln!("  Error seeking past array: {:?}", e);
                                    return Err(GgmlError::Io(e));
                                }
                            }
                        },
                        GGUF_METADATA_VALUE_TYPE_UINT16 | GGUF_METADATA_VALUE_TYPE_INT16 => {
                            eprintln!("  Array elements are 2-byte values");
                            match cursor.seek(SeekFrom::Current((array_len * 2) as i64)) {
                                Ok(_) => {},
                                Err(e) => {
                                    eprintln!("  Error seeking past array: {:?}", e);
                                    return Err(GgmlError::Io(e));
                                }
                            }
                        },
                        GGUF_METADATA_VALUE_TYPE_UINT32 | GGUF_METADATA_VALUE_TYPE_INT32 | GGUF_METADATA_VALUE_TYPE_FLOAT32 => {
                            eprintln!("  Array elements are 4-byte values");
                            match cursor.seek(SeekFrom::Current((array_len * 4) as i64)) {
                                Ok(_) => {},
                                Err(e) => {
                                    eprintln!("  Error seeking past array: {:?}", e);
                                    return Err(GgmlError::Io(e));
                                }
                            }
                        },
                        GGUF_METADATA_VALUE_TYPE_UINT64 | GGUF_METADATA_VALUE_TYPE_INT64 | GGUF_METADATA_VALUE_TYPE_FLOAT64 => {
                            eprintln!("  Array elements are 8-byte values");
                            match cursor.seek(SeekFrom::Current((array_len * 8) as i64)) {
                                Ok(_) => {},
                                Err(e) => {
                                    eprintln!("  Error seeking past array: {:?}", e);
                                    return Err(GgmlError::Io(e));
                                }
                            }
                        },
                        GGUF_METADATA_VALUE_TYPE_STRING => {
                            eprintln!("  Array elements are strings");
                            // For string arrays, we need to skip each string
                            for j in 0..array_len {
                                eprintln!("  Processing string {} in array", j);
                                let str_len = match cursor.read_u64::<LittleEndian>() {
                                    Ok(len) => {
                                        eprintln!("    String length: {}", len);
                                        len
                                    },
                                    Err(e) => {
                                        eprintln!("    Error reading string length: {:?}", e);
                                        return Err(GgmlError::Io(e));
                                    }
                                };
                                
                                match cursor.seek(SeekFrom::Current(str_len as i64)) {
                                    Ok(_) => {},
                                    Err(e) => {
                                        eprintln!("    Error seeking past string: {:?}", e);
                                        return Err(GgmlError::Io(e));
                                    }
                                }
                            }
                        },
                        _ => {
                            eprintln!("  Unsupported array type: {}", array_type);
                            return Err(GgmlError::InvalidModelConfig);
                        }
                    }
                },
                GGUF_METADATA_VALUE_TYPE_UINT64 => {
                    eprintln!("  Value is UINT64");
                    match cursor.read_u64::<LittleEndian>() {
                        Ok(val) => {
                            eprintln!("  UINT64 value: {}", val);
                        },
                        Err(e) => {
                            eprintln!("  Error reading UINT64: {:?}", e);
                            return Err(GgmlError::Io(e));
                        }
                    }
                },
                GGUF_METADATA_VALUE_TYPE_INT64 => {
                    eprintln!("  Value is INT64");
                    match cursor.read_i64::<LittleEndian>() {
                        Ok(val) => {
                            eprintln!("  INT64 value: {}", val);
                        },
                        Err(e) => {
                            eprintln!("  Error reading INT64: {:?}", e);
                            return Err(GgmlError::Io(e));
                        }
                    }
                },
                GGUF_METADATA_VALUE_TYPE_FLOAT64 => {
                    eprintln!("  Value is FLOAT64");
                    match cursor.read_f64::<LittleEndian>() {
                        Ok(val) => {
                            eprintln!("  FLOAT64 value: {}", val);
                        },
                        Err(e) => {
                            eprintln!("  Error reading FLOAT64: {:?}", e);
                            return Err(GgmlError::Io(e));
                        }
                    }
                },
                _ => {
                    eprintln!("  Unsupported value type: {}", value_type);
                    return Err(GgmlError::InvalidModelConfig);
                }
            }
            
            eprintln!("  Finished metadata {} at position {}", i, cursor.position());
        }
        
        // Parse tensor count
        let tensor_count = cursor.read_u64::<LittleEndian>()?;
        eprintln!("Found {} tensors in the model", tensor_count);
        
        // Parse tensor metadata
        for i in 0..tensor_count {
            // Get current position for debugging
            let current_pos = cursor.position();
            eprintln!("Parsing tensor {} at position {}", i, current_pos);
            
            // Parse tensor name
            let name_len = cursor.read_u64::<LittleEndian>()?;
            eprintln!("  Name length: {}", name_len);
            
            let mut name_bytes = vec![0u8; name_len as usize];
            cursor.read_exact(&mut name_bytes)?;
            let name = String::from_utf8_lossy(&name_bytes).to_string();
            eprintln!("  Tensor name: {}", name);
            
            // Parse tensor dimensions
            let n_dims = cursor.read_u32::<LittleEndian>()?;
            eprintln!("  Number of dimensions: {}", n_dims);
            
            let mut shape = Vec::with_capacity(n_dims as usize);
            for dim_idx in 0..n_dims {
                let dim_size = cursor.read_u64::<LittleEndian>()? as u32;
                eprintln!("    Dimension {}: {}", dim_idx, dim_size);
                shape.push(dim_size);
            }
            
            // Parse tensor type
            let dtype_int = cursor.read_u32::<LittleEndian>()?;
            eprintln!("  Tensor type value: {}", dtype_int);
            
            let dtype = match TensorType::from_u32(dtype_int) {
                Ok(dt) => dt,
                Err(e) => {
                    eprintln!("Error parsing tensor type for tensor '{}': {:?}", name, e);
                    return Err(e);
                }
            };
            
            // Read offset
            let offset = cursor.read_u64::<LittleEndian>()?;
            eprintln!("  Tensor offset: {}", offset);
            
            // Calculate tensor size based on shape and dtype
            let mut size: usize = 1;
            for dim in &shape {
                size *= *dim as usize;
            }
            size = (size * dtype.element_size() + alignment as usize - 1) / alignment as usize * alignment as usize; // Align to alignment bytes
            
            // Create tensor reference
            let tensor = TensorRef {
                name,
                shape,
                dtype,
                quantization: QuantizationInfo {
                    scale: None,
                    offset: None,
                },
                offset,
                size,
            };
            
            tensor_map.insert(tensor.name.clone(), i as usize);
            tensors.push(tensor);
        }
        
        Ok(GgmlFile {
            path,
            mmap,
            tensors,
            tensor_map,
        })
    }
    
    /// Get all tensors in the file
    pub fn tensors(&self) -> impl Iterator<Item = &TensorRef> {
        self.tensors.iter()
    }
    
    /// Find a specific tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<&TensorRef> {
        self.tensor_map.get(name).map(|&idx| &self.tensors[idx])
    }
    
    /// Get raw tensor data bytes (zero-copy)
    pub fn tensor_data(&self, tensor: &TensorRef) -> &[u8] {
        &self.mmap[tensor.offset as usize..tensor.offset as usize + tensor.size]
    }
}

/// Qwen3 model configuration
#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_attention_heads: u32,
    pub num_hidden_layers: u32,
    pub rms_norm_eps: f32,
    pub vocab_size: u32,
}

/// Attention weights for a Qwen3 layer
#[derive(Debug)]
pub struct AttentionWeights<'a> {
    pub q_proj: &'a TensorRef,
    pub k_proj: &'a TensorRef,
    pub v_proj: &'a TensorRef,
    pub o_proj: &'a TensorRef,
}

/// MLP weights for a Qwen3 layer
#[derive(Debug)]
pub struct MlpWeights<'a> {
    pub gate_proj: &'a TensorRef,
    pub up_proj: &'a TensorRef,
    pub down_proj: &'a TensorRef,
}

/// A single layer in the Qwen3 model
#[derive(Debug)]
pub struct Qwen3Layer<'a> {
    pub attn: AttentionWeights<'a>,
    pub mlp: MlpWeights<'a>,
    pub input_layernorm: &'a TensorRef,
    pub post_attention_layernorm: &'a TensorRef,
}

/// Complete Qwen3 model structure with tensor references
#[derive(Debug)]
pub struct Qwen3Model<'a> {
    pub config: Qwen3Config,
    pub embed_tokens: &'a TensorRef,
    pub layers: Vec<Qwen3Layer<'a>>,
    pub norm: &'a TensorRef,
    pub lm_head: &'a TensorRef,
}

impl<'a> Qwen3Model<'a> {
    /// Parse Qwen3-specific model structure from GGML file
    pub fn from_ggml(file: &'a GgmlFile) -> Result<Qwen3Model<'a>, GgmlError> {
        // Extract model configuration from tensors
        // For a real implementation, we would parse this from the metadata
        // Here we're using some reasonable defaults for Qwen3 models
        let config = Qwen3Config {
            hidden_size: 4096,
            intermediate_size: 11008,
            num_attention_heads: 32,
            num_hidden_layers: 32,
            rms_norm_eps: 1e-6,
            vocab_size: 151936,
        };
        
        // Find embedding tensor
        let embed_tokens = file.get_tensor("model.embed_tokens.weight")
            .ok_or_else(|| GgmlError::TensorNotFound("model.embed_tokens.weight".to_string()))?;
        
        // Find norm tensor
        let norm = file.get_tensor("model.norm.weight")
            .ok_or_else(|| GgmlError::TensorNotFound("model.norm.weight".to_string()))?;
        
        // Find lm_head tensor
        let lm_head = file.get_tensor("lm_head.weight")
            .ok_or_else(|| GgmlError::TensorNotFound("lm_head.weight".to_string()))?;
        
        // Parse layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        
        for i in 0..config.num_hidden_layers {
            let layer_prefix = format!("model.layers.{}", i);
            
            // Get attention weights
            let q_proj = file.get_tensor(&format!("{}.self_attn.q_proj.weight", layer_prefix))
                .ok_or_else(|| GgmlError::TensorNotFound(format!("{}.self_attn.q_proj.weight", layer_prefix)))?;
            
            let k_proj = file.get_tensor(&format!("{}.self_attn.k_proj.weight", layer_prefix))
                .ok_or_else(|| GgmlError::TensorNotFound(format!("{}.self_attn.k_proj.weight", layer_prefix)))?;
            
            let v_proj = file.get_tensor(&format!("{}.self_attn.v_proj.weight", layer_prefix))
                .ok_or_else(|| GgmlError::TensorNotFound(format!("{}.self_attn.v_proj.weight", layer_prefix)))?;
            
            let o_proj = file.get_tensor(&format!("{}.self_attn.o_proj.weight", layer_prefix))
                .ok_or_else(|| GgmlError::TensorNotFound(format!("{}.self_attn.o_proj.weight", layer_prefix)))?;
            
            // Get MLP weights
            let gate_proj = file.get_tensor(&format!("{}.mlp.gate_proj.weight", layer_prefix))
                .ok_or_else(|| GgmlError::TensorNotFound(format!("{}.mlp.gate_proj.weight", layer_prefix)))?;
            
            let up_proj = file.get_tensor(&format!("{}.mlp.up_proj.weight", layer_prefix))
                .ok_or_else(|| GgmlError::TensorNotFound(format!("{}.mlp.up_proj.weight", layer_prefix)))?;
            
            let down_proj = file.get_tensor(&format!("{}.mlp.down_proj.weight", layer_prefix))
                .ok_or_else(|| GgmlError::TensorNotFound(format!("{}.mlp.down_proj.weight", layer_prefix)))?;
            
            // Get layer norms
            let input_layernorm = file.get_tensor(&format!("{}.input_layernorm.weight", layer_prefix))
                .ok_or_else(|| GgmlError::TensorNotFound(format!("{}.input_layernorm.weight", layer_prefix)))?;
            
            let post_attention_layernorm = file.get_tensor(&format!("{}.post_attention_layernorm.weight", layer_prefix))
                .ok_or_else(|| GgmlError::TensorNotFound(format!("{}.post_attention_layernorm.weight", layer_prefix)))?;
            
            // Create layer
            layers.push(Qwen3Layer {
                attn: AttentionWeights {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                },
                mlp: MlpWeights {
                    gate_proj,
                    up_proj,
                    down_proj,
                },
                input_layernorm,
                post_attention_layernorm,
            });
        }
        
        Ok(Qwen3Model {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }
}
