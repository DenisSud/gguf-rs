//! GGUF Interface Library - Provides functionality for parsing GGUF files

pub mod metadata;

// Re-export the main types for easier access
pub use metadata::{
    GgufError, GgufHeader, GgufReader, Result, TensorType, Value, ValueType, GGUF_MAGIC,
};
