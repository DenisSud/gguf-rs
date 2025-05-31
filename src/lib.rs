//! GGUF Interface Library - Provides functionality for parsing GGUF files

pub mod metadata;
pub mod tensors;

// Re-export the main types for easier access
pub use metadata::{GgufError, GgufHeader, GgufReader, Result, TensorType, Value, ValueType, GGUF_MAGIC};
pub use tensors::{Tensor, TensorInfo, TensorLoader};  // Re-export tensors types
