//! GGUF Interface Library - Provides functionality for parsing GGUF files

pub mod config;
pub mod metadata;
pub mod model;
pub mod tensors;

// Re-export the main types for easier access
pub use config::extract_model_config;
pub use metadata::{
    GGUF_MAGIC, GgufError, GgufHeader, GgufReader, Result, TensorType, Value, ValueType,
};
pub use model::{Model, ModelBuilder, ModelConfig};
pub use tensors::{Tensor, TensorInfo, TensorLoader};
