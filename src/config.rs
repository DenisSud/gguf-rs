//! Configuration extraction from GGUF metadata
//!
//! This module provides utilities to extract model configuration
//! from GGUF metadata key-value pairs.

use crate::metadata::{GgufError, Result, Value};
use crate::model::ModelConfig;
use std::collections::HashMap;

/// Extract model configuration from GGUF metadata
pub fn extract_model_config(metadata: &HashMap<String, Value>) -> Result<ModelConfig> {
    // Extract architecture
    let architecture = metadata
        .get("general.architecture")
        .and_then(|v| v.as_string())
        .ok_or_else(|| GgufError::InvalidFormat("Missing general.architecture".to_string()))?
        .to_string();

    // Create architecture-specific prefix
    let arch_prefix = &architecture;

    // Extract required fields
    let block_count = get_u32_field(metadata, &format!("{}.block_count", arch_prefix))?;
    let context_length = get_u32_field(metadata, &format!("{}.context_length", arch_prefix))?;
    let embedding_length = get_u32_field(metadata, &format!("{}.embedding_length", arch_prefix))?;
    let feed_forward_length =
        get_u32_field(metadata, &format!("{}.feed_forward_length", arch_prefix))?;
    let attention_head_count =
        get_u32_field(metadata, &format!("{}.attention.head_count", arch_prefix))?;

    // Extract optional fields
    let attention_head_count_kv = get_optional_u32_field(
        metadata,
        &format!("{}.attention.head_count_kv", arch_prefix),
    );
    let attention_key_length =
        get_optional_u32_field(metadata, &format!("{}.attention.key_length", arch_prefix));
    let layer_norm_epsilon = get_optional_f32_field(
        metadata,
        &format!("{}.attention.layer_norm_rms_epsilon", arch_prefix),
    );
    let rope_freq_base =
        get_optional_f32_field(metadata, &format!("{}.rope.freq_base", arch_prefix));

    Ok(ModelConfig {
        architecture,
        block_count,
        context_length,
        embedding_length,
        feed_forward_length,
        attention_head_count,
        attention_head_count_kv,
        attention_key_length,
        layer_norm_epsilon,
        rope_freq_base,
    })
}

fn get_u32_field(metadata: &HashMap<String, Value>, key: &str) -> Result<u32> {
    metadata
        .get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .ok_or_else(|| GgufError::InvalidFormat(format!("Missing or invalid field: {}", key)))
}

fn get_optional_u32_field(metadata: &HashMap<String, Value>, key: &str) -> Option<u32> {
    metadata.get(key).and_then(|v| v.as_u64()).map(|v| v as u32)
}

fn get_optional_f32_field(metadata: &HashMap<String, Value>, key: &str) -> Option<f32> {
    metadata.get(key).and_then(|v| v.as_f64()).map(|v| v as f32)
}
