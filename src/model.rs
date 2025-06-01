//! Model layer organization and tensor grouping
//!
//! This module provides higher-level abstractions for organizing GGUF tensors
//! into structured model layers that can be easily used for inference.

use crate::metadata::{GgufError, Result};
use crate::tensors::Tensor;
use std::collections::HashMap;

/// Represents the embedding layer of the model
#[derive(Debug, Clone)]
pub struct EmbeddingLayer {
    /// Token embedding weights [vocab_size, embedding_dim]
    pub token_embeddings: Tensor,
}

impl EmbeddingLayer {
    /// Get the vocabulary size
    pub fn vocab_size(&self) -> u64 {
        self.token_embeddings.info.dims[1]
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> u64 {
        self.token_embeddings.info.dims[1]
    }
}

/// Represents the output/head layer of the model
#[derive(Debug, Clone)]
pub struct OutputLayer {
    /// Output projection weights [embedding_dim, vocab_size]
    pub output_weights: Tensor,
    /// Optional output normalization
    pub output_norm: Option<Tensor>,
}

impl OutputLayer {
    /// Get the vocabulary size
    pub fn vocab_size(&self) -> u64 {
        self.output_weights.info.dims[1]
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> u64 {
        self.output_weights.info.dims[0]
    }
}

/// Represents an attention layer within a transformer block
#[derive(Debug, Clone)]
pub struct AttentionLayer {
    /// Query projection weights
    pub query_weights: Tensor,
    /// Key projection weights
    pub key_weights: Tensor,
    /// Value projection weights
    pub value_weights: Tensor,
    /// Output projection weights
    pub output_weights: Tensor,

    /// Optional normalization layers
    pub query_norm: Option<Tensor>,
    pub key_norm: Option<Tensor>,
    pub attention_norm: Option<Tensor>,
}

impl AttentionLayer {
    /// Get the model dimension (embedding size)
    pub fn model_dim(&self) -> u64 {
        self.query_weights.info.dims[0]
    }

    /// Get the attention dimension (usually model_dim * num_heads)
    pub fn attention_dim(&self) -> u64 {
        self.query_weights.info.dims[1]
    }
}

/// Represents a feed-forward network layer within a transformer block
#[derive(Debug, Clone)]
pub struct FeedForwardLayer {
    /// Gate/up projection weights (for SwiGLU-style architectures)
    pub gate_weights: Option<Tensor>,
    /// Up projection weights
    pub up_weights: Tensor,
    /// Down projection weights
    pub down_weights: Tensor,
    /// Optional normalization
    pub ffn_norm: Option<Tensor>,
}

impl FeedForwardLayer {
    /// Get the model dimension (input/output size)
    pub fn model_dim(&self) -> u64 {
        self.down_weights.info.dims[0]
    }

    /// Get the intermediate dimension (hidden size)
    pub fn intermediate_dim(&self) -> u64 {
        self.up_weights.info.dims[0]
    }

    /// Check if this is a gated FFN (has gate weights)
    pub fn is_gated(&self) -> bool {
        self.gate_weights.is_some()
    }
}

/// Represents a complete transformer block/layer
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    /// Layer index (0-based)
    pub layer_index: usize,
    /// Self-attention sublayer
    pub attention: AttentionLayer,
    /// Feed-forward sublayer
    pub feed_forward: FeedForwardLayer,
}

impl TransformerBlock {
    /// Get the model dimension
    pub fn model_dim(&self) -> u64 {
        self.attention.model_dim()
    }
}

/// Complete model structure with organized layers
#[derive(Debug, Clone)]
pub struct Model {
    /// Model architecture name (e.g., "qwen3", "llama", etc.)
    pub architecture: String,
    /// Model metadata/config parameters
    pub config: ModelConfig,
    /// Token embedding layer
    pub embeddings: EmbeddingLayer,
    /// Transformer blocks/layers
    pub transformer_blocks: Vec<TransformerBlock>,
    /// Output/language modeling head
    pub output_layer: OutputLayer,
}

impl Model {
    /// Get the number of transformer layers
    pub fn num_layers(&self) -> usize {
        self.transformer_blocks.len()
    }

    /// Get a specific transformer block by index
    pub fn get_block(&self, index: usize) -> Option<&TransformerBlock> {
        self.transformer_blocks.get(index)
    }

    /// Get the model dimension
    pub fn model_dim(&self) -> u64 {
        self.config.embedding_length.into()
    }

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> u64 {
        self.embeddings.vocab_size()
    }
}

/// Model configuration extracted from GGUF metadata
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model architecture (e.g., "qwen3")
    pub architecture: String,
    /// Number of transformer blocks
    pub block_count: u32,
    /// Context length (max sequence length)
    pub context_length: u32,
    /// Embedding dimension
    pub embedding_length: u32,
    /// Feed-forward hidden dimension
    pub feed_forward_length: u32,
    /// Number of attention heads
    pub attention_head_count: u32,
    /// Number of key-value heads (for GQA)
    pub attention_head_count_kv: Option<u32>,
    /// Head dimension
    pub attention_key_length: Option<u32>,
    /// Layer norm epsilon
    pub layer_norm_epsilon: Option<f32>,
    /// RoPE frequency base
    pub rope_freq_base: Option<f32>,
}

/// Builder for constructing model from flat tensor map
pub struct ModelBuilder {
    tensors: HashMap<String, Tensor>,
    config: ModelConfig,
}

impl ModelBuilder {
    /// Create a new model builder
    pub fn new(tensors: HashMap<String, Tensor>, config: ModelConfig) -> Self {
        Self { tensors, config }
    }

    /// Build the complete model structure
    pub fn build(mut self) -> Result<Model> {
        // Build embedding layer
        let embeddings = self.build_embeddings()?;

        // Build transformer blocks
        let mut transformer_blocks = Vec::new();
        for i in 0..self.config.block_count {
            let block = self.build_transformer_block(i as usize)?;
            transformer_blocks.push(block);
        }

        // Build output layer
        let output_layer = self.build_output_layer()?;

        Ok(Model {
            architecture: self.config.architecture.clone(),
            config: self.config,
            embeddings,
            transformer_blocks,
            output_layer,
        })
    }

    fn build_embeddings(&mut self) -> Result<EmbeddingLayer> {
        let token_embeddings = self.take_tensor("token_embd.weight")?;

        Ok(EmbeddingLayer { token_embeddings })
    }

    fn build_transformer_block(&mut self, layer_idx: usize) -> Result<TransformerBlock> {
        let prefix = format!("blk.{}", layer_idx);

        // Build attention layer
        let attention = AttentionLayer {
            query_weights: self.take_tensor(&format!("{}.attn_q.weight", prefix))?,
            key_weights: self.take_tensor(&format!("{}.attn_k.weight", prefix))?,
            value_weights: self.take_tensor(&format!("{}.attn_v.weight", prefix))?,
            output_weights: self.take_tensor(&format!("{}.attn_output.weight", prefix))?,
            query_norm: self.try_take_tensor(&format!("{}.attn_q_norm.weight", prefix)),
            key_norm: self.try_take_tensor(&format!("{}.attn_k_norm.weight", prefix)),
            attention_norm: self.try_take_tensor(&format!("{}.attn_norm.weight", prefix)),
        };

        // Build feed-forward layer
        let feed_forward = FeedForwardLayer {
            gate_weights: self.try_take_tensor(&format!("{}.ffn_gate.weight", prefix)),
            up_weights: self.take_tensor(&format!("{}.ffn_up.weight", prefix))?,
            down_weights: self.take_tensor(&format!("{}.ffn_down.weight", prefix))?,
            ffn_norm: self.try_take_tensor(&format!("{}.ffn_norm.weight", prefix)),
        };

        Ok(TransformerBlock {
            layer_index: layer_idx,
            attention,
            feed_forward,
        })
    }

    fn build_output_layer(&mut self) -> Result<OutputLayer> {
        let output_weights = self.take_tensor("output.weight")?;
        let output_norm = self.try_take_tensor("output_norm.weight");

        Ok(OutputLayer {
            output_weights,
            output_norm,
        })
    }

    /// Take a required tensor from the map
    fn take_tensor(&mut self, name: &str) -> Result<Tensor> {
        self.tensors.remove(name).ok_or_else(|| {
            GgufError::InvalidFormat(format!("Required tensor '{}' not found", name))
        })
    }

    /// Try to take an optional tensor from the map
    fn try_take_tensor(&mut self, name: &str) -> Option<Tensor> {
        self.tensors.remove(name)
    }
}
