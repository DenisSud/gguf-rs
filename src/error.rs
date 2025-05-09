use std::io;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GGUFError {
    #[error("Invalid magic number, expected 'GGUF'")]
    InvalidMagic,

    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    #[error("Invalid metadata value type: {0}")]
    InvalidValueType(u32),

    #[error("Invalid tensor type: {0}")]
    InvalidTensorType(u32),

    #[error("Alignment must be at least 8, got {0}")]
    InvalidAlignment(u32),

    #[error("Invalid UTF-8 string in GGUF file")]
    InvalidString,

    #[error("Invalid tensor name: {0}")]
    InvalidTensorName(String),

    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Metadata error: {0}")]
    Metadata(String),

    #[error("Tensor error: {0}")]
    Tensor(String),
}

pub type Result<T> = std::result::Result<T, GGUFError>;
