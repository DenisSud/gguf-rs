use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::error::{Result, GGUFError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[repr(u32)]
pub enum MetadataValueType {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for MetadataValueType {
    type Error = GGUFError;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::UInt8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::UInt16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::UInt32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::UInt64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(GGUFError::InvalidValueType(value)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    pub fn type_(&self) -> MetadataValueType {
        match self {
            Self::UInt8(_) => MetadataValueType::UInt8,
            Self::Int8(_) => MetadataValueType::Int8,
            Self::UInt16(_) => MetadataValueType::UInt16,
            Self::Int16(_) => MetadataValueType::Int16,
            Self::UInt32(_) => MetadataValueType::UInt32,
            Self::Int32(_) => MetadataValueType::Int32,
            Self::Float32(_) => MetadataValueType::Float32,
            Self::Bool(_) => MetadataValueType::Bool,
            Self::String(_) => MetadataValueType::String,
            Self::Array(_) => MetadataValueType::Array,
            Self::UInt64(_) => MetadataValueType::UInt64,
            Self::Int64(_) => MetadataValueType::Int64,
            Self::Float64(_) => MetadataValueType::Float64,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        if let Self::String(s) = self {
            Some(s)
        } else {
            None
        }
    }

    pub fn as_uint32(&self) -> Option<u32> {
        match self {
            Self::UInt32(v) => Some(*v),
            _ => None,
        }
    }

    // Add more conversion methods as needed...
}

pub type Metadata = HashMap<String, MetadataValue>;
