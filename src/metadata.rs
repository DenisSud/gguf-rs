use std::collections::HashMap;
use std::fmt;
use std::io::{self, Read};

/// Magic number for GGUF files ('GGUF' in little-endian)
pub const GGUF_MAGIC: u32 = 0x46554747;

/// Result type for GGUF operations
pub type Result<T> = std::result::Result<T, GgufError>;

/// Errors that can occur when parsing GGUF files
#[derive(Debug)]
pub enum GgufError {
    /// I/O error during file operations
    Io(io::Error),
    /// Invalid file format (wrong magic number, malformed data, etc.)
    InvalidFormat(String),
    /// Unsupported GGUF version or feature
    Unsupported(String),
    /// Invalid UTF-8 string data
    InvalidUtf8(std::string::FromUtf8Error),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::Io(err) => write!(f, "I/O error: {}", err),
            GgufError::InvalidFormat(msg) => write!(f, "Invalid GGUF format: {}", msg),
            GgufError::Unsupported(msg) => write!(f, "Unsupported feature: {}", msg),
            GgufError::InvalidUtf8(err) => write!(f, "Invalid UTF-8: {}", err),
        }
    }
}

impl std::error::Error for GgufError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GgufError::Io(err) => Some(err),
            GgufError::InvalidUtf8(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for GgufError {
    fn from(err: io::Error) -> Self {
        GgufError::Io(err)
    }
}

impl From<std::string::FromUtf8Error> for GgufError {
    fn from(err: std::string::FromUtf8Error) -> Self {
        GgufError::InvalidUtf8(err)
    }
}

/// Essential header information found at the beginning of a GGUF file.
///
/// GGUF files store these values in little-endian byte order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufHeader {
    /// The magic number identifying the GGUF file format (0x46554747, or "GGUF")
    pub magic: u32,
    /// The version of the GGUF file format
    pub version: u32,
    /// The total number of tensors (model weights) contained in the file
    pub n_tensors: u64,
    /// The total number of key-value metadata entries in the file
    pub n_kv: u64,
}

impl GgufHeader {
    /// Parse a GGUF header from the beginning of a reader.
    ///
    /// This function expects the reader to be positioned at the very beginning of the GGUF file.
    /// It reads and validates the magic number, then extracts the version, number of tensors,
    /// and number of key-value pairs.
    ///
    /// # Errors
    ///
    /// Returns `GgufError::Io` if an I/O error occurs during reading.
    /// Returns `GgufError::InvalidFormat` if the magic number doesn't match the GGUF signature.
    pub fn parse<R: Read>(reader: &mut R) -> Result<Self> {
        let magic = read_u32_le(reader)?;

        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidFormat(format!(
                "Invalid magic number. Expected 0x{:08X}, got 0x{:08X}",
                GGUF_MAGIC, magic
            )));
        }

        let version = read_u32_le(reader)?;
        let n_tensors = read_u64_le(reader)?;
        let n_kv = read_u64_le(reader)?;

        Ok(GgufHeader {
            magic,
            version,
            n_tensors,
            n_kv,
        })
    }

    /// Check if this GGUF version is supported by this library
    pub fn is_version_supported(&self) -> bool {
        // Add version checks as needed
        self.version >= 1 && self.version <= 3
    }
}

/// Possible GGUF metadata value types, mapping to their u32 identifiers.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[repr(u32)]
pub enum ValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl ValueType {
    /// Convert a raw u32 type ID into a ValueType.
    ///
    /// Returns `None` if the type ID is not recognized.
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(ValueType::Uint8),
            1 => Some(ValueType::Int8),
            2 => Some(ValueType::Uint16),
            3 => Some(ValueType::Int16),
            4 => Some(ValueType::Uint32),
            5 => Some(ValueType::Int32),
            6 => Some(ValueType::Float32),
            7 => Some(ValueType::Bool),
            8 => Some(ValueType::String),
            9 => Some(ValueType::Array),
            10 => Some(ValueType::Uint64),
            11 => Some(ValueType::Int64),
            12 => Some(ValueType::Float64),
            _ => None,
        }
    }
}

/// A parsed GGUF metadata value, holding the actual data.
#[derive(Debug, PartialEq)]
pub enum Value {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(ValueType, Vec<Value>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl Value {
    /// Get the type of this value
    pub fn value_type(&self) -> ValueType {
        match self {
            Value::Uint8(_) => ValueType::Uint8,
            Value::Int8(_) => ValueType::Int8,
            Value::Uint16(_) => ValueType::Uint16,
            Value::Int16(_) => ValueType::Int16,
            Value::Uint32(_) => ValueType::Uint32,
            Value::Int32(_) => ValueType::Int32,
            Value::Float32(_) => ValueType::Float32,
            Value::Bool(_) => ValueType::Bool,
            Value::String(_) => ValueType::String,
            Value::Array(_, _) => ValueType::Array,
            Value::Uint64(_) => ValueType::Uint64,
            Value::Int64(_) => ValueType::Int64,
            Value::Float64(_) => ValueType::Float64,
        }
    }

    /// Attempt to extract a string value
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Attempt to extract a boolean value
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Attempt to extract a u64 value
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Uint64(n) => Some(*n),
            Value::Uint32(n) => Some(*n as u64),
            Value::Uint16(n) => Some(*n as u64),
            Value::Uint8(n) => Some(*n as u64),
            _ => None,
        }
    }

    /// Attempt to extract an i64 value
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int64(n) => Some(*n),
            Value::Int32(n) => Some(*n as i64),
            Value::Int16(n) => Some(*n as i64),
            Value::Int8(n) => Some(*n as i64),
            _ => None,
        }
    }

    /// Attempt to extract an f64 value
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float64(f) => Some(*f),
            Value::Float32(f) => Some(*f as f64),
            _ => None,
        }
    }
}

/// GGUF tensor data types
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[repr(u32)]
pub enum TensorType {
    F32 = 0,
    F16 = 1,
    Q40 = 2,
    Q41 = 3,
    Q50 = 6,
    Q51 = 7,
    Q80 = 8,
    Q81 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
}

impl TensorType {
    /// Convert a raw u32 type ID into a TensorType.
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(TensorType::F32),
            1 => Some(TensorType::F16),
            2 => Some(TensorType::Q40),
            3 => Some(TensorType::Q41),
            6 => Some(TensorType::Q50),
            7 => Some(TensorType::Q51),
            8 => Some(TensorType::Q80),
            9 => Some(TensorType::Q81),
            10 => Some(TensorType::Q2K),
            11 => Some(TensorType::Q3K),
            12 => Some(TensorType::Q4K),
            13 => Some(TensorType::Q5K),
            14 => Some(TensorType::Q6K),
            15 => Some(TensorType::Q8K),
            16 => Some(TensorType::Iq2Xxs),
            17 => Some(TensorType::Iq2Xs),
            18 => Some(TensorType::Iq3Xxs),
            19 => Some(TensorType::Iq1S),
            20 => Some(TensorType::Iq4Nl),
            21 => Some(TensorType::Iq3S),
            22 => Some(TensorType::Iq2S),
            23 => Some(TensorType::Iq4Xs),
            24 => Some(TensorType::I8),
            25 => Some(TensorType::I16),
            26 => Some(TensorType::I32),
            27 => Some(TensorType::I64),
            28 => Some(TensorType::F64),
            29 => Some(TensorType::Iq1M),
            _ => None,
        }
    }
}

/// Main interface for reading GGUF files
pub struct GgufReader;

impl GgufReader {
    /// Read all key-value pairs from the GGUF file's metadata section.
    ///
    /// This function assumes the reader is positioned immediately after the GGUF header.
    /// It reads `n_kv` key-value pairs as specified in the header.
    ///
    /// # Errors
    ///
    /// Returns `GgufError::Io` if an I/O error occurs during reading.
    /// Returns `GgufError::InvalidFormat` if the data is malformed.
    /// Returns `GgufError::Unsupported` if an unknown value type is encountered.
    pub fn read_metadata<R: Read>(reader: &mut R, n_kv: u64) -> Result<HashMap<String, Value>> {
        let mut metadata_map = HashMap::with_capacity(n_kv as usize);

        for kv_index in 0..n_kv {
            // Read key
            let key_len = read_u64_le(reader).map_err(|e| {
                GgufError::InvalidFormat(format!(
                    "Error reading key length for KV pair {}: {}",
                    kv_index, e
                ))
            })?;

            let mut key_bytes = vec![0u8; key_len as usize];
            reader.read_exact(&mut key_bytes)?;
            let key = String::from_utf8(key_bytes)?;

            // Read value type
            let value_type_id = read_u32_le(reader).map_err(|e| {
                GgufError::InvalidFormat(format!(
                    "Error reading value type for KV pair {}: {}",
                    kv_index, e
                ))
            })?;

            let value_type = ValueType::from_u32(value_type_id).ok_or_else(|| {
                GgufError::Unsupported(format!("Unknown GGUF value type ID: {}", value_type_id))
            })?;

            // Read value
            let value = Self::read_value(reader, value_type).map_err(|e| {
                GgufError::InvalidFormat(format!("Error reading value for key '{}': {}", key, e))
            })?;

            metadata_map.insert(key, value);
        }

        Ok(metadata_map)
    }

    /// Read a single GGUF value from the reader
    fn read_value<R: Read>(reader: &mut R, value_type: ValueType) -> Result<Value> {
        match value_type {
            ValueType::Uint8 => Ok(Value::Uint8(read_u8(reader)?)),
            ValueType::Int8 => Ok(Value::Int8(read_u8(reader)? as i8)),
            ValueType::Uint16 => Ok(Value::Uint16(read_u16_le(reader)?)),
            ValueType::Int16 => Ok(Value::Int16(read_u16_le(reader)? as i16)),
            ValueType::Uint32 => Ok(Value::Uint32(read_u32_le(reader)?)),
            ValueType::Int32 => Ok(Value::Int32(read_u32_le(reader)? as i32)),
            ValueType::Float32 => Ok(Value::Float32(f32::from_le_bytes([
                read_u8(reader)?,
                read_u8(reader)?,
                read_u8(reader)?,
                read_u8(reader)?,
            ]))),
            ValueType::Bool => Ok(Value::Bool(read_u8(reader)? != 0)),
            ValueType::String => {
                let len = read_u64_le(reader)? as usize;
                let mut string_bytes = vec![0u8; len];
                reader.read_exact(&mut string_bytes)?;
                let s = String::from_utf8(string_bytes)?;
                Ok(Value::String(s))
            }
            ValueType::Array => {
                let element_type_id = read_u32_le(reader)?;
                let element_type = ValueType::from_u32(element_type_id).ok_or_else(|| {
                    GgufError::Unsupported(format!(
                        "Unknown array element type ID: {}",
                        element_type_id
                    ))
                })?;

                let count = read_u64_le(reader)? as usize;
                let mut elements = Vec::with_capacity(count);

                for _ in 0..count {
                    elements.push(Self::read_value(reader, element_type)?);
                }

                Ok(Value::Array(element_type, elements))
            }
            ValueType::Uint64 => Ok(Value::Uint64(read_u64_le(reader)?)),
            ValueType::Int64 => Ok(Value::Int64(read_u64_le(reader)? as i64)),
            ValueType::Float64 => Ok(Value::Float64(f64::from_le_bytes([
                read_u8(reader)?,
                read_u8(reader)?,
                read_u8(reader)?,
                read_u8(reader)?,
                read_u8(reader)?,
                read_u8(reader)?,
                read_u8(reader)?,
                read_u8(reader)?,
            ]))),
        }
    }
}

// Helper functions for reading primitive types
fn read_u8<R: Read>(reader: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16_le<R: Read>(reader: &mut R) -> Result<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32_le<R: Read>(reader: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le<R: Read>(reader: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}
