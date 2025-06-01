//! GGUF tensor loading functionality
//!
//! This module provides functionality to read tensor metadata and load tensor data
//! from GGUF files. Currently focuses on FP16 (half-precision) models without quantization.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use crate::metadata::{GgufError, Result, TensorType};

/// Information about a single tensor in the GGUF file
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    /// Name of the tensor (e.g., "blk.0.attn_norm.weight")
    pub name: String,
    /// Number of dimensions in the tensor
    pub n_dims: u32,
    /// Shape of the tensor - size of each dimension
    pub dims: Vec<u64>,
    /// Data type of the tensor
    pub tensor_type: TensorType,
    /// Byte offset from the start of the tensor data section
    pub offset: u64,
}

impl TensorInfo {
    /// Calculate the total number of elements in this tensor
    pub fn element_count(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Calculate the size in bytes of this tensor's data
    pub fn byte_size(&self) -> u64 {
        let element_count = self.element_count();
        let element_size = match self.tensor_type {
            TensorType::F32 => 4,
            TensorType::F16 => 2,
            TensorType::I32 => 4,
            TensorType::I16 => 2,
            TensorType::I8 => 1,
            TensorType::F64 => 8,
            TensorType::I64 => 8,
            _ => {
                // For quantized types, we'll need more complex calculations
                // For now, return 0 to indicate unsupported
                return 0;
            }
        };
        element_count * element_size
    }

    /// Check if this tensor type is supported for loading
    pub fn is_supported(&self) -> bool {
        matches!(
            self.tensor_type,
            TensorType::F32
                | TensorType::F16
                | TensorType::I32
                | TensorType::I16
                | TensorType::I8
                | TensorType::F64
                | TensorType::I64
        )
    }
}

/// A loaded tensor with its data
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor metadata
    pub info: TensorInfo,
    /// Raw tensor data as bytes
    pub data: Vec<u8>,
}

impl Tensor {
    /// Convert the raw bytes to f32 values (assumes F16 or F32 data)
    pub fn as_f32_vec(&self) -> Result<Vec<f32>> {
        match self.info.tensor_type {
            TensorType::F32 => {
                if self.data.len() % 4 != 0 {
                    return Err(GgufError::InvalidFormat(
                        "F32 tensor data length not divisible by 4".to_string(),
                    ));
                }

                let mut result = Vec::with_capacity(self.data.len() / 4);
                for chunk in self.data.chunks_exact(4) {
                    let bytes: [u8; 4] = chunk.try_into().unwrap();
                    result.push(f32::from_le_bytes(bytes));
                }
                Ok(result)
            }
            TensorType::F16 => {
                if self.data.len() % 2 != 0 {
                    return Err(GgufError::InvalidFormat(
                        "F16 tensor data length not divisible by 2".to_string(),
                    ));
                }

                let mut result = Vec::with_capacity(self.data.len() / 2);
                for chunk in self.data.chunks_exact(2) {
                    let bytes: [u8; 2] = chunk.try_into().unwrap();
                    let f16_bits = u16::from_le_bytes(bytes);
                    result.push(f16_to_f32(f16_bits));
                }
                Ok(result)
            }
            _ => Err(GgufError::Unsupported(format!(
                "Cannot convert tensor type {:?} to f32",
                self.info.tensor_type
            ))),
        }
    }

    /// Get the tensor data as a shaped array (returns flattened data and shape)
    pub fn as_shaped_f32(&self) -> Result<(Vec<f32>, Vec<u64>)> {
        let data = self.as_f32_vec()?;
        Ok((data, self.info.dims.clone()))
    }
}

/// Main interface for loading tensors from GGUF files
pub struct TensorLoader;

impl TensorLoader {
    /// Read all tensor information blocks from the GGUF file
    ///
    /// This function reads the tensor metadata that comes after the key-value pairs
    /// but before the actual tensor data.
    pub fn read_tensor_info<R: Read>(reader: &mut R, n_tensors: u64) -> Result<Vec<TensorInfo>> {
        let mut tensors = Vec::with_capacity(n_tensors as usize);

        for tensor_index in 0..n_tensors {
            // Read tensor name
            let name_len = read_u64_le(reader).map_err(|e| {
                GgufError::InvalidFormat(format!(
                    "Error reading tensor name length for tensor {}: {}",
                    tensor_index, e
                ))
            })?;

            let mut name_bytes = vec![0u8; name_len as usize];
            reader.read_exact(&mut name_bytes)?;
            let name = String::from_utf8(name_bytes)?;

            // Read number of dimensions
            let n_dims = read_u32_le(reader).map_err(|e| {
                GgufError::InvalidFormat(format!(
                    "Error reading n_dims for tensor '{}': {}",
                    name, e
                ))
            })?;

            // Read dimensions
            let mut dims = Vec::with_capacity(n_dims as usize);
            for dim_index in 0..n_dims {
                let dim = read_u64_le(reader).map_err(|e| {
                    GgufError::InvalidFormat(format!(
                        "Error reading dimension {} for tensor '{}': {}",
                        dim_index, name, e
                    ))
                })?;
                dims.push(dim);
            }

            // Read tensor type
            let tensor_type_id = read_u32_le(reader).map_err(|e| {
                GgufError::InvalidFormat(format!(
                    "Error reading tensor type for tensor '{}': {}",
                    name, e
                ))
            })?;

            let tensor_type = TensorType::from_u32(tensor_type_id).ok_or_else(|| {
                GgufError::Unsupported(format!(
                    "Unknown tensor type ID {} for tensor '{}'",
                    tensor_type_id, name
                ))
            })?;

            // Read offset
            let offset = read_u64_le(reader).map_err(|e| {
                GgufError::InvalidFormat(format!(
                    "Error reading offset for tensor '{}': {}",
                    name, e
                ))
            })?;

            tensors.push(TensorInfo {
                name,
                n_dims,
                dims,
                tensor_type,
                offset,
            });
        }

        Ok(tensors)
    }

    /// Load a specific tensor's data from the file
    ///
    /// The reader should be a seekable reader (like a File) positioned anywhere in the file.
    /// This function will seek to the appropriate position to read the tensor data.
    pub fn load_tensor<R: Read + Seek>(
        reader: &mut R,
        tensor_info: &TensorInfo,
        tensor_data_start: u64,
    ) -> Result<Tensor> {
        if !tensor_info.is_supported() {
            return Err(GgufError::Unsupported(format!(
                "Tensor type {:?} is not supported for loading",
                tensor_info.tensor_type
            )));
        }

        let byte_size = tensor_info.byte_size();
        if byte_size == 0 {
            return Err(GgufError::Unsupported(format!(
                "Cannot determine byte size for tensor type {:?}",
                tensor_info.tensor_type
            )));
        }

        // Seek to the tensor data
        let absolute_offset = tensor_data_start + tensor_info.offset;
        reader.seek(SeekFrom::Start(absolute_offset))?;

        // Read the tensor data
        let mut data = vec![0u8; byte_size as usize];
        reader.read_exact(&mut data)?;

        Ok(Tensor {
            info: tensor_info.clone(),
            data,
        })
    }

    /// Load all tensors from the GGUF file
    ///
    /// Returns a HashMap mapping tensor names to loaded tensors.
    /// Only loads supported tensor types (FP32, FP16, etc.).
    pub fn load_all_tensors<R: Read + Seek>(
        reader: &mut R,
        tensor_infos: &[TensorInfo],
        tensor_data_start: u64,
    ) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();

        for tensor_info in tensor_infos {
            if !tensor_info.is_supported() {
                eprintln!(
                    "⚠️  Skipping unsupported tensor '{}' of type {:?}",
                    tensor_info.name, tensor_info.tensor_type
                );
                continue;
            }

            match Self::load_tensor(reader, tensor_info, tensor_data_start) {
                Ok(tensor) => {
                    tensors.insert(tensor_info.name.clone(), tensor);
                }
                Err(e) => {
                    eprintln!("⚠️  Failed to load tensor '{}': {}", tensor_info.name, e);
                    // Continue loading other tensors instead of failing completely
                }
            }
        }

        Ok(tensors)
    }

    /// Calculate the starting position of the tensor data section
    ///
    /// This is called after reading the header, metadata, and tensor info blocks.
    /// The current position in the reader should be the start of the tensor data.
    pub fn get_tensor_data_start<R: Seek>(reader: &mut R) -> Result<u64> {
        Ok(reader.stream_position()?)
    }
}

/// Convert IEEE 754 half-precision (f16) to single-precision (f32)
fn f16_to_f32(f16_bits: u16) -> f32 {
    // Extract components of f16
    let sign = (f16_bits >> 15) & 0x1;
    let exponent = (f16_bits >> 10) & 0x1f;
    let mantissa = f16_bits & 0x3ff;

    // Handle special cases
    if exponent == 0 {
        if mantissa == 0 {
            // Zero
            return if sign == 1 { -0.0 } else { 0.0 };
        } else {
            // Subnormal number
            let mut value = (mantissa as f32) / 1024.0; // 2^10
            value *= 2f32.powi(-14); // 2^(1-15)
            return if sign == 1 { -value } else { value };
        }
    } else if exponent == 31 {
        // Infinity or NaN
        if mantissa == 0 {
            return if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            };
        } else {
            return f32::NAN;
        }
    }

    // Normal number
    let f32_exponent = (exponent as i32) - 15 + 127; // Adjust bias from 15 to 127
    let f32_mantissa = (mantissa as u32) << 13; // Shift mantissa to f32 position

    // Construct f32 bits
    let f32_bits = ((sign as u32) << 31) | ((f32_exponent as u32) << 23) | f32_mantissa;
    f32::from_bits(f32_bits)
}

// Helper functions for reading primitive types
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
