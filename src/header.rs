// src/header.rs
use crate::error::{GGUFError, Result};
use crate::metadata::{Metadata, MetadataValue, MetadataValueType};
use crate::tensor::read_gguf_string;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{Read, Seek};

pub struct GGUFHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: Metadata,
    pub alignment: u32,
}

impl GGUFHeader {
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Err(GGUFError::InvalidMagic);
        }

        let version = reader.read_u32::<LittleEndian>()?;
        if version != 3 {
            return Err(GGUFError::UnsupportedVersion(version));
        }

        let tensor_count = reader.read_u64::<LittleEndian>()?;
        let metadata_kv_count = reader.read_u64::<LittleEndian>()?;

        let mut metadata = Metadata::new();
        for _ in 0..metadata_kv_count {
            let (key, value) = read_metadata_kv(reader)?;
            metadata.insert(key, value);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_uint32())
            .unwrap_or(32);

        if alignment < 8 {
            return Err(GGUFError::InvalidAlignment(alignment));
        }

        Ok(Self {
            version,
            tensor_count,
            metadata,
            alignment,
        })
    }
}

fn read_metadata_kv<R: Read + Seek>(reader: &mut R) -> Result<(String, MetadataValue)> {
    let key = read_gguf_string(reader)?;
    let value_type = MetadataValueType::try_from(reader.read_u32::<LittleEndian>()?)?;

    let value = match value_type {
        MetadataValueType::UInt8 => MetadataValue::UInt8(reader.read_u8()?),
        MetadataValueType::Int8 => MetadataValue::Int8(reader.read_i8()?),
        MetadataValueType::UInt16 => MetadataValue::UInt16(reader.read_u16::<LittleEndian>()?),
        MetadataValueType::Int16 => MetadataValue::Int16(reader.read_i16::<LittleEndian>()?),
        MetadataValueType::UInt32 => MetadataValue::UInt32(reader.read_u32::<LittleEndian>()?),
        MetadataValueType::Int32 => MetadataValue::Int32(reader.read_i32::<LittleEndian>()?),
        MetadataValueType::Float32 => MetadataValue::Float32(reader.read_f32::<LittleEndian>()?),
        MetadataValueType::Bool => MetadataValue::Bool(reader.read_u8()? != 0),
        MetadataValueType::String => MetadataValue::String(read_gguf_string(reader)?),
        MetadataValueType::Array => {
            let array_type_val = reader.read_u32::<LittleEndian>()?;
            let array_type = MetadataValueType::try_from(array_type_val)?;
            let len = reader.read_u64::<LittleEndian>()?;

            let mut items = Vec::with_capacity(len as usize);
            for _ in 0..len {
                items.push(read_typed_value(reader, &array_type)?);
            }
            MetadataValue::Array(items)
        }
        MetadataValueType::UInt64 => MetadataValue::UInt64(reader.read_u64::<LittleEndian>()?),
        MetadataValueType::Int64 => MetadataValue::Int64(reader.read_i64::<LittleEndian>()?),
        MetadataValueType::Float64 => MetadataValue::Float64(reader.read_f64::<LittleEndian>()?),
    };

    Ok((key, value))
}

fn read_typed_value<R: Read + Seek>(
    reader: &mut R,
    value_type: &MetadataValueType,
) -> Result<MetadataValue> {
    match *value_type {
        MetadataValueType::UInt8 => Ok(MetadataValue::UInt8(reader.read_u8()?)),
        MetadataValueType::Int8 => Ok(MetadataValue::Int8(reader.read_i8()?)),
        MetadataValueType::UInt16 => Ok(MetadataValue::UInt16(reader.read_u16::<LittleEndian>()?)),
        MetadataValueType::Int16 => Ok(MetadataValue::Int16(reader.read_i16::<LittleEndian>()?)),
        MetadataValueType::UInt32 => Ok(MetadataValue::UInt32(reader.read_u32::<LittleEndian>()?)),
        MetadataValueType::Int32 => Ok(MetadataValue::Int32(reader.read_i32::<LittleEndian>()?)),
        MetadataValueType::Float32 => {
            Ok(MetadataValue::Float32(reader.read_f32::<LittleEndian>()?))
        }
        MetadataValueType::Bool => Ok(MetadataValue::Bool(reader.read_u8()? != 0)),
        MetadataValueType::String => Ok(MetadataValue::String(read_gguf_string(reader)?)),
        MetadataValueType::UInt64 => Ok(MetadataValue::UInt64(reader.read_u64::<LittleEndian>()?)),
        MetadataValueType::Int64 => Ok(MetadataValue::Int64(reader.read_i64::<LittleEndian>()?)),
        MetadataValueType::Float64 => {
            Ok(MetadataValue::Float64(reader.read_f64::<LittleEndian>()?))
        }
        MetadataValueType::Array => Err(GGUFError::Metadata(
            "Nested arrays not supported".to_string(),
        )),
    }
}
