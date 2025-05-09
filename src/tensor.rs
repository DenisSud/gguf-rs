use crate::error::{GGUFError, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{Read, Seek};

#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub type_: u32,
    pub offset: u64,
}

impl TensorInfo {
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        let name = read_gguf_string(reader)?;
        validate_tensor_name(&name)?;

        let n_dims = reader.read_u32::<LittleEndian>()?;
        let mut dimensions = Vec::with_capacity(n_dims as usize);

        for _ in 0..n_dims {
            dimensions.push(reader.read_u64::<LittleEndian>()?);
        }

        let type_ = reader.read_u32::<LittleEndian>()?;
        let offset = reader.read_u64::<LittleEndian>()?;

        Ok(Self {
            name,
            dimensions,
            type_,
            offset,
        })
    }
}

fn validate_tensor_name(name: &str) -> Result<()> {
    if name.is_empty() || name.len() > 64 {
        return Err(GGUFError::InvalidTensorName(name.to_string()));
    }
    Ok(())
}

pub(crate) fn read_gguf_string<R: Read + Seek>(reader: &mut R) -> Result<String> {
    let len = reader.read_u64::<LittleEndian>()?;
    if len > 1024 * 1024 {
        return Err(GGUFError::Metadata("String too long".to_string()));
    }

    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf)?;

    String::from_utf8(buf).map_err(|_| GGUFError::InvalidString)
}
