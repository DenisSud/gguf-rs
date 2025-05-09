// src/model.rs
use crate::error::{GGUFError, Result}; // Add GGUFError import
use crate::header::GGUFHeader;
use crate::tensor::TensorInfo;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

pub struct GGUFModel {
    pub header: GGUFHeader,
    pub tensors: Vec<TensorInfo>,
    pub data: Mmap,
}

impl GGUFModel {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut cursor = std::io::Cursor::new(&mmap[..]);

        let header = GGUFHeader::read(&mut cursor)?;
        let mut tensors = Vec::with_capacity(header.tensor_count as usize);

        for _ in 0..header.tensor_count {
            tensors.push(TensorInfo::read(&mut cursor)?);
        }

        Ok(Self {
            header,
            tensors,
            data: mmap,
        })
    }

    pub fn get_tensor_data(&self, tensor: &TensorInfo) -> Result<&[u8]> {
        let start = tensor.offset as usize;
        let end = start + self.calculate_tensor_size(tensor)?;

        if end > self.data.len() {
            return Err(GGUFError::Tensor("Tensor data out of bounds".to_string()));
        }

        Ok(&self.data[start..end])
    }

    fn calculate_tensor_size(&self, tensor: &TensorInfo) -> Result<usize> {
        let element_size = match tensor.type_ {
            0 => 4, // F32
            1 => 2, // F16
            _ => return Err(GGUFError::InvalidTensorType(tensor.type_)),
        };

        let count = tensor.dimensions.iter().product::<u64>();
        Ok((count * element_size) as usize)
    }
}
