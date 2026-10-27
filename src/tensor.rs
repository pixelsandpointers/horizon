//! Burn Tensors

pub use burn::tensor::Tensor;
use burn::tensor::{DType, Device, TensorCreationOptions, TensorData};

/// The default device: burn's pure-Rust CPU backend.
pub fn default_device() -> Device {
    Device::flex()
}

pub fn tensor(values: &[f64], device: &Device) -> Tensor<1> {
    let options = TensorCreationOptions::new(device.clone()).with_dtype(DType::F64);
    Tensor::<1>::from_data(TensorData::new(values.to_vec(), [values.len()]), options).autodiff()
}

pub fn to_vec(t: Tensor<1>) -> Vec<f64> {
    t.into_data()
        .convert::<f64>()
        .try_to_vec::<f64>()
        .expect("an f64 tensor")
}
