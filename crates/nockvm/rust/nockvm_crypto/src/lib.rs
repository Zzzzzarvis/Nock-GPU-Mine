#[cfg(feature = "aes_siv")]
pub mod aes_siv;

#[cfg(feature = "ed25519")]
pub mod ed25519;

#[cfg(feature = "sha")]
pub mod sha;

// 新增 GPU FFI 模块声明
pub mod gpu_ffi;

// 可选择地重新导出 GPU FFI 中的公共函数，方便调用
// 例如，如果希望所有 gpu_... 函数都可直接通过 nockvm_crypto::xxx 调用
pub use gpu_ffi::{
    gpu_poly_mul,
    gpu_ntt_fft,
    gpu_vector_field_add,
    gpu_vector_field_mul,
    gpu_vector_field_pow,
    gpu_sha_hash,
    gpu_tip5_hash,
    CudaFFIErrorCode, // 也导出错误码枚举，如果需要在外部处理
};
