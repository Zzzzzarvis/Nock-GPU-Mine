// ntt_fft_ffi.cu - FFI implementation for NTT/FFT CUDA module

#include "ntt_fft_ffi.h"
#include "ntt_fft.cu" // Include the actual CUDA kernel implementation
// #include "field_ops.h" // <--- ***ACTION REQUIRED***: Uncomment and use correct header if field_pow_op is defined there
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio> // For printf

// If field_pow_op is not in a header but defined in another .cu file (e.g., field_ops.cu)
// and is meant to be used by other .cu files, it should be declared (e.g., in field_ops.h or here).
// For example, if it's a __device__ function:
// __device__ uint64_t field_pow_op(uint64_t base, uint64_t exp);
// This forward declaration might be necessary if not handled by an include.
// We assume PRIME is available from "ntt_fft.cu"

// Helper to check CUDA errors (can be shared or redefined)
static CudaFFIErrorCode check_cuda_error_ntt(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error during %s: %s\n", operation, cudaGetErrorString(err));
        if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_FFI_ALLOC;
        if (err == cudaErrorInvalidValue) return CUDA_ERROR_FFI_INVALID_ARGS;
        return CUDA_ERROR_FFI_KERNEL_LAUNCH;
    }
    return CUDA_SUCCESS_FFI;
}

// Helper function to compute bit reversal indices (can be precomputed on host)
void compute_bit_reverse_indices(int n, std::vector<int>& rev_indices) {
    rev_indices.resize(n);
    int log2_n = 0;
    while ((1 << log2_n) < n) {
        log2_n++;
    }
    for (int i = 0; i < n; ++i) {
        int rev = 0;
        for (int j = 0; j < log2_n; ++j) {
            if ((i >> j) & 1) {
                rev |= 1 << (log2_n - 1 - j);
            }
        }
        rev_indices[i] = rev;
    }
}

// Helper function to compute twiddle factors (can be precomputed on host)
void compute_twiddle_factors(int n, uint64_t root_of_unity, bool inverse, std::vector<uint64_t>& twiddles) {
    twiddles.resize(n / 2);
    uint64_t current_root = root_of_unity;
    if (inverse) {
        // Using PRIME directly, assuming it's the modulus from ntt_fft.cu
        // current_root = field_pow_op(root_of_unity, PRIME - 2); // Proper modular inverse
    }
    for (int i = 0; i < n / 2; ++i) {
        // Ensure field_pow_op is visible here
        twiddles[i] = field_pow_op(current_root, i);
    }
}


extern "C" CudaFFIErrorCode cuda_ntt_fft(uint64_t* h_data, int n, const int* h_rev_indices_param, uint64_t h_root_of_unity, int inverse) {
    // Parameter checks first
    if (!h_data || n <= 0 || (n & (n - 1)) != 0) { // n must be a power of two
        fprintf(stderr, "Error (cuda_ntt_fft): Invalid arguments.\n");
        return CUDA_ERROR_FFI_INVALID_ARGS;
    }

    // Variable declarations moved up, after parameter checks
    uint64_t* d_data = nullptr;
    int* d_rev_indices = nullptr;
    uint64_t* d_twiddle_factors = nullptr;
    CudaFFIErrorCode ffi_err_code = CUDA_SUCCESS_FFI;
    cudaError_t cuda_err;
    std::vector<int> rev_indices_vec;
    const int* p_rev_indices = h_rev_indices_param;
    int threads_per_block = 256; // Default threads per block
    int blocks_per_grid;         // Calculated based on n
    std::vector<uint64_t> stage_twiddles_vec; // For CPU computation of twiddles

    if (!p_rev_indices) {
        compute_bit_reverse_indices(n, rev_indices_vec);
        p_rev_indices = rev_indices_vec.data();
    }

    // 1. Allocate memory on device
    cuda_err = cudaMalloc((void**)&d_data, n * sizeof(uint64_t));
    if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaMalloc d_data")) != CUDA_SUCCESS_FFI) goto cleanup;
    cuda_err = cudaMalloc((void**)&d_rev_indices, n * sizeof(int)); 
    if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaMalloc d_rev_indices")) != CUDA_SUCCESS_FFI) goto cleanup;
    
    // 2. Copy data to device
    cuda_err = cudaMemcpy(d_data, h_data, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaMemcpy h_data to d_data")) != CUDA_SUCCESS_FFI) goto cleanup;
    cuda_err = cudaMemcpy(d_rev_indices, p_rev_indices, n * sizeof(int), cudaMemcpyHostToDevice);
    if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaMemcpy p_rev_indices to d_rev_indices")) != CUDA_SUCCESS_FFI) goto cleanup;

    // 3. Kernel launch parameters calculation
    blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    // 4. Launch bit-reversal permutation kernel
    bit_reverse_permutation_kernel<<<blocks_per_grid, threads_per_block>>>(d_data, n, d_rev_indices);
    cuda_err = cudaGetLastError();
    if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "bit_reverse_permutation_kernel launch")) != CUDA_SUCCESS_FFI) goto cleanup;
    cuda_err = cudaDeviceSynchronize();
    if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaDeviceSynchronize after bit_reverse")) != CUDA_SUCCESS_FFI) goto cleanup;

    // 5. Iterative NTT stages (Cooley-Tukey)
    for (int len = 2; len <= n; len <<= 1) { 
        // Ensure field_pow_op is visible
        uint64_t w_m_base = field_pow_op(h_root_of_unity, n / len); 
        stage_twiddles_vec.resize(len/2);
        for(int j=0; j < len/2; ++j) {
            stage_twiddles_vec[j] = field_pow_op(w_m_base, j);
        }

        d_twiddle_factors = nullptr; // Initialize before malloc
        cuda_err = cudaMalloc((void**)&d_twiddle_factors, (len/2) * sizeof(uint64_t));
        if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaMalloc d_twiddle_factors in loop")) != CUDA_SUCCESS_FFI) goto cleanup_loop;
        
        cuda_err = cudaMemcpy(d_twiddle_factors, stage_twiddles_vec.data(), (len/2) * sizeof(uint64_t), cudaMemcpyHostToDevice);
        if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaMemcpy stage_twiddles_vec in loop")) != CUDA_SUCCESS_FFI) goto cleanup_loop;

        int stage_threads = n / 2; 
        int stage_blocks = (stage_threads + threads_per_block - 1) / threads_per_block;

        ntt_fft_stage_kernel<<<stage_blocks, threads_per_block>>>(d_data, n, len, d_twiddle_factors, 0);
        cuda_err = cudaGetLastError();
        if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "ntt_fft_stage_kernel launch")) != CUDA_SUCCESS_FFI) goto cleanup_loop;
        
        cuda_err = cudaDeviceSynchronize();
        if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaDeviceSynchronize after stage kernel")) != CUDA_SUCCESS_FFI) goto cleanup_loop;
        
        cudaFree(d_twiddle_factors); 
        d_twiddle_factors = nullptr; 
    }

    // If inverse NTT, divide by N (modularly)
    if (inverse) {
        // Using PRIME directly, assuming it's the modulus from ntt_fft.cu
        uint64_t n_val = static_cast<uint64_t>(n);
        uint64_t n_inv = field_pow_op(n_val, PRIME - 2); 
        
        // Kernel to multiply all elements by n_inv
        // Placeholder: This kernel needs to be implemented and available.
        // vector_scalar_field_mul_kernel<<<blocks_per_grid, threads_per_block>>>(d_data, n_inv, n, PRIME);
        // cuda_err = cudaGetLastError();
        // if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "vector_scalar_field_mul_kernel launch")) != CUDA_SUCCESS_FFI) goto cleanup;
        // cuda_err = cudaDeviceSynchronize();
        // if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaDeviceSynchronize after N_inv mul")) != CUDA_SUCCESS_FFI) goto cleanup;
        fprintf(stdout, "INFO (cuda_ntt_fft): Inverse NTT scalar multiplication by N_inv needs a specific kernel.\n");
    }

    // 6. Copy data back to host
    cuda_err = cudaMemcpy(h_data, d_data, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) { // Check error directly
         check_cuda_error_ntt(cuda_err, "cudaMemcpy d_data to h_data");
         if (ffi_err_code == CUDA_SUCCESS_FFI) ffi_err_code = CUDA_ERROR_FFI_INTERNAL;
    }

cleanup_loop: 
    if (d_twiddle_factors && ffi_err_code != CUDA_SUCCESS_FFI) { 
         cudaFree(d_twiddle_factors);
         d_twiddle_factors = nullptr;
    }
cleanup:
    if (d_data) cudaFree(d_data);
    if (d_rev_indices) cudaFree(d_rev_indices);
    if (d_twiddle_factors) cudaFree(d_twiddle_factors); // Should be null if loop completed normally

    return ffi_err_code;
}
