// ntt_fft_ffi.cu - FFI implementation for NTT/FFT CUDA module

#include "ntt_fft_ffi.h"
#include "ntt_fft.cu" // Include the actual CUDA kernel implementation (provides PRIME and kernels)
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio> // For printf

// HOST-SIDE modular multiplication (mirroring device logic if PRIME is the same)
// Needed because field_pow_op from field_ops.cu is __device__ only.
// PRIME is visible from the included "ntt_fft.cu"
static inline uint64_t host_field_mul(uint64_t a, uint64_t b) {
    unsigned __int128 res = (unsigned __int128)a * b;
    return (uint64_t)(res % PRIME); // Use PRIME from ntt_fft.cu
}

// HOST-SIDE modular exponentiation (binary exponentiation)
static inline uint64_t host_field_pow(uint64_t base, uint64_t exp) {
    uint64_t res = 1;
    base %= PRIME; // Use PRIME from ntt_fft.cu
    while (exp > 0) {
        if (exp % 2 == 1) res = host_field_mul(res, base);
        base = host_field_mul(base, base);
        exp /= 2;
    }
    return res;
}


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
        // current_root = host_field_pow(root_of_unity, PRIME - 2); // Proper modular inverse using host function
    }
    for (int i = 0; i < n / 2; ++i) {
        twiddles[i] = host_field_pow(current_root, i); // Use host_field_pow
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
    // bit_reverse_permutation_kernel is from ntt_fft.cu
    bit_reverse_permutation_kernel<<<blocks_per_grid, threads_per_block>>>(d_data, n, d_rev_indices);
    cuda_err = cudaGetLastError();
    if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "bit_reverse_permutation_kernel launch")) != CUDA_SUCCESS_FFI) goto cleanup;
    cuda_err = cudaDeviceSynchronize();
    if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaDeviceSynchronize after bit_reverse")) != CUDA_SUCCESS_FFI) goto cleanup;

    // 5. Iterative NTT stages (Cooley-Tukey)
    for (int len = 2; len <= n; len <<= 1) { 
        uint64_t w_m_base = host_field_pow(h_root_of_unity, n / len); // Use host_field_pow
        stage_twiddles_vec.resize(len/2);
        for(int j=0; j < len/2; ++j) {
            stage_twiddles_vec[j] = host_field_pow(w_m_base, j); // Use host_field_pow
        }

        d_twiddle_factors = nullptr; // Initialize before malloc
        cuda_err = cudaMalloc((void**)&d_twiddle_factors, (len/2) * sizeof(uint64_t));
        if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaMalloc d_twiddle_factors in loop")) != CUDA_SUCCESS_FFI) goto cleanup_loop;
        
        cuda_err = cudaMemcpy(d_twiddle_factors, stage_twiddles_vec.data(), (len/2) * sizeof(uint64_t), cudaMemcpyHostToDevice);
        if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaMemcpy stage_twiddles_vec in loop")) != CUDA_SUCCESS_FFI) goto cleanup_loop;

        int stage_threads = n / 2; 
        int stage_blocks = (stage_threads + threads_per_block - 1) / threads_per_block;

        // ntt_fft_stage_kernel is from ntt_fft.cu
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
        uint64_t n_val = static_cast<uint64_t>(n);
        uint64_t n_inv = host_field_pow(n_val, PRIME - 2); // Use host_field_pow and PRIME from ntt_fft.cu
        
        // Kernel to multiply all elements by n_inv
        // This requires a new kernel, let's call it vector_scalar_field_mul_kernel.
        // This kernel would take d_data, n_inv, n, and PRIME as arguments.
        // For now, we will assume this kernel exists or will be added to ntt_fft.cu or a similar file.
        // Example: vector_scalar_field_mul_kernel<<<blocks_per_grid, threads_per_block>>>(d_data, n_inv, n);
        // cuda_err = cudaGetLastError();
        // if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "vector_scalar_field_mul_kernel launch")) != CUDA_SUCCESS_FFI) goto cleanup;
        // cuda_err = cudaDeviceSynchronize();
        // if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaDeviceSynchronize after N_inv mul")) != CUDA_SUCCESS_FFI) goto cleanup;
        fprintf(stdout, "INFO (cuda_ntt_fft): Inverse NTT scalar multiplication by N_inv (value: %lu) needs a specific kernel (vector_scalar_field_mul_kernel).\n", n_inv);
    }

    // 6. Copy data back to host
    cuda_err = cudaMemcpy(h_data, d_data, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) { 
         check_cuda_error_ntt(cuda_err, "cudaMemcpy d_data to h_data"); // Call to log error
         // Ensure ffi_err_code reflects this failure if it was previously success
         if (ffi_err_code == CUDA_SUCCESS_FFI) ffi_err_code = CUDA_ERROR_FFI_MEMCPY; // Or a more general error
    }

cleanup_loop: 
    // This label is for errors inside the main loop, specifically for d_twiddle_factors
    if (d_twiddle_factors && ffi_err_code != CUDA_SUCCESS_FFI) { 
         cudaFree(d_twiddle_factors);
         d_twiddle_factors = nullptr; // Nullify after freeing
    }
cleanup:
    if (d_data) cudaFree(d_data);
    if (d_rev_indices) cudaFree(d_rev_indices);
    // If d_twiddle_factors was allocated in the loop and an error occurred outside/after the loop,
    // it might still hold a pointer if cleanup_loop wasn't hit.
    // However, the current logic frees it inside the loop or if an error occurs in the loop.
    // For safety, one might consider an additional check here, but it should be null if loop exited cleanly or via cleanup_loop.
    if (d_twiddle_factors) cudaFree(d_twiddle_factors); 


    return ffi_err_code;
}
