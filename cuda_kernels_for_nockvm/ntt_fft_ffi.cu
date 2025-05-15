// ntt_fft_ffi.cu - FFI implementation for NTT/FFT CUDA module

#include "ntt_fft_ffi.h"
#include "ntt_fft.cu" // Include the actual CUDA kernel implementation
// #include "field_ops.h" // <<--- HYPOTHETICAL: Include if field_pow_op is declared here
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio> // For printf

// HYPOTHETICAL: If field_pow_op is defined in another .cu and not in a header,
// you might need a forward declaration if called directly from a __global__ or __device__ context,
// or ensure it's linked if called from host helper code.
// For now, assuming it's made available via an included header or is in ntt_fft.cu
// Example forward declaration (if needed and not in a header):
// __device__ uint64_t field_pow_op(uint64_t base, uint64_t exp);


// Assuming PRIME is defined in ntt_fft.cu or another included header
// If not, you might need to define it or include where it's defined.
// extern const uint64_t PRIME; // If PRIME is a global const in another .cu file

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
// This is a simple CPU version for demonstration.
// Assumes field_pow_op and PRIME are accessible
void compute_twiddle_factors(int n, uint64_t root_of_unity, bool inverse, std::vector<uint64_t>& twiddles) {
    twiddles.resize(n / 2);
    uint64_t current_root = root_of_unity;
    if (inverse) {
        // Need to find modular inverse of root_of_unity for iNTT
        // This uses PRIME, ensure it's correctly defined and accessible.
        // current_root = field_pow_op(root_of_unity, PRIME - 2); // Fermat's Little Theorem for modular inverse if PRIME is prime
    }
    for (int i = 0; i < n / 2; ++i) {
        // Ensure field_pow_op is correctly declared/included for this call
        twiddles[i] = field_pow_op(current_root, i);
    }
}


extern "C" CudaFFIErrorCode cuda_ntt_fft(uint64_t* h_data, int n, const int* h_rev_indices_param, uint64_t h_root_of_unity, int inverse) {
    // Parameter checks first
    if (!h_data || n <= 0 || (n & (n - 1)) != 0) { // n must be a power of two
        fprintf(stderr, "Error (cuda_ntt_fft): Invalid arguments. Null h_data, non-positive n, or n not power of two.\n");
        return CUDA_ERROR_FFI_INVALID_ARGS;
    }

    // Variable declarations moved to the top after initial checks
    uint64_t* d_data = nullptr;
    int* d_rev_indices = nullptr;
    uint64_t* d_twiddle_factors = nullptr; // Used inside the loop for per-stage twiddles
    CudaFFIErrorCode ffi_err_code = CUDA_SUCCESS_FFI;
    cudaError_t cuda_err;
    std::vector<int> rev_indices_vec; // For CPU computation if h_rev_indices_param is null
    const int* p_rev_indices = h_rev_indices_param;
    int threads_per_block = 256; // Common CUDA launch parameter
    int blocks_per_grid;         // Calculated based on n and threads_per_block
    std::vector<uint64_t> stage_twiddles_vec; // For CPU computation of twiddles per stage

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

    // 3. Calculate grid size for initial kernel
    blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    // 4. Launch bit-reversal permutation kernel
    bit_reverse_permutation_kernel<<<blocks_per_grid, threads_per_block>>>(d_data, n, d_rev_indices);
    cuda_err = cudaGetLastError();
    if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "bit_reverse_permutation_kernel launch")) != CUDA_SUCCESS_FFI) goto cleanup;
    cuda_err = cudaDeviceSynchronize(); // Ensure bit reversal is complete
    if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaDeviceSynchronize after bit_reverse")) != CUDA_SUCCESS_FFI) goto cleanup;

    // 5. Iterative NTT stages (Cooley-Tukey)
    for (int len = 2; len <= n; len <<= 1) {
        // Ensure field_pow_op is usable here
        uint64_t w_m_base = field_pow_op(h_root_of_unity, n / len); 
        stage_twiddles_vec.resize(len/2);
        for(int j=0; j < len/2; ++j) {
            stage_twiddles_vec[j] = field_pow_op(w_m_base, j);
        }

        // cudaFree for d_twiddle_factors should be outside loop if allocated once, or inside if per-stage.
        // Current logic is per-stage, so malloc/free inside.
        d_twiddle_factors = nullptr; // Initialize before malloc
        cuda_err = cudaMalloc((void**)&d_twiddle_factors, (len/2) * sizeof(uint64_t));
        if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaMalloc d_twiddle_factors in loop")) != CUDA_SUCCESS_FFI) goto cleanup_loop; // Use cleanup_loop
        
        cuda_err = cudaMemcpy(d_twiddle_factors, stage_twiddles_vec.data(), (len/2) * sizeof(uint64_t), cudaMemcpyHostToDevice);
        if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaMemcpy stage_twiddles_vec in loop")) != CUDA_SUCCESS_FFI) goto cleanup_loop; // Use cleanup_loop

        int stage_threads = n / 2; 
        int stage_blocks = (stage_threads + threads_per_block - 1) / threads_per_block;

        ntt_fft_stage_kernel<<<stage_blocks, threads_per_block>>>(d_data, n, len, d_twiddle_factors, 0);
        cuda_err = cudaGetLastError();
        if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "ntt_fft_stage_kernel launch")) != CUDA_SUCCESS_FFI) goto cleanup_loop; // Use cleanup_loop
        
        cuda_err = cudaDeviceSynchronize(); // Ensure stage is complete
        if ((ffi_err_code = check_cuda_error_ntt(cuda_err, "cudaDeviceSynchronize after stage kernel")) != CUDA_SUCCESS_FFI) goto cleanup_loop; // Use cleanup_loop
        
        cudaFree(d_twiddle_factors); 
        d_twiddle_factors = nullptr; 
    }

    // If inverse NTT, divide by N (modularly)
    if (inverse) {
        // Ensure PRIME is correctly defined and accessible for modular inverse.
        // Using PRIME directly if it's the modulus.
        uint64_t n_val = static_cast<uint64_t>(n); // Cast n to uint64_t for field_pow_op
        uint64_t n_inv = field_pow_op(n_val, PRIME - 2); // Ensure PRIME is the modulus
        
        // Kernel to multiply all elements by n_inv.
        // This requires a vector_scalar_field_mul_kernel.
        // For example:
        // vector_scalar_field_mul_kernel<<<blocks_per_grid, threads_per_block>>>(d_data, n_inv, n, PRIME);
        // This kernel needs to be defined and included. For now, this step is conceptual.
        // If such a kernel doesn't exist, this part needs implementation.
        // Placeholder:
        fprintf(stdout, "INFO (cuda_ntt_fft): Inverse NTT scalar multiplication by N_inv not fully implemented in this FFI version.\n");
    }

    // 6. Copy data back to host
    cuda_err = cudaMemcpy(h_data, d_data, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    // No ffi_err_code check here, if it fails, it will be caught by the final return or an earlier goto
    if (cuda_err != cudaSuccess) {
         check_cuda_error_ntt(cuda_err, "cudaMemcpy d_data to h_data"); // Log error
         // Decide on ffi_err_code if not already set by a goto
         if (ffi_err_code == CUDA_SUCCESS_FFI) ffi_err_code = CUDA_ERROR_FFI_INTERNAL; // Generic internal error
    }


cleanup_loop: // In case of error inside the loop, ensure d_twiddle_factors is handled if allocated
    if (d_twiddle_factors && ffi_err_code != CUDA_SUCCESS_FFI) { // Only free if error occurred AND it was allocated
         cudaFree(d_twiddle_factors); // d_twiddle_factors is set to nullptr after successful free in loop
         d_twiddle_factors = nullptr;
    }
cleanup:
    if (d_data) cudaFree(d_data);
    if (d_rev_indices) cudaFree(d_rev_indices);
    // d_twiddle_factors should be null here if loop completed or handled by cleanup_loop
    if (d_twiddle_factors) cudaFree(d_twiddle_factors); // Final safety net

    return ffi_err_code;
}
