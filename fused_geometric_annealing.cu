#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void fused_entropy_temperature_kernel(
    float* __restrict__ logits,
    const float* __restrict__ prev_entropies,
    int* __restrict__ patience_counts,
    float* __restrict__ output_entropies,
    const int batch_size,
    const int vocab_size,
    const float base_temp,
    const float beta,
    const int patience_cap,
    const float entropy_floor,
    const float marginal_threshold
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    float* batch_logits = logits + batch_idx * vocab_size;
    int tid = threadIdx.x;

    float local_max = -1e20f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        local_max = fmaxf(local_max, batch_logits[i]);
    }
    local_max = warp_reduce_max(local_max);

    __shared__ float s_max;
    if (tid == 0) s_max = local_max;
    __syncthreads();
    float max_val = s_max;

    float local_sum = 0.0f;
    float local_entropy_num = 0.0f;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float p = expf(batch_logits[i] - max_val);
        local_sum += p;
        local_entropy_num += (p > 1e-12f) ? (p * logf(p)) : 0.0f;
    }

    local_sum = warp_reduce_sum(local_sum);
    local_entropy_num = warp_reduce_sum(local_entropy_num);

    __shared__ float s_sum, s_entropy_num;
    if (tid == 0) {
        s_sum = local_sum;
        s_entropy_num = local_entropy_num;
    }
    __syncthreads();

    float total_sum = s_sum;
    float current_entropy = logf(total_sum) - (s_entropy_num / total_sum);

    __shared__ float active_temp;
    if (tid == 0) {
        output_entropies[batch_idx] = current_entropy;
        float prev_h = prev_entropies[batch_idx];
        float delta_h = prev_h - current_entropy;
        int p_count = patience_counts[batch_idx];

        if (delta_h < marginal_threshold && current_entropy > entropy_floor) {
            if (p_count < patience_cap) {
                p_count += 1;
                active_temp = base_temp * powf(1.0f + beta, (float)p_count);
            } else {
                active_temp = base_temp;
            }
        } else {
            p_count = 0;
            active_temp = base_temp;
        }
        patience_counts[batch_idx] = p_count;
    }
    __syncthreads();

    float inv_temp = 1.0f / fmaxf(1e-5f, active_temp);
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        batch_logits[i] *= inv_temp;
    }
}
