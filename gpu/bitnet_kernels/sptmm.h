#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <mma.h>
#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <const uint M_, const uint K_, const uint N_, const uint S_, const uint SPLIT = 32>
__global__ void __launch_bounds__(32) rowWiseSplit3Small4(
        const char* __restrict__ const X,
        const unsigned short* __restrict__ const W_map,
        const unsigned short* __restrict__ const W_map_negative,
        int* __restrict__ const C){

    static constexpr int COLS_PER_WARP = 32 / SPLIT; // 1, 4
    static_assert((S_ / 2) % (SPLIT) == 0, "Wrong SPLIT Size");

    const uint c_col = blockIdx.x * COLS_PER_WARP + threadIdx.y;
    const uint m_row = blockIdx.y;
    // assume size of thread block =< 32
    const int lane_id = threadIdx.x
                        + threadIdx.y * blockDim.x
                        + threadIdx.z * blockDim.x * blockDim.y;

    const ushort *W_map_base = &W_map[32 * blockIdx.x];
    const ushort *W_map_neg_base = &W_map_negative[32 * blockIdx.x];
    const char *X_base = &X[m_row * K_]; // must be row-major

    int accum = 0;

    static constexpr int UNROLL_FACTOR = ((S_ / 2) / SPLIT);
#pragma unroll UNROLL_FACTOR
    for(uint i = lane_id; i < ((S_ / 2) / SPLIT) * SPLIT * N_ ; i+= SPLIT * N_){
        accum += __ldg(&X_base[W_map_base[i]]);
        accum -= __ldg(&X_base[W_map_neg_base[i]]);
    }

#pragma unroll
    for(int i = SPLIT / 2; i > 0; i /= 2){
        accum += __shfl_down_sync(0xffffffff, accum, i, SPLIT);
    }

    if (threadIdx.x == 0) {
        __stwt(&C[m_row * N_ + c_col], accum);
    }
}