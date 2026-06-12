#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <mma.h>
#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cassert>
#include "AsyncCopy_PTX.cuh"

typedef unsigned char uchar;

static const char *_cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#define checkKernelErrors(expr)                             \
do {                                                      \
expr;                                                   \
\
cudaError_t __err = cudaGetLastError();                 \
if (__err != cudaSuccess) {                             \
printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
cudaGetErrorString(__err));                    \
abort();                                              \
}                                                       \
} while (0)


#ifndef NDEBUG
#define AT_0(mat, row_dim, col_dim, row, col)                                        \
    (*({                                                                             \
        uint64_t _r = (row);                                                             \
        uint64_t _c = (col);                                                             \
        uint64_t _rd = (row_dim);                                                        \
        uint64_t _cd = (col_dim);                                                        \
        if (_r >= _rd) {             \
            assert(false && "Row Out of Bound");                                         \
        }                                                                            \
        if (_c >= _cd) {             \
            assert(false && "Col Out of Bound");                                         \
        }                                                                            \
        &(mat[(uint64_t)_r * (uint64_t) _cd + (uint64_t) _c]);                                                       \
    }))
#else
    #define AT_0(mat, row_dim, col_dim, row, col) ((mat)[(uint64_t)(row) *  (uint64_t)(col_dim) + (uint64_t)(col)]) // NOTE: this could hurt performance
#endif

#ifndef NDEBUG
#define AT_1(mat, row_dim, col_dim, row, col)                                        \
    (*({                                                                             \
        auto _r = (row);                                                             \
        auto _c = (col);                                                             \
        auto _rd = (row_dim);                                                        \
        auto _cd = (col_dim);                                                        \
        if (_r >= _rd) {             \
        assert(false && "Row Out of Bound");                                         \
        }                                                                            \
        if (_c >= _cd) {             \
        assert(false && "Col Out of Bound");                                         \
        }                                                                           \
        &(mat[_c * _rd + _r]);                                                       \
    }))
#else
    #define AT_1(mat, row_dim, col_dim, row, col) ((mat)[(col) * (row_dim) + (row)])
#endif

#define AT(major) CAT(AT_, major)

#define MAJOR_ROW 0
#define MAJOR_COL 1
#define CAT(x, y) x ## y

#define W_MAJOR MAJOR_COL

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))


// helper func for creating vector
// WARN: Do not use 0 as a seed!
__host__ __device__ uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}


__global__ void prepareW_map(
        char * W, // working memory
        unsigned short* const W_map,
        unsigned short* const W_map_negative,

        unsigned char* const W_map_delta2_d,
        unsigned char* const W_map_negative_delta2_d,

        unsigned short* const W_map_32_div,
        unsigned short* const W_map_negative_32_div,

        unsigned char* const W_map_delta2_div128,
        unsigned char* const W_map_negative_delta2_div128,

        int M, int K, int N, int S){
    u_int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= N){
        // this thread won't work for init
        return;
    }

    u_int64_t col = tid;

    for(int i = 0; i < K; i++){
        AT(W_MAJOR) (W, K, N, i, col) = 0;
    }
    for(int i = 0; i < S / 2; i++){
        AT(W_MAJOR) (W, K, N, i, col) = -1;
    }
    for(int i = S / 2; i < S; i++){
        AT(W_MAJOR) (W, K, N, i, col) = 1;
    }

    uint32_t seed = (uint32_t) (0xCAFEBABE ^ col);

    for(int i = K - 1; i > 0; --i){
        int j = xorshift32(&seed) % (i + 1);
        char tmp = AT(W_MAJOR) (W, K, N, i, col);
        AT(W_MAJOR) (W, K, N, i, col) = AT(W_MAJOR) (W, K, N, j, col);
        AT(W_MAJOR) (W, K, N, j, col) = tmp;
    }

    int count_1 = 0, count_m1 = 0;
    for(int i = 0; i < K; i++){
        if(AT(W_MAJOR) (W, K, N, i, col) == 1){
            AT(W_MAJOR) (W_map, S / 2, N, count_1++, col) = i;
        }else if (AT(W_MAJOR) (W, K, N, i, col) == -1){
            AT(W_MAJOR) (W_map_negative, S / 2, N, count_m1++, col) = i;
        }
    }

    assert(count_1 == S/ 2 && count_m1 == S / 2 && "W matrix corrupt");

    __syncthreads();

    for(int i = 0; i < S/2; i++){
        if(i==0){
            AT(W_MAJOR)(W_map_delta2_d, S / 2, N, i, col)
                = AT(W_MAJOR) (W_map, S / 2, N, i, col);
        }else{
            AT(W_MAJOR)(W_map_delta2_d, S / 2, N, i, col)
                = AT(W_MAJOR) (W_map, S / 2, N, i, col) - AT(W_MAJOR) (W_map, S / 2, N, i-1, col);
        }
    }

    for(int i = 0; i < S/2; i++){
        if(i==0){
            AT(W_MAJOR)(W_map_negative_delta2_d, S / 2, N, i, col)
                = AT(W_MAJOR) (W_map_negative, S / 2, N, i, col);
        }else{
            AT(W_MAJOR)(W_map_negative_delta2_d, S / 2, N, i, col)
                = AT(W_MAJOR) (W_map_negative, S / 2, N, i, col) - AT(W_MAJOR) (W_map_negative, S / 2, N, i-1, col);
        }
    }

    __syncthreads();

    assert(S % 64 == 0 && "Wrong S setting");
    // See matrix as [32*N, S/64]
    for(int i = tid * 32; i < tid * 32 + 32; i++){
        for(int j = 0; j < S / 64; j++){
            int original_row = j * 32 + i % 32;
            int original_col = i / 32;
            AT(MAJOR_COL)(W_map_32_div, 32 * N, S / 64, i, j) = AT(W_MAJOR) (W_map, S / 2, N, original_row, original_col);
            AT(MAJOR_COL)(W_map_negative_32_div, 32 * N, S / 64, i, j) = AT(W_MAJOR) (W_map_negative, S / 2, N, original_row, original_col);
        }
    }

    int div128_cols = CEIL_DIV( (S /2), 128 );
    // See matrix as [128*N, S/256]
    for(int i = tid * 128; i < tid * 128 + 128; i++){
        for(int j = 0; j < div128_cols; j++){
            int original_row = j * 128 + i % 128;
            int original_col = i / 128;

            if (original_row < S / 2){
                AT(MAJOR_COL)(W_map_delta2_div128, 128 * N, div128_cols, i, j) = AT(W_MAJOR) (W_map_delta2_d, S / 2, N, original_row, original_col);
                AT(MAJOR_COL)(W_map_negative_delta2_div128, 128 * N, div128_cols, i, j) = AT(W_MAJOR) (W_map_negative_delta2_d, S / 2, N, original_row, original_col);
            }
        }
    }
}


template <const uint M_, const uint K_, const uint N_, const uint S_, const uint ws_num, const uint SPLIT = 32>
__global__ void __launch_bounds__(32) rowWiseSplit3Small4(
        const char* __restrict__ const X,
        const unsigned short* __restrict__ const W_map,
        const unsigned short* __restrict__ const W_map_negative,
        __nv_bfloat16* __restrict__ const C,
        const __nv_bfloat16* __restrict__ const s,
        const __nv_bfloat16* __restrict__ const ws
        ){

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

    int ws_idx = c_col / (N_ / ws_num);
    if (threadIdx.x == 0) {
        C[m_row * N_ + c_col] = (__nv_bfloat16)(  ((float)accum) / (float)s[0]*(float)ws[ws_idx]  )  ;
    }
}

// blocksize can be 32, 64, 128 to improve occupancy
// Use Unroll
// Use staged buffer
template <const uint M_, const uint K_, const uint N_, const uint S_, const uint ws_num, const uint SPLIT = 32, const uint STAGES = 2, typename COPY_TYPE = int, const uint BLOCK_SIZE = 64>
__global__ void __launch_bounds__(BLOCK_SIZE) rowWiseSplit2DeltaSmallAsync2_6(
        const char* __restrict__ const X,
        const unsigned char* __restrict__ const W_map_delta,
        const unsigned char* __restrict__ const W_map_negative_delta,
        int* __restrict__ const C,
        const __nv_bfloat16* __restrict__ const s,
        const __nv_bfloat16* __restrict__ const ws
        ){

    __builtin_assume(threadIdx.x < 32);
    __builtin_assume(threadIdx.y < (BLOCK_SIZE / 32));

    static_assert(BLOCK_SIZE % 32 == 0, "Wrong BLOCK_SIZE");
    static_assert((S_ / 2) % SPLIT == 0, "Should be completely devidable");
    static constexpr int COLS_PER_BLOCK = BLOCK_SIZE / SPLIT; // 2
    static constexpr int W_COPY_BYTE_SIZE = sizeof(COPY_TYPE); // 4, div size is *32 of this!
    static constexpr int W_COPY_NUMS = BLOCK_SIZE * W_COPY_BYTE_SIZE / sizeof(unsigned char); // 256
    static constexpr int W_COPY_SKIP_NUM = W_COPY_BYTE_SIZE / sizeof(unsigned char); // 4
    static constexpr int W_COPY_NUMS_PER_WARP = 32 * W_COPY_BYTE_SIZE / sizeof(unsigned char); // 128

    const uint c_col = blockIdx.x * COLS_PER_BLOCK + threadIdx.y;
    const uint m_row = blockIdx.y;

    const uint lane_id = threadIdx.x + threadIdx.y * blockDim.x;
    __builtin_assume(lane_id < BLOCK_SIZE);

    // This makes less IMAD operation
    const unsigned char *W_map_delta_base = &W_map_delta[W_COPY_NUMS * blockIdx.x]; // [ S/256, 128*N ]
    const unsigned char *W_map_neg_delta_base = &W_map_negative_delta[W_COPY_NUMS * blockIdx.x];
    const char *X_base = &X[m_row * K_]; // must be row-major

    int accum = 0;

    static_assert((S_ / 2) % (W_COPY_NUMS / COLS_PER_BLOCK) == 0, "Should be completely devidable");
    static constexpr int BATCH_SIZE = (S_ / 2) / (W_COPY_NUMS / COLS_PER_BLOCK);
    static_assert(STAGES <= BATCH_SIZE, "Wrong STAGES size");
    __shared__ __align__(W_COPY_BYTE_SIZE) uchar w_pos_delta_buf[W_COPY_NUMS * STAGES]; // [[ 128,128 ] * STAGES]
    __shared__ __align__(W_COPY_BYTE_SIZE) uchar w_neg_delta_buf[W_COPY_NUMS * STAGES];

    uint j = 0;
    ushort next_iter_pos = 0;
    ushort next_iter_neg = 0;

#pragma unroll
    for(uint compute_batch = 0, fetch_batch = 0; compute_batch < BATCH_SIZE; compute_batch++){
#pragma unroll
        for(; fetch_batch < BATCH_SIZE && fetch_batch < (compute_batch + STAGES); fetch_batch++ ){
            uint i = (fetch_batch % STAGES) * W_COPY_NUMS + lane_id * W_COPY_SKIP_NUM;
            cp_async<W_COPY_BYTE_SIZE>(reinterpret_cast<COPY_TYPE *>(&w_pos_delta_buf[i]),
                                   reinterpret_cast<const COPY_TYPE *>(&W_map_delta_base[j + lane_id * W_COPY_SKIP_NUM]));
            cp_async<W_COPY_BYTE_SIZE>(reinterpret_cast<COPY_TYPE *>(&w_neg_delta_buf[i]),
                                   reinterpret_cast<const COPY_TYPE *>(&W_map_neg_delta_base[j + lane_id * W_COPY_SKIP_NUM]));
            cp_async_group_commit();
            j += W_COPY_NUMS_PER_WARP * N_;
        }

        cp_async_wait_group<STAGES - 1>();

        const uint consume_i = (compute_batch % STAGES) * W_COPY_NUMS + threadIdx.y * W_COPY_NUMS_PER_WARP;

#pragma unroll
        for(uint consume_j = threadIdx.x % 32; consume_j < W_COPY_NUMS_PER_WARP; consume_j+=32){
            ushort delta_pos = w_pos_delta_buf[consume_i+consume_j];
            ushort delta_neg = w_neg_delta_buf[consume_i+consume_j];

#pragma unroll
            for(uint offset = 1; offset < 32; offset*=2){
                uint32_t packed = __shfl_up_sync(0xffffffff, ((uint32_t)delta_neg << 16) | (uint32_t)delta_pos, offset);

                delta_pos += (threadIdx.x >= offset) ? (ushort)(packed & 0xffff) : 0;
                delta_neg += (threadIdx.x >= offset) ? (ushort)((packed >> 16) & 0xffff) : 0;
            }

            delta_pos += next_iter_pos;
            delta_neg += next_iter_neg;

            next_iter_pos = __shfl_sync(0xffffffff, delta_pos, 31);
            next_iter_neg = __shfl_sync(0xffffffff, delta_neg, 31);

            accum += X_base[delta_pos];
            accum -= X_base[delta_neg];
        }
    }

    accum = __reduce_add_sync(0xffffffff, accum);

    int ws_idx = c_col / (N_ / ws_num);
    if (threadIdx.x == 0) {
        C[m_row * N_ + c_col] = (__nv_bfloat16)(  ((float)accum) / (float)s[0]*(float)ws[ws_idx]  ) ;
    }
}


// Use staged buffer
// suoppot when (S_ / 2) % W_COPY_NUMS != 0
template <const uint M_, const uint K_, const uint N_, const uint S_, const uint ws_num, const uint SPLIT = 32, const uint STAGES = 2, typename COPY_TYPE = int, const uint BLOCK_SIZE = 64>
__global__ void __launch_bounds__(BLOCK_SIZE) rowWiseSplit2DeltaSmallAsync2_7(
        const char* __restrict__ const X,
        const unsigned char* __restrict__ const W_map_delta,
        const unsigned char* __restrict__ const W_map_negative_delta,
        int* __restrict__ const C,
        const __nv_bfloat16* __restrict__ const s,
        const __nv_bfloat16* __restrict__ const ws
        ){

    __builtin_assume(threadIdx.x < 32);
    __builtin_assume(threadIdx.y < (BLOCK_SIZE / 32));

    static_assert(BLOCK_SIZE % 32 == 0, "Wrong BLOCK_SIZE");
    static_assert((S_ / 2) % SPLIT == 0, "Should be completely devidable");
    static constexpr int COLS_PER_BLOCK = BLOCK_SIZE / SPLIT; // 2
    static constexpr int W_COPY_BYTE_SIZE = sizeof(COPY_TYPE); // 4, div size is *32 of this!
    static constexpr int W_COPY_NUMS = BLOCK_SIZE * W_COPY_BYTE_SIZE / sizeof(unsigned char); // 256
    static constexpr int W_COPY_SKIP_NUM = W_COPY_BYTE_SIZE / sizeof(unsigned char); // 4
    static constexpr int W_COPY_NUMS_PER_WARP = 32 * W_COPY_BYTE_SIZE / sizeof(unsigned char); // 128

    const uint c_col = blockIdx.x * COLS_PER_BLOCK + threadIdx.y;
    const uint m_row = blockIdx.y;

    const uint lane_id = threadIdx.x + threadIdx.y * blockDim.x;
    __builtin_assume(lane_id < BLOCK_SIZE);

    // This makes less IMAD operation
    const unsigned char *W_map_delta_base = &W_map_delta[W_COPY_NUMS * blockIdx.x]; // [ S/256, 128*N ]
    const unsigned char *W_map_neg_delta_base = &W_map_negative_delta[W_COPY_NUMS * blockIdx.x];
    const char *X_base = &X[m_row * K_]; // must be row-major

    int accum = 0;

    // suppot when (S_ / 2) % W_COPY_NUMS != 0
    // static_assert((S_ / 2) % (W_COPY_NUMS / COLS_PER_BLOCK) == 0, "Should be completely devidable");
    static constexpr int BATCH_SIZE = CEIL_DIV((S_ / 2) , (W_COPY_NUMS / COLS_PER_BLOCK));
    static_assert(STAGES <= BATCH_SIZE, "Wrong STAGES size");
    __shared__ __align__(W_COPY_BYTE_SIZE) uchar w_pos_delta_buf[W_COPY_NUMS * STAGES]; // [[ 128,128 ] * STAGES]
    __shared__ __align__(W_COPY_BYTE_SIZE) uchar w_neg_delta_buf[W_COPY_NUMS * STAGES];

    uint j = 0;
    ushort next_iter_pos = 0;
    ushort next_iter_neg = 0;

#pragma unroll
    for(uint compute_batch = 0, fetch_batch = 0; compute_batch < BATCH_SIZE; compute_batch++){
        // we allow to copy unused area, just ignore at calc phase
#pragma unroll
        for(; fetch_batch < BATCH_SIZE && fetch_batch < (compute_batch + STAGES); fetch_batch++ ){
            uint i = (fetch_batch % STAGES) * W_COPY_NUMS + lane_id * W_COPY_SKIP_NUM;
            cp_async<W_COPY_BYTE_SIZE>(reinterpret_cast<COPY_TYPE *>(&w_pos_delta_buf[i]),
                                   reinterpret_cast<const COPY_TYPE *>(&W_map_delta_base[j + lane_id * W_COPY_SKIP_NUM]));
            cp_async<W_COPY_BYTE_SIZE>(reinterpret_cast<COPY_TYPE *>(&w_neg_delta_buf[i]),
                                   reinterpret_cast<const COPY_TYPE *>(&W_map_neg_delta_base[j + lane_id * W_COPY_SKIP_NUM]));
            cp_async_group_commit();
            j += W_COPY_NUMS_PER_WARP * N_;
        }

        cp_async_wait_group<STAGES - 1>();

        const uint consume_i = (compute_batch % STAGES) * W_COPY_NUMS + threadIdx.y * W_COPY_NUMS_PER_WARP;

#pragma unroll
        for(uint consume_j = threadIdx.x % 32; consume_j < W_COPY_NUMS_PER_WARP; consume_j+=32){
            if((W_COPY_NUMS_PER_WARP * compute_batch + consume_j) < S_ / 2 ){
                ushort delta_pos = w_pos_delta_buf[consume_i+consume_j];
                ushort delta_neg = w_neg_delta_buf[consume_i+consume_j];

#pragma unroll
                for(uint offset = 1; offset < 32; offset*=2){
                    uint32_t packed = __shfl_up_sync(0xffffffff, ((uint32_t)delta_neg << 16) | (uint32_t)delta_pos, offset);

                    delta_pos += (threadIdx.x >= offset) ? (ushort)(packed & 0xffff) : 0;
                    delta_neg += (threadIdx.x >= offset) ? (ushort)((packed >> 16) & 0xffff) : 0;
                }

                delta_pos += next_iter_pos;
                delta_neg += next_iter_neg;

                next_iter_pos = __shfl_sync(0xffffffff, delta_pos, 31);
                next_iter_neg = __shfl_sync(0xffffffff, delta_neg, 31);

                accum += X_base[delta_pos];
                accum -= X_base[delta_neg];
            }
        }
    }

    accum = __reduce_add_sync(0xffffffff, accum);

    int ws_idx = c_col / (N_ / ws_num);
    if (threadIdx.x == 0) {
        C[m_row * N_ + c_col] = (__nv_bfloat16)(  ((float)accum) / (float)s[0]*(float)ws[ws_idx]  ) ;
    }
}