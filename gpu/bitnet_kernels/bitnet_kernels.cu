#include "bitnet_kernels.h"
#include "sptmm.h"

extern "C" void bitlinear_int8xint2(int8_t* input0, int8_t* input1, __nv_bfloat16* output0, __nv_bfloat16* s, __nv_bfloat16* ws, int M, int N, int K, cudaStream_t stream){
    if (M == 1 && N == 3840 && K == 2560){
        ladder_int8xint2_kernel<1, 3840, 2560, 3, 8, 16><<<dim3(240, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if (M == 1 && N == 2560 && K == 2560){
        ladder_int8xint2_kernel<1, 2560, 2560, 1, 8, 16><<<dim3(160, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if (M == 1 && N == 13824 && K == 2560){
        ladder_int8xint2_kernel<1, 13824, 2560, 2, 8, 16><<<dim3(864, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if (M == 1 && N == 2560 && K == 6912){
        ladder_int8xint2_kernel<1, 2560, 6912, 1, 8, 16><<<dim3(160, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1 && N == 4800 && K == 3200){
        ladder_int8xint2_kernel<1, 4800, 3200, 6, 8, 16><<<dim3(300, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1 && N == 3200 && K == 3200){
        ladder_int8xint2_kernel<1, 3200, 3200, 1, 8, 16><<<dim3(200, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1 && N == 20480 && K == 3200){
        ladder_int8xint2_kernel<1, 20480, 3200, 2, 8, 16><<<dim3(1280, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1 && N == 3200 && K == 10240){
        ladder_int8xint2_kernel<1, 3200, 10240, 1, 8, 16><<<dim3(200, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }    
    else if(M == 1 && N == 5120 && K == 27648){
        ladder_int8xint2_kernel<1, 5120, 27648, 1, 8, 16><<<dim3(320, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1 && N == 55296 && K == 5120){
        ladder_int8xint2_kernel<1, 55296, 5120, 1, 8, 16><<<dim3(3456, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else{
        std::cout << "required ladder gemm kernel: M " << M << ", N " << N << ", K " << K << std::endl;
    }
}

extern "C" void prepare_w_map(char* W_d, unsigned short* W_map_d, unsigned short* W_map_negative_d, unsigned short* W_map_32_div_d, unsigned short* W_map_negative_32_div_d, int M, int K, int N, int S, cudaStream_t stream){
    prepareW_map<<<N/16, 16, 0, stream>>>(W_d, W_map_d, W_map_negative_d, W_map_32_div_d, W_map_negative_32_div_d, M, K, N, S);
}

extern "C" void sptmm(char* X_d, unsigned short* W_map_32_div_d, unsigned short* W_map_negative_32_div_d, int* c_d, int M, int K, int N, int S, cudaStream_t stream){
    if(M == 1 && K == 6912 && N == 2560 && S == 5120){
        rowWiseSplit3Small4<1, 6912, 2560, 5120, 32><<< 2560 / 1, dim3(32,1,1), 0, stream >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d);
    }else if(M == 1 && K == 6912 && N == 2560 && S == 5504){
        rowWiseSplit3Small4<1, 6912, 2560, 5504, 32><<< 2560 / 1, dim3(32,1,1), 0, stream >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d);
    }else if(M == 1 && K == 6912 && N == 2560 && S == 4608){
        rowWiseSplit3Small4<1, 6912, 2560, 4608, 32><<< 2560 / 1, dim3(32,1,1), 0, stream >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d);
    }else if(M == 1 && K == 6912 && N == 2560 && S == 4160){
        rowWiseSplit3Small4<1, 6912, 2560, 4160, 32><<< 2560 / 1, dim3(32,1,1), 0, stream >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d);
    }else if(M == 1 && K == 6912 && N == 2560 && S == 4096){
        rowWiseSplit3Small4<1, 6912, 2560, 4096, 32><<< 2560 / 1, dim3(32,1,1), 0, stream >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d);
    }else if(M == 1 && K == 6912 && N == 2560 && S == 3072){
        rowWiseSplit3Small4<1, 6912, 2560, 3072, 32><<< 2560 / 1, dim3(32,1,1), 0, stream >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d);
    }else if(M == 1 && K == 6912 && N == 2560 && S == 2752){
        rowWiseSplit3Small4<1, 6912, 2560, 2752, 32><<< 2560 / 1, dim3(32,1,1), 0, stream >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d);
    }else if(M == 1 && K == 6912 && N == 2560 && S == 2048){
        rowWiseSplit3Small4<1, 6912, 2560, 2048, 32><<< 2560 / 1, dim3(32,1,1), 0, stream >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d);
    }else if(M == 1 && K == 6912 && N == 2560 && S == 1024) {
        rowWiseSplit3Small4< 1, 6912, 2560, 1024, 32 ><<< 2560 / 1, dim3(32, 1, 1), 0, stream >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d);

    }else if(M == 1 && K == 2560 && N == 13824 && S == 1536){
        rowWiseSplit3Small4 < 1, 2560, 13824, 1536, 32 ><<< dim3(13824, 1, 1), dim3(32, 1, 1), 0, stream >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d);
    }else if(M == 1 && K == 2560 && N == 2560 && S == 1536) {
        rowWiseSplit3Small4 < 1, 2560, 2560, 1536, 32 ><<< dim3(2560, 1, 1), dim3(32, 1, 1), 0, stream >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d);
    }else if(M == 1 && K == 2560 && N == 3840 && S == 1536) {
        checkKernelErrors((rowWiseSplit3Small4 < 1, 2560, 3840, 1536, 32 ><<< dim3(3840, 1, 1), dim3(32, 1, 1), 0, stream >>>(X_d, W_map_32_div_d,W_map_negative_32_div_d, c_d);
    }

    else{
        std::cout << "required ladder gemm kernel: M " << M << ", N " << N << ", K " << K << std::endl;
    }
}