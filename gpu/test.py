import torch
from torch.utils import benchmark
from torch import nn
from tqdm import tqdm

from pack_weight import convert_weight_int8_to_int2
from torch.profiler import profile, record_function, ProfilerActivity
import ctypes
import numpy as np
# set all seed
torch.manual_seed(42)
np.random.seed(42)

bitnet_lib = ctypes.CDLL('bitnet_kernels/libbitnet.so')

def sptmm(x, w_map_32_div, w_map_neg_32_div, ret):
    stream = torch.cuda.current_stream()

    M = 1
    K = 6912
    N = 2560
    S = 4096

    bitnet_lib.sptmm(*[ctypes.c_void_p(x.data_ptr()),
                       ctypes.c_void_p(w_map_32_div.data_ptr()),
                       ctypes.c_void_p(w_map_neg_32_div.data_ptr()),
                       ctypes.c_void_p(ret.data_ptr()),
                       ctypes.c_int(M),
                       ctypes.c_int(K),
                       ctypes.c_int(N),
                       ctypes.c_int(S),
                       ctypes.c_void_p(stream.cuda_stream)])
    return ret


def bitnet_int8xint2_linear(input0, input1, s, ws, ret):
    out_shape = list(input0.shape)
    out_shape[-1] = input1.shape[0]

    stream = torch.cuda.current_stream()

    M = input0.shape[0]
    if len(out_shape) == 3: 
        M *= input0.shape[1]
    N = input1.shape[0]
    K = input1.shape[1] * 4

    bitnet_lib.bitlinear_int8xint2(*[ctypes.c_void_p(input0.data_ptr()), ctypes.c_void_p(input1.data_ptr()), ctypes.c_void_p(ret.data_ptr()), ctypes.c_void_p(s.data_ptr()), ctypes.c_void_p(ws.data_ptr()), ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_void_p(stream.cuda_stream)])

    return ret


def xorshift32(seed):
    x = seed
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17) & 0xFFFFFFFF
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF



def prepare_w_map_fast(m, k, n, s):
    stream = torch.cuda.current_stream()
    w = torch.zeros((k, n), dtype=torch.int8, device='cuda')
    w_map = torch.zeros((s // 2, n), dtype=torch.int16 ,device='cuda')
    w_map_negative = torch.zeros((s // 2, n), dtype=torch.int16, device='cuda')
    def alloc_div(div_nim):
        rows = div_nim * n
        cols = s // 2 // div_nim
        return torch.zeros((rows, cols), dtype=torch.int16, device='cuda')
    w_map_32_div = alloc_div(32)
    w_map_negative_32_div = alloc_div(32)

    bitnet_lib.prepare_w_map(*[
        ctypes.c_void_p(w.data_ptr()),
        ctypes.c_void_p(w_map.data_ptr()),
        ctypes.c_void_p(w_map_negative.data_ptr()),
        ctypes.c_void_p(w_map_32_div.data_ptr()),
        ctypes.c_void_p(w_map_negative_32_div.data_ptr()),
        ctypes.c_int(m),
        ctypes.c_int(k),
        ctypes.c_int(n),
        ctypes.c_int(s),
        ctypes.c_void_p(stream.cuda_stream)])

    return w_map_32_div, w_map_negative_32_div

# Obsolete and slow
def prepare_w_map(m, k, n, s):
    w = torch.zeros((k, n), dtype=torch.int8)
    w_map = torch.zeros((s // 2, n), dtype=torch.int16)
    w_map_negative = torch.zeros((s // 2, n), dtype=torch.int16)

    def alloc_div(div_nim):
        rows = div_nim * n
        cols = s // 2 // div_nim
        return torch.zeros((rows, cols), dtype=torch.int16)

    w_map_32_div = alloc_div(32)
    w_map_negative_32_div = alloc_div(32)

    for col in tqdm(range(n)):
        w[:, col] = 0
        w[:s//2, col] = -1
        w[s//2:s, col] = 1
        seed = (0xCAFEBABE ^ col) & 0xFFFFFFFF
        for i in range(k-1, 0, -1):
            j = xorshift32(seed) % (i+1)
            tmp = w[i, col].item()
            w[i,col] = w[j,col]
            w[j,col] = tmp

        count_1 = 0
        count_m1 = 0
        for i in range(k):
            val = w[i,col].item()
            if val == 1:
                w_map[count_1, col] = i
                count_1 += 1
            elif val == -1:
                w_map_negative[count_m1, col] = i
                count_m1 += 1

        assert count_1 == s // 2
        assert count_m1 == s // 2

        # div convert
        assert s % 64 == 0
        for i in range(col*32, col*32+32, 1):
            for j in range(0, s // 64, 1):
                original_row = j * 32 + i % 32
                original_col = i // 32
                w_map_32_div[i,j] = w_map[original_row, original_col]
                w_map_negative_32_div[i,j] = w_map_negative[original_row, original_col]

    # don't forget to convert to column major since pytorch is row major!
    w_map_32_div = w_map_32_div.t().contiguous().t().to('cuda')
    w_map_negative_32_div = w_map_negative_32_div.t().contiguous().t().to('cuda')
    return w_map_32_div, w_map_negative_32_div


if __name__ == '__main__':
    test_list = [
        (2560, 6912), # N, K
    ]
    for N,K in test_list:
        weight = torch.randint(-1, 2, (N, K), dtype=torch.int8, device='cuda')
        weight_scale = torch.ones(1, dtype=torch.bfloat16, device='cuda')
        weight_compressed = convert_weight_int8_to_int2(weight).to('cuda')

        w_map_32_div, w_map_negative_32_div = prepare_w_map_fast(1, K, N, 4096)

        for i in range(1):
            input0 = torch.randint(-128,127,(1, K),dtype=torch.int8, device='cuda')
            input0_bf16 = input0.to(torch.bfloat16)
            input_np = input0.cpu().to(torch.int32).numpy()
            weight_np = weight.cpu().to(torch.int32).T.numpy()
            out_np = np.matmul(input_np,weight_np)
            out_np = torch.tensor(out_np).cuda().to(torch.bfloat16)

            s = torch.ones(1, dtype=torch.bfloat16, device='cuda')
            ws = torch.ones(6, dtype=torch.bfloat16, device='cuda')

            ret = torch.empty((1,N), dtype=torch.bfloat16, device=input0.device)
            out = bitnet_int8xint2_linear(input0, weight_compressed, s, ws, ret)

            print(f'custom == np {torch.all(out==out_np)}')

        input0 = torch.randint(-128,127,(1, K),dtype=torch.int8, device='cuda')
        input0_fp16 = input0.to(torch.float16)
        input0_bf16 = input0.to(torch.bfloat16)
        weight_fp16 = weight.to(torch.float16).T
        weight_bf16 = weight.to(torch.bfloat16).T
        ret = torch.empty((1,N), dtype=torch.bfloat16, device=input0.device)
        s = torch.ones(1, dtype=torch.bfloat16, device='cuda')
        ws = torch.ones(6, dtype=torch.bfloat16, device='cuda')
        t0 = benchmark.Timer(
            stmt="bitnet_int8xint2_linear(input0, weight_compressed, s, ws, ret)",
            setup="from __main__ import input0, weight_compressed, s, ws, ret, bitnet_int8xint2_linear",
            num_threads=1,
        )

        t1 = benchmark.Timer(
            stmt="torch.matmul(input0_bf16,weight_bf16)",
            setup="from __main__ import input0_bf16, weight_bf16",
            num_threads=1,
        )

        t2 = benchmark.Timer(
            stmt="sptmm(input0, w_map_32_div, w_map_negative_32_div, ret)",
            setup="from __main__ import input0, w_map_32_div, w_map_negative_32_div, ret, sptmm",
            num_threads=1,
        )


        time0 = t0.timeit(50)
        time1 = t1.timeit(50)
        time2 = t2.timeit(50)

        print(f'Shape{N,K}, W2A8: {time0.mean * 1e6:.2f}us, torch BF16: {time1.mean * 1e6:.2f}us, SpTMM: {time2.mean * 1e6:.2f}us')

        
