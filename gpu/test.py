import torch
from torch.utils import benchmark
from torch import nn
from tqdm import tqdm

from pack_weight import convert_weight_int8_to_int2
from torch.profiler import profile, record_function, ProfilerActivity
import ctypes
import numpy as np
import argparse
import pdb
# set all seed
torch.manual_seed(42)
np.random.seed(42)

bitnet_lib = ctypes.CDLL('bitnet_kernels/libbitnet.so')

def sptmm(x, w_map_32_div, w_map_neg_32_div, s, ws, ret, M, K, N, S):
    stream = torch.cuda.current_stream()

    bitnet_lib.sptmm(*[ctypes.c_void_p(x.data_ptr()),
                       ctypes.c_void_p(w_map_32_div.data_ptr()),
                       ctypes.c_void_p(w_map_neg_32_div.data_ptr()),
                       ctypes.c_void_p(ret.data_ptr()),
                       ctypes.c_void_p(s.data_ptr()),
                       ctypes.c_void_p(ws.data_ptr()),
                       ctypes.c_int(M),
                       ctypes.c_int(K),
                       ctypes.c_int(N),
                       ctypes.c_int(S),
                       ctypes.c_void_p(stream.cuda_stream)])
    return ret


def sptmm_delta(x, w_map_delta2_div128, w_map_negative_delta2_div128, s, ws, ret, M, K, N, S):
    stream = torch.cuda.current_stream()

    bitnet_lib.sptmm_delta(*[ctypes.c_void_p(x.data_ptr()),
                       ctypes.c_void_p(w_map_delta2_div128.data_ptr()),
                       ctypes.c_void_p(w_map_negative_delta2_div128.data_ptr()),
                       ctypes.c_void_p(ret.data_ptr()),
                       ctypes.c_void_p(s.data_ptr()),
                       ctypes.c_void_p(ws.data_ptr()),
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
    def alloc_div_16bit(div_nim):
        rows = div_nim * n
        cols = s // 2 // div_nim
        return torch.zeros((rows, cols), dtype=torch.int16, device='cuda')

    def alloc_div_8bit(div_nim):
        rows = div_nim * n
        cols = s // 2 // div_nim
        return torch.zeros((rows, cols), dtype=torch.int8, device='cuda')


    W_map_delta2_d = torch.zeros((s // 2, n), dtype=torch.int8 ,device='cuda')
    W_map_negative_delta2_d = torch.zeros((s // 2, n), dtype=torch.int8, device='cuda')

    w_map_32_div = alloc_div_16bit(32)
    w_map_negative_32_div = alloc_div_16bit(32)

    W_map_delta2_div128 = alloc_div_8bit(128)
    W_map_negative_delta2_div128 = alloc_div_8bit(128)


    bitnet_lib.prepare_w_map(*[
        ctypes.c_void_p(w.data_ptr()),
        ctypes.c_void_p(w_map.data_ptr()),
        ctypes.c_void_p(w_map_negative.data_ptr()),
        ctypes.c_void_p(W_map_delta2_d.data_ptr()),
        ctypes.c_void_p(W_map_negative_delta2_d.data_ptr()),
        ctypes.c_void_p(w_map_32_div.data_ptr()),
        ctypes.c_void_p(w_map_negative_32_div.data_ptr()),
        ctypes.c_void_p(W_map_delta2_div128.data_ptr()),
        ctypes.c_void_p(W_map_negative_delta2_div128.data_ptr()),
        ctypes.c_int(m),
        ctypes.c_int(k),
        ctypes.c_int(n),
        ctypes.c_int(s),
        ctypes.c_void_p(stream.cuda_stream)])

    return w, w_map, w_map_negative, w_map_32_div, w_map_negative_32_div, W_map_delta2_div128, W_map_negative_delta2_div128


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--matmul_test',
        action='store_true',
        help='Verify SpTMM output correctness'
    )
    args = parser.parse_args()

    test_list = [
        (2560, 6912, [5504, 4096, 2752, 1408]), # N, K  # Down
        (13824, 2560, [2048, 1536, 1024, 512]), # Up & Gate
        (2560, 2560, [2048, 1536, 1024, 512]), # Output
        (3840, 2560, [2048, 1536, 1024, 512]), # QKV
    ]
    for N,K, s_list in test_list:
        weight = torch.randint(-1, 2, (N, K), dtype=torch.int8, device='cuda')
        weight_scale = torch.ones(1, dtype=torch.bfloat16, device='cuda')
        weight_compressed = convert_weight_int8_to_int2(weight).to('cuda')

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
        ret_sptmm = torch.empty((1,N), dtype=torch.bfloat16, device=input0.device)
        ret_sptmm_delta = torch.empty((1,N), dtype=torch.bfloat16, device=input0.device)
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

        time0 = t0.blocked_autorange()
        time1 = t1.blocked_autorange()

        print(f'Shape{N,K}, W2A8: {time0.median * 1e6:.2f}us, torch BF16: {time1.median * 1e6:.2f}us')

        for sparsity in s_list:
            w_original, w_map, w_map_negative, w_map_32_div, w_map_negative_32_div, W_map_delta2_div128, W_map_negative_delta2_div128 = prepare_w_map_fast(1, K, N, sparsity)

            if args.matmul_test:
                # don't forget to convert to column major since pytorch is row major!
                w_original = w_original.t().contiguous().t().to('cuda')
                w_original_bf16 = w_original.to(torch.bfloat16)
                expected = torch.matmul(input0_bf16,w_original_bf16)

                sptmm(input0, w_map_32_div, w_map_negative_32_div, s, ws, ret_sptmm, 1,K, N, sparsity)
                sptmm_success = torch.all(ret_sptmm == expected)

                sptmm_delta(input0, W_map_delta2_div128, W_map_negative_delta2_div128, s, ws, ret_sptmm_delta, 1,K, N, sparsity)
                sptmm_delta_success = torch.all(ret_sptmm_delta == expected)

                if not torch.equal(ret_sptmm, expected):
                    print("Mismatch detected!")
                    pdb.set_trace()

                print(
                    f"Sparsity {sparsity}%: "
                    f"SpTMM == numpy {sptmm_success}, "
                    f"Delta == numpy {sptmm_delta_success}"
                )

            else:
                t2 = benchmark.Timer(
                    stmt="sptmm(input0, w_map_32_div, w_map_negative_32_div, s, ws, ret_sptmm, 1,K, N, sparsity)",
                    setup="from __main__ import input0, w_map_32_div, w_map_negative_32_div, s, ws, ret_sptmm, sptmm, N, K, sparsity",
                    num_threads=1,
                )
                t3 = benchmark.Timer(
                    stmt="sptmm_delta(input0, W_map_delta2_div128, W_map_negative_delta2_div128, s, ws, ret_sptmm_delta, 1,K, N, sparsity)",
                    setup="from __main__ import input0, W_map_delta2_div128, W_map_negative_delta2_div128, s, ws, ret_sptmm_delta, sptmm_delta, N, K, sparsity",
                    num_threads=1,
                )

                time2 = t2.blocked_autorange()
                time3 = t3.blocked_autorange()
                print(f'SpTMM with {sparsity}% sparsity : {time2.median * 1e6:.2f}us, delta {time3.median * 1e6:.2f}us')

        
