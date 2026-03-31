import model
import torch
import tqdm
import ctypes
from pack_weight import convert_weight_int8_to_int2

bitnet_lib = ctypes.CDLL('bitnet_kernels/libbitnet.so')

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

    return w, w_map_32_div, w_map_negative_32_div


def col2row(w):
    return w.t().contiguous()

if __name__ == "__main__":
    # we use
    m = model.Transformer(model.ModelArgs(use_kernel=True, use_sptmm=True))

    with torch.no_grad():
        for i in range(30):
            wqkv, _,  _ = prepare_w_map_fast(1,2560,3840,1536)
            wqkv_int2 = convert_weight_int8_to_int2(col2row(wqkv))
            m.layers[i].attention.wqkv.weight.copy_(wqkv_int2)

            wo, _, _ = prepare_w_map_fast(1,2560,2560,1536)
            wo_int2 = convert_weight_int8_to_int2(col2row(wo))
            m.layers[i].attention.wo.weight.copy_(wo_int2)

            w13, _, _ = prepare_w_map_fast(1,2560,13824,1536)
            w13_int2 = convert_weight_int8_to_int2(col2row(w13))
            m.layers[i].feed_forward.w13.weight.copy_(w13_int2)

            _, w2_pos, w2_neg = prepare_w_map_fast(1,6912,2560,4096)
            m.layers[i].feed_forward.w2.w_map_32_div.copy_(w2_pos)
            m.layers[i].feed_forward.w2.w_map_negative_32_div.copy_(w2_neg)

        torch.save(m.state_dict(), "partial_sptmm.pt")