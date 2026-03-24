import model
import torch
import tqdm
import ctypes

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

    return w_map_32_div, w_map_negative_32_div


def col2row(w):
    return w.t().contiguous()

if __name__ == "__main__":
    # we use
    m = model.Transformer(model.ModelArgs(use_kernel=False, use_sptmm=True))

    with torch.no_grad():
        for i in range(30):
            wqkv, wqkv_neg = prepare_w_map_fast(1,2560,3840,1536)
            m.layers[i].attention.wqkv.w_map_32_div.copy_(wqkv)
            m.layers[i].attention.wqkv.w_map_negative_32_div.copy_(wqkv_neg)

            wo, wo_neg = prepare_w_map_fast(1,2560,2560,1536)
            m.layers[i].attention.wo.w_map_32_div.copy_(wo)
            m.layers[i].attention.wo.w_map_negative_32_div.copy_(wo_neg)

            w13, w13_neg = prepare_w_map_fast(1,2560,13824,1536)
            m.layers[i].feed_forward.w13.w_map_32_div.copy_(w13)
            m.layers[i].feed_forward.w13.w_map_negative_32_div.copy_(w13_neg)

            w2, w2_neg = prepare_w_map_fast(1,6912,2560,4096)
            m.layers[i].feed_forward.w2.w_map_32_div.copy_(w2)
            m.layers[i].feed_forward.w2.w_map_negative_32_div.copy_(w2_neg)

        torch.save(m.state_dict(), "sptmm.pt")