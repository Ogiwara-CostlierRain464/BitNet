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

    return w


def col2row(w):
    return w.t().contiguous()

if __name__ == "__main__":
    # we use
    m = model.Transformer(model.ModelArgs(use_kernel=False))
    
    with torch.no_grad():
        for i in range(30):
            wqkv = prepare_w_map_fast(1,2560,3840,1536)
            m.layers[i].attention.wqkv.weight.copy_(col2row(wqkv).to(torch.bfloat16))
            wo = prepare_w_map_fast(1,2560,2560,1536)
            m.layers[i].attention.wo.weight.copy_(col2row(wo).to(torch.bfloat16))
            w13 = prepare_w_map_fast(1,2560,13824,1536)
            m.layers[i].feed_forward.w13.weight.copy_(col2row(w13).to(torch.bfloat16))
            w2 = prepare_w_map_fast(1,6912,2560,1536)
            m.layers[i].feed_forward.w2.weight.copy_(col2row(w2).to(torch.bfloat16))

        torch.save(m.state_dict(), "test.pt")