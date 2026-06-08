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
    m_40 = model.Transformer(model.ModelArgs(use_kernel=False, use_sptmm=True, sparsity=40))

    with torch.no_grad():
        for i in range(30):
            wqkv, wqkv_neg = prepare_w_map_fast(1,2560,3840,1536)
            m_40.layers[i].attention.wqkv.w_map_32_div.copy_(wqkv)
            m_40.layers[i].attention.wqkv.w_map_negative_32_div.copy_(wqkv_neg)

            wo, wo_neg = prepare_w_map_fast(1,2560,2560,1536)
            m_40.layers[i].attention.wo.w_map_32_div.copy_(wo)
            m_40.layers[i].attention.wo.w_map_negative_32_div.copy_(wo_neg)

            w13, w13_neg = prepare_w_map_fast(1,2560,13824,1536)
            m_40.layers[i].feed_forward.w13.w_map_32_div.copy_(w13)
            m_40.layers[i].feed_forward.w13.w_map_negative_32_div.copy_(w13_neg)

            w2, w2_neg = prepare_w_map_fast(1,6912,2560,4096)
            m_40.layers[i].feed_forward.w2.w_map_32_div.copy_(w2)
            m_40.layers[i].feed_forward.w2.w_map_negative_32_div.copy_(w2_neg)

        torch.save(m_40.state_dict(), "sptmm_40.pt")

    m_60 = model.Transformer(model.ModelArgs(use_kernel=False, use_sptmm=True, sparsity=60))

    with torch.no_grad():
        for i in range(30):
            wqkv, wqkv_neg = prepare_w_map_fast(1,2560,3840,1024)
            m_60.layers[i].attention.wqkv.w_map_32_div.copy_(wqkv)
            m_60.layers[i].attention.wqkv.w_map_negative_32_div.copy_(wqkv_neg)

            wo, wo_neg = prepare_w_map_fast(1,2560,2560,1024)
            m_60.layers[i].attention.wo.w_map_32_div.copy_(wo)
            m_60.layers[i].attention.wo.w_map_negative_32_div.copy_(wo_neg)

            w13, w13_neg = prepare_w_map_fast(1,2560,13824,1024)
            m_60.layers[i].feed_forward.w13.w_map_32_div.copy_(w13)
            m_60.layers[i].feed_forward.w13.w_map_negative_32_div.copy_(w13_neg)

            w2, w2_neg = prepare_w_map_fast(1,6912,2560,2752)
            m_60.layers[i].feed_forward.w2.w_map_32_div.copy_(w2)
            m_60.layers[i].feed_forward.w2.w_map_negative_32_div.copy_(w2_neg)

        torch.save(m_60.state_dict(), "sptmm_60.pt")

    m_80 = model.Transformer(model.ModelArgs(use_kernel=False, use_sptmm=True, sparsity=80))

    with torch.no_grad():
        for i in range(30):
            wqkv, wqkv_neg = prepare_w_map_fast(1,2560,3840,512)
            m_80.layers[i].attention.wqkv.w_map_32_div.copy_(wqkv)
            m_80.layers[i].attention.wqkv.w_map_negative_32_div.copy_(wqkv_neg)

            wo, wo_neg = prepare_w_map_fast(1,2560,2560,512)
            m_80.layers[i].attention.wo.w_map_32_div.copy_(wo)
            m_80.layers[i].attention.wo.w_map_negative_32_div.copy_(wo_neg)

            w13, w13_neg = prepare_w_map_fast(1,2560,13824,512)
            m_80.layers[i].feed_forward.w13.w_map_32_div.copy_(w13)
            m_80.layers[i].feed_forward.w13.w_map_negative_32_div.copy_(w13_neg)

            w2, w2_neg = prepare_w_map_fast(1,6912,2560,1408)
            m_80.layers[i].feed_forward.w2.w_map_32_div.copy_(w2)
            m_80.layers[i].feed_forward.w2.w_map_negative_32_div.copy_(w2_neg)

        torch.save(m_80.state_dict(), "sptmm_80.pt")