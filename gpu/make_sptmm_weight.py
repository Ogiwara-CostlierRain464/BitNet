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



def col2row(w):
    return w.t().contiguous()

if __name__ == "__main__":
    # we use
    m_40 = model.Transformer(model.ModelArgs(use_kernel=False, use_sptmm=True, sparsity=40))

    with torch.no_grad():
        for i in range(30):
            result = prepare_w_map_fast(1,2560,3840,1536)
            wqkv, wqkv_neg = result[3:5]
            m_40.layers[i].attention.wqkv.w_map_32_div.copy_(wqkv)
            m_40.layers[i].attention.wqkv.w_map_negative_32_div.copy_(wqkv_neg)

            result = prepare_w_map_fast(1,2560,2560,1536)
            wo, wo_neg = result[3:5]
            m_40.layers[i].attention.wo.w_map_32_div.copy_(wo)
            m_40.layers[i].attention.wo.w_map_negative_32_div.copy_(wo_neg)

            result = prepare_w_map_fast(1,2560,13824,1536)
            w13, w13_neg = result[3:5]
            m_40.layers[i].feed_forward.w13.w_map_32_div.copy_(w13)
            m_40.layers[i].feed_forward.w13.w_map_negative_32_div.copy_(w13_neg)

            result = prepare_w_map_fast(1,6912,2560,4096)
            w2, w2_neg = result[3:5]
            m_40.layers[i].feed_forward.w2.w_map_32_div.copy_(w2)
            m_40.layers[i].feed_forward.w2.w_map_negative_32_div.copy_(w2_neg)

        torch.save(m_40.state_dict(), "./checkpoints/sptmm_40.pt")

    # m_60 = model.Transformer(model.ModelArgs(use_kernel=False, use_sptmm=True, sparsity=60))
    #
    # with torch.no_grad():
    #     for i in range(30):
    #         result = prepare_w_map_fast(1,2560,3840,1024)
    #         wqkv, wqkv_neg = result[3:5]
    #         m_60.layers[i].attention.wqkv.w_map_32_div.copy_(wqkv)
    #         m_60.layers[i].attention.wqkv.w_map_negative_32_div.copy_(wqkv_neg)
    #
    #         result = prepare_w_map_fast(1,2560,2560,1024)
    #         wo, wo_neg = result[3:5]
    #         m_60.layers[i].attention.wo.w_map_32_div.copy_(wo)
    #         m_60.layers[i].attention.wo.w_map_negative_32_div.copy_(wo_neg)
    #
    #         result = prepare_w_map_fast(1,2560,13824,1024)
    #         w13, w13_neg = result[3:5]
    #         m_60.layers[i].feed_forward.w13.w_map_32_div.copy_(w13)
    #         m_60.layers[i].feed_forward.w13.w_map_negative_32_div.copy_(w13_neg)
    #
    #         result = prepare_w_map_fast(1,6912,2560,2752)
    #         w2, w2_neg = result[3:5]
    #         m_60.layers[i].feed_forward.w2.w_map_32_div.copy_(w2)
    #         m_60.layers[i].feed_forward.w2.w_map_negative_32_div.copy_(w2_neg)
    #
    #     torch.save(m_60.state_dict(), "sptmm_60.pt")
    #
    # m_80 = model.Transformer(model.ModelArgs(use_kernel=False, use_sptmm=True, sparsity=80))
    #
    # with torch.no_grad():
    #     for i in range(30):
    #         result = prepare_w_map_fast(1,2560,3840,512)
    #         wqkv, wqkv_neg = result[3:5]
    #         m_80.layers[i].attention.wqkv.w_map_32_div.copy_(wqkv)
    #         m_80.layers[i].attention.wqkv.w_map_negative_32_div.copy_(wqkv_neg)
    #
    #         result  = prepare_w_map_fast(1,2560,2560,512)
    #         wo, wo_neg = result[3:5]
    #         m_80.layers[i].attention.wo.w_map_32_div.copy_(wo)
    #         m_80.layers[i].attention.wo.w_map_negative_32_div.copy_(wo_neg)
    #
    #         result = prepare_w_map_fast(1,2560,13824,512)
    #         w13, w13_neg = result[3:5]
    #         m_80.layers[i].feed_forward.w13.w_map_32_div.copy_(w13)
    #         m_80.layers[i].feed_forward.w13.w_map_negative_32_div.copy_(w13_neg)
    #
    #         result = prepare_w_map_fast(1,6912,2560,1408)
    #         w2, w2_neg = result[3:5]
    #         m_80.layers[i].feed_forward.w2.w_map_32_div.copy_(w2)
    #         m_80.layers[i].feed_forward.w2.w_map_negative_32_div.copy_(w2_neg)
    #
    #     torch.save(m_80.state_dict(), "sptmm_80.pt")