import model
import torch
import tqdm

if __name__ == "__main__":
    # we use
    m = model.Transformer(model.ModelArgs(use_kernel=False))
    
    with torch.no_grad():
        for i in range(30):
            m.layers[i].attention.wqkv.weight.copy_(torch.zeros((3840, 2560), dtype=torch.bfloat16))
            m.layers[i].attention.wo.weight.copy_(torch.zeros((2560, 2560), dtype=torch.bfloat16))
            m.layers[i].feed_forward.w13.weight.copy_(torch.zeros((13824, 2560), dtype=torch.bfloat16))
            m.layers[i].feed_forward.w2.weight.copy_(torch.zeros((2560, 6912), dtype=torch.bfloat16))

        torch.save(m.state_dict(), "test.pt")