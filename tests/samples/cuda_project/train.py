import torch
import flash_attn


def train(model, batch):
    model = model.cuda()
    device = "cuda:0"
    batch = batch.to(device)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    return model(batch)
