import torch


def get_device() -> torch.device:
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()
    return torch.device("cpu")


def move(model, batch):
    device = get_device()
    return model.to(device), batch.to(device)
