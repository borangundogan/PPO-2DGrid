import torch

def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Device set to: {torch.cuda.get_device_name(device)}")
    else:
        print("Device set to: CPU")
    return device
