import torch

def get_device(device_str="cpu"):
    """
    device_str:
        - "cpu"
        - "cuda"
        - "cuda:0"
        - "auto" -> choose CUDA if available
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"Device set to: {torch.cuda.get_device_name(0)} (cuda:0)")
            return torch.device("cuda:0")
        else:
            print("Device set to: CPU")
            return torch.device("cpu")

    if device_str.startswith("cuda"):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"Device set to: {torch.cuda.get_device_name(0)} ({device_str})")
            return torch.device(device_str)
        else:
            print("[WARNING] CUDA requested but not available → using CPU")
            return torch.device("cpu")

    if device_str == "cpu":
        print("Device set to: CPU")
        return torch.device("cpu")

    print("[WARNING] Unknown device flag, defaulting to CPU")
    return torch.device("cpu")
