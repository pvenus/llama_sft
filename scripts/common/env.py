import platform
import torch

def backend():
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    elif torch.cuda.is_available():
        print("cuda")
        return "cuda"
    else:
        print("cpu")
        return "cpu"