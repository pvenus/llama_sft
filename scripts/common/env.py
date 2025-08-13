import platform
import torch

def backend():
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("Mode mlx")
        return "mlx"
    elif torch.cuda.is_available():
        print("Mode cuda")
        return "cuda"
    else:
        print("Mode cpu")
        return "cpu"

if __name__ == "__main__":
    backend()