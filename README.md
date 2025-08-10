#LLAMA SFT PROJECT

## Run (Mac)

```bash
make venv
make sft
```

## Run (Windows)
>```bash
>make venv
>```
>```bash
>.\.venv\Scripts\activate
>```
>```bash
>pip uninstall torch torchvision torchaudio -y
>pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
>```
>```bash
>python -m scripts.main
>```