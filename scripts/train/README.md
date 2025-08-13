## train_sft.py option

| Option                          | Description                                                             | Default                                             |
|---------------------------------|-------------------------------------------------------------------------|-----------------------------------------------------|
| `--base-model`                  | Hugging Face repo ID or local path of the base model                    | `mlx-community/Bio-Medical-Llama-3-2-1B-CoT-012025` |
| `--train-path`                  | Path to training dataset file (`train.jsonl`)                           | **Required**                                        |
| `--eval-path`                   | Path to evaluation dataset file (`eval.jsonl`)                          | **Required**                                        |
| `--output-dir`                  | Directory to save LoRA adapters (checkpoints + `adapter_config.json`)   | `outputs/train/adapters`                            |
| `--batch-size`                  | Training batch size                                                     | `2`                                                 |
| `--iters`                       | Number of training iterations                                           | `500`                                               |
| `--eval-every-steps`            | Evaluate every N steps                                                  | `100`                                               |
| `--save-every-steps`            | Save checkpoint every N steps                                           | `200`                                               |
| `--lr`                          | Learning rate                                                           | `2e-4`                                              |
| `--max-seq-len`                 | Maximum sequence length                                                 | `512`                                               |
| `--val-batches`                 | Number of validation batches                                            | `50`                                                |
| `--no-mask-prompt`              | Do **not** mask prompt tokens in loss                                   | Disabled by default (masking enabled)               |
| `--no-grad-checkpoint`          | Disable gradient checkpointing                                          | Disabled by default (checkpointing enabled)         |
| `--fine-tune-type`              | Fine-tuning method (`lora`, `dora`, `full`)                             | `lora`                                              |
| `--steps-per-report`            | Report training progress every N steps                                  | `50`                                                |
| `--resume-adapter-file`         | Resume training from given adapters file path                           | `None`                                              |
| `--seed`                        | Random seed for reproducibility                                         | `42`                                                |
| `--gradient-accumulation-steps` | Number of steps to accumulate gradients before updating weights         | `1`                                                 |
| `--warmup-steps`                | Number of warmup steps                                                  | `50`                                                |
| `--logging-dir`                 | Directory for TensorBoard logs                                          | `logs`                                              |

## Example CLI Usage

1. Minimal example with required arguments only:

```bash
python train_sft.py --train-path train.jsonl --eval-path eval.jsonl
```

2. Full example with most parameters specified:

```bash
python train_sft.py \
  --base-model mlx-community/Bio-Medical-Llama-3-2-1B-CoT-012025 \
  --train-path data/train.jsonl \
  --eval-path data/eval.jsonl \
  --output-dir outputs/train/adapters \
  --batch-size 4 \
  --iters 1000 \
  --eval-every-steps 200 \
  --save-every-steps 400 \
  --lr 3e-4 \
  --max-seq-len 512 \
  --val-batches 100 \
  --no-mask-prompt \
  --no-grad-checkpoint \
  --fine-tune-type lora \
  --steps-per-report 100 \
  --resume-adapter-file outputs/train/adapters/checkpoint-400/pytorch_model.bin \
  --seed 1234 \
  --gradient-accumulation-steps 2 \
  --warmup-steps 100 \
  --logging-dir logs
```