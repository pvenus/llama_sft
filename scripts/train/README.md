


## train_sft.py option

| Option                  | Description                                                           | Default                                             |
|-------------------------|-----------------------------------------------------------------------|-----------------------------------------------------|
| `--base-model`          | Hugging Face repo ID or local path of the base model                  | `mlx-community/Bio-Medical-Llama-3-2-1B-CoT-012025` |
| `--train-path`          | Path to training dataset file (`train.jsonl`)                         | **Required**                                        |
| `--eval-path`           | Path to evaluation dataset file (`eval.jsonl`)                        | **Required**                                        |
| `--output-dir`          | Directory to save LoRA adapters (checkpoints + `adapter_config.json`) | `outputs/train/adapters`                            |
| `--batch-size`          | Training batch size                                                   | `2`                                                 |
| `--iters`               | Number of training iterations                                         | `500`                                               |
| `--eval-every-steps`    | Evaluate every N steps                                                | `100`                                               |
| `--save-every-steps`    | Save checkpoint every N steps                                         | `200`                                               |
| `--lr`                  | Learning rate                                                         | `2e-4`                                              |
| `--max-seq-len`         | Maximum sequence length                                               | `512`                                               |
| `--val-batches`         | Number of validation batches                                          | `50`                                                |
| `--no-mask-prompt`      | Do **not** mask prompt tokens in loss                                 | Disabled by default (masking enabled)               |
| `--no-grad-checkpoint`  | Disable gradient checkpointing                                        | Disabled by default (checkpointing enabled)         |
| `--fine-tune-type`      | Fine-tuning method (`lora`, `dora`, `full`)                           | `lora`                                              |
| `--steps-per-report`    | Report training progress every N steps                                | `50`                                                |
| `--resume-adapter-file` | Resume training from given adapters file path                         | `None`                                              |