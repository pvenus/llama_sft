

## eval_fc_cli.py Options

| Option           | Description                                                                                               | Default                                             |
|------------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| `--model`        | HF repo ID or local path of the model to use.                                                             | `mlx-community/Bio-Medical-Llama-3-2-1B-CoT-012025` |
| `--file`         | Path to evaluation JSONL file. Each line should be `{"messages":[...], "assistant":"...json string..."}`. | `outputs/datasets/eval.jsonl`                       |
| `--spec`         | Path to external JSON spec file defining prompt and regex patterns.                                       | `assets/train/fc_patterns.json`                     |
| `--out`          | Path to save the evaluation results as JSONL.                                                             | `outputs/eval/output.jsonl`                         |
| `--adapters`     | Path to LoRA adapter (`adapters.npz`). If set, runs in finetuned mode.                                    | `None`                                              |
| `--max_tokens`   | Maximum number of tokens to generate per sample.                                                          | `32`                                                |
| `--no_fallback`  | Disable rule-based fallback when model fails to output valid JSON.                                        | `False`                                             |

## eval_suite.py Options

| Option          | Description                                          | Default                                      |
|-----------------|------------------------------------------------------|----------------------------------------------|
| `--suite`       | Name of the evaluation suite to run.                 | `default`                                    |
| `--input`       | Path to input dataset file or directory.             | `None`                                       |
| `--output`      | Path to save evaluation results.                     | `outputs/eval_suite/results.jsonl`           |
| `--metrics`     | Path to metrics configuration file.                  | `assets/metrics/default_metrics.json`        |
| `--max_samples` | Limit the number of samples to evaluate.             | `None`                                       |
| `--shuffle`     | Shuffle dataset before evaluation.                   | `False`                                      |
| `--seed`        | Random seed for shuffling.                           | `42`                                         |