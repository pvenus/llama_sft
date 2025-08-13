## dataset_preparer.py option

| Option             | Description                                                 | Default                  |
|--------------------|-------------------------------------------------------------|--------------------------|
| `--in-root`        | Input root directory                                        | `scripts/data/resources` |
| `--out-dir`        | Output directory                                            | `scripts/data/outputs`   |
| `--json-pattern`   | Glob pattern(s) for JSON files (comma-separated allowed)    | `json/*.json`            |
| `--csv-pattern`    | Glob pattern(s) for CSV files (comma-separated allowed)     | `csv/*.csv`              |
| `--csv-input-col`  | CSV column name for the user text                           | `input`                  |
| `--csv-output-col` | CSV column name for the assistant JSON string               | `output_json`            |
| `--system-prompt`  | System message injected into `messages[0]`                  | function calling default |
| `--split`          | Split ratios `%` or fractions (e.g., `90,10` or `80,10,10`) | `90,10`                  |
| `--seed`           | Random seed (shuffle)                                       | `42`                     |
| `--no-shuffle`     | Disable shuffling                                           | off                      |
| `--limit`          | Cap number of items before split (0 = all)                  | `0`                      |
| `--train-name`     | Output file name for train set                              | `train.jsonl`            |
| `--eval-name`      | Output file name for eval set                               | `eval.jsonl`             |
| `--test-name`      | Output file name for test set                               | `test.jsonl`             |

## system_template.py option

| Option     | Description                                                        | Default                 |
|------------|--------------------------------------------------------------------|-------------------------|
| `--spec`   | Path to YAML spec file (local file only)                           |                         |
| `--base`   | Path to base SYSTEM message text file to prepend (local file only) |                         |
| `--out`    | Path to save output; if omitted, print to stdout                   |                         |
| `--format` | Output format: 'yaml' or 'json'                                    | `yaml`                  |
| `--pretty` | Pretty-print output (for JSON format only)                         | `off`                   |

### Example usage

```bash
python system_template.py --spec configs/spec.yaml --base configs/base_system.txt --out outputs/system_prompt.txt
```

```bash
python system_template.py --spec configs/spec.yaml --base configs/base_system.txt --format json --pretty --out outputs/system_prompt.json
```