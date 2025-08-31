from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import json
from .trainter_helper import _ensure_dir, _prepare_train, _finished_msg


def train_with_peft(cfg) -> Path:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else None

    output_dir = _ensure_dir(cfg.output_dir)

    # Prepare unified data layout
    data_dir, train_file, eval_file = _prepare_train(cfg)

    # --- load data (chat jsonl: messages + assistant) ---
    def load_jsonl(p: str | Path):
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def to_text(ex: Dict[str, Any]) -> Dict[str, str]:


        msgs = ex.get("messages", [])
        sys_txt = ""
        user_txt = ""
        for m in msgs:
            if m.get("role") == "system":
                sys_txt = m.get("content", "")
            if m.get("role") == "user":
                user_txt = m.get("content", "")

        prompt = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                  f"{sys_txt}\n"
                  "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                  f"{user_txt}\n"
                  "<|eot_id|><|start_header_id|>content<|end_header_id|>\n"
                  '[{"functionName":'
                  )
        raw_target = ex.get("assistant", "")
        try:
            parsed = json.loads(raw_target)
            if isinstance(parsed, list):
                # [{"fn":..}, {"fn":..}] → {"fn":..}{"fn":..} 식으로 flatten
                target = "".join(json.dumps(obj, ensure_ascii=False) for obj in parsed)
            else:
                target = json.dumps(parsed, ensure_ascii=False)
        except Exception:
            # 파싱 실패 시 그냥 원문
            target = raw_target
        return {"prompt": prompt, "target": target}

    train_rows = [to_text(r) for r in load_jsonl(train_file)]
    eval_rows  = [to_text(r) for r in load_jsonl(eval_file)]

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(batch):
        texts = [p + t for p, t in zip(batch["prompt"], batch["target"])]
        return tok(texts, padding=True, truncation=True, max_length=cfg.max_seq_len)

    train_ds = Dataset.from_list(train_rows).map(tokenize, batched=True, remove_columns=["prompt", "target"])  # type: ignore
    eval_ds  = Dataset.from_list(eval_rows ).map(tokenize, batched=True, remove_columns=["prompt", "target"])  # type: ignore

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype)
    if device == "cuda":
        model = model.to("cuda")
    if cfg.grad_checkpoint:
        model.gradient_checkpointing_enable()

    if cfg.fine_tune_type.lower() != "lora":
        raise ValueError("This path currently supports LoRA only. Use fine_tune_type='lora'.")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    if cfg.resume_adapter_file:
        resume_dir = Path(cfg.resume_adapter_file)
        resume_dir = resume_dir if resume_dir.is_dir() else resume_dir.parent
        model = PeftModel.from_pretrained(model, resume_dir)

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        max_steps=cfg.iters,
        eval_strategy="steps",
        eval_steps=cfg.eval_every_steps,
        save_steps=cfg.save_every_steps,
        learning_rate=cfg.lr,
        logging_steps=cfg.steps_per_report,
        fp16=(device == "cuda"),
        report_to=[],
        save_total_limit=2,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()

    model.save_pretrained(str(output_dir))
    tok.save_pretrained(str(output_dir))
    _finished_msg("HF/PEFT LoRA training", output_dir, data_dir)
    return output_dir
