"""Seed the policy: SFT on the set built by src.train.seed_data, before RL.

Usage (GPU):  python -m src.train.sft --name seed2pct
Local test:   python -m src.train.sft --name test --local-test

Full fine-tuning with TRL's SFTTrainer on prompt-completion pairs, loss on the completion only
(the policy prompt is the RL prompt). The model is saved in bf16 to runs/sft/<name>/model, so
the vLLM engines of the propensity audit and of GRPO load it in bf16 like the base policy;
GRPO upcasts it to fp32 master weights. Writes sft_log.jsonl and meta_sft.json as well.
"""

import argparse
import json
import time

from src.paths import ROOT, display_path
from src.pipeline.records import write_meta
from src.train.seed_data import CONFIG, load_sft_config


def load_dataset(path):
    from datasets import Dataset

    rows = [json.loads(line) for line in path.open()]
    return Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows]), rows


def main():  # pragma: no cover - real-model training entry point
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-test", action="store_true")
    args = ap.parse_args()
    cfg = load_sft_config(args.config, args.local_test)
    run_dir = ROOT / "runs" / "sft" / args.name
    out = run_dir / "model"
    if (out / "config.json").exists():
        raise SystemExit(f"{display_path(out)} exists; remove it to retrain")

    import torch
    from transformers import AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    if args.local_test:
        from src.train import local_compat
        local_compat.install()

    class LogToFile(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            if logs:
                with (run_dir / "sft_log.jsonl").open("a") as f:
                    f.write(json.dumps({"step": state.global_step, **logs}) + "\n")

    dataset, rows = load_dataset(run_dir / "data.jsonl")
    model_name = cfg["policy"]["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    trainer = SFTTrainer(model=model_name, train_dataset=dataset, processing_class=tokenizer,
                         args=SFTConfig(output_dir=str(run_dir / "trainer"), seed=args.seed, **cfg["trainer"]),
                         callbacks=[LogToFile()])
    start = time.time()
    trainer.train()
    minutes = (time.time() - start) / 60
    model = trainer.model.to(torch.bfloat16)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    write_meta(run_dir, name="meta_sft.json", stage="sft", base_model=model_name, n_examples=len(rows),
               steps=trainer.state.global_step, train_minutes=round(minutes, 1), sft_config=cfg,
               local_test=args.local_test)
    print(f"SFT done: {len(rows)} examples, {trainer.state.global_step} steps in {minutes:.1f} min -> "
          f"{display_path(out)}", flush=True)


if __name__ == "__main__":
    main()
