"""Run a judge locally with Hugging Face transformers (CUDA, Apple MPS or CPU).

This is for smoke tests on small judges. Audits and GRPO on the GPU serve the
same prompts with vLLM; prompt construction and parsing are shared.
"""

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.device import default_device, default_dtype
from src.judges.parse import parse_verdict
from src.judges.prompts import build_prompt, judge_answer


@dataclass
class JudgeResult:
    answer: str         # what the judge saw as the student's answer
    output: str         # raw judge output
    accepted: bool
    parse_fail: bool
    label: str | None
    n_tokens: int = 0         # generated tokens (vLLM backend only)
    truncated: bool = False   # hit max_new_tokens (vLLM backend only)


class LocalJudge:
    def __init__(self, cfg, device=None):
        self.cfg = cfg
        self.device = device or default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model"], padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=default_dtype(self.device)).to(
            self.device
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompts, batch_size=8):
        outputs = []
        sampling = self.cfg["temperature"] > 0
        for i in range(0, len(prompts), batch_size):
            batch = self.tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                                   add_special_tokens=False).to(self.device)
            kwargs = dict(max_new_tokens=self.cfg["max_new_tokens"], do_sample=sampling,
                          pad_token_id=self.tokenizer.pad_token_id)
            if sampling:
                kwargs.update(temperature=self.cfg["temperature"], top_p=self.cfg["top_p"])
            out = self.model.generate(**batch, **kwargs)
            new_tokens = out[:, batch["input_ids"].shape[1]:]
            outputs += self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return outputs

    def judge(self, items, batch_size=8):
        """items: dicts with question, reference and response. Returns JudgeResults and seconds."""
        answers = [judge_answer(it["response"]) for it in items]
        prompts = [build_prompt(self.cfg, it["question"], it["reference"], a, self.tokenizer)
                   for it, a in zip(items, answers)]
        start = time.time()
        raw = self.generate(prompts, batch_size)
        elapsed = time.time() - start
        results = []
        for a, o in zip(answers, raw):
            v = parse_verdict(self.cfg, o)
            results.append(JudgeResult(a, o, v.accepted, v.parse_fail, v.label))
        return results, elapsed
