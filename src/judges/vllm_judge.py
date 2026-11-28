"""Run a judge with vLLM on the GPU.

Same interface as src.judges.local.LocalJudge: build the native prompt, generate
with the judge's decoding settings from configs/judges.yaml, parse the verdict.
"""

import time

from src.device import configure_vllm
from src.judges.local import JudgeResult
from src.judges.parse import parse_verdict
from src.judges.prompts import build_prompt, judge_answer

PROMPT_ROOM = 2048   # tokens reserved for the judge prompt (question + answer tail)


class VLLMJudge:
    def __init__(self, cfg, gpu_memory_utilization=0.9, seed=0, **llm_kwargs):
        configure_vllm()
        from vllm import LLM

        self.cfg = cfg
        max_model_len = max(4096, cfg["max_new_tokens"] + PROMPT_ROOM)
        llm_kwargs = {**cfg.get("vllm_kwargs", {}), **llm_kwargs}  # e.g. limit_mm_per_prompt for vision-language judges
        dtype = "auto" if cfg.get("quantized") else "bfloat16"       # FP8 checkpoints keep their own dtypes
        self.llm = LLM(model=cfg["model"], dtype=dtype, seed=seed, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, **llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def judge(self, items, seed=0):
        """items: dicts with question, reference and response. Returns JudgeResults and seconds."""
        from vllm import SamplingParams

        answers = [judge_answer(it["response"]) for it in items]
        prompts = [build_prompt(self.cfg, it["question"], it["reference"], a, self.tokenizer)
                   for it, a in zip(items, answers)]
        params = SamplingParams(temperature=self.cfg["temperature"], top_p=self.cfg["top_p"],
                                max_tokens=self.cfg["max_new_tokens"], seed=seed)
        start = time.time()
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        seconds = time.time() - start
        results = []
        for a, out in zip(answers, outputs):
            c = out.outputs[0]
            v = parse_verdict(self.cfg, c.text)
            results.append(JudgeResult(a, c.text, v.accepted, v.parse_fail, v.label, len(c.token_ids),
                                       c.finish_reason == "length"))
        return results, seconds
