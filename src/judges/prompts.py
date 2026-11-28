"""Judge prompt templates and the answer string every judge receives.

Placeholders {question}, {reference} and {answer} are filled in one pass, so
LaTeX braces in the inputs are kept as they are and an answer that happens to
contain the text "{question}" is not expanded a second time.
"""

import re
from functools import lru_cache
from pathlib import Path

import yaml

from src.gold.checker import last_boxed

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "judges.yaml"

# Huang et al. [1], Appendix E, Figure 7. The figure shows the key phrases in bold;
# we render them as markdown bold. Its "\\boxed{{1}}" is a Python escape for \boxed{1}.
HUANG_WITHOUT_QUESTION = (
    'Your task is to determine if the **Extracted Answer** is mathematically equivalent to the **Ground '
    'Truth Answer**.\n'
    '**Ground Truth Answer:**\n'
    '{reference}\n'
    '**Extracted Answer:**\n'
    '{answer}\n'
    '- If **Extracted Answer** and **Ground Truth Answer** are mathematically equivalent, respond with '
    '\\boxed{1}\n'
    '- If they are not mathematically equivalent, or if the **Extracted Answer** is nonsensical (e.g., a '
    'random string), respond with \\boxed{0}'
)

HUANG_WITH_QUESTION = """\
Your task is to determine if the **Extracted Answer** is mathematically equivalent to the **Ground Truth Answer**.
**Question**
{question}
**Ground Truth Answer:**
{reference}
**Extracted Answer:**
{answer}
Please follow these steps clearly:
1. **Review the Question and Ground Truth Answer carefully.**
2. **Compare the Extracted Answer with the Ground Truth Answer.**
3. **Explain step-by-step** whether or not they express the same meaning or information.
4. **Provide your final decision clearly** at the end:
- Respond with \\boxed{1} if the answers are equivalent.
- Respond with \\boxed{0} if the answers are **not** equivalent."""

# TIGER-Lab/general-verifier model card. Used as a raw prompt, without a chat template.
GENERAL_VERIFIER = (
    "User: ### Question: {question}\n\n"
    "### Ground Truth Answer: {reference}\n\n"
    "### Student Answer: {answer}\n\n"
    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent "
    'to the ground truth answer.\n'
    'If the student\'s answer is correct, output "Final Decision: Yes". '
    'If the student\'s answer is incorrect, output "Final Decision: No". Assistant:'
)

# IAAR-Shanghai/xVerify, src/xVerify/prompts/judge_prompt.py.
XVERIFY = (
    'You are a diligent and precise assistant tasked with evaluating the correctness of responses. You '
    'will receive a question, an output sentence, and the correct answer. Your task is to determine if the '
    'output sentence accurately answers the question based on the provided correct answer. Respond with '
    'either [Correct] or [Incorrect].\n'
    '-\n'
    'Special considerations:\n'
    '\n'
    '1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers '
    'modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If '
    'the final answer is unclear or incorrect, respond with [Incorrect].\n'
    '\n'
    '2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, '
    'respond with [Correct].\n'
    '\n'
    '3. **Explicit Options**: If the question provides explicit candidate answers, the output will be '
    "considered correct if it clearly indicates the correct option's code or the correct option's "
    'content.\n'
    '\n'
    '4. **No Explicit Options**: If the question does not provide explicit options, the output must align '
    'with the correct answer in content and meaning to be considered [Correct].\n'
    '-\n'
    '\n'
    'Question: """{question}"""\n'
    '\n'
    'Output sentence: """{answer}"""\n'
    '\n'
    'Correct answer: {reference}\n'
    '\n'
    'Judgement:\n'
)

TEMPLATES = {
    "huang_without_question": HUANG_WITHOUT_QUESTION,
    "huang_with_question": HUANG_WITH_QUESTION,
    "general_verifier": GENERAL_VERIFIER,
    "xverify": XVERIFY,
}

PLACEHOLDER_RE = re.compile(r"\{(question|reference|answer)\}")


@lru_cache(maxsize=1)
def load_config():
    return yaml.safe_load(CONFIG.read_text())


def judge_config(name):
    return load_config()["judges"][name]


def judge_answer(response, tail_chars=None):
    """The student's answer shown to a judge: the last non-empty \\boxed{} content,
    otherwise the last tail_chars characters of the response."""
    if tail_chars is None:
        tail_chars = load_config()["answer_input"]["tail_chars"]
    response = response or ""
    content = last_boxed(response)
    if content is not None and content.strip():
        return content.strip()
    return response[-tail_chars:].strip()


def render(template_name, question, reference, answer):
    """Fill a template in one pass."""
    values = {"question": str(question), "reference": str(reference), "answer": str(answer)}
    return PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], TEMPLATES[template_name])


def build_prompt(cfg, question, reference, answer, tokenizer=None):
    """Full prompt string for a judge config. Chat-template judges need their tokenizer."""
    text = render(cfg["template"], question, reference, answer)
    if not cfg["chat_template"]:
        return text
    if tokenizer is None:
        raise ValueError(f"{cfg['model']} uses a chat template; pass its tokenizer")
    # chat_template_kwargs, e.g. {enable_thinking: false} for Qwen3.5 and later, which think by default
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                         add_generation_prompt=True, **cfg.get("chat_template_kwargs", {}))
