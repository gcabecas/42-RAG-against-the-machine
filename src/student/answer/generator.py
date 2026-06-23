import re
from dataclasses import dataclass
from typing import Any, cast

from student.answer.context import SourceContext
from student.answer.prompt import build_messages


@dataclass
class AnswerGenerator:
    tokenizer: Any
    model: Any
    device: str
    max_new_tokens: int

    def generate(self, question: str, contexts: list[SourceContext]) -> str:
        messages = build_messages(question, contexts, self.max_new_tokens)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        answer = str(self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ))
        return clean_answer(answer)


def clean_answer(answer: str) -> str:
    if "</think>" in answer:
        answer = answer.split("</think>", 1)[1]
    answer = answer.replace("**", "")
    answer = re.sub(
        r"\[([^\]]+)\]\(\s*\[source ([0-9]+)\]\s*\)",
        r"\1 [source \2]",
        answer,
    )
    answer = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 \2", answer)
    answer = re.sub(r"\[([^\]]+)\]\(#[^)]+\)", r"\1", answer)
    answer = re.sub(r"\[([^\]]+)\]\[[^\]]+\]", r"\1", answer)
    answer = re.sub(
        r"\bvLLM(?=(metric|model|server|service|framework|deployment)\b)",
        "vLLM ",
        answer,
    )
    answer = re.sub(
        r"\b(framework|command|script|metric|method|class|endpoint|"
        r"parameter|version)(?=(that|used|is|can|should)\b)",
        r"\1 ",
        answer,
    )
    citations = re.findall(r"\[source [0-9]+\]", answer)
    if len(citations) > 1:
        answer = re.sub(r"\s*\[source [0-9]+\]", "", answer)
        answer = f"{answer.rstrip(' .')} {citations[0]}"
    return answer.strip()


def load_answer_model(
    max_new_tokens: int,
) -> AnswerGenerator:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = cast(Any, AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"))
    model = cast(Any, AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        dtype=dtype,
    ))
    model.to(device)
    model.eval()
    return AnswerGenerator(
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=max_new_tokens,
    )
