from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class AnswerGenerator:
    """Generate grounded answers with a loaded causal language model.

    Args:
        tokenizer: Tokenizer used to prepare prompts and decode answers.
        model: Causal language model used for generation.
        device: Device on which model inference runs.
        max_new_tokens: Maximum number of tokens generated per answer.
    """

    tokenizer: Any
    model: Any
    device: str
    max_new_tokens: int

    def generate(self, question: str, contexts: list[str]) -> str:
        """Generate one answer from a question and retrieved contexts.

        Args:
            question: Question to answer.
            contexts: Retrieved source texts used as evidence.

        Returns:
            The decoded model answer.
        """
        sources = "\n\n".join(
            f"[Source {index}]\n{text}"
            for index, text in enumerate(contexts, start=1)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question using only the provided vLLM "
                    "sources. If the sources do not contain the answer, "
                    "say so. Use exact identifiers, commands, and values "
                    "when available. Keep the answer clear, concise, and "
                    "grounded."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Sources:\n{sources}\n\nQuestion:\n{question}"
                ),
            },
        ]
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
        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_length:]
        return str(
            self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
        ).strip()


def load_answer_model(max_new_tokens: int) -> AnswerGenerator:
    """Load the mandatory Qwen model on CUDA when available.

    Args:
        max_new_tokens: Maximum number of tokens generated per answer.

    Returns:
        A configured answer generator.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
    )
    model.to(device)
    model.eval()
    return AnswerGenerator(
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=max_new_tokens,
    )
