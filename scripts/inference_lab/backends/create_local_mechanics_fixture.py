"""Create the checked-in local runtime mechanics fixture."""

from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

FIXTURE_DIR = Path("tests/fixtures/backends/runtime/local_mechanics")


def main() -> None:
    """Create deterministic tokeniser and causal-model artefacts."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    vocabulary = {
        "[PAD]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "[UNK]": 3,
        "[SYSTEM]": 4,
        "[USER]": 5,
        "[ASSISTANT]": 6,
    }
    for token in "abcdefghijklmnopqrstuvwxyz!?.,":
        vocabulary[token] = len(vocabulary)
    merges = (
        ("h", "e"),
        ("he", "l"),
        ("hel", "l"),
        ("hell", "o"),
        ("w", "o"),
        ("wo", "r"),
        ("wor", "l"),
        ("worl", "d"),
        ("l", "o"),
        ("lo", "c"),
        ("loc", "a"),
        ("loca", "l"),
        ("r", "u"),
        ("ru", "n"),
        ("run", "t"),
        ("runt", "i"),
        ("runti", "m"),
        ("runtim", "e"),
        ("c", "a"),
        ("ca", "c"),
        ("cac", "h"),
        ("cach", "e"),
        ("t", "e"),
        ("te", "s"),
        ("tes", "t"),
    )
    for left, right in merges:
        vocabulary[left + right] = len(vocabulary)
    tokenizer = Tokenizer(BPE(vocab=vocabulary, merges=merges, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()  # pyright: ignore[reportAttributeAccessIssue]
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=["[SYSTEM]", "[USER]", "[ASSISTANT]"],
    )
    fast_tokenizer.chat_template = (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}[SYSTEM]"
        "{% elif message['role'] == 'user' %}[USER]"
        "{% elif message['role'] == 'assistant' %}[ASSISTANT]"
        "{% else %}[UNK]{% endif %}"
        " {{ message['content'] }}"
        "{% endfor %}"
        "[ASSISTANT]"
    )
    fast_tokenizer.save_pretrained(FIXTURE_DIR)

    torch.manual_seed(20260902)
    config = GPT2Config(
        vocab_size=len(vocabulary),
        n_positions=64,
        n_ctx=64,
        n_embd=16,
        n_layer=1,
        n_head=1,
        bos_token_id=vocabulary["[BOS]"],
        eos_token_id=vocabulary["[EOS]"],
        pad_token_id=vocabulary["[PAD]"],
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    model.save_pretrained(FIXTURE_DIR, safe_serialization=True)


if __name__ == "__main__":
    main()
