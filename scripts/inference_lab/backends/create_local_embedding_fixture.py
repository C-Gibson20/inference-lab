"""Create the small local encoder fixture used by RUN-04 acceptance tests."""

import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import AutoModel, PreTrainedTokenizerFast, Qwen3Config

FIXTURE_DIR = Path("tests/fixtures/backends/runtime/local_embedding")
SEED = 20260929


def _fixture_revision(files: dict[str, str]) -> str:
    """Return a stable revision identifier from the sorted fixture inventory."""
    digest = hashlib.sha256()
    for name, file_digest in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_digest))
    return "fixture-" + digest.hexdigest()


def _file_sha256(path: Path) -> str:
    """Return a streamed SHA-256 digest for one generated file."""
    digest = hashlib.sha256()
    with path.open("rb") as artefact:
        while chunk := artefact.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Write deterministic Qwen-compatible tokenizer and encoder files."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    vocabulary = {
        "[UNK]": 0,
        "[PAD]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
    }
    for token in (
        "Instruct:",
        "Given",
        "a",
        "scientific",
        "literature",
        "search",
        "query,",
        "retrieve",
        "passages",
        "that",
        "answer",
        "the",
        "query",
        "Query:",
        "SARM1",
        "inhibition",
        "reduces",
        "neuroinflammation",
        "injury",
        "changes",
        "response",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
    ):
        vocabulary[token] = len(vocabulary)

    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()  # pyright: ignore[reportAttributeAccessIssue]
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    fast_tokenizer.save_pretrained(FIXTURE_DIR)

    torch.manual_seed(SEED)
    config = Qwen3Config(
        vocab_size=len(vocabulary),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        bos_token_id=vocabulary["[BOS]"],
        eos_token_id=vocabulary["[EOS]"],
        pad_token_id=vocabulary["[PAD]"],
    )
    model = AutoModel.from_config(config)
    model.eval()
    model.save_pretrained(FIXTURE_DIR, safe_serialization=True)

    files = {
        path.name: _file_sha256(path)
        for path in sorted(FIXTURE_DIR.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "fixture_version": "1.0.0",
        "seed": SEED,
        "vocab_size": len(vocabulary),
        "revision": _fixture_revision(files),
        "files": files,
    }
    (FIXTURE_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
