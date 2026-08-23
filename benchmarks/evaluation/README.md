# Evaluation benchmark

The committed `cases.jsonl` is a deterministic contract fixture, not the final
corpus-backed benchmark. It exercises the schema, evidence, split, abstention,
comparison, aggregation, and tolerance contracts.

The final benchmark must contain 30–50 real corpus-backed questions and satisfy
the coverage policy enforced by `validate_final_benchmark()`.

Validate a final benchmark with:

```sh
uv run python \
  scripts/inference_lab/evaluation/validate-evaluation-benchmark.py \
  benchmarks/evaluation/manifest.json
```

Exit status `0` means the dataset passed loading, coverage, held-out
composition, and leakage checks. Any other status means it must not be used as
the final benchmark. The command deliberately exits with status `1` for the
current contract fixture until the real question set is curated and its
immutable corpus evidence is available.
