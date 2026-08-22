# Evaluation benchmark

The committed `cases.jsonl` is a deterministic contract fixture, not the final
corpus-backed benchmark. It exercises the schema, evidence, split, abstention,
comparison, aggregation, and tolerance contracts.

The final benchmark must contain 30–50 real corpus-backed questions and satisfy
the coverage policy enforced by `validate_final_benchmark()`.

Validate a final benchmark with:

```sh
PYTHONPATH=src uv run python \
  scripts/validate-evaluation-benchmark.py \
  benchmarks/evaluation/manifest.json
```

The command is expected to reject the current contract fixture until the real
question set is curated and its immutable corpus evidence is available.
