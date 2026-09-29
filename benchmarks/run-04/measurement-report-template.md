# RUN-04 local embedding measurement

Copy this template for each local measurement run. Keep generated output in
`benchmarks/run-04/results/`; do not commit private source passages or raw model
artefacts.

## Reproduction

```sh
uv run python scripts/inference_lab/backends/benchmark_local_embedding.py \
  --model-path /path/to/pinned/local/model \
  --output benchmarks/run-04/results/qwen-local-measurement.json
```

The command reads the fixed text inputs in `throughput-inputs.jsonl`. It stores
only their SHA-256 digest and aggregate token counts in the result. It does not
write the model path or input text into the report.

## Run identity

- Date and time (UTC):
- Host and hardware:
- Python, PyTorch, and Transformers versions:
- Model name and immutable revision:
- Model artefact checksum verification:
- Device and inference dtype:
- Input fixture SHA-256:
- Input count per batch:
- Backend internal batch size:
- Warmup iterations:
- Measured iterations:
- Timing method: `time.perf_counter_ns`; one cold local load, warmups, then sequential repeated batches

## Results

- Cold local load time (ms):
- Median batch duration (ms):
- Embeddings per second:
- Input tokens per second:
- Total measured input tokens:
- Output dimensions:
- Measured model tensor storage (bytes):
- Memory measurement status and adapter calibration:

## Measurement limits

List each relevant category as measured, unavailable, or unmeasured. The
included script measures unique model tensor storage through RUN-03's PyTorch
calibration adapter. It does not measure process RSS, allocator reservation,
activation peaks, framework overhead, power, or thermal behaviour. The result
is a local observation, not a hardware-independent acceptance threshold.

## Interpretation

Describe whether repeated runs used the same host, software versions, model
revision, input digest, warmup count, and measured iteration count. Explain
material differences before comparing throughput or memory figures.
