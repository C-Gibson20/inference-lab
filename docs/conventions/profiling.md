# Profiling and benchmark boundary

Inference Lab separates reusable profiling instrumentation from
experiment-specific benchmark work. This keeps operational instrumentation
stable across ingest, retrieval, serving, runtime, and evaluation while
allowing benchmark designs to change without changing the product boundary.

## Profiling package ownership

The `src/inference_lab/profiling/` package owns reusable, production-safe
instrumentation and record contracts, including:

- record schemas and validation;
- deterministic clocks and stage timers;
- resource-measurement contracts;
- failure and degraded-unit contracts;
- structured-log emission adapters.

Profiling records must remain independent of any particular benchmark,
experiment name, model-comparison policy, aggregation method, or result-storage
format.

## Record envelope and schema compatibility

Every concrete profiling record contains `record_schema_version`. Producers use
`CURRENT_RECORD_SCHEMA_VERSION`, and constructors reject versions not listed in
`SUPPORTED_RECORD_SCHEMA_VERSIONS`. A newer positive integer is not assumed to
be compatible merely because it is syntactically valid.

Every concrete record supplies its logging level. The structured-log emission
adapter rejects records with a missing or unsupported level; it does not infer
severity from other fields.

Profiling records use `observed_at` for the UTC time at which the stage event,
resource value, or failure was observed. This field survives serialization and
structured-log emission unchanged. The logging pipeline separately generates
the required `timestamp` field, which records when the log event was emitted.
Benchmark collectors must use `observed_at` for measurement ordering and may use
`timestamp` to diagnose logging delay.

## Canonical resource measurements

Resource units are determined by category and are not caller-configurable.
Samplers must normalize values before constructing `ResourceMeasurement`:

| Categories | Canonical unit | Valid range |
| --- | --- | --- |
| Model weights, KV cache, activations, embedding/reranker memory, index memory, framework overhead, pinned host memory, storage cache | `bytes` | Greater than or equal to zero |
| GPU and CPU utilization | `percent` | `0` through `100`, inclusive |
| Storage throughput | `bytes_per_second` | Greater than or equal to zero |
| Power | `watts` | Greater than or equal to zero |
| Thermal | `celsius` | Greater than or equal to `-273.15` |

All present values are finite numbers and booleans are not measurements.
`measured` records require a value, `unavailable` records prohibit one, and
`unreliable` records may retain an observed value. Sampled and interval records
require a positive finite `sample_duration_ms`; point-in-time records prohibit
that field.

## Failure privacy and instrumentation errors

`FailureEvent.exception_type`, when present, contains only a bare or
module-qualified exception class name such as `RuntimeError` or
`builtins.RuntimeError`. Exception messages, tracebacks, paths, and payloads do
not belong in profiling records.

`StageTimer` never replaces an exception raised by the measured stage with a
failure raised while emitting the completion record. When the measured stage
succeeds, a completion-emission failure is raised to the caller so missing
instrumentation is not silent. Retry and recovery policy remains the consuming
pipeline's responsibility.

## Benchmark ownership

The top-level `benchmarks/` directory owns experiment-specific material:

- experiment definitions and parameter matrices;
- benchmark fixtures and evaluation setup;
- collected measurements and result artifacts;
- comparison reports and performance conclusions.

Benchmark code consumes profiling records and decides how to aggregate,
compare, and report them. Operational profiling output remains on the
structured logging stream unless a benchmark explicitly collects it as an
experiment artifact.

## Ownership boundary

| Concern | Owner | Not owned by |
| --- | --- | --- |
| Stage timing and record serialization | `profiling/` | `benchmarks/` |
| Resource and failure record schemas | `profiling/` | `benchmarks/` |
| GPU/CPU/power/thermal samplers | `profiling/` sampler integrations | profiling record schemas |
| Cold/warm, residency, swapping, batching, and concurrency matrices | `benchmarks/` | reusable profiling records |
| Retry and recovery policy | Consuming pipeline | `FailureEvent` |
| Persistent experiment results | `benchmarks/` | operational stdout logging |

Hardware samplers and experiment matrices must not be added to the reusable
record contracts merely to support one benchmark. Conversely, benchmark code
must not duplicate the timing, resource, or failure schemas already provided by
`profiling/`.

## Record-layer completion boundary

The profiling record layer is complete when a deterministic caller can emit
timing, resource, and failure records through the structured logging stream,
while a benchmark can independently collect and analyze those records without
requiring profiling code to know the experiment design.
