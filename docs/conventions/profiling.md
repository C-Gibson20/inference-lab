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
| GPU/CPU/power/thermal samplers | PROF-03 | PROF-01 record contracts |
| Cold/warm, residency, swapping, batching, and concurrency matrices | `benchmarks/`, PROF-04 | reusable profiling records |
| Retry and recovery policy | Consuming pipeline | `FailureEvent` |
| Persistent experiment results | `benchmarks/` | operational stdout logging |

Hardware samplers and experiment matrices must not be added to the reusable
record contracts merely to support one benchmark. Conversely, benchmark code
must not duplicate the timing, resource, or failure schemas already provided by
`profiling/`.

## Completion boundary for PROF-01

PROF-01 is complete when a deterministic caller can emit timing, resource, and
failure records through the structured logging stream, while a benchmark can
independently collect and analyze those records without requiring profiling
code to know the experiment design.
