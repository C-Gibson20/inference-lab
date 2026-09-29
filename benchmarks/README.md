# Benchmarks

Experiment definitions and result artefacts live here. Reusable profiling code
belongs in `src/inference_lab/profiling`.

RUN-04's [candidate pilot](run-04/candidate-pilot.md) records provisional
development-data evidence for Qwen3-Embedding-0.6B and BGE-M3. The
[local measurement procedure](run-04/measurement-report-template.md) records
repeatable load and throughput observations for the pinned Qwen adapter using
fixed, non-corpus text inputs. It also captures the RUN-03 tensor-memory
observation and identifies resource categories the current sampler cannot
measure. These results do not replace representative corpus evaluation or
held-out EVAL-04.
