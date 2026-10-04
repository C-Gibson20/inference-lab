# Benchmarks

Experiment definitions and result artefacts live here. Reusable profiling code
belongs in `src/inference_lab/profiling`.

RUN-04's [candidate pilot](run-04/candidate-pilot.md) records provisional
development-data evidence for Qwen3-Embedding-0.6B and BGE-M3. The
[sample audit](run-04/corpus-sample-audit.md) and
[expanded development cases](run-04/development-cases-expanded.jsonl) provide a
broader set whose page labels received an independent second-pass review; see
the [expanded candidate comparison](run-04/expanded-candidate-comparison.md)
for results and the audit for remaining limitations.
[Provisional retrieval baseline](evaluation/placeholder-baseline.md) records
the sample's current development scores as a starting reference; it is not the
final EVAL-04 held-out gate.
[local measurement procedure](run-04/measurement-report-template.md) records
repeatable load and throughput observations for the pinned Qwen adapter using
fixed, non-corpus text inputs. It also captures the RUN-03 tensor-memory
observation and identifies resource categories the current sampler cannot
measure. The latest [verified local Qwen measurement](run-04/results/qwen-local-measurement.json)
does not replace representative corpus evaluation or held-out EVAL-04.
