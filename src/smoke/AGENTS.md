# src/smoke/ — Deterministic In-Binary Smoke Suite

## Purpose

Embeds the 33-check end-to-end verification suite directly in `math-draw` (`math-draw smoke`) so any release binary can self-verify core execution, `v2` recursion, verification, autodiff, quantization, sharding, symbolic equivalence, cost modeling, Layer-4 workflows, all ported pure-math families, ternary `fma`, ES strategies/PRNGs/diagnostics, portable program bundles, liveness/compaction/auto-repair/`attention_head`, symbolic VJP/HVP `grad-spec`, soak/PRNG certification/quant-search, direct `arith`/`int`/`kernel` math, pure-Rust `f64` differential oracle (`diff-ref`), and 3D geometry (`geom3`) without external fixtures.

## Ownership

- `SmokeCheck`, `SmokeReport`, timing/assertion helpers (`run_check`, `approx_eq_slice`), and suite runner (`run_smoke_suite`): [`mod.rs`](mod.rs)
- Checks #1–#15 (execution parity, `v2` repeat/macros, validation, fuzz verify, tree search, autodiff GD, ES fit, mixed `i64` + batch isolation, trace isolation, rollout scan/fixpoint/unroll/bisect, profile/compare, preconditioned GD & sensitivity ES): [`core_checks.rs`](core_checks.rs)
- Checks #16–#33 (gradcheck autodiff, quant verify/budget, multi-branch mutate, linear shard, weight-lab selftest, linear-equiv, cost-model + zoom-fuzz, harness digest & workflow, ported pure-math end-to-end pipeline, 13 parity math ops + ternary `fma` via `v2`, ES strategies/PRNGs/diagnostics, portable program bundle + tamper gate, liveness/compaction/auto-repair/`attention_head`, symbolic VJP/HVP `grad-spec`, soak/PRNG manifest/quant-search, direct `arith`/`int`/`kernel` + batch chaining, `diff-ref` `f64` oracle + tamper SIGNAL, `geom3` 17-check selftest + scene-tree + IK + batched projection): [`advanced_checks.rs`](advanced_checks.rs)

## Local Contracts

- **Zero External Dependency**: Every check must construct its JSON inputs inline and clean up any temporary directories under `std::env::temp_dir()`.
- **Strict All-Pass Gate**: `SmokeReport.ok` is true if and only if `passed == total`; any single check failure causes exit code `2` from the CLI and fails `tests/integration.rs::smoke_suite_passes_all_checks`.
- **Deterministic Outcomes**: Tolerances and seeds must produce consistent results across Linux `x86_64` CI and local developer machines.

## Work Guidance

- Keep checks fast (millisecond budget per check) and clearly numbered with concise `detail` strings for diagnostic readability.
- When adding a new major feature batch, append dedicated smoke checks to `advanced_checks.rs` (or split out a dedicated batch module if `advanced_checks.rs` exceeds ~500 lines).

## Verification

- `cargo test --locked --test integration smoke_suite_passes_all_checks`
- `./target/release/math-draw smoke` (33/33 checks passing)

## Child Graph Index

- Leaf domain directory; submodules listed under Ownership (`mod.rs`, `core_checks.rs`, `advanced_checks.rs`).
