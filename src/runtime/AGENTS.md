# src/runtime/ — Execution Engine, Tracing, Rollouts, Profiling & Cost Modeling

## Purpose

Orchestrates validated graph compilation into `PreparedGraph`, step-by-step deterministic evaluation, SHA-256 intermediate tensor fingerprinting, stateful temporal rollouts, latency benchmarking, and per-op differential microbenchmarking.

## Ownership

- `PreparedGraph`, `run_spec`, `eval_spec_with_inputs`, `eval_spec_differentiable`, and layer facade re-exports: [`engine.rs`](engine.rs)
- Spec-free direct Burn engine math (`eval_arith_op`, `eval_int_op`, `eval_kernel_op`, `eval_direct_batch` with `$prev`/`$<idx>` chaining): [`direct_math.rs`](direct_math.rs)
- Per-step SHA-256 trace collection, divergence localization, and `--isolate` prefix subspec extraction: [`trace.rs`](trace.rs)
- Recurrent `rollout scan`, `rollout fixpoint`, `rollout unroll` (`rollout_unroll_with_options` with `--compact-slots`), `rollout bisect`, and input-sequence parsing: [`rollout.rs`](rollout.rs)
- Warmup + iteration timing statistics (`min`, `median`, `p95`, `mean`, `max`) and `compare_specs_perf`: [`profile.rs`](profile.rs)
- Per-op isolated differential cost modeling (`run_cost_model`) & hotspot ranking: [`cost_model.rs`](cost_model.rs)

## Local Contracts

- **Validate-Before-Execute**: `PreparedGraph::compile` and `run_spec` must run `validate_spec` first and abort immediately on invalid graphs.
- **Direct Math Engine Parity**: [`direct_math.rs`](direct_math.rs) executes `arith`, checked `int`, and Lane-E `kernel` operations through [`crate::layers`](../layers/AGENTS.md) on `Device::flex()`, enforces strict same-length (no-broadcast) validation on `kernel`, enforces `+/-2^24` exactness on `int to-f32` unless `--allow-inexact` is set, and rejects division by zero even when `--wrap` is enabled.
- **No Unbounded Trace Allocation**: `trace_hash_spec` and `compare_trace_hashes` enforce `max_bytes` / `DEFAULT_MAX_TENSOR_BYTES` caps so trace inspection remains memory-bounded.
- **Recurrent State-Port Resolution & Domain Guards**: `rollout` operations must support both explicit port name (via `ExpansionReceipt`) and canonical slot index resolution, propagating updated state deterministically across ticks while rejecting `steps == 0` (`scan`, `unroll`, `bisect`), `max_steps == 0` (`fixpoint`), and non-finite or negative tolerances (`fixpoint` `tol`, `bisect` `abs_tol`).
- **Prefix Isolation Soundness**: `compare_trace_hashes(..., isolate=true)` must emit valid standalone `SpecV1` prefix subspecs stopping at the divergent step.

## Work Guidance

- Keep layer kernel definitions out of `runtime/`; call [`crate::layers`](../layers/AGENTS.md) helpers for tensor instantiation and step execution.
- Use `PreparedGraph` reuse inside tight loops (`profile`, `rollout`, `es`) to avoid redundant validation/allocation overhead where specs are static.

## Verification

- `cargo test --locked --test integration batch1_trace_rollout_profile_and_preconditioned`
- `cargo test --locked --test integration direct_math_cli_and_library_arith_int_kernel_end_to_end`
- `./target/release/math-draw smoke` (checks #1–#4, #10–#14, #22, #31)

## Child Graph Index

- Leaf domain directory; submodules listed under Ownership (`mod.rs`, `engine.rs`, `direct_math.rs`, `trace.rs`, `rollout.rs`, `profile.rs`, `cost_model.rs`).
