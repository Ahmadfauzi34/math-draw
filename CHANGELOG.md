# Changelog

All notable changes to `math-draw` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-18

### Added
- **Step Trace Fingerprinting & Divergence Localizer (`trace-hash`, `src/trace.rs`, `src/engine.rs`)**:
  - `trace-hash <spec.json> [other.json] [--max-bytes N] [--isolate]` computes per-step SHA-256 tensor digests with configurable byte-budget skipping (`tensor_budget_exceeded`), detects `first_divergent_step` + reason, and extracts validated prefix subspecs with `--isolate`.
- **Stateful Rollout Engine (`rollout <scan|fixpoint|unroll|bisect>`, `src/rollout.rs`)**:
  - Added cached `PreparedGraph` execution and named `v2` port resolution (`ExpansionReceipt.port_names`).
  - `rollout scan`: multi-tick recurrent/autoregressive execution tracking `delta_linf` and state SHA-256 hashes (supports `--inputs-seq`).
  - `rollout fixpoint`: converges until `||s_{t+1} - s_t||_∞ <= tol` and reports Lipschitz contraction ratio estimates.
  - `rollout unroll`: materializes an $N$-step weight-tied unrolled `SpecV1` graph.
  - `rollout bisect`: locates both the first divergent rollout `tick` and exact intra-tick `first_divergent_step`.
- **Benchmarking & Comparison (`profile`, `compare`, `src/profile.rs`)**:
  - `profile`: reports `compile_ms`, `min_ms`, `median_ms`, `p95_ms`, `mean_ms`, `max_ms`, `steps_count`, `total_params`, and `outputs_sha256`.
  - `compare`: benchmarks baseline vs candidate specs alongside equivalence verification (`outputs_match`, `max_abs_diff`, `speedup`, step/param deltas).
- **Preconditioned GD & Sensitivity-Guided ES (`src/recurse.rs`)**:
  - `recurse solve-gd --precondition`: adds second-order central-difference diagonal Hessian estimation with cross-step trust classification (`trusted` / `zero_within_noise` / `untrusted`) and damped inverse curvature scaling.
  - `recurse fit --sensitivity-guided [--freeze-threshold 1e-6]`: evaluates per-layer sensitivity (`max |dL/dw|`) and freezes low-sensitivity layers during ES optimization.
- **Expanded Verification (`src/smoke.rs`, `tests/integration.rs`)**:
  - Expanded built-in `smoke` suite from 10 to 15 checks and added `batch1_trace_rollout_profile_and_preconditioned` integration test.

## [0.1.0] - 2026-11-18

### Added
- **Standalone Rust Single-Binary CLI (`math-draw`)**: built with Rust edition `2024` and Burn `0.22.1` (`burn-flex` CPU backend + runtime `autodiff`), zero Node/WASM runtime dependencies.
- **Full 52-Constructor Catalog (`src/catalog.rs`, `src/engine.rs`)**:
  - Activations: `relu`, `gelu`, `sigmoid`, `tanh`, `hardSwish`, `leakyRelu`, `prelu`, `swiGlu`, `hardSigmoid`, `softplus`, `mish`, `softmax`, `logSoftmax`, `glu`, `sqrt`.
  - Linear, Norms & Embedding: `linear`, `batchNorm`, `groupNorm`, `instanceNorm`, `layerNorm`, `rmsNorm`, `embedding`.
  - Convolutions & Pooling: `conv1d`, `conv2d`, `convTranspose2d`, `maxPool1d`, `maxPool2d`, `avgPool1d`, `avgPool2d`, `adaptiveAvgPool2d`.
  - Custom & Transform: `featureNorm`, `shiftUp`, `shiftDown`, `shiftLeft`, `shiftRight`, `ghost`, `seBlock`, `transpose`.
  - Binary `f32`: `add`, `sub`, `mul`, `matmul`, `div` (fail-closed finite/nonzero checks), `concat`.
  - Exact Checked `i64` Ops: `intAdd`, `intSub`, `intMul`, `intMatmul`, `intDiv`, `intRem`, `intToFloat`, `intFromFloat`.
- **Recursive Spec Compiler (`v1` & `burn-math.recursive-spec.v2`, `src/composer.rs`)**:
  - Supports `subgraphs` with `$param` substitution, `repeat` (`depth`, `tie_weights`, `weights_per_depth`), `residual`, `parallel_concat`, `$ref` file includes, and built-in macros (`mlp`, `gated_unit`, `residual_mlp`, `transformer_ffn`, `recurrent_cell`).
- **Static Validator & Explainer (`validate [--explain]`, `src/validate.rs`)**:
  - Checks constructor validity, argument bounds, unique registry IDs, SSA dataflow, 4D shape inference, mixed `f32`/`i64` dtype flow, and port consumption.
- **Verification & Graph Rewrites (`verify`, `rewrite`, `src/verify.rs`, `src/rewrite.rs`)**:
  - Equivalence verification with domain-aware fuzzing (`--fuzz`), plus pure rewrite rules (`linear-branch-fusion`, `linear-chain-fusion`, `relu-idempotent`, `transpose-involution`, `dead-step-elimination`, `add-commute`).
- **Recursive Agent Engineering Subcommands (`recurse`, `src/recurse.rs`)**:
  - `recurse expand`: compiles `v2` recursive specs to canonical flat `v1`.
  - `recurse search`: tree rewrite search verified against the baseline.
  - `recurse solve-gd`: Burn `0.22.1` `Device::flex().autodiff()` gradient descent with reflective backtracking.
  - `recurse fit`: gradient-free Evolutionary Strategy optimizer across active weighted layers.
- **Built-in Smoke Suite & Integration Tests (`smoke`, `tests/integration.rs`)**:
  - 10 end-to-end smoke checks and integration coverage across all 52 constructors.
- **CI & Release Workflows (`.github/workflows/ci.yml`, `.github/workflows/release.yml`)**:
  - Automated Linux `x86_64` testing, smoke verification, and release artifact packaging (`tar.gz` + `.sha256`).
