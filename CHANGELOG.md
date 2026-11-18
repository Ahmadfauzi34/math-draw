# Changelog

All notable changes to `math-draw` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-11-18

### Added
- **Full 95-Constructor Pure-Math Kernel Parity & Ternary `fma` (`src/catalog.rs`, `src/spec.rs`, `src/layers/math_numeric.rs`, `src/layers/math_stats_prob.rs`)**:
  - Ported the remaining 13 pure-math constructors from `ai_agent_builder` (catalog expanded `82 -> 95`), achieving 100% pure-math kernel parity:
    - **Numeric Unary (7)**: `sin`, `cos`, `tan`, `floor`, `ceil`, `round` (half-to-even), and `allFinite` (emits canonical `[1, 1, 1, 1]` float `1.0`/`0.0`).
    - **Numeric Binary (3)**: `atan2(y, x)`, `powf(base, exp)` (fail-closed NaN/Inf/overflow checks), and overflow-safe scaled `hypot(x, y)`.
    - **Numeric Ternary (1)**: `fma(a, b, c)` using `f32::mul_add` with strict 3-way shape match and `"ternary"` step kind (`slots: [a, b, c, out]`).
    - **Sample Statistics (2)**: `varianceSample` and `stdSample` over `[B, F, 1, 1]` with Bessel correction (`ddof = 1`, fail-closed `F < 2`, two-pass `f64`).
- **End-to-End Ternary Arity Integration (`src/graph/`, `src/runtime/`, `src/evidence/`)**:
  - Added [`StepDecl`](src/spec.rs) arity helpers (`expected_arity`, `kind_for_arity`, `input_slots`, `output_slot`) and wired `"ternary"` steps cleanly across [`validate.rs`](src/graph/validate.rs), `v2` [`composer`](src/graph/composer/mod.rs), [`mutate.rs`](src/graph/mutate.rs), [`rewrite.rs`](src/graph/rewrite.rs), [`engine.rs`](src/runtime/engine.rs), [`trace.rs`](src/runtime/trace.rs), [`rollout.rs`](src/runtime/rollout.rs), [`cost_model.rs`](src/runtime/cost_model.rs), [`linear_equiv.rs`](src/evidence/linear_equiv.rs), and [`quant.rs`](src/evidence/quant.rs) (`fma`/`sin`/`cos`/`hypot`/`atan2`/`allFinite` bounds + fail-closed `bound` refusals for `tan`, `powf`, `floor`, `ceil`, `round`, `allFinite`).
- **Smoke Suite #25 & Integration Coverage (`src/smoke/advanced_checks.rs`, `tests/integration.rs`)**:
  - Added smoke check `#25` (`parity_13_math_ops_and_ternary_fma`, `25/25` passing), renamed/expanded `catalog_has_all_95_constructors` & `all_95_constructors_execute_cleanly`, and added `parity_13_math_ops_fail_closed_and_ternary_pipelines` (`10/10` passing).

## [0.5.0] - 2026-11-18

### Added
- **Full 82-Constructor Pure-Math Parity (`src/catalog.rs`, `src/graph/validate.rs`, `src/layers/math_numeric.rs`, `src/layers/math_stats_prob.rs`)**:
  - Ported all 30 remaining pure-math primitives from `ai_agent_builder` (catalog expanded `52 -> 82`) with strict fail-closed validation, 4D shape inference, and deterministic Burn `0.22.1` execution kernels:
    - **Numeric (4)**: `abs`, `exp` (overflow check), `log` (positive finite domain check), `clamp(min, max)` (`min <= max`).
    - **Tensor Transforms (6)**: `reshape(shape)`, `permute(axes)`, `slice(starts, ends)`, `selectAxis(axis, indices)`, `fillLike(scalar)`, `expandLike`.
    - **Linear Algebra (4)**: `dot`, `l2Norm`, `l2Distance`, `cosineSimilarity(epsilon?)` over `[B, F, 1, 1]`.
    - **Statistics (6)**: `sum`, `mean`, `min`, `max`, plus two-pass `f64` `variancePopulation` and `stdPopulation` to avoid intermediate `f32` square overflows on wide-spread inputs.
    - **Axis Reductions (4)**: keepdim rank-4 reducers `sumAxis`, `meanAxis`, `minAxis`, `maxAxis`.
    - **Probability (4)**: `normalize`, `entropy`, `crossEntropy`, `klDivergence` with non-negativity, unit-mass (`1e-5`), and `Q` zero-support enforcement.
    - **Comparison & Index Source (2)**: `lessEqual01` (exact canonical `0.0`/`1.0`, no implicit broadcast) and `indicesLike(axis)` (exact-`f32` coordinate bound `<= 2^24 + 1`).
  - Added [`ParamType::U32Array`](src/catalog.rs) and `arg_usize_vec` arg parsing for rank-4 shape/axis/slice/index array parameters.
- **Subsystem Integration & Expanded Verification (`src/evidence/quant.rs`, `src/runtime/cost_model.rs`, `src/smoke/advanced_checks.rs`, `tests/integration.rs`)**:
  - Added analytical Lipschitz bounds and forward value bounds in `quant budget --mode bound` for bounded new ops (`abs`, `clamp`, `reshape`, `permute`, `slice`, `selectAxis`, `expandLike`, `fillLike`, `indicesLike`, `mean*`, `min*`, `max*`) while keeping unbounded/domain-sensitive ops (`exp`, `log`, etc.) fail-closed with `--mode measured` guidance.
  - Updated `cost-model` synthetic probe generators for domain-sensitive ops (`log`, `normalize`, `entropy`, `crossEntropy`, `klDivergence`).
  - Added smoke check `#24` (`ported_math_ops_end_to_end`, `24/24` passing) and integration tests `catalog_has_all_82_constructors`, `all_82_constructors_execute_cleanly`, and `ported_math_ops_domain_and_shape_fail_closed` (`9/9` passing).

### Changed
- **Modular Source Architecture & Graph Framework Contracts (`src/`, `AGENTS.md`)**:
  - Reorganized flat modules into cohesive domain subtrees (`src/layers/`, `src/runtime/`, `src/graph/`, `src/es/`, `src/evidence/`, `src/harness/`, `src/smoke/`, `src/cli/`) while preserving backward-compatible root facades in `src/lib.rs`, with hierarchical `AGENTS.md` contracts across root, module trees, `tests/`, and `examples/`.

## [0.4.0] - 2026-11-18

### Added
- **Manual-Weighting Workbench (`weight-lab`, `src/weight_lab.rs`)**:
  - `weight-lab <run|gate|certify|compose|eval|selftest>`: row-major ($m \times n$) to Burn column-major (`to_column_major`) weight verification against the forward engine (`engine_agreement`), 2-step oscillatory-safe power iteration `spectral_radius`, `orthonormality_error`, `frobenius_norm`, Gaussian elimination `det`, property checker (`check_properties`), candidate transition gate (`gate_sets`), cryptographic certificate emitter (`certify_weight_set`), 3-tier 2-stage composition prover (`compose_properties`: `norm_bound` / `orthonormal` / `measured`), recursive weight expression evaluator (`evaluate_weight_expr`: `literal` / `compose` / `blend` / `scale` / `power`), and 9-case `selftest`.
- **Symbolic Linear Equivalence Prover (`linear-equiv`, `src/linear_equiv.rs`)**:
  - `linear-equiv <specA.json> <specB.json> [--tol 0.0]` reduces multi-port affine DAGs (`linear`, `add`, `sub`, `concat(dim=1)`, and constant `role: "context"` ports) to canonical $(W_{\text{eff}}, b_{\text{eff}})$ and proves exact or $\varepsilon$-bounded symbolic equivalence across sharded or fused graphs.
- **Isolated Per-Step Cost Model (`cost-model`, `src/cost_model.rs`)**:
  - `cost-model <spec.json> [--iters 11] [--table <out.json>]` benchmarks isolated per-step micro-specs (`<op> + add(0)` minus `add(0)` baseline for `f32` unary ops, isolated binary/int/embedding harnesses), reporting `ms_median`, `share_pct`, `hotspot`, and `predicted_total_ms` vs `measured_total_ms` (`prediction_error_pct`).
- **Recursive Adversarial Interval Narrowing (`verify --zoom`, `src/verify.rs`)**:
  - `verify <base.json> <cand.json> --zoom [--zoom-depth 4] [--fuzz 12] [--seed S] [--lo -5] [--hi 5]` iteratively shrinks the sampling interval around the worst-case fuzz input at each depth (`zoom_fuzz_specs`).
- **Layer-4 Agent-Harness Co-Processor (`harness`, `src/harness.rs`)**:
  - `harness <capabilities|digest|resolve-artifact|run-workflow|branch-export|branch-merge|workspace-status>` provides progressive domain discovery (`capabilities`), context-preserving semantic tensor digests (`digest`), content-addressed `.burn-harness` artifact storage (`burn://sha256:<64-hex>`) with fail-closed tamper detection (`artifact_integrity_failed`), subagent branch export/merge with multi-case equivalence gating, and multi-stage pipeline execution (`run-workflow`).
- **Expanded Smoke Suite & Integration Tests (`src/smoke.rs`, `tests/integration.rs`)**:
  - Expanded built-in `smoke` suite from 19 to 23 checks (`#20`–`#23`) and added `batch3_weight_lab_linear_equiv_cost_model_and_harness` integration test.

## [0.3.0] - 2026-11-18

### Added
- **Numeric Jacobian & Autodiff Gradient Checker (`gradcheck`, `src/gradcheck.rs`)**:
  - `gradcheck <spec.json> [analytic.json] [--eps 1e-3] [--tol 1e-3] [--weights] [--autodiff]` computes central finite-difference Jacobians for `f32` input ports (skipping `i64` ports via `port_is_i64_mask`) and active trainable weights (`--weights`), compares against optional analytic matrices, and cross-checks weight gradients against Burn `Device::flex().autodiff()` (`--autodiff`).
- **Weight Quantization Verification & Mixed-Precision Planner (`quant <verify|budget>`, `src/quant.rs`)**:
  - Supports `int8`, `q8-per-block` (block size 32), and pure-Rust IEEE-754 binary16 round-to-nearest-even (`fp16`) quantization schemes (`f32_to_f16_roundtrip`).
  - `quant verify`: quantizes all weighted layers and runs end-to-end equivalence verification with byte compression metrics.
  - `quant budget`: plans per-layer mixed precision under `--tol` using either analytical DAG Lipschitz propagation (`--mode bound`, failing closed on unbounded downstream ops such as `div` or `sqrt`) or empirical greedy measurement (`--mode measured`).
- **Transactional & Multi-Branch Equivalence-Gated Graph Mutation (`mutate`, `src/mutate.rs`)**:
  - `mutate <spec.json> <mutation.json> [--cases ...] [--abs-tol 1e-6] [--rel-tol 1e-6] [--out <out.json>]` applies `setWeights`, `replaceStep`, `insertStep`, `removeStep`, `reconnectStep`, and `setOutput` across single transactions or multi-branch candidate sets (`branches`, `promote_branch`) with a fail-closed equivalence gate on `commit`.
- **Linear Layer Output-Row Sharding (`shard`, `src/shard.rs`)**:
  - `shard <spec.json> [--max-weights 65535] [--out <out.json>]` splits oversized `linear` layers along output rows (`out_dim`), chains `concat(dim=1)` steps, verifies exact weight reassembly (`reassembly_exact`), and runs full equivalence verification.
- **Expanded Smoke Suite & Integration Tests (`src/smoke.rs`, `tests/integration.rs`)**:
  - Expanded built-in `smoke` suite from 15 to 19 checks (`#16`–`#19`) and added `batch2_gradcheck_quant_mutate_and_shard` integration test.

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
