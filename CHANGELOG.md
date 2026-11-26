# Changelog

All notable changes to `math-draw` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-11-26

### Added
- **Independent Pure-Rust `f64` Differential Oracle (`src/evidence/diff_ref/`, `math-draw diff-ref`)**:
  - Added an independent, Burn-free `f64` interpreter (`run_reference_f64`, `diff_ref_spec`, `SUPPORTED_REF_OPS`) covering all **87 `f32` constructors** (`nn_ops.rs`: 44 neural/conv/norm/pool/activation/custom/binary ops; `math_ops.rs`: 43 numeric/transform/linalg/statistics/reduction/probability/comparison ops, incl. ternary `fma`).
  - Added `math-draw diff-ref <spec.json> [--cases <cases.json>] [--ref <ref.json>] [--tol 1e-5]` supporting both the built-in `pure_rust_f64` oracle and external `--ref` JSON vectors (`[num...]`, `[[num...]...]`, `{"outputs": ...}`, `{"cases": [{"expected": [...]}]}`), returning `"PASS"` (exit `0`) or `"SIGNAL"` (exit `2`) with fail-closed validation on non-finite/non-positive tolerances, mismatched lengths, or unsupported integer ops without `--ref`.
  - Added strict fail-closed `diff_ref` stage to `harness run-workflow` and updated `harness capabilities`.
- **3D Geometry Domain Pack (`src/evidence/geom3/`, `src/cli/geom_cmds.rs`, `math-draw geom`)**:
  - Added pure `f64` row-major 4×4 and 3×3 geometry math (`math.rs`): `rot_x4`/`rot_y4`/`rot_z4`, Rodrigues `rot_axis_angle4`, `euler4`, `quat_from_euler`, `quat_to_mat4`, `translate4`, `scale4`, recursive `repeat` (`depth <= 1024`), `normal_mat3` (inverse-transpose), `look_at4`, `perspective4`, and Sutherland–Hodgman near-plane triangle clipping (`clip_triangle_near`).
  - Added `math-draw geom build <chain.json> [--compose] [--out <spec.json>]`, `scene-tree <tree.json> [--out <spec.json>]` (with cycle/duplicate/unknown-parent rejection), `solve-ik <ik.json> [--out <spec.json>]` (Polyak-stepped backtracking GD with joint limits and Burn `Device::flex()` end-effector verification), `project <model.json>` (single-vertex `linear(4,4)`), `batch <model.json>` (block-diagonal `linear(4N,4N)` chunked at `BATCH_MAX = 63`), and `selftest` (17/17 checks).
- **Expanded Smoke Suite (#32–#33) & Integration Tests (#20–#22)**:
  - Expanded in-binary smoke suite from 31 to 33 checks (`33/33`) and crate integration tests from 19 to 22 suites (`22/22`, including differential verification across all 87 `f32` constructors, full `geom` CLI/workflow coverage, and `muse_audit_edge_case_regressions`).

### Fixed
- **Audit & Fail-Closed Hardening**:
  - Replaced `div(gelu(x), x)` in symbolic `grad-spec` with a smooth Padé `[3/2]` `Φ(x)` approximation exact at `x = 0`, preserved compacted output/port metadata and SSA-final `--wrt` slots across `--order 2` HVP, and validated weight lengths / undefined `--wrt` slots before slicing.
  - Hardened `geom3` (`check_mat4_finite`, `mat4_to_col_major_f32_checked`, `compose_chain_checked`, order-independent `MAX_SCENE_TREE_DEPTH = 512`, finite projection/dehomogenize/perspective/normal-matrix checks, strict explicit field parsing, and `engine_ok` gating in `solve_ik`).
  - Hardened `diff-ref` (aligned `cosineSimilarity` default `eps = 1e-12`, pre-validated NN weight counts/finiteness, per-element pass rule consistent with `first_diff_index`), `direct-math` (strict per-op batch allowlists, 1D/2D `$prev` chaining, and `int --batch` flag guards), `liveness` (`auto_repair_spec` final re-validation), and `harness` workflow (`ok`-gated active spec promotion, non-empty `pipeline`, `null`/zero-bound rejection, and `CliUsage` error classification).

## [0.8.0] - 2026-11-25

### Added
- **Graph Liveness, SSA Normalization, Slot Compaction & Deterministic Auto-Repair (`src/graph/liveness.rs`, `src/graph/validate.rs`, `src/runtime/rollout.rs`)**:
  - Embedded pure static slot liveness analysis (`peak_bytes`, `peak_step`, `peak_slots`, `live_before`, per-slot `def_step`/`last_use_step`/`bytes` across `f32`/`i64` and `unary`/`binary`/`ternary` steps) into `validate --explain`.
  - Added `ssa_normalize_spec`, linear-scan physical register compaction `compact_slots_by_liveness` (`--compact-slots` on `recurse expand`, `recurse grad-spec`, `rollout unroll`, and harness `compile`/`unroll`), and deterministic 3-pass `auto_repair_spec` (`--auto-repair` on `recurse expand` and harness `compile`) emitting auditable remediation receipts (`insert_dtype_boundary_cast`, `insert_terminal_intToFloat`, `drain_unused_port`, `recursive_shard_linear`).
- **6th Built-In Macro `attention_head` (`src/graph/composer/macros.rs`)**:
  - Added `attention_head` supporting both `seq_axis1` (`[B, S, D, 1]`, including cross-attention `Sq != Skv`) and `heads_axis1` (`[B, H, S, D]`) layouts.
- **Symbolic In-Graph VJP & 2nd-Order HVP Compiler (`src/graph/grad_spec.rs`, `math-draw recurse grad-spec`)**:
  - Compiles forward `v1`/`v2` specs into standalone runnable `SpecV1` backward graphs (`--order 1` VJP) or 2nd-order Hessian-Vector Product graphs (`--order 2 --hvp-vec ...`) with symbolic `--wrt <slot|name>` and optional `--compact-slots` across 23 differentiable constructors (`linear`, `add`, `sub`, `mul`, `fma`, `div`, `sqrt`, `exp`, `log`, `sin`, `cos`, `sigmoid`, `tanh`, `transpose`, `matmul`, `concat`, `shift*`, `gelu`, `mish`, `swiGlu`).
- **Determinism Soak Prover, PRNG Stream Certification & Quantization Tree Search (`src/evidence/soak.rs`, `src/es/search.rs`)**:
  - `math-draw soak <spec.json> [--runs 100]`: proves N-run bit-identical execution on `Device::flex()` via canonical SHA-256 output digests.
  - `math-draw prng-check [mulberry32|xorshift64|pcg32|philox4x32] [seed] [count] [--out <manifest.json>]` & `math-draw prng-check verify <manifest.json>`: certifies and re-verifies deterministic PRNG streams.
  - `math-draw recurse search ... [--quant-tol T] [--quant-scheme q8-per-block|int8|fp16]`: explores equivalence-verified quantization branches alongside algebraic rewrites (`search_graph_tree_with_options`, scoring `active_weight_bytes`).
- **Direct Spec-Free Engine Math CLIs (`src/runtime/direct_math.rs`, `src/cli/math_cmds.rs`)**:
  - `math-draw arith <op> <a> [b] | --batch <ops.json>`: spec-free `Device::flex()` f32 vector/matrix/stats/probability arithmetic with length-1 broadcast, precision warnings, and `"$prev"`/`"$<idx>"` batch register chaining.
  - `math-draw int <add|sub|mul|div|rem|matmul|to-f32|from-f32> ... [--wrap] [--allow-inexact] | --batch <ops.json>`: exact checked `i64` arithmetic & casts with stringified `i64` JSON outputs.
  - `math-draw kernel <sin|cos|tan|floor|ceil|round|atan2|powf|hypot|fma> ... | --batch <ops.json>`: Lane-E elementwise kernels with strict same-length (no-broadcast) validation.
- **Expanded Harness Workflow Stages, Smoke Suite (#28–#31) & Integration Tests (#16–#19)**:
  - Added strict fail-closed `grad_spec`, `soak`, `gradcheck`, `shard`, and `bundle_export` stages to `harness run-workflow` plus `auto_repair`/`compact_slots`/`quant_tol` options on `compile`/`unroll`/`search_rewrites`, and updated `harness capabilities`.
  - Expanded in-binary smoke suite from 27 to 31 checks (`31/31`) and crate integration tests from 15 to 19 suites (`19/19`).

## [0.7.0] - 2026-11-19

### Added
- **Evolutionary Strategy (ES) Engine Parity (`src/es/rng.rs`, `src/es/diag.rs`, `src/es/optimizer.rs`)**:
  - Ported 4 deterministic PRNG streams (`Xorshift64`, `Mulberry32`, `Pcg32`, and counter-based `Philox4x32-10` with random-access `seek(step)`), verified against published `pcg-c-basic` and Random123 test vectors.
  - Added `(mu, lambda)` elitist selection (`EsStrategyKind::MuLambda`, `--strategy mu-lambda [--mu N]`) alongside antithetic `OpenEs` (`--strategy openes`).
  - Added per-generation population diagnostics (`worst_loss`, `mean_loss`, `std_loss`, `improvement`, `stagnation`, population `diversity`, `mean_norm`, `best_norm`, and health `flags`: `stagnation`, `low_diversity`, `non_finite_loss`) to `recurse fit` and harness workflow `fit` stages.
- **Portable Program Bundle & Canonical Parameter Binding (`src/evidence/bundle.rs`, `math-draw bundle`)**:
  - `math-draw bundle binding <spec.json>`: computes unique-first-use parameter ownership (`burn-math.graph-parameter-binding.v1`) deduplicating weight-tied layers across `spec.steps`, plus weight/input-invariant `program_identity` (`sha256:...`), `read_bound_flat_weights`, and `apply_bound_flat_weights`.
  - `math-draw bundle export <spec.json> [--no-exec] [--attest <stage>]... [--out <bundle.json>]`: packages `SpecV1`, parameter binding, optional live `Device::flex()` execution receipt, and SHA-256 provenance attestations into a self-contained `burn-math.program-bundle.v1` artifact (`bundle_sha256`).
  - `math-draw bundle verify <bundle.json>`: performs 6-check cryptographic integrity and live replay verification (`schema_version`, `bundle_digest`, `static_validation`, `parameter_binding`, `execution_replay`, `attestation_provenance`), failing closed on weight, receipt, schema, or attestation tampering.
- **Structured Error Taxonomy & JSON Error Envelopes (`src/error.rs`, `src/cli/`, `src/harness/workflow.rs`)**:
  - Added [`math_draw::error`](src/error.rs) exposing 31 machine-readable `ErrorCode` variants (`snake_case`), `MathDrawError`, backward-compatible `ErrorEnvelope` (`ok: false`, `code`, `error`, optional `stage`, optional `hint`), and deterministic `classify_error`.
  - Upgraded all CLI error exits and failed workflow stage reports (`stage_reports` + top-level workflow output) to emit structured `code`, `stage`, and `hint` fields.
- **Expanded Smoke Suite (#26–#27) & Integration Test Suites (#11–#15) (`src/smoke/advanced_checks.rs`, `tests/integration.rs`)**:
  - Added smoke check `#26` (`es_strategies_rngs_and_diagnostics`) and `#27` (`program_bundle_binding_and_verify`) (`27/27` passing).
  - Added 5 integration test suites (`es_rng_vectors_mu_lambda_and_strict_validation`, `program_bundle_binding_replay_and_tamper_detection`, `error_handling_phase1_cli_and_library_fail_closed`, `error_handling_phase2_typed_codes_and_envelopes`, `error_handling_phase3_workflow_schema_and_domain_guards`; `15/15` passing).

### Changed
- **End-to-End Fail-Closed Error Handling (Phases 1–3)**:
  - **Phase 1 (P0–P1)**: Replaced permissive CLI flag scanning with strict `ParsedArgs` allowlist validation across all subcommands (rejecting unknown flags, missing flag values, duplicate single-valued flags, unexpected extra positionals, and malformed/non-finite/out-of-domain numeric options) and eliminated 100% of production `.unwrap()`/`.expect()` calls across `src/**/*.rs`.
  - **Phase 2 (P2)**: Standardized all CLI and workflow errors on `ErrorEnvelope` (`ok`, `code`, `error`, `stage`, `hint`) with exit code `2`.
  - **Phase 3 (P3)**: Enforced strict root and per-stage JSON field allowlists, value type checks, and numeric domain bounds in `execute_harness_workflow`, plus uniform numeric domain boundary guards across public Rust library APIs (`solve_gd_autodiff`, `search_graph_tree`, `fit_es`, `verify_specs`, `zoom_fuzz_specs`, `quant_verify`, `equiv_linear`, `rollout_bisect`, `mutate_spec`, `merge_subagent_branches`, `spectral_radius`, `gate_sets`).

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
