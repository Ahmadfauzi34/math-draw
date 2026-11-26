# src/es/ — Tree Search, Autodiff GD & Evolutionary Optimization

## Purpose

Implements graph- and parameter-level optimization engines: equivalence-verified rewrite tree search, first/second-order Burn runtime autodiff gradient descent (`solve-gd`), and gradient-free Evolutionary Strategy weight fitting (`fit`).

## Ownership

- Active trainable parameter discovery (`discover_trainable_slices`, `apply_flat_weights`) & facade exports: [`mod.rs`](mod.rs)
- Bounded BFS/best-first equivalence-verified rewrite + optional quantization tree search (`search_graph_tree`, `search_graph_tree_with_options`, `SearchOptions`): [`search.rs`](search.rs)
- Burn `Autodiff` gradient descent with backtracking line search & optional diagonal Hessian preconditioning (`solve_gd_autodiff`): [`solve_gd.rs`](solve_gd.rs)
- Deterministic seeded ES optimizer (`fit_es_with_config`, legacy `fit_es`), `EsStrategyKind` (`OpenEs`, `MuLambda`), `FitEsConfig`, and sensitivity-guided layer freezing: [`optimizer.rs`](optimizer.rs)
- Deterministic PRNG suite (`Xorshift64`, `Mulberry32`, `Pcg32`, `Philox4x32` with `seek`, `RngKind`, `AnyRng`): [`rng.rs`](rng.rs)
- Population diagnostics (`mean_std`, `diversity`, `vec_l2_norm`, `compute_flags`): [`diag.rs`](diag.rs)

## Local Contracts

- **Verified Rewrite & Quantization Acceptance**: `search_graph_tree` and `search_graph_tree_with_options` require finite `abs_tol >= 0`, `rel_tol >= 0`, and optional `quant_tol >= 0`, and may only promote rewrite or quantization nodes whose outputs pass `verify_specs` within tolerance across active evaluation cases.
- **Determinism & Reference Vectors**: Given fixed `seed`, `strategy`, `rng`, `targets`, `generations`, `pop`, `mu`, `sigma`, and `lr`, `fit_es_with_config` and `fit_es` must be bit-for-bit reproducible on the `flex` backend. `Mulberry32`, `Pcg32`, and `Philox4x32` must match published reference test vectors (`pcg-c-basic` & Random123).
- **Fail-Closed ES Hyperparameters**: `fit_es_with_config` and `fit_es` must reject `generations == 0`, `pop < 2`, odd `pop` for `OpenEs` (without silent clamping), invalid `mu` (`0` or `> pop`) for `MuLambda`, `--mu` on `OpenEs`, and non-finite or non-positive `sigma`/`lr`. Non-finite candidate losses must never corrupt parent/mean states.
- **Non-Degradation / Line Search Safety (`solve_gd_autodiff`)**: Rejects `steps == 0` and non-finite or non-positive `lr` up front; backtracking line search must never accept weight updates producing `NaN`/`Inf` losses; optional `--precondition` diagonal Hessian estimation must clamp safely when curvature is near zero or negative (`hessian_trust`).
- **Active Layer Only**: `discover_trainable_slices` excludes `embedding` indices and inactive/unused layers unless explicitly filtered, failing closed if no differentiable weights exist.

## Work Guidance

- Share weight slicing (`ActiveLayerSlice`) from [`mod.rs`](mod.rs) across optimizers rather than duplicating layer traversal logic.
- Prefer `fit_es_with_config(&SpecV1, &TargetsFile, &FitEsConfig)` for new callers while retaining `fit_es(...)` for backward compatibility.

## Verification

- `cargo test --locked --test integration rewrites_and_recurse_preserve_semantics`
- `cargo test --locked --test integration batch1_trace_rollout_profile_and_preconditioned`
- `cargo test --locked --test integration es_rng_vectors_mu_lambda_and_strict_validation`
- `./target/release/math-draw smoke` (checks #7, #8, #9, #15, #26)

## Child Graph Index

- Leaf domain directory; submodules listed under Ownership (`mod.rs`, `search.rs`, `solve_gd.rs`, `optimizer.rs`, `rng.rs`, `diag.rs`).
