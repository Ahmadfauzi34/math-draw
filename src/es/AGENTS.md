# src/es/ — Tree Search, Autodiff GD & Evolutionary Optimization

## Purpose

Implements graph- and parameter-level optimization engines: equivalence-verified rewrite tree search, first/second-order Burn runtime autodiff gradient descent (`solve-gd`), and gradient-free Evolutionary Strategy weight fitting (`fit`).

## Ownership

- Active trainable parameter discovery (`discover_trainable_slices`, `apply_flat_weights`) & facade exports: [`mod.rs`](mod.rs)
- Bounded BFS/best-first equivalence-verified rewrite tree search (`search_graph_tree`): [`search.rs`](search.rs)
- Burn `Autodiff` gradient descent with backtracking line search & optional diagonal Hessian preconditioning (`solve_gd_autodiff`): [`solve_gd.rs`](solve_gd.rs)
- Deterministic seeded ES optimizer with sensitivity-guided layer freezing (`fit_es`): [`optimizer.rs`](optimizer.rs)

## Local Contracts

- **Verified Rewrite Acceptance**: `search_graph_tree` may only promote rewrite nodes whose outputs pass `verify_specs` within `abs_tol`/`rel_tol` across active evaluation cases.
- **Determinism (`fit_es`)**: Given fixed `seed`, `targets`, `generations`, `pop`, `sigma`, and `lr`, `fit_es` must be bit-for-bit reproducible on the `flex` backend.
- **Non-Degradation / Line Search Safety (`solve_gd_autodiff`)**: Backtracking line search must never accept weight updates producing `NaN`/`Inf` losses; optional `--precondition` diagonal Hessian estimation must clamp safely when curvature is near zero or negative (`hessian_trust`).
- **Active Layer Only**: `discover_trainable_slices` excludes `embedding` indices and inactive/unused layers unless explicitly filtered, failing closed if no differentiable weights exist.

## Work Guidance

- Share weight slicing (`ActiveLayerSlice`) from [`mod.rs`](mod.rs) across optimizers rather than duplicating layer traversal logic.
- Note clippy `too_many_arguments` on `fit_es` if refactoring signatures in future releases (keep public CLI/library parity unless intentionally bumping minor version).

## Verification

- `cargo test --locked --test integration rewrites_and_recurse_preserve_semantics`
- `cargo test --locked --test integration batch1_trace_rollout_profile_and_preconditioned`
- `./target/release/math-draw smoke` (checks #7, #8, #9, #15)

## Child Graph Index

- Leaf domain directory; submodules listed under Ownership (`mod.rs`, `search.rs`, `solve_gd.rs`, `optimizer.rs`).
