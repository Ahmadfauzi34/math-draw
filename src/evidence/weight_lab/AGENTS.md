# src/evidence/weight_lab/ — Linear Algebra Certification & Weight Expression Workbench

## Purpose

Provides high-precision `f64` matrix/spectral verification, Burn engine numerical probes (`engine_probe`), adaptation jump gating, Frobenius norm-bound and orthonormal composition certificates, recursive weight expression tree evaluation, and a built-in 9-check selftest suite.

## Ownership

- `WeightSet` validation & public module re-exports: [`mod.rs`](mod.rs)
- Matrix-vector/matrix-matrix ops, QR/eigenvalue spectral radius, determinants, Frobenius norms: [`linalg.rs`](linalg.rs)
- Burn probe (`engine_probe`), gating (`gate_sets`), repertoire certification (`certify_weight_set`), `compose_properties`, and recursive expr evaluator (`evaluate_weight_expr`): [`certify.rs`](certify.rs)
- Self-contained 9-case deterministic selftest harness (`run_weight_lab_selftest`): [`selftest.rs`](selftest.rs)

## Local Contracts

- **Fail-Closed Composition Bounds**: `norm_bound` composition must refuse sets when `||W_B||_F * ||W_A||_F >= rho_bound` (`refused: true`), never promoting unverified contractions.
- **Orthonormal Invariants**: `orthonormal` composition must verify factor and product orthogonality (`||W W^T - I||_inf <= tol`) and determinant conservation (`det(W_B W_A) ≈ det(W_B) det(W_A)`).
- **Dimension & Parameter Domain Integrity**: [`WeightSet::validate_dims`](mod.rs) must verify rectangular matrix row consistency and exact bias length matching before any algebra or Burn probe runs; [`spectral_radius`](linalg.rs) requires `iters >= 1` and [`gate_sets`](certify.rs) requires finite `jump_tol >= 0`.
- **Repertoire Persistence Integrity**: [`certify_weight_set`](certify.rs) must fail closed on unreadable/corrupted existing repertoire JSON, directory creation errors, or serialization errors rather than silently resetting or overwriting the repertoire file.

## Work Guidance

- Keep numerical linear algebra routines pure and zero-panic in [`linalg.rs`](linalg.rs); keep Burn spec construction and certificate formatting in [`certify.rs`](certify.rs).
- Any new composition mode or certificate field added to [`certify.rs`](certify.rs) must add a corresponding test inside [`selftest.rs`](selftest.rs).

## Verification

- `./target/release/math-draw weight-lab selftest` (9/9 passed)
- `cargo test --locked --test integration batch3_weight_lab_linear_equiv_cost_model_and_harness`

## Child Graph Index

- Leaf domain directory; submodules listed under Ownership (`mod.rs`, `linalg.rs`, `certify.rs`, `selftest.rs`).
