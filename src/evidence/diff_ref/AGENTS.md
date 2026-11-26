# src/evidence/diff_ref/ — Independent Pure-Rust `f64` Differential Oracle

## Purpose

Provides an independent, Burn-free `f64` interpreter covering all 87 `f32` constructors plus an external reference vector mode to detect Burn `Device::flex()` kernel drift, numerical bugs, or tolerance violations (`math-draw diff-ref`).

## Ownership

- Public oracle API (`SUPPORTED_REF_OPS`, `RefTensor`, `run_reference_f64`, `diff_ref_spec`, `DiffRefReport`, `DiffRefCaseReport`): [`mod.rs`](mod.rs)
- 44 core neural/conv/norm/pool/activation/custom/binary `f64` kernels: [`nn_ops.rs`](nn_ops.rs)
- 43 pure-math numeric/transform/linalg/statistics/reduction/probability/comparison `f64` kernels: [`math_ops.rs`](math_ops.rs)

## Local Contracts

- **Zero Burn Dependency**: `nn_ops.rs` and `math_ops.rs` must execute strictly in pure Rust `f64` with zero calls into `burn` or `src/layers/`.
- **Exact Kernel Parity**: Semantics must match `math-draw` layers (e.g., `featureNorm` divides by `sqrt(sum(x^2)).max(eps)`, `rmsNorm` defaults to `eps = 1e-6`, `cosineSimilarity` defaults to `eps = 1e-12`, `round` uses IEEE-754 ties-to-even `round_ties_even`, `linear` uses column-major `[in_dim, out_dim]` + optional `[out_dim]` bias, `convTranspose2d` uses `[in_ch, out_ch, kh, kw]` + `[out_ch]` bias).
- **Fail-Closed Verification**: `run_reference_f64` and `diff_ref_spec` must reject non-finite or `<= 0` `tol`, invalid specs, wrong/non-finite layer weight lengths, empty case lists, length/count mismatches in external `--ref` vectors, non-finite engine/reference values, and unsupported (`int*`) constructors when no external `--ref` is provided. A case passes iff every element satisfies `abs_err <= tol || rel_err <= tol` (`first_diff_index.is_none()`); any mismatch returns `ok: false, verdict: "SIGNAL"` with `first_diff_index: Some(i)`.

## Work Guidance

- Keep [`SUPPORTED_REF_OPS`](mod.rs) in 1-to-1 sync with all non-integer constructors in [`src/catalog.rs`](../../catalog.rs).
- When adding a new `f32` constructor to `math-draw`, add its independent `f64` formula in either [`nn_ops.rs`](nn_ops.rs) or [`math_ops.rs`](math_ops.rs) and append its name to `SUPPORTED_REF_OPS`.

## Verification

- `cargo test --locked --test integration diff_ref_all_f32_families_and_external_ref_parity`
- `./target/release/math-draw smoke` (check #32)
- `./target/release/math-draw diff-ref examples/linear-relu.json`
