# src/evidence/ — Equivalence Verification, Gradcheck, Quantization & Weight Lab

## Purpose

Provides formal, numerical, differential, and algebraic verification gates across tensor specs and parameter sets—proving functional equivalence, worst-case error amplification, finite-difference/autodiff gradient soundness, quantization error budgets, and certified manual weight composition.

## Ownership

- Numeric equivalence verification, domain-aware fuzzing, and adversarial interval zooming (`--zoom`): [`verify.rs`](verify.rs)
- Exact symbolic affine equivalence prover (`equiv_linear`) across `linear`, `add`, `sub`, and `concat(dim=1)`: [`linear_equiv.rs`](linear_equiv.rs)
- Central finite-difference Jacobian checker & Burn autodiff cross-verification (`gradcheck_spec`): [`gradcheck.rs`](gradcheck.rs)
- Weight quantization simulation (`int8`, `q8-per-block`, `fp16`) & DAG Lipschitz/measured budget planning: [`quant.rs`](quant.rs)
- Long-run determinism prover (`soak_spec`) and PRNG stream certification/verification (`certify_prng_stream`, `verify_prng_manifest`): [`soak.rs`](soak.rs)
- Portable program bundles (`burn-math.program-bundle.v1`), canonical unique-first-use parameter binding (`burn-math.graph-parameter-binding.v1`), live replay & proof provenance verification (`compute_parameter_binding`, `export_bundle`, `verify_bundle`): [`bundle.rs`](bundle.rs)
- Independent pure-Rust `f64` differential oracle across all 87 `f32` constructors & external reference mode: [`diff_ref/`](diff_ref/AGENTS.md)
- 3D geometry domain pack (4×4 builders, `normalMat3`, scene-tree compiler, 3D IK solver, single/batched Burn projection, 17-check selftest): [`geom3/`](geom3/AGENTS.md)
- Manual-weighting linear algebra workbench, certification, composition & selftests: [`weight_lab/`](weight_lab/AGENTS.md)

## Local Contracts

- **Accept/Reject Precision & Domain Guards**: Equivalence checks (`verify_specs`, `equiv_linear`, `quant_verify`) must fail closed on non-finite or negative tolerances (`abs_tol`, `rel_tol`, `tol`), shape/dtype mismatches, runtime errors, NaN/Inf outputs, or tolerance violations (`verdict: "REJECT"` / `equivalent: false` or `Err(String)`). `zoom_fuzz_specs` / `zoom_fuzz_specs_with_tol` must reject `count == 0`, non-finite `lo`/`hi`, `lo > hi`, or invalid tolerances, and fail closed (`ok: false`, `verdict: "REJECT"`, exit `2`) whenever explicit `--abs-tol`/`--rel-tol` is violated. `soak_spec` rejects `runs < 2`, `certify_prng_stream` rejects `count == 0` or `count > 10_000_000`, and `diff_ref_spec` rejects non-finite or `<= 0` `tol`.
- **Affine Restriction Soundness**: [`linear_equiv.rs`](linear_equiv.rs) must refuse graphs containing non-affine operations (`relu`, `conv2d`, `softmax`, etc.) rather than falsely certifying equivalence.
- **Gradcheck Tolerance & Step Bounds**: Central finite differences evaluate `(f(x + eps) - f(x - eps)) / (2 * eps)` strictly on finite inputs (`eps > 0`, `tol > 0`); `--autodiff` verification compares numeric weight gradients directly against Burn `Autodiff` `Gradients`.
- **Budget Soundness**: [`quant.rs`](quant.rs) requires finite `tol >= 0` and in `bound` mode must only select per-layer quantization schemes whose accumulated Lipschitz error upper bound stays within `tol`.
- **Bundle Tamper & Replay Soundness**: [`bundle.rs`](bundle.rs) deduplicates parameter owners by unique first use in `spec.steps` order, keeps `program_identity` invariant to weights/inputs, enforces `#[serde(deny_unknown_fields)]` across `ProgramBundle` and embedded `SpecV1` (`verify_bundle_json`), and fails closed in `verify_bundle` / `verify_bundle_json` on schema mismatches (`invalid_bundle_schema`), payload or injected-unknown-field tampering (`bundle_integrity_failed`), binding drift (`parameter_binding_mismatch`), live replay drift (`execution_receipt_mismatch`), or attestation digest tampering (`attestation_mismatch`).

## Work Guidance

- Keep each verification discipline isolated in its own submodule; expose shared structs cleanly via [`mod.rs`](mod.rs).
- Preserve deterministic seeded RNG behavior in fuzz/zoom fuzzing so counterexamples are reproducible.

## Verification

- `cargo test --locked --test integration batch2_gradcheck_quant_mutate_and_shard`
- `cargo test --locked --test integration batch3_weight_lab_linear_equiv_cost_model_and_harness`
- `cargo test --locked --test integration program_bundle_binding_replay_and_tamper_detection`
- `cargo test --locked --test integration soak_prng_manifest_quant_search_and_new_workflow_stages`
- `cargo test --locked --test integration diff_ref_all_f32_families_and_external_ref_parity`
- `cargo test --locked --test integration geom3_builders_scene_tree_ik_projection_batch_and_cli`
- `./target/release/math-draw smoke` (checks #6, #16, #17, #20, #21, #22, #27, #30, #32, #33)

## Child Graph Index

- [`diff_ref/AGENTS.md`](diff_ref/AGENTS.md) — Independent Burn-free `f64` interpreter across all 87 `f32` constructors & external reference differential testing
- [`geom3/AGENTS.md`](geom3/AGENTS.md) — 3D transform builders, `normalMat3`, scene-tree compiler, 3D IK solver, single/batched Burn projection & 17-check selftest
- [`weight_lab/AGENTS.md`](weight_lab/AGENTS.md) — Matrix algebra, stability/orthonormality certificates, expr evaluator, and 9-check selftest
