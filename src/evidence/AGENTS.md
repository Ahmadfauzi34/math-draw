# src/evidence/ — Equivalence Verification, Gradcheck, Quantization & Weight Lab

## Purpose

Provides formal, numerical, differential, and algebraic verification gates across tensor specs and parameter sets—proving functional equivalence, worst-case error amplification, finite-difference/autodiff gradient soundness, quantization error budgets, and certified manual weight composition.

## Ownership

- Numeric equivalence verification, domain-aware fuzzing, and adversarial interval zooming (`--zoom`): [`verify.rs`](verify.rs)
- Exact symbolic affine equivalence prover (`equiv_linear`) across `linear`, `add`, `sub`, and `concat(dim=1)`: [`linear_equiv.rs`](linear_equiv.rs)
- Central finite-difference Jacobian checker & Burn autodiff cross-verification (`gradcheck_spec`): [`gradcheck.rs`](gradcheck.rs)
- Weight quantization simulation (`int8`, `q8-per-block`, `fp16`) & DAG Lipschitz/measured budget planning: [`quant.rs`](quant.rs)
- Manual-weighting linear algebra workbench, certification, composition & selftests: [`weight_lab/`](weight_lab/AGENTS.md)

## Local Contracts

- **Accept/Reject Precision**: Equivalence checks (`verify_specs`, `equiv_linear`, `quant_verify`) must fail closed on shape, dtype, runtime, NaN/Inf, or tolerance violations (`verdict: "REJECT"` / `equivalent: false`).
- **Affine Restriction Soundness**: [`linear_equiv.rs`](linear_equiv.rs) must refuse graphs containing non-affine operations (`relu`, `conv2d`, `softmax`, etc.) rather than falsely certifying equivalence.
- **Gradcheck Tolerance & Step Bounds**: Central finite differences evaluate `(f(x + eps) - f(x - eps)) / (2 * eps)` strictly on finite inputs; `--autodiff` verification compares numeric weight gradients directly against Burn `Autodiff` `Gradients`.
- **Budget Soundness**: [`quant.rs`](quant.rs) in `bound` mode must only select per-layer quantization schemes whose accumulated Lipschitz error upper bound stays within `tol`.

## Work Guidance

- Keep each verification discipline isolated in its own submodule; expose shared structs cleanly via [`mod.rs`](mod.rs).
- Preserve deterministic seeded RNG behavior in fuzz/zoom fuzzing so counterexamples are reproducible.

## Verification

- `cargo test --locked --test integration batch2_gradcheck_quant_mutate_and_shard`
- `cargo test --locked --test integration batch3_weight_lab_linear_equiv_cost_model_and_harness`
- `./target/release/math-draw smoke` (checks #6, #16, #17, #20, #21, #22)

## Child Graph Index

- [`weight_lab/AGENTS.md`](weight_lab/AGENTS.md) — Matrix algebra, stability/orthonormality certificates, expr evaluator, and 9-check selftest
