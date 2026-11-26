# src/graph/ — Validation, Compilation, Rewrites, Mutation & Sharding

## Purpose

Manages graph topology before execution: static validation and 4D shape/dtype dataflow analysis, compiling `v2` recursive specs to canonical flat `v1`, deterministic algebraic/structural rewrites, equivalence-gated graph mutations, and linear output-row sharding.

## Ownership

- Recursive `v2` compiler, macro expansion, and node substitution: [`composer/`](composer/AGENTS.md)
- Static validation, canonical constructor arg checking, 4D shape inference, dtype tracking, and `--explain` (with peak-byte liveness) reporting: [`validate.rs`](validate.rs)
- Pure static slot liveness (`analyze_liveness`), SSA normalization (`ssa_normalize_spec`), linear-scan register compaction (`compact_slots_by_liveness`), and deterministic 3-pass auto-repair (`auto_repair_spec`): [`liveness.rs`](liveness.rs)
- Symbolic in-graph 1st-order VJP and 2nd-order HVP graph compiler (`compile_backward_spec`, `compile_grad_spec`) across 23 differentiable constructors: [`grad_spec.rs`](grad_spec.rs)
- Pure graph rewrites (`linear-branch-fusion`, `linear-chain-fusion`, `relu-idempotent`, `transpose-involution`, `dead-step-elimination`, `add-commute`) & fixpoint pass: [`rewrite.rs`](rewrite.rs)
- Transactional & multi-branch equivalence-gated mutations (`setWeights`, `replaceStep`, `insertStep`, `removeStep`, `reconnectStep`, `setOutput`): [`mutate.rs`](mutate.rs)
- Output-row `linear` sharding with `concat(dim=1)` reassembly and verification: [`shard.rs`](shard.rs)

## Local Contracts

- **Static Safety Gate**: [`validate_spec`](validate.rs) is the single source of truth for static validity (`unknown_layer_constructor`, invalid arity/args, slot OOB, uninitialized slot reads, rank-4 shape mismatches, `i64`/`f32` dtype mismatches, output slot unpopulated).
- **Liveness & Compaction Soundness**: [`compact_slots_by_liveness`](liveness.rs) SSA-normalizes first, preserves all declared port slots, avoids aliasing any step's inputs with its output slot, and validates the compacted graph before returning.
- **Symbolic Grad-Spec Soundness**: [`compile_grad_spec`](grad_spec.rs) accepts `order` in `{1, 2}`, rejects `--hvp-vec` on `order == 1`, SSA-normalizes forward steps before differentiating, fails closed on unsupported constructors (`unsupported_symbolic_grad_op`) or invalid `--wrt` (`unresolved_wrt_slot`), avoids removable origin singularities (e.g., `gelu` computes $\Phi(x)$ smoothly via Padé `[3/2]` inside `tanh` with strictly positive denominator `1 + 0.00806*x^2 >= 1` rather than `gelu(x)/x`, remaining finite and exact at `x = 0` for both VJP and HVP), and validates the emitted `SpecV1`.
- **Exact Rewrite Semantics**: Every rule in [`rewrite.rs`](rewrite.rs) must preserve input/output tensor semantics over valid graphs; dead-step elimination must maintain slot/output reachability.
- **Equivalence-Gated Mutation**: [`mutate_spec`](mutate.rs) requires finite `abs_tol >= 0` and `rel_tol >= 0`, and commits a candidate branch only when post-mutation outputs pass `verify_specs` within `abs_tol`/`rel_tol` across evaluation cases.
- **Exact Shard Reassembly**: [`shard_spec`](shard.rs) splits `linear` weights along output rows strictly bounded by `max_weights` and verifies assembled output against the unsharded baseline.

## Work Guidance

- When modifying constructor parameter schemas in [`src/catalog.rs`](../catalog.rs), always update `infer_step_shape` and arg/dtype checks in [`validate.rs`](validate.rs) in the same change.
- Keep rewrite passes pure and composable (`apply_rewrite`, `apply_rewrites_recursive`).

## Verification

- `cargo test --locked --test integration rewrites_and_recurse_preserve_semantics`
- `cargo test --locked --test integration batch2_gradcheck_quant_mutate_and_shard`
- `cargo test --locked --test integration liveness_compaction_auto_repair_and_attention_head_macros`
- `cargo test --locked --test integration symbolic_grad_spec_vjp_hvp_and_finite_diff_parity`

## Child Graph Index

- [`composer/AGENTS.md`](composer/AGENTS.md) — `burn-math.recursive-spec.v2` compiler, 6 built-in macros, and substitution passes
