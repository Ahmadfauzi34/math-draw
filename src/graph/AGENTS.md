# src/graph/ — Validation, Compilation, Rewrites, Mutation & Sharding

## Purpose

Manages graph topology before execution: static validation and 4D shape/dtype dataflow analysis, compiling `v2` recursive specs to canonical flat `v1`, deterministic algebraic/structural rewrites, equivalence-gated graph mutations, and linear output-row sharding.

## Ownership

- Recursive `v2` compiler, macro expansion, and node substitution: [`composer/`](composer/AGENTS.md)
- Static validation, canonical constructor arg checking, 4D shape inference, dtype tracking, and `--explain` reporting: [`validate.rs`](validate.rs)
- Pure graph rewrites (`linear-branch-fusion`, `linear-chain-fusion`, `relu-idempotent`, `transpose-involution`, `dead-step-elimination`, `add-commute`) & fixpoint pass: [`rewrite.rs`](rewrite.rs)
- Transactional & multi-branch equivalence-gated mutations (`setWeights`, `replaceStep`, `insertStep`, `removeStep`, `reconnectStep`, `setOutput`): [`mutate.rs`](mutate.rs)
- Output-row `linear` sharding with `concat(dim=1)` reassembly and verification: [`shard.rs`](shard.rs)

## Local Contracts

- **Static Safety Gate**: [`validate_spec`](validate.rs) is the single source of truth for static validity (`unknown_layer_constructor`, invalid arity/args, slot OOB, uninitialized slot reads, rank-4 shape mismatches, `i64`/`f32` dtype mismatches, output slot unpopulated).
- **Exact Rewrite Semantics**: Every rule in [`rewrite.rs`](rewrite.rs) must preserve input/output tensor semantics over valid graphs; dead-step elimination must maintain slot/output reachability.
- **Equivalence-Gated Mutation**: [`mutate_spec`](mutate.rs) commits a candidate branch only when post-mutation outputs pass `verify_specs` within `abs_tol`/`rel_tol` across evaluation cases.
- **Exact Shard Reassembly**: [`shard_spec`](shard.rs) splits `linear` weights along output rows strictly bounded by `max_weights` and verifies assembled output against the unsharded baseline.

## Work Guidance

- When modifying constructor parameter schemas in [`src/catalog.rs`](../catalog.rs), always update `infer_step_shape` and arg/dtype checks in [`validate.rs`](validate.rs) in the same change.
- Keep rewrite passes pure and composable (`apply_rewrite`, `apply_rewrites_recursive`).

## Verification

- `cargo test --locked --test integration rewrites_and_recurse_preserve_semantics`
- `cargo test --locked --test integration batch2_gradcheck_quant_mutate_and_shard`

## Child Graph Index

- [`composer/AGENTS.md`](composer/AGENTS.md) — `burn-math.recursive-spec.v2` compiler, macros, and substitution passes
