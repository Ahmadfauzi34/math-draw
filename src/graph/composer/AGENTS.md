# src/graph/composer/ — Recursive Spec v2 Compiler & Macro Lowering

## Purpose

Compiles hierarchical `burn-math.recursive-spec.v2` graph specifications—supporting reusable `subgraphs`, `repeat` blocks, `residual` connections, `parallel_concat`, external `$ref` includes, and built-in macros—into canonical flat [`SpecV1`](../../spec.rs) graphs accompanied by an `ExpansionReceipt`.

## Ownership

- Core `v2` schema detection (`is_recursive_spec_v2`), subgraph/repeat/residual lowering, weight binding & `ExpansionReceipt` generation: [`mod.rs`](mod.rs)
- Built-in reusable subgraph macro definitions (`mlp`, `gated_unit`, `residual_mlp`, `transformer_ffn`, `recurrent_cell`, `attention_head` with `seq_axis1`/`heads_axis1` layouts): [`macros.rs`](macros.rs)
- Node/slot/layer-index substitution and JSON template instantiation helpers: [`subst.rs`](subst.rs)

## Local Contracts

- **Bounded `$ref` Resolution**: Local `$ref` file loads resolve relative to `base_dir`, guard against cyclic recursion, and fail closed on missing or invalid targets.
- **Weight Sharing Semantics**: `repeat` blocks with `tie_weights: true` must reuse identical flattened layer parameters (`layers[i]` pointing to shared weights), whereas `weights_per_depth` allocates distinct layer weights per unrolled depth index.
- **Receipt Traceability**: Every `compile_recursive_spec` call produces a complete `ExpansionReceipt` mapping named `v2` ports back to concrete `SpecV1` slot indices for downstream `rollout` and `harness` consumers.
- **Canonical Valid Output**: Every lowered `SpecV1` emitted by `compile_recursive_spec` must pass [`validate_spec`](../validate.rs).

## Work Guidance

- Keep macro topology templates isolated in [`macros.rs`](macros.rs) and pure JSON substitution helpers in [`subst.rs`](subst.rs).
- Maintain `depth` / recursion caps to prevent runaway graph expansion from malformed `v2` inputs.

## Verification

- `cargo test --locked --test integration smoke_suite_passes_all_checks` (checks #3, #4, #12)

## Child Graph Index

- Leaf domain directory; submodules listed under Ownership (`mod.rs`, `macros.rs`, `subst.rs`).
