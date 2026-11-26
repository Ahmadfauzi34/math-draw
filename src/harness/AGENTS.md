# src/harness/ — Layer-4 Agent Workspace, Content-Addressed Store & Pipeline Runner

## Purpose

Implements the Layer-4 agent-harness co-processor runtime: persistent `.burn-harness` workspace management, SHA-256 content-addressed tensor artifact storage (`burn://sha256:...`), semantic tensor digesting, capability discovery, subagent branch coordination, and multi-stage declarative workflow execution.

## Ownership

- Workspace directory bootstrapping, artifact store/resolve integrity validation, `compute_semantic_digest`, and `get_capability_catalog`: [`workspace.rs`](workspace.rs)
- Subagent packet export (`export_subagent_branches`) and equivalence-gated winner selection (`merge_subagent_branches`): [`branches.rs`](branches.rs)
- Multi-stage pipeline executor (`execute_harness_workflow`): [`workflow.rs`](workflow.rs)
- Public harness facade re-exports: [`mod.rs`](mod.rs)

## Local Contracts

- **Content-Addressed Integrity**: [`resolve_artifact_uri`](workspace.rs) must validate URI prefix (`burn://sha256:`), exact 64 lowercase hex characters, file presence, and live SHA-256 recomputation against the stored JSON payload (`artifact_integrity_failed` on tamper).
- **Canonical Workspace State**: `workspace-state.json` adheres to schema `burn-math.harness-session.v1` and records artifact tables, proofs, and stage events. Corrupted, unreadable, or unwritable `workspace-state.json` MUST fail closed with explicit `Err(String)` diagnostics (never silently swallowed or panicking via `.unwrap()`).
- **Equivalence-Gated Branch Merge**: [`merge_subagent_branches`](branches.rs) requires finite `tol >= 0`, evaluates candidate specs across derived test inputs (`baseline`, `zeros`, `ones`, `negated`, `scaled_x2`, `scaled_half`), and promotes only verified equivalent candidates with minimum step count (tie-broken by minimum error).
- **Strict Workflow Schema & Stage Error Isolation**: [`execute_harness_workflow`](workflow.rs) enforces strict fail-closed JSON validation at both root (`schema`, `name`, `description`, `workspace`, `spec`, non-empty `pipeline`) and per-stage levels (`compile`/`compile_v2` with optional `auto_repair`/`compact_slots`/`max_weights >= 1`, `validate`, `rollout_unroll`/`unroll` with optional `compact_slots`, `search_rewrites` with optional `quant_tol`/`quant_scheme`, `quant_budget`, `fit`/`fit_hybrid`, `solve_gd`, `rollout_scan`, `rollout_fixpoint`, `run`, `grad_spec`, `soak`, `gradcheck`, `shard`, `bundle_export`, `diff_ref`) — rejecting unknown keys, explicit `null` parameter values, non-string/empty stage or metadata names, wrong JSON value types, or out-of-domain numeric parameters without silent fallback defaults. Any stage returning `Err` or `ok == false` (including `quant_budget`, `fit`, `solve_gd`, `shard`, `gradcheck`, `soak`, `diff_ref`) halts pipeline progression immediately before updating `active_spec`, classifies the error via [`classify_error`](../error.rs), best-effort persists accumulated events/proofs without clobbering the originating stage error (attaching `workspace_persist_error` if persistence fails), and returns structured `ok: false` with `code`, `error`, `hint`, `failed_stage`, and per-stage `code`/`hint` diagnostics in `stage_reports`.

## Work Guidance

- Keep [`get_capability_catalog`](workspace.rs) synchronized whenever top-level CLI subcommands are added or renamed.
- Temporary workspaces created inside tests or smoke checks must clean up cleanly in `std::env::temp_dir()`.

## Verification

- `cargo test --locked --test integration batch3_weight_lab_linear_equiv_cost_model_and_harness`
- `cargo test --locked --test integration soak_prng_manifest_quant_search_and_new_workflow_stages`
- `./target/release/math-draw harness capabilities`
- `./target/release/math-draw smoke` (check #23)

## Child Graph Index

- Leaf domain directory; submodules listed under Ownership (`mod.rs`, `workspace.rs`, `branches.rs`, `workflow.rs`).
