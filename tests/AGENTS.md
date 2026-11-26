# tests/ — Integration Test Suite Contracts

## Purpose

Houses crate-level Rust integration tests validating end-to-end public library APIs across all 95 constructors, fail-closed edge cases, rewrite/search equivalence, ternary `fma` pipelines, and Batch 1–3 + ported math engineering capabilities.

## Ownership

- Integrated regression suite (23 test suites): [`integration.rs`](integration.rs)

## Local Contracts

- **Public Facade Consumption**: [`integration.rs`](integration.rs) imports solely through `math_draw::*` public paths (e.g. `math_draw::engine`, `math_draw::direct_math`, `math_draw::error`, `math_draw::liveness`, `math_draw::grad_spec`, `math_draw::soak`, `math_draw::diff_ref`, `math_draw::geom3`, `math_draw::recurse`, `math_draw::bundle`, `math_draw::smoke`, `math_draw::verify`), verifying backward-compatible root module exports in [`src/lib.rs`](../src/lib.rs).
- **Full 95-Constructor Execution**: `catalog_has_all_95_constructors` and `all_95_constructors_execute_cleanly` guarantee every cataloged op builds and executes cleanly without panic.
- **Fail-Closed Verification**: `div_by_zero_and_int_overflow_fail_closed`, `ported_math_ops_domain_and_shape_fail_closed`, `parity_13_math_ops_fail_closed_and_ternary_pipelines`, `es_rng_vectors_mu_lambda_and_strict_validation`, `program_bundle_binding_replay_and_tamper_detection`, `error_handling_phase1_cli_and_library_fail_closed`, `error_handling_phase2_typed_codes_and_envelopes`, `error_handling_phase3_workflow_schema_and_domain_guards`, `liveness_compaction_auto_repair_and_attention_head_macros`, `symbolic_grad_spec_vjp_hvp_and_finite_diff_parity`, `soak_prng_manifest_quant_search_and_new_workflow_stages`, `direct_math_cli_and_library_arith_int_kernel_end_to_end`, `diff_ref_all_f32_families_and_external_ref_parity`, `geom3_builders_scene_tree_ik_projection_batch_and_cli`, `muse_audit_edge_case_regressions`, and `adversarial_verify_zoom_tol_and_bundle_unknown_fields` enforce explicit rejection of invalid math domains, shape mismatches, zero division, integer overflow, malformed ES configs, tampered program bundles (including injected unknown JSON fields), corrupted workspace/repertoire state, invalid CLI flags/numeric arguments, structured typed `ErrorCode` / `ErrorEnvelope` (`code`/`stage`/`hint`) diagnostics, strict workflow JSON field schemas, uniform library numeric domain boundary guards, liveness compaction/auto-repair/`attention_head`, symbolic VJP/HVP parity, soak/PRNG certification, direct `arith`/`int`/`kernel` CLIs, all 87 `f32` pure-Rust `f64` differential oracle checks, 3D geometry builders/scene trees/IK/projection/batching, `verify --zoom` tolerance gating, and edge-case fail-closed regressions.
- **Self-Contained Temp Artifacts**: Any file/workspace tests clean up created directories in `std::env::temp_dir()`.

## Work Guidance

- Add targeted integration tests for new public library modules or edge-case regressions here rather than relying solely on CLI smoke checks.
- Never hardcode assumptions about absolute workspace paths outside standard Rust temp dirs or relative fixture payloads.

## Verification

- `cargo test --locked` (must run 23/23 integration tests with 0 failures and 0 warnings)

## Child Graph Index

- Leaf directory containing [`integration.rs`](integration.rs).
