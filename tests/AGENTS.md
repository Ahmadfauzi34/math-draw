# tests/ — Integration Test Suite Contracts

## Purpose

Houses crate-level Rust integration tests validating end-to-end public library APIs across all 95 constructors, fail-closed edge cases, rewrite/search equivalence, ternary `fma` pipelines, and Batch 1–3 + ported math engineering capabilities.

## Ownership

- Integrated regression suite (10 test suites): [`integration.rs`](integration.rs)

## Local Contracts

- **Public Facade Consumption**: [`integration.rs`](integration.rs) imports solely through `math_draw::*` public paths (e.g. `math_draw::engine`, `math_draw::recurse`, `math_draw::smoke`, `math_draw::verify`), verifying backward-compatible root module exports in [`src/lib.rs`](../src/lib.rs).
- **Full 95-Constructor Execution**: `catalog_has_all_95_constructors` and `all_95_constructors_execute_cleanly` guarantee every cataloged op builds and executes cleanly without panic.
- **Fail-Closed Verification**: `div_by_zero_and_int_overflow_fail_closed`, `ported_math_ops_domain_and_shape_fail_closed`, and `parity_13_math_ops_fail_closed_and_ternary_pipelines` enforce explicit rejection of invalid math domains, shape mismatches, zero division, and integer overflow.
- **Self-Contained Temp Artifacts**: Any file/workspace tests clean up created directories in `std::env::temp_dir()`.

## Work Guidance

- Add targeted integration tests for new public library modules or edge-case regressions here rather than relying solely on CLI smoke checks.
- Never hardcode assumptions about absolute workspace paths outside standard Rust temp dirs or relative fixture payloads.

## Verification

- `cargo test --locked` (must run 10/10 integration tests with 0 failures and 0 warnings)

## Child Graph Index

- Leaf directory containing [`integration.rs`](integration.rs).
