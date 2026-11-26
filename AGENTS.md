# GRAPH FRAMEWORK CONTRACTS & RAIL (INTELLIGENT REPOSITORY RAIL)

This repository follows the graph (Documentation-as-Contract) hierarchical system to enforce global development coherence and absolute quality standards across all folders.

## Core Contracts & System Behavior

- **Durable Hierarchies**: `AGENTS.md` files located in directory subtrees hold binding work contracts, runtime side-effects, and quality specifications for that domain. All edits and code increments within a subtree must trace direct authority from its closest governing `AGENTS.md`.
- **The Graph Pass Process**:
  - **Read-Before-Edit**: Before modifying any files, always recursively trace the exact path from repository root to your target directory, reading every `AGENTS.md` document encountered along the parent chain.
  - **Reflect-and-Update**: Whenever behavior, data design, or functional dependencies are modified, immediately verify and update the nearest subtree `AGENTS.md` to keep rules fresh and aligned. Remove contradictory or stale operational records dynamically.

## Purpose

`math-draw` is a standalone Rust edition `2024` single-binary deterministic tensor graph engine, compiler, verifier, optimizer, and agent-harness CLI built on **Burn `0.22.1`** (`burn-flex` CPU backend + runtime `autodiff`).

## Ownership

- Repository-wide build/release pipeline ([`.github/workflows/ci.yml`](.github/workflows/ci.yml), [`.github/workflows/release.yml`](.github/workflows/release.yml))
- Crate manifest & locked dependency graph ([`Cargo.toml`](Cargo.toml), [`Cargo.lock`](Cargo.lock))
- End-to-end documentation and release history ([`README.md`](README.md), [`CHANGELOG.md`](CHANGELOG.md))
- Subtree contracts under [`src/`](src/AGENTS.md), [`tests/`](tests/AGENTS.md), and [`examples/`](examples/AGENTS.md)

## Local Contracts

- **Deterministic Backend**: All tensor execution runs deterministically on `Device::flex()`. Runtime autodiff uses Burn's `AutodiffBackend` on `flex` without requiring CUDA/GPU toolchains.
- **Portable Static Linux Release**: Official Linux `x86_64` CI and release binaries build as fully static `x86_64-unknown-linux-musl` executables (no `INTERP` / no `NEEDED` `glibc` shared libraries) so binaries run across all Linux distros and containers without `GLIBC_*` version mismatches.
- **Zero-Warning & Locked Dependency Gate**: Every build and test gate MUST pass with `--locked` and zero compiler warnings in `cargo test --locked` and `cargo build --release --locked`.
- **Fail-Closed Validation**: Untrusted or malformed graph specs must fail closed before execution (unknown constructors, arity/arg mismatches, shape/dtype errors, canonical registry ID uniqueness, division-by-zero, integer overflow, NaN/Inf inputs).
- **Backward-Compatible Crate Facade**: [`src/lib.rs`](src/lib.rs) must preserve root facade module paths (`engine`, `direct_math`, `error`, `trace`, `rollout`, `profile`, `cost_model`, `validate`, `composer`, `liveness`, `grad_spec`, `rewrite`, `mutate`, `shard`, `recurse`, `verify`, `diff_ref`, `geom3`, `linear_equiv`, `gradcheck`, `quant`, `soak`, `weight_lab`, `bundle`) so integration tests and external library callers never break.
- **95-Constructor Alignment**: Constructor inventory in [`src/catalog.rs`](src/catalog.rs), shape inference in [`src/graph/validate.rs`](src/graph/validate.rs), and Burn execution kernels in [`src/layers/`](src/layers/AGENTS.md) must always stay in strict 1-to-1 parity across all 95 constructors (covering `unary`, `binary`, and `ternary` step arities).

## Work Guidance

- Trace `AGENTS.md` from repo root down to your target directory before editing any code.
- Keep domain modules focused (~100–600 lines where practical) and avoid re-introducing monolithic multi-thousand-line files.
- Keep JSON CLI output schemas stable and machine-parseable (`ok` boolean, structured `code`/`error`/`stage`/`hint` diagnostics, exit code `0` on success / `2` on failure).
- Whenever adding a new subcommand, constructor, or workflow stage, update both the nearest `AGENTS.md` and [`smoke/`](src/smoke/AGENTS.md) / [`tests/`](tests/AGENTS.md).

## Verification

- `cargo test --locked` (runs all 22 integration suites in [`tests/integration.rs`](tests/integration.rs), including the 33-check smoke suite)
- `cargo clippy --all-targets --locked`
- `cargo build --release --locked`
- `./target/release/math-draw smoke`
- `./target/release/math-draw weight-lab selftest`
- `./target/release/math-draw geom selftest`

## Child Graph Index

- [`src/AGENTS.md`](src/AGENTS.md) — Rust library & CLI implementation subtrees (`layers`, `runtime`, `graph`, `es`, `evidence`, `harness`, `smoke`, `cli`)
- [`tests/AGENTS.md`](tests/AGENTS.md) — Crate-level integration tests
- [`examples/AGENTS.md`](examples/AGENTS.md) — Canonical `v1`, `v2`, and target JSON fixtures
