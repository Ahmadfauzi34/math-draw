# src/ — Core Crate Modules & Facade Contract

## Purpose

Contains the complete `math_draw` Rust library crate ([`lib.rs`](lib.rs)) and thin CLI binary wrapper ([`main.rs`](main.rs)), partitioned into eight cohesive domain subdirectories plus canonical shared IR/catalog modules ([`spec.rs`](spec.rs), [`catalog.rs`](catalog.rs)).

## Ownership

- Crate root & backward-compatible module facades: [`lib.rs`](lib.rs)
- Binary entry wrapper: [`main.rs`](main.rs) (delegates directly to [`cli::run_cli`](cli/mod.rs))
- Canonical `SpecV1`, JSON/f32/i64 parsing, SHA-256 hashing & `v1`/`v2` loading: [`spec.rs`](spec.rs)
- Authoritative 82-constructor metadata & weight-count registry: [`catalog.rs`](catalog.rs)

## Local Contracts

- **Thin Entrypoint Invariant**: [`main.rs`](main.rs) must remain a minimal delegation stub calling `math_draw::cli::run_cli()`. No CLI command logic belongs in `main.rs`.
- **Canonical Integer Key Rule**: Slot and layer maps keyed by string (`weights`, `inputs`) must use canonical unsigned integer strings (`is_canonical_uint_key`: `"0"` or non-zero-padded digits).
- **Deterministic Canonical Hashing**: `sha256_f32_slice` normalizes `-0.0f32` to `0.0f32` prior to little-endian hashing so sign-zero representation differences never spuriously invalidate traces.
- **No Circular Coupling**: [`spec.rs`](spec.rs) and [`catalog.rs`](catalog.rs) sit at the dependency base (aside from `spec.rs` dispatching `v2` specs to `composer`). Domain subtrees build upward cleanly: `layers` -> `runtime` / `graph` / `evidence` / `es` -> `harness` / `smoke` -> `cli`.

## Work Guidance

- Keep [`lib.rs`](lib.rs) strictly declarative (`pub mod ...` and facade `pub use ...` aliases).
- Place any new Burn op parameters or weight count rules in [`catalog.rs`](catalog.rs) first, then wire shape validation in [`graph/validate.rs`](graph/validate.rs) and runtime kernel construction in [`layers/`](layers/AGENTS.md).
- Update this index if adding or renaming top-level subdirectories under `src/`.

## Verification

- `cargo test --locked`
- `cargo clippy --all-targets --locked`

## Child Graph Index

- [`layers/AGENTS.md`](layers/AGENTS.md) — 82 Burn layer builders, 4D tensor/int execution kernels, and autograd `GradParam` wrappers
- [`runtime/AGENTS.md`](runtime/AGENTS.md) — `PreparedGraph` execution, step trace hashing, recurrent rollouts, profiling, and cost modeling
- [`graph/AGENTS.md`](graph/AGENTS.md) — Static validator, `v2` recursive composer, pure rewrites, transactional/branch mutations, and sharding
- [`es/AGENTS.md`](es/AGENTS.md) — Rewrite tree search, autodiff GD solver, and Evolutionary Strategy optimizer
- [`evidence/AGENTS.md`](evidence/AGENTS.md) — Equivalence verifier, symbolic affine prover, gradcheck, quantization, and `weight-lab`
- [`harness/AGENTS.md`](harness/AGENTS.md) — Layer-4 `.burn-harness` workspace, artifact store, branch coordinator, and workflow runner
- [`smoke/AGENTS.md`](smoke/AGENTS.md) — Built-in 24-check deterministic self-verification suite
- [`cli/AGENTS.md`](cli/AGENTS.md) — Command parser, JSON/exit-code formatting, and domain command handlers
