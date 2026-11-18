# examples/ — Canonical Spec & Target JSON Fixtures

## Purpose

Provides clean, minimal, runnable sample JSON inputs illustrating canonical `SpecV1`, recursive `v2` specs, and optimization target files used by documentation and CLI smoke demos.

## Ownership

- Flat `v1` linear + ReLU example: [`linear-relu.json`](linear-relu.json)
- Recursive `v2` gated residual spec with subgraphs & state ports: [`recursive-gated-residual.v2.json`](recursive-gated-residual.v2.json)
- Supervised GD/ES optimization targets for `linear-relu.json`: [`targets-linear.json`](targets-linear.json)

## Local Contracts

- **Always Valid & Runnable**: Every fixture under `examples/` must pass `math-draw validate` and `math-draw run` (or work directly as `--targets` for its paired spec) cleanly out of the box.
- **Documentation Sync**: File names and paths documented in [`README.md`](../README.md) "Quick Examples" must stay valid against files in this directory.

## Work Guidance

- Keep example payloads human-readable, small, and deterministic.
- Do not embed machine-specific absolute paths in example specs.

## Verification

- `cargo run --locked -- run examples/linear-relu.json`
- `cargo run --locked -- run examples/recursive-gated-residual.v2.json`
- `cargo run --locked -- rollout fixpoint examples/recursive-gated-residual.v2.json --tol 1e-5`

## Child Graph Index

- Leaf fixture directory (`linear-relu.json`, `recursive-gated-residual.v2.json`, `targets-linear.json`).
