# math-draw

Standalone single-binary Rust deterministic graph engine and recursive agent engineering CLI built on **Burn `0.22.1`** (`burn-flex` CPU backend + runtime `autodiff`), Rust edition `2024`.

## Build

```bash
cargo build --release
# binary: target/release/math-draw
```

## Subcommands

- `math-draw run <spec.json>` — Validate and execute a `v1` flat spec or `burn-math.recursive-spec.v2` spec on `Device::flex()`.
- `math-draw validate <spec.json> [--explain]` — Static validation (52-constructor catalog, argument bounds, shape inference, dtype flow, dataflow & ports) with optional `--explain` trace.
- `math-draw verify <base.json> <cand.json> [--cases <cases.json>] [--fuzz [N]] [--seed S] [--lo A] [--hi B] [--abs-tol T] [--rel-tol T]` — Equivalence checking across explicit or domain-aware fuzzed inputs.
- `math-draw find-op [--list | <constructor>]` — Inspect supported constructors and parameter schemas.
- `math-draw rewrite list` / `math-draw rewrite apply <rule|all> <spec.json> [--out <out.json>]` — Pure graph rewrites (`linear-branch-fusion`, `linear-chain-fusion`, `relu-idempotent`, `transpose-involution`, `dead-step-elimination`, `add-commute`).
- `math-draw recurse expand <spec.v2.json> [--out <out.v1.json>]` — Compile recursive `v2` graph specs (subgraphs, `repeat` with `tie_weights`/`weights_per_depth`, `residual`, `parallel_concat`, `$ref`, built-in macros) into canonical flat `v1`.
- `math-draw recurse search <spec.json> [--out <best.json>] [--max-depth 3] [--cases <cases.json>]` — Recursive tree rewrite search verified against the baseline.
- `math-draw recurse solve-gd <spec.json> [--targets <targets.json>] [--steps 20] [--lr 0.1] [--layer <idx|all>] [--out <out.json>]` — Burn `Device::flex().autodiff()` gradient descent with reflective backtracking.
- `math-draw recurse fit <spec.json> --targets <targets.json> [--gens 40] [--pop 16] [--sigma 0.2] [--lr 0.2] [--seed 42] [--out <out.json>]` — Gradient-free Evolutionary Strategy optimizer across active weighted layers.
- `math-draw smoke` — Run the built-in end-to-end smoke suite.

## Quick Examples

```bash
cargo run -- smoke
cargo run -- run examples/linear-relu.json
cargo run -- run examples/recursive-gated-residual.v2.json
cargo run -- validate examples/linear-relu.json --explain
cargo run -- recurse solve-gd examples/linear-relu.json --targets examples/targets-linear.json --steps 20 --lr 0.05
```