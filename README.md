# math-draw

Standalone single-binary Rust deterministic graph engine and recursive agent engineering CLI built on **Burn `0.22.1`** (`burn-flex` CPU backend + runtime `autodiff`), Rust edition `2024`.

## Build

```bash
cargo build --release
# binary: target/release/math-draw
```

## Subcommands

- `math-draw run <spec.json>` — Validate and execute a `v1` flat spec or `burn-math.recursive-spec.v2` spec on `Device::flex()`.
- `math-draw validate <spec.json> [--explain]` — Static validation (95-constructor catalog, argument bounds, shape inference, dtype flow, dataflow & ports) with optional `--explain` trace and peak-byte slot liveness analysis.
- `math-draw verify <base.json> <cand.json> [--cases <cases.json>] [--fuzz [N]] [--zoom] [--zoom-depth 4] [--seed S] [--lo A] [--hi B] [--abs-tol T] [--rel-tol T]` — Equivalence checking across explicit cases, domain-aware fuzzing, or recursive adversarial interval narrowing (`--zoom`).
- `math-draw diff-ref <spec.json> [--cases <cases.json>] [--ref <ref.json>] [--tol 1e-5]` — Differential test Burn `Device::flex()` `f32` outputs against an independent pure-Rust `f64` oracle (all 87 `f32` constructors) or external reference vectors.
- `math-draw linear-equiv <specA.json> <specB.json> [--tol 0.0]` — Exact symbolic affine equivalence prover over `linear`, `add`, `sub`, and `concat(dim=1)` graphs.
- `math-draw soak <spec.json> [--runs 100]` — Long-run bit-identical determinism proof hashing canonical raw `f32` output bytes across `N` fresh executions.
- `math-draw prng-check [mulberry32|xorshift64|pcg32|philox4x32] [seed=1] [count=1000] [--out <manifest.json>] | verify <manifest.json>` — Certify and re-verify deterministic PRNG streams.
- `math-draw arith <op> <a> [b] | --batch <ops.json>` — Spec-free `Device::flex()` f32 arithmetic, linear algebra, statistics, axis reductions, and probability ops with length-1 broadcast and `"$prev"`/`"$<idx>"` batch register chaining.
- `math-draw int <add|sub|mul|div|rem|matmul|to-f32|from-f32> <a> [b] [--wrap] [--allow-inexact] | --batch <ops.json>` — Exact checked `i64` arithmetic and casts with fail-closed overflow detection and batch register chaining.
- `math-draw kernel <sin|cos|tan|floor|ceil|round|atan2|powf|hypot|fma> <a> [b] [c] | --batch <ops.json>` — Lane-E elementwise math kernels with strict same-length (no-broadcast) validation and batch register chaining.
- `math-draw geom <build|scene-tree|solve-ik|project|batch|selftest> ...` — 3D geometry domain pack: 4×4 transform chains (incl. `repeat`), `normalMat3`, hierarchical scene-tree compiler, 3D multi-joint IK solver, single & chunked block-diagonal batched (`BATCH_MAX = 63`) Burn projection with Sutherland–Hodgman near-plane clipping, and 17-check selftest.
- `math-draw trace-hash <spec.json> [other.json] [--max-bytes N] [--isolate]` — Per-step SHA-256 tensor fingerprinting, divergence step localization, and optional prefix subspec isolation.
- `math-draw rollout scan <spec.json> [--steps N] [--state-port <slot|name>] [--inputs-seq <seq.json>]` — Stateful autoregressive/recurrent rollout tracking `delta_linf` and per-tick state hashes.
- `math-draw rollout fixpoint <spec.json> [--tol 1e-6] [--max-steps 100] [--state-port <slot|name>]` — Fixed-point iteration until `||s_{t+1} - s_t||_∞ <= tol` with Lipschitz contraction estimates.
- `math-draw rollout unroll <spec.json> [--steps N] [--state-port <slot|name>] [--compact-slots] [--out <out.json>]` — Materialize an N-step weight-tied unrolled `v1` spec with optional linear-scan liveness slot compaction.
- `math-draw rollout bisect <base.json> <cand.json> [--steps N] [--state-port <slot|name>] [--inputs-seq <seq.json>] [--abs-tol 1e-5]` — Pinpoint both the first divergent rollout tick and the exact divergent intra-tick step.
- `math-draw profile <spec.json> [--iters 30] [--warmup 5]` — Benchmark latency (`min`/`median`/`p95`/`mean`/`max` ms), step count, parameter count, and output hash using a compiled `PreparedGraph`.
- `math-draw compare <base.json> <cand.json> [--iters 30] [--warmup 5]` — Benchmark and equivalence-check a baseline vs candidate spec.
- `math-draw cost-model <spec.json> [--iters 11] [--table <out.json>]` — Isolated per-op differential microbenchmark cost model, hotspot identification, and predicted-vs-measured whole-graph comparison.
- `math-draw gradcheck <spec.json> [analytic.json] [--eps 1e-3] [--tol 1e-3] [--weights] [--autodiff]` — Numeric central-difference Jacobians w.r.t. inputs and weights, analytic verification, and Burn autodiff cross-checking.
- `math-draw quant verify <spec.json> [--scheme int8|q8-per-block|fp16] [--cases <cases.json>] [--abs-tol 1e-2] [--rel-tol 1e-2] [--out <quant.json>]` — Weight quantization simulation (`int8`, `q8-per-block`, IEEE-754 `fp16`) and end-to-end equivalence verification.
- `math-draw quant budget <spec.json> [--cases <cases.json>] [--tol 0.05] [--scheme q8-per-block|int8|fp16] [--mode bound|measured] [--out <plan.json>]` — Per-layer quantization planner using analytical DAG Lipschitz propagation (`bound`) or empirical greedy measurement (`measured`).
- `math-draw mutate <spec.json> <mutation.json> [--cases <cases.json>] [--abs-tol 1e-6] [--rel-tol 1e-6] [--out <out.json>]` — Equivalence-gated single-transaction and multi-branch graph mutation (`setWeights`, `replaceStep`, `insertStep`, `removeStep`, `reconnectStep`, `setOutput`).
- `math-draw shard <spec.json> [--max-weights 65535] [--out <out.json>]` — Output-row linear layer sharding with exact weight reassembly and equivalence verification.
- `math-draw weight-lab <run|gate|certify|compose|eval|selftest>` — Manual-weighting workbench: engine probes, spectral radius / orthonormality / determinant property checks, adaptation jump gating, Frobenius norm-bound & orthonormal composition certificates, and recursive weight expression tree evaluation.
- `math-draw bundle <binding|export|verify> ...` — Canonical unique-first-use graph parameter binding (`burn-math.graph-parameter-binding.v1`), self-contained portable program bundle export (`burn-math.program-bundle.v1`) with execution receipts & provenance attestations, and cryptographic tamper + live replay verification.
- `math-draw harness <capabilities|digest|resolve-artifact|run-workflow|branch-export|branch-merge|workspace-status>` — Unified Layer-4 agent-harness co-processor interface, content-addressed tensor artifact store (`burn://sha256:...`), subagent branch coordinator, and multi-stage pipeline runner.
- `math-draw find-op [--list | <constructor>]` — Inspect supported constructors and parameter schemas.
- `math-draw rewrite list` / `math-draw rewrite apply <rule|all> <spec.json> [--out <out.json>]` — Pure graph rewrites (`linear-branch-fusion`, `linear-chain-fusion`, `relu-idempotent`, `transpose-involution`, `dead-step-elimination`, `add-commute`).
- `math-draw recurse expand <spec.v2.json> [--auto-repair] [--max-weights N] [--compact-slots] [--out <out.v1.json>]` — Compile recursive `v2` graph specs (subgraphs, `repeat` with `tie_weights`/`weights_per_depth`, `residual`, `parallel_concat`, `$ref`, 6 built-in macros incl. `attention_head`) into canonical flat `v1`, with optional deterministic auto-repair and linear-scan liveness slot compaction.
- `math-draw recurse grad-spec <spec.json> [--wrt <slot|name>] [--order 1|2] [--dout <v1,v2,...>] [--hvp-vec <v1,v2,...>] [--compact-slots] [--out <grad.json>]` — Compile a runnable 1st-order VJP or 2nd-order Hessian-Vector Product `SpecV1` graph across 23 differentiable constructors.
- `math-draw recurse search <spec.json> [--out <best.json>] [--max-depth 3] [--cases <cases.json>] [--quant-tol T] [--quant-scheme q8-per-block|int8|fp16]` — Recursive tree rewrite & quantization search verified against the baseline.
- `math-draw recurse solve-gd <spec.json> [--targets <targets.json>] [--steps 20] [--lr 0.1] [--layer <idx|all>] [--precondition] [--out <out.json>]` — Burn `Device::flex().autodiff()` gradient descent with backtracking and optional second-order diagonal Hessian preconditioning.
- `math-draw recurse fit <spec.json> --targets <targets.json> [--strategy openes|mu-lambda] [--rng xorshift64|mulberry32|pcg32|philox4x32] [--gens 40] [--pop 16] [--mu N] [--sigma 0.2] [--lr 0.2] [--seed 42] [--sensitivity-guided] [--freeze-threshold 1e-6] [--out <out.json>]` — Gradient-free Evolutionary Strategy optimizer supporting OpenES (antithetic) & `(mu, lambda)` elitist selection across 4 deterministic PRNGs, per-generation population diagnostics, and optional sensitivity-guided layer freezing.
- `math-draw smoke` — Run the built-in end-to-end 33-check smoke suite.

## Quick Examples

```bash
cargo run -- smoke
cargo run -- run examples/linear-relu.json
cargo run -- run examples/recursive-gated-residual.v2.json
cargo run -- validate examples/linear-relu.json --explain
cargo run -- diff-ref examples/linear-relu.json --tol 1e-5
cargo run -- soak examples/linear-relu.json --runs 20
cargo run -- prng-check philox4x32 42 100
cargo run -- arith matmul "1,2;3,4" "5,6;7,8"
cargo run -- int add 16777217,100 1,-30
cargo run -- kernel fma 2,3 4,5 1,-1
cargo run -- geom selftest
cargo run -- recurse expand examples/recursive-gated-residual.v2.json --auto-repair --compact-slots
cargo run -- recurse grad-spec examples/recursive-gated-residual.v2.json --wrt state --order 1 --compact-slots
cargo run -- trace-hash examples/linear-relu.json
cargo run -- rollout fixpoint examples/recursive-gated-residual.v2.json --tol 1e-5
cargo run -- profile examples/linear-relu.json --iters 20
cargo run -- cost-model examples/linear-relu.json --iters 11
cargo run -- gradcheck examples/linear-relu.json --weights --autodiff
cargo run -- quant verify examples/linear-relu.json --scheme q8-per-block
cargo run -- weight-lab selftest
cargo run -- bundle binding examples/linear-relu.json
cargo run -- bundle export examples/linear-relu.json --attest demo_v1 --out /tmp/demo.bundle.json
cargo run -- bundle verify /tmp/demo.bundle.json
cargo run -- harness capabilities
cargo run -- recurse solve-gd examples/linear-relu.json --targets examples/targets-linear.json --steps 20 --lr 0.05 --precondition
cargo run -- recurse fit examples/linear-relu.json --targets examples/targets-linear.json --strategy mu-lambda --rng philox4x32 --pop 12 --mu 3 --gens 20
```