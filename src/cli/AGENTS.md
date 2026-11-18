# src/cli/ — CLI Argument Parsing, Subcommand Routing & JSON Output Contract

## Purpose

Implements the single-binary command-line interface for `math-draw`, parsing CLI flags/positional arguments, dispatching subcommands to library domain modules, writing optional JSON artifacts (`--out`, `--table`), and enforcing consistent JSON/exit-code output.

## Ownership

- Top-level dispatcher (`run_cli`), usage text (`print_usage`), argument helpers (`flag_val`, `has_flag`, `positional_args`), JSON & file I/O helpers: [`mod.rs`](mod.rs)
- Core execution & inspection (`run`, `validate`, `verify`, `linear-equiv`, `find-op`, `smoke`): [`core_cmds.rs`](core_cmds.rs)
- Runtime telemetry & temporal ops (`trace-hash`, `rollout`, `profile`, `compare`, `cost-model`): [`runtime_cmds.rs`](runtime_cmds.rs)
- Structural graph transformations (`rewrite`, `recurse`, `mutate`, `shard`): [`graph_cmds.rs`](graph_cmds.rs)
- Differential & numerical verification (`gradcheck`, `quant`, `weight-lab`): [`evidence_cmds.rs`](evidence_cmds.rs)
- Layer-4 agent-harness coordination (`harness`): [`harness_cmds.rs`](harness_cmds.rs)

## Local Contracts

- **JSON-Only Stdout**: Every executed subcommand (success or error) prints valid pretty-printed JSON to stdout (`{"ok": false, "error": ...}` on errors). Human usage help (`--help` / `-h`) prints to stderr via `print_usage`.
- **Exit Code Contract**: Exit `0` (`ExitCode::SUCCESS`) exclusively when the primary result is `ok` / `valid` / `equivalent` / `converged`; exit `2` (`ExitCode::from(2)`) on CLI usage errors, parse failures, or verification rejections.
- **No Business Logic in CLI Handlers**: Handlers parse flags/files, invoke library APIs in [`crate`](../AGENTS.md), optionally serialize `--out`, and format JSON. Graph algorithms belong in domain modules, never in `src/cli/`.

## Work Guidance

- Keep `positional_args(rest, &valued_flags)` accurate by listing every flag taking a value in `valued_flags` so flag values are never mistaken for positional file paths.
- Update `print_usage()` in [`mod.rs`](mod.rs) and [`README.md`](../../README.md) whenever adding or modifying subcommand flags.

## Verification

- `./target/release/math-draw --help`
- `./target/release/math-draw smoke`
- `./target/release/math-draw harness capabilities`
- `./target/release/math-draw weight-lab selftest`

## Child Graph Index

- Leaf domain directory; submodules listed under Ownership (`mod.rs`, `core_cmds.rs`, `runtime_cmds.rs`, `graph_cmds.rs`, `evidence_cmds.rs`, `harness_cmds.rs`).
