# src/cli/ — CLI Argument Parsing, Subcommand Routing & JSON Output Contract

## Purpose

Implements the single-binary command-line interface for `math-draw`, parsing CLI flags/positional arguments, dispatching subcommands to library domain modules, writing optional JSON artifacts (`--out`, `--table`), and enforcing consistent JSON/exit-code output.

## Ownership

- Top-level dispatcher (`run_cli`), usage text (`print_usage`), strict fail-closed argument parser (`ParsedArgs`), JSON & file I/O helpers: [`mod.rs`](mod.rs)
- Core execution & inspection (`run`, `validate`, `verify`, `linear-equiv`, `find-op`, `smoke`): [`core_cmds.rs`](core_cmds.rs)
- Runtime telemetry & temporal ops (`trace-hash`, `rollout`, `profile`, `compare`, `cost-model`): [`runtime_cmds.rs`](runtime_cmds.rs)
- Structural graph transformations (`rewrite`, `recurse`, `mutate`, `shard`): [`graph_cmds.rs`](graph_cmds.rs)
- Differential, numerical & bundle verification (`gradcheck`, `quant`, `weight-lab`, `bundle`): [`evidence_cmds.rs`](evidence_cmds.rs)
- Layer-4 agent-harness coordination (`harness`): [`harness_cmds.rs`](harness_cmds.rs)

## Local Contracts

- **JSON-Only Stdout & Structured Error Envelope**: Every executed subcommand (success or error) prints valid pretty-printed JSON to stdout via panic-free `print_json_exit`, `err_exit_typed`, `err_exit_stage`, `err_exit_usage`, or `err_exit_code`. Error responses emit [`ErrorEnvelope`](../error.rs) (`{"ok": false, "code": "<snake_case>", "error": "...", "stage": "...", "hint": "..."}`), preserving backward-compatible `ok` and `error` keys (while `validate` / `run` static-validation failures preserve `ValidateReport` JSON directly). Human usage help (`--help` / `-h`) prints to stderr via `print_usage`.
- **Fail-Closed Flag & Numeric Parsing**: All subcommand handlers MUST parse arguments through [`ParsedArgs`](mod.rs) (returning `Result<_, MathDrawError>` with exact `ErrorCode` and `"cli_parse"` stage) — rejecting unknown `--...`/`-<alpha>` flags, missing values on valued flags, duplicate single-valued flags, unexpected extra positionals, and malformed/non-finite/out-of-domain numeric flag values (`NaN`, `Inf`, negative tolerances, non-positive `--eps`/`--lr`/`--sigma`) without silent fallback defaults.
- **Exit Code Contract**: Exit `0` (`ExitCode::SUCCESS`) exclusively when the primary result is `ok` / `valid` / `equivalent` / `converged`; exit `2` (`ExitCode::from(2)`) on CLI usage errors, parse failures, or verification rejections.
- **No Business Logic in CLI Handlers**: Handlers parse flags/files, invoke library APIs in [`crate`](../AGENTS.md), optionally serialize `--out`, and format JSON. Graph algorithms belong in domain modules, never in `src/cli/`.

## Work Guidance

- Register every new subcommand flag in its handler's `ParsedArgs::parse` / `ParsedArgs::parse_full` allowlist (valued, optional-valued, multi-valued, or boolean) and check `ensure_max_positionals`.
- Update `print_usage()` in [`mod.rs`](mod.rs) and [`README.md`](../../README.md) whenever adding or modifying subcommand flags.

## Verification

- `./target/release/math-draw --help`
- `./target/release/math-draw smoke`
- `./target/release/math-draw harness capabilities`
- `./target/release/math-draw weight-lab selftest`

## Child Graph Index

- Leaf domain directory; submodules listed under Ownership (`mod.rs`, `core_cmds.rs`, `runtime_cmds.rs`, `graph_cmds.rs`, `evidence_cmds.rs`, `harness_cmds.rs`).
