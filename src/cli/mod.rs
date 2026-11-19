mod core_cmds;
mod evidence_cmds;
mod graph_cmds;
mod harness_cmds;
mod runtime_cmds;

use serde_json::json;
use std::process::ExitCode;

use crate::error::{ErrorCode, MathDrawError, classify_error};
use crate::verify::{EvalCase, TargetsFile};

fn print_usage() {
    eprintln!(
        r#"math-draw — Standalone Rust/Burn 0.22.1 (burn-flex) Graph Engine & Recursive Engineering CLI

USAGE:
  math-draw run <spec.json>
  math-draw validate <spec.json> [--explain]
  math-draw verify <base.json> <candidate.json> [--cases <cases.json>] [--fuzz [N]] [--zoom] [--zoom-depth 4] [--seed S] [--lo A] [--hi B] [--abs-tol T] [--rel-tol T]
  math-draw linear-equiv <specA.json> <specB.json> [--tol 0.0]
  math-draw trace-hash <spec.json> [other.json] [--max-bytes N] [--isolate]
  math-draw rollout scan <spec.json> [--steps N] [--state-port <slot|name>] [--inputs-seq <seq.json>]
  math-draw rollout fixpoint <spec.json> [--tol 1e-6] [--max-steps 100] [--state-port <slot|name>]
  math-draw rollout unroll <spec.json> [--steps N] [--state-port <slot|name>] [--out <out.json>]
  math-draw rollout bisect <base.json> <cand.json> [--steps N] [--state-port <slot|name>] [--inputs-seq <seq.json>] [--abs-tol 1e-5]
  math-draw profile <spec.json> [--iters 30] [--warmup 5]
  math-draw compare <base.json> <cand.json> [--iters 30] [--warmup 5]
  math-draw cost-model <spec.json> [--iters 11] [--table <out.json>]
  math-draw gradcheck <spec.json> [analytic.json] [--eps 1e-3] [--tol 1e-3] [--weights] [--autodiff]
  math-draw quant verify <spec.json> [--scheme int8|q8-per-block|fp16] [--cases <cases.json>] [--abs-tol 1e-2] [--rel-tol 1e-2] [--out <quant.json>]
  math-draw quant budget <spec.json> [--cases <cases.json>] [--tol 0.05] [--scheme q8-per-block|int8|fp16] [--mode bound|measured] [--out <plan.json>]
  math-draw mutate <spec.json> <mutation.json> [--cases <cases.json>] [--abs-tol 1e-6] [--rel-tol 1e-6] [--out <out.json>]
  math-draw shard <spec.json> [--max-weights 65535] [--out <out.json>]
  math-draw weight-lab <run|gate|certify|compose|eval|selftest> ...
  math-draw bundle binding <spec.json>
  math-draw bundle export <spec.json> [--out <bundle.json>] [--attest <label>] [--no-exec]
  math-draw bundle verify <bundle.json> [--out-spec <spec.json>]
  math-draw harness <capabilities|digest|resolve-artifact|run-workflow|branch-export|branch-merge|workspace-status> ...
  math-draw find-op [--list | <constructor>]
  math-draw rewrite list
  math-draw rewrite apply <rule|all> <spec.json> [--out <out.json>]
  math-draw recurse expand <spec.v2.json> [--out <out.v1.json>]
  math-draw recurse search <spec.json> [--out <best.json>] [--max-depth 3] [--cases <cases.json>] [--abs-tol 1e-5] [--rel-tol 1e-5]
  math-draw recurse solve-gd <spec.json> [--targets <targets.json>] [--steps 20] [--lr 0.1] [--layer <idx|all>] [--precondition] [--out <out.json>]
  math-draw recurse fit <spec.json> --targets <targets.json> [--strategy openes|mu-lambda] [--rng xorshift64|mulberry32|pcg32|philox4x32] [--gens 40] [--pop 16] [--mu N] [--sigma 0.2] [--lr 0.2] [--seed 42] [--sensitivity-guided] [--freeze-threshold 1e-6] [--out <out.json>]
  math-draw smoke"#
    );
}

#[derive(Debug, Default)]
pub(crate) struct ParsedArgs {
    pub(crate) positionals: Vec<String>,
    bool_flags: std::collections::BTreeSet<String>,
    single_values: std::collections::BTreeMap<String, String>,
    multi_values: std::collections::BTreeMap<String, Vec<String>>,
}

impl ParsedArgs {
    pub(crate) fn parse(
        args: &[String],
        valued: &[&str],
        bools: &[&str],
    ) -> Result<Self, MathDrawError> {
        Self::parse_full(args, valued, &[], &[], bools)
    }

    pub(crate) fn parse_full(
        args: &[String],
        valued: &[&str],
        optional_valued: &[&str],
        multi_valued: &[&str],
        bools: &[&str],
    ) -> Result<Self, MathDrawError> {
        let mut parsed = Self::default();
        let mut i = 0usize;
        while i < args.len() {
            let tok = args[i].as_str();
            if valued.contains(&tok) {
                if parsed.single_values.contains_key(tok) {
                    return Err(
                        MathDrawError::new(
                            ErrorCode::CliDuplicateFlag,
                            format!("duplicate flag '{tok}'"),
                        )
                        .with_stage("cli_parse"),
                    );
                }
                let Some(next) = args.get(i + 1) else {
                    return Err(
                        MathDrawError::new(
                            ErrorCode::CliMissingValue,
                            format!("flag '{tok}' requires a value"),
                        )
                        .with_stage("cli_parse"),
                    );
                };
                if is_flag_like(next) {
                    return Err(
                        MathDrawError::new(
                            ErrorCode::CliMissingValue,
                            format!("flag '{tok}' requires a value (got '{next}')"),
                        )
                        .with_stage("cli_parse"),
                    );
                }
                parsed.single_values.insert(tok.to_string(), next.clone());
                i += 2;
            } else if multi_valued.contains(&tok) {
                let Some(next) = args.get(i + 1) else {
                    return Err(
                        MathDrawError::new(
                            ErrorCode::CliMissingValue,
                            format!("flag '{tok}' requires a value"),
                        )
                        .with_stage("cli_parse"),
                    );
                };
                if is_flag_like(next) {
                    return Err(
                        MathDrawError::new(
                            ErrorCode::CliMissingValue,
                            format!("flag '{tok}' requires a value (got '{next}')"),
                        )
                        .with_stage("cli_parse"),
                    );
                }
                parsed
                    .multi_values
                    .entry(tok.to_string())
                    .or_default()
                    .push(next.clone());
                i += 2;
            } else if optional_valued.contains(&tok) {
                if parsed.bool_flags.contains(tok) {
                    return Err(
                        MathDrawError::new(
                            ErrorCode::CliDuplicateFlag,
                            format!("duplicate flag '{tok}'"),
                        )
                        .with_stage("cli_parse"),
                    );
                }
                parsed.bool_flags.insert(tok.to_string());
                if let Some(next) = args.get(i + 1)
                    && !is_flag_like(next)
                    && !looks_like_spec_path(next)
                {
                    parsed.single_values.insert(tok.to_string(), next.clone());
                    i += 2;
                } else {
                    i += 1;
                }
            } else if bools.contains(&tok) {
                parsed.bool_flags.insert(tok.to_string());
                i += 1;
            } else if is_flag_like(tok) {
                return Err(
                    MathDrawError::new(
                        ErrorCode::CliUnknownFlag,
                        format!("unknown flag '{tok}'"),
                    )
                    .with_stage("cli_parse"),
                );
            } else {
                parsed.positionals.push(tok.to_string());
                i += 1;
            }
        }
        Ok(parsed)
    }

    pub(crate) fn has(&self, flag: &str) -> bool {
        self.bool_flags.contains(flag)
            || self.single_values.contains_key(flag)
            || self.multi_values.contains_key(flag)
    }

    pub(crate) fn str_opt(&self, flag: &str) -> Option<&str> {
        self.single_values.get(flag).map(|s| s.as_str())
    }

    pub(crate) fn str_all(&self, flag: &str) -> Vec<String> {
        self.multi_values.get(flag).cloned().unwrap_or_default()
    }

    pub(crate) fn ensure_max_positionals(
        &self,
        max: usize,
        usage: &str,
    ) -> Result<(), MathDrawError> {
        if self.positionals.len() > max {
            return Err(
                MathDrawError::new(
                    ErrorCode::CliUnexpectedPositional,
                    format!(
                        "unexpected positional argument '{}' ({usage})",
                        self.positionals[max]
                    ),
                )
                .with_stage("cli_parse"),
            );
        }
        Ok(())
    }

    pub(crate) fn parse_usize(&self, flag: &str, default: usize) -> Result<usize, MathDrawError> {
        match self.str_opt(flag) {
            Some(raw) => raw.parse::<usize>().map_err(|_| {
                MathDrawError::new(
                    ErrorCode::CliInvalidNumber,
                    format!("invalid {flag} value '{raw}': expected non-negative integer"),
                )
                .with_stage("cli_parse")
            }),
            None => Ok(default),
        }
    }

    pub(crate) fn parse_usize_opt(&self, flag: &str) -> Result<Option<usize>, MathDrawError> {
        match self.str_opt(flag) {
            Some(raw) => raw.parse::<usize>().map(Some).map_err(|_| {
                MathDrawError::new(
                    ErrorCode::CliInvalidNumber,
                    format!("invalid {flag} value '{raw}': expected non-negative integer"),
                )
                .with_stage("cli_parse")
            }),
            None => Ok(None),
        }
    }

    pub(crate) fn parse_u64(&self, flag: &str, default: u64) -> Result<u64, MathDrawError> {
        match self.str_opt(flag) {
            Some(raw) => raw.parse::<u64>().map_err(|_| {
                MathDrawError::new(
                    ErrorCode::CliInvalidNumber,
                    format!("invalid {flag} value '{raw}': expected unsigned 64-bit integer"),
                )
                .with_stage("cli_parse")
            }),
            None => Ok(default),
        }
    }

    pub(crate) fn parse_f64(&self, flag: &str, default: f64) -> Result<f64, MathDrawError> {
        match self.str_opt(flag) {
            Some(raw) => {
                let v = raw.parse::<f64>().map_err(|_| {
                    MathDrawError::new(
                        ErrorCode::CliInvalidNumber,
                        format!("invalid {flag} value '{raw}': expected finite number"),
                    )
                    .with_stage("cli_parse")
                })?;
                if !v.is_finite() {
                    return Err(
                        MathDrawError::new(
                            ErrorCode::CliInvalidNumber,
                            format!("invalid {flag} value '{raw}': must be finite"),
                        )
                        .with_stage("cli_parse"),
                    );
                }
                Ok(v)
            }
            None => Ok(default),
        }
    }

    pub(crate) fn parse_f64_non_neg(&self, flag: &str, default: f64) -> Result<f64, MathDrawError> {
        let v = self.parse_f64(flag, default)?;
        if v < 0.0 {
            return Err(
                MathDrawError::new(
                    ErrorCode::CliInvalidRange,
                    format!("invalid {flag} value '{v}': must be >= 0"),
                )
                .with_stage("cli_parse"),
            );
        }
        Ok(v)
    }

    pub(crate) fn parse_f64_pos(&self, flag: &str, default: f64) -> Result<f64, MathDrawError> {
        let v = self.parse_f64(flag, default)?;
        if v <= 0.0 {
            return Err(
                MathDrawError::new(
                    ErrorCode::CliInvalidRange,
                    format!("invalid {flag} value '{v}': must be > 0"),
                )
                .with_stage("cli_parse"),
            );
        }
        Ok(v)
    }
}

fn is_flag_like(tok: &str) -> bool {
    tok.starts_with("--")
        || (tok.starts_with('-')
            && tok.len() >= 2
            && tok.as_bytes()[1].is_ascii_alphabetic())
}

fn looks_like_spec_path(tok: &str) -> bool {
    tok.ends_with(".json") || std::path::Path::new(tok).exists()
}

pub(crate) fn print_json_exit<T: serde::Serialize>(val: &T, ok: bool) -> ExitCode {
    match serde_json::to_string_pretty(val) {
        Ok(serialized) => {
            println!("{serialized}");
            if ok {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(2)
            }
        }
        Err(e) => err_exit_typed(
            &MathDrawError::new(
                ErrorCode::JsonSerializeFailed,
                format!("failed to serialize JSON output: {e}"),
            )
            .with_stage("serialize"),
        ),
    }
}

pub(crate) fn err_exit_typed(err: &MathDrawError) -> ExitCode {
    let env = err.to_envelope();
    match serde_json::to_string_pretty(&env) {
        Ok(s) => println!("{s}"),
        Err(_) => println!(
            "{}",
            json!({
                "ok": false,
                "code": env.code.as_str(),
                "error": env.error,
            })
        ),
    }
    ExitCode::from(2)
}

pub(crate) fn err_exit_stage(stage: &str, msg: &str) -> ExitCode {
    err_exit_typed(&classify_error(msg, Some(stage)))
}

pub(crate) fn err_exit_usage(stage: &str, usage: &str) -> ExitCode {
    err_exit_typed(&MathDrawError::new(ErrorCode::CliUsage, usage).with_stage(stage))
}

pub(crate) fn err_exit_code(code: ErrorCode, stage: &str, msg: &str) -> ExitCode {
    err_exit_typed(&MathDrawError::new(code, msg).with_stage(stage))
}

pub(crate) fn read_cases_file(path: &str) -> Result<Vec<EvalCase>, MathDrawError> {
    let raw = std::fs::read_to_string(path).map_err(|e| {
        MathDrawError::new(
            ErrorCode::IoReadFailed,
            format!("cannot read cases file '{path}': {e}"),
        )
        .with_stage("io")
    })?;
    if let Ok(tf) = serde_json::from_str::<TargetsFile>(&raw) {
        return Ok(tf.cases);
    }
    if let Ok(arr) = serde_json::from_str::<Vec<EvalCase>>(&raw) {
        return Ok(arr);
    }
    Err(
        MathDrawError::new(
            ErrorCode::JsonParseFailed,
            format!("invalid cases JSON in '{path}'"),
        )
        .with_stage("io"),
    )
}

pub(crate) fn read_targets_file(path: &str) -> Result<TargetsFile, MathDrawError> {
    let raw = std::fs::read_to_string(path).map_err(|e| {
        MathDrawError::new(
            ErrorCode::IoReadFailed,
            format!("cannot read targets file '{path}': {e}"),
        )
        .with_stage("io")
    })?;
    let tf: TargetsFile = serde_json::from_str(&raw).map_err(|e| {
        MathDrawError::new(
            ErrorCode::JsonParseFailed,
            format!("invalid targets JSON in '{path}': {e}"),
        )
        .with_stage("io")
    })?;
    if tf.cases.is_empty() {
        return Err(
            MathDrawError::new(
                ErrorCode::MissingExpectedTarget,
                "targets.cases must be non-empty",
            )
            .with_stage("io"),
        );
    }
    Ok(tf)
}

pub(crate) fn write_pretty_json<T: serde::Serialize>(
    path: &str,
    val: &T,
) -> Result<(), MathDrawError> {
    let txt = serde_json::to_string_pretty(val).map_err(|e| {
        MathDrawError::new(ErrorCode::JsonSerializeFailed, e.to_string()).with_stage("io")
    })?;
    std::fs::write(path, format!("{txt}\n")).map_err(|e| {
        MathDrawError::new(
            ErrorCode::IoWriteFailed,
            format!("failed writing '{path}': {e}"),
        )
        .with_stage("io")
    })
}

pub fn run_cli() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return if args.is_empty() {
            ExitCode::from(2)
        } else {
            ExitCode::SUCCESS
        };
    }

    let cmd = args[0].as_str();
    let rest = &args[1..];

    match cmd {
        "run" => core_cmds::handle_run(rest),
        "validate" => core_cmds::handle_validate(rest),
        "verify" => core_cmds::handle_verify(rest),
        "linear-equiv" => core_cmds::handle_linear_equiv(rest),
        "find-op" => core_cmds::handle_find_op(rest),
        "smoke" => core_cmds::handle_smoke(rest),
        "trace-hash" => runtime_cmds::handle_trace_hash(rest),
        "rollout" => runtime_cmds::handle_rollout(rest),
        "profile" => runtime_cmds::handle_profile(rest),
        "compare" => runtime_cmds::handle_compare(rest),
        "cost-model" => runtime_cmds::handle_cost_model(rest),
        "rewrite" => graph_cmds::handle_rewrite(rest),
        "recurse" => graph_cmds::handle_recurse(rest),
        "mutate" => graph_cmds::handle_mutate(rest),
        "shard" => graph_cmds::handle_shard(rest),
        "gradcheck" => evidence_cmds::handle_gradcheck(rest),
        "quant" => evidence_cmds::handle_quant(rest),
        "weight-lab" => evidence_cmds::handle_weight_lab(rest),
        "bundle" => evidence_cmds::handle_bundle(rest),
        "harness" => harness_cmds::handle_harness(rest),
        _ => err_exit_code(
            ErrorCode::CliUnknownCommand,
            "cli_dispatch",
            &format!("unknown command '{cmd}' (run `math-draw --help`)"),
        ),
    }
}
