mod core_cmds;
mod evidence_cmds;
mod graph_cmds;
mod harness_cmds;
mod runtime_cmds;

use serde_json::json;
use std::process::ExitCode;

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
  math-draw harness <capabilities|digest|resolve-artifact|run-workflow|branch-export|branch-merge|workspace-status> ...
  math-draw find-op [--list | <constructor>]
  math-draw rewrite list
  math-draw rewrite apply <rule|all> <spec.json> [--out <out.json>]
  math-draw recurse expand <spec.v2.json> [--out <out.v1.json>]
  math-draw recurse search <spec.json> [--out <best.json>] [--max-depth 3] [--cases <cases.json>] [--abs-tol 1e-5] [--rel-tol 1e-5]
  math-draw recurse solve-gd <spec.json> [--targets <targets.json>] [--steps 20] [--lr 0.1] [--layer <idx|all>] [--precondition] [--out <out.json>]
  math-draw recurse fit <spec.json> --targets <targets.json> [--gens 40] [--pop 16] [--sigma 0.2] [--lr 0.2] [--seed 42] [--sensitivity-guided] [--freeze-threshold 1e-6] [--out <out.json>]
  math-draw smoke"#
    );
}

pub(crate) fn flag_val<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].as_str())
}

pub(crate) fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

pub(crate) fn positional_args(args: &[String], valued_flags: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    let mut skip_next = false;
    for a in args {
        if skip_next {
            skip_next = false;
            continue;
        }
        if valued_flags.contains(&a.as_str()) {
            skip_next = true;
            continue;
        }
        if a.starts_with("--") {
            continue;
        }
        out.push(a.clone());
    }
    out
}

pub(crate) fn print_json_exit<T: serde::Serialize>(val: &T, ok: bool) -> ExitCode {
    println!("{}", serde_json::to_string_pretty(val).unwrap());
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    }
}

pub(crate) fn err_exit(msg: &str) -> ExitCode {
    println!("{}", json!({ "ok": false, "error": msg }));
    ExitCode::from(2)
}

pub(crate) fn read_cases_file(path: &str) -> Result<Vec<EvalCase>, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read cases file '{path}': {e}"))?;
    if let Ok(tf) = serde_json::from_str::<TargetsFile>(&raw) {
        return Ok(tf.cases);
    }
    if let Ok(arr) = serde_json::from_str::<Vec<EvalCase>>(&raw) {
        return Ok(arr);
    }
    Err(format!("invalid cases JSON in '{path}'"))
}

pub(crate) fn read_targets_file(path: &str) -> Result<TargetsFile, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read targets file '{path}': {e}"))?;
    let tf: TargetsFile =
        serde_json::from_str(&raw).map_err(|e| format!("invalid targets JSON in '{path}': {e}"))?;
    if tf.cases.is_empty() {
        return Err("targets.cases must be non-empty".into());
    }
    Ok(tf)
}

pub(crate) fn write_pretty_json<T: serde::Serialize>(path: &str, val: &T) -> Result<(), String> {
    let txt = serde_json::to_string_pretty(val).map_err(|e| e.to_string())?;
    std::fs::write(path, format!("{txt}\n"))
        .map_err(|e| format!("failed writing '{path}': {e}"))
}

pub fn run_cli() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || has_flag(&args, "--help") || has_flag(&args, "-h") {
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
        "smoke" => core_cmds::handle_smoke(),
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
        "harness" => harness_cmds::handle_harness(rest),
        _ => err_exit(&format!("unknown command '{cmd}' (run `math-draw --help`)")),
    }
}
