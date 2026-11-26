use std::path::Path;
use std::process::ExitCode;

use super::{
    ParsedArgs, err_exit_code, err_exit_stage, err_exit_typed, err_exit_usage, print_json_exit,
    write_pretty_json,
};
use crate::cost_model::run_cost_model;
use crate::error::ErrorCode;
use crate::profile::{compare_specs_perf, profile_spec};
use crate::rollout::{
    parse_inputs_sequence, rollout_bisect, rollout_fixpoint, rollout_scan,
    rollout_unroll_with_options,
};
use crate::spec::load_spec_file;
use crate::trace::{DEFAULT_MAX_TENSOR_BYTES, compare_trace_hashes, trace_hash_spec};

pub(super) fn handle_trace_hash(rest: &[String]) -> ExitCode {
    let usage = "usage: math-draw trace-hash <spec.json> [other.json] [--max-bytes N] [--isolate]";
    let parsed = match ParsedArgs::parse(rest, &["--max-bytes"], &["--isolate"]) {
        Ok(p) => p,
        Err(e) => return err_exit_typed(&e),
    };
    if let Err(e) = parsed.ensure_max_positionals(2, usage) {
        return err_exit_typed(&e);
    }
    let Some(first) = parsed.positionals.first() else {
        return err_exit_usage("trace_hash", usage);
    };
    let max_bytes = match parsed.parse_usize("--max-bytes", DEFAULT_MAX_TENSOR_BYTES) {
        Ok(v) => v,
        Err(e) => return err_exit_typed(&e),
    };
    let isolate = parsed.has("--isolate");
    let left = match load_spec_file(Path::new(first)) {
        Ok(l) => l.spec,
        Err(e) => return err_exit_stage("load_spec", &e),
    };
    if let Some(second) = parsed.positionals.get(1) {
        let right = match load_spec_file(Path::new(second)) {
            Ok(l) => l.spec,
            Err(e) => return err_exit_stage("load_spec", &e),
        };
        match compare_trace_hashes(&left, &right, max_bytes, isolate) {
            Ok(rep) => print_json_exit(&rep, rep.ok),
            Err(e) => err_exit_stage("trace_hash", &e),
        }
    } else {
        match trace_hash_spec(&left, max_bytes) {
            Ok(rep) => print_json_exit(&rep, rep.ok),
            Err(e) => err_exit_stage("trace_hash", &e),
        }
    }
}

pub(super) fn handle_rollout(rest: &[String]) -> ExitCode {
    let Some(sub) = rest.first().map(|s| s.as_str()) else {
        return err_exit_usage(
            "rollout",
            "usage: math-draw rollout <scan|fixpoint|unroll|bisect> ...",
        );
    };
    let sub_rest = &rest[1..];
    match sub {
        "scan" => {
            let usage = "usage: math-draw rollout scan <spec.json> [--steps N] [--state-port <slot|name>] [--inputs-seq <seq.json>]";
            let parsed = match ParsedArgs::parse(
                sub_rest,
                &["--steps", "--state-port", "--inputs-seq"],
                &[],
            ) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(1, usage) {
                return err_exit_typed(&e);
            }
            let Some(file) = parsed.positionals.first() else {
                return err_exit_usage("rollout.scan", usage);
            };
            let loaded = match load_spec_file(Path::new(file)) {
                Ok(l) => l,
                Err(e) => return err_exit_stage("load_spec", &e),
            };
            let steps = match parsed.parse_usize("--steps", 8) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            let state_port = parsed.str_opt("--state-port");
            let seq = if let Some(sp) = parsed.str_opt("--inputs-seq") {
                let raw = match std::fs::read_to_string(sp) {
                    Ok(r) => r,
                    Err(e) => {
                        return err_exit_code(
                            ErrorCode::IoReadFailed,
                            "rollout.scan",
                            &format!("cannot read inputs-seq '{sp}': {e}"),
                        );
                    }
                };
                match parse_inputs_sequence(&raw, loaded.expansion.as_ref()) {
                    Ok(s) => Some(s),
                    Err(e) => return err_exit_stage("rollout.scan", &e),
                }
            } else {
                None
            };
            match rollout_scan(
                &loaded.spec,
                loaded.expansion.as_ref(),
                steps,
                state_port,
                seq.as_deref(),
            ) {
                Ok(rep) => print_json_exit(&rep, rep.ok),
                Err(e) => err_exit_stage("rollout.scan", &e),
            }
        }
        "fixpoint" => {
            let usage = "usage: math-draw rollout fixpoint <spec.json> [--tol 1e-6] [--max-steps 100] [--state-port <slot|name>]";
            let parsed = match ParsedArgs::parse(
                sub_rest,
                &["--tol", "--max-steps", "--state-port"],
                &[],
            ) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(1, usage) {
                return err_exit_typed(&e);
            }
            let Some(file) = parsed.positionals.first() else {
                return err_exit_usage("rollout.fixpoint", usage);
            };
            let loaded = match load_spec_file(Path::new(file)) {
                Ok(l) => l,
                Err(e) => return err_exit_stage("load_spec", &e),
            };
            let tol = match parsed.parse_f64_non_neg("--tol", 1e-6) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            let max_steps = match parsed.parse_usize("--max-steps", 100) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            let state_port = parsed.str_opt("--state-port");
            match rollout_fixpoint(
                &loaded.spec,
                loaded.expansion.as_ref(),
                tol,
                max_steps,
                state_port,
            ) {
                Ok(rep) => print_json_exit(&rep, rep.converged),
                Err(e) => err_exit_stage("rollout.fixpoint", &e),
            }
        }
        "unroll" => {
            let usage = "usage: math-draw rollout unroll <spec.json> [--steps N] [--state-port <slot|name>] [--compact-slots] [--out <out.json>]";
            let parsed = match ParsedArgs::parse(
                sub_rest,
                &["--steps", "--state-port", "--out"],
                &["--compact-slots"],
            ) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(1, usage) {
                return err_exit_typed(&e);
            }
            let Some(file) = parsed.positionals.first() else {
                return err_exit_usage("rollout.unroll", usage);
            };
            let loaded = match load_spec_file(Path::new(file)) {
                Ok(l) => l,
                Err(e) => return err_exit_stage("load_spec", &e),
            };
            let steps = match parsed.parse_usize("--steps", 4) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            let state_port = parsed.str_opt("--state-port");
            let compact = parsed.has("--compact-slots");
            match rollout_unroll_with_options(
                &loaded.spec,
                loaded.expansion.as_ref(),
                steps,
                state_port,
                compact,
            ) {
                Ok(rep) => {
                    if let Some(op) = parsed.str_opt("--out")
                        && let Err(e) = write_pretty_json(op, &rep.unrolled_spec)
                    {
                        return err_exit_typed(&e);
                    }
                    print_json_exit(&rep, rep.ok)
                }
                Err(e) => err_exit_stage("rollout.unroll", &e),
            }
        }
        "bisect" => {
            let usage = "usage: math-draw rollout bisect <base.json> <cand.json> [--steps N] [--state-port <slot|name>] [--inputs-seq <seq.json>] [--abs-tol 1e-5]";
            let parsed = match ParsedArgs::parse(
                sub_rest,
                &["--steps", "--state-port", "--inputs-seq", "--abs-tol"],
                &[],
            ) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(2, usage) {
                return err_exit_typed(&e);
            }
            if parsed.positionals.len() < 2 {
                return err_exit_usage("rollout.bisect", usage);
            }
            let base = match load_spec_file(Path::new(&parsed.positionals[0])) {
                Ok(l) => l,
                Err(e) => return err_exit_stage("load_spec", &e),
            };
            let cand = match load_spec_file(Path::new(&parsed.positionals[1])) {
                Ok(l) => l,
                Err(e) => return err_exit_stage("load_spec", &e),
            };
            let steps = match parsed.parse_usize("--steps", 8) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            let state_port = parsed.str_opt("--state-port");
            let abs_tol = match parsed.parse_f64_non_neg("--abs-tol", 1e-5) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            let seq = if let Some(sp) = parsed.str_opt("--inputs-seq") {
                let raw = match std::fs::read_to_string(sp) {
                    Ok(r) => r,
                    Err(e) => {
                        return err_exit_code(
                            ErrorCode::IoReadFailed,
                            "rollout.bisect",
                            &format!("cannot read inputs-seq '{sp}': {e}"),
                        );
                    }
                };
                match parse_inputs_sequence(&raw, base.expansion.as_ref()) {
                    Ok(s) => Some(s),
                    Err(e) => return err_exit_stage("rollout.bisect", &e),
                }
            } else {
                None
            };
            match rollout_bisect(
                &base.spec,
                &cand.spec,
                base.expansion.as_ref(),
                steps,
                state_port,
                seq.as_deref(),
                abs_tol,
            ) {
                Ok(rep) => print_json_exit(&rep, rep.ok),
                Err(e) => err_exit_stage("rollout.bisect", &e),
            }
        }
        _ => err_exit_code(
            ErrorCode::CliUnknownCommand,
            "rollout",
            &format!("unknown rollout subcommand '{sub}' (use: scan | fixpoint | unroll | bisect)"),
        ),
    }
}

pub(super) fn handle_profile(rest: &[String]) -> ExitCode {
    let usage = "usage: math-draw profile <spec.json> [--iters 30] [--warmup 5]";
    let parsed = match ParsedArgs::parse(rest, &["--iters", "--warmup"], &[]) {
        Ok(p) => p,
        Err(e) => return err_exit_typed(&e),
    };
    if let Err(e) = parsed.ensure_max_positionals(1, usage) {
        return err_exit_typed(&e);
    }
    let Some(file) = parsed.positionals.first() else {
        return err_exit_usage("profile", usage);
    };
    let loaded = match load_spec_file(Path::new(file)) {
        Ok(l) => l,
        Err(e) => return err_exit_stage("load_spec", &e),
    };
    let iters = match parsed.parse_usize("--iters", 30) {
        Ok(v) => v,
        Err(e) => return err_exit_typed(&e),
    };
    let warmup = match parsed.parse_usize("--warmup", 5) {
        Ok(v) => v,
        Err(e) => return err_exit_typed(&e),
    };
    match profile_spec(&loaded.spec, warmup, iters) {
        Ok(rep) => print_json_exit(&rep, rep.ok),
        Err(e) => err_exit_stage("profile", &e),
    }
}

pub(super) fn handle_compare(rest: &[String]) -> ExitCode {
    let usage = "usage: math-draw compare <base.json> <cand.json> [--iters 30] [--warmup 5]";
    let parsed = match ParsedArgs::parse(rest, &["--iters", "--warmup"], &[]) {
        Ok(p) => p,
        Err(e) => return err_exit_typed(&e),
    };
    if let Err(e) = parsed.ensure_max_positionals(2, usage) {
        return err_exit_typed(&e);
    }
    if parsed.positionals.len() < 2 {
        return err_exit_usage("compare", usage);
    }
    let base = match load_spec_file(Path::new(&parsed.positionals[0])) {
        Ok(l) => l.spec,
        Err(e) => return err_exit_stage("load_spec", &e),
    };
    let cand = match load_spec_file(Path::new(&parsed.positionals[1])) {
        Ok(l) => l.spec,
        Err(e) => return err_exit_stage("load_spec", &e),
    };
    let iters = match parsed.parse_usize("--iters", 30) {
        Ok(v) => v,
        Err(e) => return err_exit_typed(&e),
    };
    let warmup = match parsed.parse_usize("--warmup", 5) {
        Ok(v) => v,
        Err(e) => return err_exit_typed(&e),
    };
    match compare_specs_perf(&base, &cand, warmup, iters) {
        Ok(rep) => print_json_exit(&rep, rep.ok),
        Err(e) => err_exit_stage("compare", &e),
    }
}

pub(super) fn handle_cost_model(rest: &[String]) -> ExitCode {
    let usage = "usage: math-draw cost-model <spec.json> [iters] [--iters 11] [--table <out.json>]";
    let parsed = match ParsedArgs::parse(rest, &["--iters", "--table"], &[]) {
        Ok(p) => p,
        Err(e) => return err_exit_typed(&e),
    };
    if let Err(e) = parsed.ensure_max_positionals(2, usage) {
        return err_exit_typed(&e);
    }
    let Some(file) = parsed.positionals.first() else {
        return err_exit_usage("cost_model", usage);
    };
    let loaded = match load_spec_file(Path::new(file)) {
        Ok(l) => l,
        Err(e) => return err_exit_stage("load_spec", &e),
    };
    let iters = match (parsed.parse_usize_opt("--iters"), parsed.positionals.get(1)) {
        (Err(e), _) => return err_exit_typed(&e),
        (Ok(Some(_)), Some(_)) => {
            return err_exit_code(
                ErrorCode::CliUsage,
                "cli_parse",
                "specify iterations either via --iters or positional [iters], not both",
            );
        }
        (Ok(Some(v)), None) => v,
        (Ok(None), Some(raw)) => match raw.parse::<usize>() {
            Ok(v) => v,
            Err(_) => {
                return err_exit_code(
                    ErrorCode::CliInvalidNumber,
                    "cli_parse",
                    &format!("invalid positional iters value '{raw}': expected non-negative integer"),
                );
            }
        },
        (Ok(None), None) => 11usize,
    };
    match run_cost_model(&loaded.spec, iters) {
        Ok(rep) => {
            if let Some(tf) = parsed.str_opt("--table")
                && let Err(e) = write_pretty_json(tf, &rep.per_op)
            {
                return err_exit_typed(&e);
            }
            print_json_exit(&rep, rep.ok)
        }
        Err(e) => err_exit_stage("cost_model", &e),
    }
}
