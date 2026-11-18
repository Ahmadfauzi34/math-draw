use std::path::Path;
use std::process::ExitCode;

use super::{err_exit, flag_val, has_flag, positional_args, print_json_exit, write_pretty_json};
use crate::cost_model::run_cost_model;
use crate::profile::{compare_specs_perf, profile_spec};
use crate::rollout::{
    parse_inputs_sequence, rollout_bisect, rollout_fixpoint, rollout_scan, rollout_unroll,
};
use crate::spec::load_spec_file;
use crate::trace::{DEFAULT_MAX_TENSOR_BYTES, compare_trace_hashes, trace_hash_spec};

pub(super) fn handle_trace_hash(rest: &[String]) -> ExitCode {
    let valued = ["--max-bytes"];
    let pos = positional_args(rest, &valued);
    let Some(first) = pos.first() else {
        return err_exit("usage: math-draw trace-hash <spec.json> [other.json] [--max-bytes N] [--isolate]");
    };
    let max_bytes = flag_val(rest, "--max-bytes")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_TENSOR_BYTES);
    let isolate = has_flag(rest, "--isolate");
    let left = match load_spec_file(Path::new(first)) {
        Ok(l) => l.spec,
        Err(e) => return err_exit(&e),
    };
    if let Some(second) = pos.get(1) {
        let right = match load_spec_file(Path::new(second)) {
            Ok(l) => l.spec,
            Err(e) => return err_exit(&e),
        };
        match compare_trace_hashes(&left, &right, max_bytes, isolate) {
            Ok(rep) => print_json_exit(&rep, rep.ok),
            Err(e) => err_exit(&e),
        }
    } else {
        match trace_hash_spec(&left, max_bytes) {
            Ok(rep) => print_json_exit(&rep, rep.ok),
            Err(e) => err_exit(&e),
        }
    }
}

pub(super) fn handle_rollout(rest: &[String]) -> ExitCode {
    let Some(sub) = rest.first().map(|s| s.as_str()) else {
        return err_exit("usage: math-draw rollout <scan|fixpoint|unroll|bisect> ...");
    };
    let sub_rest = &rest[1..];
    match sub {
        "scan" => {
            let valued = ["--steps", "--state-port", "--inputs-seq"];
            let pos = positional_args(sub_rest, &valued);
            let Some(file) = pos.first() else {
                return err_exit("usage: math-draw rollout scan <spec.json> [--steps N] [--state-port <slot|name>] [--inputs-seq <seq.json>]");
            };
            let loaded = match load_spec_file(Path::new(file)) {
                Ok(l) => l,
                Err(e) => return err_exit(&e),
            };
            let steps = flag_val(sub_rest, "--steps")
                .and_then(|s| s.parse().ok())
                .unwrap_or(8usize);
            let state_port = flag_val(sub_rest, "--state-port");
            let seq = if let Some(sp) = flag_val(sub_rest, "--inputs-seq") {
                let raw = match std::fs::read_to_string(sp) {
                    Ok(r) => r,
                    Err(e) => return err_exit(&format!("cannot read inputs-seq '{sp}': {e}")),
                };
                match parse_inputs_sequence(&raw, loaded.expansion.as_ref()) {
                    Ok(s) => Some(s),
                    Err(e) => return err_exit(&e),
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
                Err(e) => err_exit(&e),
            }
        }
        "fixpoint" => {
            let valued = ["--tol", "--max-steps", "--state-port"];
            let pos = positional_args(sub_rest, &valued);
            let Some(file) = pos.first() else {
                return err_exit("usage: math-draw rollout fixpoint <spec.json> [--tol 1e-6] [--max-steps 100] [--state-port <slot|name>]");
            };
            let loaded = match load_spec_file(Path::new(file)) {
                Ok(l) => l,
                Err(e) => return err_exit(&e),
            };
            let tol = flag_val(sub_rest, "--tol")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1e-6f64);
            let max_steps = flag_val(sub_rest, "--max-steps")
                .and_then(|s| s.parse().ok())
                .unwrap_or(100usize);
            let state_port = flag_val(sub_rest, "--state-port");
            match rollout_fixpoint(
                &loaded.spec,
                loaded.expansion.as_ref(),
                tol,
                max_steps,
                state_port,
            ) {
                Ok(rep) => print_json_exit(&rep, rep.converged),
                Err(e) => err_exit(&e),
            }
        }
        "unroll" => {
            let valued = ["--steps", "--state-port", "--out"];
            let pos = positional_args(sub_rest, &valued);
            let Some(file) = pos.first() else {
                return err_exit("usage: math-draw rollout unroll <spec.json> [--steps N] [--state-port <slot|name>] [--out <out.json>]");
            };
            let loaded = match load_spec_file(Path::new(file)) {
                Ok(l) => l,
                Err(e) => return err_exit(&e),
            };
            let steps = flag_val(sub_rest, "--steps")
                .and_then(|s| s.parse().ok())
                .unwrap_or(4usize);
            let state_port = flag_val(sub_rest, "--state-port");
            match rollout_unroll(&loaded.spec, loaded.expansion.as_ref(), steps, state_port) {
                Ok(rep) => {
                    if let Some(op) = flag_val(sub_rest, "--out")
                        && let Err(e) = write_pretty_json(op, &rep.unrolled_spec)
                    {
                        return err_exit(&e);
                    }
                    print_json_exit(&rep, rep.ok)
                }
                Err(e) => err_exit(&e),
            }
        }
        "bisect" => {
            let valued = ["--steps", "--state-port", "--inputs-seq", "--abs-tol"];
            let pos = positional_args(sub_rest, &valued);
            if pos.len() < 2 {
                return err_exit("usage: math-draw rollout bisect <base.json> <cand.json> [--steps N] [--state-port <slot|name>] [--abs-tol 1e-5]");
            }
            let base = match load_spec_file(Path::new(&pos[0])) {
                Ok(l) => l,
                Err(e) => return err_exit(&e),
            };
            let cand = match load_spec_file(Path::new(&pos[1])) {
                Ok(l) => l,
                Err(e) => return err_exit(&e),
            };
            let steps = flag_val(sub_rest, "--steps")
                .and_then(|s| s.parse().ok())
                .unwrap_or(8usize);
            let state_port = flag_val(sub_rest, "--state-port");
            let abs_tol = flag_val(sub_rest, "--abs-tol")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1e-5f64);
            let seq = if let Some(sp) = flag_val(sub_rest, "--inputs-seq") {
                let raw = match std::fs::read_to_string(sp) {
                    Ok(r) => r,
                    Err(e) => return err_exit(&format!("cannot read inputs-seq '{sp}': {e}")),
                };
                match parse_inputs_sequence(&raw, base.expansion.as_ref()) {
                    Ok(s) => Some(s),
                    Err(e) => return err_exit(&e),
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
                Err(e) => err_exit(&e),
            }
        }
        _ => err_exit(&format!(
            "unknown rollout subcommand '{sub}' (use: scan | fixpoint | unroll | bisect)"
        )),
    }
}

pub(super) fn handle_profile(rest: &[String]) -> ExitCode {
    let valued = ["--iters", "--warmup"];
    let pos = positional_args(rest, &valued);
    let Some(file) = pos.first() else {
        return err_exit("usage: math-draw profile <spec.json> [--iters 30] [--warmup 5]");
    };
    let loaded = match load_spec_file(Path::new(file)) {
        Ok(l) => l,
        Err(e) => return err_exit(&e),
    };
    let iters = flag_val(rest, "--iters")
        .and_then(|s| s.parse().ok())
        .unwrap_or(30usize);
    let warmup = flag_val(rest, "--warmup")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5usize);
    match profile_spec(&loaded.spec, warmup, iters) {
        Ok(rep) => print_json_exit(&rep, rep.ok),
        Err(e) => err_exit(&e),
    }
}

pub(super) fn handle_compare(rest: &[String]) -> ExitCode {
    let valued = ["--iters", "--warmup"];
    let pos = positional_args(rest, &valued);
    if pos.len() < 2 {
        return err_exit("usage: math-draw compare <base.json> <cand.json> [--iters 30] [--warmup 5]");
    }
    let base = match load_spec_file(Path::new(&pos[0])) {
        Ok(l) => l.spec,
        Err(e) => return err_exit(&e),
    };
    let cand = match load_spec_file(Path::new(&pos[1])) {
        Ok(l) => l.spec,
        Err(e) => return err_exit(&e),
    };
    let iters = flag_val(rest, "--iters")
        .and_then(|s| s.parse().ok())
        .unwrap_or(30usize);
    let warmup = flag_val(rest, "--warmup")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5usize);
    match compare_specs_perf(&base, &cand, warmup, iters) {
        Ok(rep) => print_json_exit(&rep, rep.ok),
        Err(e) => err_exit(&e),
    }
}

pub(super) fn handle_cost_model(rest: &[String]) -> ExitCode {
    let valued = ["--iters", "--table"];
    let pos = positional_args(rest, &valued);
    let Some(file) = pos.first() else {
        return err_exit("usage: math-draw cost-model <spec.json> [--iters 11] [--table <out.json>]");
    };
    let loaded = match load_spec_file(Path::new(file)) {
        Ok(l) => l,
        Err(e) => return err_exit(&e),
    };
    let iters = flag_val(rest, "--iters")
        .and_then(|s| s.parse().ok())
        .or_else(|| pos.get(1).and_then(|s| s.parse().ok()))
        .unwrap_or(11usize);
    match run_cost_model(&loaded.spec, iters) {
        Ok(rep) => {
            if let Some(tf) = flag_val(rest, "--table")
                && let Err(e) = write_pretty_json(tf, &rep.per_op)
            {
                return err_exit(&e);
            }
            print_json_exit(&rep, rep.ok)
        }
        Err(e) => err_exit(&e),
    }
}
