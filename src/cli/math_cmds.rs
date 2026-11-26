use std::path::Path;
use std::process::ExitCode;

use super::{
    ParsedArgs, err_exit_code, err_exit_stage, err_exit_typed, err_exit_usage, print_json_exit,
    write_pretty_json,
};
use crate::direct_math::{
    DirectBatchKind, eval_arith_op, eval_direct_batch, eval_int_op, eval_kernel_op,
};
use crate::error::ErrorCode;
use crate::recurse::RngKind;
use crate::soak::{certify_prng_stream, soak_spec, verify_prng_manifest};
use crate::spec::load_spec_file;

fn run_batch_file(kind: DirectBatchKind, stage: &str, path: &str) -> ExitCode {
    let raw = match std::fs::read_to_string(path) {
        Ok(r) => r,
        Err(e) => {
            return err_exit_code(
                ErrorCode::IoReadFailed,
                stage,
                &format!("cannot read batch file '{path}': {e}"),
            );
        }
    };
    let doc: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            return err_exit_code(
                ErrorCode::JsonParseFailed,
                stage,
                &format!("invalid batch JSON '{path}': {e}"),
            );
        }
    };
    match eval_direct_batch(kind, &doc) {
        Ok(val) => print_json_exit(&val, true),
        Err(e) => err_exit_stage(stage, &e),
    }
}

pub(super) fn handle_arith(rest: &[String]) -> ExitCode {
    let usage = "usage: math-draw arith <op> <a> [b] | math-draw arith --batch <ops.json>";
    let parsed = match ParsedArgs::parse(rest, &["--batch"], &[]) {
        Ok(p) => p,
        Err(e) => return err_exit_typed(&e),
    };
    if let Some(bf) = parsed.str_opt("--batch") {
        if let Err(e) = parsed.ensure_max_positionals(0, usage) {
            return err_exit_typed(&e);
        }
        return run_batch_file(DirectBatchKind::Arith, "arith.batch", bf);
    }
    if let Err(e) = parsed.ensure_max_positionals(3, usage) {
        return err_exit_typed(&e);
    }
    if parsed.positionals.len() < 2 {
        return err_exit_usage("arith", usage);
    }
    let op = &parsed.positionals[0];
    let a = &parsed.positionals[1];
    let b = parsed.positionals.get(2).map(|s| s.as_str());
    match eval_arith_op(op, a, b) {
        Ok(val) => print_json_exit(&val, true),
        Err(e) => err_exit_stage("arith", &e),
    }
}

pub(super) fn handle_int(rest: &[String]) -> ExitCode {
    let usage = "usage: math-draw int <add|sub|mul|div|rem|matmul|to-f32|from-f32> <a> [b] [--wrap] [--allow-inexact] | math-draw int --batch <ops.json>";
    let parsed = match ParsedArgs::parse(rest, &["--batch"], &["--wrap", "--allow-inexact"]) {
        Ok(p) => p,
        Err(e) => return err_exit_typed(&e),
    };
    if let Some(bf) = parsed.str_opt("--batch") {
        if let Err(e) = parsed.ensure_max_positionals(0, usage) {
            return err_exit_typed(&e);
        }
        if parsed.has("--wrap") || parsed.has("--allow-inexact") {
            return err_exit_code(
                ErrorCode::CliUsage,
                "int.batch",
                "--wrap and --allow-inexact cannot be combined with --batch (set per-op 'wrap'/'allow_inexact' booleans inside batch JSON)",
            );
        }
        return run_batch_file(DirectBatchKind::Int, "int.batch", bf);
    }
    if let Err(e) = parsed.ensure_max_positionals(3, usage) {
        return err_exit_typed(&e);
    }
    if parsed.positionals.len() < 2 {
        return err_exit_usage("int", usage);
    }
    let op = &parsed.positionals[0];
    let a = &parsed.positionals[1];
    let b = parsed.positionals.get(2).map(|s| s.as_str());
    match eval_int_op(op, a, b, parsed.has("--wrap"), parsed.has("--allow-inexact")) {
        Ok(val) => print_json_exit(&val, true),
        Err(e) => err_exit_stage("int", &e),
    }
}

pub(super) fn handle_kernel(rest: &[String]) -> ExitCode {
    let usage = "usage: math-draw kernel <sin|cos|tan|floor|ceil|round|atan2|powf|hypot|fma> <a> [b] [c] | math-draw kernel --batch <ops.json>";
    let parsed = match ParsedArgs::parse(rest, &["--batch"], &[]) {
        Ok(p) => p,
        Err(e) => return err_exit_typed(&e),
    };
    if let Some(bf) = parsed.str_opt("--batch") {
        if let Err(e) = parsed.ensure_max_positionals(0, usage) {
            return err_exit_typed(&e);
        }
        return run_batch_file(DirectBatchKind::Kernel, "kernel.batch", bf);
    }
    if let Err(e) = parsed.ensure_max_positionals(4, usage) {
        return err_exit_typed(&e);
    }
    if parsed.positionals.len() < 2 {
        return err_exit_usage("kernel", usage);
    }
    let op = &parsed.positionals[0];
    let a = &parsed.positionals[1];
    let b = parsed.positionals.get(2).map(|s| s.as_str());
    let c = parsed.positionals.get(3).map(|s| s.as_str());
    match eval_kernel_op(op, a, b, c) {
        Ok(val) => print_json_exit(&val, true),
        Err(e) => err_exit_stage("kernel", &e),
    }
}

pub(super) fn handle_soak(rest: &[String]) -> ExitCode {
    let usage = "usage: math-draw soak <spec.json> [--runs 100]";
    let parsed = match ParsedArgs::parse(rest, &["--runs"], &[]) {
        Ok(p) => p,
        Err(e) => return err_exit_typed(&e),
    };
    if let Err(e) = parsed.ensure_max_positionals(2, usage) {
        return err_exit_typed(&e);
    }
    let Some(file) = parsed.positionals.first() else {
        return err_exit_usage("soak", usage);
    };
    let runs = if parsed.has("--runs") {
        if parsed.positionals.len() > 1 {
            return err_exit_code(
                ErrorCode::CliUsage,
                "cli_parse",
                "specify run count either via --runs or positional [N], not both",
            );
        }
        match parsed.parse_usize("--runs", 100) {
            Ok(v) => v,
            Err(e) => return err_exit_typed(&e),
        }
    } else if let Some(pos_n) = parsed.positionals.get(1) {
        match pos_n.parse::<usize>() {
            Ok(v) => v,
            Err(_) => {
                return err_exit_code(
                    ErrorCode::CliInvalidNumber,
                    "cli_parse",
                    &format!("invalid runs count '{pos_n}': expected integer >= 2"),
                );
            }
        }
    } else {
        100
    };
    let loaded = match load_spec_file(Path::new(file)) {
        Ok(l) => l,
        Err(e) => return err_exit_stage("load_spec", &e),
    };
    match soak_spec(&loaded.spec, runs) {
        Ok(rep) => print_json_exit(&rep, rep.identical),
        Err(e) => err_exit_stage("soak", &e),
    }
}

pub(super) fn handle_prng_check(rest: &[String]) -> ExitCode {
    if rest.first().map(|s| s.as_str()) == Some("verify") {
        let usage = "usage: math-draw prng-check verify <manifest.json>";
        let parsed = match ParsedArgs::parse(&rest[1..], &[], &[]) {
            Ok(p) => p,
            Err(e) => return err_exit_typed(&e),
        };
        if let Err(e) = parsed.ensure_max_positionals(1, usage) {
            return err_exit_typed(&e);
        }
        let Some(mf) = parsed.positionals.first() else {
            return err_exit_usage("prng_check.verify", usage);
        };
        let raw = match std::fs::read_to_string(mf) {
            Ok(r) => r,
            Err(e) => {
                return err_exit_code(
                    ErrorCode::IoReadFailed,
                    "prng_check.verify",
                    &format!("cannot read PRNG manifest '{mf}': {e}"),
                );
            }
        };
        let val: serde_json::Value = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(e) => {
                return err_exit_code(
                    ErrorCode::JsonParseFailed,
                    "prng_check.verify",
                    &format!("invalid PRNG manifest JSON '{mf}': {e}"),
                );
            }
        };
        let (Some(algo), Some(seed), Some(count), Some(sha)) = (
            val.get("algorithm").and_then(|v| v.as_str()),
            val.get("seed").and_then(|v| v.as_u64()),
            val.get("count").and_then(|v| v.as_u64()).map(|n| n as usize),
            val.get("sha256").and_then(|v| v.as_str()),
        ) else {
            return err_exit_code(
                ErrorCode::JsonParseFailed,
                "prng_check.verify",
                "manifest must contain {algorithm, seed, count, sha256}",
            );
        };
        return match verify_prng_manifest(algo, seed, count, sha) {
            Ok(rep) => print_json_exit(&rep, rep.verified),
            Err(e) => err_exit_stage("prng_check.verify", &e),
        };
    }

    let usage = "usage: math-draw prng-check [mulberry32|xorshift64|pcg32|philox4x32] [seed=1] [count=1000] [--out <manifest.json>] | math-draw prng-check verify <manifest.json>";
    let parsed = match ParsedArgs::parse(rest, &["--algo", "--seed", "--count", "--out"], &[]) {
        Ok(p) => p,
        Err(e) => return err_exit_typed(&e),
    };
    if let Err(e) = parsed.ensure_max_positionals(3, usage) {
        return err_exit_typed(&e);
    }
    let algo_s = parsed
        .str_opt("--algo")
        .or_else(|| parsed.positionals.first().map(|s| s.as_str()))
        .unwrap_or("mulberry32");
    let kind = match RngKind::parse(algo_s) {
        Ok(k) => k,
        Err(e) => return err_exit_code(ErrorCode::CliUsage, "prng_check", &e),
    };
    let seed = if parsed.has("--seed") {
        match parsed.parse_u64("--seed", 1) {
            Ok(v) => v,
            Err(e) => return err_exit_typed(&e),
        }
    } else if let Some(s_pos) = parsed.positionals.get(1) {
        match s_pos.parse::<u64>() {
            Ok(v) => v,
            Err(_) => {
                return err_exit_code(
                    ErrorCode::CliInvalidNumber,
                    "cli_parse",
                    &format!("invalid seed '{s_pos}': expected unsigned 64-bit integer"),
                );
            }
        }
    } else {
        1
    };
    let count = if parsed.has("--count") {
        match parsed.parse_usize("--count", 1000) {
            Ok(v) => v,
            Err(e) => return err_exit_typed(&e),
        }
    } else if let Some(c_pos) = parsed.positionals.get(2) {
        match c_pos.parse::<usize>() {
            Ok(v) => v,
            Err(_) => {
                return err_exit_code(
                    ErrorCode::CliInvalidNumber,
                    "cli_parse",
                    &format!("invalid count '{c_pos}': expected integer >= 1"),
                );
            }
        }
    } else {
        1000
    };

    match certify_prng_stream(kind, seed, count) {
        Ok(manifest) => {
            if let Some(op) = parsed.str_opt("--out")
                && let Err(e) = write_pretty_json(op, &manifest)
            {
                return err_exit_typed(&e);
            }
            print_json_exit(&manifest, manifest.ok)
        }
        Err(e) => err_exit_stage("prng_check", &e),
    }
}
