use serde_json::json;
use std::path::Path;
use std::process::ExitCode;

use super::{
    ParsedArgs, err_exit_code, err_exit_stage, err_exit_typed, err_exit_usage, print_json_exit,
    read_cases_file,
};
use crate::catalog::{CATALOG, find_constructor};
use crate::engine::run_spec;
use crate::error::ErrorCode;
use crate::linear_equiv::equiv_linear;
use crate::smoke::run_smoke_suite;
use crate::spec::load_spec_file;
use crate::validate::validate_spec;
use crate::verify::{
    FuzzConfig, default_cases_from_spec, generate_fuzz_cases, verify_specs,
    zoom_fuzz_specs_with_tol,
};

pub(super) fn handle_run(rest: &[String]) -> ExitCode {
    let usage = "usage: math-draw run <spec.json>";
    let parsed = match ParsedArgs::parse(rest, &[], &[]) {
        Ok(p) => p,
        Err(e) => return err_exit_typed(&e),
    };
    if let Err(e) = parsed.ensure_max_positionals(1, usage) {
        return err_exit_typed(&e);
    }
    let Some(file) = parsed.positionals.first() else {
        return err_exit_usage("run", usage);
    };
    let loaded = match load_spec_file(Path::new(file)) {
        Ok(l) => l,
        Err(e) => return err_exit_stage("load_spec", &e),
    };
    match run_spec(&loaded.spec) {
        Ok(res) => print_json_exit(&res, true),
        Err(e) => {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&e) {
                print_json_exit(&v, false)
            } else {
                err_exit_stage("run", &e)
            }
        }
    }
}

pub(super) fn handle_validate(rest: &[String]) -> ExitCode {
    let usage = "usage: math-draw validate <spec.json> [--explain]";
    let parsed = match ParsedArgs::parse(rest, &[], &["--explain"]) {
        Ok(p) => p,
        Err(e) => return err_exit_typed(&e),
    };
    if let Err(e) = parsed.ensure_max_positionals(1, usage) {
        return err_exit_typed(&e);
    }
    let Some(file) = parsed.positionals.first() else {
        return err_exit_usage("validate", usage);
    };
    let loaded = match load_spec_file(Path::new(file)) {
        Ok(l) => l,
        Err(e) => return err_exit_stage("load_spec", &e),
    };
    let rep = validate_spec(&loaded.spec, parsed.has("--explain"));
    print_json_exit(&rep, rep.valid)
}

pub(super) fn handle_verify(rest: &[String]) -> ExitCode {
    let usage = "usage: math-draw verify <base.json> <candidate.json> [--cases <cases.json>] [--fuzz [N]] [--zoom] [--zoom-depth 4] [--seed S] [--lo A] [--hi B] [--abs-tol T] [--rel-tol T]";
    let valued = [
        "--cases",
        "--zoom-depth",
        "--seed",
        "--lo",
        "--hi",
        "--abs-tol",
        "--rel-tol",
    ];
    let parsed = match ParsedArgs::parse_full(rest, &valued, &["--fuzz"], &[], &["--zoom"]) {
        Ok(p) => p,
        Err(e) => return err_exit_typed(&e),
    };
    if let Err(e) = parsed.ensure_max_positionals(2, usage) {
        return err_exit_typed(&e);
    }
    if parsed.positionals.len() < 2 {
        return err_exit_usage("verify", usage);
    }
    if parsed.has("--zoom") && parsed.has("--cases") {
        return err_exit_code(
            ErrorCode::CliUsage,
            "cli_parse",
            "verify --zoom cannot be combined with --cases",
        );
    }
    let base = match load_spec_file(Path::new(&parsed.positionals[0])) {
        Ok(l) => l.spec,
        Err(e) => return err_exit_stage("load_spec", &e),
    };
    let cand = match load_spec_file(Path::new(&parsed.positionals[1])) {
        Ok(l) => l.spec,
        Err(e) => return err_exit_stage("load_spec", &e),
    };
    let abs_tol = match parsed.parse_f64_non_neg("--abs-tol", 1e-5) {
        Ok(v) => v,
        Err(e) => return err_exit_typed(&e),
    };
    let rel_tol = match parsed.parse_f64_non_neg("--rel-tol", 1e-5) {
        Ok(v) => v,
        Err(e) => return err_exit_typed(&e),
    };

    if parsed.has("--zoom") {
        let count = match parsed.parse_usize("--fuzz", 12) {
            Ok(v) => v,
            Err(e) => return err_exit_typed(&e),
        };
        let seed = match parsed.parse_u64("--seed", FuzzConfig::default().seed) {
            Ok(v) => v,
            Err(e) => return err_exit_typed(&e),
        };
        let lo = match parsed.parse_f64("--lo", -5.0) {
            Ok(v) => v,
            Err(e) => return err_exit_typed(&e),
        };
        let hi = match parsed.parse_f64("--hi", 5.0) {
            Ok(v) => v,
            Err(e) => return err_exit_typed(&e),
        };
        if lo > hi {
            return err_exit_code(
                ErrorCode::CliInvalidRange,
                "cli_parse",
                &format!("invalid fuzz range: --lo ({lo}) must be <= --hi ({hi})"),
            );
        }
        let zoom_depth = match parsed.parse_usize("--zoom-depth", 4) {
            Ok(v) => v,
            Err(e) => return err_exit_typed(&e),
        };
        let zoom_abs_tol = if parsed.has("--abs-tol") {
            Some(abs_tol)
        } else {
            None
        };
        let zoom_rel_tol = if parsed.has("--rel-tol") {
            Some(rel_tol)
        } else {
            None
        };
        match zoom_fuzz_specs_with_tol(
            &base,
            &cand,
            &FuzzConfig { count, seed, lo, hi },
            zoom_depth,
            zoom_abs_tol,
            zoom_rel_tol,
        ) {
            Ok(rep) => return print_json_exit(&rep, rep.ok),
            Err(e) => return err_exit_stage("verify", &e),
        }
    }

    let cases = if let Some(cf) = parsed.str_opt("--cases") {
        match read_cases_file(cf) {
            Ok(c) => c,
            Err(e) => return err_exit_typed(&e),
        }
    } else if parsed.has("--fuzz") {
        let count = match parsed.parse_usize("--fuzz", 8) {
            Ok(v) => v,
            Err(e) => return err_exit_typed(&e),
        };
        let seed = match parsed.parse_u64("--seed", FuzzConfig::default().seed) {
            Ok(v) => v,
            Err(e) => return err_exit_typed(&e),
        };
        let lo = match parsed.parse_f64("--lo", -2.0) {
            Ok(v) => v,
            Err(e) => return err_exit_typed(&e),
        };
        let hi = match parsed.parse_f64("--hi", 2.0) {
            Ok(v) => v,
            Err(e) => return err_exit_typed(&e),
        };
        if lo > hi {
            return err_exit_code(
                ErrorCode::CliInvalidRange,
                "cli_parse",
                &format!("invalid fuzz range: --lo ({lo}) must be <= --hi ({hi})"),
            );
        }
        generate_fuzz_cases(&base, &FuzzConfig { count, seed, lo, hi })
    } else {
        default_cases_from_spec(&base)
    };

    let rep = verify_specs(&base, &cand, &cases, abs_tol, rel_tol);
    print_json_exit(&rep, rep.equivalent)
}

pub(super) fn handle_linear_equiv(rest: &[String]) -> ExitCode {
    let usage = "usage: math-draw linear-equiv <specA.json> <specB.json> [--tol 0.0]";
    let parsed = match ParsedArgs::parse(rest, &["--tol"], &[]) {
        Ok(p) => p,
        Err(e) => return err_exit_typed(&e),
    };
    if let Err(e) = parsed.ensure_max_positionals(2, usage) {
        return err_exit_typed(&e);
    }
    if parsed.positionals.len() < 2 {
        return err_exit_usage("linear_equiv", usage);
    }
    let spec_a = match load_spec_file(Path::new(&parsed.positionals[0])) {
        Ok(l) => l.spec,
        Err(e) => return err_exit_stage("load_spec", &e),
    };
    let spec_b = match load_spec_file(Path::new(&parsed.positionals[1])) {
        Ok(l) => l.spec,
        Err(e) => return err_exit_stage("load_spec", &e),
    };
    let tol = match parsed.parse_f64_non_neg("--tol", 0.0) {
        Ok(v) => v,
        Err(e) => return err_exit_typed(&e),
    };
    let rep = equiv_linear(&spec_a, &spec_b, tol);
    print_json_exit(&rep, rep.ok)
}

pub(super) fn handle_find_op(rest: &[String]) -> ExitCode {
    let usage = "usage: math-draw find-op [--list | <constructor>]";
    let parsed = match ParsedArgs::parse(rest, &[], &["--list"]) {
        Ok(p) => p,
        Err(e) => return err_exit_typed(&e),
    };
    if parsed.has("--list") {
        if let Err(e) = parsed.ensure_max_positionals(0, usage) {
            return err_exit_typed(&e);
        }
    } else if let Err(e) = parsed.ensure_max_positionals(1, usage) {
        return err_exit_typed(&e);
    }

    if parsed.has("--list") || parsed.positionals.is_empty() {
        let list: Vec<_> = CATALOG
            .iter()
            .map(|c| {
                json!({
                    "name": c.name,
                    "family": c.family,
                    "arity": c.arity,
                    "signature": c.signature,
                    "expected_layout": c.expected_layout,
                })
            })
            .collect();
        print_json_exit(&json!({ "ok": true, "count": list.len(), "constructors": list }), true)
    } else {
        let query = &parsed.positionals[0];
        if let Some(meta) = find_constructor(query) {
            print_json_exit(&json!({ "ok": true, "constructor": meta }), true)
        } else {
            let q_low = query.to_lowercase();
            let matches: Vec<_> = CATALOG
                .iter()
                .filter(|c| {
                    c.name.to_lowercase().contains(&q_low)
                        || c.family.to_lowercase().contains(&q_low)
                })
                .collect();
            print_json_exit(
                &json!({ "ok": !matches.is_empty(), "query": query, "matches": matches }),
                !matches.is_empty(),
            )
        }
    }
}

pub(super) fn handle_smoke(rest: &[String]) -> ExitCode {
    let usage = "usage: math-draw smoke";
    let parsed = match ParsedArgs::parse(rest, &[], &[]) {
        Ok(p) => p,
        Err(e) => return err_exit_typed(&e),
    };
    if let Err(e) = parsed.ensure_max_positionals(0, usage) {
        return err_exit_typed(&e);
    }
    let rep = run_smoke_suite();
    print_json_exit(&rep, rep.ok)
}
