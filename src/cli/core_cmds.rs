use serde_json::json;
use std::path::Path;
use std::process::ExitCode;

use super::{
    err_exit, flag_val, has_flag, positional_args, print_json_exit, read_cases_file,
};
use crate::catalog::{CATALOG, find_constructor};
use crate::engine::run_spec;
use crate::linear_equiv::equiv_linear;
use crate::smoke::run_smoke_suite;
use crate::spec::load_spec_file;
use crate::validate::validate_spec;
use crate::verify::{
    FuzzConfig, default_cases_from_spec, generate_fuzz_cases, verify_specs, zoom_fuzz_specs,
};

pub(super) fn handle_run(rest: &[String]) -> ExitCode {
    let pos = positional_args(rest, &[]);
    let Some(file) = pos.first() else {
        return err_exit("usage: math-draw run <spec.json>");
    };
    let loaded = match load_spec_file(Path::new(file)) {
        Ok(l) => l,
        Err(e) => return err_exit(&e),
    };
    match run_spec(&loaded.spec) {
        Ok(res) => print_json_exit(&res, true),
        Err(e) => {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&e) {
                println!("{}", serde_json::to_string_pretty(&v).unwrap());
                ExitCode::from(2)
            } else {
                err_exit(&e)
            }
        }
    }
}

pub(super) fn handle_validate(rest: &[String]) -> ExitCode {
    let explain = has_flag(rest, "--explain");
    let pos = positional_args(rest, &[]);
    let Some(file) = pos.first() else {
        return err_exit("usage: math-draw validate <spec.json> [--explain]");
    };
    let loaded = match load_spec_file(Path::new(file)) {
        Ok(l) => l,
        Err(e) => return err_exit(&e),
    };
    let rep = validate_spec(&loaded.spec, explain);
    print_json_exit(&rep, rep.valid)
}

pub(super) fn handle_verify(rest: &[String]) -> ExitCode {
    let valued = [
        "--cases",
        "--fuzz",
        "--zoom-depth",
        "--seed",
        "--lo",
        "--hi",
        "--abs-tol",
        "--rel-tol",
    ];
    let pos = positional_args(rest, &valued);
    if pos.len() < 2 {
        return err_exit("usage: math-draw verify <base.json> <candidate.json> [--cases <cases.json>] [--fuzz N] [--zoom]");
    }
    let base = match load_spec_file(Path::new(&pos[0])) {
        Ok(l) => l.spec,
        Err(e) => return err_exit(&e),
    };
    let cand = match load_spec_file(Path::new(&pos[1])) {
        Ok(l) => l.spec,
        Err(e) => return err_exit(&e),
    };
    let abs_tol: f64 = flag_val(rest, "--abs-tol")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-5);
    let rel_tol: f64 = flag_val(rest, "--rel-tol")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-5);

    if has_flag(rest, "--zoom") {
        let count = flag_val(rest, "--fuzz")
            .and_then(|s| s.parse().ok())
            .unwrap_or(12usize);
        let seed = flag_val(rest, "--seed")
            .and_then(|s| s.parse().ok())
            .unwrap_or(FuzzConfig::default().seed);
        let lo = flag_val(rest, "--lo")
            .and_then(|s| s.parse().ok())
            .unwrap_or(-5.0);
        let hi = flag_val(rest, "--hi")
            .and_then(|s| s.parse().ok())
            .unwrap_or(5.0);
        let zoom_depth = flag_val(rest, "--zoom-depth")
            .and_then(|s| s.parse().ok())
            .unwrap_or(4usize);
        match zoom_fuzz_specs(&base, &cand, &FuzzConfig { count, seed, lo, hi }, zoom_depth) {
            Ok(rep) => return print_json_exit(&rep, rep.ok),
            Err(e) => return err_exit(&e),
        }
    }

    let cases = if let Some(cf) = flag_val(rest, "--cases") {
        match read_cases_file(cf) {
            Ok(c) => c,
            Err(e) => return err_exit(&e),
        }
    } else if has_flag(rest, "--fuzz") {
        let count = flag_val(rest, "--fuzz")
            .and_then(|s| s.parse().ok())
            .unwrap_or(8usize);
        let seed = flag_val(rest, "--seed")
            .and_then(|s| s.parse().ok())
            .unwrap_or(FuzzConfig::default().seed);
        let lo = flag_val(rest, "--lo")
            .and_then(|s| s.parse().ok())
            .unwrap_or(-2.0);
        let hi = flag_val(rest, "--hi")
            .and_then(|s| s.parse().ok())
            .unwrap_or(2.0);
        generate_fuzz_cases(&base, &FuzzConfig { count, seed, lo, hi })
    } else {
        default_cases_from_spec(&base)
    };

    let rep = verify_specs(&base, &cand, &cases, abs_tol, rel_tol);
    print_json_exit(&rep, rep.equivalent)
}

pub(super) fn handle_linear_equiv(rest: &[String]) -> ExitCode {
    let valued = ["--tol"];
    let pos = positional_args(rest, &valued);
    if pos.len() < 2 {
        return err_exit("usage: math-draw linear-equiv <specA.json> <specB.json> [--tol 0.0]");
    }
    let spec_a = match load_spec_file(Path::new(&pos[0])) {
        Ok(l) => l.spec,
        Err(e) => return err_exit(&e),
    };
    let spec_b = match load_spec_file(Path::new(&pos[1])) {
        Ok(l) => l.spec,
        Err(e) => return err_exit(&e),
    };
    let tol = flag_val(rest, "--tol")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0f64);
    let rep = equiv_linear(&spec_a, &spec_b, tol);
    print_json_exit(&rep, rep.ok)
}

pub(super) fn handle_find_op(rest: &[String]) -> ExitCode {
    if has_flag(rest, "--list") || rest.is_empty() {
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
        let query = &rest[0];
        if let Some(meta) = find_constructor(query) {
            print_json_exit(&json!({ "ok": true, "constructor": meta }), true)
        } else {
            let q_low = query.to_lowercase();
            let matches: Vec<_> = CATALOG
                .iter()
                .filter(|c| c.name.to_lowercase().contains(&q_low) || c.family.to_lowercase().contains(&q_low))
                .collect();
            print_json_exit(
                &json!({ "ok": !matches.is_empty(), "query": query, "matches": matches }),
                !matches.is_empty(),
            )
        }
    }
}

pub(super) fn handle_smoke() -> ExitCode {
    let rep = run_smoke_suite();
    print_json_exit(&rep, rep.ok)
}
