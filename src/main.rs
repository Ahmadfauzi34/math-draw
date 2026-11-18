use serde_json::json;
use std::path::Path;
use std::process::ExitCode;

use math_draw::catalog::{CATALOG, find_constructor};
use math_draw::engine::run_spec;
use math_draw::recurse::{fit_es, search_graph_tree, solve_gd_autodiff};
use math_draw::rewrite::{REWRITE_RULES, apply_rewrite, apply_rewrites_recursive};
use math_draw::smoke::run_smoke_suite;
use math_draw::spec::{load_spec_file, sha256_json};
use math_draw::validate::validate_spec;
use math_draw::verify::{
    EvalCase, FuzzConfig, TargetsFile, default_cases_from_spec, generate_fuzz_cases, verify_specs,
};

fn print_usage() {
    eprintln!(
        r#"math-draw — Standalone Rust/Burn 0.22.1 (burn-flex) Graph Engine & Recursive Engineering CLI

USAGE:
  math-draw run <spec.json>
  math-draw validate <spec.json> [--explain]
  math-draw verify <base.json> <candidate.json> [--cases <cases.json>] [--fuzz [N]] [--seed S] [--lo A] [--hi B] [--abs-tol T] [--rel-tol T]
  math-draw find-op [--list | <constructor>]
  math-draw rewrite list
  math-draw rewrite apply <rule|all> <spec.json> [--out <out.json>]
  math-draw recurse expand <spec.v2.json> [--out <out.v1.json>]
  math-draw recurse search <spec.json> [--out <best.json>] [--max-depth 3] [--cases <cases.json>] [--abs-tol 1e-5] [--rel-tol 1e-5]
  math-draw recurse solve-gd <spec.json> [--targets <targets.json>] [--steps 20] [--lr 0.1] [--layer <idx|all>] [--out <out.json>]
  math-draw recurse fit <spec.json> --targets <targets.json> [--gens 40] [--pop 16] [--sigma 0.2] [--lr 0.2] [--seed 42] [--out <out.json>]
  math-draw smoke"#
    );
}

fn flag_val<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].as_str())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn positional_args(args: &[String], valued_flags: &[&str]) -> Vec<String> {
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

fn print_json_exit<T: serde::Serialize>(val: &T, ok: bool) -> ExitCode {
    println!("{}", serde_json::to_string_pretty(val).unwrap());
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    }
}

fn err_exit(msg: &str) -> ExitCode {
    println!("{}", json!({ "ok": false, "error": msg }));
    ExitCode::from(2)
}

fn read_cases_file(path: &str) -> Result<Vec<EvalCase>, String> {
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

fn read_targets_file(path: &str) -> Result<TargetsFile, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read targets file '{path}': {e}"))?;
    let tf: TargetsFile =
        serde_json::from_str(&raw).map_err(|e| format!("invalid targets JSON in '{path}': {e}"))?;
    if tf.cases.is_empty() {
        return Err("targets.cases must be non-empty".into());
    }
    Ok(tf)
}

fn write_pretty_json<T: serde::Serialize>(path: &str, val: &T) -> Result<(), String> {
    let txt = serde_json::to_string_pretty(val).map_err(|e| e.to_string())?;
    std::fs::write(path, format!("{txt}\n"))
        .map_err(|e| format!("failed writing '{path}': {e}"))
}

fn main() -> ExitCode {
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
        "run" => {
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
        "validate" => {
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
        "verify" => {
            let valued = ["--cases", "--fuzz", "--seed", "--lo", "--hi", "--abs-tol", "--rel-tol"];
            let pos = positional_args(rest, &valued);
            if pos.len() < 2 {
                return err_exit("usage: math-draw verify <base.json> <candidate.json> [--cases <cases.json>] [--fuzz N]");
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
        "find-op" => {
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
        "rewrite" => {
            let Some(sub) = rest.first().map(|s| s.as_str()) else {
                return err_exit("usage: math-draw rewrite <list|apply> ...");
            };
            let sub_rest = &rest[1..];
            match sub {
                "list" => print_json_exit(&json!({ "ok": true, "rules": REWRITE_RULES }), true),
                "apply" => {
                    let pos = positional_args(sub_rest, &["--out"]);
                    if pos.len() < 2 {
                        return err_exit("usage: math-draw rewrite apply <rule|all> <spec.json> [--out <out.json>]");
                    }
                    let rule = &pos[0];
                    let loaded = match load_spec_file(Path::new(&pos[1])) {
                        Ok(l) => l,
                        Err(e) => return err_exit(&e),
                    };
                    let (rewritten, applied) = if rule == "all" || rule == "fixpoint" {
                        let (s, app) = apply_rewrites_recursive(&loaded.spec, 32);
                        if app.is_empty() {
                            return err_exit("fixpoint: no rewrite patterns matched");
                        }
                        (s, app)
                    } else {
                        match apply_rewrite(rule, &loaded.spec) {
                            Ok(s) => (s, vec![rule.clone()]),
                            Err(e) => return err_exit(&e),
                        }
                    };
                    if let Some(out_path) = flag_val(sub_rest, "--out")
                        && let Err(e) = write_pretty_json(out_path, &rewritten)
                    {
                        return err_exit(&e);
                    }
                    print_json_exit(
                        &json!({
                            "ok": true,
                            "applied_rewrites": applied,
                            "before_steps": loaded.spec.steps.len(),
                            "after_steps": rewritten.steps.len(),
                            "sha256": sha256_json(&rewritten),
                            "spec": rewritten,
                        }),
                        true,
                    )
                }
                _ => err_exit(&format!("unknown rewrite subcommand '{sub}'")),
            }
        }
        "recurse" => {
            let Some(sub) = rest.first().map(|s| s.as_str()) else {
                return err_exit("usage: math-draw recurse <expand|search|solve-gd|fit> ...");
            };
            let sub_rest = &rest[1..];
            match sub {
                "expand" => {
                    let pos = positional_args(sub_rest, &["--out"]);
                    let Some(file) = pos.first() else {
                        return err_exit("usage: math-draw recurse expand <spec.v2.json> [--out <out.v1.json>]");
                    };
                    let loaded = match load_spec_file(Path::new(file)) {
                        Ok(l) => l,
                        Err(e) => return err_exit(&e),
                    };
                    let val_rep = validate_spec(&loaded.spec, false);
                    if let Some(out_path) = flag_val(sub_rest, "--out")
                        && let Err(e) = write_pretty_json(out_path, &loaded.spec)
                    {
                        return err_exit(&e);
                    }
                    print_json_exit(
                        &json!({
                            "ok": val_rep.valid,
                            "was_v2": loaded.was_v2,
                            "validation": val_rep,
                            "expansion": loaded.expansion,
                            "spec": loaded.spec,
                        }),
                        val_rep.valid,
                    )
                }
                "search" => {
                    let valued = ["--out", "--max-depth", "--cases", "--abs-tol", "--rel-tol"];
                    let pos = positional_args(sub_rest, &valued);
                    let Some(file) = pos.first() else {
                        return err_exit("usage: math-draw recurse search <spec.json> [--out <best.json>] [--max-depth 3]");
                    };
                    let loaded = match load_spec_file(Path::new(file)) {
                        Ok(l) => l,
                        Err(e) => return err_exit(&e),
                    };
                    let max_depth = flag_val(sub_rest, "--max-depth")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(3usize);
                    let abs_tol = flag_val(sub_rest, "--abs-tol")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1e-5f64);
                    let rel_tol = flag_val(sub_rest, "--rel-tol")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1e-5f64);
                    let cases = if let Some(cf) = flag_val(sub_rest, "--cases") {
                        match read_cases_file(cf) {
                            Ok(c) => Some(c),
                            Err(e) => return err_exit(&e),
                        }
                    } else {
                        None
                    };
                    match search_graph_tree(&loaded.spec, max_depth, cases.as_deref(), abs_tol, rel_tol) {
                        Ok(res) => {
                            let out_file = flag_val(sub_rest, "--out")
                                .map(String::from)
                                .or_else(|| pos.get(1).cloned());
                            if let Some(op) = out_file
                                && let Err(e) = write_pretty_json(&op, &res.best_spec)
                            {
                                return err_exit(&e);
                            }
                            print_json_exit(&res, res.ok)
                        }
                        Err(e) => err_exit(&e),
                    }
                }
                "solve-gd" => {
                    let valued = ["--targets", "--steps", "--lr", "--layer", "--out"];
                    let pos = positional_args(sub_rest, &valued);
                    let Some(file) = pos.first() else {
                        return err_exit("usage: math-draw recurse solve-gd <spec.json> [--targets <targets.json>] [--steps 20] [--lr 0.1] [--layer <idx|all>] [--out <out.json>]");
                    };
                    let loaded = match load_spec_file(Path::new(file)) {
                        Ok(l) => l,
                        Err(e) => return err_exit(&e),
                    };
                    let targets = if let Some(tf) = flag_val(sub_rest, "--targets")
                        .or_else(|| pos.get(1).map(|s| s.as_str()))
                    {
                        match read_targets_file(tf) {
                            Ok(t) => Some(t),
                            Err(e) => return err_exit(&e),
                        }
                    } else {
                        None
                    };
                    let steps = flag_val(sub_rest, "--steps")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(20usize);
                    let lr = flag_val(sub_rest, "--lr")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.1f64);
                    let layer_filter = match flag_val(sub_rest, "--layer") {
                        None | Some("all") => None,
                        Some(s) => match s.parse::<usize>() {
                            Ok(idx) => Some(idx),
                            Err(_) => return err_exit(&format!("invalid --layer value '{s}'")),
                        },
                    };
                    match solve_gd_autodiff(&loaded.spec, targets.as_ref(), layer_filter, steps, lr) {
                        Ok(res) => {
                            if let Some(op) = flag_val(sub_rest, "--out")
                                && let Err(e) = write_pretty_json(op, &res.optimized_spec)
                            {
                                return err_exit(&e);
                            }
                            print_json_exit(&res, res.ok)
                        }
                        Err(e) => err_exit(&e),
                    }
                }
                "fit" => {
                    let valued = ["--targets", "--gens", "--pop", "--sigma", "--lr", "--seed", "--out"];
                    let pos = positional_args(sub_rest, &valued);
                    let Some(file) = pos.first() else {
                        return err_exit("usage: math-draw recurse fit <spec.json> --targets <targets.json> [--gens 40] [--pop 16] [--sigma 0.2] [--lr 0.2] [--seed 42] [--out <out.json>]");
                    };
                    let Some(tf_path) = flag_val(sub_rest, "--targets")
                        .or_else(|| pos.get(1).map(|s| s.as_str()))
                    else {
                        return err_exit("fit requires --targets <targets.json>");
                    };
                    let loaded = match load_spec_file(Path::new(file)) {
                        Ok(l) => l,
                        Err(e) => return err_exit(&e),
                    };
                    let targets = match read_targets_file(tf_path) {
                        Ok(t) => t,
                        Err(e) => return err_exit(&e),
                    };
                    let gens = flag_val(sub_rest, "--gens")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(40usize);
                    let pop = flag_val(sub_rest, "--pop")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(16usize);
                    let sigma = flag_val(sub_rest, "--sigma")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.2f64);
                    let lr = flag_val(sub_rest, "--lr")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.2f64);
                    let seed = flag_val(sub_rest, "--seed")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(42u64);
                    match fit_es(&loaded.spec, &targets, gens, pop, sigma, lr, seed) {
                        Ok(res) => {
                            if let Some(op) = flag_val(sub_rest, "--out")
                                && let Err(e) = write_pretty_json(op, &res.optimized_spec)
                            {
                                return err_exit(&e);
                            }
                            print_json_exit(&res, res.ok)
                        }
                        Err(e) => err_exit(&e),
                    }
                }
                _ => err_exit(&format!("unknown recurse command '{sub}' (use: expand | search | solve-gd | fit)")),
            }
        }
        "smoke" => {
            let rep = run_smoke_suite();
            print_json_exit(&rep, rep.ok)
        }
        _ => err_exit(&format!("unknown command '{cmd}' (run `math-draw --help`)")),
    }
}
