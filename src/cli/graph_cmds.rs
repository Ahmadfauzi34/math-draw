use serde_json::json;
use std::path::Path;
use std::process::ExitCode;

use super::{
    err_exit, flag_val, has_flag, positional_args, print_json_exit, read_cases_file,
    read_targets_file, write_pretty_json,
};
use crate::mutate::{MutationRequest, mutate_spec};
use crate::recurse::{fit_es, search_graph_tree, solve_gd_autodiff};
use crate::rewrite::{REWRITE_RULES, apply_rewrite, apply_rewrites_recursive};
use crate::shard::shard_spec;
use crate::spec::{load_spec_file, sha256_json};
use crate::validate::validate_spec;

pub(super) fn handle_rewrite(rest: &[String]) -> ExitCode {
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

pub(super) fn handle_recurse(rest: &[String]) -> ExitCode {
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
                return err_exit("usage: math-draw recurse solve-gd <spec.json> [--targets <targets.json>] [--steps 20] [--lr 0.1] [--layer <idx|all>] [--precondition] [--out <out.json>]");
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
            let precondition = has_flag(sub_rest, "--precondition");
            match solve_gd_autodiff(&loaded.spec, targets.as_ref(), layer_filter, steps, lr, precondition) {
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
            let valued = [
                "--targets",
                "--gens",
                "--pop",
                "--sigma",
                "--lr",
                "--seed",
                "--freeze-threshold",
                "--out",
            ];
            let pos = positional_args(sub_rest, &valued);
            let Some(file) = pos.first() else {
                return err_exit("usage: math-draw recurse fit <spec.json> --targets <targets.json> [--gens 40] [--pop 16] [--sigma 0.2] [--lr 0.2] [--seed 42] [--sensitivity-guided] [--freeze-threshold 1e-6] [--out <out.json>]");
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
            let sensitivity_guided = has_flag(sub_rest, "--sensitivity-guided");
            let freeze_threshold = flag_val(sub_rest, "--freeze-threshold")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1e-6f64);
            match fit_es(
                &loaded.spec,
                &targets,
                gens,
                pop,
                sigma,
                lr,
                seed,
                sensitivity_guided,
                freeze_threshold,
            ) {
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
        _ => err_exit(&format!(
            "unknown recurse command '{sub}' (use: expand | search | solve-gd | fit)"
        )),
    }
}

pub(super) fn handle_mutate(rest: &[String]) -> ExitCode {
    let valued = ["--cases", "--abs-tol", "--rel-tol", "--out"];
    let pos = positional_args(rest, &valued);
    if pos.len() < 2 {
        return err_exit("usage: math-draw mutate <spec.json> <mutation.json> [--cases <cases.json>] [--abs-tol 1e-6] [--rel-tol 1e-6] [--out <out.json>]");
    }
    let base = match load_spec_file(Path::new(&pos[0])) {
        Ok(l) => l.spec,
        Err(e) => return err_exit(&e),
    };
    let mut_raw = match std::fs::read_to_string(&pos[1]) {
        Ok(r) => r,
        Err(e) => return err_exit(&format!("cannot read mutation JSON '{}': {e}", pos[1])),
    };
    let req: MutationRequest = match serde_json::from_str(&mut_raw) {
        Ok(r) => r,
        Err(e) => return err_exit(&format!("invalid mutation JSON '{}': {e}", pos[1])),
    };
    let abs_tol = flag_val(rest, "--abs-tol")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-6f64);
    let rel_tol = flag_val(rest, "--rel-tol")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-6f64);
    let cases = if let Some(cf) = flag_val(rest, "--cases") {
        match read_cases_file(cf) {
            Ok(c) => Some(c),
            Err(e) => return err_exit(&e),
        }
    } else {
        None
    };
    match mutate_spec(&base, &req, cases.as_deref(), abs_tol, rel_tol) {
        Ok(rep) => {
            if let Some(op) = flag_val(rest, "--out")
                && let Some(ref sp) = rep.resulting_spec
                && rep.ok
                && let Err(e) = write_pretty_json(op, sp)
            {
                return err_exit(&e);
            }
            print_json_exit(&rep, rep.ok)
        }
        Err(e) => err_exit(&e),
    }
}

pub(super) fn handle_shard(rest: &[String]) -> ExitCode {
    let valued = ["--max-weights", "--out"];
    let pos = positional_args(rest, &valued);
    let Some(file) = pos.first() else {
        return err_exit("usage: math-draw shard <spec.json> [--max-weights 65535] [--out <out.json>]");
    };
    let loaded = match load_spec_file(Path::new(file)) {
        Ok(l) => l,
        Err(e) => return err_exit(&e),
    };
    let max_weights = flag_val(rest, "--max-weights")
        .and_then(|s| s.parse().ok())
        .unwrap_or(65535usize);
    match shard_spec(&loaded.spec, max_weights) {
        Ok(rep) => {
            if let Some(op) = flag_val(rest, "--out")
                && let Err(e) = write_pretty_json(op, &rep.spec)
            {
                return err_exit(&e);
            }
            print_json_exit(&rep, rep.ok)
        }
        Err(e) => err_exit(&e),
    }
}
