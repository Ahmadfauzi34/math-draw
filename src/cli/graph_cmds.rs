use serde_json::json;
use std::path::Path;
use std::process::ExitCode;

use super::{
    ParsedArgs, err_exit_code, err_exit_stage, err_exit_typed, err_exit_usage, print_json_exit,
    read_cases_file, read_targets_file, write_pretty_json,
};
use crate::direct_math::parse_f32_vector_str;
use crate::error::ErrorCode;
use crate::grad_spec::{GradSpecOptions, compile_grad_spec};
use crate::liveness::{auto_repair_spec, compact_slots_by_liveness};
use crate::mutate::{MutationRequest, mutate_spec};
use crate::quant::QuantScheme;
use crate::recurse::{
    EsStrategyKind, FitEsConfig, RngKind, SearchOptions, fit_es_with_config,
    search_graph_tree_with_options, solve_gd_autodiff,
};
use crate::rewrite::{REWRITE_RULES, apply_rewrite, apply_rewrites_recursive};
use crate::shard::shard_spec;
use crate::spec::{load_spec_file, sha256_json};
use crate::validate::validate_spec;

pub(super) fn handle_rewrite(rest: &[String]) -> ExitCode {
    let Some(sub) = rest.first().map(|s| s.as_str()) else {
        return err_exit_usage("rewrite", "usage: math-draw rewrite <list|apply> ...");
    };
    let sub_rest = &rest[1..];
    match sub {
        "list" => {
            let parsed = match ParsedArgs::parse(sub_rest, &[], &[]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(0, "usage: math-draw rewrite list") {
                return err_exit_typed(&e);
            }
            print_json_exit(&json!({ "ok": true, "rules": REWRITE_RULES }), true)
        }
        "apply" => {
            let usage = "usage: math-draw rewrite apply <rule|all> <spec.json> [--out <out.json>]";
            let parsed = match ParsedArgs::parse(sub_rest, &["--out"], &[]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(2, usage) {
                return err_exit_typed(&e);
            }
            if parsed.positionals.len() < 2 {
                return err_exit_usage("rewrite.apply", usage);
            }
            let rule = &parsed.positionals[0];
            let loaded = match load_spec_file(Path::new(&parsed.positionals[1])) {
                Ok(l) => l,
                Err(e) => return err_exit_stage("load_spec", &e),
            };
            let (rewritten, applied) = if rule == "all" || rule == "fixpoint" {
                let (s, app) = apply_rewrites_recursive(&loaded.spec, 32);
                if app.is_empty() {
                    return err_exit_code(
                        ErrorCode::RewriteFailed,
                        "rewrite.apply",
                        "fixpoint: no rewrite patterns matched",
                    );
                }
                (s, app)
            } else {
                match apply_rewrite(rule, &loaded.spec) {
                    Ok(s) => (s, vec![rule.clone()]),
                    Err(e) => return err_exit_stage("rewrite.apply", &e),
                }
            };
            if let Some(out_path) = parsed.str_opt("--out")
                && let Err(e) = write_pretty_json(out_path, &rewritten)
            {
                return err_exit_typed(&e);
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
        _ => err_exit_code(
            ErrorCode::CliUnknownCommand,
            "rewrite",
            &format!("unknown rewrite subcommand '{sub}'"),
        ),
    }
}

pub(super) fn handle_recurse(rest: &[String]) -> ExitCode {
    let Some(sub) = rest.first().map(|s| s.as_str()) else {
        return err_exit_usage(
            "recurse",
            "usage: math-draw recurse <expand|grad-spec|search|solve-gd|fit> ...",
        );
    };
    let sub_rest = &rest[1..];
    match sub {
        "expand" => {
            let usage = "usage: math-draw recurse expand <spec.v2.json> [--auto-repair] [--max-weights N] [--compact-slots] [--out <out.v1.json>]";
            let parsed = match ParsedArgs::parse(
                sub_rest,
                &["--out", "--max-weights"],
                &["--auto-repair", "--compact-slots"],
            ) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(1, usage) {
                return err_exit_typed(&e);
            }
            let Some(file) = parsed.positionals.first() else {
                return err_exit_usage("recurse.expand", usage);
            };
            let loaded = match load_spec_file(Path::new(file)) {
                Ok(l) => l,
                Err(e) => return err_exit_stage("load_spec", &e),
            };
            let max_weights = match parsed.parse_usize("--max-weights", 65535) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            let mut spec = loaded.spec;
            let mut remediation_receipt = Vec::new();
            if parsed.has("--auto-repair") {
                match auto_repair_spec(&spec, max_weights) {
                    Ok(rep) => {
                        spec = rep.spec;
                        remediation_receipt = rep.remediation_receipt;
                    }
                    Err(e) => return err_exit_stage("recurse.expand", &e),
                }
            }
            let mut slot_compaction = None;
            if parsed.has("--compact-slots") {
                match compact_slots_by_liveness(&spec) {
                    Ok(comp) => {
                        spec = comp.spec;
                        slot_compaction = Some(json!({
                            "slots_before": comp.slots_before,
                            "slots_after": comp.slots_after,
                            "slots_saved": comp.slots_saved,
                        }));
                    }
                    Err(e) => return err_exit_stage("recurse.expand", &e),
                }
            }
            let val_rep = validate_spec(&spec, false);
            if let Some(out_path) = parsed.str_opt("--out")
                && let Err(e) = write_pretty_json(out_path, &spec)
            {
                return err_exit_typed(&e);
            }
            print_json_exit(
                &json!({
                    "ok": val_rep.valid,
                    "was_v2": loaded.was_v2,
                    "validation": val_rep,
                    "expansion": loaded.expansion,
                    "remediation_receipt": remediation_receipt,
                    "slot_compaction": slot_compaction,
                    "spec": spec,
                }),
                val_rep.valid,
            )
        }
        "grad-spec" => {
            let usage = "usage: math-draw recurse grad-spec <spec.json> [--wrt <slot|name>] [--order 1|2] [--dout <v1,v2,...>] [--hvp-vec <v1,v2,...>] [--compact-slots] [--out <grad.json>]";
            let parsed = match ParsedArgs::parse(
                sub_rest,
                &["--wrt", "--order", "--dout", "--hvp-vec", "--out"],
                &["--compact-slots"],
            ) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(2, usage) {
                return err_exit_typed(&e);
            }
            let Some(file) = parsed.positionals.first() else {
                return err_exit_usage("recurse.grad_spec", usage);
            };
            let loaded = match load_spec_file(Path::new(file)) {
                Ok(l) => l,
                Err(e) => return err_exit_stage("load_spec", &e),
            };
            let wrt = parsed.str_opt("--wrt").unwrap_or("0").to_string();
            let order = match parsed.parse_usize("--order", 1) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            let dout = match parsed.str_opt("--dout") {
                Some(raw) => match parse_f32_vector_str(raw, "--dout") {
                    Ok((v, _)) => Some(v),
                    Err(e) => return err_exit_code(ErrorCode::CliInvalidNumber, "cli_parse", &e),
                },
                None => None,
            };
            let hvp_vec = match parsed.str_opt("--hvp-vec") {
                Some(raw) => match parse_f32_vector_str(raw, "--hvp-vec") {
                    Ok((v, _)) => Some(v),
                    Err(e) => return err_exit_code(ErrorCode::CliInvalidNumber, "cli_parse", &e),
                },
                None => None,
            };
            let out_file = match (parsed.str_opt("--out"), parsed.positionals.get(1)) {
                (Some(_), Some(_)) => {
                    return err_exit_code(
                        ErrorCode::CliUsage,
                        "cli_parse",
                        "specify output path either via --out or positional [grad.json], not both",
                    );
                }
                (Some(op), None) => Some(op.to_string()),
                (None, Some(op)) => Some(op.clone()),
                (None, None) => None,
            };
            let opts = GradSpecOptions {
                wrt,
                order,
                dout,
                hvp_vec,
                compact_slots: parsed.has("--compact-slots"),
            };
            match compile_grad_spec(&loaded.spec, loaded.expansion.as_ref(), &opts) {
                Ok(rep) => {
                    if let Some(op) = out_file
                        && let Err(e) = write_pretty_json(&op, &rep.spec)
                    {
                        return err_exit_typed(&e);
                    }
                    print_json_exit(&rep, rep.ok)
                }
                Err(e) => err_exit_stage("recurse.grad_spec", &e),
            }
        }
        "search" => {
            let usage = "usage: math-draw recurse search <spec.json> [--out <best.json>] [--max-depth 3] [--cases <cases.json>] [--abs-tol 1e-5] [--rel-tol 1e-5] [--quant-tol T] [--quant-scheme q8-per-block|int8|fp16]";
            let parsed = match ParsedArgs::parse(
                sub_rest,
                &[
                    "--out",
                    "--max-depth",
                    "--cases",
                    "--abs-tol",
                    "--rel-tol",
                    "--quant-tol",
                    "--quant-scheme",
                ],
                &[],
            ) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(2, usage) {
                return err_exit_typed(&e);
            }
            let Some(file) = parsed.positionals.first() else {
                return err_exit_usage("recurse.search", usage);
            };
            let loaded = match load_spec_file(Path::new(file)) {
                Ok(l) => l,
                Err(e) => return err_exit_stage("load_spec", &e),
            };
            let max_depth = match parsed.parse_usize("--max-depth", 3) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            let abs_tol = match parsed.parse_f64_non_neg("--abs-tol", 1e-5) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            let rel_tol = match parsed.parse_f64_non_neg("--rel-tol", 1e-5) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            let quant_tol = if parsed.has("--quant-tol") {
                match parsed.parse_f64_non_neg("--quant-tol", 0.0) {
                    Ok(v) => Some(v),
                    Err(e) => return err_exit_typed(&e),
                }
            } else {
                None
            };
            let quant_scheme = match QuantScheme::parse(
                parsed.str_opt("--quant-scheme").unwrap_or("q8-per-block"),
            ) {
                Ok(s) => s,
                Err(e) => return err_exit_code(ErrorCode::CliUsage, "recurse.search", &e),
            };
            let cases = if let Some(cf) = parsed.str_opt("--cases") {
                match read_cases_file(cf) {
                    Ok(c) => Some(c),
                    Err(e) => return err_exit_typed(&e),
                }
            } else {
                None
            };
            let out_file = match (parsed.str_opt("--out"), parsed.positionals.get(1)) {
                (Some(_), Some(_)) => {
                    return err_exit_code(
                        ErrorCode::CliUsage,
                        "cli_parse",
                        "specify output path either via --out or positional [out.json], not both",
                    );
                }
                (Some(op), None) => Some(op.to_string()),
                (None, Some(op)) => Some(op.clone()),
                (None, None) => None,
            };
            let opts = SearchOptions {
                max_depth,
                abs_tol,
                rel_tol,
                quant_tol,
                quant_scheme,
            };
            match search_graph_tree_with_options(&loaded.spec, cases.as_deref(), &opts) {
                Ok(res) => {
                    if let Some(op) = out_file
                        && let Err(e) = write_pretty_json(&op, &res.best_spec)
                    {
                        return err_exit_typed(&e);
                    }
                    print_json_exit(&res, res.ok)
                }
                Err(e) => err_exit_stage("recurse.search", &e),
            }
        }
        "solve-gd" => {
            let usage = "usage: math-draw recurse solve-gd <spec.json> [--targets <targets.json>] [--steps 20] [--lr 0.1] [--layer <idx|all>] [--precondition] [--out <out.json>]";
            let parsed = match ParsedArgs::parse(
                sub_rest,
                &["--targets", "--steps", "--lr", "--layer", "--out"],
                &["--precondition"],
            ) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(2, usage) {
                return err_exit_typed(&e);
            }
            let Some(file) = parsed.positionals.first() else {
                return err_exit_usage("recurse.solve_gd", usage);
            };
            let loaded = match load_spec_file(Path::new(file)) {
                Ok(l) => l,
                Err(e) => return err_exit_stage("load_spec", &e),
            };
            let target_path = match (parsed.str_opt("--targets"), parsed.positionals.get(1)) {
                (Some(_), Some(_)) => {
                    return err_exit_code(
                        ErrorCode::CliUsage,
                        "cli_parse",
                        "specify targets either via --targets or positional [targets.json], not both",
                    );
                }
                (Some(tf), None) => Some(tf),
                (None, Some(tf)) => Some(tf.as_str()),
                (None, None) => None,
            };
            let targets = if let Some(tf) = target_path {
                match read_targets_file(tf) {
                    Ok(t) => Some(t),
                    Err(e) => return err_exit_typed(&e),
                }
            } else {
                None
            };
            let steps = match parsed.parse_usize("--steps", 20) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            let lr = match parsed.parse_f64_pos("--lr", 0.1) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            let layer_filter = match parsed.str_opt("--layer") {
                None | Some("all") => None,
                Some(s) => match s.parse::<usize>() {
                    Ok(idx) => Some(idx),
                    Err(_) => {
                        return err_exit_code(
                            ErrorCode::CliInvalidNumber,
                            "cli_parse",
                            &format!("invalid --layer value '{s}'"),
                        );
                    }
                },
            };
            let precondition = parsed.has("--precondition");
            match solve_gd_autodiff(&loaded.spec, targets.as_ref(), layer_filter, steps, lr, precondition) {
                Ok(res) => {
                    if let Some(op) = parsed.str_opt("--out")
                        && let Err(e) = write_pretty_json(op, &res.optimized_spec)
                    {
                        return err_exit_typed(&e);
                    }
                    print_json_exit(&res, res.ok)
                }
                Err(e) => err_exit_stage("recurse.solve_gd", &e),
            }
        }
        "fit" => {
            let usage = "usage: math-draw recurse fit <spec.json> --targets <targets.json> [--strategy openes|mu-lambda] [--rng xorshift64|mulberry32|pcg32|philox4x32] [--gens 40] [--pop 16] [--mu N] [--sigma 0.2] [--lr 0.2] [--seed 42] [--sensitivity-guided] [--freeze-threshold 1e-6] [--out <out.json>]";
            let valued = [
                "--targets",
                "--gens",
                "--pop",
                "--mu",
                "--sigma",
                "--lr",
                "--seed",
                "--strategy",
                "--rng",
                "--freeze-threshold",
                "--out",
            ];
            let parsed = match ParsedArgs::parse(sub_rest, &valued, &["--sensitivity-guided"]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(2, usage) {
                return err_exit_typed(&e);
            }
            let Some(file) = parsed.positionals.first() else {
                return err_exit_usage("recurse.fit", usage);
            };
            let tf_path = match (parsed.str_opt("--targets"), parsed.positionals.get(1)) {
                (Some(_), Some(_)) => {
                    return err_exit_code(
                        ErrorCode::CliUsage,
                        "cli_parse",
                        "specify targets either via --targets or positional [targets.json], not both",
                    );
                }
                (Some(tf), None) => tf,
                (None, Some(tf)) => tf.as_str(),
                (None, None) => {
                    return err_exit_usage("recurse.fit", "fit requires --targets <targets.json>");
                }
            };
            let loaded = match load_spec_file(Path::new(file)) {
                Ok(l) => l,
                Err(e) => return err_exit_stage("load_spec", &e),
            };
            let targets = match read_targets_file(tf_path) {
                Ok(t) => t,
                Err(e) => return err_exit_typed(&e),
            };
            let strategy = match parsed.str_opt("--strategy") {
                Some(raw) => match EsStrategyKind::parse(raw) {
                    Ok(s) => s,
                    Err(e) => return err_exit_code(ErrorCode::EsConfigInvalid, "recurse.fit", &e),
                },
                None => EsStrategyKind::OpenEs,
            };
            let rng = match parsed.str_opt("--rng") {
                Some(raw) => match RngKind::parse(raw) {
                    Ok(r) => r,
                    Err(e) => return err_exit_code(ErrorCode::EsConfigInvalid, "recurse.fit", &e),
                },
                None => RngKind::Xorshift64,
            };
            let gens = match parsed.parse_usize("--gens", 40) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            let pop = match parsed.parse_usize("--pop", 16) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            let mu = match parsed.parse_usize_opt("--mu") {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            let sigma = match parsed.parse_f64_pos("--sigma", 0.2) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            let lr = match parsed.parse_f64_pos("--lr", 0.2) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            let seed = match parsed.parse_u64("--seed", 42) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            let sensitivity_guided = parsed.has("--sensitivity-guided");
            let freeze_threshold = match parsed.parse_f64_non_neg("--freeze-threshold", 1e-6) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            let cfg = FitEsConfig {
                generations: gens,
                pop,
                mu,
                sigma,
                lr,
                seed,
                strategy,
                rng,
                sensitivity_guided,
                freeze_threshold,
            };
            match fit_es_with_config(&loaded.spec, &targets, &cfg) {
                Ok(res) => {
                    if let Some(op) = parsed.str_opt("--out")
                        && let Err(e) = write_pretty_json(op, &res.optimized_spec)
                    {
                        return err_exit_typed(&e);
                    }
                    print_json_exit(&res, res.ok)
                }
                Err(e) => err_exit_stage("recurse.fit", &e),
            }
        }
        _ => err_exit_code(
            ErrorCode::CliUnknownCommand,
            "recurse",
            &format!("unknown recurse command '{sub}' (use: expand | search | solve-gd | fit)"),
        ),
    }
}

pub(super) fn handle_mutate(rest: &[String]) -> ExitCode {
    let usage = "usage: math-draw mutate <spec.json> <mutation.json> [--cases <cases.json>] [--abs-tol 1e-6] [--rel-tol 1e-6] [--out <out.json>]";
    let parsed = match ParsedArgs::parse(
        rest,
        &["--cases", "--abs-tol", "--rel-tol", "--out"],
        &[],
    ) {
        Ok(p) => p,
        Err(e) => return err_exit_typed(&e),
    };
    if let Err(e) = parsed.ensure_max_positionals(2, usage) {
        return err_exit_typed(&e);
    }
    if parsed.positionals.len() < 2 {
        return err_exit_usage("mutate", usage);
    }
    let base = match load_spec_file(Path::new(&parsed.positionals[0])) {
        Ok(l) => l.spec,
        Err(e) => return err_exit_stage("load_spec", &e),
    };
    let mut_raw = match std::fs::read_to_string(&parsed.positionals[1]) {
        Ok(r) => r,
        Err(e) => {
            return err_exit_code(
                ErrorCode::IoReadFailed,
                "mutate",
                &format!("cannot read mutation JSON '{}': {e}", parsed.positionals[1]),
            );
        }
    };
    let req: MutationRequest = match serde_json::from_str(&mut_raw) {
        Ok(r) => r,
        Err(e) => {
            return err_exit_code(
                ErrorCode::JsonParseFailed,
                "mutate",
                &format!("invalid mutation JSON '{}': {e}", parsed.positionals[1]),
            );
        }
    };
    let abs_tol = match parsed.parse_f64_non_neg("--abs-tol", 1e-6) {
        Ok(v) => v,
        Err(e) => return err_exit_typed(&e),
    };
    let rel_tol = match parsed.parse_f64_non_neg("--rel-tol", 1e-6) {
        Ok(v) => v,
        Err(e) => return err_exit_typed(&e),
    };
    let cases = if let Some(cf) = parsed.str_opt("--cases") {
        match read_cases_file(cf) {
            Ok(c) => Some(c),
            Err(e) => return err_exit_typed(&e),
        }
    } else {
        None
    };
    match mutate_spec(&base, &req, cases.as_deref(), abs_tol, rel_tol) {
        Ok(rep) => {
            if let Some(op) = parsed.str_opt("--out")
                && let Some(ref sp) = rep.resulting_spec
                && rep.ok
                && let Err(e) = write_pretty_json(op, sp)
            {
                return err_exit_typed(&e);
            }
            print_json_exit(&rep, rep.ok)
        }
        Err(e) => err_exit_stage("mutate", &e),
    }
}

pub(super) fn handle_shard(rest: &[String]) -> ExitCode {
    let usage = "usage: math-draw shard <spec.json> [--max-weights 65535] [--out <out.json>]";
    let parsed = match ParsedArgs::parse(rest, &["--max-weights", "--out"], &[]) {
        Ok(p) => p,
        Err(e) => return err_exit_typed(&e),
    };
    if let Err(e) = parsed.ensure_max_positionals(1, usage) {
        return err_exit_typed(&e);
    }
    let Some(file) = parsed.positionals.first() else {
        return err_exit_usage("shard", usage);
    };
    let loaded = match load_spec_file(Path::new(file)) {
        Ok(l) => l,
        Err(e) => return err_exit_stage("load_spec", &e),
    };
    let max_weights = match parsed.parse_usize("--max-weights", 65535) {
        Ok(v) => v,
        Err(e) => return err_exit_typed(&e),
    };
    match shard_spec(&loaded.spec, max_weights) {
        Ok(rep) => {
            if let Some(op) = parsed.str_opt("--out")
                && let Err(e) = write_pretty_json(op, &rep.spec)
            {
                return err_exit_typed(&e);
            }
            print_json_exit(&rep, rep.ok)
        }
        Err(e) => err_exit_stage("shard", &e),
    }
}
