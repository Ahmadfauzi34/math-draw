use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;

use super::{
    ParsedArgs, err_exit_code, err_exit_stage, err_exit_typed, err_exit_usage, print_json_exit,
    read_cases_file, write_pretty_json,
};
use crate::bundle::{ProgramBundle, compute_parameter_binding, export_bundle, verify_bundle};
use crate::diff_ref::diff_ref_spec;
use crate::error::ErrorCode;
use crate::gradcheck::gradcheck_spec;
use crate::quant::{BudgetMode, QuantScheme, quant_budget, quant_verify};
use crate::spec::load_spec_file;
use crate::weight_lab::{
    WeightSet, certify_weight_set, compose_properties, engine_probe, evaluate_weight_expr,
    gate_sets, mat_vec_mul, run_weight_lab_selftest,
};

pub(super) fn handle_gradcheck(rest: &[String]) -> ExitCode {
    let usage = "usage: math-draw gradcheck <spec.json> [analytic.json] [--eps 1e-3] [--tol 1e-3] [--weights] [--autodiff]";
    let parsed = match ParsedArgs::parse(rest, &["--eps", "--tol"], &["--weights", "--autodiff"]) {
        Ok(p) => p,
        Err(e) => return err_exit_typed(&e),
    };
    if let Err(e) = parsed.ensure_max_positionals(2, usage) {
        return err_exit_typed(&e);
    }
    let Some(spec_file) = parsed.positionals.first() else {
        return err_exit_usage("gradcheck", usage);
    };
    let loaded = match load_spec_file(Path::new(spec_file)) {
        Ok(l) => l,
        Err(e) => return err_exit_stage("load_spec", &e),
    };
    let analytic = if let Some(af) = parsed.positionals.get(1) {
        let raw = match std::fs::read_to_string(af) {
            Ok(r) => r,
            Err(e) => {
                return err_exit_code(
                    ErrorCode::IoReadFailed,
                    "gradcheck",
                    &format!("cannot read analytic JSON '{af}': {e}"),
                );
            }
        };
        match serde_json::from_str::<serde_json::Value>(&raw) {
            Ok(v) => Some(v),
            Err(e) => {
                return err_exit_code(
                    ErrorCode::JsonParseFailed,
                    "gradcheck",
                    &format!("invalid analytic JSON '{af}': {e}"),
                );
            }
        }
    } else {
        None
    };
    let eps = match parsed.parse_f64_pos("--eps", 1e-3) {
        Ok(v) => v,
        Err(e) => return err_exit_typed(&e),
    };
    let tol = match parsed.parse_f64_non_neg("--tol", 1e-3) {
        Ok(v) => v,
        Err(e) => return err_exit_typed(&e),
    };
    let include_weights = parsed.has("--weights");
    let check_autodiff = parsed.has("--autodiff");
    match gradcheck_spec(
        &loaded.spec,
        analytic.as_ref(),
        eps,
        tol,
        include_weights,
        check_autodiff,
    ) {
        Ok(rep) => print_json_exit(&rep, rep.ok),
        Err(e) => err_exit_stage("gradcheck", &e),
    }
}

pub(super) fn handle_quant(rest: &[String]) -> ExitCode {
    let Some(sub) = rest.first().map(|s| s.as_str()) else {
        return err_exit_usage("quant", "usage: math-draw quant <verify|budget> ...");
    };
    let sub_rest = &rest[1..];
    match sub {
        "verify" => {
            let usage = "usage: math-draw quant verify <spec.json> [--scheme int8|q8-per-block|fp16] [--cases <cases.json>] [--abs-tol 1e-2] [--rel-tol 1e-2] [--out <quant.json>]";
            let parsed = match ParsedArgs::parse(
                sub_rest,
                &["--scheme", "--cases", "--abs-tol", "--rel-tol", "--out"],
                &[],
            ) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(1, usage) {
                return err_exit_typed(&e);
            }
            let Some(file) = parsed.positionals.first() else {
                return err_exit_usage("quant.verify", usage);
            };
            let loaded = match load_spec_file(Path::new(file)) {
                Ok(l) => l,
                Err(e) => return err_exit_stage("load_spec", &e),
            };
            let scheme = match QuantScheme::parse(parsed.str_opt("--scheme").unwrap_or("int8")) {
                Ok(s) => s,
                Err(e) => return err_exit_code(ErrorCode::CliUsage, "quant.verify", &e),
            };
            let abs_tol = match parsed.parse_f64_non_neg("--abs-tol", 1e-2) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            let rel_tol = match parsed.parse_f64_non_neg("--rel-tol", 1e-2) {
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
            match quant_verify(&loaded.spec, scheme, cases.as_deref(), abs_tol, rel_tol) {
                Ok(rep) => {
                    if let Some(op) = parsed.str_opt("--out")
                        && let Err(e) = write_pretty_json(op, &rep.quantized_spec)
                    {
                        return err_exit_typed(&e);
                    }
                    print_json_exit(&rep, rep.ok)
                }
                Err(e) => err_exit_stage("quant.verify", &e),
            }
        }
        "budget" => {
            let usage = "usage: math-draw quant budget <spec.json> [--cases <cases.json>] [--tol 0.05] [--scheme q8-per-block|int8|fp16] [--mode bound|measured] [--out <plan.json>]";
            let parsed = match ParsedArgs::parse(
                sub_rest,
                &["--cases", "--tol", "--scheme", "--mode", "--out"],
                &[],
            ) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(1, usage) {
                return err_exit_typed(&e);
            }
            let Some(file) = parsed.positionals.first() else {
                return err_exit_usage("quant.budget", usage);
            };
            let loaded = match load_spec_file(Path::new(file)) {
                Ok(l) => l,
                Err(e) => return err_exit_stage("load_spec", &e),
            };
            let scheme =
                match QuantScheme::parse(parsed.str_opt("--scheme").unwrap_or("q8-per-block")) {
                    Ok(s) => s,
                    Err(e) => return err_exit_code(ErrorCode::CliUsage, "quant.budget", &e),
                };
            let mode = match BudgetMode::parse(parsed.str_opt("--mode").unwrap_or("bound")) {
                Ok(m) => m,
                Err(e) => return err_exit_code(ErrorCode::CliUsage, "quant.budget", &e),
            };
            let tol = match parsed.parse_f64_non_neg("--tol", 0.05) {
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
            match quant_budget(&loaded.spec, cases.as_deref(), tol, scheme, mode) {
                Ok(rep) => {
                    if let Some(op) = parsed.str_opt("--out")
                        && let Err(e) = write_pretty_json(op, &rep.planned_spec)
                    {
                        return err_exit_typed(&e);
                    }
                    print_json_exit(&rep, rep.ok)
                }
                Err(e) => err_exit_stage("quant.budget", &e),
            }
        }
        _ => err_exit_code(
            ErrorCode::CliUnknownCommand,
            "quant",
            &format!("unknown quant subcommand '{sub}' (use: verify | budget)"),
        ),
    }
}

pub(super) fn handle_weight_lab(rest: &[String]) -> ExitCode {
    if rest.first().map(|s| s.as_str()) == Some("--selftest")
        || rest.first().map(|s| s.as_str()) == Some("selftest")
    {
        let parsed = match ParsedArgs::parse(&rest[1..], &[], &[]) {
            Ok(p) => p,
            Err(e) => return err_exit_typed(&e),
        };
        if let Err(e) = parsed.ensure_max_positionals(0, "usage: math-draw weight-lab selftest") {
            return err_exit_typed(&e);
        }
        return match run_weight_lab_selftest() {
            Ok(v) => {
                let ok = v.get("ok").and_then(|b| b.as_bool()).unwrap_or(false);
                print_json_exit(&v, ok)
            }
            Err(e) => err_exit_stage("weight_lab.selftest", &e),
        };
    }
    let Some(sub) = rest.first().map(|s| s.as_str()) else {
        return err_exit_usage(
            "weight_lab",
            "usage: math-draw weight-lab <run|gate|certify|compose|eval|selftest> ...",
        );
    };
    let sub_rest = &rest[1..];
    let load_ws = |p: &str| -> Result<WeightSet, String> {
        let raw = std::fs::read_to_string(p)
            .map_err(|e| format!("cannot read weight set '{p}': {e}"))?;
        let ws: WeightSet = serde_json::from_str(&raw)
            .map_err(|e| format!("invalid weight set '{p}': {e}"))?;
        ws.validate_dims()?;
        Ok(ws)
    };
    match sub {
        "run" => {
            let usage = "usage: math-draw weight-lab run <weights.json> --in 80,80";
            let parsed = match ParsedArgs::parse(sub_rest, &["--in"], &[]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(1, usage) {
                return err_exit_typed(&e);
            }
            let (Some(wf), Some(in_str)) = (parsed.positionals.first(), parsed.str_opt("--in"))
            else {
                return err_exit_usage("weight_lab.run", usage);
            };
            let ws = match load_ws(wf) {
                Ok(w) => w,
                Err(e) => return err_exit_stage("weight_lab.run", &e),
            };
            let v: Result<Vec<f64>, String> = in_str
                .split(',')
                .map(|s| {
                    let val = s
                        .trim()
                        .parse::<f64>()
                        .map_err(|e| format!("invalid --in vector '{in_str}': {e}"))?;
                    if !val.is_finite() {
                        return Err(format!("invalid --in vector '{in_str}': values must be finite"));
                    }
                    Ok(val)
                })
                .collect();
            let v = match v {
                Ok(vec) => vec,
                Err(e) => return err_exit_code(ErrorCode::CliInvalidNumber, "cli_parse", &e),
            };
            let eng = match engine_probe(&ws.w, &ws.b, std::slice::from_ref(&v)) {
                Ok(mut outs) => outs.remove(0),
                Err(e) => return err_exit_stage("weight_lab.run", &e),
            };
            let js = match mat_vec_mul(&ws.w, &v) {
                Ok(mv) => mv
                    .into_iter()
                    .zip(ws.b.iter())
                    .map(|(x, &bi)| x + bi)
                    .collect::<Vec<f64>>(),
                Err(e) => return err_exit_stage("weight_lab.run", &e),
            };
            let diff = eng
                .iter()
                .zip(js.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            print_json_exit(
                &json!({
                    "ok": true,
                    "name": ws.name,
                    "kind": ws.kind,
                    "engine_f32": eng,
                    "ref_f64": js,
                    "max_abs_diff": diff,
                }),
                true,
            )
        }
        "gate" => {
            let usage =
                "usage: math-draw weight-lab gate <current.json> <candidate.json> [--jump 1.0]";
            let parsed = match ParsedArgs::parse(sub_rest, &["--jump"], &[]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(2, usage) {
                return err_exit_typed(&e);
            }
            if parsed.positionals.len() < 2 {
                return err_exit_usage("weight_lab.gate", usage);
            }
            let cur = match load_ws(&parsed.positionals[0]) {
                Ok(w) => w,
                Err(e) => return err_exit_stage("weight_lab.gate", &e),
            };
            let cand = match load_ws(&parsed.positionals[1]) {
                Ok(w) => w,
                Err(e) => return err_exit_stage("weight_lab.gate", &e),
            };
            let jump_tol = match parsed.parse_f64_non_neg("--jump", 1.0) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            match gate_sets(&cur, &cand, jump_tol) {
                Ok(rep) => print_json_exit(&rep, rep.ok),
                Err(e) => err_exit_stage("weight_lab.gate", &e),
            }
        }
        "certify" => {
            let usage =
                "usage: math-draw weight-lab certify <weights.json> [--repertoire <path>]";
            let parsed = match ParsedArgs::parse(sub_rest, &["--repertoire"], &[]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(1, usage) {
                return err_exit_typed(&e);
            }
            let Some(wf) = parsed.positionals.first() else {
                return err_exit_usage("weight_lab.certify", usage);
            };
            let ws = match load_ws(wf) {
                Ok(w) => w,
                Err(e) => return err_exit_stage("weight_lab.certify", &e),
            };
            let def_name = Path::new(wf)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("weights");
            let rep_path = parsed.str_opt("--repertoire").map(Path::new);
            match certify_weight_set(&ws, def_name, rep_path) {
                Ok(val) => {
                    let ok = val.get("ok").and_then(|b| b.as_bool()).unwrap_or(false);
                    print_json_exit(&val, ok)
                }
                Err(e) => err_exit_stage("weight_lab.certify", &e),
            }
        }
        "compose" => {
            let usage = "usage: math-draw weight-lab compose <A.json> <B.json> [--out <C.json>]";
            let parsed = match ParsedArgs::parse(sub_rest, &["--out"], &[]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(2, usage) {
                return err_exit_typed(&e);
            }
            if parsed.positionals.len() < 2 {
                return err_exit_usage("weight_lab.compose", usage);
            }
            let a = match load_ws(&parsed.positionals[0]) {
                Ok(w) => w,
                Err(e) => return err_exit_stage("weight_lab.compose", &e),
            };
            let b = match load_ws(&parsed.positionals[1]) {
                Ok(w) => w,
                Err(e) => return err_exit_stage("weight_lab.compose", &e),
            };
            match compose_properties(&a, &b) {
                Ok(rep) => {
                    if let Some(op) = parsed.str_opt("--out")
                        && rep.ok
                        && let Err(e) = write_pretty_json(
                            op,
                            &json!({
                                "name": rep.name,
                                "kind": rep.kind,
                                "W": rep.w,
                                "b": rep.b,
                                "composition_method": rep.method,
                            }),
                        )
                    {
                        return err_exit_typed(&e);
                    }
                    print_json_exit(&rep, rep.ok)
                }
                Err(e) => err_exit_stage("weight_lab.compose", &e),
            }
        }
        "eval" => {
            let usage = "usage: math-draw weight-lab eval <expr.json> [--repertoire <path>] [--out <C.json>]";
            let parsed = match ParsedArgs::parse(sub_rest, &["--repertoire", "--out"], &[]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(1, usage) {
                return err_exit_typed(&e);
            }
            let Some(ef) = parsed.positionals.first() else {
                return err_exit_usage("weight_lab.eval", usage);
            };
            let raw = match std::fs::read_to_string(ef) {
                Ok(r) => r,
                Err(e) => {
                    return err_exit_code(
                        ErrorCode::IoReadFailed,
                        "weight_lab.eval",
                        &format!("cannot read expr JSON '{ef}': {e}"),
                    );
                }
            };
            let doc: serde_json::Value = match serde_json::from_str(&raw) {
                Ok(v) => v,
                Err(e) => {
                    return err_exit_code(
                        ErrorCode::JsonParseFailed,
                        "weight_lab.eval",
                        &format!("invalid expr JSON '{ef}': {e}"),
                    );
                }
            };
            let mut rep_map: BTreeMap<String, WeightSet> = BTreeMap::new();
            if let Some(rp) = parsed.str_opt("--repertoire") {
                let r_raw = match std::fs::read_to_string(rp) {
                    Ok(r) => r,
                    Err(e) => {
                        return err_exit_code(
                            ErrorCode::IoReadFailed,
                            "weight_lab.eval",
                            &format!("cannot read repertoire '{rp}': {e}"),
                        );
                    }
                };
                let r_val: serde_json::Value = match serde_json::from_str(&r_raw) {
                    Ok(v) => v,
                    Err(e) => {
                        return err_exit_code(
                            ErrorCode::RepertoireInvalid,
                            "weight_lab.eval",
                            &format!("invalid repertoire JSON '{rp}': {e}"),
                        );
                    }
                };
                let Some(sets) = r_val.get("sets").and_then(|v| v.as_array()) else {
                    return err_exit_code(
                        ErrorCode::RepertoireInvalid,
                        "weight_lab.eval",
                        &format!("repertoire '{rp}' missing 'sets' array"),
                    );
                };
                for (idx, s) in sets.iter().enumerate() {
                    let ws: WeightSet = match serde_json::from_value(s.clone()) {
                        Ok(w) => w,
                        Err(e) => {
                            return err_exit_code(
                                ErrorCode::RepertoireInvalid,
                                "weight_lab.eval",
                                &format!("invalid weight set at repertoire '{rp}' sets[{idx}]: {e}"),
                            );
                        }
                    };
                    if let Err(e) = ws.validate_dims() {
                        return err_exit_code(
                            ErrorCode::RepertoireInvalid,
                            "weight_lab.eval",
                            &format!("invalid dimensions at repertoire '{rp}' sets[{idx}]: {e}"),
                        );
                    }
                    let Some(ref nm) = ws.name else {
                        return err_exit_code(
                            ErrorCode::RepertoireInvalid,
                            "weight_lab.eval",
                            &format!("repertoire '{rp}' sets[{idx}] is missing 'name'"),
                        );
                    };
                    rep_map.insert(nm.clone(), ws);
                }
            }
            if let Some(inline_rep_val) = doc.get("repertoire") {
                let Some(inline_rep) = inline_rep_val.as_object() else {
                    return err_exit_code(
                        ErrorCode::RepertoireInvalid,
                        "weight_lab.eval",
                        "inline 'repertoire' must be a JSON object",
                    );
                };
                for (k, v) in inline_rep {
                    let ws: WeightSet = match serde_json::from_value(v.clone()) {
                        Ok(w) => w,
                        Err(e) => {
                            return err_exit_code(
                                ErrorCode::RepertoireInvalid,
                                "weight_lab.eval",
                                &format!("invalid inline repertoire entry '{k}': {e}"),
                            );
                        }
                    };
                    if let Err(e) = ws.validate_dims() {
                        return err_exit_code(
                            ErrorCode::RepertoireInvalid,
                            "weight_lab.eval",
                            &format!("invalid dimensions for inline repertoire entry '{k}': {e}"),
                        );
                    }
                    rep_map.insert(k.clone(), ws);
                }
            }
            let expr_node = doc.get("expr").unwrap_or(&doc);
            match evaluate_weight_expr(expr_node, &rep_map) {
                Ok(rep) => {
                    if let Some(op) = parsed.str_opt("--out")
                        && rep.ok
                        && let Err(e) = write_pretty_json(
                            op,
                            &json!({
                                "name": rep.name,
                                "kind": rep.kind,
                                "W": rep.w,
                                "b": rep.b,
                                "composition_method": rep.method,
                            }),
                        )
                    {
                        return err_exit_typed(&e);
                    }
                    print_json_exit(&rep, rep.ok)
                }
                Err(e) => err_exit_stage("weight_lab.eval", &e),
            }
        }
        _ => err_exit_code(
            ErrorCode::CliUnknownCommand,
            "weight_lab",
            &format!(
                "unknown weight-lab subcommand '{sub}' (use: run | gate | certify | compose | eval | selftest)"
            ),
        ),
    }
}

pub(super) fn handle_bundle(rest: &[String]) -> ExitCode {
    let Some(sub) = rest.first().map(|s| s.as_str()) else {
        return err_exit_usage("bundle", "usage: math-draw bundle <binding|export|verify> ...");
    };
    let sub_rest = &rest[1..];
    match sub {
        "binding" => {
            let usage = "usage: math-draw bundle binding <spec.json>";
            let parsed = match ParsedArgs::parse(sub_rest, &[], &[]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(1, usage) {
                return err_exit_typed(&e);
            }
            let Some(file) = parsed.positionals.first() else {
                return err_exit_usage("bundle.binding", usage);
            };
            let loaded = match load_spec_file(Path::new(file)) {
                Ok(l) => l,
                Err(e) => return err_exit_stage("load_spec", &e),
            };
            match compute_parameter_binding(&loaded.spec) {
                Ok(binding) => print_json_exit(
                    &json!({
                        "ok": true,
                        "was_v2": loaded.was_v2,
                        "binding": binding,
                    }),
                    true,
                ),
                Err(e) => err_exit_stage("bundle.binding", &e),
            }
        }
        "export" => {
            let usage = "usage: math-draw bundle export <spec.json> [--out <bundle.json>] [--attest <label>] [--no-exec]";
            let parsed = match ParsedArgs::parse_full(
                sub_rest,
                &["--out"],
                &[],
                &["--attest"],
                &["--no-exec"],
            ) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(1, usage) {
                return err_exit_typed(&e);
            }
            let Some(file) = parsed.positionals.first() else {
                return err_exit_usage("bundle.export", usage);
            };
            let loaded = match load_spec_file(Path::new(file)) {
                Ok(l) => l,
                Err(e) => return err_exit_stage("load_spec", &e),
            };
            let include_exec = !parsed.has("--no-exec");
            let attest_labels = parsed.str_all("--attest");
            match export_bundle(&loaded.spec, include_exec, &attest_labels) {
                Ok(bundle) => {
                    if let Some(op) = parsed.str_opt("--out")
                        && let Err(e) = write_pretty_json(op, &bundle)
                    {
                        return err_exit_typed(&e);
                    }
                    print_json_exit(
                        &json!({
                            "ok": true,
                            "schema": bundle.schema,
                            "program_identity": bundle.program_identity,
                            "bundle_sha256": bundle.bundle_sha256,
                            "total_params": bundle.parameter_binding.total_params,
                            "trainable_params": bundle.parameter_binding.trainable_params,
                            "attestations_count": bundle.attestations.len(),
                            "has_reference_execution": bundle.reference_execution.is_some(),
                            "bundle": bundle,
                        }),
                        true,
                    )
                }
                Err(e) => err_exit_stage("bundle.export", &e),
            }
        }
        "verify" => {
            let usage = "usage: math-draw bundle verify <bundle.json> [--out-spec <spec.json>]";
            let parsed = match ParsedArgs::parse(sub_rest, &["--out-spec"], &[]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(1, usage) {
                return err_exit_typed(&e);
            }
            let Some(file) = parsed.positionals.first() else {
                return err_exit_usage("bundle.verify", usage);
            };
            let raw = match std::fs::read_to_string(file) {
                Ok(r) => r,
                Err(e) => {
                    return err_exit_code(
                        ErrorCode::IoReadFailed,
                        "bundle.verify",
                        &format!("cannot read bundle '{file}': {e}"),
                    );
                }
            };
            let bundle: ProgramBundle = match serde_json::from_str(&raw) {
                Ok(b) => b,
                Err(e) => {
                    return err_exit_code(
                        ErrorCode::BundleInvalid,
                        "bundle.verify",
                        &format!("invalid bundle JSON '{file}': {e}"),
                    );
                }
            };
            let rep = verify_bundle(&bundle);
            if rep.ok
                && let Some(op) = parsed.str_opt("--out-spec")
                && let Err(e) = write_pretty_json(op, &bundle.spec)
            {
                return err_exit_typed(&e);
            }
            print_json_exit(&rep, rep.ok)
        }
        _ => err_exit_code(
            ErrorCode::CliUnknownCommand,
            "bundle",
            &format!("unknown bundle subcommand '{sub}' (use: binding | export | verify)"),
        ),
    }
}

fn parse_f64_array(arr: &[serde_json::Value], ctx: &str) -> Result<Vec<f64>, String> {
    let mut out = Vec::with_capacity(arr.len());
    for (i, v) in arr.iter().enumerate() {
        let f = v
            .as_f64()
            .ok_or_else(|| format!("{ctx}[{i}]: expected finite number, got {v}"))?;
        if !f.is_finite() {
            return Err(format!("{ctx}[{i}]: non-finite number"));
        }
        out.push(f);
    }
    Ok(out)
}

fn parse_external_ref_json(val: &serde_json::Value) -> Result<Vec<Vec<f64>>, String> {
    if let Some(arr) = val.as_array() {
        if arr.is_empty() {
            return Err("external ref JSON array must not be empty".into());
        }
        if arr[0].is_array() {
            let mut out = Vec::with_capacity(arr.len());
            for (ci, item) in arr.iter().enumerate() {
                let inner = item
                    .as_array()
                    .ok_or_else(|| format!("ref[{ci}]: expected array of numbers"))?;
                out.push(parse_f64_array(inner, &format!("ref[{ci}]"))?);
            }
            return Ok(out);
        }
        return Ok(vec![parse_f64_array(arr, "ref")?]);
    }
    if let Some(obj) = val.as_object() {
        if let Some(outputs) = obj.get("outputs") {
            return parse_external_ref_json(outputs);
        }
        if let Some(cases) = obj.get("cases").and_then(|v| v.as_array()) {
            let mut out = Vec::with_capacity(cases.len());
            for (ci, c) in cases.iter().enumerate() {
                let exp = c
                    .get("expected")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| format!("ref.cases[{ci}]: missing 'expected' numeric array"))?;
                out.push(parse_f64_array(exp, &format!("ref.cases[{ci}].expected"))?);
            }
            return Ok(out);
        }
    }
    Err(
        "external ref JSON must be [num...], [[num...]...], {\"outputs\": ...}, or {\"cases\": [{\"expected\": [...]}]}"
            .into(),
    )
}

pub(super) fn handle_diff_ref(rest: &[String]) -> ExitCode {
    let usage =
        "usage: math-draw diff-ref <spec.json> [--cases <cases.json>] [--ref <ref.json>] [--tol 1e-5]";
    let parsed = match ParsedArgs::parse(rest, &["--cases", "--ref", "--tol"], &[]) {
        Ok(p) => p,
        Err(e) => return err_exit_typed(&e),
    };
    if let Err(e) = parsed.ensure_max_positionals(1, usage) {
        return err_exit_typed(&e);
    }
    let Some(spec_file) = parsed.positionals.first() else {
        return err_exit_usage("diff_ref", usage);
    };
    let loaded = match load_spec_file(Path::new(spec_file)) {
        Ok(l) => l,
        Err(e) => return err_exit_stage("load_spec", &e),
    };
    let tol = match parsed.parse_f64_pos("--tol", 1e-5) {
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
    let ext_ref = if let Some(rf) = parsed.str_opt("--ref") {
        let raw = match std::fs::read_to_string(rf) {
            Ok(r) => r,
            Err(e) => {
                return err_exit_code(
                    ErrorCode::IoReadFailed,
                    "diff_ref",
                    &format!("cannot read ref JSON '{rf}': {e}"),
                );
            }
        };
        let val: serde_json::Value = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(e) => {
                return err_exit_code(
                    ErrorCode::JsonParseFailed,
                    "diff_ref",
                    &format!("invalid ref JSON '{rf}': {e}"),
                );
            }
        };
        match parse_external_ref_json(&val) {
            Ok(v) => Some(v),
            Err(e) => return err_exit_stage("diff_ref", &e),
        }
    } else {
        None
    };

    match diff_ref_spec(&loaded.spec, cases.as_deref(), ext_ref.as_deref(), tol) {
        Ok(rep) => print_json_exit(&rep, rep.ok),
        Err(e) => err_exit_stage("diff_ref", &e),
    }
}

