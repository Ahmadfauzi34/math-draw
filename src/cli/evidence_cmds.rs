use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;

use super::{
    err_exit, flag_val, has_flag, positional_args, print_json_exit, read_cases_file,
    write_pretty_json,
};
use crate::gradcheck::gradcheck_spec;
use crate::quant::{BudgetMode, QuantScheme, quant_budget, quant_verify};
use crate::spec::load_spec_file;
use crate::weight_lab::{
    WeightSet, certify_weight_set, compose_properties, engine_probe, evaluate_weight_expr,
    gate_sets, mat_vec_mul, run_weight_lab_selftest,
};

pub(super) fn handle_gradcheck(rest: &[String]) -> ExitCode {
    let valued = ["--eps", "--tol"];
    let pos = positional_args(rest, &valued);
    let Some(spec_file) = pos.first() else {
        return err_exit("usage: math-draw gradcheck <spec.json> [analytic.json] [--eps 1e-3] [--tol 1e-3] [--weights] [--autodiff]");
    };
    let loaded = match load_spec_file(Path::new(spec_file)) {
        Ok(l) => l,
        Err(e) => return err_exit(&e),
    };
    let analytic = if let Some(af) = pos.get(1) {
        let raw = match std::fs::read_to_string(af) {
            Ok(r) => r,
            Err(e) => return err_exit(&format!("cannot read analytic JSON '{af}': {e}")),
        };
        match serde_json::from_str::<serde_json::Value>(&raw) {
            Ok(v) => Some(v),
            Err(e) => return err_exit(&format!("invalid analytic JSON '{af}': {e}")),
        }
    } else {
        None
    };
    let eps = flag_val(rest, "--eps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3f64);
    let tol = flag_val(rest, "--tol")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3f64);
    let include_weights = has_flag(rest, "--weights");
    let check_autodiff = has_flag(rest, "--autodiff");
    match gradcheck_spec(
        &loaded.spec,
        analytic.as_ref(),
        eps,
        tol,
        include_weights,
        check_autodiff,
    ) {
        Ok(rep) => print_json_exit(&rep, rep.ok),
        Err(e) => err_exit(&e),
    }
}

pub(super) fn handle_quant(rest: &[String]) -> ExitCode {
    let Some(sub) = rest.first().map(|s| s.as_str()) else {
        return err_exit("usage: math-draw quant <verify|budget> ...");
    };
    let sub_rest = &rest[1..];
    match sub {
        "verify" => {
            let valued = ["--scheme", "--cases", "--abs-tol", "--rel-tol", "--out"];
            let pos = positional_args(sub_rest, &valued);
            let Some(file) = pos.first() else {
                return err_exit("usage: math-draw quant verify <spec.json> [--scheme int8|q8-per-block|fp16] [--cases <cases.json>] [--abs-tol 1e-2] [--rel-tol 1e-2] [--out <quant.json>]");
            };
            let loaded = match load_spec_file(Path::new(file)) {
                Ok(l) => l,
                Err(e) => return err_exit(&e),
            };
            let scheme =
                match QuantScheme::parse(flag_val(sub_rest, "--scheme").unwrap_or("int8")) {
                    Ok(s) => s,
                    Err(e) => return err_exit(&e),
                };
            let abs_tol = flag_val(sub_rest, "--abs-tol")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1e-2f64);
            let rel_tol = flag_val(sub_rest, "--rel-tol")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1e-2f64);
            let cases = if let Some(cf) = flag_val(sub_rest, "--cases") {
                match read_cases_file(cf) {
                    Ok(c) => Some(c),
                    Err(e) => return err_exit(&e),
                }
            } else {
                None
            };
            match quant_verify(&loaded.spec, scheme, cases.as_deref(), abs_tol, rel_tol) {
                Ok(rep) => {
                    if let Some(op) = flag_val(sub_rest, "--out")
                        && let Err(e) = write_pretty_json(op, &rep.quantized_spec)
                    {
                        return err_exit(&e);
                    }
                    print_json_exit(&rep, rep.ok)
                }
                Err(e) => err_exit(&e),
            }
        }
        "budget" => {
            let valued = ["--cases", "--tol", "--scheme", "--mode", "--out"];
            let pos = positional_args(sub_rest, &valued);
            let Some(file) = pos.first() else {
                return err_exit("usage: math-draw quant budget <spec.json> [--cases <cases.json>] [--tol 0.05] [--scheme q8-per-block|int8|fp16] [--mode bound|measured] [--out <plan.json>]");
            };
            let loaded = match load_spec_file(Path::new(file)) {
                Ok(l) => l,
                Err(e) => return err_exit(&e),
            };
            let scheme = match QuantScheme::parse(
                flag_val(sub_rest, "--scheme").unwrap_or("q8-per-block"),
            ) {
                Ok(s) => s,
                Err(e) => return err_exit(&e),
            };
            let mode = match BudgetMode::parse(flag_val(sub_rest, "--mode").unwrap_or("bound")) {
                Ok(m) => m,
                Err(e) => return err_exit(&e),
            };
            let tol = flag_val(sub_rest, "--tol")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.05f64);
            let cases = if let Some(cf) = flag_val(sub_rest, "--cases") {
                match read_cases_file(cf) {
                    Ok(c) => Some(c),
                    Err(e) => return err_exit(&e),
                }
            } else {
                None
            };
            match quant_budget(&loaded.spec, cases.as_deref(), tol, scheme, mode) {
                Ok(rep) => {
                    if let Some(op) = flag_val(sub_rest, "--out")
                        && let Err(e) = write_pretty_json(op, &rep.planned_spec)
                    {
                        return err_exit(&e);
                    }
                    print_json_exit(&rep, rep.ok)
                }
                Err(e) => err_exit(&e),
            }
        }
        _ => err_exit(&format!(
            "unknown quant subcommand '{sub}' (use: verify | budget)"
        )),
    }
}

pub(super) fn handle_weight_lab(rest: &[String]) -> ExitCode {
    if has_flag(rest, "--selftest") || rest.first().map(|s| s.as_str()) == Some("selftest") {
        return match run_weight_lab_selftest() {
            Ok(v) => {
                let ok = v.get("ok").and_then(|b| b.as_bool()).unwrap_or(false);
                print_json_exit(&v, ok)
            }
            Err(e) => err_exit(&e),
        };
    }
    let Some(sub) = rest.first().map(|s| s.as_str()) else {
        return err_exit("usage: math-draw weight-lab <run|gate|certify|compose|eval|selftest> ...");
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
            let valued = ["--in"];
            let pos = positional_args(sub_rest, &valued);
            let (Some(wf), Some(in_str)) = (pos.first(), flag_val(sub_rest, "--in")) else {
                return err_exit("usage: math-draw weight-lab run <weights.json> --in 80,80");
            };
            let ws = match load_ws(wf) {
                Ok(w) => w,
                Err(e) => return err_exit(&e),
            };
            let v: Result<Vec<f64>, _> =
                in_str.split(',').map(|s| s.trim().parse::<f64>()).collect();
            let v = match v {
                Ok(vec) => vec,
                Err(e) => return err_exit(&format!("invalid --in vector '{in_str}': {e}")),
            };
            let eng = match engine_probe(&ws.w, &ws.b, std::slice::from_ref(&v)) {
                Ok(mut outs) => outs.remove(0),
                Err(e) => return err_exit(&e),
            };
            let js = match mat_vec_mul(&ws.w, &v) {
                Ok(mv) => mv
                    .into_iter()
                    .zip(ws.b.iter())
                    .map(|(x, &bi)| x + bi)
                    .collect::<Vec<f64>>(),
                Err(e) => return err_exit(&e),
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
            let valued = ["--jump"];
            let pos = positional_args(sub_rest, &valued);
            if pos.len() < 2 {
                return err_exit(
                    "usage: math-draw weight-lab gate <current.json> <candidate.json> [--jump 1.0]",
                );
            }
            let cur = match load_ws(&pos[0]) {
                Ok(w) => w,
                Err(e) => return err_exit(&e),
            };
            let cand = match load_ws(&pos[1]) {
                Ok(w) => w,
                Err(e) => return err_exit(&e),
            };
            let jump_tol = flag_val(sub_rest, "--jump")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0f64);
            match gate_sets(&cur, &cand, jump_tol) {
                Ok(rep) => print_json_exit(&rep, rep.ok),
                Err(e) => err_exit(&e),
            }
        }
        "certify" => {
            let valued = ["--repertoire"];
            let pos = positional_args(sub_rest, &valued);
            let Some(wf) = pos.first() else {
                return err_exit("usage: math-draw weight-lab certify <weights.json> [--repertoire <path>]");
            };
            let ws = match load_ws(wf) {
                Ok(w) => w,
                Err(e) => return err_exit(&e),
            };
            let def_name = Path::new(wf)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("weights");
            let rep_path = flag_val(sub_rest, "--repertoire").map(Path::new);
            match certify_weight_set(&ws, def_name, rep_path) {
                Ok(val) => {
                    let ok = val.get("ok").and_then(|b| b.as_bool()).unwrap_or(false);
                    print_json_exit(&val, ok)
                }
                Err(e) => err_exit(&e),
            }
        }
        "compose" => {
            let valued = ["--out"];
            let pos = positional_args(sub_rest, &valued);
            if pos.len() < 2 {
                return err_exit(
                    "usage: math-draw weight-lab compose <A.json> <B.json> [--out <C.json>]",
                );
            }
            let a = match load_ws(&pos[0]) {
                Ok(w) => w,
                Err(e) => return err_exit(&e),
            };
            let b = match load_ws(&pos[1]) {
                Ok(w) => w,
                Err(e) => return err_exit(&e),
            };
            match compose_properties(&a, &b) {
                Ok(rep) => {
                    if let Some(op) = flag_val(sub_rest, "--out")
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
                        return err_exit(&e);
                    }
                    print_json_exit(&rep, rep.ok)
                }
                Err(e) => err_exit(&e),
            }
        }
        "eval" => {
            let valued = ["--repertoire", "--out"];
            let pos = positional_args(sub_rest, &valued);
            let Some(ef) = pos.first() else {
                return err_exit("usage: math-draw weight-lab eval <expr.json> [--repertoire <path>] [--out <C.json>]");
            };
            let raw = match std::fs::read_to_string(ef) {
                Ok(r) => r,
                Err(e) => return err_exit(&format!("cannot read expr JSON '{ef}': {e}")),
            };
            let doc: serde_json::Value = match serde_json::from_str(&raw) {
                Ok(v) => v,
                Err(e) => return err_exit(&format!("invalid expr JSON '{ef}': {e}")),
            };
            let mut rep_map: BTreeMap<String, WeightSet> = BTreeMap::new();
            if let Some(rp) = flag_val(sub_rest, "--repertoire")
                && let Ok(r_raw) = std::fs::read_to_string(rp)
                && let Ok(r_val) = serde_json::from_str::<serde_json::Value>(&r_raw)
                && let Some(sets) = r_val.get("sets").and_then(|v| v.as_array())
            {
                for s in sets {
                    if let Ok(ws) = serde_json::from_value::<WeightSet>(s.clone())
                        && let Some(ref nm) = ws.name
                    {
                        rep_map.insert(nm.clone(), ws);
                    }
                }
            }
            if let Some(inline_rep) = doc.get("repertoire").and_then(|v| v.as_object()) {
                for (k, v) in inline_rep {
                    if let Ok(ws) = serde_json::from_value::<WeightSet>(v.clone()) {
                        rep_map.insert(k.clone(), ws);
                    }
                }
            }
            let expr_node = doc.get("expr").unwrap_or(&doc);
            match evaluate_weight_expr(expr_node, &rep_map) {
                Ok(rep) => {
                    if let Some(op) = flag_val(sub_rest, "--out")
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
                        return err_exit(&e);
                    }
                    print_json_exit(&rep, rep.ok)
                }
                Err(e) => err_exit(&e),
            }
        }
        _ => err_exit(&format!(
            "unknown weight-lab subcommand '{sub}' (use: run | gate | certify | compose | eval | selftest)"
        )),
    }
}
