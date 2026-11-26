use serde_json::{Value, json};
use std::process::ExitCode;

use super::{
    ParsedArgs, err_exit_code, err_exit_stage, err_exit_typed, err_exit_usage, print_json_exit,
    write_pretty_json,
};
use crate::error::ErrorCode;
use crate::geom3::{
    build_chain_spec, compile_scene_tree, project_model, run_geom_selftest, solve_ik,
};

fn read_json_doc(path: &str, stage: &str) -> Result<Value, ExitCode> {
    let raw = match std::fs::read_to_string(path) {
        Ok(r) => r,
        Err(e) => {
            return Err(err_exit_code(
                ErrorCode::IoReadFailed,
                stage,
                &format!("cannot read JSON file '{path}': {e}"),
            ));
        }
    };
    match serde_json::from_str::<Value>(&raw) {
        Ok(v) => Ok(v),
        Err(e) => Err(err_exit_code(
            ErrorCode::JsonParseFailed,
            stage,
            &format!("invalid JSON in '{path}': {e}"),
        )),
    }
}

pub(super) fn handle_geom(rest: &[String]) -> ExitCode {
    let usage = "usage: math-draw geom <build|scene-tree|solve-ik|project|batch|selftest> ...";
    let Some(sub) = rest.first().map(|s| s.as_str()) else {
        return err_exit_usage("geom", usage);
    };
    let sub_rest = &rest[1..];

    match sub {
        "selftest" | "--selftest" => {
            let parsed = match ParsedArgs::parse(sub_rest, &[], &[]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) =
                parsed.ensure_max_positionals(0, "usage: math-draw geom selftest")
            {
                return err_exit_typed(&e);
            }
            match run_geom_selftest() {
                Ok(v) => {
                    let ok = v.get("ok").and_then(|b| b.as_bool()).unwrap_or(false);
                    print_json_exit(&v, ok)
                }
                Err(e) => err_exit_stage("geom.selftest", &e),
            }
        }
        "build" => {
            let u = "usage: math-draw geom build <chain.json> [--compose] [--out <spec.json>]";
            let parsed = match ParsedArgs::parse(sub_rest, &["--out"], &["--compose"]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(1, u) {
                return err_exit_typed(&e);
            }
            let Some(file) = parsed.positionals.first() else {
                return err_exit_usage("geom.build", u);
            };
            let doc = match read_json_doc(file, "geom.build") {
                Ok(v) => v,
                Err(code) => return code,
            };
            let compose = parsed.has("--compose");
            match build_chain_spec(&doc, compose) {
                Ok(rep) => {
                    if let Some(op) = parsed.str_opt("--out") {
                        if let Err(e) = write_pretty_json(op, &rep.spec) {
                            return err_exit_typed(&e);
                        }
                        return print_json_exit(
                            &json!({
                                "ok": rep.ok,
                                "composed": rep.composed,
                                "steps": rep.steps,
                                "out_shape": rep.out_shape,
                                "reference_point_f64": rep.reference_point_f64,
                                "composed_matrix_rowmajor": rep.composed_matrix_rowmajor,
                                "note": rep.note,
                                "out_file": op,
                            }),
                            rep.ok,
                        );
                    }
                    print_json_exit(&rep, rep.ok)
                }
                Err(e) => err_exit_stage("geom.build", &e),
            }
        }
        "scene-tree" => {
            let u = "usage: math-draw geom scene-tree <tree.json> [--out <spec.json>]";
            let parsed = match ParsedArgs::parse(sub_rest, &["--out"], &[]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(1, u) {
                return err_exit_typed(&e);
            }
            let Some(file) = parsed.positionals.first() else {
                return err_exit_usage("geom.scene_tree", u);
            };
            let doc = match read_json_doc(file, "geom.scene_tree") {
                Ok(v) => v,
                Err(code) => return code,
            };
            match compile_scene_tree(&doc) {
                Ok(rep) => {
                    if let Some(op) = parsed.str_opt("--out") {
                        if let Err(e) = write_pretty_json(op, &rep.spec) {
                            return err_exit_typed(&e);
                        }
                        return print_json_exit(
                            &json!({
                                "ok": rep.ok,
                                "nodes": rep.nodes,
                                "out_file": op,
                            }),
                            rep.ok,
                        );
                    }
                    print_json_exit(&rep, rep.ok)
                }
                Err(e) => err_exit_stage("geom.scene_tree", &e),
            }
        }
        "solve-ik" => {
            let u = "usage: math-draw geom solve-ik <ik.json> [--out <spec.json>]";
            let parsed = match ParsedArgs::parse(sub_rest, &["--out"], &[]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(1, u) {
                return err_exit_typed(&e);
            }
            let Some(file) = parsed.positionals.first() else {
                return err_exit_usage("geom.solve_ik", u);
            };
            let doc = match read_json_doc(file, "geom.solve_ik") {
                Ok(v) => v,
                Err(code) => return code,
            };
            match solve_ik(&doc) {
                Ok(rep) => {
                    if let Some(op) = parsed.str_opt("--out") {
                        if rep.ok
                            && let Err(e) = write_pretty_json(op, &rep.scene_tree.spec)
                        {
                            return err_exit_typed(&e);
                        }
                        return print_json_exit(
                            &json!({
                                "ok": rep.ok,
                                "converged": rep.converged,
                                "iterations": rep.iterations,
                                "start_error": rep.start_error,
                                "final_error": rep.final_error,
                                "final_distance": rep.final_distance,
                                "tol": rep.tol,
                                "target": rep.target,
                                "end_effector": rep.end_effector,
                                "engine_end_effector": rep.engine_end_effector,
                                "engine_max_abs_diff": rep.engine_max_abs_diff,
                                "solved_angles_deg": rep.solved_angles_deg,
                                "out_file": if rep.ok { Some(op) } else { None },
                            }),
                            rep.ok,
                        );
                    }
                    print_json_exit(&rep, rep.ok)
                }
                Err(e) => err_exit_stage("geom.solve_ik", &e),
            }
        }
        "project" | "batch" => {
            let batched = sub == "batch";
            let u = if batched {
                "usage: math-draw geom batch <model.json> [--out <out.json>]"
            } else {
                "usage: math-draw geom project <model.json> [--out <out.json>]"
            };
            let stage = if batched { "geom.batch" } else { "geom.project" };
            let parsed = match ParsedArgs::parse(sub_rest, &["--out"], &[]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(1, u) {
                return err_exit_typed(&e);
            }
            let Some(file) = parsed.positionals.first() else {
                return err_exit_usage(stage, u);
            };
            let doc = match read_json_doc(file, stage) {
                Ok(v) => v,
                Err(code) => return code,
            };
            match project_model(&doc, batched) {
                Ok(rep) => {
                    if let Some(op) = parsed.str_opt("--out")
                        && let Err(e) = write_pretty_json(op, &rep)
                    {
                        return err_exit_typed(&e);
                    }
                    print_json_exit(&rep, rep.ok)
                }
                Err(e) => err_exit_stage(stage, &e),
            }
        }
        _ => err_exit_code(
            ErrorCode::CliUnknownCommand,
            "geom",
            &format!(
                "unknown geom subcommand '{sub}' (use: build | scene-tree | solve-ik | project | batch | selftest)"
            ),
        ),
    }
}
