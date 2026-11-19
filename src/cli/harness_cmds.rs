use serde_json::json;
use std::path::Path;
use std::process::ExitCode;

use super::{
    ParsedArgs, err_exit_code, err_exit_stage, err_exit_typed, err_exit_usage, print_json_exit,
};
use crate::engine::run_spec;
use crate::error::ErrorCode;
use crate::harness::{
    compute_semantic_digest, ensure_workspace_dir, execute_harness_workflow,
    export_subagent_branches, get_capability_catalog, merge_subagent_branches,
    resolve_artifact_uri,
};
use crate::spec::load_spec_file;

pub(super) fn handle_harness(rest: &[String]) -> ExitCode {
    let Some(sub) = rest.first().map(|s| s.as_str()) else {
        return err_exit_usage(
            "harness",
            "usage: math-draw harness <capabilities|digest|resolve-artifact|run-workflow|branch-export|branch-merge|workspace-status> ...",
        );
    };
    let sub_rest = &rest[1..];
    match sub {
        "capabilities" => {
            let usage = "usage: math-draw harness capabilities [category]";
            let parsed = match ParsedArgs::parse(sub_rest, &[], &[]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(1, usage) {
                return err_exit_typed(&e);
            }
            let cat = get_capability_catalog(parsed.positionals.first().map(|s| s.as_str()));
            let ok = cat.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
            print_json_exit(&cat, ok)
        }
        "workspace-status" => {
            let usage = "usage: math-draw harness workspace-status [--workspace .burn-harness]";
            let parsed = match ParsedArgs::parse(sub_rest, &["--workspace"], &[]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(0, usage) {
                return err_exit_typed(&e);
            }
            let ws_dir = Path::new(parsed.str_opt("--workspace").unwrap_or(".burn-harness"));
            match ensure_workspace_dir(ws_dir) {
                Ok(ws) => {
                    let raw = match std::fs::read_to_string(&ws.state_file) {
                        Ok(r) => r,
                        Err(e) => {
                            return err_exit_code(
                                ErrorCode::IoReadFailed,
                                "harness.workspace_status",
                                &format!(
                                    "failed to read workspace state '{}': {e}",
                                    ws.state_file.display()
                                ),
                            );
                        }
                    };
                    let st: serde_json::Value = match serde_json::from_str(&raw) {
                        Ok(v) => v,
                        Err(e) => {
                            return err_exit_code(
                                ErrorCode::WorkspaceCorrupted,
                                "harness.workspace_status",
                                &format!(
                                    "corrupted workspace state '{}': {e}",
                                    ws.state_file.display()
                                ),
                            );
                        }
                    };
                    print_json_exit(
                        &json!({
                            "ok": true,
                            "workspace": ws.root.display().to_string(),
                            "state": st,
                        }),
                        true,
                    )
                }
                Err(e) => err_exit_stage("harness.workspace_status", &e),
            }
        }
        "resolve-artifact" => {
            let usage = "usage: math-draw harness resolve-artifact <burn://sha256:...> [--workspace .burn-harness]";
            let parsed = match ParsedArgs::parse(sub_rest, &["--workspace"], &[]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(1, usage) {
                return err_exit_typed(&e);
            }
            let Some(uri) = parsed.positionals.first() else {
                return err_exit_usage("harness.resolve_artifact", usage);
            };
            let ws_dir = Path::new(parsed.str_opt("--workspace").unwrap_or(".burn-harness"));
            match resolve_artifact_uri(uri, ws_dir) {
                Ok(val) => {
                    print_json_exit(&json!({ "ok": true, "uri": uri, "artifact": val }), true)
                }
                Err(e) => err_exit_stage("harness.resolve_artifact", &e),
            }
        }
        "digest" => {
            let usage = "usage: math-draw harness digest <spec.json> [--workspace .burn-harness]";
            let parsed = match ParsedArgs::parse(sub_rest, &["--workspace"], &[]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(1, usage) {
                return err_exit_typed(&e);
            }
            let Some(sf) = parsed.positionals.first() else {
                return err_exit_usage("harness.digest", usage);
            };
            let ws_dir = Path::new(parsed.str_opt("--workspace").unwrap_or(".burn-harness"));
            let loaded = match load_spec_file(Path::new(sf)) {
                Ok(l) => l,
                Err(e) => return err_exit_stage("load_spec", &e),
            };
            let out = match run_spec(&loaded.spec) {
                Ok(o) => o,
                Err(e) => return err_exit_stage("harness.digest", &e),
            };
            match compute_semantic_digest(
                &out.outputs.values,
                out.outputs.shape,
                Some(ws_dir),
                1,
                false,
            ) {
                Ok(dig) => print_json_exit(
                    &json!({
                        "ok": true,
                        "mode": "semantic_digest",
                        "digest": dig,
                    }),
                    true,
                ),
                Err(e) => err_exit_stage("harness.digest", &e),
            }
        }
        "branch-export" => {
            let usage =
                "usage: math-draw harness branch-export <spec.json> [--workspace .burn-harness]";
            let parsed = match ParsedArgs::parse(sub_rest, &["--workspace"], &[]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(1, usage) {
                return err_exit_typed(&e);
            }
            let Some(sf) = parsed.positionals.first() else {
                return err_exit_usage("harness.branch_export", usage);
            };
            let ws_dir = Path::new(parsed.str_opt("--workspace").unwrap_or(".burn-harness"));
            let loaded = match load_spec_file(Path::new(sf)) {
                Ok(l) => l,
                Err(e) => return err_exit_stage("load_spec", &e),
            };
            match export_subagent_branches(&loaded.spec, ws_dir) {
                Ok(val) => print_json_exit(&val, true),
                Err(e) => err_exit_stage("harness.branch_export", &e),
            }
        }
        "branch-merge" => {
            let usage = "usage: math-draw harness branch-merge <baseline.json> <cand1.json> [cand2.json ...] [--tol 1e-4]";
            let parsed = match ParsedArgs::parse(sub_rest, &["--tol"], &[]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if parsed.positionals.len() < 2 {
                return err_exit_usage("harness.branch_merge", usage);
            }
            let base = match load_spec_file(Path::new(&parsed.positionals[0])) {
                Ok(l) => l.spec,
                Err(e) => return err_exit_stage("load_spec", &e),
            };
            let tol = match parsed.parse_f64_non_neg("--tol", 1e-4) {
                Ok(v) => v,
                Err(e) => return err_exit_typed(&e),
            };
            match merge_subagent_branches(&base, &parsed.positionals[1..], tol) {
                Ok(val) => {
                    let ok = val.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
                    print_json_exit(&val, ok)
                }
                Err(e) => err_exit_stage("harness.branch_merge", &e),
            }
        }
        "run-workflow" => {
            let usage =
                "usage: math-draw harness run-workflow <workflow.json> [--workspace .burn-harness]";
            let parsed = match ParsedArgs::parse(sub_rest, &["--workspace"], &[]) {
                Ok(p) => p,
                Err(e) => return err_exit_typed(&e),
            };
            if let Err(e) = parsed.ensure_max_positionals(1, usage) {
                return err_exit_typed(&e);
            }
            let Some(wf_file) = parsed.positionals.first() else {
                return err_exit_usage("harness.run_workflow", usage);
            };
            let raw = match std::fs::read_to_string(wf_file) {
                Ok(r) => r,
                Err(e) => {
                    return err_exit_code(
                        ErrorCode::IoReadFailed,
                        "harness.run_workflow",
                        &format!("cannot read workflow '{wf_file}': {e}"),
                    );
                }
            };
            let wf_val: serde_json::Value = match serde_json::from_str(&raw) {
                Ok(v) => v,
                Err(e) => {
                    return err_exit_code(
                        ErrorCode::JsonParseFailed,
                        "harness.run_workflow",
                        &format!("invalid workflow JSON '{wf_file}': {e}"),
                    );
                }
            };
            let base_dir = Path::new(wf_file).parent().unwrap_or_else(|| Path::new("."));
            let ws_ov = parsed.str_opt("--workspace").map(Path::new);
            match execute_harness_workflow(&wf_val, base_dir, ws_ov) {
                Ok(val) => {
                    let ok = val.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
                    print_json_exit(&val, ok)
                }
                Err(e) => err_exit_stage("harness.run_workflow", &e),
            }
        }
        _ => err_exit_code(
            ErrorCode::CliUnknownCommand,
            "harness",
            &format!("unknown harness subcommand '{sub}'"),
        ),
    }
}
