use serde_json::json;
use std::path::Path;
use std::process::ExitCode;

use super::{err_exit, flag_val, positional_args, print_json_exit};
use crate::engine::run_spec;
use crate::harness::{
    compute_semantic_digest, ensure_workspace_dir, execute_harness_workflow,
    export_subagent_branches, get_capability_catalog, merge_subagent_branches,
    resolve_artifact_uri,
};
use crate::spec::load_spec_file;

pub(super) fn handle_harness(rest: &[String]) -> ExitCode {
    let Some(sub) = rest.first().map(|s| s.as_str()) else {
        return err_exit("usage: math-draw harness <capabilities|digest|resolve-artifact|run-workflow|branch-export|branch-merge|workspace-status> ...");
    };
    let sub_rest = &rest[1..];
    let ws_dir = Path::new(flag_val(sub_rest, "--workspace").unwrap_or(".burn-harness"));
    let valued = ["--workspace", "--tol"];
    let pos = positional_args(sub_rest, &valued);
    match sub {
        "capabilities" => {
            let cat = get_capability_catalog(pos.first().map(|s| s.as_str()));
            let ok = cat.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
            print_json_exit(&cat, ok)
        }
        "workspace-status" => match ensure_workspace_dir(ws_dir) {
            Ok(ws) => {
                let raw =
                    std::fs::read_to_string(&ws.state_file).unwrap_or_else(|_| "{}".into());
                let st: serde_json::Value =
                    serde_json::from_str(&raw).unwrap_or_else(|_| json!({}));
                print_json_exit(
                    &json!({
                        "ok": true,
                        "workspace": ws.root.display().to_string(),
                        "state": st,
                    }),
                    true,
                )
            }
            Err(e) => err_exit(&e),
        },
        "resolve-artifact" => {
            let Some(uri) = pos.first() else {
                return err_exit("usage: math-draw harness resolve-artifact <burn://sha256:...> [--workspace .burn-harness]");
            };
            match resolve_artifact_uri(uri, ws_dir) {
                Ok(val) => {
                    print_json_exit(&json!({ "ok": true, "uri": uri, "artifact": val }), true)
                }
                Err(e) => err_exit(&e),
            }
        }
        "digest" => {
            let Some(sf) = pos.first() else {
                return err_exit("usage: math-draw harness digest <spec.json> [--workspace .burn-harness]");
            };
            let loaded = match load_spec_file(Path::new(sf)) {
                Ok(l) => l,
                Err(e) => return err_exit(&e),
            };
            let out = match run_spec(&loaded.spec) {
                Ok(o) => o,
                Err(e) => return err_exit(&e),
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
                Err(e) => err_exit(&e),
            }
        }
        "branch-export" => {
            let Some(sf) = pos.first() else {
                return err_exit("usage: math-draw harness branch-export <spec.json> [--workspace .burn-harness]");
            };
            let loaded = match load_spec_file(Path::new(sf)) {
                Ok(l) => l,
                Err(e) => return err_exit(&e),
            };
            match export_subagent_branches(&loaded.spec, ws_dir) {
                Ok(val) => print_json_exit(&val, true),
                Err(e) => err_exit(&e),
            }
        }
        "branch-merge" => {
            if pos.len() < 2 {
                return err_exit("usage: math-draw harness branch-merge <baseline.json> <cand1.json> [cand2.json ...] [--tol 1e-4]");
            }
            let base = match load_spec_file(Path::new(&pos[0])) {
                Ok(l) => l.spec,
                Err(e) => return err_exit(&e),
            };
            let tol = flag_val(sub_rest, "--tol")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1e-4f64);
            match merge_subagent_branches(&base, &pos[1..], tol) {
                Ok(val) => {
                    let ok = val.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
                    print_json_exit(&val, ok)
                }
                Err(e) => err_exit(&e),
            }
        }
        "run-workflow" => {
            let Some(wf_file) = pos.first() else {
                return err_exit("usage: math-draw harness run-workflow <workflow.json> [--workspace .burn-harness]");
            };
            let raw = match std::fs::read_to_string(wf_file) {
                Ok(r) => r,
                Err(e) => return err_exit(&format!("cannot read workflow '{wf_file}': {e}")),
            };
            let wf_val: serde_json::Value = match serde_json::from_str(&raw) {
                Ok(v) => v,
                Err(e) => return err_exit(&format!("invalid workflow JSON '{wf_file}': {e}")),
            };
            let base_dir = Path::new(wf_file).parent().unwrap_or_else(|| Path::new("."));
            let ws_ov = flag_val(sub_rest, "--workspace").map(Path::new);
            match execute_harness_workflow(&wf_val, base_dir, ws_ov) {
                Ok(val) => {
                    let ok = val.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
                    print_json_exit(&val, ok)
                }
                Err(e) => err_exit(&e),
            }
        }
        _ => err_exit(&format!("unknown harness subcommand '{sub}'")),
    }
}
