use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Path;

use super::workspace::ensure_workspace_dir;
use crate::engine::PreparedGraph;
use crate::spec::{SpecV1, load_spec_file, sha256_json};
use crate::verify::{EvalCase, verify_specs};

pub fn export_subagent_branches(spec: &SpecV1, workspace_dir: &Path) -> Result<Value, String> {
    let ws = ensure_workspace_dir(workspace_dir)?;
    let baseline_sha = sha256_json(spec);
    let baseline_path = ws.branches_dir.join("baseline.spec.json");
    let baseline_txt = serde_json::to_string_pretty(spec)
        .map_err(|e| format!("failed serializing baseline spec: {e}"))?;
    std::fs::write(&baseline_path, format!("{baseline_txt}\n"))
        .map_err(|e| format!("failed writing '{}': {e}", baseline_path.display()))?;

    let default_branches = [
        (
            "sub-rewrite",
            "structural_rewrite",
            "Run math-draw recurse search on baseline.spec.json and save to sub-rewrite.candidate.spec.json",
        ),
        (
            "sub-quant",
            "quantization",
            "Run math-draw quant budget on baseline.spec.json and save to sub-quant.candidate.spec.json",
        ),
    ];

    let mut packets = Vec::new();
    for (bid, goal, instr) in default_branches {
        let cand_path = ws.branches_dir.join(format!("{bid}.candidate.spec.json"));
        let pkt = json!({
            "branch_id": bid,
            "goal": goal,
            "baseline_state_sha256": baseline_sha,
            "baseline_spec_path": baseline_path.display().to_string(),
            "candidate_output_path": cand_path.display().to_string(),
            "instructions": instr,
        });
        let pkt_path = ws.branches_dir.join(format!("{bid}.packet.json"));
        let pkt_txt = serde_json::to_string_pretty(&pkt)
            .map_err(|e| format!("failed serializing branch packet '{bid}': {e}"))?;
        std::fs::write(&pkt_path, format!("{pkt_txt}\n"))
            .map_err(|e| format!("failed writing '{}': {e}", pkt_path.display()))?;
        packets.push(pkt);
    }

    Ok(json!({
        "ok": true,
        "baseline_state_sha256": baseline_sha,
        "baseline_spec_path": baseline_path.display().to_string(),
        "branch_packets": packets,
    }))
}

pub fn merge_subagent_branches(
    baseline: &SpecV1,
    candidate_files: &[String],
    tol: f64,
) -> Result<Value, String> {
    if !(tol.is_finite() && tol >= 0.0) {
        return Err(format!(
            "merge_subagent_branches: tol must be finite and >= 0, got {tol}"
        ));
    }
    let base_graph = PreparedGraph::compile(baseline)?;
    let baseline_sha = sha256_json(baseline);

    let derive_inputs = |scale: f64, const_val: Option<f64>| -> BTreeMap<String, Vec<Value>> {
        let mut out = BTreeMap::new();
        for (k, arr) in &baseline.inputs {
            let mapped = arr
                .iter()
                .map(|v| {
                    if let Some(f) = v.as_f64() {
                        Value::from(const_val.unwrap_or(f * scale))
                    } else {
                        v.clone()
                    }
                })
                .collect();
            out.insert(k.clone(), mapped);
        }
        out
    };

    let named_candidates = [
        ("baseline", baseline.inputs.clone()),
        ("zeros", derive_inputs(0.0, Some(0.0))),
        ("ones", derive_inputs(0.0, Some(1.0))),
        ("negated", derive_inputs(-1.0, None)),
        ("scaled_x2", derive_inputs(2.0, None)),
        ("scaled_half", derive_inputs(0.5, None)),
    ];

    let mut cases = Vec::new();
    let mut case_names = Vec::new();
    for (name, inp) in named_candidates {
        if let Ok(out) = base_graph.run_once(baseline, Some(&inp))
            && out.values.iter().all(|v| v.is_finite())
        {
            cases.push(EvalCase {
                inputs: inp,
                expected: None,
            });
            case_names.push(name);
        }
    }

    let mut evaluations = Vec::new();
    let mut best_winner: Option<(usize, f64, String, String, SpecV1)> = None;

    for (i, file) in candidate_files.iter().enumerate() {
        let branch_id = format!("merge.{}", i + 1);
        let cand = match load_spec_file(Path::new(file)) {
            Ok(l) => l.spec,
            Err(e) => {
                evaluations.push(json!({
                    "file": file,
                    "branch_id": branch_id,
                    "accepted": false,
                    "reason": format!("unreadable spec: {e}"),
                }));
                continue;
            }
        };
        let ver = verify_specs(baseline, &cand, &cases, tol, tol);
        let cand_sha = sha256_json(&cand);
        let receipt = sha256_json(&(&baseline_sha, &cand_sha, ver.equivalent, ver.max_abs_diff));
        let steps_len = cand.steps.len();
        evaluations.push(json!({
            "file": file,
            "branch_id": branch_id,
            "accepted": ver.equivalent,
            "equivalent": ver.equivalent,
            "max_abs_error": ver.max_abs_diff,
            "receipt_digest": receipt,
            "steps_count": steps_len,
        }));
        if ver.equivalent {
            let take = match &best_winner {
                None => true,
                Some((best_steps, best_err, ..)) => {
                    steps_len < *best_steps
                        || (steps_len == *best_steps && ver.max_abs_diff < *best_err)
                }
            };
            if take {
                best_winner = Some((
                    steps_len,
                    ver.max_abs_diff,
                    file.clone(),
                    branch_id,
                    cand,
                ));
            }
        }
    }

    let ok = best_winner.is_some();
    let (promoted_winner, promoted_spec) =
        if let Some((steps_count, max_err, file, bid, sp)) = best_winner {
            let csha = sha256_json(&sp);
            let receipt = sha256_json(&(&baseline_sha, &csha, true, max_err));
            (
                Some(json!({
                    "file": file,
                    "branch_id": bid,
                    "steps_count": steps_count,
                    "max_abs_error": max_err,
                    "receipt_digest": receipt,
                })),
                Some(sp),
            )
        } else {
            (None, None)
        };

    Ok(json!({
        "ok": ok,
        "baseline_state_sha256": baseline_sha,
        "equivalence_cases": case_names,
        "cases_count": cases.len(),
        "candidates_evaluated": evaluations.len(),
        "evaluations": evaluations,
        "promoted_winner": promoted_winner,
        "promoted_spec": promoted_spec,
    }))
}
