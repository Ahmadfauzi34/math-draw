use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use crate::catalog::find_constructor;
use crate::spec::{LayerDecl, SpecV1, StepDecl, sha256_json};
use crate::validate::validate_spec;
use crate::verify::{EvalCase, default_cases_from_spec, verify_specs};

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "op")]
pub enum GraphEdit {
    #[serde(rename = "setWeights")]
    SetWeights {
        layer_index: usize,
        values: Vec<f32>,
    },
    #[serde(rename = "replaceStep")]
    ReplaceStep {
        index: usize,
        layer: LayerDecl,
        slots: Vec<usize>,
    },
    #[serde(rename = "insertStep")]
    InsertStep {
        index: usize,
        layer: LayerDecl,
        slots: Vec<usize>,
    },
    #[serde(rename = "removeStep")]
    RemoveStep { index: usize },
    #[serde(rename = "reconnectStep")]
    ReconnectStep { index: usize, slots: Vec<usize> },
    #[serde(rename = "setOutput")]
    SetOutput { slot: usize },
}

#[derive(Debug, Clone, Deserialize)]
pub struct BranchEditSpec {
    pub branch_id: String,
    pub edits: Vec<GraphEdit>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MutationRequest {
    #[serde(default)]
    pub commit: bool,
    pub promote_branch: Option<String>,
    pub edits: Option<Vec<GraphEdit>>,
    pub branches: Option<Vec<BranchEditSpec>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MutationDiffSummary {
    pub steps_delta: isize,
    pub layers_added: usize,
    pub output_slot_changed: bool,
    pub changed_weight_layers: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BranchReport {
    pub branch_id: String,
    pub valid: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_error: Option<String>,
    pub candidate_sha256: String,
    pub equivalent: bool,
    pub max_abs_error: f64,
    pub max_rel_error: f64,
    pub receipt_digest: String,
    pub diff: MutationDiffSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct MutateReport {
    pub ok: bool,
    pub mode: &'static str,
    pub commit_requested: bool,
    pub committed: bool,
    pub equivalent: bool,
    pub baseline_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub winning_branch: Option<String>,
    pub branches: Vec<BranchReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resulting_spec: Option<SpecV1>,
}

fn layer_is_binary(decl: &LayerDecl, slots_len: usize) -> Result<bool, String> {
    let meta = find_constructor(&decl.constructor)
        .ok_or_else(|| format!("unknown layer constructor '{}'", decl.constructor))?;
    match meta.arity {
        1 => Ok(false),
        2 => Ok(true),
        _ => Ok(slots_len == 3),
    }
}

pub fn apply_edits_to_spec(base: &SpecV1, edits: &[GraphEdit]) -> Result<(SpecV1, MutationDiffSummary), String> {
    let mut s = base.clone();
    let mut changed_weights = BTreeSet::new();

    for edit in edits {
        match edit {
            GraphEdit::SetWeights { layer_index, values } => {
                if *layer_index >= s.layers.len() {
                    return Err(format!("setWeights: invalid layer_index {layer_index}"));
                }
                for (i, &v) in values.iter().enumerate() {
                    if !v.is_finite() {
                        return Err(format!("setWeights[{layer_index}][{i}]: non-finite weight {v}"));
                    }
                }
                s.weights.insert(layer_index.to_string(), values.clone());
                changed_weights.insert(*layer_index);
            }
            GraphEdit::ReplaceStep { index, layer, slots } => {
                if *index >= s.steps.len() {
                    return Err(format!("replaceStep: index {index} out of bounds"));
                }
                let is_bin = layer_is_binary(layer, slots.len())?;
                let expected = if is_bin { 3 } else { 2 };
                if slots.len() < expected {
                    return Err(format!(
                        "replaceStep: expected at least {expected} slots for {}, got {}",
                        layer.constructor,
                        slots.len()
                    ));
                }
                let new_li = s.layers.len();
                s.layers.push(layer.clone());
                let used_slots = slots[..expected].to_vec();
                for &sl in &used_slots {
                    if sl + 1 > s.num_slots {
                        s.num_slots = sl + 1;
                    }
                }
                s.steps[*index] = StepDecl {
                    kind: if is_bin { "binary" } else { "unary" }.to_string(),
                    layer: new_li,
                    slots: used_slots,
                };
            }
            GraphEdit::InsertStep { index, layer, slots } => {
                if *index > s.steps.len() {
                    return Err(format!("insertStep: index {index} > steps len {}", s.steps.len()));
                }
                let is_bin = layer_is_binary(layer, slots.len())?;
                let expected = if is_bin { 3 } else { 2 };
                if slots.len() < expected {
                    return Err(format!(
                        "insertStep: expected at least {expected} slots for {}, got {}",
                        layer.constructor,
                        slots.len()
                    ));
                }
                let new_li = s.layers.len();
                s.layers.push(layer.clone());
                let used_slots = slots[..expected].to_vec();
                for &sl in &used_slots {
                    if sl + 1 > s.num_slots {
                        s.num_slots = sl + 1;
                    }
                }
                s.steps.insert(
                    *index,
                    StepDecl {
                        kind: if is_bin { "binary" } else { "unary" }.to_string(),
                        layer: new_li,
                        slots: used_slots,
                    },
                );
            }
            GraphEdit::RemoveStep { index } => {
                if *index >= s.steps.len() {
                    return Err(format!("removeStep: index {index} out of bounds"));
                }
                s.steps.remove(*index);
            }
            GraphEdit::ReconnectStep { index, slots } => {
                let st = s
                    .steps
                    .get_mut(*index)
                    .ok_or_else(|| format!("reconnectStep: index {index} out of bounds"))?;
                let expected = if st.kind == "binary" { 3 } else { 2 };
                if slots.len() < expected {
                    return Err(format!(
                        "reconnectStep: expected {expected} slots for {} step",
                        st.kind
                    ));
                }
                let used_slots = slots[..expected].to_vec();
                for &sl in &used_slots {
                    if sl + 1 > s.num_slots {
                        s.num_slots = sl + 1;
                    }
                }
                st.slots = used_slots;
            }
            GraphEdit::SetOutput { slot } => {
                if *slot + 1 > s.num_slots {
                    s.num_slots = *slot + 1;
                }
                s.output_slot = *slot;
            }
        }
    }

    let diff = MutationDiffSummary {
        steps_delta: (s.steps.len() as isize) - (base.steps.len() as isize),
        layers_added: s.layers.len().saturating_sub(base.layers.len()),
        output_slot_changed: s.output_slot != base.output_slot,
        changed_weight_layers: changed_weights.into_iter().collect(),
    };
    Ok((s, diff))
}

pub fn mutate_spec(
    base: &SpecV1,
    req: &MutationRequest,
    cases_override: Option<&[EvalCase]>,
    abs_tol: f64,
    rel_tol: f64,
) -> Result<MutateReport, String> {
    let base_val = validate_spec(base, false);
    if !base_val.valid {
        return Err(
            serde_json::to_string(&base_val)
                .unwrap_or_else(|_| format!("invalid baseline spec: {:?}", base_val.code)),
        );
    }

    let cases = match cases_override {
        Some(c) if !c.is_empty() => c.to_vec(),
        _ => default_cases_from_spec(base),
    };
    let baseline_sha256 = sha256_json(base);

    let (mode, branch_list): (&'static str, Vec<(String, Vec<GraphEdit>)>) =
        if let Some(branches) = &req.branches
            && !branches.is_empty()
        {
            (
                "branch_set",
                branches
                    .iter()
                    .map(|b| (b.branch_id.clone(), b.edits.clone()))
                    .collect(),
            )
        } else if let Some(edits) = &req.edits
            && !edits.is_empty()
        {
            ("single_transaction", vec![("main".to_string(), edits.clone())])
        } else {
            return Err("mutation JSON must contain non-empty 'edits' or 'branches'".into());
        };

    let mut reports = Vec::with_capacity(branch_list.len());
    let mut specs_by_id: Vec<(String, SpecV1)> = Vec::new();

    for (bid, edits) in &branch_list {
        if bid.trim().is_empty() {
            return Err("branch_id must be non-empty".into());
        }
        if edits.is_empty() {
            return Err(format!("branch '{bid}' has empty edits"));
        }
        let (cand, diff) = apply_edits_to_spec(base, edits)?;
        let cval = validate_spec(&cand, false);
        let cand_sha = sha256_json(&cand);

        if !cval.valid {
            reports.push(BranchReport {
                branch_id: bid.clone(),
                valid: false,
                validation_error: Some(cval.code.to_string()),
                candidate_sha256: cand_sha.clone(),
                equivalent: false,
                max_abs_error: f64::INFINITY,
                max_rel_error: f64::INFINITY,
                receipt_digest: sha256_json(&(baseline_sha256.clone(), cand_sha, false)),
                diff,
            });
            continue;
        }

        let ver = verify_specs(base, &cand, &cases, abs_tol, rel_tol);
        let receipt_digest = sha256_json(&(
            baseline_sha256.clone(),
            cand_sha.clone(),
            ver.equivalent,
            ver.max_abs_diff,
        ));
        if ver.equivalent {
            specs_by_id.push((bid.clone(), cand));
        }
        reports.push(BranchReport {
            branch_id: bid.clone(),
            valid: true,
            validation_error: None,
            candidate_sha256: cand_sha,
            equivalent: ver.equivalent,
            max_abs_error: ver.max_abs_diff,
            max_rel_error: ver.max_rel_diff,
            receipt_digest,
            diff,
        });
    }

    let mut passing: Vec<&BranchReport> = reports.iter().filter(|r| r.equivalent).collect();
    passing.sort_by(|a, b| a.max_abs_error.total_cmp(&b.max_abs_error));

    let winner_id = if let Some(pref) = &req.promote_branch {
        passing
            .iter()
            .find(|r| &r.branch_id == pref)
            .map(|r| r.branch_id.clone())
    } else {
        passing.first().map(|r| r.branch_id.clone())
    };

    let any_equiv = winner_id.is_some();
    let committed = req.commit && any_equiv;
    let ok = if req.commit { committed } else { any_equiv };
    let resulting_spec = winner_id
        .as_ref()
        .and_then(|wid| specs_by_id.into_iter().find(|(id, _)| id == wid).map(|(_, sp)| sp));

    Ok(MutateReport {
        ok,
        mode,
        commit_requested: req.commit,
        committed,
        equivalent: any_equiv,
        baseline_sha256,
        winning_branch: winner_id,
        branches: reports,
        resulting_spec,
    })
}
