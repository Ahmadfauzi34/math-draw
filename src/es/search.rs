use serde::Serialize;
use std::collections::{HashSet, VecDeque};

use crate::catalog::expected_weight_count;
use crate::quant::{QuantScheme, quant_verify};
use crate::rewrite::{SIMPLIFY_ORDER, apply_rewrite};
use crate::spec::{SpecV1, sha256_json};
use crate::validate::validate_spec;
use crate::verify::{EvalCase, default_cases_from_spec, verify_specs};

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub max_depth: usize,
    pub abs_tol: f64,
    pub rel_tol: f64,
    pub quant_tol: Option<f64>,
    pub quant_scheme: QuantScheme,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            max_depth: 3,
            abs_tol: 1e-5,
            rel_tol: 1e-5,
            quant_tol: None,
            quant_scheme: QuantScheme::Q8PerBlock,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchNodeSummary {
    pub id: String,
    pub depth: usize,
    pub parent_id: Option<String>,
    pub transform_chain: Vec<String>,
    pub steps_count: usize,
    pub active_params: usize,
    pub active_weight_bytes: usize,
    pub equivalent: bool,
    pub max_abs_diff: f64,
    pub state_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub ok: bool,
    pub baseline_sha256: String,
    pub best_node_id: String,
    pub best_sha256: String,
    pub baseline_steps: usize,
    pub best_steps: usize,
    pub baseline_params: usize,
    pub best_params: usize,
    pub baseline_weight_bytes: usize,
    pub best_weight_bytes: usize,
    pub visited_nodes: usize,
    pub transform_chain: Vec<String>,
    pub nodes: Vec<SearchNodeSummary>,
    pub best_spec: SpecV1,
}

fn count_active_params(spec: &SpecV1) -> usize {
    let used: HashSet<usize> = spec.steps.iter().map(|s| s.layer).collect();
    used.into_iter()
        .filter_map(|li| spec.layers.get(li))
        .filter_map(|decl| expected_weight_count(&decl.constructor, &decl.args))
        .sum()
}

pub fn search_graph_tree(
    base: &SpecV1,
    max_depth: usize,
    cases_override: Option<&[EvalCase]>,
    abs_tol: f64,
    rel_tol: f64,
) -> Result<SearchResult, String> {
    search_graph_tree_with_options(
        base,
        cases_override,
        &SearchOptions {
            max_depth,
            abs_tol,
            rel_tol,
            quant_tol: None,
            quant_scheme: QuantScheme::Q8PerBlock,
        },
    )
}

pub fn search_graph_tree_with_options(
    base: &SpecV1,
    cases_override: Option<&[EvalCase]>,
    opts: &SearchOptions,
) -> Result<SearchResult, String> {
    if !(opts.abs_tol.is_finite() && opts.abs_tol >= 0.0) {
        return Err(format!(
            "search_graph_tree: abs_tol must be finite and >= 0, got {}",
            opts.abs_tol
        ));
    }
    if !(opts.rel_tol.is_finite() && opts.rel_tol >= 0.0) {
        return Err(format!(
            "search_graph_tree: rel_tol must be finite and >= 0, got {}",
            opts.rel_tol
        ));
    }
    if let Some(qt) = opts.quant_tol
        && !(qt.is_finite() && qt >= 0.0)
    {
        return Err(format!(
            "search_graph_tree: quant_tol must be finite and >= 0, got {qt}"
        ));
    }
    let v = validate_spec(base, false);
    if !v.valid {
        return Err(format!("baseline spec invalid: {:?}", v.code));
    }

    let eval_cases: Vec<EvalCase> = match cases_override {
        Some(c) if !c.is_empty() => c.to_vec(),
        _ => default_cases_from_spec(base),
    };

    let base_sha = sha256_json(base);
    let base_steps = base.steps.len();
    let base_params = count_active_params(base);
    let base_bytes = base_params * 4;

    let mut nodes = vec![SearchNodeSummary {
        id: "root".to_string(),
        depth: 0,
        parent_id: None,
        transform_chain: vec![],
        steps_count: base_steps,
        active_params: base_params,
        active_weight_bytes: base_bytes,
        equivalent: true,
        max_abs_diff: 0.0,
        state_sha256: base_sha.clone(),
    }];

    let mut best_id = "root".to_string();
    let mut best_sha = base_sha.clone();
    let mut best_spec = base.clone();
    let mut best_steps = base_steps;
    let mut best_params = base_params;
    let mut best_bytes = base_bytes;
    let mut best_chain = Vec::new();

    let mut seen_shas = HashSet::from([base_sha.clone()]);
    let mut queue: VecDeque<(String, usize, Vec<String>, SpecV1, bool)> = VecDeque::new();
    queue.push_back(("root".to_string(), 0, vec![], base.clone(), false));

    let max_total_nodes = 64usize;
    let mut counter = 0usize;

    while let Some((parent_id, depth, chain, cur_spec, already_quantized)) = queue.pop_front() {
        if depth >= opts.max_depth || nodes.len() >= max_total_nodes {
            continue;
        }
        for &rule in SIMPLIFY_ORDER {
            if nodes.len() >= max_total_nodes {
                break;
            }
            let Ok(cand) = apply_rewrite(rule, &cur_spec) else {
                continue;
            };
            let cand_sha = sha256_json(&cand);
            if !seen_shas.insert(cand_sha.clone()) {
                continue;
            }
            let cand_valid = validate_spec(&cand, false);
            if !cand_valid.valid {
                continue;
            }
            let ver = verify_specs(base, &cand, &eval_cases, opts.abs_tol, opts.rel_tol);
            counter += 1;
            let node_id = format!("n{counter}");
            let mut next_chain = chain.clone();
            next_chain.push(rule.to_string());

            let steps_cnt = cand.steps.len();
            let params_cnt = count_active_params(&cand);
            let weight_bytes = params_cnt * 4;

            nodes.push(SearchNodeSummary {
                id: node_id.clone(),
                depth: depth + 1,
                parent_id: Some(parent_id.clone()),
                transform_chain: next_chain.clone(),
                steps_count: steps_cnt,
                active_params: params_cnt,
                active_weight_bytes: weight_bytes,
                equivalent: ver.equivalent,
                max_abs_diff: ver.max_abs_diff,
                state_sha256: cand_sha.clone(),
            });

            if ver.equivalent {
                let better = steps_cnt < best_steps
                    || (steps_cnt == best_steps && weight_bytes < best_bytes)
                    || (steps_cnt == best_steps
                        && weight_bytes == best_bytes
                        && params_cnt < best_params);
                if better {
                    best_id = node_id.clone();
                    best_sha = cand_sha;
                    best_spec = cand.clone();
                    best_steps = steps_cnt;
                    best_params = params_cnt;
                    best_bytes = weight_bytes;
                    best_chain = next_chain.clone();
                }
                queue.push_back((node_id, depth + 1, next_chain, cand, already_quantized));
            }
        }

        // Optional verified quantization branch when --quant-tol is enabled
        if let Some(qt) = opts.quant_tol
            && !already_quantized
            && nodes.len() < max_total_nodes
            && count_active_params(&cur_spec) > 0
            && let Ok(qrep) = quant_verify(&cur_spec, opts.quant_scheme, Some(&eval_cases), qt, qt)
        {
            let cand = qrep.quantized_spec;
            let cand_sha = sha256_json(&cand);
            if seen_shas.insert(cand_sha.clone()) {
                let ver = verify_specs(base, &cand, &eval_cases, qt, qt);
                counter += 1;
                let node_id = format!("n{counter}");
                let mut next_chain = chain.clone();
                next_chain.push(format!("quant:{}", opts.quant_scheme.as_str()));
                let steps_cnt = cand.steps.len();
                let params_cnt = count_active_params(&cand);
                let weight_bytes = qrep.totals.bytes_quant;

                nodes.push(SearchNodeSummary {
                    id: node_id.clone(),
                    depth: depth + 1,
                    parent_id: Some(parent_id.clone()),
                    transform_chain: next_chain.clone(),
                    steps_count: steps_cnt,
                    active_params: params_cnt,
                    active_weight_bytes: weight_bytes,
                    equivalent: ver.equivalent,
                    max_abs_diff: ver.max_abs_diff,
                    state_sha256: cand_sha.clone(),
                });

                if ver.equivalent {
                    let better = steps_cnt < best_steps
                        || (steps_cnt == best_steps && weight_bytes < best_bytes)
                        || (steps_cnt == best_steps
                            && weight_bytes == best_bytes
                            && params_cnt < best_params);
                    if better {
                        best_id = node_id.clone();
                        best_sha = cand_sha;
                        best_spec = cand.clone();
                        best_steps = steps_cnt;
                        best_params = params_cnt;
                        best_bytes = weight_bytes;
                        best_chain = next_chain.clone();
                    }
                    queue.push_back((node_id, depth + 1, next_chain, cand, true));
                }
            }
        }
    }

    Ok(SearchResult {
        ok: true,
        baseline_sha256: base_sha,
        best_node_id: best_id,
        best_sha256: best_sha,
        baseline_steps: base_steps,
        best_steps,
        baseline_params: base_params,
        best_params,
        baseline_weight_bytes: base_bytes,
        best_weight_bytes: best_bytes,
        visited_nodes: nodes.len(),
        transform_chain: best_chain,
        nodes,
        best_spec,
    })
}
