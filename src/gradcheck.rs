use burn::tensor::Device;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

use crate::engine::{PreparedGraph, SlotValue, eval_spec_with_inputs, execute_forward, port_is_i64_mask};
use crate::recurse::{apply_flat_weights, discover_trainable_slices};
use crate::spec::{SpecV1, parse_f32_values};
use crate::validate::validate_spec;

#[derive(Debug, Clone, Serialize)]
pub struct JacobianMatrix {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SlotComparison {
    pub max_abs_error: f64,
    pub max_rel_error: f64,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct AutodiffCheckReport {
    pub ok: bool,
    pub checked_weights: usize,
    pub max_abs_error: f64,
    pub max_rel_error: f64,
    pub layers: BTreeMap<String, SlotComparison>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GradcheckReport {
    pub ok: bool,
    pub eps: f64,
    pub tol: f64,
    pub out_len: usize,
    pub input_jacobians: BTreeMap<String, JacobianMatrix>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub weight_jacobians: BTreeMap<String, JacobianMatrix>,
    pub compared: bool,
    pub passed: bool,
    pub max_abs_error: f64,
    pub max_rel_error: f64,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub per_input_comparison: BTreeMap<String, SlotComparison>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub per_weight_comparison: BTreeMap<String, SlotComparison>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub autodiff_check: Option<AutodiffCheckReport>,
}

fn parse_analytic_matrix(val: &Value, rows: usize, cols: usize, label: &str) -> Result<Vec<Vec<f64>>, String> {
    let outer = val
        .as_array()
        .ok_or_else(|| format!("analytic[{label}] must be a {rows}x{cols} 2D array"))?;
    if outer.len() != rows {
        return Err(format!(
            "analytic[{label}] row count mismatch: expected {rows}, got {}",
            outer.len()
        ));
    }
    let mut mat = Vec::with_capacity(rows);
    for (r, row_val) in outer.iter().enumerate() {
        let row_arr = row_val
            .as_array()
            .ok_or_else(|| format!("analytic[{label}][{r}] must be an array"))?;
        if row_arr.len() != cols {
            return Err(format!(
                "analytic[{label}][{r}] col count mismatch: expected {cols}, got {}",
                row_arr.len()
            ));
        }
        let mut row_f = Vec::with_capacity(cols);
        for (c, cell) in row_arr.iter().enumerate() {
            let num = cell
                .as_f64()
                .ok_or_else(|| format!("analytic[{label}][{r}][{c}] must be finite number"))?;
            if !num.is_finite() {
                return Err(format!("analytic[{label}][{r}][{c}] is non-finite"));
            }
            row_f.push(num);
        }
        mat.push(row_f);
    }
    Ok(mat)
}

fn compare_matrices(numeric: &[Vec<f64>], analytic: &[Vec<f64>], tol: f64) -> SlotComparison {
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    for (nr, ar) in numeric.iter().zip(analytic.iter()) {
        for (&nv, &av) in nr.iter().zip(ar.iter()) {
            let abs = (nv - av).abs();
            let denom = av.abs().max(1e-12);
            let rel = abs / denom;
            if abs > max_abs {
                max_abs = abs;
            }
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }
    SlotComparison {
        max_abs_error: max_abs,
        max_rel_error: max_rel,
        passed: max_abs <= tol || max_rel <= tol,
    }
}

pub fn gradcheck_spec(
    spec: &SpecV1,
    analytic: Option<&Value>,
    eps: f64,
    tol: f64,
    include_weights: bool,
    check_autodiff: bool,
) -> Result<GradcheckReport, String> {
    if !(eps.is_finite() && eps > 0.0) || !(tol.is_finite() && tol > 0.0) {
        return Err("gradcheck requires positive finite eps and tol".into());
    }
    let val = validate_spec(spec, false);
    if !val.valid {
        return Err(
            serde_json::to_string(&val).unwrap_or_else(|_| format!("invalid spec: {:?}", val.code)),
        );
    }

    let graph = PreparedGraph::compile(spec)?;
    let base_out = graph.run_once(spec, None)?;
    let out_len = base_out.values.len();
    let i64_mask = port_is_i64_mask(spec);

    let want_weights = include_weights
        || check_autodiff
        || analytic
            .and_then(|v| v.get("weights"))
            .and_then(|v| v.as_object())
            .is_some();

    let mut input_jacobians = BTreeMap::new();
    let mut per_input_comparison = BTreeMap::new();
    let mut compared = false;
    let mut passed = true;
    let mut global_max_abs = 0.0f64;
    let mut global_max_rel = 0.0f64;

    for port in &spec.ports {
        if port.slot < i64_mask.len() && i64_mask[port.slot] {
            continue;
        }
        let slot_key = port.slot.to_string();
        let vals = parse_f32_values(
            spec.inputs
                .get(&slot_key)
                .ok_or_else(|| format!("missing inputs for slot {slot_key}"))?,
            &format!("inputs[{slot_key}]"),
        )?;
        let in_len = vals.len();
        let mut mat = vec![vec![0.0f64; in_len]; out_len];

        for j in 0..in_len {
            let mut plus_ov = spec.inputs.clone();
            let mut minus_ov = spec.inputs.clone();
            let mut p_vals = vals.clone();
            let mut m_vals = vals.clone();
            p_vals[j] = ((vals[j] as f64) + eps) as f32;
            m_vals[j] = ((vals[j] as f64) - eps) as f32;
            plus_ov.insert(
                slot_key.clone(),
                p_vals.iter().map(|&v| Value::from(v as f64)).collect(),
            );
            minus_ov.insert(
                slot_key.clone(),
                m_vals.iter().map(|&v| Value::from(v as f64)).collect(),
            );
            let fp = graph.run_once(spec, Some(&plus_ov))?;
            let fm = graph.run_once(spec, Some(&minus_ov))?;
            for i in 0..out_len {
                mat[i][j] = ((fp.values[i] as f64) - (fm.values[i] as f64)) / (2.0 * eps);
            }
        }

        if let Some(an_root) = analytic {
            let maybe_mat = an_root
                .get("inputs")
                .and_then(|m| m.get(&slot_key))
                .or_else(|| an_root.get(&slot_key));
            if let Some(mval) = maybe_mat {
                compared = true;
                let parsed = parse_analytic_matrix(mval, out_len, in_len, &slot_key)?;
                let cmp = compare_matrices(&mat, &parsed, tol);
                if !cmp.passed {
                    passed = false;
                }
                global_max_abs = global_max_abs.max(cmp.max_abs_error);
                global_max_rel = global_max_rel.max(cmp.max_rel_error);
                per_input_comparison.insert(slot_key.clone(), cmp);
            }
        }

        input_jacobians.insert(
            slot_key,
            JacobianMatrix {
                rows: out_len,
                cols: in_len,
                values: mat,
            },
        );
    }

    let mut weight_jacobians = BTreeMap::new();
    let mut per_weight_comparison = BTreeMap::new();
    let mut autodiff_check = None;

    if want_weights {
        let (slices, flat_w) = discover_trainable_slices(spec, None)?;
        let mut work_w = flat_w.clone();

        for sl in &slices {
            let l_key = sl.layer_index.to_string();
            let mut mat = vec![vec![0.0f64; sl.length]; out_len];

            for local_j in 0..sl.length {
                let d = sl.offset + local_j;
                let orig = flat_w[d];
                work_w[d] = ((orig as f64) + eps) as f32;
                let sp_p = apply_flat_weights(spec, &slices, &work_w);
                let fp = eval_spec_with_inputs(&sp_p, None)?;

                work_w[d] = ((orig as f64) - eps) as f32;
                let sp_m = apply_flat_weights(spec, &slices, &work_w);
                let fm = eval_spec_with_inputs(&sp_m, None)?;
                work_w[d] = orig;

                for i in 0..out_len {
                    mat[i][local_j] = ((fp.values[i] as f64) - (fm.values[i] as f64)) / (2.0 * eps);
                }
            }

            if let Some(an_root) = analytic
                && let Some(w_val) = an_root.get("weights").and_then(|w| w.get(&l_key))
            {
                compared = true;
                let parsed = parse_analytic_matrix(w_val, out_len, sl.length, &format!("weights.{l_key}"))?;
                let cmp = compare_matrices(&mat, &parsed, tol);
                if !cmp.passed {
                    passed = false;
                }
                global_max_abs = global_max_abs.max(cmp.max_abs_error);
                global_max_rel = global_max_rel.max(cmp.max_rel_error);
                per_weight_comparison.insert(l_key.clone(), cmp);
            }

            weight_jacobians.insert(
                l_key,
                JacobianMatrix {
                    rows: out_len,
                    cols: sl.length,
                    values: mat,
                },
            );
        }

        if check_autodiff {
            let device = Device::flex().autodiff();
            let grad_layers: Vec<usize> = slices.iter().map(|s| s.layer_index).collect();
            let (slot_val, grad_params) = execute_forward(spec, &device, &grad_layers, None)?;
            let y = match slot_val {
                SlotValue::F32(t) => t,
                SlotValue::I64(..) => {
                    return Err("gradcheck --autodiff requires f32 graph output".into());
                }
            };
            let grads = y.sum().backward();
            let mut ad_grad = vec![0.0f64; flat_w.len()];
            let slice_map: BTreeMap<usize, usize> =
                slices.iter().map(|s| (s.layer_index, s.offset)).collect();
            for gp in grad_params {
                let gv = gp.tensor.extract_grad(&grads)?;
                if let Some(&base_off) = slice_map.get(&gp.layer_idx) {
                    for (i, &val) in gv.iter().enumerate() {
                        ad_grad[base_off + gp.offset + i] += val as f64;
                    }
                }
            }

            let mut ad_layers = BTreeMap::new();
            let mut ad_max_abs = 0.0f64;
            let mut ad_max_rel = 0.0f64;
            let mut ad_ok = true;
            for sl in &slices {
                let l_key = sl.layer_index.to_string();
                let jac = &weight_jacobians[&l_key].values;
                let mut num_sum = vec![0.0f64; sl.length];
                for j in 0..sl.length {
                    let mut s = 0.0f64;
                    for row in jac {
                        s += row[j];
                    }
                    num_sum[j] = s;
                }
                let ad_slice = &ad_grad[sl.offset..sl.offset + sl.length];
                let cmp = compare_matrices(&[num_sum], &[ad_slice.to_vec()], tol);
                if !cmp.passed {
                    ad_ok = false;
                }
                ad_max_abs = ad_max_abs.max(cmp.max_abs_error);
                ad_max_rel = ad_max_rel.max(cmp.max_rel_error);
                ad_layers.insert(l_key, cmp);
            }
            if !ad_ok {
                passed = false;
            }
            global_max_abs = global_max_abs.max(ad_max_abs);
            global_max_rel = global_max_rel.max(ad_max_rel);
            compared = true;
            autodiff_check = Some(AutodiffCheckReport {
                ok: ad_ok,
                checked_weights: flat_w.len(),
                max_abs_error: ad_max_abs,
                max_rel_error: ad_max_rel,
                layers: ad_layers,
            });
        }
    }

    Ok(GradcheckReport {
        ok: passed,
        eps,
        tol,
        out_len,
        input_jacobians,
        weight_jacobians,
        compared,
        passed,
        max_abs_error: global_max_abs,
        max_rel_error: global_max_rel,
        per_input_comparison,
        per_weight_comparison,
        autodiff_check,
    })
}
