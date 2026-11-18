use serde::Serialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

use crate::spec::{PortDecl, SpecV1, parse_f32_values};
use crate::weight_lab::{mat_mul, mat_vec_mul, weight_fingerprint};

#[derive(Debug, Clone)]
pub struct EffectiveAffine {
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    pub in_dim: usize,
    pub out_dim: usize,
    pub chain: Vec<usize>,
    pub by_port: BTreeMap<usize, Vec<Vec<f64>>>,
    pub primary_origin: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct LinearEquivReport {
    pub ok: bool,
    pub verdict: &'static str,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dims: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_ports: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conditional_on_context: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_ports_folded: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chain_a: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chain_b: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_diff: Option<Value>,
}

fn kernel_to_rows(flat: &[f32], in_dim: usize, out_dim: usize) -> Vec<Vec<f64>> {
    let mut w = Vec::with_capacity(out_dim);
    for i in 0..out_dim {
        let mut row = Vec::with_capacity(in_dim);
        for k in 0..in_dim {
            row.push(flat[k * out_dim + i] as f64);
        }
        w.push(row);
    }
    w
}

fn identity_rows(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

fn zero_mat(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    vec![vec![0.0; cols]; rows]
}

fn is_zero_mat(w: &[Vec<f64>], tol: f64) -> bool {
    w.iter().all(|row| row.iter().all(|&v| v.abs() <= tol))
}

fn add_mat(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(&x, &y)| x + y).collect())
        .collect()
}

fn sub_mat(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(&x, &y)| x - y).collect())
        .collect()
}

fn add_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

fn sub_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

fn compute_live_step_indices(spec: &SpecV1) -> BTreeSet<usize> {
    let mut live_slots = BTreeSet::new();
    live_slots.insert(spec.output_slot);
    let mut live_steps = BTreeSet::new();
    for (si, st) in spec.steps.iter().enumerate().rev() {
        let Some(&out_slot) = st.slots.last() else {
            continue;
        };
        if live_slots.contains(&out_slot) {
            live_steps.insert(si);
            let in_count = st.expected_arity();
            for &in_s in st.slots.iter().take(in_count) {
                live_slots.insert(in_s);
            }
        }
    }
    live_steps
}

#[derive(Clone)]
struct SlotAffine {
    out_dim: usize,
    by_port: BTreeMap<usize, Vec<Vec<f64>>>,
    b: Vec<f64>,
    primary_origin: usize,
}

pub fn effective_affine(spec: &SpecV1) -> Result<EffectiveAffine, String> {
    let ports: BTreeMap<usize, &PortDecl> = spec.ports.iter().map(|p| (p.slot, p)).collect();
    let live_steps = compute_live_step_indices(spec);
    let mut aff: BTreeMap<usize, SlotAffine> = BTreeMap::new();
    let mut chain = Vec::new();

    let get_or_init = |aff_map: &BTreeMap<usize, SlotAffine>,
                       slot: usize,
                       expected_dim: Option<usize>|
     -> Result<SlotAffine, String> {
        if let Some(existing) = aff_map.get(&slot) {
            return Ok(existing.clone());
        }
        let Some(p) = ports.get(&slot) else {
            return Err(format!(
                "linear chain reads slot {slot} with no affine state and no declared input port"
            ));
        };
        let in_dim = p.shape[1];
        if let Some(exp) = expected_dim
            && in_dim != exp
        {
            return Err(format!("input port dim {in_dim} != layer in_dim {exp}"));
        }
        let mut by_port = BTreeMap::new();
        by_port.insert(slot, identity_rows(in_dim));
        Ok(SlotAffine {
            out_dim: in_dim,
            by_port,
            b: vec![0.0; in_dim],
            primary_origin: slot,
        })
    };

    for (si, st) in spec.steps.iter().enumerate() {
        if !live_steps.contains(&si) {
            continue;
        }
        let Some(decl) = spec.layers.get(st.layer) else {
            return Err(format!("step {si} references invalid layer {}", st.layer));
        };
        if st.kind == "unary" {
            if decl.constructor != "linear" {
                return Err(format!(
                    "unary step {si} references non-linear layer {} ('{}')",
                    st.layer, decl.constructor
                ));
            }
            let in_dim = decl.args.get(1).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let out_dim = decl.args.get(2).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let bias = decl.args.get(3).and_then(|v| v.as_bool()).unwrap_or(false);
            let flat = spec
                .weights
                .get(&st.layer.to_string())
                .ok_or_else(|| format!("linear layer {} has no weights in spec", st.layer))?;
            let need = in_dim * out_dim + if bias { out_dim } else { 0 };
            if flat.len() < need {
                return Err(format!(
                    "linear layer {}: weights length {} < {need}",
                    st.layer,
                    flat.len()
                ));
            }
            let lw = kernel_to_rows(flat, in_dim, out_dim);
            let lb: Vec<f64> = if bias {
                flat[in_dim * out_dim..in_dim * out_dim + out_dim]
                    .iter()
                    .map(|&v| v as f64)
                    .collect()
            } else {
                vec![0.0; out_dim]
            };

            let in_slot = st.slots[0];
            let out_slot = st.slots[1];
            let prev = get_or_init(&aff, in_slot, Some(in_dim))?;
            if prev.out_dim != in_dim {
                return Err(format!(
                    "dim mismatch into linear layer {}: {} vs {in_dim}",
                    st.layer, prev.out_dim
                ));
            }
            let mut next_by_port = BTreeMap::new();
            for (&p_slot, wp) in &prev.by_port {
                next_by_port.insert(p_slot, mat_mul(&lw, wp)?);
            }
            let w_b = mat_vec_mul(&lw, &prev.b)?;
            let next_b = add_vec(&w_b, &lb);
            aff.insert(
                out_slot,
                SlotAffine {
                    out_dim,
                    by_port: next_by_port,
                    b: next_b,
                    primary_origin: prev.primary_origin,
                },
            );
            chain.push(st.layer);
        } else if st.kind == "binary" {
            let ctor = decl.constructor.as_str();
            let a_slot = st.slots[0];
            let b_slot = st.slots[1];
            let out_slot = st.slots[2];

            if ctor == "add" || ctor == "sub" {
                let sa = get_or_init(&aff, a_slot, None)?;
                let sb = get_or_init(&aff, b_slot, None)?;
                if sa.out_dim != sb.out_dim {
                    return Err(format!("{ctor} dim mismatch {} vs {}", sa.out_dim, sb.out_dim));
                }
                let out_dim = sa.out_dim;
                let all_ports: BTreeSet<usize> =
                    sa.by_port.keys().chain(sb.by_port.keys()).copied().collect();
                let mut next_by_port = BTreeMap::new();
                for p_slot in all_ports {
                    let p_dim = ports
                        .get(&p_slot)
                        .map(|p| p.shape[1])
                        .or_else(|| sa.by_port.get(&p_slot).map(|m| m[0].len()))
                        .or_else(|| sb.by_port.get(&p_slot).map(|m| m[0].len()))
                        .unwrap_or(0);
                    let wa = sa
                        .by_port
                        .get(&p_slot)
                        .cloned()
                        .unwrap_or_else(|| zero_mat(out_dim, p_dim));
                    let wb = sb
                        .by_port
                        .get(&p_slot)
                        .cloned()
                        .unwrap_or_else(|| zero_mat(out_dim, p_dim));
                    next_by_port.insert(
                        p_slot,
                        if ctor == "add" {
                            add_mat(&wa, &wb)
                        } else {
                            sub_mat(&wa, &wb)
                        },
                    );
                }
                let next_b = if ctor == "add" {
                    add_vec(&sa.b, &sb.b)
                } else {
                    sub_vec(&sa.b, &sb.b)
                };
                aff.insert(
                    out_slot,
                    SlotAffine {
                        out_dim,
                        by_port: next_by_port,
                        b: next_b,
                        primary_origin: sa.primary_origin,
                    },
                );
                chain.push(st.layer);
            } else if ctor == "concat" {
                let dim = decl.args.get(1).and_then(|v| v.as_u64()).unwrap_or(0);
                if dim != 1 {
                    return Err(format!(
                        "concat dim {dim} unsupported: only dim=1 (feature axis) preserves affinity"
                    ));
                }
                let Some(sa) = aff.get(&a_slot).cloned() else {
                    return Err("concat reads left slot with no affine map".into());
                };
                let Some(sb) = aff.get(&b_slot).cloned() else {
                    return Err("concat reads right slot with no affine map".into());
                };
                if sa.primary_origin != sb.primary_origin {
                    return Err(format!(
                        "concat of affines from different primary inputs (slots {} vs {})",
                        sa.primary_origin, sb.primary_origin
                    ));
                }
                let all_ports: BTreeSet<usize> =
                    sa.by_port.keys().chain(sb.by_port.keys()).copied().collect();
                let mut next_by_port = BTreeMap::new();
                for p_slot in all_ports {
                    let p_dim = ports
                        .get(&p_slot)
                        .map(|p| p.shape[1])
                        .or_else(|| sa.by_port.get(&p_slot).map(|m| m[0].len()))
                        .or_else(|| sb.by_port.get(&p_slot).map(|m| m[0].len()))
                        .unwrap_or(0);
                    let mut wa = sa
                        .by_port
                        .get(&p_slot)
                        .cloned()
                        .unwrap_or_else(|| zero_mat(sa.out_dim, p_dim));
                    let wb = sb
                        .by_port
                        .get(&p_slot)
                        .cloned()
                        .unwrap_or_else(|| zero_mat(sb.out_dim, p_dim));
                    wa.extend(wb);
                    next_by_port.insert(p_slot, wa);
                }
                let mut next_b = sa.b;
                next_b.extend(sb.b);
                aff.insert(
                    out_slot,
                    SlotAffine {
                        out_dim: sa.out_dim + sb.out_dim,
                        by_port: next_by_port,
                        b: next_b,
                        primary_origin: sa.primary_origin,
                    },
                );
                chain.push(st.layer);
            } else {
                return Err(format!(
                    "binary step on layer {} is '{ctor}', only add, sub, and concat(dim=1) are supported",
                    st.layer
                ));
            }
        } else {
            return Err(format!("unsupported step kind '{}'", st.kind));
        }
    }

    let Some(out) = aff.remove(&spec.output_slot) else {
        return Err(format!(
            "outputSlot {} is not the end of a linear/affine chain",
            spec.output_slot
        ));
    };
    let primary_w = out
        .by_port
        .get(&out.primary_origin)
        .cloned()
        .or_else(|| out.by_port.values().next().cloned())
        .ok_or_else(|| "no active input port in affine chain".to_string())?;
    let in_dim = primary_w.first().map(|r| r.len()).unwrap_or(0);

    Ok(EffectiveAffine {
        w: primary_w,
        b: out.b,
        in_dim,
        out_dim: out.out_dim,
        chain,
        by_port: out.by_port,
        primary_origin: out.primary_origin,
    })
}

fn fold_extra_ports(
    aff: &EffectiveAffine,
    spec: &SpecV1,
    other_spec: &SpecV1,
    shared_ports: &BTreeSet<usize>,
    tol: f64,
) -> Result<(Vec<f64>, Vec<usize>), String> {
    let mut b = aff.b.clone();
    let mut folded = Vec::new();
    let ports: BTreeMap<usize, &PortDecl> = spec.ports.iter().map(|p| (p.slot, p)).collect();
    let other_ports: BTreeMap<usize, &PortDecl> =
        other_spec.ports.iter().map(|p| (p.slot, p)).collect();

    for (&p_slot, wp) in &aff.by_port {
        if shared_ports.contains(&p_slot) || is_zero_mat(wp, tol) {
            continue;
        }
        let p_def = ports.get(&p_slot);
        let o_def = other_ports.get(&p_slot);
        let is_ctx = p_def.map(|p| p.role == "context").unwrap_or(false)
            && o_def.map(|p| p.role == "context").unwrap_or(true);
        if !is_ctx {
            let role = p_def.map(|p| p.role.as_str()).unwrap_or("unknown");
            return Err(format!(
                "port {p_slot} (role '{role}') has non-zero linear dependency in one graph but is absent in the other (only role='context' ports may be folded into constant bias)"
            ));
        }
        let raw_in = spec
            .inputs
            .get(&p_slot.to_string())
            .ok_or_else(|| format!("context port {p_slot} has no inputs array to fold"))?;
        let in_f32 = parse_f32_values(raw_in, &format!("inputs[{p_slot}]"))?;
        let in_vals: Vec<f64> = in_f32.into_iter().map(|v| v as f64).collect();
        if in_vals.len() != wp[0].len() {
            return Err(format!(
                "context port {p_slot} input length {} != expected {}",
                in_vals.len(),
                wp[0].len()
            ));
        }
        if let Some(other_raw) = other_spec.inputs.get(&p_slot.to_string())
            && let Ok(other_f32) = parse_f32_values(other_raw, "other_inputs")
            && other_f32.len() == in_vals.len()
        {
            for (idx, (&va, &vb)) in in_vals.iter().zip(other_f32.iter()).enumerate() {
                if (va - (vb as f64)).abs() > tol {
                    return Err(format!(
                        "context port {p_slot} input value at index {idx} differs between specA ({va}) and specB ({vb})"
                    ));
                }
            }
        }
        let contrib = mat_vec_mul(wp, &in_vals)?;
        b = add_vec(&b, &contrib);
        folded.push(p_slot);
    }
    Ok((b, folded))
}

pub fn equiv_linear(spec_a: &SpecV1, spec_b: &SpecV1, tol: f64) -> LinearEquivReport {
    let reject = |reason: String, first_diff: Option<Value>| LinearEquivReport {
        ok: false,
        verdict: "REJECT",
        reason,
        effective_fingerprint: None,
        dims: None,
        active_ports: None,
        conditional_on_context: None,
        context_ports_folded: None,
        chain_a: None,
        chain_b: None,
        first_diff,
    };

    let a = match effective_affine(spec_a) {
        Ok(v) => v,
        Err(e) => return reject(format!("specA: {e}"), None),
    };
    let b = match effective_affine(spec_b) {
        Ok(v) => v,
        Err(e) => return reject(format!("specB: {e}"), None),
    };

    if a.out_dim != b.out_dim {
        return reject(
            format!(
                "dim mismatch A({}->{}) vs B({}->{})",
                a.in_dim, a.out_dim, b.in_dim, b.out_dim
            ),
            None,
        );
    }

    let ports_a: Vec<usize> = a.by_port.keys().copied().collect();
    let ports_b: Vec<usize> = b.by_port.keys().copied().collect();
    let shared_ports: BTreeSet<usize> = ports_a
        .iter()
        .filter(|p| b.by_port.contains_key(p))
        .copied()
        .collect();

    let mut b_a = a.b.clone();
    let mut b_b = b.b.clone();
    let mut folded_info: Option<Value> = None;

    if ports_a != ports_b {
        if shared_ports.is_empty() {
            return reject(
                format!("disjoint input ports A({ports_a:?}) vs B({ports_b:?})"),
                None,
            );
        }
        let (fa_b, fa_ports) = match fold_extra_ports(&a, spec_a, spec_b, &shared_ports, tol) {
            Ok(res) => res,
            Err(e) => {
                return reject(
                    format!("active port set mismatch A({ports_a:?}) vs B({ports_b:?}): {e}"),
                    None,
                )
            }
        };
        let (fb_b, fb_ports) = match fold_extra_ports(&b, spec_b, spec_a, &shared_ports, tol) {
            Ok(res) => res,
            Err(e) => {
                return reject(
                    format!("active port set mismatch A({ports_a:?}) vs B({ports_b:?}): {e}"),
                    None,
                )
            }
        };
        b_a = fa_b;
        b_b = fb_b;
        if !fa_ports.is_empty() || !fb_ports.is_empty() {
            folded_info = Some(json!({
                "folded_in_A": fa_ports,
                "folded_in_B": fb_ports,
            }));
        }
    }

    for &p_slot in &shared_ports {
        let wa = &a.by_port[&p_slot];
        let wb = &b.by_port[&p_slot];
        if wa[0].len() != wb[0].len() {
            return reject(
                format!(
                    "dim mismatch on port {p_slot}: A({}->{}) vs B({}->{})",
                    wa[0].len(),
                    a.out_dim,
                    wb[0].len(),
                    b.out_dim
                ),
                None,
            );
        }
        for i in 0..a.out_dim {
            for k in 0..wa[0].len() {
                if (wa[i][k] - wb[i][k]).abs() > tol {
                    return reject(
                        format!("W[{i}][{k}] (port {p_slot}) differs"),
                        Some(json!({
                            "kind": "W",
                            "port": p_slot,
                            "i": i,
                            "k": k,
                            "a": wa[i][k],
                            "b": wb[i][k],
                        })),
                    );
                }
            }
        }
    }

    for i in 0..a.out_dim {
        if (b_a[i] - b_b[i]).abs() > tol {
            return reject(
                format!("b[{i}] differs"),
                Some(json!({
                    "kind": "b",
                    "i": i,
                    "a": b_a[i],
                    "b": b_b[i],
                })),
            );
        }
    }

    let fp = format!("sha256:{}", weight_fingerprint(&a.w, &b_a));
    LinearEquivReport {
        ok: true,
        verdict: "ACCEPT",
        reason: format!(
            "exact affine map {}->{} over {} linear/affine steps",
            a.in_dim,
            a.out_dim,
            a.chain.len() + b.chain.len()
        ),
        effective_fingerprint: Some(fp),
        dims: Some(json!({ "inDim": a.in_dim, "outDim": a.out_dim })),
        active_ports: Some(shared_ports.into_iter().collect()),
        conditional_on_context: folded_info.as_ref().map(|_| true),
        context_ports_folded: folded_info,
        chain_a: Some(a.chain),
        chain_b: Some(b.chain),
        first_diff: None,
    }
}
