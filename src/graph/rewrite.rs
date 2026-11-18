use serde::Serialize;
use serde_json::Value;
use std::collections::HashSet;

use crate::spec::{LayerDecl, SpecV1, StepDecl};

#[derive(Debug, Clone, Serialize)]
pub struct RewriteRuleInfo {
    pub name: &'static str,
    pub description: &'static str,
    pub preconditions: &'static str,
}

pub const REWRITE_RULES: &[RewriteRuleInfo] = &[
    RewriteRuleInfo {
        name: "linear-branch-fusion",
        description: "linear(a)(x) + linear(b)(x) -> linear(a+b)(x) for bias-free linears with private intermediates.",
        preconditions: "two bias-free unary linear layers sharing the same input slot, summed by one binary add",
    },
    RewriteRuleInfo {
        name: "linear-chain-fusion",
        description: "linear2(linear1(x)) -> linear_fused(x) via exact affine composition W = W2*W1, b = W2*b1 + b2.",
        preconditions: "two chained unary linear layers with explicit weights and private intermediate slot",
    },
    RewriteRuleInfo {
        name: "relu-idempotent",
        description: "relu(relu(x)) -> relu(x): second relu is eliminated.",
        preconditions: "two chained unary relu layers with private intermediate slot",
    },
    RewriteRuleInfo {
        name: "transpose-involution",
        description: "transpose(d0,d1)(transpose(d0,d1)(x)) -> x when same axis pair is swapped back.",
        preconditions: "two chained unary transpose steps on the same axis pair with >= 1 remaining step",
    },
    RewriteRuleInfo {
        name: "dead-step-elimination",
        description: "Remove steps whose outputs never reach outputSlot while preserving port consumption.",
        preconditions: "at least one dead step not contributing to outputSlot or sole port consumer",
    },
    RewriteRuleInfo {
        name: "add-commute",
        description: "add(a,b) -> add(b,a): canonicalizes operand order.",
        preconditions: "at least one binary add step with distinct operands",
    },
];

pub const SIMPLIFY_ORDER: &[&str] = &[
    "linear-branch-fusion",
    "linear-chain-fusion",
    "relu-idempotent",
    "transpose-involution",
    "dead-step-elimination",
];

fn inputs_of(st: &StepDecl) -> &[usize] {
    let arity = st.expected_arity();
    if st.slots.len() > arity {
        &st.slots[..arity]
    } else if !st.slots.is_empty() {
        &st.slots[..st.slots.len() - 1]
    } else {
        &[]
    }
}

fn out_of(st: &StepDecl) -> usize {
    *st.slots.last().unwrap_or(&0)
}

fn uses_of(spec: &SpecV1, slot: usize) -> (Vec<usize>, Vec<usize>) {
    let mut reads = Vec::new();
    let mut writes = Vec::new();
    for (i, st) in spec.steps.iter().enumerate() {
        if inputs_of(st).contains(&slot) {
            reads.push(i);
        }
        if out_of(st) == slot {
            writes.push(i);
        }
    }
    (reads, writes)
}

fn max_registry_id(spec: &SpecV1) -> usize {
    spec.layers
        .iter()
        .filter_map(|l| l.args.first().and_then(|v| v.as_u64()))
        .max()
        .unwrap_or(0) as usize
}

fn apply_linear_branch_fusion(spec: &SpecV1) -> Result<SpecV1, String> {
    let mut s = spec.clone();
    let n = s.steps.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let (si, sj) = (&s.steps[i], &s.steps[j]);
            if si.kind != "unary" || sj.kind != "unary" {
                continue;
            }
            let (la, lb) = (&s.layers[si.layer], &s.layers[sj.layer]);
            if la.constructor != "linear" || lb.constructor != "linear" {
                continue;
            }
            let in_a = la.args.get(1).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let out_a = la.args.get(2).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let bias_a = la.args.get(3).and_then(|v| v.as_bool()).unwrap_or(false);
            let in_b = lb.args.get(1).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let out_b = lb.args.get(2).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let bias_b = lb.args.get(3).and_then(|v| v.as_bool()).unwrap_or(false);
            if bias_a || bias_b || in_a != in_b || out_a != out_b {
                continue;
            }
            let (in_slot, t1) = (si.slots[0], si.slots[1]);
            let (in_slot2, t2) = (sj.slots[0], sj.slots[1]);
            if in_slot != in_slot2 || t1 == s.output_slot || t2 == s.output_slot {
                continue;
            }
            let (Some(wa), Some(wb)) = (
                s.weights.get(&si.layer.to_string()),
                s.weights.get(&sj.layer.to_string()),
            ) else {
                continue;
            };
            if wa.len() != wb.len() {
                continue;
            }
            for k in (j + 1)..n {
                let sk = &s.steps[k];
                if sk.kind != "binary" || s.layers[sk.layer].constructor != "add" {
                    continue;
                }
                let (a, b, t3) = (sk.slots[0], sk.slots[1], sk.slots[2]);
                if !((a == t1 && b == t2) || (a == t2 && b == t1)) {
                    continue;
                }
                let (r1, w1) = uses_of(&s, t1);
                let (r2, w2) = uses_of(&s, t2);
                if r1 != vec![k] || w1.len() != 1 || r2 != vec![k] || w2.len() != 1 {
                    continue;
                }
                let new_id = max_registry_id(&s) + 1;
                let fused_w: Vec<f32> = wa.iter().zip(wb.iter()).map(|(x, y)| x + y).collect();
                let new_layer_idx = s.layers.len();
                s.layers.push(LayerDecl {
                    constructor: "linear".into(),
                    args: vec![Value::from(new_id), Value::from(in_a), Value::from(out_a), Value::from(false)],
                });
                s.weights.insert(new_layer_idx.to_string(), fused_w);
                let fused_step = StepDecl {
                    kind: "unary".into(),
                    layer: new_layer_idx,
                    slots: vec![in_slot, t3],
                };
                let mut new_steps: Vec<StepDecl> = s
                    .steps
                    .into_iter()
                    .enumerate()
                    .filter(|(idx, _)| *idx != i && *idx != j && *idx != k)
                    .map(|(_, st)| st)
                    .collect();
                new_steps.insert(i, fused_step);
                s.steps = new_steps;
                return Ok(s);
            }
        }
    }
    Err("linear-branch-fusion: no matching pattern".into())
}

fn apply_linear_chain_fusion(spec: &SpecV1) -> Result<SpecV1, String> {
    let mut s = spec.clone();
    let n = s.steps.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let (si, sj) = (&s.steps[i], &s.steps[j]);
            if si.kind != "unary" || sj.kind != "unary" {
                continue;
            }
            let (l1, l2) = (&s.layers[si.layer], &s.layers[sj.layer]);
            if l1.constructor != "linear" || l2.constructor != "linear" {
                continue;
            }
            let (in_slot, t1) = (si.slots[0], si.slots[1]);
            let (mid_in, t2) = (sj.slots[0], sj.slots[1]);
            if t1 != mid_in || t1 == s.output_slot {
                continue;
            }
            let (reads, writes) = uses_of(&s, t1);
            if reads != vec![j] || writes != vec![i] {
                continue;
            }
            let in1 = l1.args.get(1).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let out1 = l1.args.get(2).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let bias1 = l1.args.get(3).and_then(|v| v.as_bool()).unwrap_or(false);
            let in2 = l2.args.get(1).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let out2 = l2.args.get(2).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let bias2 = l2.args.get(3).and_then(|v| v.as_bool()).unwrap_or(false);
            if out1 != in2 {
                continue;
            }
            let (Some(w1_flat), Some(w2_flat)) = (
                s.weights.get(&si.layer.to_string()),
                s.weights.get(&sj.layer.to_string()),
            ) else {
                continue;
            };
            // Burn Linear layout: weight matrix is [in_dim, out_dim] row-major, followed by bias [out_dim].
            // Fused W_fused[k, o] = sum_m W1[k, m] * W2[m, o]
            // Fused b_fused[o] = (sum_m b1[m] * W2[m, o]) + b2[o]
            let mut w_fused = vec![0.0f32; in1 * out2];
            for k in 0..in1 {
                for o in 0..out2 {
                    let mut acc = 0.0f64;
                    for m in 0..out1 {
                        acc += (w1_flat[k * out1 + m] as f64) * (w2_flat[m * out2 + o] as f64);
                    }
                    w_fused[k * out2 + o] = acc as f32;
                }
            }
            let has_bias = bias1 || bias2;
            if has_bias {
                let b1 = if bias1 {
                    &w1_flat[in1 * out1..]
                } else {
                    &[][..]
                };
                let b2 = if bias2 {
                    &w2_flat[in2 * out2..]
                } else {
                    &[][..]
                };
                for o in 0..out2 {
                    let mut acc = if bias2 { b2[o] as f64 } else { 0.0 };
                    if bias1 {
                        for m in 0..out1 {
                            acc += (b1[m] as f64) * (w2_flat[m * out2 + o] as f64);
                        }
                    }
                    w_fused.push(acc as f32);
                }
            }

            let new_id = max_registry_id(&s) + 1;
            let new_layer_idx = s.layers.len();
            s.layers.push(LayerDecl {
                constructor: "linear".into(),
                args: vec![
                    Value::from(new_id),
                    Value::from(in1),
                    Value::from(out2),
                    Value::from(has_bias),
                ],
            });
            s.weights.insert(new_layer_idx.to_string(), w_fused);
            let fused_step = StepDecl {
                kind: "unary".into(),
                layer: new_layer_idx,
                slots: vec![in_slot, t2],
            };
            let mut new_steps: Vec<StepDecl> = s
                .steps
                .into_iter()
                .enumerate()
                .filter(|(idx, _)| *idx != i && *idx != j)
                .map(|(_, st)| st)
                .collect();
            new_steps.insert(i, fused_step);
            s.steps = new_steps;
            return Ok(s);
        }
    }
    Err("linear-chain-fusion: no matching pattern".into())
}

fn apply_relu_idempotent(spec: &SpecV1) -> Result<SpecV1, String> {
    let mut s = spec.clone();
    let n = s.steps.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let (si, sj) = (&s.steps[i], &s.steps[j]);
            if si.kind != "unary" || sj.kind != "unary" {
                continue;
            }
            if s.layers[si.layer].constructor != "relu" || s.layers[sj.layer].constructor != "relu" {
                continue;
            }
            let t1 = si.slots[1];
            let t2 = sj.slots[1];
            if sj.slots[0] != t1 || t1 == s.output_slot {
                continue;
            }
            let (reads, writes) = uses_of(&s, t1);
            if reads != vec![j] || writes != vec![i] {
                continue;
            }
            s.steps.remove(j);
            for st in &mut s.steps {
                for sl in &mut st.slots {
                    if *sl == t2 {
                        *sl = t1;
                    }
                }
            }
            if s.output_slot == t2 {
                s.output_slot = t1;
            }
            return Ok(s);
        }
    }
    Err("relu-idempotent: no matching pattern".into())
}

fn apply_transpose_involution(spec: &SpecV1) -> Result<SpecV1, String> {
    let mut s = spec.clone();
    if s.steps.len() <= 2 {
        return Err("transpose-involution: requires at least 1 remaining step".into());
    }
    let n = s.steps.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let (si, sj) = (&s.steps[i], &s.steps[j]);
            if si.kind != "unary" || sj.kind != "unary" {
                continue;
            }
            let (la, lb) = (&s.layers[si.layer], &s.layers[sj.layer]);
            if la.constructor != "transpose" || lb.constructor != "transpose" {
                continue;
            }
            let (a0, a1) = (la.args[1].as_u64(), la.args[2].as_u64());
            let (b0, b1) = (lb.args[1].as_u64(), lb.args[2].as_u64());
            if !((a0 == b0 && a1 == b1) || (a0 == b1 && a1 == b0)) {
                continue;
            }
            let (in_slot, t1) = (si.slots[0], si.slots[1]);
            let (mid_in, t2) = (sj.slots[0], sj.slots[1]);
            if t1 != mid_in || t1 == s.output_slot {
                continue;
            }
            let (reads, writes) = uses_of(&s, t1);
            if reads != vec![j] || writes != vec![i] {
                continue;
            }
            s.steps = s
                .steps
                .into_iter()
                .enumerate()
                .filter(|(idx, _)| *idx != i && *idx != j)
                .map(|(_, st)| st)
                .collect();
            for st in &mut s.steps {
                for sl in &mut st.slots {
                    if *sl == t2 {
                        *sl = in_slot;
                    }
                }
            }
            if s.output_slot == t2 {
                s.output_slot = in_slot;
            }
            return Ok(s);
        }
    }
    Err("transpose-involution: no matching pattern".into())
}

fn apply_dead_step_elimination(spec: &SpecV1) -> Result<SpecV1, String> {
    let mut s = spec.clone();
    let mut live_slots: HashSet<usize> = HashSet::from([s.output_slot]);
    let mut live_steps: HashSet<usize> = HashSet::new();

    for si in (0..s.steps.len()).rev() {
        let st = &s.steps[si];
        if live_slots.contains(&out_of(st)) {
            live_steps.insert(si);
            for &ins in inputs_of(st) {
                live_slots.insert(ins);
            }
        }
    }
    let port_slots: HashSet<usize> = s.ports.iter().map(|p| p.slot).collect();
    let mut consumed_ports: HashSet<usize> = HashSet::new();
    for &si in &live_steps {
        for &ins in inputs_of(&s.steps[si]) {
            if port_slots.contains(&ins) {
                consumed_ports.insert(ins);
            }
        }
    }
    for (si, st) in s.steps.iter().enumerate() {
        if live_steps.contains(&si) {
            continue;
        }
        for &ins in inputs_of(st) {
            if port_slots.contains(&ins) && !consumed_ports.contains(&ins) {
                live_steps.insert(si);
                consumed_ports.insert(ins);
            }
        }
    }
    if live_steps.len() == s.steps.len() {
        return Err("dead-step-elimination: no dead steps found".into());
    }
    s.steps = s
        .steps
        .into_iter()
        .enumerate()
        .filter(|(idx, _)| live_steps.contains(idx))
        .map(|(_, st)| st)
        .collect();
    Ok(s)
}

fn apply_add_commute(spec: &SpecV1) -> Result<SpecV1, String> {
    let mut s = spec.clone();
    for st in &mut s.steps {
        if st.kind == "binary"
            && s.layers[st.layer].constructor == "add"
            && st.slots[0] != st.slots[1]
        {
            st.slots.swap(0, 1);
            return Ok(s);
        }
    }
    Err("add-commute: no binary add with distinct operands".into())
}

pub fn apply_rewrite(name: &str, spec: &SpecV1) -> Result<SpecV1, String> {
    match name {
        "linear-branch-fusion" => apply_linear_branch_fusion(spec),
        "linear-chain-fusion" => apply_linear_chain_fusion(spec),
        "relu-idempotent" => apply_relu_idempotent(spec),
        "transpose-involution" => apply_transpose_involution(spec),
        "dead-step-elimination" => apply_dead_step_elimination(spec),
        "add-commute" => apply_add_commute(spec),
        _ => Err(format!("unknown rewrite rule '{name}'")),
    }
}

pub fn apply_rewrites_recursive(
    spec: &SpecV1,
    max_passes: usize,
) -> (SpecV1, Vec<String>) {
    let mut cur = spec.clone();
    let mut applied = Vec::new();
    for _ in 0..max_passes {
        let mut changed = false;
        for &rule in SIMPLIFY_ORDER {
            if let Ok(next) = apply_rewrite(rule, &cur) {
                cur = next;
                applied.push(rule.to_string());
                changed = true;
                break;
            }
        }
        if !changed {
            break;
        }
    }
    (cur, applied)
}
