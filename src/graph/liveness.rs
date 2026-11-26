use serde::Serialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::catalog::{expected_weight_count, is_int_constructor};
use crate::graph::shard::shard_spec;
use crate::graph::validate::validate_spec;
use crate::spec::{LayerDecl, SpecV1, StepDecl};

#[derive(Debug, Clone, Serialize)]
pub struct LivenessSlotRow {
    pub slot: usize,
    pub def_step: Option<isize>,
    pub last_use_step: Option<isize>,
    pub bytes: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveBeforeStep {
    pub step: usize,
    pub bytes: Option<usize>,
    pub slots: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LivenessReport {
    pub peak_bytes: usize,
    pub peak_step: usize,
    pub peak_slots: Vec<usize>,
    pub peak_is_lower_bound: bool,
    pub unknown_bytes_slots: Vec<usize>,
    pub live_before: Vec<LiveBeforeStep>,
    pub slots: Vec<LivenessSlotRow>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SsaNormalizeReport {
    pub ssa_spec: SpecV1,
    pub final_slot_map: BTreeMap<usize, usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SlotCompactionReport {
    pub spec: SpecV1,
    pub slots_before: usize,
    pub slots_after: usize,
    pub slots_saved: usize,
    pub slot_remap: BTreeMap<usize, usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AutoRepairReport {
    pub spec: SpecV1,
    pub repaired: bool,
    pub remediation_receipt: Vec<Value>,
}

fn shape_bytes(shape: [usize; 4], dtype: Option<&str>) -> Option<usize> {
    let elem_bytes = match dtype {
        Some("i64") => 8usize,
        _ => 4usize,
    };
    let mut numel = 1usize;
    for d in shape {
        numel = numel.checked_mul(d)?;
    }
    numel.checked_mul(elem_bytes)
}

/// Pure static liveness analysis over a validated or structurally sound `SpecV1`.
///
/// `live_before[k]` (for `k = 0..=steps.len()`) lists every slot with `def_step < k`
/// and `last_use_step >= k`. Ports have `def_step = -1`; `output_slot` has
/// `last_use_step = steps.len()`.
pub fn analyze_liveness(
    spec: &SpecV1,
    slot_shapes: &BTreeMap<usize, [usize; 4]>,
    slot_dtypes: Option<&HashMap<usize, &'static str>>,
) -> LivenessReport {
    let n = spec.steps.len() as isize;
    let mut def: BTreeMap<usize, isize> = BTreeMap::new();
    let mut last_use: BTreeMap<usize, isize> = BTreeMap::new();
    let mut bytes_by_slot: BTreeMap<usize, Option<usize>> = BTreeMap::new();

    for p in &spec.ports {
        def.insert(p.slot, -1);
        last_use.insert(p.slot, -1);
        let dt = slot_dtypes.and_then(|m| m.get(&p.slot).copied());
        let sh = slot_shapes.get(&p.slot).copied().unwrap_or(p.shape);
        bytes_by_slot.insert(p.slot, shape_bytes(sh, dt));
    }

    for (si, st) in spec.steps.iter().enumerate() {
        let si_i = si as isize;
        for &in_s in st.input_slots() {
            def.entry(in_s).or_insert(-1);
            let entry = last_use.entry(in_s).or_insert(-1);
            if si_i > *entry {
                *entry = si_i;
            }
            bytes_by_slot.entry(in_s).or_insert_with(|| {
                let dt = slot_dtypes.and_then(|m| m.get(&in_s).copied());
                slot_shapes
                    .get(&in_s)
                    .copied()
                    .and_then(|sh| shape_bytes(sh, dt))
            });
        }
        if let Some(out_s) = st.output_slot() {
            def.entry(out_s).or_insert(si_i);
            last_use.entry(out_s).or_insert(si_i);
            let dt = slot_dtypes.and_then(|m| m.get(&out_s).copied());
            let b = slot_shapes
                .get(&out_s)
                .copied()
                .and_then(|sh| shape_bytes(sh, dt));
            bytes_by_slot.insert(out_s, b);
        }
    }

    last_use.insert(spec.output_slot, n);
    bytes_by_slot.entry(spec.output_slot).or_insert_with(|| {
        let dt = slot_dtypes.and_then(|m| m.get(&spec.output_slot).copied());
        slot_shapes
            .get(&spec.output_slot)
            .copied()
            .and_then(|sh| shape_bytes(sh, dt))
    });

    let slot_rows: Vec<LivenessSlotRow> = bytes_by_slot
        .iter()
        .map(|(&slot, &bytes)| LivenessSlotRow {
            slot,
            def_step: def.get(&slot).copied(),
            last_use_step: last_use.get(&slot).copied(),
            bytes,
        })
        .collect();

    let mut live_before = Vec::with_capacity(spec.steps.len() + 1);
    let mut ever_live_unknown = false;
    let mut unknown_seen: BTreeSet<usize> = BTreeSet::new();

    for k in 0..=spec.steps.len() {
        let k_i = k as isize;
        let mut live_slots = Vec::new();
        let mut step_bytes = 0usize;
        let mut has_unknown = false;
        for row in &slot_rows {
            if let (Some(d), Some(lu)) = (row.def_step, row.last_use_step)
                && d < k_i
                && lu >= k_i
            {
                live_slots.push(row.slot);
                match row.bytes {
                    Some(b) => step_bytes = step_bytes.saturating_add(b),
                    None => {
                        has_unknown = true;
                        ever_live_unknown = true;
                        unknown_seen.insert(row.slot);
                    }
                }
            }
        }
        live_before.push(LiveBeforeStep {
            step: k,
            bytes: if has_unknown { None } else { Some(step_bytes) },
            slots: live_slots,
        });
    }

    let mut peak_step = 0usize;
    let mut peak_bytes = live_before.first().and_then(|lb| lb.bytes).unwrap_or(0);
    for (k, lb) in live_before.iter().enumerate().skip(1) {
        let b = lb.bytes.unwrap_or(0);
        if b > peak_bytes {
            peak_bytes = b;
            peak_step = k;
        }
    }
    let peak_slots = live_before
        .get(peak_step)
        .map(|lb| lb.slots.clone())
        .unwrap_or_default();

    LivenessReport {
        peak_bytes,
        peak_step,
        peak_slots,
        peak_is_lower_bound: ever_live_unknown,
        unknown_bytes_slots: unknown_seen.into_iter().collect(),
        live_before,
        slots: slot_rows,
    }
}

/// Convert a `SpecV1` into strict single-static-assignment (SSA) form where
/// no step overwrites an input port or a previously written slot.
pub fn ssa_normalize_spec(spec: &SpecV1) -> SsaNormalizeReport {
    let mut cur_map: BTreeMap<usize, usize> = BTreeMap::new();
    let mut allocated: HashSet<usize> = HashSet::new();
    let mut next_slot = spec.num_slots;

    for p in &spec.ports {
        cur_map.insert(p.slot, p.slot);
        allocated.insert(p.slot);
        if p.slot >= next_slot {
            next_slot = p.slot + 1;
        }
    }

    let mut new_steps = Vec::with_capacity(spec.steps.len());
    for st in &spec.steps {
        let mut mapped_slots = Vec::with_capacity(st.slots.len());
        for &in_s in st.input_slots() {
            let s = cur_map.get(&in_s).copied().unwrap_or(in_s);
            mapped_slots.push(s);
        }
        if let Some(out_s) = st.output_slot() {
            let target = if allocated.contains(&out_s) {
                let fresh = next_slot;
                next_slot += 1;
                fresh
            } else {
                if out_s >= next_slot {
                    next_slot = out_s + 1;
                }
                out_s
            };
            allocated.insert(target);
            cur_map.insert(out_s, target);
            mapped_slots.push(target);
        }
        new_steps.push(StepDecl {
            kind: st.kind.clone(),
            layer: st.layer,
            slots: mapped_slots,
        });
    }

    let new_out = cur_map
        .get(&spec.output_slot)
        .copied()
        .unwrap_or(spec.output_slot);
    let mut ssa_spec = spec.clone();
    ssa_spec.steps = new_steps;
    ssa_spec.output_slot = new_out;
    ssa_spec.num_slots = next_slot.max(new_out + 1);

    SsaNormalizeReport {
        ssa_spec,
        final_slot_map: cur_map,
    }
}

/// Linear-scan liveness register allocator for `SpecV1` slots.
///
/// SSA-normalizes the graph first, computes exact liveness intervals across
/// unary/binary/ternary steps, preserves all declared input port slots, and
/// recycles dead intermediate physical slots.
pub fn compact_slots_by_liveness(spec: &SpecV1) -> Result<SlotCompactionReport, String> {
    let val = validate_spec(spec, false);
    if !val.valid {
        return Err(format!(
            "compact_slots_by_liveness: invalid spec ({})",
            val.code
        ));
    }

    let old_num_slots = spec.num_slots;
    let ssa = ssa_normalize_spec(spec).ssa_spec;
    let liveness = analyze_liveness(&ssa, &BTreeMap::new(), None);
    let port_slots: HashSet<usize> = ssa.ports.iter().map(|p| p.slot).collect();

    let mut next_phys = port_slots
        .iter()
        .copied()
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    let mut free_pool: BTreeSet<usize> = BTreeSet::new();
    let mut virt_to_phys: BTreeMap<usize, usize> = BTreeMap::new();
    for &ps in &port_slots {
        virt_to_phys.insert(ps, ps);
    }

    let mut release_at_step: HashMap<usize, Vec<usize>> = HashMap::new();
    for row in &liveness.slots {
        if port_slots.contains(&row.slot) {
            continue;
        }
        if let Some(lu) = row.last_use_step
            && lu >= 0
            && (lu as usize) < ssa.steps.len()
        {
            release_at_step
                .entry(lu as usize)
                .or_default()
                .push(row.slot);
        }
    }

    let alloc_phys = |virt: usize,
                      virt_to_phys: &mut BTreeMap<usize, usize>,
                      free_pool: &mut BTreeSet<usize>,
                      next_phys: &mut usize|
     -> usize {
        if let Some(&phys) = virt_to_phys.get(&virt) {
            return phys;
        }
        let phys = if let Some(reused) = free_pool.pop_first() {
            reused
        } else {
            let p = *next_phys;
            *next_phys += 1;
            p
        };
        virt_to_phys.insert(virt, phys);
        phys
    };

    let mut compacted = ssa.clone();
    for (si, st) in compacted.steps.iter_mut().enumerate() {
        let in_slots = st.input_slots().to_vec();
        let mut mapped = Vec::with_capacity(st.slots.len());
        for v in in_slots {
            mapped.push(alloc_phys(v, &mut virt_to_phys, &mut free_pool, &mut next_phys));
        }
        if let Some(virt_out) = st.output_slot() {
            let phys_out = alloc_phys(virt_out, &mut virt_to_phys, &mut free_pool, &mut next_phys);
            mapped.push(phys_out);
        }
        st.slots = mapped;

        if let Some(dead_list) = release_at_step.get(&si) {
            for &dead_virt in dead_list {
                if dead_virt == ssa.output_slot {
                    continue;
                }
                if let Some(&dead_phys) = virt_to_phys.get(&dead_virt) {
                    free_pool.insert(dead_phys);
                }
            }
        }
    }

    compacted.output_slot = virt_to_phys
        .get(&ssa.output_slot)
        .copied()
        .unwrap_or(ssa.output_slot);
    compacted.num_slots = next_phys.max(compacted.output_slot + 1);

    let post_val = validate_spec(&compacted, false);
    if !post_val.valid {
        return Err(format!(
            "compact_slots_by_liveness internal error: compacted spec failed validation ({})",
            post_val.code
        ));
    }

    let slots_after = compacted.num_slots;
    Ok(SlotCompactionReport {
        spec: compacted,
        slots_before: old_num_slots,
        slots_after,
        slots_saved: old_num_slots.saturating_sub(slots_after),
        slot_remap: virt_to_phys,
    })
}

/// Deterministic 3-pass auto-repair with auditable remediation receipts:
/// 1. Insert missing `intFromFloat`/`intToFloat` dtype boundary casts (and terminal `intToFloat` if output is `i64`).
/// 2. Drain unused declared ports (`plan_unused_ports`).
/// 3. Shard oversized `linear` layers exceeding `max_weights_per_layer` (when `> 0`).
pub fn auto_repair_spec(
    spec: &SpecV1,
    max_weights_per_layer: usize,
) -> Result<AutoRepairReport, String> {
    let mut current = spec.clone();
    let mut receipts: Vec<Value> = Vec::new();

    let next_reg_id_of = |s: &SpecV1| -> usize {
        s.layers
            .iter()
            .filter_map(|l| l.args.first().and_then(|v| v.as_u64()))
            .max()
            .map(|m| (m as usize) + 1)
            .unwrap_or(0)
    };

    // Pass 1: Mixed f32/i64 dtype boundary casts
    let mut has_int_layer = false;
    for l in &current.layers {
        if is_int_constructor(&l.constructor) {
            has_int_layer = true;
            break;
        }
    }
    if has_int_layer {
        let mut slot_dtype: HashMap<usize, &'static str> = HashMap::new();
        for p in &current.ports {
            slot_dtype.insert(p.slot, "f32");
        }
        // Only mark a port as native i64 if it is consumed exclusively by i64-input ops
        // (matching validate_spec's convention without breaking mixed f32+i64 port fans).
        let mut port_fed_f32: HashSet<usize> = HashSet::new();
        let mut port_fed_i64: HashSet<usize> = HashSet::new();
        for st in &current.steps {
            if let Some(layer) = current.layers.get(st.layer) {
                let wants_i64 = is_int_constructor(&layer.constructor)
                    && layer.constructor != "intFromFloat";
                for &in_s in st.input_slots() {
                    if current.port_by_slot(in_s).is_some() {
                        if wants_i64 {
                            port_fed_i64.insert(in_s);
                        } else {
                            port_fed_f32.insert(in_s);
                        }
                    }
                }
            }
        }
        for p in &current.ports {
            if port_fed_i64.contains(&p.slot) && !port_fed_f32.contains(&p.slot) {
                slot_dtype.insert(p.slot, "i64");
            }
        }

        let mut next_reg_id = next_reg_id_of(&current);
        let mut next_slot = current.num_slots;
        let orig_steps = current.steps.clone();
        let mut new_steps = Vec::with_capacity(orig_steps.len());

        for (si, st) in orig_steps.iter().enumerate() {
            let Some(decl) = current.layers.get(st.layer) else {
                new_steps.push(st.clone());
                continue;
            };
            let ctor = decl.constructor.clone();
            let expected_in = if is_int_constructor(&ctor) && ctor != "intFromFloat" {
                "i64"
            } else {
                "f32"
            };
            let arity = st.expected_arity();
            let mut fixed_slots = st.slots.clone();
            for (ii, slot_ref) in fixed_slots.iter_mut().take(arity).enumerate() {
                let s = *slot_ref;
                if let Some(&actual) = slot_dtype.get(&s)
                    && actual != expected_in
                {
                    let cast_ctor = if expected_in == "i64" {
                        "intFromFloat"
                    } else {
                        "intToFloat"
                    };
                    let cast_layer_idx = current.layers.len();
                    let cast_reg_id = next_reg_id;
                    next_reg_id += 1;
                    let cast_slot = next_slot;
                    next_slot += 1;
                    current.layers.push(LayerDecl {
                        constructor: cast_ctor.to_string(),
                        args: vec![Value::from(cast_reg_id)],
                    });
                    new_steps.push(StepDecl {
                        kind: "unary".to_string(),
                        layer: cast_layer_idx,
                        slots: vec![s, cast_slot],
                    });
                    slot_dtype.insert(cast_slot, expected_in);
                    *slot_ref = cast_slot;
                    receipts.push(json!({
                        "repair": "insert_dtype_boundary_cast",
                        "at_step": si,
                        "input_position": ii,
                        "from_slot": s,
                        "to_slot": cast_slot,
                        "cast_constructor": cast_ctor,
                        "reason": format!("step {si} ({ctor}) expects {expected_in} but slot {s} holds {actual}"),
                    }));
                }
            }
            if let Some(out_s) = fixed_slots.get(arity).copied() {
                let out_dt = if ctor == "intToFloat" {
                    "f32"
                } else if is_int_constructor(&ctor) {
                    "i64"
                } else {
                    "f32"
                };
                slot_dtype.insert(out_s, out_dt);
            }
            new_steps.push(StepDecl {
                kind: st.kind.clone(),
                layer: st.layer,
                slots: fixed_slots,
            });
        }

        if slot_dtype.get(&current.output_slot).copied() == Some("i64") {
            let cast_layer_idx = current.layers.len();
            let cast_reg_id = next_reg_id;
            let f32_out = next_slot;
            next_slot += 1;
            current.layers.push(LayerDecl {
                constructor: "intToFloat".to_string(),
                args: vec![Value::from(cast_reg_id)],
            });
            new_steps.push(StepDecl {
                kind: "unary".to_string(),
                layer: cast_layer_idx,
                slots: vec![current.output_slot, f32_out],
            });
            receipts.push(json!({
                "repair": "insert_terminal_intToFloat",
                "from_slot": current.output_slot,
                "to_slot": f32_out,
                "reason": "mixed-dtype v1 requires f32 outputSlot",
            }));
            current.output_slot = f32_out;
        }

        current.steps = new_steps;
        current.num_slots = current.num_slots.max(next_slot);
    }

    // Pass 2: Drain unused declared ports (plan_unused_ports)
    let mut consumed: HashSet<usize> = HashSet::new();
    for st in &current.steps {
        for &s in st.input_slots() {
            consumed.insert(s);
        }
    }
    let unused_ports: Vec<usize> = current
        .ports
        .iter()
        .map(|p| p.slot)
        .filter(|s| !consumed.contains(s))
        .collect();
    if !unused_ports.is_empty() {
        let base_reg_id = next_reg_id_of(&current);
        let base_slot = current.num_slots;
        let drain_count = unused_ports.len();
        for (offset, u_slot) in unused_ports.into_iter().enumerate() {
            let drain_layer_idx = current.layers.len();
            let drain_reg_id = base_reg_id + offset;
            let drain_out_slot = base_slot + offset;
            current.layers.push(LayerDecl {
                constructor: "relu".to_string(),
                args: vec![Value::from(drain_reg_id)],
            });
            current.steps.push(StepDecl {
                kind: "unary".to_string(),
                layer: drain_layer_idx,
                slots: vec![u_slot, drain_out_slot],
            });
            receipts.push(json!({
                "repair": "drain_unused_port",
                "port_slot": u_slot,
                "drain_slot": drain_out_slot,
                "reason": format!("port slot {u_slot} was declared but not read by any step (plan_unused_ports)"),
            }));
        }
        current.num_slots = current.num_slots.max(base_slot + drain_count);
    }

    // Pass 3: Oversized linear layer sharding
    if max_weights_per_layer > 0 {
        let has_oversized = current.layers.iter().enumerate().any(|(li, decl)| {
            if decl.constructor != "linear" {
                return false;
            }
            let exp = expected_weight_count(&decl.constructor, &decl.args).unwrap_or(0);
            let actual = current
                .weights
                .get(&li.to_string())
                .map(|w| w.len())
                .unwrap_or(exp);
            actual > max_weights_per_layer
        });
        if has_oversized {
            let sharded = shard_spec(&current, max_weights_per_layer)?;
            if sharded.sharded {
                for sh in &sharded.shards {
                    receipts.push(json!({
                        "repair": "recursive_shard_linear",
                        "layer_index": sh.original_layer,
                        "in_dim": sh.in_dim,
                        "out_dim": sh.out_dim,
                        "num_shards": sh.num_shards,
                        "max_weights": max_weights_per_layer,
                    }));
                }
                current = sharded.spec;
            }
        }
    }

    Ok(AutoRepairReport {
        spec: current,
        repaired: !receipts.is_empty(),
        remediation_receipt: receipts,
    })
}
