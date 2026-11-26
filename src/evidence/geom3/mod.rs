pub mod math;

use serde::Serialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

use crate::runtime::engine::PreparedGraph;
use crate::spec::{LayerDecl, PortDecl, SpecV1, StepDecl};

pub use math::*;

/// Maximum vertices packed into a single block-diagonal `linear(4N, 4N)` layer:
/// `(4 * 63)^2 = 63,504 <= 65,535` weights.
pub const BATCH_MAX: usize = 63;

fn make_f4_port(slot: usize, role: &str, dim: usize) -> PortDecl {
    PortDecl {
        slot,
        role: role.to_string(),
        shape: [1, dim, 1, 1],
        layout: "feature_axis1_singleton".to_string(),
    }
}

fn vec_f64_to_json(vals: &[f64]) -> Vec<Value> {
    vals.iter().map(|&v| json!(v)).collect()
}

#[derive(Debug, Clone, Serialize)]
pub struct GeomBuildReport {
    pub ok: bool,
    pub composed: bool,
    pub steps: usize,
    pub out_shape: [usize; 4],
    pub reference_point_f64: Vec3,
    pub composed_matrix_rowmajor: Mat4,
    pub note: &'static str,
    pub spec: SpecV1,
}

/// Compile a JSON chain spec `{"input": [x,y,z], "chain": [ops...]}` into a `SpecV1`.
pub fn build_chain_spec(doc: &Value, compose: bool) -> Result<GeomBuildReport, String> {
    let obj = doc
        .as_object()
        .ok_or_else(|| "geom build: input document must be a JSON object".to_string())?;
    let input_val = obj
        .get("input")
        .or_else(|| obj.get("point"))
        .ok_or_else(|| "geom build: document needs 'input' (or 'point'): [x, y, z]".to_string())?;
    let input = parse_vec3_value(input_val, "input")?;

    let chain_arr = obj
        .get("chain")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "geom build: document needs a non-empty 'chain' array".to_string())?;
    if chain_arr.is_empty() {
        return Err("geom build: 'chain' array must not be empty".into());
    }

    let mats: Vec<Mat4> = chain_arr
        .iter()
        .map(op_to_mat4)
        .collect::<Result<Vec<_>, _>>()?;
    let composed_mat = compose_chain_checked(&mats, "geom build")?;
    let built: Vec<Mat4> = if compose { vec![composed_mat] } else { mats };

    let mut layers = Vec::with_capacity(built.len() + 1);
    let mut steps = Vec::with_capacity(built.len() + 1);
    let mut weights: BTreeMap<String, Vec<f32>> = BTreeMap::new();

    for (i, m) in built.iter().enumerate() {
        layers.push(LayerDecl {
            constructor: "linear".to_string(),
            args: vec![json!(100 + i), json!(4), json!(4), json!(false)],
        });
        weights.insert(
            i.to_string(),
            mat4_to_col_major_f32_checked(m, "geom build")?,
        );
    }

    let mut prev = 0usize;
    let mut next = 2usize;
    for i in 0..built.len() {
        steps.push(StepDecl {
            kind: "unary".to_string(),
            layer: i,
            slots: vec![prev, next],
        });
        prev = next;
        next += 1;
    }

    let add_idx = built.len();
    layers.push(LayerDecl {
        constructor: "add".to_string(),
        args: vec![json!(100 + built.len() + 100)],
    });
    steps.push(StepDecl {
        kind: "binary".to_string(),
        layer: add_idx,
        slots: vec![prev, 1, next],
    });

    let mut inputs = BTreeMap::new();
    inputs.insert("0".to_string(), vec_f64_to_json(&homogenize(input)));
    inputs.insert("1".to_string(), vec_f64_to_json(&[0.0, 0.0, 0.0, 0.0]));

    let spec = SpecV1 {
        num_slots: next + 1,
        layers,
        steps,
        output_slot: next,
        ports: vec![
            make_f4_port(0, "observation", 4),
            make_f4_port(1, "observation", 4),
        ],
        weights,
        inputs,
    };

    Ok(GeomBuildReport {
        ok: true,
        composed: compose,
        steps: built.len(),
        out_shape: [1, 4, 1, 1],
        reference_point_f64: apply_mat4(&composed_mat, input)?,
        composed_matrix_rowmajor: composed_mat,
        note: "output is homogeneous [x,y,z,1]; dehomogenize host-side",
        spec,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct SceneNodeSummary {
    pub id: String,
    pub parent: Option<String>,
    pub local_point: Vec3,
    pub world_point: Vec3,
    pub world_matrix_rowmajor: Mat4,
}

#[derive(Debug, Clone, Serialize)]
pub struct SceneTreeReport {
    pub ok: bool,
    pub nodes: Vec<SceneNodeSummary>,
    pub spec: SpecV1,
}

struct ParsedTreeNode {
    id: String,
    parent: Option<String>,
    local_mat: Mat4,
    point: Vec3,
}

const MAX_SCENE_TREE_DEPTH: usize = 512;

fn resolve_world_mat(
    id: &str,
    by_id: &BTreeMap<String, usize>,
    parsed: &[ParsedTreeNode],
    world_mats: &mut BTreeMap<String, (Mat4, usize)>,
    visiting: &mut BTreeSet<String>,
    rec_depth: usize,
) -> Result<(Mat4, usize), String> {
    if rec_depth > MAX_SCENE_TREE_DEPTH {
        return Err(format!(
            "geom scene-tree: hierarchy depth exceeded maximum ({MAX_SCENE_TREE_DEPTH}) at node '{id}'"
        ));
    }
    if let Some(&cached) = world_mats.get(id) {
        return Ok(cached);
    }
    if visiting.contains(id) {
        return Err(format!(
            "geom scene-tree: cycle detected in kinematic tree at node '{id}'"
        ));
    }
    let &idx = by_id
        .get(id)
        .ok_or_else(|| format!("geom scene-tree: unknown parent node '{id}'"))?;
    visiting.insert(id.to_string());
    let node = &parsed[idx];
    let (world_m, tree_depth) = match &node.parent {
        Some(p) => {
            let (pm, pd) =
                resolve_world_mat(p, by_id, parsed, world_mats, visiting, rec_depth + 1)?;
            let d = pd + 1;
            if d > MAX_SCENE_TREE_DEPTH {
                return Err(format!(
                    "geom scene-tree: hierarchy depth exceeded maximum ({MAX_SCENE_TREE_DEPTH}) at node '{id}'"
                ));
            }
            (
                check_mat4_finite(&mat4_mul(&pm, &node.local_mat), "geom scene-tree")?,
                d,
            )
        }
        None => (check_mat4_finite(&node.local_mat, "geom scene-tree")?, 0),
    };
    visiting.remove(id);
    world_mats.insert(id.to_string(), (world_m, tree_depth));
    Ok((world_m, tree_depth))
}

/// Compile a hierarchical kinematic scene tree `{"nodes": [{id, parent?, chain?, point?}]}`.
pub fn compile_scene_tree(doc: &Value) -> Result<SceneTreeReport, String> {
    let nodes_arr = doc
        .get("nodes")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "geom scene-tree: 'nodes' must be a non-empty array".to_string())?;
    if nodes_arr.is_empty() {
        return Err("geom scene-tree: 'nodes' must be a non-empty array".into());
    }

    let mut by_id: BTreeMap<String, usize> = BTreeMap::new();
    let mut parsed = Vec::with_capacity(nodes_arr.len());

    for (idx, nv) in nodes_arr.iter().enumerate() {
        let obj = nv
            .as_object()
            .ok_or_else(|| format!("geom scene-tree: nodes[{idx}] must be a JSON object"))?;
        let id = obj
            .get("id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| format!("geom scene-tree: nodes[{idx}] missing non-empty string 'id'"))?
            .to_string();
        if by_id.contains_key(&id) {
            return Err(format!("geom scene-tree: duplicate node id '{id}'"));
        }
        let parent = match obj.get("parent") {
            None | Some(Value::Null) => None,
            Some(v) => Some(
                v.as_str()
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| format!("geom scene-tree: node '{id}' parent must be string or null"))?
                    .to_string(),
            ),
        };
        let chain_mats: Vec<Mat4> = match obj.get("chain") {
            None | Some(Value::Null) => Vec::new(),
            Some(v) => {
                let arr = v
                    .as_array()
                    .ok_or_else(|| format!("geom scene-tree: node '{id}' chain must be an array"))?;
                arr.iter()
                    .map(op_to_mat4)
                    .collect::<Result<Vec<_>, _>>()?
            }
        };
        let point = match obj.get("point") {
            None | Some(Value::Null) => [0.0, 0.0, 0.0],
            Some(v) => parse_vec3_value(v, &format!("node '{id}'.point"))?,
        };
        let local_mat = compose_chain_checked(&chain_mats, &format!("node '{id}'"))?;
        by_id.insert(id.clone(), idx);
        parsed.push(ParsedTreeNode {
            id,
            parent,
            local_mat,
            point,
        });
    }

    let mut world_mats: BTreeMap<String, (Mat4, usize)> = BTreeMap::new();
    let mut visiting: BTreeSet<String> = BTreeSet::new();
    for node in &parsed {
        resolve_world_mat(&node.id, &by_id, &parsed, &mut world_mats, &mut visiting, 0)?;
    }

    let mut layers = Vec::new();
    let mut steps = Vec::new();
    let mut weights = BTreeMap::new();
    let mut ports = Vec::new();
    let mut inputs = BTreeMap::new();
    let mut next_reg_id = 100u64;
    let mut next_slot = if parsed.len() < 2 { 2 } else { parsed.len() };

    for (idx, node) in parsed.iter().enumerate() {
        ports.push(make_f4_port(idx, "observation", 4));
        inputs.insert(idx.to_string(), vec_f64_to_json(&homogenize(node.point)));
    }
    if parsed.len() == 1 {
        ports.push(make_f4_port(1, "context", 4));
        inputs.insert("1".to_string(), vec_f64_to_json(&[0.0, 0.0, 0.0, 0.0]));
    }

    let mut out_slots = Vec::with_capacity(parsed.len());
    for (idx, node) in parsed.iter().enumerate() {
        let w_mat = world_mats[&node.id].0;
        let lin_idx = layers.len();
        layers.push(LayerDecl {
            constructor: "linear".to_string(),
            args: vec![json!(next_reg_id), json!(4), json!(4), json!(false)],
        });
        next_reg_id += 1;
        weights.insert(
            lin_idx.to_string(),
            mat4_to_col_major_f32_checked(&w_mat, &format!("scene node '{}'", node.id))?,
        );
        let out_slot = next_slot;
        next_slot += 1;
        steps.push(StepDecl {
            kind: "unary".to_string(),
            layer: lin_idx,
            slots: vec![idx, out_slot],
        });
        out_slots.push(out_slot);
    }

    let mut final_slot = out_slots[0];
    if parsed.len() == 1 {
        let add_idx = layers.len();
        layers.push(LayerDecl {
            constructor: "add".to_string(),
            args: vec![json!(next_reg_id)],
        });
        let s_out = next_slot;
        next_slot += 1;
        steps.push(StepDecl {
            kind: "binary".to_string(),
            layer: add_idx,
            slots: vec![final_slot, 1, s_out],
        });
        final_slot = s_out;
    } else {
        let concat_idx = layers.len();
        layers.push(LayerDecl {
            constructor: "concat".to_string(),
            args: vec![json!(next_reg_id), json!(1)],
        });
        for &slot in out_slots.iter().skip(1) {
            let target = next_slot;
            next_slot += 1;
            steps.push(StepDecl {
                kind: "binary".to_string(),
                layer: concat_idx,
                slots: vec![final_slot, slot, target],
            });
            final_slot = target;
        }
    }

    let mut summaries = Vec::with_capacity(parsed.len());
    for node in &parsed {
        let w_mat = world_mats[&node.id].0;
        summaries.push(SceneNodeSummary {
            id: node.id.clone(),
            parent: node.parent.clone(),
            local_point: node.point,
            world_point: apply_mat4(&w_mat, node.point)?,
            world_matrix_rowmajor: w_mat,
        });
    }

    Ok(SceneTreeReport {
        ok: true,
        nodes: summaries,
        spec: SpecV1 {
            num_slots: next_slot,
            layers,
            steps,
            output_slot: final_slot,
            ports,
            weights,
            inputs,
        },
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct IkSolveReport {
    pub ok: bool,
    pub converged: bool,
    pub iterations: usize,
    pub start_error: f64,
    pub final_error: f64,
    pub final_distance: f64,
    pub tol: f64,
    pub target: Vec3,
    pub end_effector: Vec3,
    pub engine_end_effector: Vec3,
    pub engine_max_abs_diff: f64,
    pub solved_angles_deg: Vec<f64>,
    pub scene_tree: SceneTreeReport,
}

struct ParsedJoint {
    id: String,
    axis: char,
    tx: f64,
    ty: f64,
    tz: f64,
    initial_deg: f64,
    min_deg: Option<f64>,
    max_deg: Option<f64>,
}

fn forward_ee(joints: &[ParsedJoint], angles: &[f64]) -> Result<Vec3, String> {
    let mut m = mat4_identity();
    for (j, &deg) in joints.iter().zip(angles.iter()) {
        let rot = match j.axis {
            'x' => rot_x4(deg)?,
            'y' => rot_y4(deg)?,
            _ => rot_z4(deg)?,
        };
        let tr = translate4(j.tx, j.ty, j.tz)?;
        m = mat4_mul(&m, &mat4_mul(&rot, &tr));
    }
    apply_mat4(&m, [0.0, 0.0, 0.0])
}

fn dist3(a: Vec3, b: Vec3) -> f64 {
    let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Multi-joint 3D Inverse Kinematics solver with Polyak-stepped backtracking GD
/// and Burn `Device::flex()` verification of the compiled scene-tree spec.
pub fn solve_ik(doc: &Value) -> Result<IkSolveReport, String> {
    let obj = doc
        .as_object()
        .ok_or_else(|| "geom solve-ik: input document must be a JSON object".to_string())?;
    let joints_arr = obj
        .get("joints")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "geom solve-ik: 'joints' must be a non-empty array".to_string())?;
    if joints_arr.is_empty() {
        return Err("geom solve-ik: 'joints' must be a non-empty array".into());
    }
    let target_val = obj
        .get("target")
        .ok_or_else(|| "geom solve-ik: missing 'target': [tx, ty, tz]".to_string())?;
    let target = parse_vec3_value(target_val, "target")?;

    let tol = match obj.get("tol") {
        None | Some(Value::Null) => 1e-3,
        Some(v) => v
            .as_f64()
            .ok_or_else(|| "geom solve-ik: tol must be a finite number".to_string())?,
    };
    if !tol.is_finite() || tol <= 0.0 {
        return Err(format!("geom solve-ik: tol must be finite and > 0, got {tol}"));
    }
    let max_iters = match obj.get("max_iters") {
        None | Some(Value::Null) => 150usize,
        Some(v) => v
            .as_u64()
            .ok_or_else(|| "geom solve-ik: max_iters must be a positive integer".to_string())?
            as usize,
    };
    if max_iters == 0 {
        return Err("geom solve-ik: max_iters must be >= 1".into());
    }

    let mut joints = Vec::with_capacity(joints_arr.len());
    for (idx, jv) in joints_arr.iter().enumerate() {
        let jo = jv
            .as_object()
            .ok_or_else(|| format!("geom solve-ik: joints[{idx}] must be a JSON object"))?;
        let id = match jo.get("id") {
            None | Some(Value::Null) => format!("joint_{idx}"),
            Some(v) => v
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    format!("geom solve-ik: joints[{idx}].id must be a non-empty string")
                })?
                .to_string(),
        };
        let ax_str = match jo.get("axis") {
            None | Some(Value::Null) => "z",
            Some(v) => v
                .as_str()
                .ok_or_else(|| format!("geom solve-ik: joints[{idx}].axis must be a string"))?,
        };
        let axis = match ax_str.to_ascii_lowercase().as_str() {
            "x" => 'x',
            "y" => 'y',
            "z" => 'z',
            other => {
                return Err(format!(
                    "geom solve-ik: joints[{idx}] axis must be x/y/z, got '{other}'"
                ));
            }
        };
        let tx = match jo.get("length").or_else(|| jo.get("tx")) {
            None | Some(Value::Null) => 1.0,
            Some(v) => check_finite(
                v.as_f64()
                    .ok_or_else(|| format!("geom solve-ik: joints[{idx}] length/tx must be a number"))?,
                "joint.length",
            )?,
        };
        let ty = match jo.get("ty") {
            None | Some(Value::Null) => 0.0,
            Some(v) => check_finite(
                v.as_f64()
                    .ok_or_else(|| format!("geom solve-ik: joints[{idx}] ty must be a number"))?,
                "joint.ty",
            )?,
        };
        let tz = match jo.get("tz") {
            None | Some(Value::Null) => 0.0,
            Some(v) => check_finite(
                v.as_f64()
                    .ok_or_else(|| format!("geom solve-ik: joints[{idx}] tz must be a number"))?,
                "joint.tz",
            )?,
        };
        let initial_deg = match jo.get("initial_deg").or_else(|| jo.get("angle_deg")) {
            None | Some(Value::Null) => 0.0,
            Some(v) => check_finite(
                v.as_f64()
                    .ok_or_else(|| format!("geom solve-ik: joints[{idx}] initial_deg must be a number"))?,
                "joint.initial_deg",
            )?,
        };
        let min_deg = match jo.get("min_deg") {
            None | Some(Value::Null) => None,
            Some(v) => Some(check_finite(
                v.as_f64()
                    .ok_or_else(|| format!("geom solve-ik: joints[{idx}] min_deg must be a number"))?,
                "joint.min_deg",
            )?),
        };
        let max_deg = match jo.get("max_deg") {
            None | Some(Value::Null) => None,
            Some(v) => Some(check_finite(
                v.as_f64()
                    .ok_or_else(|| format!("geom solve-ik: joints[{idx}] max_deg must be a number"))?,
                "joint.max_deg",
            )?),
        };
        if let (Some(lo), Some(hi)) = (min_deg, max_deg)
            && lo > hi
        {
            return Err(format!(
                "geom solve-ik: joints[{idx}] min_deg ({lo}) > max_deg ({hi})"
            ));
        }
        joints.push(ParsedJoint {
            id,
            axis,
            tx,
            ty,
            tz,
            initial_deg,
            min_deg,
            max_deg,
        });
    }

    let mut angles: Vec<f64> = joints
        .iter()
        .map(|j| {
            let mut a = j.initial_deg;
            if let Some(lo) = j.min_deg {
                a = a.max(lo);
            }
            if let Some(hi) = j.max_deg {
                a = a.min(hi);
            }
            a
        })
        .collect();

    let mut cur_pos = forward_ee(&joints, &angles)?;
    let mut cur_err = dist3(cur_pos, target);
    let start_err = cur_err;
    let mut iters_run = 0usize;

    for it in 0..max_iters {
        if cur_err <= tol {
            break;
        }
        iters_run = it + 1;
        let eps = 1e-3;
        let f_val = cur_err * cur_err;
        let mut grad = vec![0.0f64; joints.len()];
        let mut grad_norm_sq = 0.0f64;

        for j in 0..joints.len() {
            let mut a_plus = angles.clone();
            a_plus[j] += eps;
            let mut a_minus = angles.clone();
            a_minus[j] -= eps;
            let ep = dist3(forward_ee(&joints, &a_plus)?, target);
            let em = dist3(forward_ee(&joints, &a_minus)?, target);
            let g = (ep * ep - em * em) / (2.0 * eps);
            grad[j] = g;
            grad_norm_sq += g * g;
        }
        if grad_norm_sq < 1e-20 {
            break;
        }
        let polyak_scale = f_val / grad_norm_sq;
        let mut alpha = 1.0;
        let mut improved = false;
        for _ in 0..16 {
            let step_scale = alpha * polyak_scale;
            let cand: Vec<f64> = angles
                .iter()
                .enumerate()
                .map(|(j, &a)| {
                    let mut next = a - step_scale * grad[j];
                    if let Some(lo) = joints[j].min_deg {
                        next = next.max(lo);
                    }
                    if let Some(hi) = joints[j].max_deg {
                        next = next.min(hi);
                    }
                    next
                })
                .collect();
            let cand_pos = forward_ee(&joints, &cand)?;
            let cand_err = dist3(cand_pos, target);
            if cand_err < cur_err {
                angles = cand;
                cur_pos = cand_pos;
                cur_err = cand_err;
                improved = true;
                break;
            }
            alpha *= 0.5;
        }
        if !improved {
            break;
        }
    }

    let tree_nodes: Vec<Value> = joints
        .iter()
        .enumerate()
        .map(|(idx, j)| {
            let parent = if idx == 0 {
                Value::Null
            } else {
                json!(joints[idx - 1].id)
            };
            json!({
                "id": j.id,
                "parent": parent,
                "chain": [
                    { "op": "translate", "by": [j.tx, j.ty, j.tz] },
                    { "op": format!("rotate_{}", j.axis), "degrees": angles[idx] }
                ],
                "point": [0.0, 0.0, 0.0]
            })
        })
        .collect();
    let scene_tree = compile_scene_tree(&json!({ "nodes": tree_nodes }))?;

    // Verify end-effector position through Burn Device::flex() execution
    let graph = PreparedGraph::compile(&scene_tree.spec)?;
    let out = graph.run_once(&scene_tree.spec, None)?;
    let last_off = (joints.len() - 1) * 4;
    if out.values.len() < last_off + 4 {
        return Err("geom solve-ik: engine output length shorter than expected".into());
    }
    let eng_clip = [
        out.values[last_off] as f64,
        out.values[last_off + 1] as f64,
        out.values[last_off + 2] as f64,
        out.values[last_off + 3] as f64,
    ];
    let eng_ee = dehomogenize(eng_clip)?;
    let eng_diff = (0..3)
        .map(|k| (eng_ee[k] - cur_pos[k]).abs())
        .fold(0.0f64, f64::max);

    let converged = cur_err <= tol;
    let engine_tol = tol.max(1e-3) * (1.0 + cur_pos.iter().map(|v| v.abs()).fold(0.0f64, f64::max));
    let engine_ok = eng_diff.is_finite() && eng_diff <= engine_tol;
    Ok(IkSolveReport {
        ok: converged && engine_ok,
        converged,
        iterations: iters_run,
        start_error: start_err,
        final_error: cur_err,
        final_distance: cur_err,
        tol,
        target,
        end_effector: cur_pos,
        engine_end_effector: eng_ee,
        engine_max_abs_diff: eng_diff,
        solved_angles_deg: angles,
        scene_tree,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct ProjectedVertex {
    pub behind: bool,
    pub clip: Vec4,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ndc: Option<Vec3>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub screen: Option<[f64; 2]>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProjectedSceneNode {
    pub id: String,
    pub parent: Option<String>,
    pub world_point: Vec3,
    pub projection: ProjectedVertex,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProjectedTriangle {
    pub input: [usize; 3],
    pub spans_camera: bool,
    pub clipped_count: usize,
    pub clipped: Vec<[ProjectedVertex; 3]>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GeomProjectReport {
    pub ok: bool,
    pub batched: bool,
    pub chunks: usize,
    pub vertices: usize,
    pub projected: usize,
    pub behind_camera: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scene_nodes: Option<Vec<ProjectedSceneNode>>,
    pub results: Vec<ProjectedVertex>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub triangles: Option<Vec<ProjectedTriangle>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub triangles_clipped_away: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub triangles_spans_camera: Option<usize>,
}

fn project_clip(clip: Vec4, vp_w: f64, vp_h: f64) -> Result<ProjectedVertex, String> {
    if !clip.iter().all(|v| v.is_finite()) {
        return Err("geom project: non-finite clip coordinates produced by engine".into());
    }
    let [x, y, z, w] = clip;
    if w <= 0.0 {
        return Ok(ProjectedVertex {
            behind: true,
            clip,
            ndc: None,
            screen: None,
        });
    }
    let nx = x / w;
    let ny = y / w;
    let nz = z / w;
    let sx = (nx * 0.5 + 0.5) * vp_w;
    let sy = (1.0 - (ny * 0.5 + 0.5)) * vp_h;
    if ![nx, ny, nz, sx, sy].iter().all(|v| v.is_finite()) {
        return Err(
            "geom project: non-finite projected NDC or screen coordinates (overflow)".into(),
        );
    }
    Ok(ProjectedVertex {
        behind: false,
        clip,
        ndc: Some([nx, ny, nz]),
        screen: Some([sx, sy]),
    })
}

fn build_single_mvp_spec(mvp: &Mat4) -> SpecV1 {
    let mut weights = BTreeMap::new();
    weights.insert("0".to_string(), mat4_to_col_major_f32(mvp));
    let mut inputs = BTreeMap::new();
    inputs.insert("0".to_string(), vec_f64_to_json(&[0.0, 0.0, 0.0, 1.0]));
    inputs.insert("1".to_string(), vec_f64_to_json(&[0.0, 0.0, 0.0, 0.0]));

    SpecV1 {
        num_slots: 4,
        layers: vec![
            LayerDecl {
                constructor: "linear".to_string(),
                args: vec![json!(100), json!(4), json!(4), json!(false)],
            },
            LayerDecl {
                constructor: "add".to_string(),
                args: vec![json!(200)],
            },
        ],
        steps: vec![
            StepDecl {
                kind: "unary".to_string(),
                layer: 0,
                slots: vec![0, 2],
            },
            StepDecl {
                kind: "binary".to_string(),
                layer: 1,
                slots: vec![2, 1, 3],
            },
        ],
        output_slot: 3,
        ports: vec![
            make_f4_port(0, "observation", 4),
            make_f4_port(1, "observation", 4),
        ],
        weights,
        inputs,
    }
}

fn build_batched_mvp_spec(mvp: &Mat4, n: usize) -> SpecV1 {
    let dim = 4 * n;
    let mut w = vec![0.0f32; dim * dim];
    for k in 0..n {
        for r in 0..4 {
            for c in 0..4 {
                w[(k * 4 + c) * dim + (k * 4 + r)] = mvp[r * 4 + c] as f32;
            }
        }
    }
    let mut weights = BTreeMap::new();
    weights.insert("0".to_string(), w);
    let zeros = vec![json!(0.0); dim];
    let mut inputs = BTreeMap::new();
    inputs.insert("0".to_string(), zeros.clone());
    inputs.insert("1".to_string(), zeros);

    SpecV1 {
        num_slots: 4,
        layers: vec![
            LayerDecl {
                constructor: "linear".to_string(),
                args: vec![json!(100), json!(dim), json!(dim), json!(false)],
            },
            LayerDecl {
                constructor: "add".to_string(),
                args: vec![json!(200)],
            },
        ],
        steps: vec![
            StepDecl {
                kind: "unary".to_string(),
                layer: 0,
                slots: vec![0, 2],
            },
            StepDecl {
                kind: "binary".to_string(),
                layer: 1,
                slots: vec![2, 1, 3],
            },
        ],
        output_slot: 3,
        ports: vec![
            make_f4_port(0, "observation", dim),
            make_f4_port(1, "observation", dim),
        ],
        weights,
        inputs,
    }
}

fn run_clips_single(mvp: &Mat4, vertices: &[Vec3]) -> Result<Vec<Vec4>, String> {
    let spec = build_single_mvp_spec(mvp);
    let graph = PreparedGraph::compile(&spec)?;
    let mut clips = Vec::with_capacity(vertices.len());
    for &v in vertices {
        let mut ov = BTreeMap::new();
        ov.insert("0".to_string(), vec_f64_to_json(&homogenize(v)));
        let out = graph.run_once(&spec, Some(&ov))?;
        if out.values.len() < 4 {
            return Err("geom project: engine returned < 4 clip components".into());
        }
        clips.push([
            out.values[0] as f64,
            out.values[1] as f64,
            out.values[2] as f64,
            out.values[3] as f64,
        ]);
    }
    Ok(clips)
}

fn run_clips_batched(mvp: &Mat4, vertices: &[Vec3]) -> Result<(Vec<Vec4>, usize), String> {
    let mut clips = Vec::with_capacity(vertices.len());
    let mut chunks = 0usize;
    let mut offset = 0usize;
    while offset < vertices.len() {
        let end = (offset + BATCH_MAX).min(vertices.len());
        let slice = &vertices[offset..end];
        let n = slice.len();
        let spec = build_batched_mvp_spec(mvp, n);
        let graph = PreparedGraph::compile(&spec)?;
        let mut flat = Vec::with_capacity(n * 4);
        for &v in slice {
            let h = homogenize(v);
            flat.push(json!(h[0]));
            flat.push(json!(h[1]));
            flat.push(json!(h[2]));
            flat.push(json!(h[3]));
        }
        let mut ov = BTreeMap::new();
        ov.insert("0".to_string(), flat);
        let out = graph.run_once(&spec, Some(&ov))?;
        if out.values.len() < n * 4 {
            return Err("geom batch: engine output length shorter than expected".into());
        }
        for k in 0..n {
            clips.push([
                out.values[k * 4] as f64,
                out.values[k * 4 + 1] as f64,
                out.values[k * 4 + 2] as f64,
                out.values[k * 4 + 3] as f64,
            ]);
        }
        chunks += 1;
        offset = end;
    }
    Ok((clips, chunks))
}

fn build_mvp_from_doc(obj: &serde_json::Map<String, Value>) -> Result<(Mat4, f64, f64), String> {
    let model_mats: Vec<Mat4> = match obj.get("model") {
        None | Some(Value::Null) => Vec::new(),
        Some(v) => {
            let arr = v
                .as_array()
                .ok_or_else(|| "geom project: 'model' must be an array of transform ops".to_string())?;
            arr.iter()
                .map(op_to_mat4)
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    let m_mat = compose_chain_checked(&model_mats, "geom project.model")?;

    let cam = obj
        .get("camera")
        .and_then(|v| v.as_object())
        .ok_or_else(|| "geom project: missing 'camera' object".to_string())?;
    let eye = parse_vec3_value(
        cam.get("eye")
            .ok_or_else(|| "geom project: camera.eye missing".to_string())?,
        "camera.eye",
    )?;
    let target = parse_vec3_value(
        cam.get("target")
            .ok_or_else(|| "geom project: camera.target missing".to_string())?,
        "camera.target",
    )?;
    let up = parse_vec3_value(
        cam.get("up")
            .ok_or_else(|| "geom project: camera.up missing".to_string())?,
        "camera.up",
    )?;
    let fov_deg = cam
        .get("fov_deg")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| "geom project: camera.fov_deg missing".to_string())?;
    let aspect = match cam.get("aspect") {
        None | Some(Value::Null) => 1.0,
        Some(v) => v
            .as_f64()
            .ok_or_else(|| format!("geom project: camera.aspect must be a finite number, got {v}"))?,
    };
    let near = cam
        .get("near")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| "geom project: camera.near missing".to_string())?;
    let far = cam
        .get("far")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| "geom project: camera.far missing".to_string())?;

    let vp = obj
        .get("viewport")
        .and_then(|v| v.as_object())
        .ok_or_else(|| "geom project: missing 'viewport' object".to_string())?;
    let vp_w = vp
        .get("w")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| "geom project: viewport.w missing".to_string())?;
    let vp_h = vp
        .get("h")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| "geom project: viewport.h missing".to_string())?;
    if !vp_w.is_finite() || vp_w <= 0.0 || !vp_h.is_finite() || vp_h <= 0.0 {
        return Err(format!(
            "geom project: viewport w/h must be finite and > 0, got ({vp_w}, {vp_h})"
        ));
    }

    let v_mat = look_at4(eye, target, up)?;
    let p_mat = perspective4(fov_deg, aspect, near, far)?;
    let vm = check_mat4_finite(&mat4_mul(&v_mat, &m_mat), "geom project.vm")?;
    let mvp = check_mat4_finite(&mat4_mul(&p_mat, &vm), "geom project.mvp")?;
    let _ = mat4_to_col_major_f32_checked(&mvp, "geom project.mvp")?;
    Ok((mvp, vp_w, vp_h))
}

/// Project 3D vertices or hierarchical scene-tree nodes to 2D screen space via Burn `Device::flex()`.
/// Supports single-vertex `linear(4,4)` (`batched = false`) and block-diagonal `linear(4N,4N)`
/// chunked at `BATCH_MAX = 63` (`batched = true`), plus optional near-plane triangle clipping.
pub fn project_model(doc: &Value, batched: bool) -> Result<GeomProjectReport, String> {
    let obj = doc
        .as_object()
        .ok_or_else(|| "geom project: input document must be a JSON object".to_string())?;

    let mut vertices: Vec<Vec3> = Vec::new();
    let mut scene_summaries: Option<Vec<SceneNodeSummary>> = None;

    if let Some(verts_arr) = obj.get("vertices").and_then(|v| v.as_array())
        && !verts_arr.is_empty()
    {
        for (i, vv) in verts_arr.iter().enumerate() {
            vertices.push(parse_vec3_value(vv, &format!("vertices[{i}]"))?);
        }
    } else if obj.get("nodes").and_then(|v| v.as_array()).is_some() {
        let compiled = compile_scene_tree(doc)?;
        vertices = compiled.nodes.iter().map(|n| n.world_point).collect();
        scene_summaries = Some(compiled.nodes);
    }

    if vertices.is_empty() {
        return Err(
            "geom project: document needs non-empty 'vertices': [[x,y,z], ...] or 'nodes': [...]"
                .into(),
        );
    }

    let (mvp, vp_w, vp_h) = build_mvp_from_doc(obj)?;
    let (clips, chunks) = if batched {
        run_clips_batched(&mvp, &vertices)?
    } else {
        (run_clips_single(&mvp, &vertices)?, vertices.len())
    };

    let results: Vec<ProjectedVertex> = clips
        .iter()
        .map(|&c| project_clip(c, vp_w, vp_h))
        .collect::<Result<Vec<_>, _>>()?;
    let projected = results.iter().filter(|r| !r.behind).count();
    let behind_camera = results.len() - projected;

    let scene_nodes = scene_summaries.map(|nodes| {
        nodes
            .into_iter()
            .enumerate()
            .map(|(i, n)| ProjectedSceneNode {
                id: n.id,
                parent: n.parent,
                world_point: n.world_point,
                projection: results[i].clone(),
            })
            .collect()
    });

    let mut triangles_out = None;
    let mut clipped_away = None;
    let mut spans_cam = None;

    if let Some(tris_arr) = obj.get("triangles") {
        let arr = tris_arr
            .as_array()
            .ok_or_else(|| "geom project: 'triangles' must be an array of [a,b,c]".to_string())?;
        let mut out_tris = Vec::with_capacity(arr.len());
        let mut n_clipped_away = 0usize;
        let mut n_spans = 0usize;
        for (ti, tv) in arr.iter().enumerate() {
            let tri_arr = tv
                .as_array()
                .ok_or_else(|| format!("geom project: triangles[{ti}] must be [a,b,c]"))?;
            if tri_arr.len() != 3 {
                return Err(format!(
                    "geom project: triangles[{ti}] must have 3 indices, got {}",
                    tri_arr.len()
                ));
            }
            let mut idxs = [0usize; 3];
            for (k, iv) in tri_arr.iter().enumerate() {
                let idx = iv.as_u64().ok_or_else(|| {
                    format!("geom project: triangles[{ti}][{k}] must be a non-negative integer")
                })? as usize;
                if idx >= clips.len() {
                    return Err(format!(
                        "geom project: triangle {ti}: vertex index {idx} out of range (0..{})",
                        clips.len()
                    ));
                }
                idxs[k] = idx;
            }
            let res = clip_triangle_near(clips[idxs[0]], clips[idxs[1]], clips[idxs[2]]);
            if res.spans_camera {
                n_spans += 1;
            } else if res.triangles.is_empty() {
                n_clipped_away += 1;
            }
            let clipped_proj: Vec<[ProjectedVertex; 3]> = res
                .triangles
                .iter()
                .map(|tri| -> Result<[ProjectedVertex; 3], String> {
                    Ok([
                        project_clip(tri[0], vp_w, vp_h)?,
                        project_clip(tri[1], vp_w, vp_h)?,
                        project_clip(tri[2], vp_w, vp_h)?,
                    ])
                })
                .collect::<Result<Vec<_>, _>>()?;
            out_tris.push(ProjectedTriangle {
                input: idxs,
                spans_camera: res.spans_camera,
                clipped_count: clipped_proj.len(),
                clipped: clipped_proj,
            });
        }
        triangles_out = Some(out_tris);
        clipped_away = Some(n_clipped_away);
        spans_cam = Some(n_spans);
    }

    Ok(GeomProjectReport {
        ok: true,
        batched,
        chunks,
        vertices: vertices.len(),
        projected,
        behind_camera,
        scene_nodes,
        results,
        triangles: triangles_out,
        triangles_clipped_away: clipped_away,
        triangles_spans_camera: spans_cam,
    })
}

/// Self-contained 17-check geometry verification suite covering builders, `normal_mat3`,
/// near-plane clipping, cube projection vs independent scalar `f64` reference,
/// moved-camera negative control, behind-camera flagging, and batched-vs-single parity.
pub fn run_geom_selftest() -> Result<Value, String> {
    let close = |a: &[f64], b: &[f64], tol: f64| -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b.iter())
                .all(|(&x, &y)| (x - y).abs() < tol)
    };

    let mut checks: Vec<(&'static str, bool)> = vec![
        // 1..6: basic transform builders
        (
            "Rz(90)*[1,0,0] = [0,1,0]",
            close(&apply_mat4(&rot_z4(90.0)?, [1.0, 0.0, 0.0])?, &[0.0, 1.0, 0.0], 1e-9),
        ),
        (
            "Rx(90)*[0,1,0] = [0,0,1]",
            close(&apply_mat4(&rot_x4(90.0)?, [0.0, 1.0, 0.0])?, &[0.0, 0.0, 1.0], 1e-9),
        ),
        (
            "Ry(90)*[0,0,1] = [1,0,0]",
            close(&apply_mat4(&rot_y4(90.0)?, [0.0, 0.0, 1.0])?, &[1.0, 0.0, 0.0], 1e-9),
        ),
        (
            "Rz(45)^2*[1,0,0] = Rz(90)*[1,0,0]",
            close(
                &apply_mat4(&mat4_mul(&rot_z4(45.0)?, &rot_z4(45.0)?), [1.0, 0.0, 0.0])?,
                &apply_mat4(&rot_z4(90.0)?, [1.0, 0.0, 0.0])?,
                1e-9,
            ),
        ),
        (
            "translate*[1,2,3] + [10,0,0]",
            close(&apply_mat4(&translate4(10.0, 0.0, 0.0)?, [1.0, 2.0, 3.0])?, &[11.0, 2.0, 3.0], 1e-9),
        ),
        (
            "scale*[1,2,3]",
            close(&apply_mat4(&scale4(2.0, 3.0, 4.0)?, [1.0, 2.0, 3.0])?, &[2.0, 6.0, 12.0], 1e-9),
        ),
    ];
    let qz90 = quat_from_euler(0.0, 0.0, 90.0, "ZYX")?;
    checks.push((
        "quat(Rz90)*[1,0,0] = [0,1,0]",
        close(
            &apply_mat4(&quat_to_mat4(qz90[0], qz90[1], qz90[2], qz90[3])?, [1.0, 0.0, 0.0])?,
            &[0.0, 1.0, 0.0],
            1e-9,
        ),
    ));
    let q123 = quat_from_euler(10.0, 20.0, 30.0, "ZYX")?;
    checks.push((
        "euler(10,20,30) = quat(euler(10,20,30))",
        close(
            &apply_mat4(&euler4(10.0, 20.0, 30.0, "ZYX")?, [1.0, 2.0, 3.0])?,
            &apply_mat4(&quat_to_mat4(q123[0], q123[1], q123[2], q123[3])?, [1.0, 2.0, 3.0])?,
            1e-9,
        ),
    ));
    checks.push((
        "rotAxisAngle z 90 = rotZ 90",
        close(
            &apply_mat4(&rot_axis_angle4(0.0, 0.0, 5.0, 90.0)?, [1.0, 0.0, 0.0])?,
            &[0.0, 1.0, 0.0],
            1e-9,
        ),
    ));

    // 10..11: normalMat3 perpendicularity + naive negative control
    {
        let m = mat4_mul(
            &translate4(5.0, 0.0, 0.0)?,
            &mat4_mul(&rot_z4(30.0)?, &scale4(2.0, 1.0, 1.0)?),
        );
        let dir = |m4: &Mat4, v: Vec3| -> Vec3 {
            [
                m4[0] * v[0] + m4[1] * v[1] + m4[2] * v[2],
                m4[4] * v[0] + m4[5] * v[1] + m4[6] * v[2],
                m4[8] * v[0] + m4[9] * v[1] + m4[10] * v[2],
            ]
        };
        let dot = |a: Vec3, b: Vec3| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let t = [s, s, 0.0];
        let n = [-s, s, 0.0];
        let mt = dir(&m, t);
        let nn = apply_mat3(&normal_mat3(&m)?, n);
        let naive = dir(&m, n);
        checks.push((
            "normalMat3 preserves perpendicularity",
            dot(mt, nn).abs() < 1e-9,
        ));
        checks.push((
            "naive model-matrix breaks perpendicularity (control)",
            dot(mt, naive).abs() > 0.1,
        ));
    }

    // 12..15: near-plane Sutherland–Hodgman clipping
    {
        let r = clip_triangle_near([0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
        checks.push(("clip near: all-inside -> 1 triangle", r.triangles.len() == 1 && !r.spans_camera));
    }
    {
        let r = clip_triangle_near([0.0, 0.0, -2.0, 1.0], [1.0, 0.0, -2.0, 1.0], [0.0, 1.0, -2.0, 1.0]);
        checks.push(("clip near: all-outside -> 0 triangles", r.triangles.is_empty() && !r.spans_camera));
    }
    {
        let r = clip_triangle_near([0.0, 0.0, 0.5, 1.0], [0.0, 0.0, -2.0, 1.0], [1.0, 0.0, 0.5, 1.0]);
        let on_plane = r
            .triangles
            .iter()
            .flat_map(|t| t.iter())
            .all(|v| v[2] + v[3] >= -1e-9);
        checks.push((
            "clip near: straddle -> 2 triangles on/inside plane",
            r.triangles.len() == 2 && on_plane && !r.spans_camera,
        ));
    }
    {
        let r = clip_triangle_near([0.0, 0.0, 0.0, -1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
        checks.push((
            "clip near: w<=0 -> spans_camera flag",
            r.spans_camera && r.triangles.is_empty(),
        ));
    }

    // 16..17: cube projection vs independent scalar f64 ref + negative control + batched-vs-single parity
    let mut verts = Vec::with_capacity(9);
    for &sx in &[-1.0, 1.0] {
        for &sy in &[-1.0, 1.0] {
            for &sz in &[-1.0, 1.0] {
                verts.push([sx, sy, sz]);
            }
        }
    }
    verts.push([0.0, 0.0, 12.0]); // behind camera after translate(0, 0, -8)

    let model_doc = json!({
        "vertices": verts,
        "model": [
            { "op": "rotate_y", "degrees": 30.0 },
            { "op": "translate", "by": [0.0, 0.0, -8.0] }
        ],
        "camera": {
            "eye": [0.0, 0.0, 0.0],
            "target": [0.0, 0.0, -1.0],
            "up": [0.0, 1.0, 0.0],
            "fov_deg": 60.0,
            "aspect": 4.0 / 3.0,
            "near": 0.1,
            "far": 100.0
        },
        "viewport": { "w": 800.0, "h": 600.0 }
    });

    let single = project_model(&model_doc, false)?;
    let batched = project_model(&model_doc, true)?;

    // Independent scalar f64 projection for cube (no geom3 builder calls)
    let ang = 30.0f64 * std::f64::consts::PI / 180.0;
    let (s30, c30) = ang.sin_cos();
    let f = 1.0 / (30.0f64 * std::f64::consts::PI / 180.0).tan();
    let asp = 4.0 / 3.0;
    let (near, far) = (0.1f64, 100.0f64);
    let mut max_px_err = 0.0f64;
    let mut behind_ok = true;
    for (i, &[vx, vy, vz]) in verts.iter().enumerate() {
        // Rotate Y by 30 deg, then translate by (0, 0, -8); camera is at origin looking down -z
        let rx = c30 * vx + s30 * vz;
        let ry = vy;
        let rz = -s30 * vx + c30 * vz - 8.0;
        let cx = (f / asp) * rx;
        let cy = f * ry;
        let cw = -rz;
        let ref_behind = !cw.is_finite() || cw <= 0.0;
        if single.results[i].behind != ref_behind {
            behind_ok = false;
        }
        if !ref_behind
            && let Some([sx, sy]) = single.results[i].screen
        {
            let ref_sx = ((cx / cw) * 0.5 + 0.5) * 800.0;
            let ref_sy = (1.0 - ((cy / cw) * 0.5 + 0.5)) * 600.0;
            max_px_err = max_px_err.max((sx - ref_sx).abs()).max((sy - ref_sy).abs());
        }
        let _ = (near, far);
    }

    // Moved camera negative control
    let mut moved_doc = model_doc.clone();
    if let Some(cam) = moved_doc.get_mut("camera").and_then(|v| v.as_object_mut()) {
        cam.insert("eye".to_string(), json!([2.5, 0.0, 0.0]));
    }
    let moved = project_model(&moved_doc, false)?;
    let mut ctrl_diff = 0.0f64;
    for i in 0..verts.len() {
        if let (Some(a), Some(b)) = (single.results[i].screen, moved.results[i].screen) {
            ctrl_diff = ctrl_diff
                .max((a[0] - b[0]).abs())
                .max((a[1] - b[1]).abs());
        }
    }

    let proj_ok =
        max_px_err <= 1e-4 && behind_ok && single.results[8].behind && ctrl_diff > 1.0;
    checks.push((
        "cube projection vs scalar f64 ref + behind-flag + moved-camera control",
        proj_ok,
    ));

    // Batched vs single parity
    let mut batch_max_diff = 0.0f64;
    let mut batch_behind_ok = true;
    for (a, b) in single.results.iter().zip(batched.results.iter()) {
        if a.behind != b.behind {
            batch_behind_ok = false;
        }
        for k in 0..4 {
            batch_max_diff = batch_max_diff.max((a.clip[k] - b.clip[k]).abs());
        }
    }
    checks.push((
        "batched linear(4N,4N) matches single linear(4,4) within 1e-6",
        batch_behind_ok && batch_max_diff <= 1e-6,
    ));

    let passed = checks.iter().filter(|(_, ok)| *ok).count();
    let total = checks.len();
    let all_ok = passed == total;
    Ok(json!({
        "ok": all_ok,
        "passed": passed,
        "total": total,
        "max_abs_screen_px": max_px_err,
        "negative_control_moved_camera_px": ctrl_diff,
        "batch_vs_single_max_clip_diff": batch_max_diff,
        "checks": checks.iter().map(|(name, ok)| json!({ "name": name, "ok": ok })).collect::<Vec<_>>(),
    }))
}
