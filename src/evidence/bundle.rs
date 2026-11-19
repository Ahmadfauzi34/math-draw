use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashSet;

use crate::engine::get_layer_weight_len;
use crate::spec::{SpecV1, sha256_f32_slice, sha256_json};
use crate::trace::{DEFAULT_MAX_TENSOR_BYTES, trace_hash_spec};
use crate::validate::validate_spec;

pub const PARAMETER_BINDING_SCHEMA: &str = "burn-math.graph-parameter-binding.v1";
pub const PARAMETER_ORDERING: &str = "unique_first_use_graph_plan";
pub const PROGRAM_BUNDLE_SCHEMA: &str = "burn-math.program-bundle.v1";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GraphParameterOwner {
    pub layer_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub registry_id: Option<u64>,
    pub constructor: String,
    pub offset: usize,
    pub len: usize,
    pub trainable: bool,
    pub explicit_weights: bool,
    pub weight_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GraphParameterBinding {
    pub schema: String,
    pub ordering: String,
    pub program_identity: String,
    pub total_params: usize,
    pub trainable_params: usize,
    pub stateless_layers_count: usize,
    pub unique_referenced_layers: usize,
    pub owners: Vec<GraphParameterOwner>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BundleExecutionReceipt {
    pub output_shape: [usize; 4],
    pub output_values: Vec<f32>,
    pub outputs_sha256: String,
    pub trace_sha256: String,
    pub captured_steps: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleAttestation {
    pub label: String,
    pub kind: String,
    pub subject_identity: String,
    pub digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProgramBundle {
    pub schema: String,
    pub program_identity: String,
    pub spec: SpecV1,
    pub parameter_binding: GraphParameterBinding,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_execution: Option<BundleExecutionReceipt>,
    pub attestations: Vec<BundleAttestation>,
    pub bundle_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleVerifyCheck {
    pub name: String,
    pub passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleVerifyReport {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    pub program_identity: String,
    pub bundle_sha256: String,
    pub total_params: usize,
    pub trainable_params: usize,
    pub checked_attestations: usize,
    pub replayed_execution: bool,
    pub checks: Vec<BundleVerifyCheck>,
}

/// Structural program identity independent of input data and weight values.
pub fn compute_program_identity(spec: &SpecV1) -> String {
    let topology = json!({
        "numSlots": spec.num_slots,
        "outputSlot": spec.output_slot,
        "ports": &spec.ports,
        "layers": &spec.layers,
        "steps": &spec.steps,
    });
    sha256_json(&topology)
}

fn resolve_layer_weights(spec: &SpecV1, layer_idx: usize) -> Result<(Vec<f32>, bool), String> {
    let decl = spec
        .layers
        .get(layer_idx)
        .ok_or_else(|| format!("graph parameter binding: layer index {layer_idx} out of range"))?;
    let len = get_layer_weight_len(decl);
    if len == 0 {
        return Ok((Vec::new(), false));
    }

    if let Some(existing) = spec.weights.get(&layer_idx.to_string()) {
        if existing.len() != len {
            return Err(format!(
                "graph parameter binding: layer {layer_idx} ({}) expected {len} weights, got {}",
                decl.constructor,
                existing.len()
            ));
        }
        if let Some((i, _)) = existing.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(format!(
                "graph parameter binding: layer {layer_idx} weight[{i}] is non-finite"
            ));
        }
        return Ok((existing.clone(), true));
    }

    let defaults = match decl.constructor.as_str() {
        "batchNorm" | "groupNorm" | "instanceNorm" | "layerNorm" => {
            let mut v = vec![0.0f32; len];
            for x in v.iter_mut().take(len / 2) {
                *x = 1.0;
            }
            v
        }
        "rmsNorm" => vec![1.0f32; len],
        "prelu" => {
            let alpha = decl
                .args
                .get(2)
                .and_then(|v| v.as_f64())
                .unwrap_or(0.25) as f32;
            vec![alpha; len]
        }
        _ => vec![0.0f32; len],
    };
    Ok((defaults, false))
}

/// Build canonical unique-first-use parameter binding for a validated `SpecV1`.
///
/// Deduplicates referenced layers in structural `spec.steps` execution order so weight-tied
/// layers appear once with a single canonical flat parameter slice.
pub fn compute_parameter_binding(spec: &SpecV1) -> Result<GraphParameterBinding, String> {
    let val = validate_spec(spec, false);
    if !val.valid {
        let detail = if !val.issues.is_empty() {
            val.issues.join("; ")
        } else if !val.unknown_constructors.is_empty() {
            format!("unknown constructors: {}", val.unknown_constructors.join(", "))
        } else {
            val.stage.to_string()
        };
        return Err(format!(
            "graph parameter binding: invalid spec ({}: {detail})",
            val.code
        ));
    }

    let program_identity = compute_program_identity(spec);
    let mut seen: HashSet<usize> = HashSet::new();
    let mut owners: Vec<GraphParameterOwner> = Vec::new();
    let mut total_params = 0usize;
    let mut trainable_params = 0usize;
    let mut stateless_layers_count = 0usize;

    for step in &spec.steps {
        if !seen.insert(step.layer) {
            continue;
        }
        let decl = &spec.layers[step.layer];
        let len = get_layer_weight_len(decl);
        if len == 0 {
            stateless_layers_count += 1;
            continue;
        }
        let (weights, explicit_weights) = resolve_layer_weights(spec, step.layer)?;
        let offset = total_params;
        total_params = total_params
            .checked_add(len)
            .ok_or_else(|| "graph parameter binding: total parameter length overflow".to_string())?;
        let trainable = decl.constructor != "embedding";
        if trainable {
            trainable_params = trainable_params.checked_add(len).ok_or_else(|| {
                "graph parameter binding: trainable parameter length overflow".to_string()
            })?;
        }
        let registry_id = decl.args.first().and_then(|v| v.as_u64());
        owners.push(GraphParameterOwner {
            layer_index: step.layer,
            registry_id,
            constructor: decl.constructor.clone(),
            offset,
            len,
            trainable,
            explicit_weights,
            weight_sha256: sha256_f32_slice(&weights),
        });
    }

    Ok(GraphParameterBinding {
        schema: PARAMETER_BINDING_SCHEMA.to_string(),
        ordering: PARAMETER_ORDERING.to_string(),
        program_identity,
        total_params,
        trainable_params,
        stateless_layers_count,
        unique_referenced_layers: seen.len(),
        owners,
    })
}

/// Read all bound layer weights into a single flat vector matching `binding.owners`.
pub fn read_bound_flat_weights(
    spec: &SpecV1,
    binding: &GraphParameterBinding,
) -> Result<Vec<f32>, String> {
    let cur_id = compute_program_identity(spec);
    if cur_id != binding.program_identity {
        return Err(
            "graph parameter read: spec structural identity differs from binding identity".into(),
        );
    }
    let mut out = Vec::with_capacity(binding.total_params);
    for owner in &binding.owners {
        let (weights, _) = resolve_layer_weights(spec, owner.layer_index)?;
        if weights.len() != owner.len {
            return Err(format!(
                "graph parameter read: layer {} length mismatch (expected {}, got {})",
                owner.layer_index,
                owner.len,
                weights.len()
            ));
        }
        out.extend_from_slice(&weights);
    }
    if out.len() != binding.total_params {
        return Err(format!(
            "graph parameter read: total length mismatch (expected {}, got {})",
            binding.total_params,
            out.len()
        ));
    }
    Ok(out)
}

/// Atomically apply a flat candidate parameter vector according to `binding`,
/// failing closed before any weight mutation if identity, length, or finiteness checks fail.
pub fn apply_bound_flat_weights(
    spec: &SpecV1,
    binding: &GraphParameterBinding,
    candidate: &[f32],
) -> Result<SpecV1, String> {
    let cur_id = compute_program_identity(spec);
    if cur_id != binding.program_identity {
        return Err(
            "graph parameter apply: spec structural identity differs from binding identity".into(),
        );
    }
    if candidate.len() != binding.total_params {
        return Err(format!(
            "graph parameter apply: expected {} floats, got {}",
            binding.total_params,
            candidate.len()
        ));
    }
    if let Some((idx, _)) = candidate.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(format!(
            "graph parameter apply: candidate contains non-finite value at index {idx}"
        ));
    }
    for owner in &binding.owners {
        let end = owner
            .offset
            .checked_add(owner.len)
            .ok_or_else(|| "graph parameter apply: slice boundary overflow".to_string())?;
        if end > candidate.len() {
            return Err(format!(
                "graph parameter apply: layer {} slice {}..{} exceeds candidate length {}",
                owner.layer_index,
                owner.offset,
                end,
                candidate.len()
            ));
        }
        let decl = spec.layers.get(owner.layer_index).ok_or_else(|| {
            format!(
                "graph parameter apply: layer {} missing from spec",
                owner.layer_index
            )
        })?;
        let expected_len = get_layer_weight_len(decl);
        if expected_len != owner.len {
            return Err(format!(
                "graph parameter apply: layer {} expected {} weights, binding has {}",
                owner.layer_index, expected_len, owner.len
            ));
        }
    }

    let mut updated = spec.clone();
    for owner in &binding.owners {
        let end = owner.offset + owner.len;
        updated.weights.insert(
            owner.layer_index.to_string(),
            candidate[owner.offset..end].to_vec(),
        );
    }
    Ok(updated)
}

fn compute_attestation_digest(
    kind: &str,
    label: &str,
    subject_identity: &str,
    binding: &GraphParameterBinding,
    receipt: Option<&BundleExecutionReceipt>,
) -> Result<String, String> {
    match kind {
        "program_binding" => Ok(sha256_json(&json!({
            "kind": kind,
            "label": label,
            "subject_identity": subject_identity,
            "binding_sha256": sha256_json(binding),
        }))),
        "execution_receipt" => {
            let Some(exec) = receipt else {
                return Err(
                    "execution_receipt attestation requires reference_execution to be present"
                        .into(),
                );
            };
            Ok(sha256_json(&json!({
                "kind": kind,
                "label": label,
                "subject_identity": subject_identity,
                "outputs_sha256": exec.outputs_sha256,
                "trace_sha256": exec.trace_sha256,
                "output_shape": exec.output_shape,
                "captured_steps": exec.captured_steps,
            })))
        }
        "user_attestation" => Ok(sha256_json(&json!({
            "kind": kind,
            "label": label,
            "subject_identity": subject_identity,
            "binding_sha256": sha256_json(binding),
            "receipt_sha256": receipt.map(sha256_json),
        }))),
        other => Err(format!("unknown bundle attestation kind '{other}'")),
    }
}

fn compute_bundle_sha256(
    schema: &str,
    program_identity: &str,
    spec: &SpecV1,
    parameter_binding: &GraphParameterBinding,
    reference_execution: &Option<BundleExecutionReceipt>,
    attestations: &[BundleAttestation],
) -> String {
    sha256_json(&json!({
        "schema": schema,
        "program_identity": program_identity,
        "spec": spec,
        "parameter_binding": parameter_binding,
        "reference_execution": reference_execution,
        "attestations": attestations,
    }))
}

/// Export a self-contained, cryptographically attested portable `ProgramBundle`.
pub fn export_bundle(
    spec: &SpecV1,
    include_execution: bool,
    attest_labels: &[String],
) -> Result<ProgramBundle, String> {
    let binding = compute_parameter_binding(spec)?;
    let program_identity = binding.program_identity.clone();

    let reference_execution = if include_execution {
        let trace = trace_hash_spec(spec, DEFAULT_MAX_TENSOR_BYTES)?;
        let trace_sha256 = sha256_json(&trace.steps);
        Some(BundleExecutionReceipt {
            output_shape: trace.output.shape,
            output_values: trace.output.values,
            outputs_sha256: trace.output_sha256,
            trace_sha256,
            captured_steps: trace.captured_steps,
        })
    } else {
        None
    };

    let mut attestations = Vec::new();
    let binding_digest = compute_attestation_digest(
        "program_binding",
        "canonical_parameter_binding",
        &program_identity,
        &binding,
        reference_execution.as_ref(),
    )?;
    attestations.push(BundleAttestation {
        label: "canonical_parameter_binding".to_string(),
        kind: "program_binding".to_string(),
        subject_identity: program_identity.clone(),
        digest: binding_digest,
    });

    if reference_execution.is_some() {
        let exec_digest = compute_attestation_digest(
            "execution_receipt",
            "deterministic_flex_replay",
            &program_identity,
            &binding,
            reference_execution.as_ref(),
        )?;
        attestations.push(BundleAttestation {
            label: "deterministic_flex_replay".to_string(),
            kind: "execution_receipt".to_string(),
            subject_identity: program_identity.clone(),
            digest: exec_digest,
        });
    }

    for label in attest_labels {
        let trimmed = label.trim();
        if trimmed.is_empty() {
            return Err("export_bundle: attestation label must be non-empty".into());
        }
        let user_digest = compute_attestation_digest(
            "user_attestation",
            trimmed,
            &program_identity,
            &binding,
            reference_execution.as_ref(),
        )?;
        attestations.push(BundleAttestation {
            label: trimmed.to_string(),
            kind: "user_attestation".to_string(),
            subject_identity: program_identity.clone(),
            digest: user_digest,
        });
    }

    let bundle_sha256 = compute_bundle_sha256(
        PROGRAM_BUNDLE_SCHEMA,
        &program_identity,
        spec,
        &binding,
        &reference_execution,
        &attestations,
    );

    Ok(ProgramBundle {
        schema: PROGRAM_BUNDLE_SCHEMA.to_string(),
        program_identity,
        spec: spec.clone(),
        parameter_binding: binding,
        reference_execution,
        attestations,
        bundle_sha256,
    })
}

/// Fail-closed verification of a `ProgramBundle`: schema, SHA-256 tamper check,
/// static spec + parameter binding parity, live Burn `flex` replay, and provenance digests.
pub fn verify_bundle(bundle: &ProgramBundle) -> BundleVerifyReport {
    let mut checks: Vec<BundleVerifyCheck> = Vec::new();
    let mut first_code: Option<String> = None;
    let mut record = |name: &str, passed: bool, code: Option<&str>, detail: String| {
        if !passed && first_code.is_none() {
            first_code = code.map(String::from);
        }
        checks.push(BundleVerifyCheck {
            name: name.to_string(),
            passed,
            code: if passed { None } else { code.map(String::from) },
            detail,
        });
    };

    // 1. Schema check
    let schema_ok = bundle.schema == PROGRAM_BUNDLE_SCHEMA;
    record(
        "schema",
        schema_ok,
        Some("invalid_bundle_schema"),
        if schema_ok {
            format!("schema={}", bundle.schema)
        } else {
            format!(
                "expected schema '{PROGRAM_BUNDLE_SCHEMA}', got '{}'",
                bundle.schema
            )
        },
    );

    // 2. Cryptographic bundle integrity check
    let recomputed_sha = compute_bundle_sha256(
        &bundle.schema,
        &bundle.program_identity,
        &bundle.spec,
        &bundle.parameter_binding,
        &bundle.reference_execution,
        &bundle.attestations,
    );
    let integrity_ok = recomputed_sha == bundle.bundle_sha256;
    record(
        "bundle_integrity",
        integrity_ok,
        Some("bundle_integrity_failed"),
        if integrity_ok {
            format!("bundle_sha256 verified ({})", bundle.bundle_sha256)
        } else {
            format!(
                "bundle_sha256 mismatch: expected {}, recomputed {recomputed_sha}",
                bundle.bundle_sha256
            )
        },
    );

    // 3. Static validation + parameter binding & structural identity parity
    match compute_parameter_binding(&bundle.spec) {
        Ok(recomputed) => {
            let id_ok = recomputed.program_identity == bundle.program_identity;
            let bind_ok = recomputed == bundle.parameter_binding;
            let ok = id_ok && bind_ok;
            record(
                "parameter_binding",
                ok,
                Some("parameter_binding_mismatch"),
                if ok {
                    format!(
                        "owners={} total_params={} trainable_params={}",
                        recomputed.owners.len(),
                        recomputed.total_params,
                        recomputed.trainable_params
                    )
                } else if !id_ok {
                    format!(
                        "program_identity mismatch: bundle={}, recomputed={}",
                        bundle.program_identity, recomputed.program_identity
                    )
                } else {
                    "recomputed graph parameter binding differs from embedded binding".to_string()
                },
            );
        }
        Err(e) => {
            record(
                "parameter_binding",
                false,
                Some("parameter_binding_mismatch"),
                e,
            );
        }
    }

    // 4. Live execution replay against reference_execution (if embedded)
    let mut replayed_execution = false;
    if let Some(ref expected) = bundle.reference_execution {
        replayed_execution = true;
        match trace_hash_spec(&bundle.spec, DEFAULT_MAX_TENSOR_BYTES) {
            Ok(trace) => {
                let actual_trace_sha = sha256_json(&trace.steps);
                let shape_ok = trace.output.shape == expected.output_shape;
                let out_sha_ok = trace.output_sha256 == expected.outputs_sha256;
                let trace_sha_ok = actual_trace_sha == expected.trace_sha256;
                let steps_ok = trace.captured_steps == expected.captured_steps;
                let values_sha_ok =
                    sha256_f32_slice(&expected.output_values) == expected.outputs_sha256;
                let ok = shape_ok && out_sha_ok && trace_sha_ok && steps_ok && values_sha_ok;
                record(
                    "execution_replay",
                    ok,
                    Some("execution_receipt_mismatch"),
                    if ok {
                        format!(
                            "replayed shape={:?} steps={} outputs_sha256={}",
                            trace.output.shape, trace.captured_steps, trace.output_sha256
                        )
                    } else {
                        format!(
                            "replay mismatch (shape_ok={shape_ok}, out_sha_ok={out_sha_ok}, trace_sha_ok={trace_sha_ok}, steps_ok={steps_ok}, values_sha_ok={values_sha_ok})"
                        )
                    },
                );
            }
            Err(e) => {
                record(
                    "execution_replay",
                    false,
                    Some("execution_receipt_mismatch"),
                    format!("replay failed: {e}"),
                );
            }
        }
    }

    // 5. Provenance attestation digest verification
    let mut att_ok = true;
    let mut att_err = String::new();
    for att in &bundle.attestations {
        if att.subject_identity != bundle.program_identity {
            att_ok = false;
            att_err = format!(
                "attestation '{}' subject_identity mismatch ({})",
                att.label, att.subject_identity
            );
            break;
        }
        match compute_attestation_digest(
            &att.kind,
            &att.label,
            &att.subject_identity,
            &bundle.parameter_binding,
            bundle.reference_execution.as_ref(),
        ) {
            Ok(expected_digest) => {
                if expected_digest != att.digest {
                    att_ok = false;
                    att_err = format!(
                        "attestation '{}' ({}) digest mismatch: expected {expected_digest}, got {}",
                        att.label, att.kind, att.digest
                    );
                    break;
                }
            }
            Err(e) => {
                att_ok = false;
                att_err = format!("attestation '{}' invalid: {e}", att.label);
                break;
            }
        }
    }
    record(
        "attestation_provenance",
        att_ok,
        Some("attestation_mismatch"),
        if att_ok {
            format!("verified {} attestation(s)", bundle.attestations.len())
        } else {
            att_err
        },
    );

    let ok = checks.iter().all(|c| c.passed);
    BundleVerifyReport {
        ok,
        code: first_code,
        program_identity: bundle.program_identity.clone(),
        bundle_sha256: bundle.bundle_sha256.clone(),
        total_params: bundle.parameter_binding.total_params,
        trainable_params: bundle.parameter_binding.trainable_params,
        checked_attestations: bundle.attestations.len(),
        replayed_execution,
        checks,
    }
}
