use serde::Serialize;
use std::fmt;

/// Machine-readable error codes for `math-draw` library and CLI diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    // CLI argument parsing & command dispatch
    CliUsage,
    CliUnknownCommand,
    CliUnknownFlag,
    CliMissingValue,
    CliDuplicateFlag,
    CliUnexpectedPositional,
    CliInvalidNumber,
    CliInvalidRange,

    // Filesystem I/O & JSON codecs
    IoReadFailed,
    IoWriteFailed,
    JsonParseFailed,
    JsonSerializeFailed,

    // Spec, graph validation & structural transforms
    SpecInvalid,
    ValidationFailed,
    RewriteFailed,
    ShardFailed,

    // Runtime & numeric math domains
    DivisionByZero,
    IntOverflow,
    NonFiniteOutput,
    ShapeMismatch,
    MathDomainError,
    ExecutionFailed,

    // Optimization, verification & evidence
    EsConfigInvalid,
    MissingExpectedTarget,
    QuantBoundUnsupported,
    RepertoireInvalid,
    BundleInvalid,

    // Harness workspace & workflows
    WorkspaceCorrupted,
    ArtifactIntegrityFailed,
    WorkflowFailed,

    // Fallback
    InternalError,
}

impl ErrorCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CliUsage => "cli_usage",
            Self::CliUnknownCommand => "cli_unknown_command",
            Self::CliUnknownFlag => "cli_unknown_flag",
            Self::CliMissingValue => "cli_missing_value",
            Self::CliDuplicateFlag => "cli_duplicate_flag",
            Self::CliUnexpectedPositional => "cli_unexpected_positional",
            Self::CliInvalidNumber => "cli_invalid_number",
            Self::CliInvalidRange => "cli_invalid_range",
            Self::IoReadFailed => "io_read_failed",
            Self::IoWriteFailed => "io_write_failed",
            Self::JsonParseFailed => "json_parse_failed",
            Self::JsonSerializeFailed => "json_serialize_failed",
            Self::SpecInvalid => "spec_invalid",
            Self::ValidationFailed => "validation_failed",
            Self::RewriteFailed => "rewrite_failed",
            Self::ShardFailed => "shard_failed",
            Self::DivisionByZero => "division_by_zero",
            Self::IntOverflow => "int_overflow",
            Self::NonFiniteOutput => "non_finite_output",
            Self::ShapeMismatch => "shape_mismatch",
            Self::MathDomainError => "math_domain_error",
            Self::ExecutionFailed => "execution_failed",
            Self::EsConfigInvalid => "es_config_invalid",
            Self::MissingExpectedTarget => "missing_expected_target",
            Self::QuantBoundUnsupported => "quant_bound_unsupported",
            Self::RepertoireInvalid => "repertoire_invalid",
            Self::BundleInvalid => "bundle_invalid",
            Self::WorkspaceCorrupted => "workspace_corrupted",
            Self::ArtifactIntegrityFailed => "artifact_integrity_failed",
            Self::WorkflowFailed => "workflow_failed",
            Self::InternalError => "internal_error",
        }
    }

    pub const fn default_hint(self) -> Option<&'static str> {
        match self {
            Self::CliUsage | Self::CliUnknownCommand => {
                Some("Run `math-draw --help` to see available commands and usage syntax.")
            }
            Self::CliUnknownFlag => {
                Some("Check flag spelling or run `math-draw --help` to list supported flags.")
            }
            Self::CliMissingValue => {
                Some("Provide an explicit value immediately after this flag.")
            }
            Self::CliDuplicateFlag => {
                Some("Pass this single-valued flag at most once per command invocation.")
            }
            Self::CliUnexpectedPositional => {
                Some("Remove extra positional arguments or check flag syntax via `math-draw --help`.")
            }
            Self::CliInvalidNumber | Self::CliInvalidRange => {
                Some("Provide finite numeric values within the valid domain for this option.")
            }
            Self::IoReadFailed => {
                Some("Verify that the file path exists and is readable.")
            }
            Self::IoWriteFailed => {
                Some("Verify that the target directory exists and is writable.")
            }
            Self::JsonParseFailed => {
                Some("Validate the input file is well-formed JSON matching the expected schema.")
            }
            Self::JsonSerializeFailed => {
                Some("Ensure all output values are finite and JSON-serializable.")
            }
            Self::SpecInvalid | Self::ValidationFailed => {
                Some("Run `math-draw validate <spec.json> --explain` to inspect structural diagnostics.")
            }
            Self::RewriteFailed => {
                Some("Run `math-draw rewrite list` to inspect supported rewrite rules.")
            }
            Self::ShardFailed => {
                Some("Increase `--max-weights` or reduce layer input dimension so each shard fits.")
            }
            Self::DivisionByZero => {
                Some("Ensure divisor tensors/constants are non-zero across all elements.")
            }
            Self::IntOverflow => {
                Some("Keep integer tensor values within checked i64 bounds.")
            }
            Self::NonFiniteOutput | Self::MathDomainError => {
                Some("Ensure inputs and weights stay within the valid finite mathematical domain.")
            }
            Self::ShapeMismatch => {
                Some("Verify input port shapes, tensor broadcast compatibility, and layer dimensions.")
            }
            Self::ExecutionFailed => {
                Some("Run `math-draw validate <spec.json> --explain` and `math-draw trace-hash <spec.json>` to isolate the failing step.")
            }
            Self::EsConfigInvalid => {
                Some("Check ES hyperparameters: gens >= 1, pop >= 2 (even for openes), 1 <= mu <= pop (mu-lambda), finite positive sigma/lr.")
            }
            Self::MissingExpectedTarget => {
                Some("Include an 'expected' numeric output vector in every case inside the targets JSON.")
            }
            Self::QuantBoundUnsupported => {
                Some("Use `--mode measured` for graphs containing unbounded ops, or restrict to Lipschitz-bounded ops.")
            }
            Self::RepertoireInvalid => {
                Some("Verify repertoire JSON contains a valid 'sets' array of named WeightSet objects.")
            }
            Self::BundleInvalid => {
                Some("Re-export the bundle with `math-draw bundle export` and verify no fields were altered.")
            }
            Self::WorkspaceCorrupted => {
                Some("Inspect or recreate `workspace-state.json` in the harness workspace directory.")
            }
            Self::ArtifactIntegrityFailed => {
                Some("Artifact payload does not match its SHA-256 URI; regenerate the artifact cleanly.")
            }
            Self::WorkflowFailed => {
                Some("Inspect `failed_stage` and `stage_reports` in the workflow output.")
            }
            Self::InternalError => None,
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Structured error type carrying a stable `ErrorCode`, human message, optional stage, and recovery hint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MathDrawError {
    pub code: ErrorCode,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

/// Canonical CLI JSON error envelope preserving backward-compatible `ok` and `error` keys
/// while adding machine-readable `code`, `stage`, and `hint`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ErrorEnvelope {
    pub ok: bool,
    pub code: ErrorCode,
    pub error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

impl MathDrawError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            stage: None,
            hint: code.default_hint().map(String::from),
        }
    }

    pub fn with_stage(mut self, stage: impl Into<String>) -> Self {
        self.stage = Some(stage.into());
        self
    }

    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    pub fn to_envelope(&self) -> ErrorEnvelope {
        ErrorEnvelope {
            ok: false,
            code: self.code,
            error: self.message.clone(),
            stage: self.stage.clone(),
            hint: self.hint.clone(),
        }
    }
}

impl fmt::Display for MathDrawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for MathDrawError {}

impl From<MathDrawError> for String {
    fn from(err: MathDrawError) -> Self {
        err.message
    }
}

/// Deterministically classifies a domain `String` error into a structured [`MathDrawError`].
pub fn classify_error(msg: &str, stage: Option<&str>) -> MathDrawError {
    let low = msg.to_lowercase();
    let code = if low.contains("artifact_integrity_failed") {
        ErrorCode::ArtifactIntegrityFailed
    } else if low.contains("corrupted workspace state") || low.contains("workspace state") {
        ErrorCode::WorkspaceCorrupted
    } else if low.contains("unknown flag") {
        ErrorCode::CliUnknownFlag
    } else if low.contains("requires a value") {
        ErrorCode::CliMissingValue
    } else if low.contains("duplicate flag") {
        ErrorCode::CliDuplicateFlag
    } else if low.contains("unexpected positional") {
        ErrorCode::CliUnexpectedPositional
    } else if low.contains("unknown command")
        || (low.contains("unknown ")
            && (low.contains("subcommand")
                || low.contains("arith op")
                || low.contains("int op")
                || low.contains("kernel op")
                || low.contains("geom op")))
    {
        ErrorCode::CliUnknownCommand
    } else if low.starts_with("usage:")
        || low.contains("unknown quant scheme")
        || low.contains("unknown budget mode")
        || low.contains("grad-spec: order must be")
        || low.contains("grad-spec: --hvp-vec")
        || low.contains("unknown or unsupported batch field")
        || (low.starts_with("batch[") && low.contains("specify either "))
    {
        ErrorCode::CliUsage
    } else if low.contains("cannot read") || low.contains("failed to read") {
        ErrorCode::IoReadFailed
    } else if low.contains("failed writing")
        || low.contains("failed to write")
        || low.contains("failed to create")
    {
        ErrorCode::IoWriteFailed
    } else if low.contains("failed to serialize") {
        ErrorCode::JsonSerializeFailed
    } else if low.contains("repertoire") {
        ErrorCode::RepertoireInvalid
    } else if low.contains("invalid json")
        || low.contains("failed to parse")
        || (low.contains("invalid ") && low.contains(" json"))
    {
        ErrorCode::JsonParseFailed
    } else if low.contains("invalid v1 spec")
        || low.contains("recursive-spec.v2")
        || low.contains("unknown_layer_constructor")
        || low.contains("unsupported_symbolic_grad_op")
        || low.contains("unresolved_wrt_slot")
        || low.contains("cycle detected")
        || low.contains("duplicate node id")
        || low.contains("unknown parent node")
        || low.contains("unsupported constructor")
    {
        ErrorCode::SpecInvalid
    } else if low.contains("missing 'expected'")
        || low.contains("missing expected")
        || low.contains("targets.cases")
    {
        ErrorCode::MissingExpectedTarget
    } else if low.starts_with("workflow ") || low.contains("unknown workflow pipeline stage") {
        ErrorCode::WorkflowFailed
    } else if low.contains("unknown es strategy")
        || low.contains("unknown rng")
        || low.starts_with("fit: ")
        || (matches!(stage, Some("recurse.fit" | "fit" | "fit_hybrid"))
            && (low.contains("pop")
                || low.contains("generations")
                || low.contains("mu")
                || low.contains("strategy")
                || low.contains("rng")
                || low.contains("sigma")
                || low.contains("lr")))
    {
        ErrorCode::EsConfigInvalid
    } else if low.contains("division by zero")
        || low.contains("modulo by zero")
        || low.contains("divide by zero")
        || low.contains("zero denominator")
    {
        ErrorCode::DivisionByZero
    } else if low.contains("overflow") {
        ErrorCode::IntOverflow
    } else if low.contains("non-finite") || low.contains("nan") || low.contains("must be finite") {
        ErrorCode::NonFiniteOutput
    } else if low.contains("unshardable") {
        ErrorCode::ShardFailed
    } else if low.contains("unknown rewrite") || low.contains("no rewrite patterns matched") {
        ErrorCode::RewriteFailed
    } else if low.contains("unbounded") || low.contains("bound mode") {
        ErrorCode::QuantBoundUnsupported
    } else if low.contains("bundle") {
        ErrorCode::BundleInvalid
    } else if low.contains("shape") || low.contains("length mismatch") || low.contains("dimension") {
        ErrorCode::ShapeMismatch
    } else if low.contains("domain")
        || low.contains("must be > 0")
        || low.contains("must be >= 0")
        || low.contains("must be >= 1")
        || low.contains("must be >= 2")
        || low.contains("must sum to 1")
        || low.contains("zero support")
        || low.contains("negative")
        || low.contains("out of range")
        || low.contains("exceeds exact f32")
        || low.contains("singular")
        || low.contains("zero quaternion")
        || low.contains("zero-length")
        || low.contains("requires 0 < near < far")
        || low.contains("exceeds maximum")
    {
        ErrorCode::MathDomainError
    } else if low.contains("workflow") || low.contains("unknown workflow pipeline stage") {
        ErrorCode::WorkflowFailed
    } else if matches!(stage, Some("validate")) {
        ErrorCode::ValidationFailed
    } else if matches!(
        stage,
        Some(
            "run"
                | "execute"
                | "trace_hash"
                | "rollout"
                | "profile"
                | "compare"
                | "soak"
                | "diff_ref"
                | "arith"
                | "int"
                | "kernel"
                | "geom.project"
                | "geom.batch"
                | "geom.solve_ik"
        )
    ) {
        ErrorCode::ExecutionFailed
    } else {
        ErrorCode::InternalError
    };

    let mut err = MathDrawError::new(code, msg);
    if let Some(st) = stage {
        err = err.with_stage(st);
    }
    err
}
