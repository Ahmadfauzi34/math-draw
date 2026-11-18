# src/layers/ — Burn Tensor Kernels & 95-Constructor Dispatch

## Purpose

Implements construction (`build_layers`, `build_linear`) and deterministic forward execution (`exec_layer`) for all 95 supported constructors using Burn `0.22.1` primitives on `Device::flex()`, along with autograd parameter tracking (`GradParam`, `GradTensor`).

## Ownership

- Layer dispatch, `RuntimeLayer`, slot value representation (`SlotValue::F32` / `SlotValue::I64`), weight-length & trainable helpers: [`mod.rs`](mod.rs)
- Activations, elementwise nonlinearities, `softmax`, `logSoftmax`, `glu`, `sqrt`: [`activation.rs`](activation.rs)
- Linear, normalization (`batchNorm`, `groupNorm`, `instanceNorm`, `layerNorm`, `rmsNorm`), and `embedding`: [`norm_linear.rs`](norm_linear.rs)
- Convolutions (`conv1d`, `conv2d`, `convTranspose2d`) and pooling (`maxPool1d`, `avgPool1d`, `maxPool2d`, `avgPool2d`, `adaptiveAvgPool2d`): [`conv_pool.rs`](conv_pool.rs)
- Custom spatial/channel layers (`featureNorm`, `shiftUp`/`Down`/`Left`/`Right`, `ghost`, `seBlock`): [`custom.rs`](custom.rs)
- Binary float ops (`add`, `sub`, `mul`, `matmul`, `div`, `concat`), `transpose`, and exact checked `i64` ops (`intAdd`, `intSub`, `intMul`, `intMatmul`, `intDiv`, `intRem`, `intToFloat`, `intFromFloat`): [`binary_int.rs`](binary_int.rs)
- Numeric primitives (`abs`, `exp`, `log`, `clamp`, `sin`, `cos`, `tan`, `floor`, `ceil`, `round`, `atan2`, `powf`, `hypot`, ternary `fma`, `allFinite`), tensor transforms (`reshape`, `permute`, `slice`, `selectAxis`, `fillLike`, `expandLike`), axis reductions (`sumAxis`, `meanAxis`, `minAxis`, `maxAxis`), `lessEqual01`, and `indicesLike`: [`math_numeric.rs`](math_numeric.rs)
- Feature-vector linalg (`dot`, `l2Norm`, `l2Distance`, `cosineSimilarity`), statistics (`sum`, `mean`, `variancePopulation`, `stdPopulation`, `varianceSample`, `stdSample`, `min`, `max`), and probability ops (`normalize`, `entropy`, `crossEntropy`, `klDivergence`): [`math_stats_prob.rs`](math_stats_prob.rs)

## Local Contracts

- **4D Tensor & Batch Isolation**: All float tensors execute as rank-4 Burn `Tensor<4>`. Channel-wise custom ops (`featureNorm`) and normalizations must preserve strict batch isolation across `dim=0`.
- **Fail-Closed Arithmetic & Bounds**: `div`, `intDiv`, and `intRem` must reject zero divisors explicitly (`Err(...)`) rather than producing `Inf`/`NaN` or panicking; integer ops must use checked arithmetic (`checked_add`, `checked_sub`, `checked_mul`, `checked_div`, `checked_rem`) to fail closed on overflow. `log`, `exp`, `tan`, `powf`, scaled overflow-safe `hypot`, `f32::mul_add` ternary `fma`, `allFinite` (`[1, 1, 1, 1]` canonical `1.0`/`0.0`), population & Bessel-corrected (`features >= 2`, `ddof = 1`) sample `variance`/`std` (two-pass `f64`), `normalize`/`entropy`/`crossEntropy`/`klDivergence` (unit-mass & zero-support checks), `lessEqual01` (exact canonical `0.0`/`1.0`), and `indicesLike` (`axis_len <= 2^24 + 1`) must fail closed on invalid domains or non-finite outputs.
- **Weight Layout Exactness**: Resolved weights in `resolved_weights` must match `expected_weight_count(constructor, args)` when explicit weights are supplied, and `GradParam` offsets must accurately mirror flat slice layouts expected by [`src/es/`](../es/AGENTS.md) and [`src/evidence/gradcheck.rs`](../evidence/gradcheck.rs).
- **Explicit Weight Precedence**: User-provided weights must never be silently overwritten by initializers.

## Work Guidance

- Keep constructors grouped by family in the seven kernel submodules; `mod.rs` owns `RuntimeLayer`, shared param builders, and high-level `build_layers` / `exec_layer` routing.
- Avoid panicking `.unwrap()` on tensor shapes or JSON args inside kernels; surface actionable `Err(String)` messages.

## Verification

- `cargo test --locked --test integration all_95_constructors_execute_cleanly`
- `cargo test --locked --test integration div_by_zero_and_int_overflow_fail_closed`
- `cargo test --locked --test integration ported_math_ops_domain_and_shape_fail_closed`
- `cargo test --locked --test integration parity_13_math_ops_fail_closed_and_ternary_pipelines`

## Child Graph Index

- Leaf domain directory; core submodules listed under Ownership (`mod.rs`, `activation.rs`, `norm_linear.rs`, `conv_pool.rs`, `custom.rs`, `binary_int.rs`, `math_numeric.rs`, `math_stats_prob.rs`).
