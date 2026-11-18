use burn::module::{Initializer, Param};
use burn::nn::activation::{
    HardSigmoid, HardSigmoidConfig, HardSwish, LeakyRelu, LeakyReluConfig, PRelu, PReluConfig,
    SwiGlu, SwiGluConfig,
};
use burn::nn::conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::pool::{
    AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, AvgPool1d, AvgPool1dConfig, AvgPool2d,
    AvgPool2dConfig, MaxPool1d, MaxPool1dConfig, MaxPool2d, MaxPool2dConfig,
};
use burn::nn::{
    BatchNorm, BatchNormConfig, Embedding, EmbeddingConfig, Gelu, GroupNorm, GroupNormConfig,
    InstanceNorm, InstanceNormConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    PaddingConfig1d, PaddingConfig2d, Relu, RmsNorm, RmsNormConfig, Sigmoid, Softplus,
    SoftplusConfig, Tanh,
};
use burn::tensor::{Device, Int, Tensor, TensorData};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

use crate::catalog::{expected_weight_count, is_int_constructor};
use crate::spec::{LayerDecl, SpecV1, parse_f32_values, parse_i64_values, sha256_f32_slice};
use crate::validate::validate_spec;

#[derive(Debug, Clone, Serialize)]
pub struct TensorOutput {
    pub shape: [usize; 4],
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunResult {
    pub ok: bool,
    pub mode: &'static str,
    pub outputs: TensorOutput,
    pub trace_sha256: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum GradTensor {
    D1(Tensor<1>),
    D2(Tensor<2>),
    D3(Tensor<3>),
    D4(Tensor<4>),
}

impl GradTensor {
    pub fn extract_grad(&self, grads: &burn::tensor::Gradients) -> Result<Vec<f32>, String> {
        let data = match self {
            GradTensor::D1(t) => t.grad(grads).map(|g| g.into_data()),
            GradTensor::D2(t) => t.grad(grads).map(|g| g.into_data()),
            GradTensor::D3(t) => t.grad(grads).map(|g| g.into_data()),
            GradTensor::D4(t) => t.grad(grads).map(|g| g.into_data()),
        };
        let Some(td) = data else {
            return Ok(Vec::new());
        };
        td.try_into_vec::<f32>()
            .map_err(|e| format!("autodiff gradient read failed: {e}"))
    }
}

#[derive(Debug, Clone)]
pub struct GradParam {
    pub layer_idx: usize,
    pub offset: usize,
    pub len: usize,
    pub tensor: GradTensor,
}

#[derive(Clone)]
pub enum SlotValue {
    F32(Tensor<4>),
    I64(Vec<i64>, [usize; 4]),
}

impl SlotValue {
    fn as_f32(&self, ctx: &str) -> Result<Tensor<4>, String> {
        match self {
            SlotValue::F32(t) => Ok(t.clone()),
            SlotValue::I64(..) => Err(format!("{ctx}: expected f32 slot, got i64")),
        }
    }

    fn as_i64(&self, ctx: &str) -> Result<(&[i64], [usize; 4]), String> {
        match self {
            SlotValue::I64(v, shape) => Ok((v.as_slice(), *shape)),
            SlotValue::F32(..) => Err(format!("{ctx}: expected i64 slot, got f32")),
        }
    }
}

pub fn get_layer_weight_len(decl: &LayerDecl) -> usize {
    expected_weight_count(&decl.constructor, &decl.args).unwrap_or(0)
}

pub fn layer_has_trainable_weights(decl: &LayerDecl) -> bool {
    get_layer_weight_len(decl) > 0 && decl.constructor != "embedding"
}

pub fn is_linear_like(decl: &LayerDecl) -> bool {
    get_layer_weight_len(decl) > 0
}

fn arg_usize(args: &[Value], idx: usize) -> Result<usize, String> {
    args.get(idx)
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .ok_or_else(|| format!("missing usize arg[{idx}]"))
}

fn arg_opt_usize(args: &[Value], idx: usize) -> Option<usize> {
    match args.get(idx) {
        Some(Value::Null) | None => None,
        Some(v) => v.as_u64().map(|n| n as usize),
    }
}

fn arg_bool(args: &[Value], idx: usize) -> Result<bool, String> {
    args.get(idx)
        .and_then(|v| v.as_bool())
        .ok_or_else(|| format!("missing bool arg[{idx}]"))
}

fn arg_f64(args: &[Value], idx: usize) -> Result<f64, String> {
    args.get(idx)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("missing f64 arg[{idx}]"))
}

fn arg_opt_f64(args: &[Value], idx: usize) -> Option<f64> {
    match args.get(idx) {
        Some(Value::Null) | None => None,
        Some(v) => v.as_f64(),
    }
}

fn resolved_weights(spec: &SpecV1, layer_idx: usize, expected: usize) -> Result<Vec<f32>, String> {
    match spec.weights.get(&layer_idx.to_string()) {
        Some(w) => {
            if w.len() != expected {
                return Err(format!(
                    "weights[{layer_idx}] ({}): expected {expected} floats, got {}",
                    spec.layers[layer_idx].constructor,
                    w.len()
                ));
            }
            for (i, &v) in w.iter().enumerate() {
                if !v.is_finite() {
                    return Err(format!("weights[{layer_idx}][{i}] is non-finite: {v}"));
                }
            }
            Ok(w.clone())
        }
        None => Ok(vec![0.0f32; expected]),
    }
}

fn resolved_weights_default_ones(
    spec: &SpecV1,
    layer_idx: usize,
    expected: usize,
    first_half_ones: usize,
) -> Result<Vec<f32>, String> {
    if spec.weights.contains_key(&layer_idx.to_string()) {
        return resolved_weights(spec, layer_idx, expected);
    }
    let mut v = vec![0.0f32; expected];
    for x in v.iter_mut().take(first_half_ones) {
        *x = 1.0;
    }
    Ok(v)
}

fn make_param_1d(
    slice: &[f32],
    dims: [usize; 1],
    device: &Device,
    need_grad: bool,
    layer_idx: usize,
    offset: usize,
    grad_params: &mut Vec<GradParam>,
) -> Param<Tensor<1>> {
    let t = Tensor::<1>::from_data(TensorData::new(slice.to_vec(), dims), device);
    let t = if need_grad {
        let g = t.require_grad();
        grad_params.push(GradParam {
            layer_idx,
            offset,
            len: slice.len(),
            tensor: GradTensor::D1(g.clone()),
        });
        g
    } else {
        t
    };
    Param::from_tensor(t)
}

fn make_param_2d(
    slice: &[f32],
    dims: [usize; 2],
    device: &Device,
    need_grad: bool,
    layer_idx: usize,
    offset: usize,
    grad_params: &mut Vec<GradParam>,
) -> Param<Tensor<2>> {
    let t = Tensor::<2>::from_data(TensorData::new(slice.to_vec(), dims), device);
    let t = if need_grad {
        let g = t.require_grad();
        grad_params.push(GradParam {
            layer_idx,
            offset,
            len: slice.len(),
            tensor: GradTensor::D2(g.clone()),
        });
        g
    } else {
        t
    };
    Param::from_tensor(t)
}

fn make_param_3d(
    slice: &[f32],
    dims: [usize; 3],
    device: &Device,
    need_grad: bool,
    layer_idx: usize,
    offset: usize,
    grad_params: &mut Vec<GradParam>,
) -> Param<Tensor<3>> {
    let t = Tensor::<3>::from_data(TensorData::new(slice.to_vec(), dims), device);
    let t = if need_grad {
        let g = t.require_grad();
        grad_params.push(GradParam {
            layer_idx,
            offset,
            len: slice.len(),
            tensor: GradTensor::D3(g.clone()),
        });
        g
    } else {
        t
    };
    Param::from_tensor(t)
}

fn make_param_4d(
    slice: &[f32],
    dims: [usize; 4],
    device: &Device,
    need_grad: bool,
    layer_idx: usize,
    offset: usize,
    grad_params: &mut Vec<GradParam>,
) -> Param<Tensor<4>> {
    let t = Tensor::<4>::from_data(TensorData::new(slice.to_vec(), dims), device);
    let t = if need_grad {
        let g = t.require_grad();
        grad_params.push(GradParam {
            layer_idx,
            offset,
            len: slice.len(),
            tensor: GradTensor::D4(g.clone()),
        });
        g
    } else {
        t
    };
    Param::from_tensor(t)
}

enum RuntimeLayer {
    Relu(Relu),
    Gelu(Gelu),
    Sigmoid(Sigmoid),
    Tanh(Tanh),
    HardSwish(HardSwish),
    LeakyRelu(LeakyRelu),
    PRelu(PRelu),
    SwiGlu(SwiGlu),
    HardSigmoid(HardSigmoid),
    Softplus(Softplus),
    Mish,
    Softmax(usize),
    LogSoftmax(usize),
    Glu(usize),
    Sqrt,
    Linear(Linear),
    BatchNorm(BatchNorm),
    GroupNorm(GroupNorm),
    InstanceNorm(InstanceNorm),
    LayerNorm(LayerNorm),
    RmsNorm(RmsNorm),
    Conv1d(Conv1d),
    Conv2d(Conv2d),
    ConvTranspose2d(ConvTranspose2d),
    Embedding { module: Embedding, vocab_size: usize },
    MaxPool1d(MaxPool1d),
    MaxPool2d(MaxPool2d),
    AvgPool1d(AvgPool1d),
    AvgPool2d(AvgPool2d),
    AdaptiveAvgPool2d(AdaptiveAvgPool2d),
    FeatureNorm(f64),
    Shift { dir: ShiftDir, size: usize },
    Ghost { primary: Conv2d, cheap: Conv2d },
    SeBlock { squeeze: AdaptiveAvgPool2d, fc1: Linear, fc2: Linear },
    Add,
    Sub,
    Mul,
    Matmul,
    Div,
    Concat(usize),
    Transpose(usize, usize),
    IntAdd,
    IntSub,
    IntMul,
    IntMatmul,
    IntDiv,
    IntRem,
    IntToFloat,
    IntFromFloat,
}

#[derive(Clone, Copy)]
enum ShiftDir {
    Up,
    Down,
    Left,
    Right,
}

fn build_linear(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    w: &[f32],
    device: &Device,
    need_grad: bool,
    layer_idx: usize,
    base_off: usize,
    grad_params: &mut Vec<GradParam>,
) -> Linear {
    let mut lin = LinearConfig::new(in_dim, out_dim)
        .with_bias(bias)
        .with_initializer(Initializer::Zeros)
        .init(device);
    let w_len = in_dim * out_dim;
    lin.weight = make_param_2d(
        &w[..w_len],
        [in_dim, out_dim],
        device,
        need_grad,
        layer_idx,
        base_off,
        grad_params,
    );
    if bias {
        lin.bias = Some(make_param_1d(
            &w[w_len..w_len + out_dim],
            [out_dim],
            device,
            need_grad,
            layer_idx,
            base_off + w_len,
            grad_params,
        ));
    }
    lin
}

fn build_layers(
    spec: &SpecV1,
    device: &Device,
    grad_layers: &[usize],
) -> Result<(Vec<RuntimeLayer>, Vec<GradParam>), String> {
    let mut out = Vec::with_capacity(spec.layers.len());
    let mut grad_params = Vec::new();

    for (li, decl) in spec.layers.iter().enumerate() {
        let need_grad = grad_layers.contains(&li);
        let args = &decl.args;
        let layer = match decl.constructor.as_str() {
            "relu" => RuntimeLayer::Relu(Relu::new()),
            "gelu" => RuntimeLayer::Gelu(Gelu::new()),
            "sigmoid" => RuntimeLayer::Sigmoid(Sigmoid::new()),
            "tanh" => RuntimeLayer::Tanh(Tanh::new()),
            "hardSwish" => RuntimeLayer::HardSwish(HardSwish::new()),
            "leakyRelu" => {
                let slope = arg_f64(args, 1)?;
                RuntimeLayer::LeakyRelu(LeakyReluConfig::new().with_negative_slope(slope).init())
            }
            "prelu" => {
                let np = arg_usize(args, 1)?;
                let alpha = arg_f64(args, 2)?;
                let mut m = PReluConfig::new()
                    .with_num_parameters(np)
                    .with_alpha(alpha)
                    .init(device);
                let w = if spec.weights.contains_key(&li.to_string()) {
                    resolved_weights(spec, li, np)?
                } else {
                    vec![alpha as f32; np]
                };
                m.alpha = make_param_1d(&w, [np], device, need_grad, li, 0, &mut grad_params);
                RuntimeLayer::PRelu(m)
            }
            "swiGlu" => {
                let d_in = arg_usize(args, 1)?;
                let d_out = arg_usize(args, 2)?;
                let bias = arg_bool(args, 3)?;
                let lin_sz = d_in * d_out + if bias { d_out } else { 0 };
                let w = resolved_weights(spec, li, 2 * lin_sz)?;
                let mut m = SwiGluConfig::new(d_in, d_out)
                    .with_bias(bias)
                    .with_initializer(Initializer::Zeros)
                    .init(device);
                m.linear_inner = build_linear(
                    d_in,
                    d_out,
                    bias,
                    &w[..lin_sz],
                    device,
                    need_grad,
                    li,
                    0,
                    &mut grad_params,
                );
                m.linear_outer = build_linear(
                    d_in,
                    d_out,
                    bias,
                    &w[lin_sz..],
                    device,
                    need_grad,
                    li,
                    lin_sz,
                    &mut grad_params,
                );
                RuntimeLayer::SwiGlu(m)
            }
            "hardSigmoid" => {
                let alpha = arg_f64(args, 1)?;
                let beta = arg_f64(args, 2)?;
                RuntimeLayer::HardSigmoid(
                    HardSigmoidConfig::new()
                        .with_alpha(alpha)
                        .with_beta(beta)
                        .init(),
                )
            }
            "softplus" => {
                let beta = arg_f64(args, 1)?;
                RuntimeLayer::Softplus(SoftplusConfig::new().with_beta(beta).init())
            }
            "mish" => RuntimeLayer::Mish,
            "softmax" => RuntimeLayer::Softmax(arg_usize(args, 1)?),
            "logSoftmax" => RuntimeLayer::LogSoftmax(arg_usize(args, 1)?),
            "glu" => RuntimeLayer::Glu(arg_usize(args, 1)?),
            "sqrt" => RuntimeLayer::Sqrt,
            "linear" => {
                let in_dim = arg_usize(args, 1)?;
                let out_dim = arg_usize(args, 2)?;
                let bias = arg_bool(args, 3)?;
                let exp = in_dim * out_dim + if bias { out_dim } else { 0 };
                let w = resolved_weights(spec, li, exp)?;
                RuntimeLayer::Linear(build_linear(
                    in_dim,
                    out_dim,
                    bias,
                    &w,
                    device,
                    need_grad,
                    li,
                    0,
                    &mut grad_params,
                ))
            }
            "batchNorm" => {
                let nf = arg_usize(args, 1)?;
                let eps = arg_opt_f64(args, 2).unwrap_or(1e-5);
                let mut m = BatchNormConfig::new(nf).with_epsilon(eps).init(device);
                let w = resolved_weights_default_ones(spec, li, 2 * nf, nf)?;
                m.gamma = make_param_1d(&w[..nf], [nf], device, need_grad, li, 0, &mut grad_params);
                m.beta = make_param_1d(&w[nf..], [nf], device, need_grad, li, nf, &mut grad_params);
                RuntimeLayer::BatchNorm(m)
            }
            "groupNorm" => {
                let ng = arg_usize(args, 1)?;
                let nc = arg_usize(args, 2)?;
                let eps = arg_opt_f64(args, 3).unwrap_or(1e-5);
                let mut m = GroupNormConfig::new(ng, nc).with_epsilon(eps).init(device);
                let w = resolved_weights_default_ones(spec, li, 2 * nc, nc)?;
                m.gamma = Some(make_param_1d(&w[..nc], [nc], device, need_grad, li, 0, &mut grad_params));
                m.beta = Some(make_param_1d(&w[nc..], [nc], device, need_grad, li, nc, &mut grad_params));
                RuntimeLayer::GroupNorm(m)
            }
            "instanceNorm" => {
                let nc = arg_usize(args, 1)?;
                let eps = arg_opt_f64(args, 2).unwrap_or(1e-5);
                let mut m = InstanceNormConfig::new(nc).with_epsilon(eps).init(device);
                let w = resolved_weights_default_ones(spec, li, 2 * nc, nc)?;
                m.gamma = Some(make_param_1d(&w[..nc], [nc], device, need_grad, li, 0, &mut grad_params));
                m.beta = Some(make_param_1d(&w[nc..], [nc], device, need_grad, li, nc, &mut grad_params));
                RuntimeLayer::InstanceNorm(m)
            }
            "layerNorm" => {
                let sz = arg_usize(args, 1)?;
                let eps = arg_opt_f64(args, 2).unwrap_or(1e-5);
                let mut m = LayerNormConfig::new(sz).with_epsilon(eps).init(device);
                let w = resolved_weights_default_ones(spec, li, 2 * sz, sz)?;
                m.gamma = make_param_1d(&w[..sz], [sz], device, need_grad, li, 0, &mut grad_params);
                m.beta = Some(make_param_1d(&w[sz..], [sz], device, need_grad, li, sz, &mut grad_params));
                RuntimeLayer::LayerNorm(m)
            }
            "rmsNorm" => {
                let sz = arg_usize(args, 1)?;
                let eps = arg_opt_f64(args, 2).unwrap_or(1e-6);
                let mut m = RmsNormConfig::new(sz).with_epsilon(eps).init(device);
                let w = resolved_weights_default_ones(spec, li, sz, sz)?;
                m.gamma = make_param_1d(&w, [sz], device, need_grad, li, 0, &mut grad_params);
                RuntimeLayer::RmsNorm(m)
            }
            "conv1d" => {
                let in_ch = arg_usize(args, 1)?;
                let out_ch = arg_usize(args, 2)?;
                let k = arg_usize(args, 3)?;
                let s = arg_opt_usize(args, 4).unwrap_or(1);
                let p = arg_opt_usize(args, 5).unwrap_or(0);
                let mut m = Conv1dConfig::new(in_ch, out_ch, k)
                    .with_stride(s)
                    .with_padding(PaddingConfig1d::Explicit(p, p))
                    .with_bias(true)
                    .with_initializer(Initializer::Zeros)
                    .init(device);
                let w_len = out_ch * in_ch * k;
                let w = resolved_weights(spec, li, w_len + out_ch)?;
                m.weight = make_param_3d(&w[..w_len], [out_ch, in_ch, k], device, need_grad, li, 0, &mut grad_params);
                m.bias = Some(make_param_1d(&w[w_len..], [out_ch], device, need_grad, li, w_len, &mut grad_params));
                RuntimeLayer::Conv1d(m)
            }
            "conv2d" => {
                let in_ch = arg_usize(args, 1)?;
                let out_ch = arg_usize(args, 2)?;
                let kh = arg_usize(args, 3)?;
                let kw = arg_usize(args, 4)?;
                let sh = arg_opt_usize(args, 5).unwrap_or(1);
                let sw = arg_opt_usize(args, 6).unwrap_or(1);
                let ph = arg_opt_usize(args, 7).unwrap_or(0);
                let pw = arg_opt_usize(args, 8).unwrap_or(0);
                let mut m = Conv2dConfig::new([in_ch, out_ch], [kh, kw])
                    .with_stride([sh, sw])
                    .with_padding(PaddingConfig2d::Explicit(ph, ph, pw, pw))
                    .with_bias(true)
                    .with_initializer(Initializer::Zeros)
                    .init(device);
                let w_len = out_ch * in_ch * kh * kw;
                let w = resolved_weights(spec, li, w_len + out_ch)?;
                m.weight = make_param_4d(&w[..w_len], [out_ch, in_ch, kh, kw], device, need_grad, li, 0, &mut grad_params);
                m.bias = Some(make_param_1d(&w[w_len..], [out_ch], device, need_grad, li, w_len, &mut grad_params));
                RuntimeLayer::Conv2d(m)
            }
            "convTranspose2d" => {
                let in_ch = arg_usize(args, 1)?;
                let out_ch = arg_usize(args, 2)?;
                let kh = arg_usize(args, 3)?;
                let kw = arg_usize(args, 4)?;
                let sh = arg_opt_usize(args, 5).unwrap_or(1);
                let sw = arg_opt_usize(args, 6).unwrap_or(1);
                let ph = arg_opt_usize(args, 7).unwrap_or(0);
                let pw = arg_opt_usize(args, 8).unwrap_or(0);
                let mut m = ConvTranspose2dConfig::new([in_ch, out_ch], [kh, kw])
                    .with_stride([sh, sw])
                    .with_padding([ph, pw])
                    .with_bias(true)
                    .with_initializer(Initializer::Zeros)
                    .init(device);
                let w_len = in_ch * out_ch * kh * kw;
                let w = resolved_weights(spec, li, w_len + out_ch)?;
                m.weight = make_param_4d(&w[..w_len], [in_ch, out_ch, kh, kw], device, need_grad, li, 0, &mut grad_params);
                m.bias = Some(make_param_1d(&w[w_len..], [out_ch], device, need_grad, li, w_len, &mut grad_params));
                RuntimeLayer::ConvTranspose2d(m)
            }
            "embedding" => {
                let vocab = arg_usize(args, 1)?;
                let d_model = arg_usize(args, 2)?;
                let mut m = EmbeddingConfig::new(vocab, d_model)
                    .with_initializer(Initializer::Zeros)
                    .init(device);
                let w = resolved_weights(spec, li, vocab * d_model)?;
                m.weight = make_param_2d(&w, [vocab, d_model], device, need_grad, li, 0, &mut grad_params);
                RuntimeLayer::Embedding { module: m, vocab_size: vocab }
            }
            "maxPool1d" => {
                let k = arg_usize(args, 1)?;
                let s = arg_opt_usize(args, 2).unwrap_or(k);
                let p = arg_opt_usize(args, 3).unwrap_or(0);
                RuntimeLayer::MaxPool1d(
                    MaxPool1dConfig::new(k)
                        .with_stride(s)
                        .with_padding(PaddingConfig1d::Explicit(p, p))
                        .init(),
                )
            }
            "avgPool1d" => {
                let k = arg_usize(args, 1)?;
                let s = arg_opt_usize(args, 2).unwrap_or(k);
                let p = arg_opt_usize(args, 3).unwrap_or(0);
                RuntimeLayer::AvgPool1d(
                    AvgPool1dConfig::new(k)
                        .with_stride(s)
                        .with_padding(PaddingConfig1d::Explicit(p, p))
                        .init(),
                )
            }
            "maxPool2d" => {
                let kh = arg_usize(args, 1)?;
                let kw = arg_usize(args, 2)?;
                let sh = arg_opt_usize(args, 3).unwrap_or(kh);
                let sw = arg_opt_usize(args, 4).unwrap_or(kw);
                let ph = arg_opt_usize(args, 5).unwrap_or(0);
                let pw = arg_opt_usize(args, 6).unwrap_or(0);
                RuntimeLayer::MaxPool2d(
                    MaxPool2dConfig::new([kh, kw])
                        .with_strides([sh, sw])
                        .with_padding(PaddingConfig2d::Explicit(ph, ph, pw, pw))
                        .init(),
                )
            }
            "avgPool2d" => {
                let kh = arg_usize(args, 1)?;
                let kw = arg_usize(args, 2)?;
                let sh = arg_opt_usize(args, 3).unwrap_or(kh);
                let sw = arg_opt_usize(args, 4).unwrap_or(kw);
                let ph = arg_opt_usize(args, 5).unwrap_or(0);
                let pw = arg_opt_usize(args, 6).unwrap_or(0);
                RuntimeLayer::AvgPool2d(
                    AvgPool2dConfig::new([kh, kw])
                        .with_strides([sh, sw])
                        .with_padding(PaddingConfig2d::Explicit(ph, ph, pw, pw))
                        .init(),
                )
            }
            "adaptiveAvgPool2d" => {
                let oh = arg_usize(args, 1)?;
                let ow = arg_usize(args, 2)?;
                RuntimeLayer::AdaptiveAvgPool2d(AdaptiveAvgPool2dConfig::new([oh, ow]).init())
            }
            "featureNorm" => {
                let eps = arg_opt_f64(args, 1).unwrap_or(1e-12);
                RuntimeLayer::FeatureNorm(eps)
            }
            "shiftUp" => RuntimeLayer::Shift { dir: ShiftDir::Up, size: arg_usize(args, 1)? },
            "shiftDown" => RuntimeLayer::Shift { dir: ShiftDir::Down, size: arg_usize(args, 1)? },
            "shiftLeft" => RuntimeLayer::Shift { dir: ShiftDir::Left, size: arg_usize(args, 1)? },
            "shiftRight" => RuntimeLayer::Shift { dir: ShiftDir::Right, size: arg_usize(args, 1)? },
            "ghost" => {
                let in_ch = arg_usize(args, 1)?;
                let out_ch = arg_usize(args, 2)?;
                let kh = arg_usize(args, 3)?;
                let kw = arg_usize(args, 4)?;
                let ratio = arg_usize(args, 5)?;
                let sh = arg_opt_usize(args, 6).unwrap_or(1);
                let sw = arg_opt_usize(args, 7).unwrap_or(1);
                let ph = arg_opt_usize(args, 8).unwrap_or(0);
                let pw = arg_opt_usize(args, 9).unwrap_or(0);
                let pch = out_ch / ratio;
                let pw_len = pch * in_ch * kh * kw;
                let total = pw_len + pch + pch;
                let w = resolved_weights(spec, li, total)?;
                let mut primary = Conv2dConfig::new([in_ch, pch], [kh, kw])
                    .with_stride([sh, sw])
                    .with_padding(PaddingConfig2d::Explicit(ph, ph, pw, pw))
                    .with_bias(true)
                    .with_initializer(Initializer::Zeros)
                    .init(device);
                primary.weight = make_param_4d(&w[..pw_len], [pch, in_ch, kh, kw], device, need_grad, li, 0, &mut grad_params);
                primary.bias = Some(make_param_1d(&w[pw_len..pw_len + pch], [pch], device, need_grad, li, pw_len, &mut grad_params));
                let mut cheap = Conv2dConfig::new([pch, pch], [1, 1])
                    .with_groups(pch)
                    .with_bias(false)
                    .with_initializer(Initializer::Zeros)
                    .init(device);
                cheap.weight = make_param_4d(&w[pw_len + pch..], [pch, 1, 1, 1], device, need_grad, li, pw_len + pch, &mut grad_params);
                RuntimeLayer::Ghost { primary, cheap }
            }
            "seBlock" => {
                let ch = arg_usize(args, 1)?;
                let red = arg_usize(args, 2)?;
                let r = ch / red;
                let l1 = ch * r + r;
                let l2 = r * ch + ch;
                let w = resolved_weights(spec, li, l1 + l2)?;
                let squeeze = AdaptiveAvgPool2dConfig::new([1, 1]).init();
                let fc1 = build_linear(ch, r, true, &w[..l1], device, need_grad, li, 0, &mut grad_params);
                let fc2 = build_linear(r, ch, true, &w[l1..], device, need_grad, li, l1, &mut grad_params);
                RuntimeLayer::SeBlock { squeeze, fc1, fc2 }
            }
            "add" => RuntimeLayer::Add,
            "sub" => RuntimeLayer::Sub,
            "mul" => RuntimeLayer::Mul,
            "matmul" => RuntimeLayer::Matmul,
            "div" => RuntimeLayer::Div,
            "concat" => RuntimeLayer::Concat(arg_usize(args, 1)?),
            "transpose" => RuntimeLayer::Transpose(arg_usize(args, 1)?, arg_usize(args, 2)?),
            "intAdd" => RuntimeLayer::IntAdd,
            "intSub" => RuntimeLayer::IntSub,
            "intMul" => RuntimeLayer::IntMul,
            "intMatmul" => RuntimeLayer::IntMatmul,
            "intDiv" => RuntimeLayer::IntDiv,
            "intRem" => RuntimeLayer::IntRem,
            "intToFloat" => RuntimeLayer::IntToFloat,
            "intFromFloat" => RuntimeLayer::IntFromFloat,
            other => return Err(format!("unsupported constructor '{other}'")),
        };
        out.push(layer);
    }
    Ok((out, grad_params))
}

fn checked_div_f32(a: Tensor<4>, b: Tensor<4>) -> Result<Tensor<4>, String> {
    let ad = a.clone().into_data();
    let av = ad.as_slice::<f32>().map_err(|_| "div lhs not f32".to_string())?;
    for (i, &v) in av.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("binary div lhs: non-finite value at index {i}: {v}"));
        }
    }
    let bd = b.clone().into_data();
    let bv = bd.as_slice::<f32>().map_err(|_| "div rhs not f32".to_string())?;
    for (i, &v) in bv.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("binary div rhs: non-finite value at index {i}: {v}"));
        }
        if v == 0.0 {
            return Err(format!("binary div rhs: zero denominator at index {i}"));
        }
    }
    let out = a.div(b);
    let od = out.clone().into_data();
    let ov = od.as_slice::<f32>().map_err(|_| "div output not f32".to_string())?;
    for (i, &v) in ov.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("binary div output: non-finite value at index {i}: {v}"));
        }
    }
    Ok(out)
}

fn run_shift(input: Tensor<4>, dir: ShiftDir, size: usize) -> Tensor<4> {
    let [b, c, h, w] = input.dims();
    let dev = input.device();
    match dir {
        ShiftDir::Left => {
            if size >= w {
                Tensor::zeros([b, c, h, w], &dev)
            } else {
                let main = input.slice([0..b, 0..c, 0..h, size..w]);
                let pad = Tensor::zeros([b, c, h, size], &dev);
                Tensor::cat(vec![main, pad], 3)
            }
        }
        ShiftDir::Right => {
            if size >= w {
                Tensor::zeros([b, c, h, w], &dev)
            } else {
                let pad = Tensor::zeros([b, c, h, size], &dev);
                let main = input.slice([0..b, 0..c, 0..h, 0..(w - size)]);
                Tensor::cat(vec![pad, main], 3)
            }
        }
        ShiftDir::Up => {
            if size >= h {
                Tensor::zeros([b, c, h, w], &dev)
            } else {
                let main = input.slice([0..b, 0..c, size..h, 0..w]);
                let pad = Tensor::zeros([b, c, size, w], &dev);
                Tensor::cat(vec![main, pad], 2)
            }
        }
        ShiftDir::Down => {
            if size >= h {
                Tensor::zeros([b, c, h, w], &dev)
            } else {
                let pad = Tensor::zeros([b, c, size, w], &dev);
                let main = input.slice([0..b, 0..c, 0..(h - size), 0..w]);
                Tensor::cat(vec![pad, main], 2)
            }
        }
    }
}

fn exec_layer(
    layer: &RuntimeLayer,
    inputs: &[SlotValue],
    device: &Device,
) -> Result<SlotValue, String> {
    match layer {
        RuntimeLayer::Relu(m) => Ok(SlotValue::F32(m.forward(inputs[0].as_f32("relu")?))),
        RuntimeLayer::Gelu(m) => Ok(SlotValue::F32(m.forward(inputs[0].as_f32("gelu")?))),
        RuntimeLayer::Sigmoid(m) => Ok(SlotValue::F32(m.forward(inputs[0].as_f32("sigmoid")?))),
        RuntimeLayer::Tanh(m) => Ok(SlotValue::F32(m.forward(inputs[0].as_f32("tanh")?))),
        RuntimeLayer::HardSwish(m) => Ok(SlotValue::F32(m.forward(inputs[0].as_f32("hardSwish")?))),
        RuntimeLayer::LeakyRelu(m) => Ok(SlotValue::F32(m.forward(inputs[0].as_f32("leakyRelu")?))),
        RuntimeLayer::PRelu(m) => Ok(SlotValue::F32(m.forward(inputs[0].as_f32("prelu")?))),
        RuntimeLayer::SwiGlu(m) => Ok(SlotValue::F32(m.forward(inputs[0].as_f32("swiGlu")?))),
        RuntimeLayer::HardSigmoid(m) => Ok(SlotValue::F32(m.forward(inputs[0].as_f32("hardSigmoid")?))),
        RuntimeLayer::Softplus(m) => Ok(SlotValue::F32(m.forward(inputs[0].as_f32("softplus")?))),
        RuntimeLayer::Mish => Ok(SlotValue::F32(burn::tensor::activation::mish(
            inputs[0].as_f32("mish")?,
        ))),
        RuntimeLayer::Softmax(dim) => Ok(SlotValue::F32(burn::tensor::activation::softmax(
            inputs[0].as_f32("softmax")?,
            *dim,
        ))),
        RuntimeLayer::LogSoftmax(dim) => Ok(SlotValue::F32(burn::tensor::activation::log_softmax(
            inputs[0].as_f32("logSoftmax")?,
            *dim,
        ))),
        RuntimeLayer::Glu(dim) => Ok(SlotValue::F32(burn::tensor::activation::glu(
            inputs[0].as_f32("glu")?,
            *dim,
        ))),
        RuntimeLayer::Sqrt => Ok(SlotValue::F32(inputs[0].as_f32("sqrt")?.sqrt())),
        RuntimeLayer::Linear(m) => {
            let x = inputs[0].as_f32("linear")?;
            let [b, f, _, _] = x.dims();
            let y2 = m.forward(x.reshape([b, f]));
            let [b2, o] = y2.dims();
            Ok(SlotValue::F32(y2.reshape([b2, o, 1, 1])))
        }
        RuntimeLayer::BatchNorm(m) => Ok(SlotValue::F32(m.forward(inputs[0].as_f32("batchNorm")?))),
        RuntimeLayer::GroupNorm(m) => Ok(SlotValue::F32(m.forward(inputs[0].as_f32("groupNorm")?))),
        RuntimeLayer::InstanceNorm(m) => {
            Ok(SlotValue::F32(m.forward(inputs[0].as_f32("instanceNorm")?)))
        }
        RuntimeLayer::LayerNorm(m) => Ok(SlotValue::F32(m.forward(inputs[0].as_f32("layerNorm")?))),
        RuntimeLayer::RmsNorm(m) => Ok(SlotValue::F32(m.forward(inputs[0].as_f32("rmsNorm")?))),
        RuntimeLayer::Conv1d(m) => {
            let x = inputs[0].as_f32("conv1d")?;
            let [b, c, l, _] = x.dims();
            let y = m.forward(x.reshape([b, c, l]));
            let [b2, c2, l2] = y.dims();
            Ok(SlotValue::F32(y.reshape([b2, c2, l2, 1])))
        }
        RuntimeLayer::Conv2d(m) => Ok(SlotValue::F32(m.forward(inputs[0].as_f32("conv2d")?))),
        RuntimeLayer::ConvTranspose2d(m) => {
            Ok(SlotValue::F32(m.forward(inputs[0].as_f32("convTranspose2d")?)))
        }
        RuntimeLayer::Embedding { module, vocab_size } => {
            let x = inputs[0].as_f32("embedding")?;
            let [b, s, _, _] = x.dims();
            let td = x.into_data();
            let vals = td
                .as_slice::<f32>()
                .map_err(|_| "embedding: expected f32 token IDs".to_string())?;
            let mut ids = Vec::with_capacity(vals.len());
            for (idx, &v) in vals.iter().enumerate() {
                if !v.is_finite() || v < 0.0 || v.fract() != 0.0 || (v as usize) >= *vocab_size {
                    return Err(format!(
                        "embedding: token ID[{idx}]={v} must be integer in 0..{vocab_size}"
                    ));
                }
                ids.push(v as i64);
            }
            let tok = Tensor::<2, Int>::from_data(TensorData::new(ids, [b, s]), device);
            let out3 = module.forward(tok);
            let [b2, s2, d] = out3.dims();
            Ok(SlotValue::F32(out3.reshape([b2, s2, d, 1])))
        }
        RuntimeLayer::MaxPool1d(m) => {
            let x = inputs[0].as_f32("maxPool1d")?;
            let [b, c, l, _] = x.dims();
            let y = m.forward(x.reshape([b, c, l]));
            let [b2, c2, l2] = y.dims();
            Ok(SlotValue::F32(y.reshape([b2, c2, l2, 1])))
        }
        RuntimeLayer::AvgPool1d(m) => {
            let x = inputs[0].as_f32("avgPool1d")?;
            let [b, c, l, _] = x.dims();
            let y = m.forward(x.reshape([b, c, l]));
            let [b2, c2, l2] = y.dims();
            Ok(SlotValue::F32(y.reshape([b2, c2, l2, 1])))
        }
        RuntimeLayer::MaxPool2d(m) => Ok(SlotValue::F32(m.forward(inputs[0].as_f32("maxPool2d")?))),
        RuntimeLayer::AvgPool2d(m) => Ok(SlotValue::F32(m.forward(inputs[0].as_f32("avgPool2d")?))),
        RuntimeLayer::AdaptiveAvgPool2d(m) => {
            Ok(SlotValue::F32(m.forward(inputs[0].as_f32("adaptiveAvgPool2d")?)))
        }
        RuntimeLayer::FeatureNorm(eps) => {
            let x = inputs[0].as_f32("featureNorm")?;
            let norm = x.clone().square().sum_dim(1).sqrt().clamp_min(*eps as f32);
            Ok(SlotValue::F32(x / norm))
        }
        RuntimeLayer::Shift { dir, size } => {
            Ok(SlotValue::F32(run_shift(inputs[0].as_f32("shift")?, *dir, *size)))
        }
        RuntimeLayer::Ghost { primary, cheap } => {
            let x = inputs[0].as_f32("ghost")?;
            let intrinsic = primary.forward(x);
            let ghost = cheap.forward(intrinsic.clone());
            Ok(SlotValue::F32(Tensor::cat(vec![intrinsic, ghost], 1)))
        }
        RuntimeLayer::SeBlock { squeeze, fc1, fc2 } => {
            let x = inputs[0].as_f32("seBlock")?;
            let [b, c, _, _] = x.dims();
            let pooled = squeeze.forward(x.clone()).reshape([b, c]);
            let s = burn::tensor::activation::relu(fc1.forward(pooled));
            let s = burn::tensor::activation::sigmoid(fc2.forward(s)).reshape([b, c, 1, 1]);
            Ok(SlotValue::F32(x * s))
        }
        RuntimeLayer::Add => Ok(SlotValue::F32(
            inputs[0].as_f32("add")?.add(inputs[1].as_f32("add")?),
        )),
        RuntimeLayer::Sub => Ok(SlotValue::F32(
            inputs[0].as_f32("sub")?.sub(inputs[1].as_f32("sub")?),
        )),
        RuntimeLayer::Mul => Ok(SlotValue::F32(
            inputs[0].as_f32("mul")?.mul(inputs[1].as_f32("mul")?),
        )),
        RuntimeLayer::Matmul => Ok(SlotValue::F32(
            inputs[0].as_f32("matmul")?.matmul(inputs[1].as_f32("matmul")?),
        )),
        RuntimeLayer::Div => Ok(SlotValue::F32(checked_div_f32(
            inputs[0].as_f32("div")?,
            inputs[1].as_f32("div")?,
        )?)),
        RuntimeLayer::Concat(dim) => Ok(SlotValue::F32(Tensor::cat(
            vec![inputs[0].as_f32("concat")?, inputs[1].as_f32("concat")?],
            *dim,
        ))),
        RuntimeLayer::Transpose(d0, d1) => Ok(SlotValue::F32(
            inputs[0].as_f32("transpose")?.swap_dims(*d0, *d1),
        )),
        RuntimeLayer::IntAdd | RuntimeLayer::IntSub | RuntimeLayer::IntMul | RuntimeLayer::IntDiv | RuntimeLayer::IntRem => {
            let (va, sa) = inputs[0].as_i64("int binary lhs")?;
            let (vb, sb) = inputs[1].as_i64("int binary rhs")?;
            if sa != sb {
                return Err(format!("int binary shape mismatch {sa:?} vs {sb:?}"));
            }
            let mut out = Vec::with_capacity(va.len());
            for (i, (&x, &y)) in va.iter().zip(vb.iter()).enumerate() {
                let res = match layer {
                    RuntimeLayer::IntAdd => x.checked_add(y).ok_or_else(|| format!("intAdd i64 overflow at {i}")),
                    RuntimeLayer::IntSub => x.checked_sub(y).ok_or_else(|| format!("intSub i64 overflow at {i}")),
                    RuntimeLayer::IntMul => x.checked_mul(y).ok_or_else(|| format!("intMul i64 overflow at {i}")),
                    RuntimeLayer::IntDiv => {
                        if y == 0 {
                            Err(format!("intDiv division by zero at {i}"))
                        } else if x == i64::MIN && y == -1 {
                            Err(format!("intDiv i64 overflow at {i}"))
                        } else {
                            Ok(x / y)
                        }
                    }
                    RuntimeLayer::IntRem => {
                        if y == 0 {
                            Err(format!("intRem division by zero at {i}"))
                        } else if x == i64::MIN && y == -1 {
                            Err(format!("intRem i64 overflow at {i}"))
                        } else {
                            Ok(x % y)
                        }
                    }
                    _ => unreachable!(),
                }?;
                out.push(res);
            }
            Ok(SlotValue::I64(out, sa))
        }
        RuntimeLayer::IntMatmul => {
            let (va, da) = inputs[0].as_i64("intMatmul lhs")?;
            let (vb, db) = inputs[1].as_i64("intMatmul rhs")?;
            if da[0] != db[0] || da[1] != db[1] || da[3] != db[2] {
                return Err(format!("intMatmul shape mismatch {da:?} @ {db:?}"));
            }
            let (b0n, b1n, m, k, n) = (da[0], da[1], da[2], da[3], db[3]);
            let mut out = Vec::with_capacity(b0n * b1n * m * n);
            for b0 in 0..b0n {
                for b1 in 0..b1n {
                    let batch = b0 * b1n + b1;
                    for mi in 0..m {
                        for ni in 0..n {
                            let mut acc: i64 = 0;
                            for ki in 0..k {
                                let x = va[(batch * m + mi) * k + ki];
                                let y = vb[(batch * k + ki) * n + ni];
                                let prod = x.checked_mul(y).ok_or_else(|| {
                                    format!("intMatmul product i64 overflow at [{b0},{b1},{mi},{ni}]")
                                })?;
                                acc = acc.checked_add(prod).ok_or_else(|| {
                                    format!("intMatmul accumulator i64 overflow at [{b0},{b1},{mi},{ni}]")
                                })?;
                            }
                            out.push(acc);
                        }
                    }
                }
            }
            Ok(SlotValue::I64(out, [b0n, b1n, m, n]))
        }
        RuntimeLayer::IntToFloat => {
            let (vi, shape) = inputs[0].as_i64("intToFloat")?;
            let mut vf = Vec::with_capacity(vi.len());
            for (i, &v) in vi.iter().enumerate() {
                if v.unsigned_abs() > 16_777_216 {
                    return Err(format!(
                        "intToFloat: value {v} at index {i} exceeds exact f32 integer range (|x| <= 2^24)"
                    ));
                }
                vf.push(v as f32);
            }
            Ok(SlotValue::F32(Tensor::from_data(TensorData::new(vf, shape), device)))
        }
        RuntimeLayer::IntFromFloat => {
            let t = inputs[0].as_f32("intFromFloat")?;
            let shape = t.dims();
            let td = t.into_data();
            let vf = td.as_slice::<f32>().map_err(|_| "intFromFloat not f32".to_string())?;
            let mut vi = Vec::with_capacity(vf.len());
            for (i, &v) in vf.iter().enumerate() {
                if !v.is_finite() || v < (i64::MIN as f32) || v >= (i64::MAX as f32) {
                    return Err(format!("intFromFloat: out of i64 range at index {i}: {v}"));
                }
                vi.push(v as i64);
            }
            Ok(SlotValue::I64(vi, shape))
        }
    }
}

fn load_port_inputs(
    spec: &SpecV1,
    device: &Device,
    overrides: Option<&BTreeMap<String, Vec<Value>>>,
) -> Result<Vec<Option<SlotValue>>, String> {
    let mut slots: Vec<Option<SlotValue>> = vec![None; spec.num_slots];
    let mut i64_ports = vec![false; spec.num_slots];
    for st in &spec.steps {
        if let Some(layer) = spec.layers.get(st.layer)
            && is_int_constructor(&layer.constructor)
            && layer.constructor != "intFromFloat"
        {
            for &in_s in st.slots.iter().take(st.slots.len().saturating_sub(1)) {
                if in_s < spec.num_slots && spec.port_by_slot(in_s).is_some() {
                    i64_ports[in_s] = true;
                }
            }
        }
    }

    for port in &spec.ports {
        let key = port.slot.to_string();
        let prod: usize = port.shape.iter().product();
        let raw_opt = overrides
            .and_then(|m| m.get(&key))
            .or_else(|| spec.inputs.get(&key));
        let Some(raw) = raw_opt else {
            return Err(format!("missing inputs for port slot {}", port.slot));
        };
        if raw.len() != prod {
            return Err(format!(
                "inputs[{}] length mismatch: expected {prod} for shape {:?}, got {}",
                port.slot,
                port.shape,
                raw.len()
            ));
        }
        if i64_ports[port.slot] {
            let vals = parse_i64_values(raw, &format!("inputs[{}]", port.slot))?;
            slots[port.slot] = Some(SlotValue::I64(vals, port.shape));
        } else {
            let vals = parse_f32_values(raw, &format!("inputs[{}]", port.slot))?;
            let tensor = Tensor::<4>::from_data(TensorData::new(vals, port.shape), device);
            slots[port.slot] = Some(SlotValue::F32(tensor));
        }
    }
    Ok(slots)
}

pub fn execute_forward(
    spec: &SpecV1,
    device: &Device,
    grad_layers: &[usize],
    input_overrides: Option<&BTreeMap<String, Vec<Value>>>,
) -> Result<(SlotValue, Vec<GradParam>), String> {
    let (layers, grad_params) = build_layers(spec, device, grad_layers)?;
    let mut slots = load_port_inputs(spec, device, input_overrides)?;

    for (si, st) in spec.steps.iter().enumerate() {
        let layer = layers
            .get(st.layer)
            .ok_or_else(|| format!("step {si}: missing layer {}", st.layer))?;
        let arity = if st.kind == "binary" { 2 } else { 1 };
        let mut ins = Vec::with_capacity(arity);
        for &s in st.slots.iter().take(arity) {
            let val = slots
                .get(s)
                .and_then(|x| x.as_ref())
                .cloned()
                .ok_or_else(|| format!("step {si}: slot {s} read before written"))?;
            ins.push(val);
        }
        let out_slot = st.slots[arity];
        let res = exec_layer(layer, &ins, device)?;
        slots[out_slot] = Some(res);
    }

    let final_val = slots[spec.output_slot]
        .take()
        .ok_or_else(|| format!("outputSlot {} was not produced", spec.output_slot))?;
    Ok((final_val, grad_params))
}

pub fn eval_spec_with_inputs(
    spec: &SpecV1,
    input_overrides: Option<&BTreeMap<String, Vec<Value>>>,
) -> Result<TensorOutput, String> {
    let device = Device::flex();
    let (val, _) = execute_forward(spec, &device, &[], input_overrides)?;
    match val {
        SlotValue::F32(t) => {
            let shape = t.dims();
            let td = t.into_data();
            let values = td
                .try_into_vec::<f32>()
                .map_err(|e| format!("failed reading output f32 tensor: {e}"))?;
            for (i, &v) in values.iter().enumerate() {
                if !v.is_finite() {
                    return Err(format!("non-finite output at index {i}: {v}"));
                }
            }
            Ok(TensorOutput { shape, values })
        }
        SlotValue::I64(vi, shape) => {
            let values: Vec<f32> = vi.into_iter().map(|v| v as f32).collect();
            Ok(TensorOutput { shape, values })
        }
    }
}

pub fn run_spec(spec: &SpecV1) -> Result<RunResult, String> {
    let report = validate_spec(spec, false);
    if !report.valid {
        return Err(serde_json::to_string(&report).unwrap_or_else(|_| "validation failed".into()));
    }
    let outputs = eval_spec_with_inputs(spec, None)?;
    let trace_sha256 = sha256_f32_slice(&outputs.values);
    Ok(RunResult {
        ok: true,
        mode: "native_flex",
        outputs,
        trace_sha256,
        warnings: report.static_warnings,
    })
}
