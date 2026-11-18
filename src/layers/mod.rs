pub mod activation;
pub mod binary_int;
pub mod conv_pool;
pub mod custom;
pub mod math_numeric;
pub mod math_stats_prob;
pub mod norm_linear;

use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::nn::activation::{HardSigmoid, HardSwish, LeakyRelu, PRelu, SwiGlu};
use burn::nn::conv::{Conv1d, Conv2d, ConvTranspose2d};
use burn::nn::pool::{AdaptiveAvgPool2d, AvgPool1d, AvgPool2d, MaxPool1d, MaxPool2d};
use burn::nn::{
    BatchNorm, Embedding, Gelu, GroupNorm, InstanceNorm, LayerNorm, Relu, RmsNorm, Sigmoid,
    Softplus, Tanh,
};
use burn::module::Initializer;
use burn::tensor::{Device, Tensor, TensorData};
use serde_json::Value;

use crate::catalog::expected_weight_count;
use crate::spec::{LayerDecl, SpecV1};

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
    pub fn as_f32(&self, ctx: &str) -> Result<Tensor<4>, String> {
        match self {
            SlotValue::F32(t) => Ok(t.clone()),
            SlotValue::I64(..) => Err(format!("{ctx}: expected f32 slot, got i64")),
        }
    }

    pub fn as_i64(&self, ctx: &str) -> Result<(&[i64], [usize; 4]), String> {
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

pub(crate) fn arg_usize(args: &[Value], idx: usize) -> Result<usize, String> {
    args.get(idx)
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .ok_or_else(|| format!("missing usize arg[{idx}]"))
}

pub(crate) fn arg_opt_usize(args: &[Value], idx: usize) -> Option<usize> {
    match args.get(idx) {
        Some(Value::Null) | None => None,
        Some(v) => v.as_u64().map(|n| n as usize),
    }
}

pub(crate) fn arg_bool(args: &[Value], idx: usize) -> Result<bool, String> {
    args.get(idx)
        .and_then(|v| v.as_bool())
        .ok_or_else(|| format!("missing bool arg[{idx}]"))
}

pub(crate) fn arg_f64(args: &[Value], idx: usize) -> Result<f64, String> {
    args.get(idx)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("missing f64 arg[{idx}]"))
}

pub(crate) fn arg_opt_f64(args: &[Value], idx: usize) -> Option<f64> {
    match args.get(idx) {
        Some(Value::Null) | None => None,
        Some(v) => v.as_f64(),
    }
}

pub(crate) fn arg_usize_vec(args: &[Value], idx: usize) -> Result<Vec<usize>, String> {
    let arr = args
        .get(idx)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("missing u32_array arg[{idx}]"))?;
    let mut out = Vec::with_capacity(arr.len());
    for (pos, v) in arr.iter().enumerate() {
        let n = v
            .as_u64()
            .ok_or_else(|| format!("arg[{idx}][{pos}] must be non-negative integer"))?;
        let u = usize::try_from(n)
            .map_err(|_| format!("arg[{idx}][{pos}] overflows usize"))?;
        out.push(u);
    }
    Ok(out)
}

pub(crate) fn resolved_weights(
    spec: &SpecV1,
    layer_idx: usize,
    expected: usize,
) -> Result<Vec<f32>, String> {
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

pub(crate) fn resolved_weights_default_ones(
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

pub(crate) fn make_param_1d(
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

pub(crate) fn make_param_2d(
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

pub(crate) fn make_param_3d(
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

pub(crate) fn make_param_4d(
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

pub(crate) fn build_linear(
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

#[derive(Clone, Copy)]
pub enum ShiftDir {
    Up,
    Down,
    Left,
    Right,
}

pub enum RuntimeLayer {
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
    // Ported pure math ops (43)
    Abs,
    Exp,
    Log,
    Clamp { min: f32, max: f32 },
    Sin,
    Cos,
    Tan,
    Floor,
    Ceil,
    Round,
    Atan2,
    Powf,
    Hypot,
    Fma,
    AllFinite,
    Reshape([usize; 4]),
    Permute([usize; 4]),
    Slice { starts: [usize; 4], ends: [usize; 4] },
    SelectAxis { axis: usize, indices: Vec<usize> },
    FillLike(f32),
    ExpandLike,
    Dot,
    L2Norm,
    L2Distance,
    CosineSimilarity(f32),
    Sum,
    Mean,
    VariancePopulation,
    StdPopulation,
    VarianceSample,
    StdSample,
    Min,
    Max,
    SumAxis(usize),
    MeanAxis(usize),
    MinAxis(usize),
    MaxAxis(usize),
    Normalize,
    Entropy,
    CrossEntropy,
    KlDivergence,
    LessEqual01,
    IndicesLike(usize),
}

pub fn build_layers(
    spec: &SpecV1,
    device: &Device,
    grad_layers: &[usize],
) -> Result<(Vec<RuntimeLayer>, Vec<GradParam>), String> {
    let mut out = Vec::with_capacity(spec.layers.len());
    let mut grad_params = Vec::new();

    for (li, decl) in spec.layers.iter().enumerate() {
        let need_grad = grad_layers.contains(&li);
        let layer = if let Some(l) =
            activation::try_build(spec, li, decl, device, need_grad, &mut grad_params)?
        {
            l
        } else if let Some(l) =
            norm_linear::try_build(spec, li, decl, device, need_grad, &mut grad_params)?
        {
            l
        } else if let Some(l) =
            conv_pool::try_build(spec, li, decl, device, need_grad, &mut grad_params)?
        {
            l
        } else if let Some(l) =
            custom::try_build(spec, li, decl, device, need_grad, &mut grad_params)?
        {
            l
        } else if let Some(l) = binary_int::try_build(decl)? {
            l
        } else if let Some(l) = math_numeric::try_build(decl)? {
            l
        } else if let Some(l) = math_stats_prob::try_build(decl)? {
            l
        } else {
            return Err(format!("unsupported constructor '{}'", decl.constructor));
        };
        out.push(layer);
    }
    Ok((out, grad_params))
}

pub fn exec_layer(
    layer: &RuntimeLayer,
    inputs: &[SlotValue],
    device: &Device,
) -> Result<SlotValue, String> {
    if let Some(v) = activation::try_exec(layer, inputs)? {
        return Ok(v);
    }
    if let Some(v) = norm_linear::try_exec(layer, inputs, device)? {
        return Ok(v);
    }
    if let Some(v) = conv_pool::try_exec(layer, inputs)? {
        return Ok(v);
    }
    if let Some(v) = custom::try_exec(layer, inputs)? {
        return Ok(v);
    }
    if let Some(v) = binary_int::try_exec(layer, inputs, device)? {
        return Ok(v);
    }
    if let Some(v) = math_numeric::try_exec(layer, inputs, device)? {
        return Ok(v);
    }
    if let Some(v) = math_stats_prob::try_exec(layer, inputs, device)? {
        return Ok(v);
    }
    Err("unhandled runtime layer".into())
}
