use burn::module::Initializer;
use burn::nn::activation::{
    HardSigmoidConfig, HardSwish, LeakyReluConfig, PReluConfig, SwiGluConfig,
};
use burn::nn::{Gelu, Relu, Sigmoid, SoftplusConfig, Tanh};
use burn::tensor::Device;

use crate::spec::{LayerDecl, SpecV1};

use super::{
    GradParam, RuntimeLayer, SlotValue, arg_bool, arg_f64, arg_usize, build_linear, make_param_1d,
    resolved_weights,
};

pub(super) fn try_build(
    spec: &SpecV1,
    li: usize,
    decl: &LayerDecl,
    device: &Device,
    need_grad: bool,
    grad_params: &mut Vec<GradParam>,
) -> Result<Option<RuntimeLayer>, String> {
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
            m.alpha = make_param_1d(&w, [np], device, need_grad, li, 0, grad_params);
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
                grad_params,
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
                grad_params,
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
        _ => return Ok(None),
    };
    Ok(Some(layer))
}

pub(super) fn try_exec(layer: &RuntimeLayer, inputs: &[SlotValue]) -> Result<Option<SlotValue>, String> {
    let val = match layer {
        RuntimeLayer::Relu(m) => SlotValue::F32(m.forward(inputs[0].as_f32("relu")?)),
        RuntimeLayer::Gelu(m) => SlotValue::F32(m.forward(inputs[0].as_f32("gelu")?)),
        RuntimeLayer::Sigmoid(m) => SlotValue::F32(m.forward(inputs[0].as_f32("sigmoid")?)),
        RuntimeLayer::Tanh(m) => SlotValue::F32(m.forward(inputs[0].as_f32("tanh")?)),
        RuntimeLayer::HardSwish(m) => SlotValue::F32(m.forward(inputs[0].as_f32("hardSwish")?)),
        RuntimeLayer::LeakyRelu(m) => SlotValue::F32(m.forward(inputs[0].as_f32("leakyRelu")?)),
        RuntimeLayer::PRelu(m) => SlotValue::F32(m.forward(inputs[0].as_f32("prelu")?)),
        RuntimeLayer::SwiGlu(m) => SlotValue::F32(m.forward(inputs[0].as_f32("swiGlu")?)),
        RuntimeLayer::HardSigmoid(m) => SlotValue::F32(m.forward(inputs[0].as_f32("hardSigmoid")?)),
        RuntimeLayer::Softplus(m) => SlotValue::F32(m.forward(inputs[0].as_f32("softplus")?)),
        RuntimeLayer::Mish => SlotValue::F32(burn::tensor::activation::mish(
            inputs[0].as_f32("mish")?,
        )),
        RuntimeLayer::Softmax(dim) => SlotValue::F32(burn::tensor::activation::softmax(
            inputs[0].as_f32("softmax")?,
            *dim,
        )),
        RuntimeLayer::LogSoftmax(dim) => SlotValue::F32(burn::tensor::activation::log_softmax(
            inputs[0].as_f32("logSoftmax")?,
            *dim,
        )),
        RuntimeLayer::Glu(dim) => SlotValue::F32(burn::tensor::activation::glu(
            inputs[0].as_f32("glu")?,
            *dim,
        )),
        RuntimeLayer::Sqrt => SlotValue::F32(inputs[0].as_f32("sqrt")?.sqrt()),
        _ => return Ok(None),
    };
    Ok(Some(val))
}
