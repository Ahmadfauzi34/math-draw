use burn::module::Initializer;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::AdaptiveAvgPool2dConfig;
use burn::tensor::{Device, Tensor};

use crate::spec::{LayerDecl, SpecV1};

use super::{
    GradParam, RuntimeLayer, ShiftDir, SlotValue, arg_opt_f64, arg_opt_usize, arg_usize,
    build_linear, make_param_1d, make_param_4d, resolved_weights,
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
        "featureNorm" => {
            let eps = arg_opt_f64(args, 1).unwrap_or(1e-12);
            RuntimeLayer::FeatureNorm(eps)
        }
        "shiftUp" => RuntimeLayer::Shift {
            dir: ShiftDir::Up,
            size: arg_usize(args, 1)?,
        },
        "shiftDown" => RuntimeLayer::Shift {
            dir: ShiftDir::Down,
            size: arg_usize(args, 1)?,
        },
        "shiftLeft" => RuntimeLayer::Shift {
            dir: ShiftDir::Left,
            size: arg_usize(args, 1)?,
        },
        "shiftRight" => RuntimeLayer::Shift {
            dir: ShiftDir::Right,
            size: arg_usize(args, 1)?,
        },
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
            primary.weight = make_param_4d(&w[..pw_len], [pch, in_ch, kh, kw], device, need_grad, li, 0, grad_params);
            primary.bias = Some(make_param_1d(&w[pw_len..pw_len + pch], [pch], device, need_grad, li, pw_len, grad_params));
            let mut cheap = Conv2dConfig::new([pch, pch], [1, 1])
                .with_groups(pch)
                .with_bias(false)
                .with_initializer(Initializer::Zeros)
                .init(device);
            cheap.weight = make_param_4d(&w[pw_len + pch..], [pch, 1, 1, 1], device, need_grad, li, pw_len + pch, grad_params);
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
            let fc1 = build_linear(ch, r, true, &w[..l1], device, need_grad, li, 0, grad_params);
            let fc2 = build_linear(r, ch, true, &w[l1..], device, need_grad, li, l1, grad_params);
            RuntimeLayer::SeBlock { squeeze, fc1, fc2 }
        }
        _ => return Ok(None),
    };
    Ok(Some(layer))
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

pub(super) fn try_exec(layer: &RuntimeLayer, inputs: &[SlotValue]) -> Result<Option<SlotValue>, String> {
    let val = match layer {
        RuntimeLayer::FeatureNorm(eps) => {
            let x = inputs[0].as_f32("featureNorm")?;
            let norm = x.clone().square().sum_dim(1).sqrt().clamp_min(*eps as f32);
            SlotValue::F32(x / norm)
        }
        RuntimeLayer::Shift { dir, size } => {
            SlotValue::F32(run_shift(inputs[0].as_f32("shift")?, *dir, *size))
        }
        RuntimeLayer::Ghost { primary, cheap } => {
            let x = inputs[0].as_f32("ghost")?;
            let intrinsic = primary.forward(x);
            let ghost = cheap.forward(intrinsic.clone());
            SlotValue::F32(Tensor::cat(vec![intrinsic, ghost], 1))
        }
        RuntimeLayer::SeBlock { squeeze, fc1, fc2 } => {
            let x = inputs[0].as_f32("seBlock")?;
            let [b, c, _, _] = x.dims();
            let pooled = squeeze.forward(x.clone()).reshape([b, c]);
            let s = burn::tensor::activation::relu(fc1.forward(pooled));
            let s = burn::tensor::activation::sigmoid(fc2.forward(s)).reshape([b, c, 1, 1]);
            SlotValue::F32(x * s)
        }
        _ => return Ok(None),
    };
    Ok(Some(val))
}
