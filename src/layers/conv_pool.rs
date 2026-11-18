use burn::module::Initializer;
use burn::nn::conv::{Conv1dConfig, Conv2dConfig, ConvTranspose2dConfig};
use burn::nn::pool::{
    AdaptiveAvgPool2dConfig, AvgPool1dConfig, AvgPool2dConfig, MaxPool1dConfig, MaxPool2dConfig,
};
use burn::nn::{PaddingConfig1d, PaddingConfig2d};
use burn::tensor::Device;

use crate::spec::{LayerDecl, SpecV1};

use super::{
    GradParam, RuntimeLayer, SlotValue, arg_opt_usize, arg_usize, make_param_1d, make_param_3d,
    make_param_4d, resolved_weights,
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
            m.weight = make_param_3d(&w[..w_len], [out_ch, in_ch, k], device, need_grad, li, 0, grad_params);
            m.bias = Some(make_param_1d(&w[w_len..], [out_ch], device, need_grad, li, w_len, grad_params));
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
            m.weight = make_param_4d(&w[..w_len], [out_ch, in_ch, kh, kw], device, need_grad, li, 0, grad_params);
            m.bias = Some(make_param_1d(&w[w_len..], [out_ch], device, need_grad, li, w_len, grad_params));
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
            m.weight = make_param_4d(&w[..w_len], [in_ch, out_ch, kh, kw], device, need_grad, li, 0, grad_params);
            m.bias = Some(make_param_1d(&w[w_len..], [out_ch], device, need_grad, li, w_len, grad_params));
            RuntimeLayer::ConvTranspose2d(m)
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
        _ => return Ok(None),
    };
    Ok(Some(layer))
}

pub(super) fn try_exec(layer: &RuntimeLayer, inputs: &[SlotValue]) -> Result<Option<SlotValue>, String> {
    let val = match layer {
        RuntimeLayer::Conv1d(m) => {
            let x = inputs[0].as_f32("conv1d")?;
            let [b, c, l, _] = x.dims();
            let y = m.forward(x.reshape([b, c, l]));
            let [b2, c2, l2] = y.dims();
            SlotValue::F32(y.reshape([b2, c2, l2, 1]))
        }
        RuntimeLayer::Conv2d(m) => SlotValue::F32(m.forward(inputs[0].as_f32("conv2d")?)),
        RuntimeLayer::ConvTranspose2d(m) => {
            SlotValue::F32(m.forward(inputs[0].as_f32("convTranspose2d")?))
        }
        RuntimeLayer::MaxPool1d(m) => {
            let x = inputs[0].as_f32("maxPool1d")?;
            let [b, c, l, _] = x.dims();
            let y = m.forward(x.reshape([b, c, l]));
            let [b2, c2, l2] = y.dims();
            SlotValue::F32(y.reshape([b2, c2, l2, 1]))
        }
        RuntimeLayer::AvgPool1d(m) => {
            let x = inputs[0].as_f32("avgPool1d")?;
            let [b, c, l, _] = x.dims();
            let y = m.forward(x.reshape([b, c, l]));
            let [b2, c2, l2] = y.dims();
            SlotValue::F32(y.reshape([b2, c2, l2, 1]))
        }
        RuntimeLayer::MaxPool2d(m) => SlotValue::F32(m.forward(inputs[0].as_f32("maxPool2d")?)),
        RuntimeLayer::AvgPool2d(m) => SlotValue::F32(m.forward(inputs[0].as_f32("avgPool2d")?)),
        RuntimeLayer::AdaptiveAvgPool2d(m) => {
            SlotValue::F32(m.forward(inputs[0].as_f32("adaptiveAvgPool2d")?))
        }
        _ => return Ok(None),
    };
    Ok(Some(val))
}
