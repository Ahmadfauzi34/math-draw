use burn::module::Initializer;
use burn::nn::{
    BatchNormConfig, EmbeddingConfig, GroupNormConfig, InstanceNormConfig, LayerNormConfig,
    RmsNormConfig,
};
use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::spec::{LayerDecl, SpecV1};

use super::{
    GradParam, RuntimeLayer, SlotValue, arg_bool, arg_opt_f64, arg_usize, build_linear,
    make_param_1d, make_param_2d, resolved_weights, resolved_weights_default_ones,
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
                grad_params,
            ))
        }
        "batchNorm" => {
            let nf = arg_usize(args, 1)?;
            let eps = arg_opt_f64(args, 2).unwrap_or(1e-5);
            let mut m = BatchNormConfig::new(nf).with_epsilon(eps).init(device);
            let w = resolved_weights_default_ones(spec, li, 2 * nf, nf)?;
            m.gamma = make_param_1d(&w[..nf], [nf], device, need_grad, li, 0, grad_params);
            m.beta = make_param_1d(&w[nf..], [nf], device, need_grad, li, nf, grad_params);
            RuntimeLayer::BatchNorm(m)
        }
        "groupNorm" => {
            let ng = arg_usize(args, 1)?;
            let nc = arg_usize(args, 2)?;
            let eps = arg_opt_f64(args, 3).unwrap_or(1e-5);
            let mut m = GroupNormConfig::new(ng, nc).with_epsilon(eps).init(device);
            let w = resolved_weights_default_ones(spec, li, 2 * nc, nc)?;
            m.gamma = Some(make_param_1d(&w[..nc], [nc], device, need_grad, li, 0, grad_params));
            m.beta = Some(make_param_1d(&w[nc..], [nc], device, need_grad, li, nc, grad_params));
            RuntimeLayer::GroupNorm(m)
        }
        "instanceNorm" => {
            let nc = arg_usize(args, 1)?;
            let eps = arg_opt_f64(args, 2).unwrap_or(1e-5);
            let mut m = InstanceNormConfig::new(nc).with_epsilon(eps).init(device);
            let w = resolved_weights_default_ones(spec, li, 2 * nc, nc)?;
            m.gamma = Some(make_param_1d(&w[..nc], [nc], device, need_grad, li, 0, grad_params));
            m.beta = Some(make_param_1d(&w[nc..], [nc], device, need_grad, li, nc, grad_params));
            RuntimeLayer::InstanceNorm(m)
        }
        "layerNorm" => {
            let sz = arg_usize(args, 1)?;
            let eps = arg_opt_f64(args, 2).unwrap_or(1e-5);
            let mut m = LayerNormConfig::new(sz).with_epsilon(eps).init(device);
            let w = resolved_weights_default_ones(spec, li, 2 * sz, sz)?;
            m.gamma = make_param_1d(&w[..sz], [sz], device, need_grad, li, 0, grad_params);
            m.beta = Some(make_param_1d(&w[sz..], [sz], device, need_grad, li, sz, grad_params));
            RuntimeLayer::LayerNorm(m)
        }
        "rmsNorm" => {
            let sz = arg_usize(args, 1)?;
            let eps = arg_opt_f64(args, 2).unwrap_or(1e-6);
            let mut m = RmsNormConfig::new(sz).with_epsilon(eps).init(device);
            let w = resolved_weights_default_ones(spec, li, sz, sz)?;
            m.gamma = make_param_1d(&w, [sz], device, need_grad, li, 0, grad_params);
            RuntimeLayer::RmsNorm(m)
        }
        "embedding" => {
            let vocab = arg_usize(args, 1)?;
            let d_model = arg_usize(args, 2)?;
            let mut m = EmbeddingConfig::new(vocab, d_model)
                .with_initializer(Initializer::Zeros)
                .init(device);
            let w = resolved_weights(spec, li, vocab * d_model)?;
            m.weight = make_param_2d(&w, [vocab, d_model], device, need_grad, li, 0, grad_params);
            RuntimeLayer::Embedding {
                module: m,
                vocab_size: vocab,
            }
        }
        _ => return Ok(None),
    };
    Ok(Some(layer))
}

pub(super) fn try_exec(
    layer: &RuntimeLayer,
    inputs: &[SlotValue],
    device: &Device,
) -> Result<Option<SlotValue>, String> {
    let val = match layer {
        RuntimeLayer::Linear(m) => {
            let x = inputs[0].as_f32("linear")?;
            let [b, f, _, _] = x.dims();
            let y2 = m.forward(x.reshape([b, f]));
            let [b2, o] = y2.dims();
            SlotValue::F32(y2.reshape([b2, o, 1, 1]))
        }
        RuntimeLayer::BatchNorm(m) => SlotValue::F32(m.forward(inputs[0].as_f32("batchNorm")?)),
        RuntimeLayer::GroupNorm(m) => SlotValue::F32(m.forward(inputs[0].as_f32("groupNorm")?)),
        RuntimeLayer::InstanceNorm(m) => {
            SlotValue::F32(m.forward(inputs[0].as_f32("instanceNorm")?))
        }
        RuntimeLayer::LayerNorm(m) => SlotValue::F32(m.forward(inputs[0].as_f32("layerNorm")?)),
        RuntimeLayer::RmsNorm(m) => SlotValue::F32(m.forward(inputs[0].as_f32("rmsNorm")?)),
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
            SlotValue::F32(out3.reshape([b2, s2, d, 1]))
        }
        _ => return Ok(None),
    };
    Ok(Some(val))
}
