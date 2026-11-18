use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamType {
    U32,
    Bool,
    F64,
    OptionU32,
    OptionF64,
    U32Array,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ParamMeta {
    pub name: &'static str,
    pub ty: ParamType,
    pub optional: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ConstructorMeta {
    pub name: &'static str,
    pub family: &'static str,
    pub arity: usize,
    pub signature: &'static str,
    pub expected_layout: &'static str,
    pub parameters: &'static [ParamMeta],
}

const P_ID: ParamMeta = ParamMeta { name: "id", ty: ParamType::U32, optional: false };
const P_DIM: ParamMeta = ParamMeta { name: "dim", ty: ParamType::U32, optional: false };
const P_DIM0: ParamMeta = ParamMeta { name: "dim0", ty: ParamType::U32, optional: false };
const P_DIM1: ParamMeta = ParamMeta { name: "dim1", ty: ParamType::U32, optional: false };

static PARAMS_ID_ONLY: &[ParamMeta] = &[P_ID];
static PARAMS_DIM: &[ParamMeta] = &[P_ID, P_DIM];
static PARAMS_TRANSPOSE: &[ParamMeta] = &[P_ID, P_DIM0, P_DIM1];

static PARAMS_LEAKY_RELU: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "negative_slope", ty: ParamType::F64, optional: false },
];
static PARAMS_PRELU: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "num_parameters", ty: ParamType::U32, optional: false },
    ParamMeta { name: "alpha", ty: ParamType::F64, optional: false },
];
static PARAMS_SWIGLU: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "d_input", ty: ParamType::U32, optional: false },
    ParamMeta { name: "d_output", ty: ParamType::U32, optional: false },
    ParamMeta { name: "bias", ty: ParamType::Bool, optional: false },
];
static PARAMS_HARDSIGMOID: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "alpha", ty: ParamType::F64, optional: false },
    ParamMeta { name: "beta", ty: ParamType::F64, optional: false },
];
static PARAMS_SOFTPLUS: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "beta", ty: ParamType::F64, optional: false },
];
static PARAMS_LINEAR: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "in_dim", ty: ParamType::U32, optional: false },
    ParamMeta { name: "out_dim", ty: ParamType::U32, optional: false },
    ParamMeta { name: "bias", ty: ParamType::Bool, optional: false },
];
static PARAMS_BATCH_NORM: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "num_features", ty: ParamType::U32, optional: false },
    ParamMeta { name: "epsilon", ty: ParamType::OptionF64, optional: true },
];
static PARAMS_GROUP_NORM: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "num_groups", ty: ParamType::U32, optional: false },
    ParamMeta { name: "num_channels", ty: ParamType::U32, optional: false },
    ParamMeta { name: "epsilon", ty: ParamType::OptionF64, optional: true },
];
static PARAMS_INSTANCE_NORM: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "num_channels", ty: ParamType::U32, optional: false },
    ParamMeta { name: "epsilon", ty: ParamType::OptionF64, optional: true },
];
static PARAMS_NORM_SIZE: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "size", ty: ParamType::U32, optional: false },
    ParamMeta { name: "epsilon", ty: ParamType::OptionF64, optional: true },
];
static PARAMS_EPS_OPT: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "epsilon", ty: ParamType::OptionF64, optional: true },
];
static PARAMS_CONV1D: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "in_ch", ty: ParamType::U32, optional: false },
    ParamMeta { name: "out_ch", ty: ParamType::U32, optional: false },
    ParamMeta { name: "kernel", ty: ParamType::U32, optional: false },
    ParamMeta { name: "stride", ty: ParamType::OptionU32, optional: true },
    ParamMeta { name: "padding", ty: ParamType::OptionU32, optional: true },
];
static PARAMS_CONV2D: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "in_ch", ty: ParamType::U32, optional: false },
    ParamMeta { name: "out_ch", ty: ParamType::U32, optional: false },
    ParamMeta { name: "kh", ty: ParamType::U32, optional: false },
    ParamMeta { name: "kw", ty: ParamType::U32, optional: false },
    ParamMeta { name: "sh", ty: ParamType::OptionU32, optional: true },
    ParamMeta { name: "sw", ty: ParamType::OptionU32, optional: true },
    ParamMeta { name: "ph", ty: ParamType::OptionU32, optional: true },
    ParamMeta { name: "pw", ty: ParamType::OptionU32, optional: true },
];
static PARAMS_EMBEDDING: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "vocab_size", ty: ParamType::U32, optional: false },
    ParamMeta { name: "d_model", ty: ParamType::U32, optional: false },
];
static PARAMS_POOL1D: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "kernel", ty: ParamType::U32, optional: false },
    ParamMeta { name: "stride", ty: ParamType::OptionU32, optional: true },
    ParamMeta { name: "padding", ty: ParamType::OptionU32, optional: true },
];
static PARAMS_POOL2D: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "kh", ty: ParamType::U32, optional: false },
    ParamMeta { name: "kw", ty: ParamType::U32, optional: false },
    ParamMeta { name: "sh", ty: ParamType::OptionU32, optional: true },
    ParamMeta { name: "sw", ty: ParamType::OptionU32, optional: true },
    ParamMeta { name: "ph", ty: ParamType::OptionU32, optional: true },
    ParamMeta { name: "pw", ty: ParamType::OptionU32, optional: true },
];
static PARAMS_ADAPTIVE_POOL2D: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "output_h", ty: ParamType::U32, optional: false },
    ParamMeta { name: "output_w", ty: ParamType::U32, optional: false },
];
static PARAMS_SHIFT: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "shift_size", ty: ParamType::U32, optional: false },
];
static PARAMS_GHOST: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "in_ch", ty: ParamType::U32, optional: false },
    ParamMeta { name: "out_ch", ty: ParamType::U32, optional: false },
    ParamMeta { name: "kh", ty: ParamType::U32, optional: false },
    ParamMeta { name: "kw", ty: ParamType::U32, optional: false },
    ParamMeta { name: "ratio", ty: ParamType::U32, optional: false },
    ParamMeta { name: "sh", ty: ParamType::OptionU32, optional: true },
    ParamMeta { name: "sw", ty: ParamType::OptionU32, optional: true },
    ParamMeta { name: "ph", ty: ParamType::OptionU32, optional: true },
    ParamMeta { name: "pw", ty: ParamType::OptionU32, optional: true },
];
static PARAMS_SEBLOCK: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "channels", ty: ParamType::U32, optional: false },
    ParamMeta { name: "reduction", ty: ParamType::U32, optional: false },
];
static PARAMS_CLAMP: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "min", ty: ParamType::F64, optional: false },
    ParamMeta { name: "max", ty: ParamType::F64, optional: false },
];
static PARAMS_FILL_LIKE: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "scalar", ty: ParamType::F64, optional: false },
];
static PARAMS_RESHAPE: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "shape", ty: ParamType::U32Array, optional: false },
];
static PARAMS_PERMUTE: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "axes", ty: ParamType::U32Array, optional: false },
];
static PARAMS_SLICE: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "starts", ty: ParamType::U32Array, optional: false },
    ParamMeta { name: "ends", ty: ParamType::U32Array, optional: false },
];
static PARAMS_SELECT_AXIS: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "axis", ty: ParamType::U32, optional: false },
    ParamMeta { name: "indices", ty: ParamType::U32Array, optional: false },
];
static PARAMS_AXIS: &[ParamMeta] = &[
    P_ID,
    ParamMeta { name: "axis", ty: ParamType::U32, optional: false },
];

pub static CATALOG: &[ConstructorMeta] = &[
    // Activations (15)
    ConstructorMeta { name: "relu", family: "activation", arity: 1, signature: "relu(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "gelu", family: "activation", arity: 1, signature: "gelu(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "sigmoid", family: "activation", arity: 1, signature: "sigmoid(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "tanh", family: "activation", arity: 1, signature: "tanh(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "hardSwish", family: "activation", arity: 1, signature: "hardSwish(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "leakyRelu", family: "activation", arity: 1, signature: "leakyRelu(id,negative_slope)", expected_layout: "any_rank4", parameters: PARAMS_LEAKY_RELU },
    ConstructorMeta { name: "prelu", family: "activation", arity: 1, signature: "prelu(id,num_parameters,alpha)", expected_layout: "channel_first", parameters: PARAMS_PRELU },
    ConstructorMeta { name: "swiGlu", family: "activation", arity: 1, signature: "swiGlu(id,d_input,d_output,bias)", expected_layout: "feature_last", parameters: PARAMS_SWIGLU },
    ConstructorMeta { name: "hardSigmoid", family: "activation", arity: 1, signature: "hardSigmoid(id,alpha,beta)", expected_layout: "any_rank4", parameters: PARAMS_HARDSIGMOID },
    ConstructorMeta { name: "softplus", family: "activation", arity: 1, signature: "softplus(id,beta)", expected_layout: "any_rank4", parameters: PARAMS_SOFTPLUS },
    ConstructorMeta { name: "mish", family: "activation", arity: 1, signature: "mish(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "softmax", family: "activation", arity: 1, signature: "softmax(id,dim)", expected_layout: "any_rank4", parameters: PARAMS_DIM },
    ConstructorMeta { name: "logSoftmax", family: "activation", arity: 1, signature: "logSoftmax(id,dim)", expected_layout: "any_rank4", parameters: PARAMS_DIM },
    ConstructorMeta { name: "glu", family: "activation", arity: 1, signature: "glu(id,dim)", expected_layout: "any_rank4", parameters: PARAMS_DIM },
    ConstructorMeta { name: "sqrt", family: "activation", arity: 1, signature: "sqrt(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    // Linear (1)
    ConstructorMeta { name: "linear", family: "linear", arity: 1, signature: "linear(id,in_dim,out_dim,bias)", expected_layout: "feature_axis1_singleton", parameters: PARAMS_LINEAR },
    // Norms (5)
    ConstructorMeta { name: "batchNorm", family: "norm", arity: 1, signature: "batchNorm(id,num_features,epsilon?)", expected_layout: "channel_first", parameters: PARAMS_BATCH_NORM },
    ConstructorMeta { name: "groupNorm", family: "norm", arity: 1, signature: "groupNorm(id,num_groups,num_channels,epsilon?)", expected_layout: "channel_first", parameters: PARAMS_GROUP_NORM },
    ConstructorMeta { name: "instanceNorm", family: "norm", arity: 1, signature: "instanceNorm(id,num_channels,epsilon?)", expected_layout: "channel_first", parameters: PARAMS_INSTANCE_NORM },
    ConstructorMeta { name: "layerNorm", family: "norm", arity: 1, signature: "layerNorm(id,size,epsilon?)", expected_layout: "feature_last", parameters: PARAMS_NORM_SIZE },
    ConstructorMeta { name: "rmsNorm", family: "norm", arity: 1, signature: "rmsNorm(id,size,epsilon?)", expected_layout: "feature_last", parameters: PARAMS_NORM_SIZE },
    // Convs (3)
    ConstructorMeta { name: "conv1d", family: "conv", arity: 1, signature: "conv1d(id,in_ch,out_ch,kernel,stride?,padding?)", expected_layout: "channel_first_singleton_width", parameters: PARAMS_CONV1D },
    ConstructorMeta { name: "conv2d", family: "conv", arity: 1, signature: "conv2d(id,in_ch,out_ch,kh,kw,sh?,sw?,ph?,pw?)", expected_layout: "channel_first", parameters: PARAMS_CONV2D },
    ConstructorMeta { name: "convTranspose2d", family: "conv", arity: 1, signature: "convTranspose2d(id,in_ch,out_ch,kh,kw,sh?,sw?,ph?,pw?)", expected_layout: "channel_first", parameters: PARAMS_CONV2D },
    // Embedding (1)
    ConstructorMeta { name: "embedding", family: "embedding", arity: 1, signature: "embedding(id,vocab_size,d_model)", expected_layout: "token_ids_axis1_singleton", parameters: PARAMS_EMBEDDING },
    // Pools (5)
    ConstructorMeta { name: "maxPool1d", family: "pool", arity: 1, signature: "maxPool1d(id,kernel,stride?,padding?)", expected_layout: "channel_first_singleton_width", parameters: PARAMS_POOL1D },
    ConstructorMeta { name: "maxPool2d", family: "pool", arity: 1, signature: "maxPool2d(id,kh,kw,sh?,sw?,ph?,pw?)", expected_layout: "channel_first", parameters: PARAMS_POOL2D },
    ConstructorMeta { name: "avgPool1d", family: "pool", arity: 1, signature: "avgPool1d(id,kernel,stride?,padding?)", expected_layout: "channel_first_singleton_width", parameters: PARAMS_POOL1D },
    ConstructorMeta { name: "avgPool2d", family: "pool", arity: 1, signature: "avgPool2d(id,kh,kw,sh?,sw?,ph?,pw?)", expected_layout: "channel_first", parameters: PARAMS_POOL2D },
    ConstructorMeta { name: "adaptiveAvgPool2d", family: "pool", arity: 1, signature: "adaptiveAvgPool2d(id,output_h,output_w)", expected_layout: "channel_first", parameters: PARAMS_ADAPTIVE_POOL2D },
    // Custom (7)
    ConstructorMeta { name: "featureNorm", family: "feature_norm", arity: 1, signature: "featureNorm(id,epsilon?)", expected_layout: "feature_axis1_singleton", parameters: PARAMS_EPS_OPT },
    ConstructorMeta { name: "shiftUp", family: "shift", arity: 1, signature: "shiftUp(id,shift_size)", expected_layout: "channel_first", parameters: PARAMS_SHIFT },
    ConstructorMeta { name: "shiftDown", family: "shift", arity: 1, signature: "shiftDown(id,shift_size)", expected_layout: "channel_first", parameters: PARAMS_SHIFT },
    ConstructorMeta { name: "shiftLeft", family: "shift", arity: 1, signature: "shiftLeft(id,shift_size)", expected_layout: "channel_first", parameters: PARAMS_SHIFT },
    ConstructorMeta { name: "shiftRight", family: "shift", arity: 1, signature: "shiftRight(id,shift_size)", expected_layout: "channel_first", parameters: PARAMS_SHIFT },
    ConstructorMeta { name: "ghost", family: "ghost", arity: 1, signature: "ghost(id,in_ch,out_ch,kh,kw,ratio,sh?,sw?,ph?,pw?)", expected_layout: "channel_first", parameters: PARAMS_GHOST },
    ConstructorMeta { name: "seBlock", family: "seblock", arity: 1, signature: "seBlock(id,channels,reduction)", expected_layout: "channel_first", parameters: PARAMS_SEBLOCK },
    // Binary f32 (6)
    ConstructorMeta { name: "add", family: "binary", arity: 2, signature: "add(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "sub", family: "binary", arity: 2, signature: "sub(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "mul", family: "binary", arity: 2, signature: "mul(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "matmul", family: "binary", arity: 2, signature: "matmul(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "div", family: "binary", arity: 2, signature: "div(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "concat", family: "binary", arity: 2, signature: "concat(id,dim)", expected_layout: "any_rank4", parameters: PARAMS_DIM },
    // Transform (1)
    ConstructorMeta { name: "transpose", family: "unary", arity: 1, signature: "transpose(id,dim0,dim1)", expected_layout: "any_rank4", parameters: PARAMS_TRANSPOSE },
    // Integer i64 ops (8)
    ConstructorMeta { name: "intAdd", family: "int_binary", arity: 2, signature: "intAdd(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "intSub", family: "int_binary", arity: 2, signature: "intSub(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "intMul", family: "int_binary", arity: 2, signature: "intMul(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "intMatmul", family: "int_binary", arity: 2, signature: "intMatmul(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "intDiv", family: "int_binary", arity: 2, signature: "intDiv(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "intRem", family: "int_binary", arity: 2, signature: "intRem(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "intToFloat", family: "int_cast", arity: 1, signature: "intToFloat(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "intFromFloat", family: "int_cast", arity: 1, signature: "intFromFloat(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    // Numeric primitives (15)
    ConstructorMeta { name: "abs", family: "numeric", arity: 1, signature: "abs(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "exp", family: "numeric", arity: 1, signature: "exp(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "log", family: "numeric", arity: 1, signature: "log(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "clamp", family: "numeric", arity: 1, signature: "clamp(id,min,max)", expected_layout: "any_rank4", parameters: PARAMS_CLAMP },
    ConstructorMeta { name: "sin", family: "numeric", arity: 1, signature: "sin(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "cos", family: "numeric", arity: 1, signature: "cos(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "tan", family: "numeric", arity: 1, signature: "tan(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "floor", family: "numeric", arity: 1, signature: "floor(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "ceil", family: "numeric", arity: 1, signature: "ceil(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "round", family: "numeric", arity: 1, signature: "round(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "atan2", family: "numeric", arity: 2, signature: "atan2(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "powf", family: "numeric", arity: 2, signature: "powf(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "hypot", family: "numeric", arity: 2, signature: "hypot(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "fma", family: "numeric", arity: 3, signature: "fma(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "allFinite", family: "numeric", arity: 1, signature: "allFinite(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    // Tensor transforms (6)
    ConstructorMeta { name: "reshape", family: "transform", arity: 1, signature: "reshape(id,shape)", expected_layout: "any_rank4", parameters: PARAMS_RESHAPE },
    ConstructorMeta { name: "permute", family: "transform", arity: 1, signature: "permute(id,axes)", expected_layout: "any_rank4", parameters: PARAMS_PERMUTE },
    ConstructorMeta { name: "slice", family: "transform", arity: 1, signature: "slice(id,starts,ends)", expected_layout: "any_rank4", parameters: PARAMS_SLICE },
    ConstructorMeta { name: "selectAxis", family: "transform", arity: 1, signature: "selectAxis(id,axis,indices)", expected_layout: "any_rank4", parameters: PARAMS_SELECT_AXIS },
    ConstructorMeta { name: "fillLike", family: "transform", arity: 1, signature: "fillLike(id,scalar)", expected_layout: "any_rank4", parameters: PARAMS_FILL_LIKE },
    ConstructorMeta { name: "expandLike", family: "transform", arity: 2, signature: "expandLike(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    // Linear algebra vectors (4)
    ConstructorMeta { name: "dot", family: "linalg", arity: 2, signature: "dot(id)", expected_layout: "feature_axis1_singleton", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "l2Norm", family: "linalg", arity: 1, signature: "l2Norm(id)", expected_layout: "feature_axis1_singleton", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "l2Distance", family: "linalg", arity: 2, signature: "l2Distance(id)", expected_layout: "feature_axis1_singleton", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "cosineSimilarity", family: "linalg", arity: 2, signature: "cosineSimilarity(id,epsilon?)", expected_layout: "feature_axis1_singleton", parameters: PARAMS_EPS_OPT },
    // Statistics over [B,F,1,1] (8)
    ConstructorMeta { name: "sum", family: "statistics", arity: 1, signature: "sum(id)", expected_layout: "feature_axis1_singleton", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "mean", family: "statistics", arity: 1, signature: "mean(id)", expected_layout: "feature_axis1_singleton", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "variancePopulation", family: "statistics", arity: 1, signature: "variancePopulation(id)", expected_layout: "feature_axis1_singleton", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "stdPopulation", family: "statistics", arity: 1, signature: "stdPopulation(id)", expected_layout: "feature_axis1_singleton", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "varianceSample", family: "statistics", arity: 1, signature: "varianceSample(id)", expected_layout: "feature_axis1_singleton", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "stdSample", family: "statistics", arity: 1, signature: "stdSample(id)", expected_layout: "feature_axis1_singleton", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "min", family: "statistics", arity: 1, signature: "min(id)", expected_layout: "feature_axis1_singleton", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "max", family: "statistics", arity: 1, signature: "max(id)", expected_layout: "feature_axis1_singleton", parameters: PARAMS_ID_ONLY },
    // Rank-4 axis reduction keepdim (4)
    ConstructorMeta { name: "sumAxis", family: "reduction", arity: 1, signature: "sumAxis(id,axis)", expected_layout: "any_rank4", parameters: PARAMS_AXIS },
    ConstructorMeta { name: "meanAxis", family: "reduction", arity: 1, signature: "meanAxis(id,axis)", expected_layout: "any_rank4", parameters: PARAMS_AXIS },
    ConstructorMeta { name: "minAxis", family: "reduction", arity: 1, signature: "minAxis(id,axis)", expected_layout: "any_rank4", parameters: PARAMS_AXIS },
    ConstructorMeta { name: "maxAxis", family: "reduction", arity: 1, signature: "maxAxis(id,axis)", expected_layout: "any_rank4", parameters: PARAMS_AXIS },
    // Probability distributions over [B,F,1,1] (4)
    ConstructorMeta { name: "normalize", family: "probability", arity: 1, signature: "normalize(id)", expected_layout: "feature_axis1_singleton", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "entropy", family: "probability", arity: 1, signature: "entropy(id)", expected_layout: "feature_axis1_singleton", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "crossEntropy", family: "probability", arity: 2, signature: "crossEntropy(id)", expected_layout: "feature_axis1_singleton", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "klDivergence", family: "probability", arity: 2, signature: "klDivergence(id)", expected_layout: "feature_axis1_singleton", parameters: PARAMS_ID_ONLY },
    // Comparison & Index source (2)
    ConstructorMeta { name: "lessEqual01", family: "comparison", arity: 2, signature: "lessEqual01(id)", expected_layout: "any_rank4", parameters: PARAMS_ID_ONLY },
    ConstructorMeta { name: "indicesLike", family: "index_source", arity: 1, signature: "indicesLike(id,axis)", expected_layout: "any_rank4", parameters: PARAMS_AXIS },
];

pub fn find_constructor(name: &str) -> Option<&'static ConstructorMeta> {
    CATALOG.iter().find(|c| c.name == name)
}

pub fn is_int_constructor(name: &str) -> bool {
    matches!(
        name,
        "intAdd" | "intSub" | "intMul" | "intMatmul" | "intDiv" | "intRem" | "intToFloat" | "intFromFloat"
    )
}

pub fn is_int_binary(name: &str) -> bool {
    matches!(
        name,
        "intAdd" | "intSub" | "intMul" | "intMatmul" | "intDiv" | "intRem"
    )
}

pub fn expected_weight_count(ctor: &str, args: &[serde_json::Value]) -> Option<usize> {
    let u = |idx: usize| args.get(idx).and_then(|v| v.as_u64()).map(|v| v as usize);
    let b = |idx: usize| args.get(idx).and_then(|v| v.as_bool());
    match ctor {
        "linear" => {
            let in_dim = u(1)?;
            let out_dim = u(2)?;
            let bias = b(3).unwrap_or(false);
            Some(in_dim * out_dim + if bias { out_dim } else { 0 })
        }
        "batchNorm" => Some(2 * u(1)?),
        "groupNorm" | "instanceNorm" => Some(2 * u(2)?),
        "layerNorm" => Some(2 * u(1)?),
        "rmsNorm" => Some(u(1)?),
        "conv1d" => {
            let in_ch = u(1)?;
            let out_ch = u(2)?;
            let k = u(3)?;
            Some(in_ch * out_ch * k + out_ch)
        }
        "conv2d" | "convTranspose2d" => {
            let in_ch = u(1)?;
            let out_ch = u(2)?;
            let kh = u(3)?;
            let kw = u(4)?;
            Some(in_ch * out_ch * kh * kw + out_ch)
        }
        "embedding" => Some(u(1)? * u(2)?),
        "prelu" => Some(u(1)?),
        "swiGlu" => {
            let d_in = u(1)?;
            let d_out = u(2)?;
            let bias = b(3).unwrap_or(false);
            let lin = d_in * d_out + if bias { d_out } else { 0 };
            Some(2 * lin)
        }
        "seBlock" => {
            let ch = u(1)?;
            let red = u(2)?;
            if red == 0 || ch < red {
                return None;
            }
            let r = ch / red;
            Some((ch * r + r) + (r * ch + ch))
        }
        "ghost" => {
            let in_ch = u(1)?;
            let out_ch = u(2)?;
            let kh = u(3)?;
            let kw = u(4)?;
            let ratio = u(5)?;
            if ratio == 0 || out_ch % ratio != 0 {
                return None;
            }
            let primary_ch = out_ch / ratio;
            let primary = in_ch * primary_ch * kh * kw + primary_ch;
            let cheap = primary_ch; // depthwise 1x1, no bias
            Some(primary + cheap)
        }
        _ => None,
    }
}
