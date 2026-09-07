// Copyright 2026 OpenAI.
// Apache-2.0 weight-compatible CPU inference kernel for Mamba-3 SISO.
// The recurrence follows the public state-spaces/mamba reference equations.
//
// PyTorch bindings around the torch-free kernels in montlok_kernel.h,
// montlok_mhc.h and montlok_layers.h:
//   mamba3_siso_recurrent  vectorised recurrence over prepared q/k/v/gate
//   mamba3_siso_forward    fused in_proj-output -> pre-out_proj forward
//   mhc_prepare            mHC coefficients + aggregated input for one residual
//   mhc_combine            mHC stream mixing + write of the wrapped layer output
//   mhc_combine_prepare    the write of one residual fused with the read of the next
//   mhc_combine_collapse   the final write fused with the mean over streams
//   rms_norm, layer_norm   (Grouped)RMSNorm / LayerNorm in one op
//   swiglu                 silu(gate) * up in one op
//   mla_rope_qk            MLA rotary embedding of q (in place) + key assembly
//   build_info             which SIMD path this build compiled to

#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/scaled_dot_product_attention.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "montlok_kernel.h"
#include "montlok_layers.h"
#include "montlok_mhc.h"

namespace {

void require_f32_cpu(const torch::Tensor& value, const char* name) {
    TORCH_CHECK(value.defined(), name, " must be defined");
    TORCH_CHECK(value.device().is_cpu(), name, " must be on CPU");
    TORCH_CHECK(value.scalar_type() == torch::kFloat32, name, " must be float32");
}

// The kernels index the leading dimensions through explicit strides and only
// require the innermost dimension to be dense, so slices of the in_proj
// output can be consumed without .contiguous() copies.
void require_inner_dense(const torch::Tensor& value, const char* name) {
    TORCH_CHECK(value.dim() >= 1 && (value.stride(-1) == 1 || value.size(-1) == 1),
                name, " must be contiguous along its last dimension");
}

montlok::Strides3 leading_strides(const torch::Tensor& value) {
    return {value.stride(0), value.stride(1), value.stride(2)};
}

torch::Tensor dense_parameter(const torch::Tensor& value, int64_t numel, const char* name) {
    require_f32_cpu(value, name);
    TORCH_CHECK(value.numel() == numel, name, " must have ", numel, " elements, got ", value.numel());
    return value.contiguous();
}

torch::Tensor linear_dispatch(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias = std::nullopt) {
    const char* dnnl_env = std::getenv("MONTLOK_CPP_DNNL");
    const char* min_rows_env = std::getenv("MONTLOK_DNNL_MIN_ROWS");
    const int64_t min_rows = min_rows_env != nullptr ? std::max<int64_t>(1, std::atoll(min_rows_env)) : 128;
    const int64_t rows = input.dim() > 0 && input.size(-1) > 0 ? input.numel() / input.size(-1) : 0;
    if (!at::hasMKLDNN() || rows < min_rows || (dnnl_env != nullptr && std::string_view(dnnl_env) == "0")) {
        return at::linear(input, weight, bias);
    }
    static const auto op = c10::Dispatcher::singleton().findSchemaOrThrow("mkldnn::_linear_pointwise", "");
    c10::Stack stack;
    stack.reserve(6);
    stack.emplace_back(input);
    stack.emplace_back(weight);
    stack.emplace_back(bias.has_value() ? c10::IValue(*bias) : c10::IValue());
    stack.emplace_back(std::string("none"));
    stack.emplace_back(c10::List<std::optional<at::Scalar>>());
    stack.emplace_back(c10::IValue());
    op.callBoxed(&stack);
    TORCH_INTERNAL_ASSERT(stack.size() == 1, "oneDNN linear returned an invalid boxed stack");
    return std::move(stack.back()).toTensor();
}

// Runs fn(begin, end) over [0, items) on torch's intra-op thread pool.
//
// Linux builds use -fopenmp and bind to the libgomp.so.1 already loaded by the
// PyTorch wheel, so at::parallel_for can dispatch directly. Builds without an
// OpenMP compiler use TensorIteratorBase::for_each as the thread-pool bridge.
// Both paths honour torch.set_num_threads().
//
// Small problems are run inline: a fork/join costs tens of microseconds on a
// 16-32 thread pool, more than the arithmetic of a 480-token RMSNorm.
// `work_per_item` is an estimate in scalar float operations; the threshold is
// configurable with MONTLOK_SERIAL_WORK (default 100000).
int64_t serial_work_threshold() {
    static const int64_t threshold = [] {
        const char* env = std::getenv("MONTLOK_SERIAL_WORK");
        return env ? static_cast<int64_t>(std::atoll(env)) : int64_t{100000};
    }();
    return threshold;
}

template <typename Fn>
void parallel_items(
    int64_t items,
    int64_t grain,
    const Fn& fn,
    int64_t work_per_item = 1 << 20,
    const torch::Tensor& dispatch_storage = torch::Tensor()) {
    if (items <= 0) {
        return;
    }
    if (items * work_per_item < serial_work_threshold()) {
        fn(0, items);
        return;
    }
#ifdef _OPENMP
    // Linux wheels already load PyTorch's bundled libgomp.so.1. Building this
    // extension with -fopenmp binds to that same SONAME and lets ATen dispatch
    // directly without a TensorIterator trampoline.
    at::parallel_for(0, items, std::max<int64_t>(grain, 1), fn);
    return;
#endif
    // The tensor is never read: the chunk position comes from its pointer
    // offset. Reuse an already allocated contiguous output whenever possible;
    // otherwise fall back to a tiny uninitialized dispatch buffer.
    torch::Tensor indices;
    if (dispatch_storage.defined() && dispatch_storage.is_contiguous() && dispatch_storage.numel() >= items) {
        indices = dispatch_storage.view({-1}).narrow(0, 0, items);
    } else {
        indices = torch::empty({items}, torch::dtype(torch::kLong));
    }
    const char* base = reinterpret_cast<const char*>(indices.data_ptr());
    const int64_t element_size = indices.element_size();
    auto iter = at::TensorIteratorConfig().add_output(indices).add_input(indices).build();
    iter.for_each(
        [&](char** data, const int64_t* /*strides*/, int64_t size) {
            const int64_t begin = static_cast<int64_t>((data[1] - base) / static_cast<std::ptrdiff_t>(element_size));
            fn(begin, begin + size);
        },
        std::max<int64_t>(grain, 1));
}

void run_recurrence(const montlok::SisoPlan& plan, const torch::Tensor& dispatch_storage) {
    const int64_t items = montlok::siso_work_items(plan);
    const int64_t scratch_floats = montlok::siso_scratch_floats(plan);
    parallel_items(items, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> scratch(static_cast<size_t>(scratch_floats));
        for (int64_t item = begin; item < end; ++item) {
            montlok::siso_run_work_item(plan, item, scratch.data());
        }
    }, 1 << 20, dispatch_storage);
}

}  // namespace

// q, k: [B, L, H, N]; v, gate: [B, L, H, D]; adt, dt, trap_logit: [B, L, H];
// skip: [H]. Any leading-dimension strides are accepted (only the last
// dimension must be dense). gate must already be z * sigmoid(z). Returns the
// gated SSM output [B, L, H, D].
torch::Tensor mamba3_siso_recurrent(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& gate,
    const torch::Tensor& adt,
    const torch::Tensor& dt,
    const torch::Tensor& trap_logit,
    const torch::Tensor& skip) {
    require_f32_cpu(q, "q");
    require_f32_cpu(k, "k");
    require_f32_cpu(v, "v");
    require_f32_cpu(gate, "gate");
    require_f32_cpu(adt, "adt");
    require_f32_cpu(dt, "dt");
    require_f32_cpu(trap_logit, "trap_logit");
    TORCH_CHECK(q.dim() == 4 && k.sizes() == q.sizes(), "q/k must be [B,L,H,N]");
    TORCH_CHECK(v.dim() == 4 && gate.sizes() == v.sizes(), "v/gate must be [B,L,H,D]");
    const int64_t batch = q.size(0);
    const int64_t length = q.size(1);
    const int64_t heads = q.size(2);
    const int64_t state_dim = q.size(3);
    const int64_t value_dim = v.size(3);
    TORCH_CHECK(v.size(0) == batch && v.size(1) == length && v.size(2) == heads, "v shape mismatch");
    TORCH_CHECK(adt.sizes() == torch::IntArrayRef({batch, length, heads}), "adt must be [B,L,H]");
    TORCH_CHECK(dt.sizes() == adt.sizes() && trap_logit.sizes() == adt.sizes(), "dt/trap_logit shape mismatch");
    require_inner_dense(q, "q");
    require_inner_dense(k, "k");
    require_inner_dense(v, "v");
    require_inner_dense(gate, "gate");
    const torch::Tensor skip_dense = dense_parameter(skip, heads, "skip");

    // Per-token coefficients are shared by all headdim blocks of a head, so
    // they are computed once here instead of inside every work item.
    auto coefficients = torch::empty({3, batch, length, heads}, q.options());
    float* alpha = coefficients.data_ptr<float>();
    float* beta = alpha + batch * length * heads;
    float* gamma = beta + batch * length * heads;
    {
        const auto adt_acc = adt.accessor<float, 3>();
        const auto dt_acc = dt.accessor<float, 3>();
        const auto trap_acc = trap_logit.accessor<float, 3>();
        parallel_items(batch * length * heads, 2048, [&](int64_t begin, int64_t end) {
            for (int64_t index = begin; index < end; ++index) {
                const int64_t h = index % heads;
                const int64_t t = (index / heads) % length;
                const int64_t b = index / (heads * length);
                const montlok::StepCoefficients coef =
                    montlok::step_coefficients(adt_acc[b][t][h], dt_acc[b][t][h], trap_acc[b][t][h]);
                alpha[index] = coef.alpha;
                beta[index] = coef.beta;
                gamma[index] = coef.gamma;
            }
        }, 1 << 20, coefficients);
    }

    auto output = torch::empty({batch, length, heads, value_dim}, v.options());
    montlok::SisoPlan plan{};
    plan.B = batch;
    plan.L = length;
    plan.H = heads;
    plan.N = state_dim;
    plan.D = value_dim;
    plan.q = q.data_ptr<float>();
    plan.qs = leading_strides(q);
    plan.k = k.data_ptr<float>();
    plan.ks = leading_strides(k);
    plan.v = v.data_ptr<float>();
    plan.vs = leading_strides(v);
    plan.gate = gate.data_ptr<float>();
    plan.gs = leading_strides(gate);
    plan.gate_logits = false;
    plan.alpha = alpha;
    plan.beta = beta;
    plan.gamma = gamma;
    plan.skip = skip_dense.data_ptr<float>();
    plan.out = output.data_ptr<float>();
    plan.os = leading_strides(output);
    run_recurrence(plan, output);
    return output;
}

// Fused Mamba-3 SISO forward between in_proj and out_proj.
//   projected: [B, L, W] in_proj output, W = 2*d_inner + 2*ngroups*d_state + 3*nheads + num_rope_angles
//   gate:      [B, L, d_inner] = silu(projected[..., :d_inner]), or an empty
//              tensor to evaluate the silu of the z lanes inside the kernel
//   b_bias, c_bias: [H, 1, N]; b_norm_w, c_norm_w: [N]; dt_bias, skip (D): [H]
// Returns [B, L, d_inner] ready for out_proj.
torch::Tensor mamba3_siso_forward(
    const torch::Tensor& projected,
    const torch::Tensor& gate,
    const torch::Tensor& b_bias,
    const torch::Tensor& c_bias,
    const torch::Tensor& b_norm_w,
    const torch::Tensor& c_norm_w,
    const torch::Tensor& dt_bias,
    const torch::Tensor& skip,
    int64_t d_state,
    int64_t ngroups,
    int64_t nheads,
    int64_t headdim,
    int64_t num_rope_angles,
    double b_eps,
    double c_eps,
    double a_floor) {
    require_f32_cpu(projected, "projected");
    const bool gate_logits = gate.numel() == 0;
    if (!gate_logits) {
        require_f32_cpu(gate, "gate");
    }
    TORCH_CHECK(projected.dim() == 3, "projected must be [B,L,W]");
    TORCH_CHECK(d_state > 0 && ngroups > 0 && nheads > 0 && headdim > 0, "invalid Mamba-3 geometry");
    TORCH_CHECK(nheads % ngroups == 0, "nheads must be divisible by ngroups");
    TORCH_CHECK(num_rope_angles >= 0 && 2 * num_rope_angles <= d_state, "num_rope_angles must fit in d_state");
    require_inner_dense(projected, "projected");

    const int64_t batch = projected.size(0);
    const int64_t length = projected.size(1);
    const int64_t d_inner = nheads * headdim;
    if (!gate_logits) {
        require_inner_dense(gate, "gate");
        TORCH_CHECK(gate.dim() == 3 && gate.size(0) == batch && gate.size(1) == length && gate.size(2) == d_inner,
                    "gate must be [B,L,d_inner]");
    }

    montlok::PrepPlan prep{};
    prep.B = batch;
    prep.L = length;
    prep.W = projected.size(2);
    prep.d_inner = d_inner;
    prep.N = d_state;
    prep.G = ngroups;
    prep.H = nheads;
    prep.D = headdim;
    prep.R = num_rope_angles;
    TORCH_CHECK(prep.W == montlok::prep_expected_width(prep), "projected width ", prep.W,
                " does not match the Mamba-3 layout width ", montlok::prep_expected_width(prep));

    const torch::Tensor b_bias_dense = dense_parameter(b_bias, nheads * d_state, "B_bias");
    const torch::Tensor c_bias_dense = dense_parameter(c_bias, nheads * d_state, "C_bias");
    const torch::Tensor b_norm_dense = dense_parameter(b_norm_w, d_state, "B_norm.weight");
    const torch::Tensor c_norm_dense = dense_parameter(c_norm_w, d_state, "C_norm.weight");
    const torch::Tensor dt_bias_dense = dense_parameter(dt_bias, nheads, "dt_bias");
    const torch::Tensor skip_dense = dense_parameter(skip, nheads, "D");

    const auto options = projected.options();
    auto q = torch::empty({batch, length, nheads, d_state}, options);
    auto k = torch::empty({batch, length, nheads, d_state}, options);
    auto coefficients = torch::empty({3, batch, length, nheads}, options);
    auto angle_term = torch::empty({batch, length, nheads, std::max<int64_t>(num_rope_angles, 1)}, options);
    auto cumulative = torch::empty({batch, nheads, std::max<int64_t>(num_rope_angles, 1), length}, options);
    auto output = torch::empty({batch, length, d_inner}, options);

    prep.proj = projected.data_ptr<float>();
    prep.proj_sb = projected.stride(0);
    prep.proj_st = projected.stride(1);
    prep.b_bias = b_bias_dense.data_ptr<float>();
    prep.c_bias = c_bias_dense.data_ptr<float>();
    prep.b_norm_w = b_norm_dense.data_ptr<float>();
    prep.c_norm_w = c_norm_dense.data_ptr<float>();
    prep.b_eps = static_cast<float>(b_eps);
    prep.c_eps = static_cast<float>(c_eps);
    prep.dt_bias = dt_bias_dense.data_ptr<float>();
    prep.a_floor = static_cast<float>(a_floor);
    prep.q = q.data_ptr<float>();
    prep.k = k.data_ptr<float>();
    prep.alpha = coefficients.data_ptr<float>();
    prep.beta = prep.alpha + batch * length * nheads;
    prep.gamma = prep.beta + batch * length * nheads;
    prep.angle_term = angle_term.data_ptr<float>();
    prep.cum = cumulative.data_ptr<float>();

    const int64_t tokens = batch * length;
    const int64_t prep_scratch = montlok::prep_scratch_floats(prep);
    parallel_items(tokens, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> scratch(static_cast<size_t>(prep_scratch));
        for (int64_t token = begin; token < end; ++token) {
            montlok::prep_token(prep, token / length, token % length, scratch.data());
        }
    }, 1 << 20, q);
    if (num_rope_angles > 0) {
        parallel_items(batch * nheads * num_rope_angles, 1, [&](int64_t begin, int64_t end) {
            for (int64_t chain = begin; chain < end; ++chain) {
                montlok::prep_chain(prep, chain);
            }
        }, 1 << 20, cumulative);
        parallel_items(tokens, 1, [&](int64_t begin, int64_t end) {
            for (int64_t token = begin; token < end; ++token) {
                montlok::prep_rotate(prep, token / length, token % length);
            }
        }, 1 << 20, q);
    }

    montlok::SisoPlan plan{};
    plan.B = batch;
    plan.L = length;
    plan.H = nheads;
    plan.N = d_state;
    plan.D = headdim;
    plan.q = prep.q;
    plan.qs = {length * nheads * d_state, nheads * d_state, d_state};
    plan.k = prep.k;
    plan.ks = plan.qs;
    plan.v = prep.proj + d_inner;  // x lanes of the in_proj row, no copy
    plan.vs = {prep.proj_sb, prep.proj_st, headdim};
    if (gate_logits) {
        plan.gate = prep.proj;  // z lanes of the in_proj row
        plan.gs = {prep.proj_sb, prep.proj_st, headdim};
    } else {
        plan.gate = gate.data_ptr<float>();
        plan.gs = {gate.stride(0), gate.stride(1), headdim};
    }
    plan.gate_logits = gate_logits;
    plan.alpha = prep.alpha;
    plan.beta = prep.beta;
    plan.gamma = prep.gamma;
    plan.skip = skip_dense.data_ptr<float>();
    plan.out = output.data_ptr<float>();
    plan.os = {length * d_inner, d_inner, headdim};
    run_recurrence(plan, output);
    return output;
}

// Complete Mamba-3 SISO mixer in one native call. The two GEMMs dispatch to
// oneDNN on supported CPU builds; the middle recurrence stays in montlok.cpp.
torch::Tensor mamba3_siso_layer(
    const torch::Tensor& inputs,
    const torch::Tensor& in_weight,
    const torch::Tensor& out_weight,
    const torch::Tensor& b_bias,
    const torch::Tensor& c_bias,
    const torch::Tensor& b_norm_w,
    const torch::Tensor& c_norm_w,
    const torch::Tensor& dt_bias,
    const torch::Tensor& skip,
    int64_t d_state,
    int64_t ngroups,
    int64_t nheads,
    int64_t headdim,
    int64_t num_rope_angles,
    double b_eps,
    double c_eps,
    double a_floor) {
    auto projected = linear_dispatch(inputs, in_weight);
    auto gate = torch::empty({0}, inputs.options());
    auto output = mamba3_siso_forward(projected, gate, b_bias, c_bias, b_norm_w, c_norm_w, dt_bias, skip, d_state,
                                      ngroups, nheads, headdim, num_rope_angles, b_eps, c_eps, a_floor);
    return linear_dispatch(output, out_weight);
}

// Manifold-constrained hyper-connection (Model/layers/mhc.py,
// constrain=True). `streams` is [B, L, n, d] contiguous, or the [B, L, d]
// backbone when every stream still equals it (TwoStageCore._expand is then
// never materialised; `n_streams` supplies n).
struct StreamsView {
    int64_t batch;
    int64_t length;
    int64_t n;
    int64_t d;
    int64_t stream_stride;
    int64_t token_stride;
    const float* data;
};

StreamsView streams_view(const torch::Tensor& streams, int64_t n_streams, const char* name) {
    require_f32_cpu(streams, name);
    TORCH_CHECK(streams.is_contiguous(), name, " must be contiguous");
    StreamsView view{};
    if (streams.dim() == 4) {
        view.n = streams.size(2);
        view.d = streams.size(3);
        view.stream_stride = view.d;
        TORCH_CHECK(n_streams <= 0 || n_streams == view.n, name, " has ", view.n, " streams, expected ", n_streams);
    } else {
        TORCH_CHECK(streams.dim() == 3, name, " must be [B,L,n,d] or a broadcast [B,L,d]");
        TORCH_CHECK(n_streams > 0, "n_streams is required for a broadcast [B,L,d] ", name);
        view.n = n_streams;
        view.d = streams.size(2);
        view.stream_stride = 0;
    }
    view.batch = streams.size(0);
    view.length = streams.size(1);
    view.token_stride = view.stream_stride == 0 ? view.d : view.n * view.d;
    TORCH_CHECK(view.n > 0 && view.d > 0, name, " must be non-empty");
    view.data = streams.data_ptr<float>();
    return view;
}

struct MhcPrepareOutputs {
    torch::Tensor agg;
    torch::Tensor pre;
    torch::Tensor post;
    torch::Tensor res;
};

// Validates the hyper-connection parameters, allocates the outputs and fills
// everything of the plan except the stream pointers.
MhcPrepareOutputs mhc_prepare_plan(
    montlok::MhcPlan& plan,
    const StreamsView& view,
    const torch::Tensor& norm_w,
    double eps,
    const torch::Tensor& proj_w,
    const torch::Tensor& pre_bias,
    const torch::Tensor& post_bias,
    const torch::Tensor& res_bias,
    const torch::Tensor& pre_alpha,
    const torch::Tensor& post_alpha,
    const torch::Tensor& res_alpha,
    int64_t sinkhorn_iters,
    const std::optional<torch::Tensor>& agg_norm_w,
    double agg_norm_eps,
    std::vector<torch::Tensor>& keep_alive) {
    const int64_t n = view.n;
    const int64_t d = view.d;
    TORCH_CHECK(sinkhorn_iters >= 0, "sinkhorn_iters must be non-negative");
    const size_t parameter_base = keep_alive.size();
    keep_alive.push_back(dense_parameter(norm_w, d, "dyn_norm.weight"));
    keep_alive.push_back(dense_parameter(pre_bias, n, "pre_bias"));
    keep_alive.push_back(dense_parameter(post_bias, n, "post_bias"));
    keep_alive.push_back(dense_parameter(res_bias, n * n, "res_bias"));
    TORCH_CHECK(proj_w.device().is_cpu() && proj_w.scalar_type() == torch::kFloat64 && proj_w.is_contiguous(),
                "proj_w must be a contiguous float64 CPU tensor");
    TORCH_CHECK(proj_w.dim() == 2 && proj_w.size(0) == montlok::mhc_proj_rows(n) && proj_w.size(1) == d,
                "proj_w must be [2n + n*n, d]");

    auto scalar_value = [](const torch::Tensor& value, const char* name) {
        require_f32_cpu(value, name);
        TORCH_CHECK(value.numel() == 1, name, " must be a scalar");
        return static_cast<double>(value.data_ptr<float>()[0]);
    };

    const auto options = torch::dtype(torch::kFloat32).device(torch::kCPU);
    MhcPrepareOutputs outputs{
        torch::empty({view.batch, view.length, d}, options),
        torch::empty({view.batch, view.length, n}, options),
        torch::empty({view.batch, view.length, n}, options),
        torch::empty({view.batch, view.length, n, n}, options),
    };

    plan.norm_w = keep_alive[parameter_base].data_ptr<float>();
    plan.proj_w = proj_w.data_ptr<double>();
    plan.pre_bias = keep_alive[parameter_base + 1].data_ptr<float>();
    plan.post_bias = keep_alive[parameter_base + 2].data_ptr<float>();
    plan.res_bias = keep_alive[parameter_base + 3].data_ptr<float>();
    plan.pre_alpha = scalar_value(pre_alpha, "pre_alpha");
    plan.post_alpha = scalar_value(post_alpha, "post_alpha");
    plan.res_alpha = scalar_value(res_alpha, "res_alpha");
    // The reference adds the Python float eps to a float32 tensor.
    plan.eps = static_cast<double>(static_cast<float>(eps));
    plan.tokens = view.batch * view.length;
    plan.n = n;
    plan.d = d;
    plan.sinkhorn_iters = static_cast<int>(std::min<int64_t>(sinkhorn_iters, 1 << 20));
    plan.agg = outputs.agg.data_ptr<float>();
    if (agg_norm_w.has_value() && agg_norm_w->defined() && agg_norm_w->numel() > 0) {
        keep_alive.push_back(dense_parameter(*agg_norm_w, d, "agg_norm_w"));
        plan.agg_norm_w = keep_alive.back().data_ptr<float>();
        plan.agg_norm_eps = static_cast<double>(static_cast<float>(agg_norm_eps));
    }
    plan.pre = outputs.pre.data_ptr<float>();
    plan.post = outputs.post.data_ptr<float>();
    plan.res = outputs.res.data_ptr<float>();
    return outputs;
}

// Estimated scalar operations of one token of mhc_prepare_block (used for the
// serial/parallel decision).
int64_t mhc_prepare_work(int64_t n, int64_t d, int64_t sinkhorn_iters) {
    return 2 * montlok::mhc_proj_rows(n) * d + 4 * n * d + 8 * n * n * sinkhorn_iters;
}

// First half of the hyper-connection (everything before the wrapped layer).
// Returns (agg [B, L, d], pre [B, L, n], post [B, L, n], res [B, L, n, n]).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> mhc_prepare(
    const torch::Tensor& streams,
    const torch::Tensor& norm_w,
    double eps,
    const torch::Tensor& proj_w,
    const torch::Tensor& pre_bias,
    const torch::Tensor& post_bias,
    const torch::Tensor& res_bias,
    const torch::Tensor& pre_alpha,
    const torch::Tensor& post_alpha,
    const torch::Tensor& res_alpha,
    int64_t sinkhorn_iters,
    int64_t n_streams,
    const std::optional<torch::Tensor>& agg_norm_w,
    double agg_norm_eps) {
    const StreamsView view = streams_view(streams, n_streams, "streams");
    std::vector<torch::Tensor> keep_alive;
    keep_alive.reserve(5);
    montlok::MhcPlan plan{};
    MhcPrepareOutputs outputs = mhc_prepare_plan(plan, view, norm_w, eps, proj_w, pre_bias, post_bias, res_bias,
                                                 pre_alpha, post_alpha, res_alpha, sinkhorn_iters, agg_norm_w,
                                                 agg_norm_eps, keep_alive);
    plan.streams = view.data;
    plan.stream_stride = view.stream_stride;
    plan.token_stride = view.token_stride;

    const int64_t scratch_doubles = montlok::mhc_scratch_doubles(plan);
    parallel_items(plan.tokens, montlok::kMhcTokenBlock, [&](int64_t begin, int64_t end) {
        std::vector<double> scratch(static_cast<size_t>(scratch_doubles));
        montlok::mhc_prepare_range(plan, begin, end, scratch.data());
    }, mhc_prepare_work(view.n, view.d, plan.sinkhorn_iters), outputs.agg);
    return std::make_tuple(outputs.agg, outputs.pre, outputs.post, outputs.res);
}

montlok::MhcCombinePlan mhc_combine_plan(
    const StreamsView& view,
    const torch::Tensor& res,
    const torch::Tensor& post,
    const torch::Tensor& out) {
    require_f32_cpu(res, "res");
    require_f32_cpu(post, "post");
    require_f32_cpu(out, "out");
    const int64_t batch = view.batch;
    const int64_t length = view.length;
    const int64_t n = view.n;
    const int64_t d = view.d;
    TORCH_CHECK(res.sizes() == torch::IntArrayRef({batch, length, n, n}) && res.is_contiguous(),
                "res must be contiguous [B,L,n,n]");
    TORCH_CHECK(post.sizes() == torch::IntArrayRef({batch, length, n}) && post.is_contiguous(),
                "post must be contiguous [B,L,n]");
    TORCH_CHECK(out.sizes() == torch::IntArrayRef({batch, length, d}) && out.is_contiguous(),
                "out must be contiguous [B,L,d]");
    montlok::MhcCombinePlan plan{};
    plan.streams = view.data;
    plan.stream_stride = view.stream_stride;
    plan.token_stride = view.token_stride;
    plan.res = res.data_ptr<float>();
    plan.post = post.data_ptr<float>();
    plan.out = out.data_ptr<float>();
    plan.n = n;
    plan.d = d;
    return plan;
}

// Second half of the hyper-connection: res @ streams + post (x) out.
//   streams: [B, L, n, d] (or broadcast [B, L, d]); res: [B, L, n, n];
//   post: [B, L, n]; out: [B, L, d]. Returns the new streams [B, L, n, d].
torch::Tensor mhc_combine(
    const torch::Tensor& streams,
    const torch::Tensor& res,
    const torch::Tensor& post,
    const torch::Tensor& out) {
    const StreamsView view = streams_view(streams, res.dim() == 4 ? res.size(-1) : 0, "streams");
    montlok::MhcCombinePlan plan = mhc_combine_plan(view, res, post, out);
    auto new_streams = torch::empty({view.batch, view.length, view.n, view.d}, streams.options());
    plan.new_streams = new_streams.data_ptr<float>();
    parallel_items(view.batch * view.length, 1, [&](int64_t begin, int64_t end) {
        montlok::mhc_combine_range(plan, begin, end);
    }, 2 * view.n * (view.n + 1) * view.d, new_streams);
    return new_streams;
}

// mhc_combine followed by TwoStageCore._collapse: returns only the mean over
// the new streams, [B, L, d]. The streams themselves are never materialised.
torch::Tensor mhc_combine_collapse(
    const torch::Tensor& streams,
    const torch::Tensor& res,
    const torch::Tensor& post,
    const torch::Tensor& out) {
    const StreamsView view = streams_view(streams, res.dim() == 4 ? res.size(-1) : 0, "streams");
    montlok::MhcCombinePlan plan = mhc_combine_plan(view, res, post, out);
    auto hidden = torch::empty({view.batch, view.length, view.d}, streams.options());
    plan.collapsed = hidden.data_ptr<float>();
    const int64_t token_floats = view.n * view.d;
    const int64_t n = view.n;
    const int64_t d = view.d;
    parallel_items(view.batch * view.length, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> scratch(static_cast<size_t>(token_floats));
        for (int64_t token = begin; token < end; ++token) {
            montlok::mhc_combine_token(plan.streams + token * plan.token_stride, plan.stream_stride,
                                       plan.res + token * n * n, plan.post + token * n, plan.out + token * d,
                                       scratch.data(), plan.collapsed + token * d, n, d);
        }
    }, 2 * n * (n + 1) * d, hidden);
    return hidden;
}

// Write step of one hyper-connection fused with the read step of the next one
// (whose parameters follow `out`). Returns
// (new_streams [B, L, n, d], agg [B, L, d], pre [B, L, n], post [B, L, n], res [B, L, n, n]).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> mhc_combine_prepare(
    const torch::Tensor& streams,
    const torch::Tensor& res,
    const torch::Tensor& post,
    const torch::Tensor& out,
    const torch::Tensor& norm_w,
    double eps,
    const torch::Tensor& proj_w,
    const torch::Tensor& pre_bias,
    const torch::Tensor& post_bias,
    const torch::Tensor& res_bias,
    const torch::Tensor& pre_alpha,
    const torch::Tensor& post_alpha,
    const torch::Tensor& res_alpha,
    int64_t sinkhorn_iters,
    const std::optional<torch::Tensor>& agg_norm_w,
    double agg_norm_eps) {
    const StreamsView view = streams_view(streams, res.dim() == 4 ? res.size(-1) : 0, "streams");
    montlok::MhcCombinePlan combine = mhc_combine_plan(view, res, post, out);
    auto new_streams = torch::empty({view.batch, view.length, view.n, view.d}, streams.options());
    combine.new_streams = new_streams.data_ptr<float>();

    std::vector<torch::Tensor> keep_alive;
    keep_alive.reserve(5);
    montlok::MhcPlan prepare{};
    MhcPrepareOutputs outputs = mhc_prepare_plan(prepare, view, norm_w, eps, proj_w, pre_bias, post_bias, res_bias,
                                                 pre_alpha, post_alpha, res_alpha, sinkhorn_iters, agg_norm_w,
                                                 agg_norm_eps, keep_alive);
    prepare.streams = combine.new_streams;
    prepare.stream_stride = view.d;
    prepare.token_stride = view.n * view.d;

    const int64_t scratch_doubles = montlok::mhc_scratch_doubles(prepare);
    parallel_items(prepare.tokens, montlok::kMhcTokenBlock, [&](int64_t begin, int64_t end) {
        std::vector<double> scratch(static_cast<size_t>(scratch_doubles));
        montlok::mhc_combine_prepare_range(combine, prepare, begin, end, scratch.data());
    }, mhc_prepare_work(view.n, view.d, prepare.sinkhorn_iters) + 2 * view.n * (view.n + 1) * view.d,
       outputs.agg);
    return std::make_tuple(new_streams, outputs.agg, outputs.pre, outputs.post, outputs.res);
}

// Internal Stage-2 state transition. Once the broadcast backbone has been
// materialised, every subsequent mHC write can reuse that allocation. The
// SIMD kernel loads all old stream lanes before storing any replacement lane.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> mhc_combine_prepare_inplace(
    torch::Tensor& streams,
    const torch::Tensor& res,
    const torch::Tensor& post,
    const torch::Tensor& out,
    const torch::Tensor& norm_w,
    double eps,
    const torch::Tensor& proj_w,
    const torch::Tensor& pre_bias,
    const torch::Tensor& post_bias,
    const torch::Tensor& res_bias,
    const torch::Tensor& pre_alpha,
    const torch::Tensor& post_alpha,
    const torch::Tensor& res_alpha,
    int64_t sinkhorn_iters,
    const std::optional<torch::Tensor>& agg_norm_w,
    double agg_norm_eps) {
    const StreamsView view = streams_view(streams, res.dim() == 4 ? res.size(-1) : 0, "streams");
    TORCH_CHECK(streams.dim() == 4 && view.stream_stride == view.d,
                "in-place mHC requires materialised contiguous streams");
    TORCH_CHECK(view.n <= 8, "in-place mHC supports at most eight streams");
    montlok::MhcCombinePlan combine = mhc_combine_plan(view, res, post, out);
    combine.new_streams = streams.data_ptr<float>();

    std::vector<torch::Tensor> keep_alive;
    keep_alive.reserve(5);
    montlok::MhcPlan prepare{};
    MhcPrepareOutputs outputs = mhc_prepare_plan(prepare, view, norm_w, eps, proj_w, pre_bias, post_bias, res_bias,
                                                 pre_alpha, post_alpha, res_alpha, sinkhorn_iters, agg_norm_w,
                                                 agg_norm_eps, keep_alive);
    prepare.streams = combine.new_streams;
    prepare.stream_stride = view.d;
    prepare.token_stride = view.n * view.d;

    const int64_t scratch_doubles = montlok::mhc_scratch_doubles(prepare);
    parallel_items(prepare.tokens, montlok::kMhcTokenBlock, [&](int64_t begin, int64_t end) {
        std::vector<double> scratch(static_cast<size_t>(scratch_doubles));
        montlok::mhc_combine_prepare_inplace_range(combine, prepare, begin, end, scratch.data());
    }, mhc_prepare_work(view.n, view.d, prepare.sinkhorn_iters) + 2 * view.n * (view.n + 1) * view.d,
       outputs.agg);
    return std::make_tuple(outputs.agg, outputs.pre, outputs.post, outputs.res);
}

// RMSNorm / GroupedRMSNorm (Model/layers/rmsnorm.py) over the last dimension.
//   x: [..., dim] float32 CPU; weight: [dim]; groups divides dim.
torch::Tensor rms_norm(const torch::Tensor& x, const torch::Tensor& weight, double eps, int64_t groups) {
    require_f32_cpu(x, "x");
    TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
    const int64_t dim = x.size(-1);
    TORCH_CHECK(groups > 0 && dim % groups == 0, "groups must divide the normalised dimension");
    const torch::Tensor weight_dense = dense_parameter(weight, dim, "weight");
    torch::Tensor rows = x.reshape({-1, dim});
    if (rows.stride(1) != 1 && dim != 1) {
        rows = rows.contiguous();
    }
    const int64_t count = rows.size(0);
    auto out = torch::empty({count, dim}, x.options());
    const float* x_ptr = rows.data_ptr<float>();
    const int64_t row_stride = rows.stride(0);
    const float* w_ptr = weight_dense.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    // The reference adds the Python float eps to a float32 tensor.
    const double eps_f = static_cast<double>(static_cast<float>(eps));
    parallel_items(count, 1, [&](int64_t begin, int64_t end) {
        montlok::rms_norm_rows(x_ptr, row_stride, w_ptr, eps_f, dim, groups, begin, end, out_ptr);
    }, 3 * dim, out);
    return out.view(x.sizes());
}

// SwiGLU gate (Model/layers/swiglu.py): x [..., 2 * hidden] = w_in output as
// [gate | up]; returns silu(gate) * up [..., hidden] for w_down.
torch::Tensor swiglu(const torch::Tensor& x) {
    require_f32_cpu(x, "x");
    TORCH_CHECK(x.dim() >= 1 && x.size(-1) % 2 == 0, "x must end in an even dimension [gate | up]");
    const int64_t width = x.size(-1);
    const int64_t hidden = width / 2;
    torch::Tensor rows = x.reshape({-1, width});
    if (rows.stride(1) != 1 && width != 1) {
        rows = rows.contiguous();
    }
    const int64_t count = rows.size(0);
    auto out = torch::empty({count, hidden}, x.options());
    const float* x_ptr = rows.data_ptr<float>();
    const int64_t row_stride = rows.stride(0);
    float* out_ptr = out.data_ptr<float>();
    parallel_items(count, 1, [&](int64_t begin, int64_t end) {
        montlok::swiglu_rows(x_ptr, row_stride, hidden, begin, end, out_ptr);
    }, 24 * hidden, out);
    std::vector<int64_t> shape(x.sizes().begin(), x.sizes().end());
    shape.back() = hidden;
    return out.view(shape);
}

// Complete SwiGLU MLP in one native call: input projection, fused gate and
// output projection. oneDNN owns the GEMM primitives when available.
torch::Tensor swiglu_mlp(
    const torch::Tensor& x,
    const torch::Tensor& in_weight,
    const std::optional<torch::Tensor>& in_bias,
    const torch::Tensor& out_weight,
    const std::optional<torch::Tensor>& out_bias) {
    auto projected = linear_dispatch(x, in_weight, in_bias);
    auto activated = swiglu(projected);
    return linear_dispatch(activated, out_weight, out_bias);
}

std::tuple<torch::Tensor, torch::Tensor> add_rms_norm(
    const torch::Tensor& left,
    const torch::Tensor& right,
    const torch::Tensor& weight,
    double eps,
    int64_t groups) {
    require_f32_cpu(left, "left");
    require_f32_cpu(right, "right");
    TORCH_CHECK(left.sizes() == right.sizes() && left.dim() >= 1, "residual operands must have the same shape");
    const int64_t dim = left.size(-1);
    TORCH_CHECK(groups > 0 && dim % groups == 0, "groups must divide the normalized dimension");
    auto left_rows = left.reshape({-1, dim}).contiguous();
    auto right_rows = right.reshape({-1, dim}).contiguous();
    const auto weight_dense = dense_parameter(weight, dim, "weight");
    const int64_t count = left_rows.size(0);
    auto residual = torch::empty({count, dim}, left.options());
    auto normalized = torch::empty_like(residual);
    const float* left_ptr = left_rows.data_ptr<float>();
    const float* right_ptr = right_rows.data_ptr<float>();
    const float* weight_ptr = weight_dense.data_ptr<float>();
    float* residual_ptr = residual.data_ptr<float>();
    float* normalized_ptr = normalized.data_ptr<float>();
    const double eps_f = static_cast<double>(static_cast<float>(eps));
    parallel_items(count, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const float* a = left_ptr + row * dim;
            const float* b = right_ptr + row * dim;
            float* out = residual_ptr + row * dim;
            int64_t k = 0;
#ifdef MONTLOK_VECTOR_EXT
            for (; k + 8 <= dim; k += 8) {
                montlok::store8(out + k, montlok::load8(a + k) + montlok::load8(b + k));
            }
#endif
            for (; k < dim; ++k) {
                out[k] = a[k] + b[k];
            }
        }
        montlok::rms_norm_rows(residual_ptr, dim, weight_ptr, eps_f, dim, groups, begin, end, normalized_ptr);
    }, 4 * dim, normalized);
    return std::make_tuple(residual.view(left.sizes()), normalized.view(left.sizes()));
}

// Complete Stage-1 stack in one native call. Each tensor pack contains:
// outer norm, Mamba in/out, B/C bias+norm, dt/D, FFN norm and FFN in/out.
torch::Tensor stage1_forward(
    const torch::Tensor& inputs,
    const std::vector<std::vector<torch::Tensor>>& layers,
    double mamba_norm_eps,
    int64_t mamba_norm_groups,
    double ffn_norm_eps,
    int64_t d_state,
    int64_t ngroups,
    int64_t nheads,
    int64_t headdim,
    int64_t num_rope_angles,
    double b_eps,
    double c_eps,
    double a_floor) {
    TORCH_CHECK(!layers.empty(), "stage1 layers must not be empty");
    for (const auto& layer : layers) {
        TORCH_CHECK(layer.size() == 14, "each stage1 tensor pack must contain 14 tensors");
    }
    torch::Tensor hidden = inputs;
    auto mamba_input = rms_norm(hidden, layers[0][0], mamba_norm_eps, mamba_norm_groups);
    for (size_t index = 0; index < layers.size(); ++index) {
        const auto& layer = layers[index];
        auto mamba_output = mamba3_siso_layer(mamba_input, layer[1], layer[2], layer[3], layer[4], layer[5],
                                              layer[6], layer[7], layer[8], d_state, ngroups, nheads, headdim,
                                              num_rope_angles, b_eps, c_eps, a_floor);
        auto mamba_residual = add_rms_norm(hidden, mamba_output, layer[9], ffn_norm_eps, 1);
        hidden = std::get<0>(mamba_residual);
        const std::optional<torch::Tensor> in_bias =
            layer[11].numel() ? std::optional<torch::Tensor>(layer[11]) : std::nullopt;
        const std::optional<torch::Tensor> out_bias =
            layer[13].numel() ? std::optional<torch::Tensor>(layer[13]) : std::nullopt;
        auto ffn_output = swiglu_mlp(std::get<1>(mamba_residual), layer[10], in_bias, layer[12], out_bias);
        if (index + 1 == layers.size()) {
            hidden = hidden + ffn_output;
        } else {
            auto ffn_residual = add_rms_norm(hidden, ffn_output, layers[index + 1][0], mamba_norm_eps,
                                             mamba_norm_groups);
            hidden = std::get<0>(ffn_residual);
            mamba_input = std::get<1>(ffn_residual);
        }
    }
    return hidden;
}

// torch.nn.LayerNorm over the last dimension.
//   x: [..., dim] float32 CPU; weight: [dim]; bias: [dim] or an empty tensor.
torch::Tensor layer_norm(const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& bias, double eps) {
    require_f32_cpu(x, "x");
    TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
    const int64_t dim = x.size(-1);
    const torch::Tensor weight_dense = dense_parameter(weight, dim, "weight");
    const bool has_bias = bias.defined() && bias.numel() > 0;
    const torch::Tensor bias_dense = has_bias ? dense_parameter(bias, dim, "bias") : torch::Tensor();
    torch::Tensor rows = x.reshape({-1, dim});
    if (rows.stride(1) != 1 && dim != 1) {
        rows = rows.contiguous();
    }
    const int64_t count = rows.size(0);
    auto out = torch::empty({count, dim}, x.options());
    const float* x_ptr = rows.data_ptr<float>();
    const int64_t row_stride = rows.stride(0);
    const float* w_ptr = weight_dense.data_ptr<float>();
    const float* b_ptr = has_bias ? bias_dense.data_ptr<float>() : nullptr;
    float* out_ptr = out.data_ptr<float>();
    // The reference casts eps to the float32 accumulation type.
    const double eps_f = static_cast<double>(static_cast<float>(eps));
    parallel_items(count, 1, [&](int64_t begin, int64_t end) {
        montlok::layer_norm_rows(x_ptr, row_stride, w_ptr, b_ptr, eps_f, dim, begin, end, out_ptr);
    }, 6 * dim, out);
    return out.view(x.sizes());
}

// MLA rotary step (Model/layers/mla.py + rope.py apply_rope):
//   q:      [B, L, H, head_dim], rotated in place on its last rope_dim lanes
//   kv:     [B, L, H, nope_dim + head_dim] contiguous kv_up output (k_nope | v)
//   k_rope: [B, L, rope_dim]; cosine, sine: [L, rope_dim] contiguous
// q and k_rope may be column slices of one fused projection output (dense
// within a token, any token stride). q may hold only the trailing Lq <= L
// positions of each sequence ([B, Lq, H, head_dim]); they are rotated with the
// tables of positions [L - Lq, L). Returns k = cat(k_nope, rope(k_rope)) as
// [B, L, H, head_dim].
torch::Tensor mla_rope_qk(
    torch::Tensor q,
    const torch::Tensor& kv,
    const torch::Tensor& k_rope,
    const torch::Tensor& cosine,
    const torch::Tensor& sine,
    int64_t nope_dim) {
    require_f32_cpu(q, "q");
    require_f32_cpu(kv, "kv");
    require_f32_cpu(k_rope, "k_rope");
    require_f32_cpu(cosine, "cos");
    require_f32_cpu(sine, "sin");
    TORCH_CHECK(q.dim() == 4, "q must be [B,Lq,H,head_dim]");
    TORCH_CHECK(kv.dim() == 4 && kv.is_contiguous(), "kv must be contiguous [B,L,H,nope_dim+head_dim]");
    const int64_t batch = q.size(0);
    const int64_t q_length = q.size(1);
    const int64_t length = kv.size(1);
    const int64_t heads = q.size(2);
    const int64_t head_dim = q.size(3);
    const int64_t rope_dim = head_dim - nope_dim;
    TORCH_CHECK(nope_dim >= 0 && rope_dim > 0 && rope_dim % 2 == 0, "head_dim - nope_dim must be a positive even rope_dim");
    TORCH_CHECK(q_length >= 1 && q_length <= length, "q must cover between 1 and L trailing positions");
    // Tokens of a sequence are q.stride(1) apart and sequences q_length tokens
    // apart, so a [B, Lq, ...] column slice of a wider [B, Lq, W] tensor works.
    TORCH_CHECK((q.stride(3) == 1 || head_dim == 1) && (q.stride(2) == head_dim || heads == 1) &&
                    (q.stride(0) == q_length * q.stride(1) || batch == 1),
                "q must be dense within a token with a uniform token stride");
    TORCH_CHECK(kv.sizes() == torch::IntArrayRef({batch, length, heads, nope_dim + head_dim}),
                "kv must be [B,L,H,nope_dim+head_dim]");
    TORCH_CHECK(k_rope.sizes() == torch::IntArrayRef({batch, length, rope_dim}), "k_rope must be [B,L,rope_dim]");
    TORCH_CHECK((k_rope.stride(2) == 1 || rope_dim == 1) && (k_rope.stride(0) == length * k_rope.stride(1) || batch == 1),
                "k_rope must be dense within a token with a uniform token stride");
    TORCH_CHECK(cosine.sizes() == torch::IntArrayRef({length, rope_dim}) && cosine.is_contiguous(),
                "cos must be contiguous [L,rope_dim]");
    TORCH_CHECK(sine.sizes() == cosine.sizes() && sine.is_contiguous(), "sin must be contiguous [L,rope_dim]");

    auto k = torch::empty({batch, length, heads, head_dim}, q.options());
    montlok::MlaRopePlan plan{};
    plan.q = q.data_ptr<float>();
    plan.q_token_stride = q.stride(1);
    plan.q_head_stride = head_dim;
    plan.q_length = q_length;
    plan.kv = kv.data_ptr<float>();
    plan.kv_token_stride = heads * (nope_dim + head_dim);
    plan.kv_head_stride = nope_dim + head_dim;
    plan.k_rope = k_rope.data_ptr<float>();
    plan.k_rope_token_stride = k_rope.stride(1);
    plan.cosine = cosine.data_ptr<float>();
    plan.sine = sine.data_ptr<float>();
    plan.k = k.data_ptr<float>();
    plan.length = length;
    plan.heads = heads;
    plan.head_dim = head_dim;
    plan.nope_dim = nope_dim;
    plan.rope_dim = rope_dim;
    parallel_items(batch * length, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> scratch(static_cast<size_t>(rope_dim));
        for (int64_t token = begin; token < end; ++token) {
            montlok::mla_rope_token(plan, token, scratch.data());
        }
    }, heads * (head_dim + 3 * rope_dim), k);
    return k;
}

// Complete MLA inference path for the two serving shapes: every causal query,
// or the final query only. Projection, latent normalization, RoPE, CPU flash
// attention and the output projection remain inside one native call.
torch::Tensor mla_forward(
    const torch::Tensor& x,
    const torch::Tensor& in_proj_weight,
    const torch::Tensor& q_weight,
    const torch::Tensor& kv_norm_weight,
    const torch::Tensor& kv_norm_bias,
    double kv_norm_eps,
    const torch::Tensor& kv_up_weight,
    const torch::Tensor& out_weight,
    const torch::Tensor& cosine,
    const torch::Tensor& sine,
    int64_t n_heads,
    int64_t head_dim,
    int64_t nope_dim,
    int64_t kv_lora_rank,
    int64_t n_last,
    bool causal,
    double scale) {
    require_f32_cpu(x, "x");
    TORCH_CHECK(x.dim() == 3, "x must be [B,L,d_model]");
    const int64_t batch = x.size(0);
    const int64_t length = x.size(1);
    const int64_t d_model = x.size(2);
    TORCH_CHECK(n_heads > 0 && head_dim > 0 && d_model == n_heads * head_dim, "invalid MLA head geometry");
    TORCH_CHECK(nope_dim >= 0 && nope_dim < head_dim && kv_lora_rank > 0, "invalid MLA latent geometry");
    TORCH_CHECK(n_last == 1 || n_last == length, "native MLA supports the final query or the full sequence");

    const bool full_queries = n_last == length;
    auto projected = linear_dispatch(x, in_proj_weight);
    const int64_t q_width = full_queries ? d_model : 0;
    torch::Tensor q;
    if (full_queries) {
        q = projected.narrow(-1, 0, q_width);
    } else {
        TORCH_CHECK(q_weight.numel() == d_model * d_model, "q_weight must be [d_model,d_model] for a tail query");
        q = linear_dispatch(x.narrow(1, length - 1, 1), q_weight);
    }
    q = q.view({batch, n_last, n_heads, head_dim});

    auto kv_latent = projected.narrow(-1, q_width, kv_lora_rank);
    auto k_rope = projected.narrow(-1, q_width + kv_lora_rank, head_dim - nope_dim);
    auto kv_normalized = layer_norm(kv_latent, kv_norm_weight, kv_norm_bias, kv_norm_eps);
    auto kv = linear_dispatch(kv_normalized, kv_up_weight)
                  .view({batch, length, n_heads, nope_dim + head_dim});
    auto k = mla_rope_qk(q, kv, k_rope, cosine, sine, nope_dim);
    auto v = kv.narrow(-1, nope_dim, head_dim);

    const bool is_causal = full_queries && causal;
    auto attended = at::scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                     std::nullopt, 0.0, is_causal, scale, false);
    auto flat = attended.transpose(1, 2).reshape({batch, n_last, d_model});
    return linear_dispatch(flat, out_weight);
}

// Complete recurrent stage-2 tail inference for the deployed one-layer mHC
// geometry. The same layer weights are reused `depth` times; Python binds the
// tensors once per call, while all residual transitions and wrapped operators
// execute inside this native loop.
torch::Tensor stage2_tail_forward(
    const torch::Tensor& backbone,
    int64_t depth,
    int64_t n_streams,
    const std::vector<torch::Tensor>& attn_hc,
    double attn_hc_eps,
    int64_t attn_sinkhorn_iters,
    const torch::Tensor& attn_norm_weight,
    double attn_norm_eps,
    const std::vector<torch::Tensor>& mla,
    double kv_norm_eps,
    int64_t n_heads,
    int64_t head_dim,
    int64_t nope_dim,
    int64_t kv_lora_rank,
    double attn_scale,
    const std::vector<torch::Tensor>& ffn_hc,
    double ffn_hc_eps,
    int64_t ffn_sinkhorn_iters,
    const torch::Tensor& ffn_norm_weight,
    double ffn_norm_eps,
    const std::vector<torch::Tensor>& ffn) {
    TORCH_CHECK(depth > 0, "depth must be positive");
    TORCH_CHECK(n_streams > 0, "n_streams must be positive");
    TORCH_CHECK(attn_hc.size() == 8 && ffn_hc.size() == 8,
                "each mHC tensor pack must contain dyn/proj/biases/alphas");
    TORCH_CHECK(mla.size() == 9, "MLA tensor pack must contain full/tail/q/norm/kv/out/rope tensors");
    TORCH_CHECK(ffn.size() == 4, "FFN tensor pack must contain in/out weights and biases");

    auto prepare = [&](const torch::Tensor& streams, const std::vector<torch::Tensor>& hc, double hc_eps,
                       int64_t sinkhorn_iters, int64_t broadcast_streams, const torch::Tensor& output_norm_weight,
                       double output_norm_eps) {
        return mhc_prepare(streams, hc[0], hc_eps, hc[1], hc[2], hc[3], hc[4], hc[5], hc[6], hc[7],
                           sinkhorn_iters, broadcast_streams, output_norm_weight, output_norm_eps);
    };
    auto combine_prepare = [&](const torch::Tensor& streams, const torch::Tensor& res, const torch::Tensor& post,
                               const torch::Tensor& out, const std::vector<torch::Tensor>& next_hc,
                               double next_eps, int64_t next_iters, const torch::Tensor& output_norm_weight,
                               double output_norm_eps) {
        return mhc_combine_prepare(streams, res, post, out, next_hc[0], next_eps, next_hc[1], next_hc[2],
                                   next_hc[3], next_hc[4], next_hc[5], next_hc[6], next_hc[7], next_iters,
                                   output_norm_weight, output_norm_eps);
    };
    auto combine_prepare_same_storage = [&](torch::Tensor& streams, const torch::Tensor& res,
                                            const torch::Tensor& post, const torch::Tensor& out,
                                            const std::vector<torch::Tensor>& next_hc, double next_eps,
                                            int64_t next_iters, const torch::Tensor& output_norm_weight,
                                            double output_norm_eps) {
        return mhc_combine_prepare_inplace(streams, res, post, out, next_hc[0], next_eps, next_hc[1], next_hc[2],
                                           next_hc[3], next_hc[4], next_hc[5], next_hc[6], next_hc[7], next_iters,
                                           output_norm_weight, output_norm_eps);
    };
    auto run_mla = [&](const torch::Tensor& input, int64_t n_last) {
        const torch::Tensor& projection = n_last == input.size(1) ? mla[0] : mla[1];
        return mla_forward(input, projection, mla[2], mla[3], mla[4], kv_norm_eps, mla[5], mla[6], mla[7], mla[8],
                           n_heads, head_dim, nope_dim, kv_lora_rank, n_last, true, attn_scale);
    };

    auto first = prepare(backbone, attn_hc, attn_hc_eps, attn_sinkhorn_iters, n_streams, attn_norm_weight,
                         attn_norm_eps);
    torch::Tensor streams = backbone;
    torch::Tensor agg = std::get<0>(first);
    torch::Tensor post = std::get<2>(first);
    torch::Tensor res = std::get<3>(first);
    const char* inplace_env = std::getenv("MONTLOK_STAGE2_INPLACE");
    const bool use_inplace = n_streams <= 8 && (inplace_env == nullptr || std::string_view(inplace_env) != "0");

    const std::optional<torch::Tensor> ffn_in_bias = ffn[1].numel() ? std::optional<torch::Tensor>(ffn[1]) : std::nullopt;
    const std::optional<torch::Tensor> ffn_out_bias = ffn[3].numel() ? std::optional<torch::Tensor>(ffn[3]) : std::nullopt;
    for (int64_t step = 0; step < depth; ++step) {
        const bool final_step = step + 1 == depth;
        if (final_step) {
            auto attn_out = run_mla(agg, 1);
            const int64_t last = streams.size(1) - 1;
            auto tail_streams = streams.narrow(1, last, 1).contiguous();
            auto tail_res = res.narrow(1, last, 1).contiguous();
            auto tail_post = post.narrow(1, last, 1).contiguous();
            auto written = mhc_combine(tail_streams, tail_res, tail_post, attn_out);
            auto ffn_prepared = prepare(written, ffn_hc, ffn_hc_eps, ffn_sinkhorn_iters, 0, ffn_norm_weight,
                                        ffn_norm_eps);
            auto ffn_out = swiglu_mlp(std::get<0>(ffn_prepared), ffn[0], ffn_in_bias, ffn[2], ffn_out_bias);
            return mhc_combine_collapse(written, std::get<3>(ffn_prepared), std::get<2>(ffn_prepared), ffn_out);
        }

        auto attn_out = run_mla(agg, agg.size(1));
        torch::Tensor ffn_agg;
        torch::Tensor ffn_post;
        torch::Tensor ffn_res;
        if (use_inplace && streams.dim() == 4) {
            auto ffn_prepared = combine_prepare_same_storage(streams, res, post, attn_out, ffn_hc, ffn_hc_eps,
                                                             ffn_sinkhorn_iters, ffn_norm_weight, ffn_norm_eps);
            ffn_agg = std::get<0>(ffn_prepared);
            ffn_post = std::get<2>(ffn_prepared);
            ffn_res = std::get<3>(ffn_prepared);
        } else {
            auto ffn_prepared = combine_prepare(streams, res, post, attn_out, ffn_hc, ffn_hc_eps,
                                                ffn_sinkhorn_iters, ffn_norm_weight, ffn_norm_eps);
            streams = std::get<0>(ffn_prepared);
            ffn_agg = std::get<1>(ffn_prepared);
            ffn_post = std::get<3>(ffn_prepared);
            ffn_res = std::get<4>(ffn_prepared);
        }
        auto ffn_out = swiglu_mlp(ffn_agg, ffn[0], ffn_in_bias, ffn[2], ffn_out_bias);
        if (use_inplace) {
            auto next_attn = combine_prepare_same_storage(streams, ffn_res, ffn_post, ffn_out, attn_hc,
                                                           attn_hc_eps, attn_sinkhorn_iters, attn_norm_weight,
                                                           attn_norm_eps);
            agg = std::get<0>(next_attn);
            post = std::get<2>(next_attn);
            res = std::get<3>(next_attn);
        } else {
            auto next_attn = combine_prepare(streams, ffn_res, ffn_post, ffn_out, attn_hc, attn_hc_eps,
                                             attn_sinkhorn_iters, attn_norm_weight, attn_norm_eps);
            streams = std::get<0>(next_attn);
            agg = std::get<1>(next_attn);
            post = std::get<3>(next_attn);
            res = std::get<4>(next_attn);
        }
    }
    TORCH_INTERNAL_ASSERT(false, "unreachable stage-2 loop");
    return torch::Tensor();
}

py::dict build_info() {
    py::dict info;
    info["vector_extensions"] =
#ifdef MONTLOK_VECTOR_EXT
        true;
#else
        false;
#endif
    info["avx2"] =
#ifdef __AVX2__
        true;
#else
        false;
#endif
    info["fma"] =
#ifdef __FMA__
        true;
#else
        false;
#endif
    info["avx512f"] =
#ifdef __AVX512F__
        true;
#else
        false;
#endif
    info["x86_denormal_guard"] =
#ifdef MONTLOK_X86
        true;
#else
        false;
#endif
    info["openmp_dispatch"] =
#ifdef _OPENMP
        true;
#else
        false;
#endif
    info["block_width_headdim_64"] = montlok::siso_block_width(64);
    info["compiler"] =
#if defined(__clang__)
        std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
        std::string("gcc ") + __VERSION__;
#else
        std::string("unknown");
#endif
    return info;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("linear", &linear_dispatch, "oneDNN-backed inference linear", py::arg("input"), py::arg("weight"),
               py::arg("bias") = py::none());
    module.def("mamba3_siso_recurrent", &mamba3_siso_recurrent,
               "Mamba-3 SISO recurrent CPU forward over prepared q/k/v/gate");
    module.def("mamba3_siso_forward", &mamba3_siso_forward,
               "Fused Mamba-3 SISO CPU forward from the in_proj output to the out_proj input");
    module.def("mamba3_siso_layer", &mamba3_siso_layer,
               "Complete oneDNN-backed Mamba-3 SISO mixer in one native call");
    module.def("mhc_prepare", &mhc_prepare,
               "mHC coefficients (pre, post, Sinkhorn res) and aggregated input for one residual",
               py::arg("streams"), py::arg("norm_w"), py::arg("eps"), py::arg("proj_w"), py::arg("pre_bias"),
               py::arg("post_bias"), py::arg("res_bias"), py::arg("pre_alpha"), py::arg("post_alpha"),
               py::arg("res_alpha"), py::arg("sinkhorn_iters"), py::arg("n_streams") = 0,
               py::arg("agg_norm_w") = py::none(), py::arg("agg_norm_eps") = 0.0);
    module.def("mhc_combine", &mhc_combine, "mHC stream mixing plus write of the wrapped layer output");
    module.def("mhc_combine_collapse", &mhc_combine_collapse,
               "mHC stream mixing plus write, returning only the mean over the new streams");
    module.def("mhc_combine_prepare", &mhc_combine_prepare,
               "mHC write step of one residual fused with the coefficients of the next residual",
               py::arg("streams"), py::arg("res"), py::arg("post"), py::arg("out"), py::arg("norm_w"),
               py::arg("eps"), py::arg("proj_w"), py::arg("pre_bias"), py::arg("post_bias"), py::arg("res_bias"),
               py::arg("pre_alpha"), py::arg("post_alpha"), py::arg("res_alpha"), py::arg("sinkhorn_iters"),
               py::arg("agg_norm_w") = py::none(), py::arg("agg_norm_eps") = 0.0);
    module.def("rms_norm", &rms_norm, "RMSNorm / GroupedRMSNorm over the last dimension in one op");
    module.def("layer_norm", &layer_norm, "LayerNorm over the last dimension in one op");
    module.def("swiglu", &swiglu, "silu(gate) * up of a [gate | up] tensor in one op");
    module.def("swiglu_mlp", &swiglu_mlp, "Complete oneDNN-backed SwiGLU MLP in one native call",
               py::arg("input"), py::arg("in_weight"), py::arg("in_bias"), py::arg("out_weight"),
               py::arg("out_bias"));
    module.def("stage1_forward", &stage1_forward, "Complete Mamba/SwiGLU Stage-1 stack in one native call");
    module.def("mla_rope_qk", &mla_rope_qk, "MLA rotary embedding of q (in place) and key assembly");
    module.def("mla_forward", &mla_forward, "Complete oneDNN/flash-attention MLA inference path");
    module.def("stage2_tail_forward", &stage2_tail_forward,
               "Complete recurrent one-layer mHC stage-2 tail inference in one native call");
    module.def("build_info", &build_info, "SIMD/compiler details of this build");
}
