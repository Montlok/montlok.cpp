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
//   rms_norm               (Grouped)RMSNorm in one op
//   mla_rope_qk            MLA rotary embedding of q (in place) + key assembly
//   build_info             which SIMD path this build compiled to

#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
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

// Runs fn(begin, end) over [0, items) on torch's intra-op thread pool.
//
// at::parallel_for is header-inline: with the OpenMP backend of the pip wheels
// it is only parallel when the extension itself is compiled with -fopenmp,
// and torch.utils.cpp_extension builds without it (adding it would load a
// second OpenMP runtime next to the libgomp bundled with torch). Chunks of
// `grain` items are dispatched through TensorIteratorBase::for_each, which is
// compiled inside libtorch_cpu, so the work lands on the same threads as the
// rest of the model and honours torch.set_num_threads().
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
void parallel_items(int64_t items, int64_t grain, const Fn& fn, int64_t work_per_item = 1 << 20) {
    if (items <= 0) {
        return;
    }
    if (items * work_per_item < serial_work_threshold()) {
        fn(0, items);
        return;
    }
    auto indices = torch::arange(items, torch::dtype(torch::kLong));
    auto iter = at::TensorIteratorConfig().add_output(indices).add_input(indices).build();
    iter.for_each(
        [&](char** data, const int64_t* /*strides*/, int64_t size) {
            const int64_t begin = *reinterpret_cast<const int64_t*>(data[1]);
            fn(begin, begin + size);
        },
        std::max<int64_t>(grain, 1));
}

void run_recurrence(const montlok::SisoPlan& plan) {
    const int64_t items = montlok::siso_work_items(plan);
    const int64_t scratch_floats = montlok::siso_scratch_floats(plan);
    parallel_items(items, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> scratch(static_cast<size_t>(scratch_floats));
        for (int64_t item = begin; item < end; ++item) {
            montlok::siso_run_work_item(plan, item, scratch.data());
        }
    });
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
        });
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
    plan.alpha = alpha;
    plan.beta = beta;
    plan.gamma = gamma;
    plan.skip = skip_dense.data_ptr<float>();
    plan.out = output.data_ptr<float>();
    plan.os = leading_strides(output);
    run_recurrence(plan);
    return output;
}

// Fused Mamba-3 SISO forward between in_proj and out_proj.
//   projected: [B, L, W] in_proj output, W = 2*d_inner + 2*ngroups*d_state + 3*nheads + num_rope_angles
//   gate:      [B, L, d_inner] = silu(projected[..., :d_inner])
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
    require_f32_cpu(gate, "gate");
    TORCH_CHECK(projected.dim() == 3, "projected must be [B,L,W]");
    TORCH_CHECK(d_state > 0 && ngroups > 0 && nheads > 0 && headdim > 0, "invalid Mamba-3 geometry");
    TORCH_CHECK(nheads % ngroups == 0, "nheads must be divisible by ngroups");
    TORCH_CHECK(num_rope_angles >= 0 && 2 * num_rope_angles <= d_state, "num_rope_angles must fit in d_state");
    require_inner_dense(projected, "projected");
    require_inner_dense(gate, "gate");

    const int64_t batch = projected.size(0);
    const int64_t length = projected.size(1);
    const int64_t d_inner = nheads * headdim;
    TORCH_CHECK(gate.dim() == 3 && gate.size(0) == batch && gate.size(1) == length && gate.size(2) == d_inner,
                "gate must be [B,L,d_inner]");

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
    });
    if (num_rope_angles > 0) {
        parallel_items(batch * nheads * num_rope_angles, 1, [&](int64_t begin, int64_t end) {
            for (int64_t chain = begin; chain < end; ++chain) {
                montlok::prep_chain(prep, chain);
            }
        });
        parallel_items(tokens, 1, [&](int64_t begin, int64_t end) {
            for (int64_t token = begin; token < end; ++token) {
                montlok::prep_rotate(prep, token / length, token % length);
            }
        });
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
    plan.gate = gate.data_ptr<float>();
    plan.gs = {gate.stride(0), gate.stride(1), headdim};
    plan.alpha = prep.alpha;
    plan.beta = prep.beta;
    plan.gamma = prep.gamma;
    plan.skip = skip_dense.data_ptr<float>();
    plan.out = output.data_ptr<float>();
    plan.os = {length * d_inner, d_inner, headdim};
    run_recurrence(plan);
    return output;
}

// Manifold-constrained hyper-connection, first half (everything before the
// wrapped layer runs).
//   streams: [B, L, n, d] contiguous; norm_w: [d];
//   proj_w: [2n + n*n, d] float64 = cat(pre_proj, post_proj, res_proj).weight
//   pre_bias, post_bias: [n]; res_bias: [n, n];
//   pre_alpha, post_alpha, res_alpha: 0-dim tensors.
// Returns (agg [B, L, d], pre [B, L, n], post [B, L, n], res [B, L, n, n])
// matching ManifoldHyperConnection.forward in Model/layers/mhc.py with
// constrain=True.
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
    int64_t sinkhorn_iters) {
    require_f32_cpu(streams, "streams");
    TORCH_CHECK(streams.dim() == 4, "streams must be [B,L,n,d]");
    TORCH_CHECK(streams.is_contiguous(), "streams must be contiguous");
    const int64_t batch = streams.size(0);
    const int64_t length = streams.size(1);
    const int64_t n = streams.size(2);
    const int64_t d = streams.size(3);
    TORCH_CHECK(n > 0 && d > 0, "streams must be non-empty");
    TORCH_CHECK(sinkhorn_iters >= 0, "sinkhorn_iters must be non-negative");

    const torch::Tensor norm_dense = dense_parameter(norm_w, d, "dyn_norm.weight");
    const torch::Tensor pre_bias_dense = dense_parameter(pre_bias, n, "pre_bias");
    const torch::Tensor post_bias_dense = dense_parameter(post_bias, n, "post_bias");
    const torch::Tensor res_bias_dense = dense_parameter(res_bias, n * n, "res_bias");
    TORCH_CHECK(proj_w.device().is_cpu() && proj_w.scalar_type() == torch::kFloat64 && proj_w.is_contiguous(),
                "proj_w must be a contiguous float64 CPU tensor");
    TORCH_CHECK(proj_w.dim() == 2 && proj_w.size(0) == montlok::mhc_proj_rows(n) && proj_w.size(1) == d,
                "proj_w must be [2n + n*n, d]");

    auto scalar_value = [](const torch::Tensor& value, const char* name) {
        require_f32_cpu(value, name);
        TORCH_CHECK(value.numel() == 1, name, " must be a scalar");
        return static_cast<double>(value.item<float>());
    };

    const int64_t tokens = batch * length;
    const auto options = streams.options();
    auto agg = torch::empty({batch, length, d}, options);
    auto pre = torch::empty({batch, length, n}, options);
    auto post = torch::empty({batch, length, n}, options);
    auto res = torch::empty({batch, length, n, n}, options);

    montlok::MhcPlan plan{};
    plan.streams = streams.data_ptr<float>();
    plan.norm_w = norm_dense.data_ptr<float>();
    plan.proj_w = proj_w.data_ptr<double>();
    plan.pre_bias = pre_bias_dense.data_ptr<float>();
    plan.post_bias = post_bias_dense.data_ptr<float>();
    plan.res_bias = res_bias_dense.data_ptr<float>();
    plan.pre_alpha = scalar_value(pre_alpha, "pre_alpha");
    plan.post_alpha = scalar_value(post_alpha, "post_alpha");
    plan.res_alpha = scalar_value(res_alpha, "res_alpha");
    // The reference adds the Python float eps to a float32 tensor.
    plan.eps = static_cast<double>(static_cast<float>(eps));
    plan.tokens = tokens;
    plan.n = n;
    plan.d = d;
    plan.sinkhorn_iters = static_cast<int>(std::min<int64_t>(sinkhorn_iters, 1 << 20));
    plan.agg = agg.data_ptr<float>();
    plan.pre = pre.data_ptr<float>();
    plan.post = post.data_ptr<float>();
    plan.res = res.data_ptr<float>();

    const int64_t scratch_doubles = montlok::mhc_scratch_doubles(plan);
    parallel_items(tokens, montlok::kMhcTokenBlock, [&](int64_t begin, int64_t end) {
        std::vector<double> scratch(static_cast<size_t>(scratch_doubles));
        montlok::mhc_prepare_range(plan, begin, end, scratch.data());
    });
    return std::make_tuple(agg, pre, post, res);
}

// Second half of the hyper-connection: res @ streams + post (x) out.
//   streams: [B, L, n, d]; res: [B, L, n, n]; post: [B, L, n]; out: [B, L, d]
// Returns the new streams [B, L, n, d].
torch::Tensor mhc_combine(
    const torch::Tensor& streams,
    const torch::Tensor& res,
    const torch::Tensor& post,
    const torch::Tensor& out) {
    require_f32_cpu(streams, "streams");
    require_f32_cpu(res, "res");
    require_f32_cpu(post, "post");
    require_f32_cpu(out, "out");
    TORCH_CHECK(streams.dim() == 4 && streams.is_contiguous(), "streams must be contiguous [B,L,n,d]");
    const int64_t batch = streams.size(0);
    const int64_t length = streams.size(1);
    const int64_t n = streams.size(2);
    const int64_t d = streams.size(3);
    TORCH_CHECK(res.sizes() == torch::IntArrayRef({batch, length, n, n}) && res.is_contiguous(),
                "res must be contiguous [B,L,n,n]");
    TORCH_CHECK(post.sizes() == torch::IntArrayRef({batch, length, n}) && post.is_contiguous(),
                "post must be contiguous [B,L,n]");
    TORCH_CHECK(out.sizes() == torch::IntArrayRef({batch, length, d}) && out.is_contiguous(),
                "out must be contiguous [B,L,d]");

    auto new_streams = torch::empty_like(streams);
    const float* streams_ptr = streams.data_ptr<float>();
    const float* res_ptr = res.data_ptr<float>();
    const float* post_ptr = post.data_ptr<float>();
    const float* out_ptr = out.data_ptr<float>();
    float* new_ptr = new_streams.data_ptr<float>();
    parallel_items(batch * length, 1, [&](int64_t begin, int64_t end) {
        for (int64_t token = begin; token < end; ++token) {
            montlok::mhc_combine_token(streams_ptr + token * n * d, res_ptr + token * n * n,
                                       post_ptr + token * n, out_ptr + token * d,
                                       new_ptr + token * n * d, n, d);
        }
    });
    return new_streams;
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
    }, 3 * dim);
    return out.view(x.sizes());
}

// MLA rotary step (Model/layers/mla.py + rope.py apply_rope):
//   q:      [B, L, H, head_dim] contiguous, rotated in place on its last rope_dim lanes
//   kv:     [B, L, H, nope_dim + head_dim] contiguous kv_up output (k_nope | v)
//   k_rope: [B, L, rope_dim] contiguous; cosine, sine: [L, rope_dim] contiguous
// Returns k = cat(k_nope, rope(k_rope)) as [B, L, H, head_dim] contiguous.
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
    TORCH_CHECK(q.dim() == 4 && q.is_contiguous(), "q must be contiguous [B,L,H,head_dim]");
    const int64_t batch = q.size(0);
    const int64_t length = q.size(1);
    const int64_t heads = q.size(2);
    const int64_t head_dim = q.size(3);
    const int64_t rope_dim = head_dim - nope_dim;
    TORCH_CHECK(nope_dim >= 0 && rope_dim > 0 && rope_dim % 2 == 0, "head_dim - nope_dim must be a positive even rope_dim");
    TORCH_CHECK(kv.sizes() == torch::IntArrayRef({batch, length, heads, nope_dim + head_dim}) && kv.is_contiguous(),
                "kv must be contiguous [B,L,H,nope_dim+head_dim]");
    TORCH_CHECK(k_rope.sizes() == torch::IntArrayRef({batch, length, rope_dim}) && k_rope.is_contiguous(),
                "k_rope must be contiguous [B,L,rope_dim]");
    TORCH_CHECK(cosine.sizes() == torch::IntArrayRef({length, rope_dim}) && cosine.is_contiguous(),
                "cos must be contiguous [L,rope_dim]");
    TORCH_CHECK(sine.sizes() == cosine.sizes() && sine.is_contiguous(), "sin must be contiguous [L,rope_dim]");

    auto k = torch::empty({batch, length, heads, head_dim}, q.options());
    montlok::MlaRopePlan plan{};
    plan.q = q.data_ptr<float>();
    plan.q_token_stride = heads * head_dim;
    plan.q_head_stride = head_dim;
    plan.kv = kv.data_ptr<float>();
    plan.kv_token_stride = heads * (nope_dim + head_dim);
    plan.kv_head_stride = nope_dim + head_dim;
    plan.k_rope = k_rope.data_ptr<float>();
    plan.k_rope_token_stride = rope_dim;
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
    }, heads * (head_dim + 3 * rope_dim));
    return k;
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
    module.def("mamba3_siso_recurrent", &mamba3_siso_recurrent,
               "Mamba-3 SISO recurrent CPU forward over prepared q/k/v/gate");
    module.def("mamba3_siso_forward", &mamba3_siso_forward,
               "Fused Mamba-3 SISO CPU forward from the in_proj output to the out_proj input");
    module.def("mhc_prepare", &mhc_prepare,
               "mHC coefficients (pre, post, Sinkhorn res) and aggregated input for one residual");
    module.def("mhc_combine", &mhc_combine, "mHC stream mixing plus write of the wrapped layer output");
    module.def("rms_norm", &rms_norm, "RMSNorm / GroupedRMSNorm over the last dimension in one op");
    module.def("mla_rope_qk", &mla_rope_qk, "MLA rotary embedding of q (in place) and key assembly");
    module.def("build_info", &build_info, "SIMD/compiler details of this build");
}
