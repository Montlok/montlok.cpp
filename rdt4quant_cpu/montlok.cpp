// Copyright 2026 OpenAI.
// Apache-2.0 weight-compatible CPU inference kernel for Mamba-3 SISO.
// The recurrence follows the public state-spaces/mamba reference equations.

#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

void require_f32_cpu_contiguous(const torch::Tensor& value, const char* name) {
    TORCH_CHECK(value.device().is_cpu(), name, " must be on CPU");
    TORCH_CHECK(value.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor mamba3_siso_recurrent(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& adt,
    const torch::Tensor& dt,
    const torch::Tensor& trap,
    const torch::Tensor& skip,
    const torch::Tensor& z) {
    require_f32_cpu_contiguous(q, "q");
    require_f32_cpu_contiguous(k, "k");
    require_f32_cpu_contiguous(v, "v");
    require_f32_cpu_contiguous(adt, "adt");
    require_f32_cpu_contiguous(dt, "dt");
    require_f32_cpu_contiguous(trap, "trap");
    require_f32_cpu_contiguous(skip, "skip");
    require_f32_cpu_contiguous(z, "z");
    TORCH_CHECK(q.dim() == 4 && k.sizes() == q.sizes(), "q/k must be [B,L,H,N]");
    TORCH_CHECK(v.dim() == 4 && z.sizes() == v.sizes(), "v/z must be [B,L,H,D]");
    const auto batch = q.size(0);
    const auto length = q.size(1);
    const auto heads = q.size(2);
    const auto state_dim = q.size(3);
    const auto value_dim = v.size(3);
    TORCH_CHECK(v.size(0) == batch && v.size(1) == length && v.size(2) == heads, "v shape mismatch");
    TORCH_CHECK(adt.sizes() == torch::IntArrayRef({batch, heads, length}), "adt must be [B,H,L]");
    TORCH_CHECK(dt.sizes() == adt.sizes() && trap.sizes() == adt.sizes(), "dt/trap shape mismatch");
    TORCH_CHECK(skip.numel() == heads, "skip must have H elements");

    auto output = torch::empty_like(v);
    const float* q_ptr = q.data_ptr<float>();
    const float* k_ptr = k.data_ptr<float>();
    const float* v_ptr = v.data_ptr<float>();
    const float* adt_ptr = adt.data_ptr<float>();
    const float* dt_ptr = dt.data_ptr<float>();
    const float* trap_ptr = trap.data_ptr<float>();
    const float* skip_ptr = skip.data_ptr<float>();
    const float* z_ptr = z.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    at::parallel_for(0, batch * heads, 1, [&](int64_t begin, int64_t end) {
        for (int64_t work = begin; work < end; ++work) {
            const int64_t b = work / heads;
            const int64_t h = work % heads;
            std::vector<float> state(value_dim * state_dim, 0.0f);
            std::vector<float> previous_k(state_dim, 0.0f);
            std::vector<float> previous_v(value_dim, 0.0f);
            for (int64_t t = 0; t < length; ++t) {
                const int64_t scalar_index = (b * heads + h) * length + t;
                const float alpha = std::exp(adt_ptr[scalar_index]);
                const float trap_value = 1.0f / (1.0f + std::exp(-trap_ptr[scalar_index]));
                const float delta = dt_ptr[scalar_index];
                const float beta = (1.0f - trap_value) * delta * alpha;
                const float gamma = trap_value * delta;
                const int64_t q_base = ((b * length + t) * heads + h) * state_dim;
                const int64_t v_base = ((b * length + t) * heads + h) * value_dim;
                for (int64_t d = 0; d < value_dim; ++d) {
                    float result = 0.0f;
                    const float current_v = v_ptr[v_base + d];
                    const int64_t state_base = d * state_dim;
                    for (int64_t n = 0; n < state_dim; ++n) {
                        const float updated = alpha * state[state_base + n]
                            + beta * previous_k[n] * previous_v[d]
                            + gamma * k_ptr[q_base + n] * current_v;
                        state[state_base + n] = updated;
                        result += updated * q_ptr[q_base + n];
                    }
                    result += skip_ptr[h] * current_v;
                    const float gate = z_ptr[v_base + d];
                    result *= gate / (1.0f + std::exp(-gate));
                    out_ptr[v_base + d] = result;
                }
                std::copy(k_ptr + q_base, k_ptr + q_base + state_dim, previous_k.begin());
                std::copy(v_ptr + v_base, v_ptr + v_base + value_dim, previous_v.begin());
            }
        }
    });
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("mamba3_siso_recurrent", &mamba3_siso_recurrent, "Mamba-3 SISO recurrent CPU forward");
}
