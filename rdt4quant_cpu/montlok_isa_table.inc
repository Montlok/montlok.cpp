// Copyright 2026 OpenAI.
// Apache-2.0. Defines the KernelTable accessor of one variant translation unit.
//
// Included by montlok_isa_*.cpp after the target pragma has been popped, so the
// accessor itself is compiled for the baseline target and can be called from
// anywhere; the table merely holds the addresses of the variant's kernels,
// which were compiled for the variant's target. Requires MONTLOK_ISA_NAMESPACE,
// MONTLOK_ISA_NAME, MONTLOK_ISA_LEVEL, MONTLOK_ISA_TABLE and the
// MONTLOK_ISA_FMA / MONTLOK_ISA_AVX2 / MONTLOK_ISA_AVX512 flags.

namespace montlok {

const KernelTable* MONTLOK_ISA_TABLE() noexcept {
    static const KernelTable table = {
        MONTLOK_ISA_NAME,
        MONTLOK_ISA_LEVEL,
        MONTLOK_VEC_BYTES,
#ifdef MONTLOK_VECTOR_EXT
        true,
#else
        false,
#endif
        MONTLOK_ISA_FMA,
        MONTLOK_ISA_AVX2,
        MONTLOK_ISA_AVX512,
        &MONTLOK_ISA_NAMESPACE::siso_run_work_item,
        &MONTLOK_ISA_NAMESPACE::prep_token,
        &MONTLOK_ISA_NAMESPACE::prep_chain,
        &MONTLOK_ISA_NAMESPACE::prep_rotate,
        &MONTLOK_ISA_NAMESPACE::mhc_prepare_range,
        &MONTLOK_ISA_NAMESPACE::mhc_combine_range,
        &MONTLOK_ISA_NAMESPACE::mhc_combine_collapse_range,
        &MONTLOK_ISA_NAMESPACE::mhc_combine_prepare_range,
        &MONTLOK_ISA_NAMESPACE::mhc_combine_prepare_inplace_range,
        &MONTLOK_ISA_NAMESPACE::rms_norm_rows,
        &MONTLOK_ISA_NAMESPACE::add_rms_norm_rows,
        &MONTLOK_ISA_NAMESPACE::swiglu_rows,
        &MONTLOK_ISA_NAMESPACE::layer_norm_rows,
        &MONTLOK_ISA_NAMESPACE::mla_rope_range,
    };
    return &table;
}

}  // namespace montlok
