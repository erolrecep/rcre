/**
 * @file lifecycle.c
 * @brief Model Lifecycle Engine: Magnitude Pruning, Lossless RCRZ Compression ("RGC1")
 *        & Post-Training Quantization (F16 / BF16 / Q8_0 / Q4_0, "RGQ1")
 *
 * Applies uniformly across all RCRE generative & vision models:
 *   RCRE_ConvAE, RCRE_ConvVAE, RCRE_TinyUNet, RCRE_ViT, RCRE_DiT.
 *
 * Strict ISO C23 compliant, zero external dependencies.
 */

#include "rcre_gen.h"
#include "rcre_io.h"
#include "rcre_llm.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint8_t  magic[4];       /* "RGC1" */
    uint32_t version;        /* 1      */
    uint32_t num_params;
    uint32_t reserved;
    uint64_t total_floats;
    uint64_t compressed_nbytes;
} GenRgc1Header;

typedef struct {
    uint8_t  magic[4];       /* "RGQ1" */
    uint32_t version;        /* 1      */
    uint32_t qtype;          /* RCRE_GenQuantType */
    uint32_t num_params;
    uint64_t total_floats;
    uint64_t padded_floats;
    uint64_t payload_nbytes;
} GenRgq1Header;

static int gen_lc_cmp_f32_asc(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

static size_t gen_lc_count_total(RCRE_Tensor* const* params, size_t num_params) {
    size_t total = 0;
    for (size_t i = 0; i < num_params; ++i) {
        if (!params[i] || params[i]->dtype != RCRE_DTYPE_F32) return 0;
        total += params[i]->numel;
    }
    return total;
}

static void gen_lc_compute_error_stats(
    const float* orig, const float* approx, size_t n,
    float* out_rmse, float* out_max_err, float* out_sqnr_db)
{
    double sig_pow = 0.0;
    double err_pow = 0.0;
    float  max_err = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        double o = (double)orig[i];
        double d = (double)approx[i] - o;
        sig_pow += o * o;
        err_pow += d * d;
        float ad = fabsf(approx[i] - orig[i]);
        if (ad > max_err) max_err = ad;
    }
    *out_rmse    = (n > 0) ? (float)sqrt(err_pow / (double)n) : 0.0f;
    *out_max_err = max_err;
    if (err_pow <= 1e-20) {
        *out_sqnr_db = 99.0f;
    } else if (sig_pow <= 1e-20) {
        *out_sqnr_db = 0.0f;
    } else {
        *out_sqnr_db = (float)(10.0 * log10(sig_pow / err_pow));
    }
}

int rcre_gen_prune_magnitude(
    RCRE_Tensor* const* params,
    size_t num_params,
    float target_sparsity,
    bool skip_1d_tensors,
    RCRE_Arena* scratch,
    RCRE_GenModelStats* out_stats)
{
    if (!params || num_params == 0 || !scratch ||
        !(target_sparsity >= 0.0f && target_sparsity < 1.0f)) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t total_all = gen_lc_count_total(params, num_params);
    if (total_all == 0) return RCRE_ERR_INVALID_ARG;

    size_t prunable = 0;
    for (size_t p = 0; p < num_params; ++p) {
        if (skip_1d_tensors && params[p]->ndim <= 1) continue;
        prunable += params[p]->numel;
    }

    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(scratch);
    float* mags = nullptr;
    float threshold = 0.0f;

    if (prunable > 0 && target_sparsity > 0.0f) {
        mags = (float*)rcre_arena_alloc(scratch, prunable * sizeof(float), 64);
        if (!mags) {
            rcre_arena_temp_end(tmp);
            return RCRE_ERR_OUT_OF_MEMORY;
        }
        size_t idx = 0;
        for (size_t p = 0; p < num_params; ++p) {
            if (skip_1d_tensors && params[p]->ndim <= 1) continue;
            const float* d = rcre_tensor_f32_data(params[p]);
            for (size_t i = 0; i < params[p]->numel; ++i) {
                mags[idx++] = fabsf(d[i]);
            }
        }
        qsort(mags, prunable, sizeof(float), gen_lc_cmp_f32_asc);
        size_t cut = (size_t)((double)target_sparsity * (double)prunable);
        if (cut >= prunable) cut = prunable - 1U;
        threshold = mags[cut];
    }

    double sig_pow = 0.0;
    double err_pow = 0.0;
    float  max_err = 0.0f;
    size_t zeros   = 0;

    for (size_t p = 0; p < num_params; ++p) {
        float* d = rcre_tensor_f32_data(params[p]);
        bool can_prune = !(skip_1d_tensors && params[p]->ndim <= 1) && (target_sparsity > 0.0f);
        for (size_t i = 0; i < params[p]->numel; ++i) {
            float v = d[i];
            sig_pow += (double)v * (double)v;
            if (can_prune && fabsf(v) <= threshold) {
                err_pow += (double)v * (double)v;
                if (fabsf(v) > max_err) max_err = fabsf(v);
                d[i] = 0.0f;
            }
            if (d[i] == 0.0f) zeros++;
        }
    }

    rcre_arena_temp_end(tmp);

    if (out_stats) {
        memset(out_stats, 0, sizeof(*out_stats));
        out_stats->total_params      = total_all;
        out_stats->zero_params       = zeros;
        out_stats->sparsity          = (float)zeros / (float)total_all;
        out_stats->raw_bytes         = total_all * sizeof(float);
        out_stats->artifact_bytes    = out_stats->raw_bytes;
        out_stats->compression_ratio = 1.0f;
        out_stats->rmse              = (float)sqrt(err_pow / (double)total_all);
        out_stats->max_abs_err       = max_err;
        out_stats->sqnr_db           = (err_pow <= 1e-20)
                                           ? 99.0f
                                           : (float)(10.0 * log10(fmax(sig_pow, 1e-20) / err_pow));
    }
    return RCRE_OK;
}

int rcre_gen_save_compressed(
    RCRE_Tensor* const* params,
    size_t num_params,
    const char* filepath,
    RCRE_Arena* scratch,
    RCRE_GenModelStats* out_stats)
{
    if (!params || num_params == 0 || !filepath || !scratch) return RCRE_ERR_INVALID_ARG;
    size_t total = gen_lc_count_total(params, num_params);
    if (total == 0) return RCRE_ERR_INVALID_ARG;

    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(scratch);
    size_t flat_shape[1] = { total };
    RCRE_Tensor flat = rcre_tensor_create(flat_shape, 1, RCRE_DTYPE_F32, RCRE_LAYOUT_TABULAR, scratch);
    uint64_t* numels = (uint64_t*)rcre_arena_alloc(scratch, num_params * sizeof(uint64_t), alignof(uint64_t));
    if (!flat.data || !numels) {
        rcre_arena_temp_end(tmp);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    float* fd = (float*)flat.data;
    size_t pos = 0;
    size_t zeros = 0;
    for (size_t p = 0; p < num_params; ++p) {
        numels[p] = (uint64_t)params[p]->numel;
        const float* pd = rcre_tensor_f32_data(params[p]);
        memcpy(fd + pos, pd, params[p]->numel * sizeof(float));
        for (size_t i = 0; i < params[p]->numel; ++i) {
            if (pd[i] == 0.0f) zeros++;
        }
        pos += params[p]->numel;
    }

    RCRE_Slice comp = {0};
    int rc = rcre_io_compress_tensor(&flat, scratch, &comp);
    if (rc != RCRE_OK) {
        rcre_arena_temp_end(tmp);
        return rc;
    }

    GenRgc1Header hdr;
    hdr.magic[0]          = 'R';
    hdr.magic[1]          = 'G';
    hdr.magic[2]          = 'C';
    hdr.magic[3]          = '1';
    hdr.version           = 1U;
    hdr.num_params        = (uint32_t)num_params;
    hdr.reserved          = 0U;
    hdr.total_floats      = (uint64_t)total;
    hdr.compressed_nbytes = (uint64_t)comp.len;

    FILE* fp = fopen(filepath, "wb");
    if (!fp) {
        rcre_arena_temp_end(tmp);
        return RCRE_ERR_IO;
    }

    bool ok = (fwrite(&hdr, sizeof(hdr), 1, fp) == 1) &&
              (fwrite(numels, sizeof(uint64_t), num_params, fp) == num_params) &&
              (fwrite(comp.data, 1, comp.len, fp) == comp.len);
    fclose(fp);
    rcre_arena_temp_end(tmp);

    if (!ok) return RCRE_ERR_IO;

    if (out_stats) {
        size_t file_bytes = sizeof(hdr) + num_params * sizeof(uint64_t) + comp.len;
        memset(out_stats, 0, sizeof(*out_stats));
        out_stats->total_params      = total;
        out_stats->zero_params       = zeros;
        out_stats->sparsity          = (float)zeros / (float)total;
        out_stats->raw_bytes         = total * sizeof(float);
        out_stats->artifact_bytes    = file_bytes;
        out_stats->compression_ratio = (float)out_stats->raw_bytes / (float)file_bytes;
        out_stats->rmse              = 0.0f;
        out_stats->max_abs_err       = 0.0f;
        out_stats->sqnr_db           = 99.0f; /* Lossless 0-ULP */
    }
    return RCRE_OK;
}

int rcre_gen_load_compressed(
    RCRE_Tensor* const* params,
    size_t num_params,
    const char* filepath,
    RCRE_Arena* scratch)
{
    if (!params || num_params == 0 || !filepath || !scratch) return RCRE_ERR_INVALID_ARG;
    size_t total = gen_lc_count_total(params, num_params);
    if (total == 0) return RCRE_ERR_INVALID_ARG;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return RCRE_ERR_IO;

    GenRgc1Header hdr;
    if (fread(&hdr, sizeof(hdr), 1, fp) != 1 ||
        memcmp(hdr.magic, "RGC1", 4) != 0 ||
        hdr.version != 1U ||
        hdr.num_params != (uint32_t)num_params ||
        hdr.total_floats != (uint64_t)total ||
        hdr.compressed_nbytes == 0) {
        fclose(fp);
        return RCRE_ERR_INVALID_ARG;
    }

    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(scratch);
    uint64_t* numels = (uint64_t*)rcre_arena_alloc(scratch, num_params * sizeof(uint64_t), alignof(uint64_t));
    uint8_t*  cbuf   = (uint8_t*)rcre_arena_alloc(scratch, (size_t)hdr.compressed_nbytes, 64);
    if (!numels || !cbuf) {
        fclose(fp);
        rcre_arena_temp_end(tmp);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    bool ok = (fread(numels, sizeof(uint64_t), num_params, fp) == num_params) &&
              (fread(cbuf, 1, (size_t)hdr.compressed_nbytes, fp) == (size_t)hdr.compressed_nbytes);
    fclose(fp);
    if (!ok) {
        rcre_arena_temp_end(tmp);
        return RCRE_ERR_IO;
    }

    for (size_t p = 0; p < num_params; ++p) {
        if (numels[p] != (uint64_t)params[p]->numel) {
            rcre_arena_temp_end(tmp);
            return RCRE_ERR_SHAPE_MISMATCH;
        }
    }

    RCRE_Slice frame = rcre_slice_from_parts(cbuf, (size_t)hdr.compressed_nbytes);
    RCRE_Tensor flat = {0};
    int rc = rcre_io_decompress_tensor(frame, scratch, &flat);
    if (rc != RCRE_OK || flat.numel != total || flat.dtype != RCRE_DTYPE_F32) {
        rcre_arena_temp_end(tmp);
        return (rc != RCRE_OK) ? rc : RCRE_ERR_SHAPE_MISMATCH;
    }

    const float* fd = (const float*)flat.data;
    size_t pos = 0;
    for (size_t p = 0; p < num_params; ++p) {
        memcpy(params[p]->data, fd + pos, params[p]->numel * sizeof(float));
        pos += params[p]->numel;
    }

    rcre_arena_temp_end(tmp);
    return RCRE_OK;
}

static size_t gen_lc_quant_payload_bytes(RCRE_GenQuantType qtype, size_t padded_floats) {
    switch (qtype) {
        case RCRE_GEN_QUANT_F16:  return padded_floats * sizeof(uint16_t);
        case RCRE_GEN_QUANT_BF16: return padded_floats * sizeof(uint16_t);
        case RCRE_GEN_QUANT_Q8_0: return (padded_floats / RCRE_QK8_0) * sizeof(RCRE_BlockQ8_0);
        case RCRE_GEN_QUANT_Q4_0: return (padded_floats / RCRE_QK4_0) * sizeof(RCRE_BlockQ4_0);
        default: return 0;
    }
}

int rcre_gen_save_quantized(
    RCRE_Tensor* const* params,
    size_t num_params,
    RCRE_GenQuantType qtype,
    const char* filepath,
    RCRE_Arena* scratch,
    RCRE_GenModelStats* out_stats)
{
    if (!params || num_params == 0 || !filepath || !scratch) return RCRE_ERR_INVALID_ARG;
    if (qtype != RCRE_GEN_QUANT_F16 && qtype != RCRE_GEN_QUANT_BF16 &&
        qtype != RCRE_GEN_QUANT_Q8_0 && qtype != RCRE_GEN_QUANT_Q4_0) {
        return RCRE_ERR_INVALID_ARG;
    }

    size_t total = gen_lc_count_total(params, num_params);
    if (total == 0) return RCRE_ERR_INVALID_ARG;

    /* Pad to block size 32 for Q8_0 / Q4_0 */
    size_t padded = (qtype == RCRE_GEN_QUANT_Q8_0 || qtype == RCRE_GEN_QUANT_Q4_0)
                        ? ((total + 31U) & ~((size_t)31U))
                        : total;
    size_t payload_bytes = gen_lc_quant_payload_bytes(qtype, padded);

    RCRE_ArenaTemp tmp = rcre_arena_temp_begin(scratch);
    float*    flat    = (float*)rcre_arena_alloc(scratch, padded * sizeof(float), 64);
    uint8_t*  payload = (uint8_t*)rcre_arena_alloc(scratch, payload_bytes, 64);
    uint64_t* numels  = (uint64_t*)rcre_arena_alloc(scratch, num_params * sizeof(uint64_t), alignof(uint64_t));
    if (!flat || !payload || !numels) {
        rcre_arena_temp_end(tmp);
        return RCRE_ERR_OUT_OF_MEMORY;
    }
    memset(flat, 0, padded * sizeof(float));

    size_t pos = 0;
    size_t zeros = 0;
    for (size_t p = 0; p < num_params; ++p) {
        numels[p] = (uint64_t)params[p]->numel;
        const float* pd = rcre_tensor_f32_data(params[p]);
        memcpy(flat + pos, pd, params[p]->numel * sizeof(float));
        for (size_t i = 0; i < params[p]->numel; ++i) {
            if (pd[i] == 0.0f) zeros++;
        }
        pos += params[p]->numel;
    }

    if (qtype == RCRE_GEN_QUANT_F16) {
        rcre_f32_to_f16(flat, payload, padded);
    } else if (qtype == RCRE_GEN_QUANT_BF16) {
        rcre_f32_to_bf16(flat, payload, padded);
    } else if (qtype == RCRE_GEN_QUANT_Q8_0) {
        rcre_quantize_q8_0(flat, payload, padded);
    } else {
        rcre_quantize_q4_0(flat, payload, padded);
    }

    GenRgq1Header hdr;
    hdr.magic[0]       = 'R';
    hdr.magic[1]       = 'G';
    hdr.magic[2]       = 'Q';
    hdr.magic[3]       = '1';
    hdr.version        = 1U;
    hdr.qtype          = (uint32_t)qtype;
    hdr.num_params     = (uint32_t)num_params;
    hdr.total_floats   = (uint64_t)total;
    hdr.padded_floats  = (uint64_t)padded;
    hdr.payload_nbytes = (uint64_t)payload_bytes;

    FILE* fp = fopen(filepath, "wb");
    if (!fp) {
        rcre_arena_temp_end(tmp);
        return RCRE_ERR_IO;
    }
    bool ok = (fwrite(&hdr, sizeof(hdr), 1, fp) == 1) &&
              (fwrite(numels, sizeof(uint64_t), num_params, fp) == num_params) &&
              (fwrite(payload, 1, payload_bytes, fp) == payload_bytes);
    fclose(fp);

    if (!ok) {
        rcre_arena_temp_end(tmp);
        return RCRE_ERR_IO;
    }

    if (out_stats) {
        float* deq = (float*)rcre_arena_alloc(scratch, padded * sizeof(float), 64);
        memset(out_stats, 0, sizeof(*out_stats));
        size_t file_bytes = sizeof(hdr) + num_params * sizeof(uint64_t) + payload_bytes;
        out_stats->total_params      = total;
        out_stats->zero_params       = zeros;
        out_stats->sparsity          = (float)zeros / (float)total;
        out_stats->raw_bytes         = total * sizeof(float);
        out_stats->artifact_bytes    = file_bytes;
        out_stats->compression_ratio = (float)out_stats->raw_bytes / (float)file_bytes;

        if (deq) {
            if (qtype == RCRE_GEN_QUANT_F16)       rcre_f16_to_f32(payload, deq, padded);
            else if (qtype == RCRE_GEN_QUANT_BF16) rcre_bf16_to_f32(payload, deq, padded);
            else if (qtype == RCRE_GEN_QUANT_Q8_0) rcre_dequantize_q8_0(payload, deq, padded);
            else                                   rcre_dequantize_q4_0(payload, deq, padded);
            gen_lc_compute_error_stats(flat, deq, total,
                                       &out_stats->rmse, &out_stats->max_abs_err, &out_stats->sqnr_db);
        }
    }

    rcre_arena_temp_end(tmp);
    return RCRE_OK;
}

int rcre_gen_load_quantized(
    RCRE_Tensor* const* params,
    size_t num_params,
    const char* filepath,
    RCRE_GenQuantType* out_qtype)
{
    if (!params || num_params == 0 || !filepath) return RCRE_ERR_INVALID_ARG;
    size_t total = gen_lc_count_total(params, num_params);
    if (total == 0) return RCRE_ERR_INVALID_ARG;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return RCRE_ERR_IO;

    GenRgq1Header hdr;
    if (fread(&hdr, sizeof(hdr), 1, fp) != 1 ||
        memcmp(hdr.magic, "RGQ1", 4) != 0 ||
        hdr.version != 1U ||
        hdr.num_params != (uint32_t)num_params ||
        hdr.total_floats != (uint64_t)total ||
        hdr.padded_floats < (uint64_t)total) {
        fclose(fp);
        return RCRE_ERR_INVALID_ARG;
    }

    RCRE_GenQuantType qtype = (RCRE_GenQuantType)hdr.qtype;
    size_t expected_bytes = gen_lc_quant_payload_bytes(qtype, (size_t)hdr.padded_floats);
    if (expected_bytes == 0 || hdr.payload_nbytes != (uint64_t)expected_bytes) {
        fclose(fp);
        return RCRE_ERR_INVALID_ARG;
    }

    size_t need = num_params * sizeof(uint64_t) + expected_bytes + (size_t)hdr.padded_floats * sizeof(float) + 4096U;
    RCRE_Arena* tmp_arena = rcre_arena_create(need);
    if (!tmp_arena) {
        fclose(fp);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    uint64_t* numels  = (uint64_t*)rcre_arena_alloc(tmp_arena, num_params * sizeof(uint64_t), alignof(uint64_t));
    uint8_t*  payload = (uint8_t*)rcre_arena_alloc(tmp_arena, expected_bytes, 64);
    float*    deq     = (float*)rcre_arena_alloc(tmp_arena, (size_t)hdr.padded_floats * sizeof(float), 64);
    if (!numels || !payload || !deq) {
        fclose(fp);
        rcre_arena_destroy(tmp_arena);
        return RCRE_ERR_OUT_OF_MEMORY;
    }

    bool ok = (fread(numels, sizeof(uint64_t), num_params, fp) == num_params) &&
              (fread(payload, 1, expected_bytes, fp) == expected_bytes);
    fclose(fp);
    if (!ok) {
        rcre_arena_destroy(tmp_arena);
        return RCRE_ERR_IO;
    }

    for (size_t p = 0; p < num_params; ++p) {
        if (numels[p] != (uint64_t)params[p]->numel) {
            rcre_arena_destroy(tmp_arena);
            return RCRE_ERR_SHAPE_MISMATCH;
        }
    }

    if (qtype == RCRE_GEN_QUANT_F16)       rcre_f16_to_f32(payload, deq, (size_t)hdr.padded_floats);
    else if (qtype == RCRE_GEN_QUANT_BF16) rcre_bf16_to_f32(payload, deq, (size_t)hdr.padded_floats);
    else if (qtype == RCRE_GEN_QUANT_Q8_0) rcre_dequantize_q8_0(payload, deq, (size_t)hdr.padded_floats);
    else                                   rcre_dequantize_q4_0(payload, deq, (size_t)hdr.padded_floats);

    size_t pos = 0;
    for (size_t p = 0; p < num_params; ++p) {
        memcpy(params[p]->data, deq + pos, params[p]->numel * sizeof(float));
        pos += params[p]->numel;
    }

    rcre_arena_destroy(tmp_arena);
    if (out_qtype) *out_qtype = qtype;
    return RCRE_OK;
}

int rcre_gen_load_any(
    RCRE_Tensor* const* params,
    size_t num_params,
    const char* filepath,
    RCRE_Arena* scratch,
    char* out_format,
    size_t max_fmt)
{
    if (!params || num_params == 0 || !filepath || !scratch) return RCRE_ERR_INVALID_ARG;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return RCRE_ERR_IO;
    uint8_t magic[4] = {0};
    size_t nread = fread(magic, 1, 4, fp);
    fclose(fp);
    if (nread < 4) return RCRE_ERR_IO;

    if (memcmp(magic, "RGC1", 4) == 0) {
        if (out_format && max_fmt > 0) snprintf(out_format, max_fmt, "RGC1/RCRZ");
        return rcre_gen_load_compressed(params, num_params, filepath, scratch);
    }
    if (memcmp(magic, "RGQ1", 4) == 0) {
        RCRE_GenQuantType qt = RCRE_GEN_QUANT_Q8_0;
        int rc = rcre_gen_load_quantized(params, num_params, filepath, &qt);
        if (rc == RCRE_OK && out_format && max_fmt > 0) {
            const char* tag = (qt == RCRE_GEN_QUANT_F16)  ? "RGQ1/F16"  :
                              (qt == RCRE_GEN_QUANT_BF16) ? "RGQ1/BF16" :
                              (qt == RCRE_GEN_QUANT_Q8_0) ? "RGQ1/Q8_0" : "RGQ1/Q4_0";
            snprintf(out_format, max_fmt, "%s", tag);
        }
        return rc;
    }

    if (out_format && max_fmt > 0) snprintf(out_format, max_fmt, "SafeTensors/F32");
    return rcre_gen_load_safetensors(params, num_params, filepath);
}
