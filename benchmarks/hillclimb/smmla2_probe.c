// H72 probe: a 2-bit SMMLA (i8mm) batched kernel against the shipped 4-query
// nibble-LUT kernel, and the price the single-query LUT kernel pays if the
// codes move to the pair-interleaved layout the SMMLA kernel wants.
//
// Layouts (per 32-vector block, 192 byte-groups at dim 768 / 2 bits):
//   sequential  codes[g*32 + v]                (shipped arm 2-bit layout)
//   pair        codes[c*64 + v*2 + j], g = 2c+j (each vector's two consecutive
//                                              byte-groups adjacent)
//
// Variants:
//   0  LUT, 4 queries, sequential          (transcribes scan_groups_neon)
//   1  SMMLA, 8 queries, pair              (4 query pairs)
//   2  SMMLA, 12 queries, pair             (6 pairs; expected to spill)
//   3  LUT, 1 query, sequential            (transcribes score_4bit_block_neon)
//   4  LUT, 1 query, pair, LD2 deinterleave
//   5  LUT, 1 query, pair, LDR + UZP deinterleave
//   6  SMMLA, 4 queries, pair              (2 pairs)
//   7  SMMLA, 8 queries, vm8               (8 groups per vector adjacent: each
//                                           TBL output is a B operand, no zips)
//   8  SMMLA, 12 queries, vm8
//   9  LUT, 1 query, vm8, UZP tree deinterleave (8 group registers per 128 B)
//  10  LUT, 1 query, vm8, two LD4 + one UZP level per 128 B
//  11  SMMLA, 1 query duplicated (NP=1), vm8 — the 4-bit path's nq=1 shape
//
// Build:  cc -O3 -march=armv8.6-a+i8mm -o smmla2_probe smmla2_probe.c
// Run:    ./smmla2_probe [variant] [n_vectors]
// Prints ms per full pass of the batch and G(query.dim)/s.

#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BLOCK 32
#define N_GROUPS 192
#define N_CHUNKS (N_GROUPS / 2)
#define BLOCK_BYTES (N_GROUPS * BLOCK)

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double clock_hz(void) {
    uint64_t acc = 0;
    const uint64_t n = 10000000;
    double t0 = now();
    for (uint64_t i = 0; i < n; i++) {
        __asm__ volatile("add %0, %0, #1\n add %0, %0, #1\n"
                         "add %0, %0, #1\n add %0, %0, #1\n"
                         "add %0, %0, #1\n add %0, %0, #1\n"
                         "add %0, %0, #1\n add %0, %0, #1\n"
                         : "+r"(acc)::);
    }
    return (double)n * 8.0 / (now() - t0);
}

// ---- LUT kernels (u8 tables, [hi16 | lo16] per group per query) ----------

static inline __attribute__((always_inline)) float
lut_flush(uint16x8_t a0, uint16x8_t a1, uint16x8_t a2, uint16x8_t a3) {
    float32x4_t f = vcvtq_f32_u32(vmovl_u16(vget_low_u16(a0)));
    f = vaddq_f32(f, vcvtq_f32_u32(vmovl_u16(vget_high_u16(a1))));
    f = vaddq_f32(f, vcvtq_f32_u32(vmovl_u16(vget_low_u16(a2))));
    f = vaddq_f32(f, vcvtq_f32_u32(vmovl_u16(vget_high_u16(a3))));
    return vaddvq_f32(f);
}

// variant 0: 4 queries, sequential layout — scan_groups_neon transcribed.
static inline __attribute__((always_inline)) float
lut4_block_seq(const uint8_t *codes, const uint8_t *const luts[4]) {
    uint16x8_t acc[4][4];
    for (int q = 0; q < 4; q++)
        for (int i = 0; i < 4; i++) acc[q][i] = vdupq_n_u16(0);
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    for (int g = 0; g < N_GROUPS; g++) {
        const uint8_t *cp = codes + g * BLOCK;
        uint8x16_t c0 = vld1q_u8(cp), c1 = vld1q_u8(cp + 16);
        uint8x16_t lo0 = vandq_u8(c0, mask), lo1 = vandq_u8(c1, mask);
        uint8x16_t hi0 = vshrq_n_u8(c0, 4), hi1 = vshrq_n_u8(c1, 4);
        for (int q = 0; q < 4; q++) {
            const uint8_t *lp = luts[q] + g * 32;
            uint8x16_t lut_hi = vld1q_u8(lp), lut_lo = vld1q_u8(lp + 16);
            uint8x16_t s0 = vaddq_u8(vqtbl1q_u8(lut_lo, lo0), vqtbl1q_u8(lut_hi, hi0));
            uint8x16_t s1 = vaddq_u8(vqtbl1q_u8(lut_lo, lo1), vqtbl1q_u8(lut_hi, hi1));
            acc[q][0] = vaddw_u8(acc[q][0], vget_low_u8(s0));
            acc[q][1] = vaddw_u8(acc[q][1], vget_high_u8(s0));
            acc[q][2] = vaddw_u8(acc[q][2], vget_low_u8(s1));
            acc[q][3] = vaddw_u8(acc[q][3], vget_high_u8(s1));
        }
    }
    float sink = 0;
    for (int q = 0; q < 4; q++) sink += lut_flush(acc[q][0], acc[q][1], acc[q][2], acc[q][3]);
    return sink;
}

// Single-query group step shared by variants 3/4/5.
#define LUT1_GROUP(c0, c1, lp)                                                          \
    do {                                                                                \
        uint8x16_t lut_hi = vld1q_u8(lp), lut_lo = vld1q_u8((lp) + 16);                 \
        uint8x16_t s0 = vaddq_u8(vqtbl1q_u8(lut_lo, vandq_u8(c0, mask)),                \
                                 vqtbl1q_u8(lut_hi, vshrq_n_u8(c0, 4)));                \
        uint8x16_t s1 = vaddq_u8(vqtbl1q_u8(lut_lo, vandq_u8(c1, mask)),                \
                                 vqtbl1q_u8(lut_hi, vshrq_n_u8(c1, 4)));                \
        a0 = vaddw_u8(a0, vget_low_u8(s0));                                             \
        a1 = vaddw_u8(a1, vget_high_u8(s0));                                            \
        a2 = vaddw_u8(a2, vget_low_u8(s1));                                             \
        a3 = vaddw_u8(a3, vget_high_u8(s1));                                            \
    } while (0)

static inline __attribute__((always_inline)) float
lut1_block(const uint8_t *codes, const uint8_t *lut, const int variant) {
    uint16x8_t a0 = vdupq_n_u16(0), a1 = a0, a2 = a0, a3 = a0;
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    uint8x16_t gsave[8];
    if (variant == 3) {
        for (int g = 0; g < N_GROUPS; g++) {
            const uint8_t *cp = codes + g * BLOCK;
            uint8x16_t c0 = vld1q_u8(cp), c1 = vld1q_u8(cp + 16);
            LUT1_GROUP(c0, c1, lut + g * 32);
        }
    } else if (variant == 9) {
        for (int o = 0; o < N_GROUPS / 8; o++) {
            for (int half = 0; half < 2; half++) {  // vectors 0-15, 16-31
                const uint8_t *cp = codes + o * 256 + half * 128;
                uint8x16_t r[8];
                for (int i = 0; i < 8; i++) r[i] = vld1q_u8(cp + i * 16);
                uint8x16_t t[8], u[8];
                for (int i = 0; i < 4; i++) { t[i] = vuzp1q_u8(r[2*i], r[2*i+1]); t[4+i] = vuzp2q_u8(r[2*i], r[2*i+1]); }
                for (int i = 0; i < 2; i++) { u[i] = vuzp1q_u8(t[2*i], t[2*i+1]); u[2+i] = vuzp2q_u8(t[2*i], t[2*i+1]);
                                              u[4+i] = vuzp1q_u8(t[4+2*i], t[5+2*i]); u[6+i] = vuzp2q_u8(t[4+2*i], t[5+2*i]); }
                uint8x16_t gq[8];
                gq[0] = vuzp1q_u8(u[0], u[1]); gq[4] = vuzp2q_u8(u[0], u[1]);
                gq[2] = vuzp1q_u8(u[2], u[3]); gq[6] = vuzp2q_u8(u[2], u[3]);
                gq[1] = vuzp1q_u8(u[4], u[5]); gq[5] = vuzp2q_u8(u[4], u[5]);
                gq[3] = vuzp1q_u8(u[6], u[7]); gq[7] = vuzp2q_u8(u[6], u[7]);
                if (half == 0) { for (int j = 0; j < 8; j++) gsave[j] = gq[j]; }
                else { for (int j = 0; j < 8; j++) LUT1_GROUP(gsave[j], gq[j], lut + (8 * o + j) * 32); }
            }
        }
    } else if (variant == 10) {
        for (int o = 0; o < N_GROUPS / 8; o++) {
            for (int half = 0; half < 2; half++) {
                const uint8_t *cp = codes + o * 256 + half * 128;
                uint8x16x4_t d0 = vld4q_u8(cp), d1 = vld4q_u8(cp + 64);
                // d.val[k] = groups k and k+4 alternating (v0gk, v0gk+4, v1gk, ...)
                uint8x16_t gq[8];
                for (int k = 0; k < 4; k++) { gq[k] = vuzp1q_u8(d0.val[k], d1.val[k]); gq[k + 4] = vuzp2q_u8(d0.val[k], d1.val[k]); }
                if (half == 0) { for (int j = 0; j < 8; j++) gsave[j] = gq[j]; }
                else { for (int j = 0; j < 8; j++) LUT1_GROUP(gsave[j], gq[j], lut + (8 * o + j) * 32); }
            }
        }
    } else {
        for (int c = 0; c < N_CHUNKS; c++) {
            const uint8_t *cp = codes + c * 64;
            uint8x16_t g0a, g1a, g0b, g1b;  // group 2c (a: v0-15, b: v16-31), group 2c+1
            if (variant == 4) {
                uint8x16x2_t d0 = vld2q_u8(cp), d1 = vld2q_u8(cp + 32);
                g0a = d0.val[0]; g1a = d0.val[1]; g0b = d1.val[0]; g1b = d1.val[1];
            } else {
                uint8x16_t r0 = vld1q_u8(cp), r1 = vld1q_u8(cp + 16);
                uint8x16_t r2 = vld1q_u8(cp + 32), r3 = vld1q_u8(cp + 48);
                g0a = vuzp1q_u8(r0, r1); g1a = vuzp2q_u8(r0, r1);
                g0b = vuzp1q_u8(r2, r3); g1b = vuzp2q_u8(r2, r3);
            }
            LUT1_GROUP(g0a, g0b, lut + (2 * c) * 32);
            LUT1_GROUP(g1a, g1b, lut + (2 * c + 1) * 32);
        }
    }
    return lut_flush(a0, a1, a2, a3);
}

// ---- SMMLA kernel, vm8 layout: 16 B = 2 vectors x 8 groups; field k of the
// 16 lanes is (v0: dims 4g+k, g=0..7 | v1: same) = one B row per vector.
static inline __attribute__((always_inline)) float
smmla_block_vm8(const uint8_t *codes, const int8_t *a_buf, const int8x16_t ta,
                const int8x16_t tb, const int NP) {
    float sink = 0;
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    for (int part = 0; part < 4; part++) {          // 8 vectors per part = 4 loads per octet
        int32x4_t acc[6][4];
        for (int p = 0; p < NP; p++)
            for (int k = 0; k < 4; k++) acc[p][k] = vdupq_n_s32(0);
        for (int o = 0; o < N_GROUPS / 8; o++) {   // 24 octets
            const int8_t *ap = a_buf + (size_t)o * NP * 64;
            for (int i = 0; i < 4; i++) {           // 4 x 16 B = 8 vectors
                uint8x16_t cc = vld1q_u8(codes + o * 256 + part * 64 + i * 16);
                uint8x16_t lo = vandq_u8(cc, mask), hi = vshrq_n_u8(cc, 4);
                int8x16_t f0 = vqtbl1q_s8(tb, lo), f1 = vqtbl1q_s8(ta, lo);
                int8x16_t f2 = vqtbl1q_s8(tb, hi), f3 = vqtbl1q_s8(ta, hi);
                for (int p = 0; p < NP; p++) {
                    // A for field k: query pair's weights at dims 4g+k, g=0..7
                    acc[p][i] = vmmlaq_s32(acc[p][i], vld1q_s8(ap + p * 64 + 0), f0);
                    acc[p][i] = vmmlaq_s32(acc[p][i], vld1q_s8(ap + p * 64 + 16), f1);
                    acc[p][i] = vmmlaq_s32(acc[p][i], vld1q_s8(ap + p * 64 + 32), f2);
                    acc[p][i] = vmmlaq_s32(acc[p][i], vld1q_s8(ap + p * 64 + 48), f3);
                }
            }
        }
        for (int p = 0; p < NP; p++)
            for (int k = 0; k < 4; k++) sink += (float)vaddvq_s32(acc[p][k]);
    }
    return sink;
}

// ---- SMMLA kernel, pair layout ----------------------------------------------
//
// Byte fields at 2 bits (from build_query_neon_lut_from_slice): bits 3:2 = dim
// 4g+0, 1:0 = 4g+1, 7:6 = 4g+2, 5:4 = 4g+3. Two 16-entry tables serve all
// four: Ta[i] = level[i & 3], Tb[i] = level[(i >> 2) & 3], indexed by the
// masked low nibble and the shifted high nibble.
//
// A operand: a_buf[(c * NP + p) * 16] = [q(2p): dims 8c..8c+7][q(2p+1): same].

static inline __attribute__((always_inline)) float
smmla_block(const uint8_t *codes, const int8_t *a_buf, const int8x16_t ta,
            const int8x16_t tb, const int NP) {
    float sink = 0;
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    for (int part = 0; part < 4; part++) {          // 8 vectors per part
        int32x4_t acc[6][4];
        for (int p = 0; p < NP; p++)
            for (int k = 0; k < 4; k++) acc[p][k] = vdupq_n_s32(0);
        for (int c = 0; c < N_CHUNKS; c++) {
            uint8x16_t cc = vld1q_u8(codes + c * 64 + part * 16);
            uint8x16_t lo = vandq_u8(cc, mask), hi = vshrq_n_u8(cc, 4);
            int8x16_t f0 = vqtbl1q_s8(tb, lo);   // dim 4g+0
            int8x16_t f1 = vqtbl1q_s8(ta, lo);   // dim 4g+1
            int8x16_t f2 = vqtbl1q_s8(tb, hi);   // dim 4g+2
            int8x16_t f3 = vqtbl1q_s8(ta, hi);   // dim 4g+3
            int8x16_t z01l = vzip1q_s8(f0, f1), z01h = vzip2q_s8(f0, f1);
            int8x16_t z23l = vzip1q_s8(f2, f3), z23h = vzip2q_s8(f2, f3);
            int8x16_t b0 = vreinterpretq_s8_s16(vzip1q_s16(vreinterpretq_s16_s8(z01l), vreinterpretq_s16_s8(z23l)));
            int8x16_t b1 = vreinterpretq_s8_s16(vzip2q_s16(vreinterpretq_s16_s8(z01l), vreinterpretq_s16_s8(z23l)));
            int8x16_t b2 = vreinterpretq_s8_s16(vzip1q_s16(vreinterpretq_s16_s8(z01h), vreinterpretq_s16_s8(z23h)));
            int8x16_t b3 = vreinterpretq_s8_s16(vzip2q_s16(vreinterpretq_s16_s8(z01h), vreinterpretq_s16_s8(z23h)));
            const int8_t *ap = a_buf + (size_t)c * NP * 16;
            for (int p = 0; p < NP; p++) {
                int8x16_t a = vld1q_s8(ap + p * 16);
                acc[p][0] = vmmlaq_s32(acc[p][0], a, b0);
                acc[p][1] = vmmlaq_s32(acc[p][1], a, b1);
                acc[p][2] = vmmlaq_s32(acc[p][2], a, b2);
                acc[p][3] = vmmlaq_s32(acc[p][3], a, b3);
            }
        }
        for (int p = 0; p < NP; p++)
            for (int k = 0; k < 4; k++) sink += (float)vaddvq_s32(acc[p][k]);
    }
    return sink;
}

int main(int argc, char **argv) {
    int variant = argc > 1 ? atoi(argv[1]) : 0;
    size_t n_vectors = argc > 2 ? strtoull(argv[2], NULL, 10) : 200000;
    size_t n_blocks = n_vectors / BLOCK;
    size_t bytes = n_blocks * BLOCK_BYTES;
    uint8_t *codes = aligned_alloc(64, bytes);
    uint32_t s = 12345;
    for (size_t i = 0; i < bytes; i++) { s = s * 1664525u + 1013904223u; codes[i] = (uint8_t)(s >> 24); }
    uint8_t *luts = aligned_alloc(64, 4 * N_GROUPS * 32);
    for (size_t i = 0; i < 4 * N_GROUPS * 32; i++) { s = s * 1664525u + 1013904223u; luts[i] = (uint8_t)((s >> 25) & 127); }
    const uint8_t *lut4[4] = {luts, luts + N_GROUPS * 32, luts + 2 * N_GROUPS * 32, luts + 3 * N_GROUPS * 32};
    int NP = (variant == 1 || variant == 7) ? 4 : (variant == 2 || variant == 8) ? 6 : variant == 6 ? 2 : variant == 11 ? 1 : 0;
    int8_t *a_buf = aligned_alloc(64, (size_t)N_CHUNKS * 6 * 16 * 2);
    for (size_t i = 0; i < (size_t)N_CHUNKS * 6 * 16 * 2; i++) { s = s * 1664525u + 1013904223u; a_buf[i] = (int8_t)((int)(s >> 24) - 128); }
    int8_t lv[4] = {-100, -33, 33, 100};
    int8_t ta_[16], tb_[16];
    for (int i = 0; i < 16; i++) { ta_[i] = lv[i & 3]; tb_[i] = lv[(i >> 2) & 3]; }
    int8x16_t ta = vld1q_s8(ta_), tb = vld1q_s8(tb_);

    double hz = clock_hz();
    int nq = variant == 0 ? 4 : (variant == 1 || variant == 7) ? 8 : (variant == 2 || variant == 8) ? 12 : variant == 6 ? 4 : 1;
    double best = 1e9;
    volatile float sink = 0;
    for (int rep = 0; rep < 5; rep++) {
        double t0 = now();
        float acc = 0;
        for (size_t b = 0; b < n_blocks; b++) {
            const uint8_t *cb = codes + b * BLOCK_BYTES;
            switch (variant) {
                case 0: acc += lut4_block_seq(cb, lut4); break;
                case 1: acc += smmla_block(cb, a_buf, ta, tb, 4); break;
                case 2: acc += smmla_block(cb, a_buf, ta, tb, 6); break;
                case 6: acc += smmla_block(cb, a_buf, ta, tb, 2); break;
                case 7: acc += smmla_block_vm8(cb, a_buf, ta, tb, 4); break;
                case 8: acc += smmla_block_vm8(cb, a_buf, ta, tb, 6); break;
                case 9: acc += lut1_block(cb, luts, 9); break;
                case 10: acc += lut1_block(cb, luts, 10); break;
                case 11: acc += smmla_block_vm8(cb, a_buf, ta, tb, 1); break;
                case 3: acc += lut1_block(cb, luts, 3); break;
                case 4: acc += lut1_block(cb, luts, 4); break;
                case 5: acc += lut1_block(cb, luts, 5); break;
            }
        }
        sink = acc;
        double t = now() - t0;
        if (t < best) best = t;
    }
    double qd = (double)nq * (double)n_vectors * 768.0;
    double cy_per_block = best * hz / (double)n_blocks;
    printf("variant %d  nq=%2d  %8.3f ms/pass  %6.1f G(q.dim)/s  %7.0f cy/block  %6.2f cy/group  (sink %g)\n",
           variant, nq, best * 1e3, qd / best / 1e9, cy_per_block, cy_per_block / N_GROUPS, (double)sink);
    return 0;
}
