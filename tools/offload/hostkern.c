/* Host-side fp4 matrix-vector kernel with the card's MM semantics (docs/isa.md "MM", "Weight
 * formats"; docs/offload.md, "Hybrid"): what the host needs to compute an expert the card's
 * cache misses, from the expert's image in host RAM.
 *
 *   gcc -O3 -mavx2 -ffp-contract=off -fopenmp -o hostkern tools/offload/hostkern.c
 *   ./hostkern [N K experts threads]      default: Gemma 4 26B-A4B's gate_up (1408 x 2816)
 *
 * y[n] = isum_4 over the K/128 blocks k of t[k] = (i2f(isum_k) * ws_k) * ascale_k, with
 * isum_k = sum_b m_b * sum_{i in sub-block b} a[128k + i] * w[i] the exact integer of the block
 * (w: E2M1 codes as twice their value, {0, 1, 2, 3, 4, 6, 8, 12} and a sign; ws: the bf16 block
 * scale, m_b: its four 4-bit multipliers), as the ISA simulator computes it (-ffp-contract=off:
 * every multiply and add rounds on its own, as on the card; no FMA contraction). The bench
 * checks the AVX2 kernel against a scalar version of the same formula (bit for bit), then times
 * it over
 * `experts` different matrices (more bytes than the caches) and reports GB/s of weight bytes
 * (nibbles + scale words) and the time per matrix.
 */
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const int8_t E2M1[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

static float bf16(uint32_t sw) {
  uint32_t u = (sw & 0xFFFFu) << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

/* one row: nib = K/2 bytes, sw = K/128 scale words; a = K int8, as = K/128 fp32 */
static float row_ref(const uint8_t *nib, const uint32_t *sw, const int8_t *a, const float *as,
                     int K) {
  int KB = K / 128;
  float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int k = 0; k < KB; k++) {
    int32_t isum = 0;
    for (int b = 0; b < 4; b++) {
      int32_t s = 0;
      for (int i = 0; i < 32; i++) {
        int e = 128 * k + 32 * b + i;
        uint8_t c = (nib[e / 2] >> (4 * (e & 1))) & 15;
        s += a[e] * E2M1[c];
      }
      isum += (int32_t)((sw[k] >> (16 + 4 * b)) & 15) * s;
    }
    p[k & 3] += ((float)isum * bf16(sw[k])) * as[k];
  }
  return (p[0] + p[2]) + (p[1] + p[3]);
}

/* the activation split into even / odd elements, as the nibbles are (per 64-element half) */
static void split_act(const int8_t *a, int8_t *ae, int8_t *ao, int K) {
  for (int h = 0; h < K / 64; h++)
    for (int i = 0; i < 32; i++) {
      ae[32 * h + i] = a[64 * h + 2 * i];
      ao[32 * h + i] = a[64 * h + 2 * i + 1];
    }
}

static inline __m256i dot32(__m256i w, __m256i a) {   /* 8 x int32 sums of w * a */
  __m256i p = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(a, w));
  return _mm256_madd_epi16(p, _mm256_set1_epi16(1));
}

static float row_avx2(const uint8_t *nib, const uint32_t *sw, const int8_t *ae, const int8_t *ao,
                      const float *as, int K) {
  const __m256i lut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)E2M1));
  const __m256i m4 = _mm256_set1_epi8(15);
  int KB = K / 128;
  float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int k = 0; k < KB; k++) {
    uint32_t s = sw[k];
    __m256i acc = _mm256_setzero_si256();
    for (int h = 0; h < 2; h++) {                /* sub-blocks 2h, 2h + 1 */
      __m256i v = _mm256_loadu_si256((const __m256i *)(nib + 64 * k + 32 * h));
      __m256i lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(v, m4));
      __m256i hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v, 4), m4));
      /* byte j of the 32 is elements 2j, 2j+1 of this 64-element half; the activation halves are
       * split the same way. Lanes 0-3: sub-block 2h, lanes 4-7: sub-block 2h + 1. */
      __m256i ve = _mm256_loadu_si256((const __m256i *)(ae + 64 * k + 32 * h));
      __m256i vo = _mm256_loadu_si256((const __m256i *)(ao + 64 * k + 32 * h));
      __m256i d = _mm256_add_epi32(dot32(lo, ve), dot32(hi, vo));
      __m256i m = _mm256_setr_epi32((s >> (16 + 8 * h)) & 15, (s >> (16 + 8 * h)) & 15,
                                    (s >> (16 + 8 * h)) & 15, (s >> (16 + 8 * h)) & 15,
                                    (s >> (20 + 8 * h)) & 15, (s >> (20 + 8 * h)) & 15,
                                    (s >> (20 + 8 * h)) & 15, (s >> (20 + 8 * h)) & 15);
      acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(d, m));
    }
    __m128i x = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0x4E));
    x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0xB1));
    int32_t isum = _mm_cvtsi128_si32(x);
    p[k & 3] += ((float)isum * bf16(s)) * as[k];
  }
  return (p[0] + p[2]) + (p[1] + p[3]);
}

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(int argc, char **argv) {
  /* the ISA flushes denormal inputs and results to zero (docs/isa.md "Arithmetic"); so does
   * SSE with FTZ + DAZ, in every thread OpenMP starts (they inherit the mode from here) */
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  int N = argc > 1 ? atoi(argv[1]) : 1408, K = argc > 2 ? atoi(argv[2]) : 2816;
  int E = argc > 3 ? atoi(argv[3]) : 256, T = argc > 4 ? atoi(argv[4]) : 4;
  int KB = K / 128;
  size_t nb = (size_t)N * K / 2, sb = (size_t)N * KB * 4, per = nb + sb;
  uint8_t *img = aligned_alloc(64, per * E);
  srand(1);
  for (size_t i = 0; i < per * E; i++) img[i] = (uint8_t)rand();
  for (int e = 0; e < E; e++) {                  /* scale words: a sane bf16, any multipliers */
    uint32_t *sw = (uint32_t *)(img + per * e + nb);
    for (size_t i = 0; i < (size_t)N * KB; i++)
      sw[i] = (0x3C00u + (rand() & 0x3F)) | ((uint32_t)rand() << 16);
  }
  int8_t *a = malloc(K), *ae = aligned_alloc(64, K), *ao = aligned_alloc(64, K);
  float *as = malloc(4 * KB), *y = malloc(4 * N);
  for (int i = 0; i < K; i++) a[i] = (int8_t)(rand() % 255 - 127);
  for (int k = 0; k < KB; k++) as[k] = 1.0f / (1 + k);
  split_act(a, ae, ao, K);
  int bad = 0;                                   /* bit for bit against the reference */
  for (int n = 0; n < N; n++) {
    const uint8_t *nib = img + (size_t)n * K / 2;
    const uint32_t *sw = (const uint32_t *)(img + nb) + (size_t)n * KB;
    float r = row_ref(nib, sw, a, as, K), v = row_avx2(nib, sw, ae, ao, as, K);
    bad += memcmp(&r, &v, 4) != 0;
  }
  printf("check: %d of %d rows differ from the reference\n", bad, N);
  omp_set_num_threads(T);
  for (int rep = 0; rep < 3; rep++) {
    double t0 = now();
    for (int e = 0; e < E; e++) {
      const uint8_t *base = img + per * e;
#pragma omp parallel for schedule(static)
      for (int n = 0; n < N; n++)
        y[n] = row_avx2(base + (size_t)n * K / 2, (const uint32_t *)(base + nb) + (size_t)n * KB,
                        ae, ao, as, K);
    }
    double dt = now() - t0;
    printf("%d x %d, %d matrices (%.0f MB), %d threads: %.2f GB/s, %.1f us per matrix\n", N, K,
           E, per * E / 1e6, T, per * E / dt / 1e9, dt / E * 1e6);
  }
  return bad != 0;
}
