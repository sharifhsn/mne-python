"""Prototype fully fused Numba ttest + threshold + index extraction.

Strategy: prange for ttest computation, then serial index extraction in same
Numba function. This avoids:
1. 8 separate memory passes for elementwise ops (→ 1 fused pass)
2. np.where Python overhead (306 µs → ~30 µs serial scan in Numba)
3. Boolean array allocation for threshold comparison

Two-pass within Numba:
- Phase 1 (prange): compute t_buf[j] for all j
- Phase 2 (serial): scan t_buf for |t| > threshold, write indices
"""

import time
import numpy as np
import sys
sys.path.insert(0, "/Users/sharif/Code/mne-python")
from numba import njit, prange

# ---- Fused functions ----

@njit(parallel=True, cache=True)
def _fused_ttest(X_T, signs, sum_sq, inv_n, neg_n, sqrt_n_nm1, t_buf):
    """Phase 1: Fused dot + elementwise t-stat computation (parallel)."""
    n_vars = X_T.shape[0]
    n_samp = X_T.shape[1]
    for j in prange(n_vars):
        s = 0.0
        for i in range(n_samp):
            s += signs[i] * X_T[j, i]
        mean_s = s * inv_n
        denom_sq = mean_s * mean_s * neg_n + sum_sq[j]
        if denom_sq < 0.0:
            denom_sq = 0.0
        denom = denom_sq ** 0.5
        if denom > 0.0:
            t_buf[j] = mean_s / denom * sqrt_n_nm1
        else:
            t_buf[j] = 0.0


@njit(cache=True)
def _threshold_indices(t_buf, threshold, idx_buf):
    """Phase 2: threshold + index extraction (serial, cache-hot from phase 1)."""
    n = t_buf.shape[0]
    n_active = 0
    for j in range(n):
        if t_buf[j] > threshold:
            idx_buf[n_active] = j
            n_active += 1
    return n_active


@njit(cache=True)
def _threshold_indices_neg(t_buf, threshold, idx_buf):
    """Phase 2: negative threshold (tail=-1)."""
    n = t_buf.shape[0]
    n_active = 0
    for j in range(n):
        if t_buf[j] < threshold:
            idx_buf[n_active] = j
            n_active += 1
    return n_active


@njit(cache=True)
def _threshold_indices_abs(t_buf, threshold, idx_buf_pos, idx_buf_neg):
    """Phase 2: two-tailed — extract positive and negative tails in one pass."""
    n = t_buf.shape[0]
    n_pos = 0
    n_neg = 0
    for j in range(n):
        t = t_buf[j]
        if t > threshold:
            idx_buf_pos[n_pos] = j
            n_pos += 1
        elif t < -threshold:
            idx_buf_neg[n_neg] = j
            n_neg += 1
    return n_pos, n_neg


# ---- Setup ----
n_samp = 7
n_vars = 307_260
np.random.seed(42)
X = np.random.randn(n_samp, n_vars)
X_T = np.ascontiguousarray(X.T)  # (n_vars, n_samp) contiguous
_sum_sq = np.sum(X**2, axis=0)
_sqrt_n_nm1 = np.sqrt(n_samp * (n_samp - 1))
inv_n = 1.0 / n_samp
neg_n = -float(n_samp)

_mean_s = np.empty(n_vars, dtype=np.float64)
_denom_sq = np.empty(n_vars, dtype=np.float64)
_t_buf = np.empty(n_vars, dtype=np.float64)
_idx_buf = np.empty(n_vars, dtype=np.intp)
_idx_buf_neg = np.empty(n_vars, dtype=np.intp)

n_perms = 2000
orders = [np.random.randint(0, 2, size=n_samp).astype(bool) for _ in range(n_perms)]
threshold = 1.67

# Warmup
print("Warming up Numba JIT...")
for _ in range(5):
    signs_1d = 2.0 * orders[0] - 1.0
    _fused_ttest(X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1, _t_buf)
    _threshold_indices(_t_buf, threshold, _idx_buf)
    _threshold_indices_neg(_t_buf, -threshold, _idx_buf)
    _threshold_indices_abs(_t_buf, threshold, _idx_buf, _idx_buf_neg)

print(f"Benchmarking: {n_samp} subjects × {n_vars:,} tests, {n_perms} perms")
print("=" * 70)

# ---- Benchmark: NumPy baseline (current v7 code) ----
t_numpy_total = 0
for order in orders:
    signs_1d = 2.0 * order - 1.0
    t0 = time.perf_counter()
    np.dot(signs_1d, X, out=_mean_s)
    _mean_s /= n_samp
    np.multiply(_mean_s, _mean_s, out=_denom_sq)
    _denom_sq *= -n_samp
    _denom_sq += _sum_sq
    np.maximum(_denom_sq, 0, out=_denom_sq)
    np.sqrt(_denom_sq, out=_denom_sq)
    np.divide(_mean_s, _denom_sq, out=_t_buf)
    _t_buf *= _sqrt_n_nm1
    x_in = _t_buf > threshold
    act = np.where(x_in)[0]
    t1 = time.perf_counter()
    t_numpy_total += t1 - t0

print(f"\nNumPy baseline (dot + 8 elem + threshold + np.where):")
print(f"  {t_numpy_total/n_perms*1e6:.1f} µs/perm")

# ---- Benchmark: Fused ttest + serial index extraction (tail=1) ----
t_fused_total = 0
t_ttest_only = 0
t_thresh_only = 0
for order in orders:
    signs_1d = 2.0 * order - 1.0
    t0 = time.perf_counter()
    _fused_ttest(X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1, _t_buf)
    t1 = time.perf_counter()
    n_act = _threshold_indices(_t_buf, threshold, _idx_buf)
    t2 = time.perf_counter()
    t_fused_total += t2 - t0
    t_ttest_only += t1 - t0
    t_thresh_only += t2 - t1

print(f"\nFused Numba (ttest prange + serial threshold/indices):")
print(f"  ttest:         {t_ttest_only/n_perms*1e6:.1f} µs/perm")
print(f"  thresh+idx:    {t_thresh_only/n_perms*1e6:.1f} µs/perm")
print(f"  total:         {t_fused_total/n_perms*1e6:.1f} µs/perm")
print(f"  vs numpy:      {t_numpy_total/t_fused_total:.2f}x speedup")

# ---- Benchmark: two-tailed (tail=0) ----
t_np_2t = 0
for order in orders:
    signs_1d = 2.0 * order - 1.0
    t0 = time.perf_counter()
    np.dot(signs_1d, X, out=_mean_s)
    _mean_s /= n_samp
    np.multiply(_mean_s, _mean_s, out=_denom_sq)
    _denom_sq *= -n_samp
    _denom_sq += _sum_sq
    np.maximum(_denom_sq, 0, out=_denom_sq)
    np.sqrt(_denom_sq, out=_denom_sq)
    np.divide(_mean_s, _denom_sq, out=_t_buf)
    _t_buf *= _sqrt_n_nm1
    act_pos = np.where(_t_buf > threshold)[0]
    act_neg = np.where(_t_buf < -threshold)[0]
    t1 = time.perf_counter()
    t_np_2t += t1 - t0

t_fused_2t = 0
for order in orders:
    signs_1d = 2.0 * order - 1.0
    t0 = time.perf_counter()
    _fused_ttest(X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1, _t_buf)
    n_pos, n_neg = _threshold_indices_abs(_t_buf, threshold, _idx_buf, _idx_buf_neg)
    t1 = time.perf_counter()
    t_fused_2t += t1 - t0

print(f"\nTwo-tailed comparison:")
print(f"  NumPy:  {t_np_2t/n_perms*1e6:.1f} µs/perm")
print(f"  Fused:  {t_fused_2t/n_perms*1e6:.1f} µs/perm")
print(f"  Speedup: {t_np_2t/t_fused_2t:.2f}x")

# ---- Parity check ----
print("\n\n--- Parity check ---")
signs_1d = 2.0 * orders[0] - 1.0

# NumPy reference
np.dot(signs_1d, X, out=_mean_s)
_mean_s /= n_samp
np.multiply(_mean_s, _mean_s, out=_denom_sq)
_denom_sq *= -n_samp
_denom_sq += _sum_sq
np.maximum(_denom_sq, 0, out=_denom_sq)
np.sqrt(_denom_sq, out=_denom_sq)
t_ref = np.where(_denom_sq > 0, _mean_s / _denom_sq * _sqrt_n_nm1, 0.0)
ref_pos = np.where(t_ref > threshold)[0]
ref_neg = np.where(t_ref < -threshold)[0]

# Numba
_fused_ttest(X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1, _t_buf)
n_pos = _threshold_indices(_t_buf, threshold, _idx_buf)
fused_pos = _idx_buf[:n_pos].copy()

n_pos2, n_neg2 = _threshold_indices_abs(_t_buf, threshold, _idx_buf, _idx_buf_neg)
fused_pos2 = _idx_buf[:n_pos2].copy()
fused_neg2 = _idx_buf_neg[:n_neg2].copy()

max_err = np.max(np.abs(t_ref - _t_buf))
print(f"t-stat max error: {max_err:.2e}")
print(f"tail=1 indices match: {np.array_equal(ref_pos, fused_pos)}")
print(f"  NumPy: {len(ref_pos)} active, Numba: {len(fused_pos)} active")
print(f"tail=0 pos indices match: {np.array_equal(ref_pos, fused_pos2)}")
print(f"tail=0 neg indices match: {np.array_equal(ref_neg, fused_neg2)}")

# ---- signs_1d cost ----
print("\n\n--- signs_1d optimization ---")
t_signs_np = 0
for order in orders:
    t0 = time.perf_counter()
    signs_1d = 2.0 * order - 1.0
    t1 = time.perf_counter()
    t_signs_np += t1 - t0

# Alternative: pre-allocate signs buffer, use lookup
_signs_buf = np.empty(n_samp, dtype=np.float64)

@njit(cache=True)
def _make_signs(order, out):
    for i in range(order.shape[0]):
        out[i] = 1.0 if order[i] else -1.0

# warmup
_make_signs(orders[0], _signs_buf)

t_signs_nb = 0
for order in orders:
    t0 = time.perf_counter()
    _make_signs(order, _signs_buf)
    t1 = time.perf_counter()
    t_signs_nb += t1 - t0

print(f"  NumPy (2.0*order-1.0): {t_signs_np/n_perms*1e6:.1f} µs")
print(f"  Numba:                 {t_signs_nb/n_perms*1e6:.1f} µs")
print(f"  (both negligible at n_samp={n_samp})")
