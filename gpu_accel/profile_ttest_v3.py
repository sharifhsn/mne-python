"""Optimize fused ttest: try different strategies to eliminate threshold+where cost.

v2 showed:
- prange ttest: 334 µs (good)
- serial threshold+indices: 263 µs (bad — cross-core cache invalidation)
- Total: 598 µs (1.89x vs numpy 1129 µs)

Strategies to try:
A. Fully serial Numba (no prange) — avoids cross-core cache issue but single-threaded
B. prange ttest writes bool mask (uint8), serial scan of 300KB mask (8x smaller)
C. prange with chunked index collection, then merge
D. prange ttest + parallel prefix sum for index extraction
"""

import time
import numpy as np
import sys
sys.path.insert(0, "/Users/sharif/Code/mne-python")
from numba import njit, prange

n_samp = 7
n_vars = 307_260
np.random.seed(42)
X = np.random.randn(n_samp, n_vars)
X_T = np.ascontiguousarray(X.T)
_sum_sq = np.sum(X**2, axis=0)
_sqrt_n_nm1 = np.sqrt(n_samp * (n_samp - 1))
inv_n = 1.0 / n_samp
neg_n = -float(n_samp)

_t_buf = np.empty(n_vars, dtype=np.float64)
_mask_buf = np.empty(n_vars, dtype=np.uint8)
_idx_buf = np.empty(n_vars, dtype=np.intp)
_idx_buf2 = np.empty(n_vars, dtype=np.intp)
_t_active = np.empty(n_vars, dtype=np.float64)  # buffer for active t-values

n_perms = 2000
orders = [np.random.randint(0, 2, size=n_samp).astype(bool) for _ in range(n_perms)]
threshold = 1.67


# ---- Strategy A: Fully serial ttest + threshold + index extraction ----

@njit(cache=True)
def _serial_ttest_indices(X_T, signs, sum_sq, inv_n, neg_n, sqrt_n_nm1,
                           threshold, t_buf, idx_buf):
    """Fully serial: dot + elementwise + threshold + index in one pass."""
    n_vars = X_T.shape[0]
    n_samp = X_T.shape[1]
    n_active = 0
    for j in range(n_vars):
        s = 0.0
        for i in range(n_samp):
            s += signs[i] * X_T[j, i]
        mean_s = s * inv_n
        denom_sq = mean_s * mean_s * neg_n + sum_sq[j]
        if denom_sq < 0.0:
            denom_sq = 0.0
        denom = denom_sq ** 0.5
        if denom > 0.0:
            t = mean_s / denom * sqrt_n_nm1
        else:
            t = 0.0
        t_buf[j] = t
        if t > threshold:
            idx_buf[n_active] = j
            n_active += 1
    return n_active


# ---- Strategy B: prange ttest + bool mask, serial scan of mask ----

@njit(parallel=True, cache=True)
def _par_ttest_mask(X_T, signs, sum_sq, inv_n, neg_n, sqrt_n_nm1,
                     threshold, t_buf, mask_buf):
    """Parallel ttest + write both t_buf and mask_buf."""
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
            t = mean_s / denom * sqrt_n_nm1
        else:
            t = 0.0
        t_buf[j] = t
        mask_buf[j] = 1 if t > threshold else 0


@njit(cache=True)
def _scan_mask(mask_buf, idx_buf, n):
    """Serial scan of uint8 mask (300 KB for 307K elements)."""
    n_active = 0
    for j in range(n):
        if mask_buf[j]:
            idx_buf[n_active] = j
            n_active += 1
    return n_active


# ---- Strategy C: prange with output active values + indices ----

@njit(parallel=True, cache=True)
def _par_ttest_only(X_T, signs, sum_sq, inv_n, neg_n, sqrt_n_nm1, t_buf):
    """Just the parallel ttest (from v2)."""
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
def _scan_tbuf(t_buf, threshold, idx_buf, n):
    """Serial scan of float64 t_buf (2.5 MB for 307K elements)."""
    n_active = 0
    for j in range(n):
        if t_buf[j] > threshold:
            idx_buf[n_active] = j
            n_active += 1
    return n_active


# ---- Strategy D: serial ttest that outputs (idx, t_val) pairs only ----

@njit(cache=True)
def _serial_ttest_compact(X_T, signs, sum_sq, inv_n, neg_n, sqrt_n_nm1,
                           threshold, idx_buf, t_active_buf):
    """Serial ttest outputting only active indices and their t-values.
    Skips writing full t_buf entirely."""
    n_vars = X_T.shape[0]
    n_samp = X_T.shape[1]
    n_active = 0
    for j in range(n_vars):
        s = 0.0
        for i in range(n_samp):
            s += signs[i] * X_T[j, i]
        mean_s = s * inv_n
        denom_sq = mean_s * mean_s * neg_n + sum_sq[j]
        if denom_sq < 0.0:
            denom_sq = 0.0
        denom = denom_sq ** 0.5
        if denom > 0.0:
            t = mean_s / denom * sqrt_n_nm1
        else:
            t = 0.0
        if t > threshold:
            idx_buf[n_active] = j
            t_active_buf[n_active] = t
            n_active += 1
    return n_active


# ---- Strategy E: serial ttest both tails, output compact ----

@njit(cache=True)
def _serial_ttest_compact_2tail(X_T, signs, sum_sq, inv_n, neg_n, sqrt_n_nm1,
                                 threshold, idx_pos, t_pos, idx_neg, t_neg):
    """Serial ttest outputting compact (idx, t_val) for both tails in one pass."""
    n_vars = X_T.shape[0]
    n_samp = X_T.shape[1]
    n_pos = 0
    n_neg = 0
    neg_threshold = -threshold
    for j in range(n_vars):
        s = 0.0
        for i in range(n_samp):
            s += signs[i] * X_T[j, i]
        mean_s = s * inv_n
        denom_sq = mean_s * mean_s * neg_n + sum_sq[j]
        if denom_sq < 0.0:
            denom_sq = 0.0
        denom = denom_sq ** 0.5
        if denom > 0.0:
            t = mean_s / denom * sqrt_n_nm1
        else:
            t = 0.0
        if t > threshold:
            idx_pos[n_pos] = j
            t_pos[n_pos] = t
            n_pos += 1
        elif t < neg_threshold:
            idx_neg[n_neg] = j
            t_neg[n_neg] = t
            n_neg += 1
    return n_pos, n_neg


# ---- Warmup all ----
print("Warming up Numba JIT...")
signs_1d = 2.0 * orders[0] - 1.0
for _ in range(5):
    _serial_ttest_indices(X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1,
                          threshold, _t_buf, _idx_buf)
    _par_ttest_mask(X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1,
                    threshold, _t_buf, _mask_buf)
    _scan_mask(_mask_buf, _idx_buf, n_vars)
    _par_ttest_only(X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1, _t_buf)
    _scan_tbuf(_t_buf, threshold, _idx_buf, n_vars)
    _serial_ttest_compact(X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1,
                          threshold, _idx_buf, _t_active)
    _serial_ttest_compact_2tail(X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1,
                                threshold, _idx_buf, _t_active, _idx_buf2, _t_active)

print(f"Benchmarking: {n_samp} subjects × {n_vars:,} tests, {n_perms} perms, threshold={threshold}")
print("=" * 70)

# ---- NumPy baseline ----
_mean_s = np.empty(n_vars, dtype=np.float64)
_denom_sq = np.empty(n_vars, dtype=np.float64)
t_numpy = 0
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
    act = np.where(_t_buf > threshold)[0]
    t1 = time.perf_counter()
    t_numpy += t1 - t0
print(f"NumPy baseline:                       {t_numpy/n_perms*1e6:7.1f} µs/perm (1.00x)")

# ---- Strategy A: fully serial ----
t_A = 0
for order in orders:
    signs_1d = 2.0 * order - 1.0
    t0 = time.perf_counter()
    n_act = _serial_ttest_indices(X_T, signs_1d, _sum_sq, inv_n, neg_n,
                                   _sqrt_n_nm1, threshold, _t_buf, _idx_buf)
    t1 = time.perf_counter()
    t_A += t1 - t0
print(f"A. Serial ttest+thresh+idx:           {t_A/n_perms*1e6:7.1f} µs/perm ({t_numpy/t_A:.2f}x)")

# ---- Strategy B: prange + mask + scan mask ----
t_B = 0
for order in orders:
    signs_1d = 2.0 * order - 1.0
    t0 = time.perf_counter()
    _par_ttest_mask(X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1,
                    threshold, _t_buf, _mask_buf)
    n_act = _scan_mask(_mask_buf, _idx_buf, n_vars)
    t1 = time.perf_counter()
    t_B += t1 - t0
print(f"B. Parallel ttest+mask, scan mask:    {t_B/n_perms*1e6:7.1f} µs/perm ({t_numpy/t_B:.2f}x)")

# ---- Strategy C: prange ttest, scan tbuf ----
t_C = 0
for order in orders:
    signs_1d = 2.0 * order - 1.0
    t0 = time.perf_counter()
    _par_ttest_only(X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1, _t_buf)
    n_act = _scan_tbuf(_t_buf, threshold, _idx_buf, n_vars)
    t1 = time.perf_counter()
    t_C += t1 - t0
print(f"C. Parallel ttest, scan tbuf:         {t_C/n_perms*1e6:7.1f} µs/perm ({t_numpy/t_C:.2f}x)")

# ---- Strategy D: serial compact (no full t_buf) ----
t_D = 0
for order in orders:
    signs_1d = 2.0 * order - 1.0
    t0 = time.perf_counter()
    n_act = _serial_ttest_compact(X_T, signs_1d, _sum_sq, inv_n, neg_n,
                                   _sqrt_n_nm1, threshold, _idx_buf, _t_active)
    t1 = time.perf_counter()
    t_D += t1 - t0
print(f"D. Serial compact (no full t_buf):    {t_D/n_perms*1e6:7.1f} µs/perm ({t_numpy/t_D:.2f}x)")

# ---- Strategy E: serial compact 2-tailed ----
t_E = 0
for order in orders:
    signs_1d = 2.0 * order - 1.0
    t0 = time.perf_counter()
    n_pos, n_neg = _serial_ttest_compact_2tail(
        X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1, threshold,
        _idx_buf, _t_active, _idx_buf2, _t_active)
    t1 = time.perf_counter()
    t_E += t1 - t0
print(f"E. Serial compact 2-tail:             {t_E/n_perms*1e6:7.1f} µs/perm ({t_numpy/t_E:.2f}x)")

# ---- Parity check all strategies ----
print("\n--- Parity check ---")
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

# A
_serial_ttest_indices(X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1,
                      threshold, _t_buf, _idx_buf)
n_A = _serial_ttest_indices(X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1,
                            threshold, _t_buf, _idx_buf)
print(f"A: t-stat err={np.max(np.abs(t_ref - _t_buf)):.2e}, "
      f"idx match={np.array_equal(ref_pos, _idx_buf[:n_A])}, n={n_A}")

# B
_par_ttest_mask(X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1,
                threshold, _t_buf, _mask_buf)
n_B = _scan_mask(_mask_buf, _idx_buf, n_vars)
print(f"B: t-stat err={np.max(np.abs(t_ref - _t_buf)):.2e}, "
      f"idx match={np.array_equal(ref_pos, _idx_buf[:n_B])}, n={n_B}")

# D
n_D = _serial_ttest_compact(X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1,
                             threshold, _idx_buf, _t_active)
print(f"D: idx match={np.array_equal(ref_pos, _idx_buf[:n_D])}, "
      f"t_active match={np.allclose(t_ref[ref_pos], _t_active[:n_D])}, n={n_D}")

# E
n_pos, n_neg = _serial_ttest_compact_2tail(
    X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1, threshold,
    _idx_buf, _t_active, _idx_buf2, _t_active)
print(f"E: pos match={np.array_equal(ref_pos, _idx_buf[:n_pos])}, "
      f"neg match={np.array_equal(ref_neg, _idx_buf2[:n_neg])}")
