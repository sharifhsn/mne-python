"""Profile the ttest breakdown to understand where 0.735ms/perm is spent.

Break down into:
1. signs_1d = 2.0 * order - 1.0  (bool→float conversion)
2. np.dot(signs, X)              (BLAS GEMV: (7,) @ (7, 307K))
3. elementwise chain (7 ops)     (mean, sq, var, clip, sqrt, div, scale)
4. threshold comparison           (t > thresh)
5. np.where                       (index extraction)
"""

import time
import numpy as np
import sys
sys.path.insert(0, "/Users/sharif/Code/mne-python")

# Simulate the real fsaverage workload
n_samp = 7       # subjects
n_vars = 307_260  # 20,484 vertices × 15 timepoints

np.random.seed(42)
X = np.random.randn(n_samp, n_vars)
_sum_sq = np.sum(X**2, axis=0)
_sqrt_n_nm1 = np.sqrt(n_samp * (n_samp - 1))

# Pre-allocated buffers
_mean_s = np.empty(n_vars, dtype=np.float64)
_denom_sq = np.empty(n_vars, dtype=np.float64)
_t_buf = np.empty(n_vars, dtype=np.float64)

# Generate random sign-flip orders
n_perms = 2000
orders = [np.random.randint(0, 2, size=n_samp).astype(bool) for _ in range(n_perms)]

threshold = 1.67  # typical p=0.05 threshold for t-test

# Warmup
for order in orders[:10]:
    signs_1d = 2.0 * order - 1.0
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
    np.where(x_in)

print(f"Profiling ttest breakdown: {n_samp} subjects × {n_vars:,} tests, {n_perms} perms")
print("=" * 70)

# Profile each stage
t_signs = 0
t_dot = 0
t_elem = 0
t_thresh = 0
t_where = 0
t_total = 0

for order in orders:
    t0 = time.perf_counter()

    # Stage 1: signs computation
    signs_1d = 2.0 * order - 1.0
    t1 = time.perf_counter()

    # Stage 2: BLAS GEMV dot product
    np.dot(signs_1d, X, out=_mean_s)
    t2 = time.perf_counter()

    # Stage 3: elementwise chain (7 operations)
    _mean_s /= n_samp                          # 1
    np.multiply(_mean_s, _mean_s, out=_denom_sq) # 2
    _denom_sq *= -n_samp                        # 3
    _denom_sq += _sum_sq                        # 4
    np.maximum(_denom_sq, 0, out=_denom_sq)     # 5
    np.sqrt(_denom_sq, out=_denom_sq)           # 6
    np.divide(_mean_s, _denom_sq, out=_t_buf)   # 7
    _t_buf *= _sqrt_n_nm1                       # 8
    t3 = time.perf_counter()

    # Stage 4: threshold
    x_in = _t_buf > threshold
    t4 = time.perf_counter()

    # Stage 5: np.where
    act_idx = np.where(x_in)[0]
    t5 = time.perf_counter()

    t_signs += t1 - t0
    t_dot += t2 - t1
    t_elem += t3 - t2
    t_thresh += t4 - t3
    t_where += t5 - t4
    t_total += t5 - t0

print(f"\nPer-permutation breakdown ({n_perms} perms):")
print(f"  signs (2*order-1):      {t_signs/n_perms*1e6:7.1f} µs  ({t_signs/t_total*100:5.1f}%)")
print(f"  dot (signs @ X):        {t_dot/n_perms*1e6:7.1f} µs  ({t_dot/t_total*100:5.1f}%)")
print(f"  elementwise (8 ops):    {t_elem/n_perms*1e6:7.1f} µs  ({t_elem/t_total*100:5.1f}%)")
print(f"  threshold (t > thresh): {t_thresh/n_perms*1e6:7.1f} µs  ({t_thresh/t_total*100:5.1f}%)")
print(f"  np.where:               {t_where/n_perms*1e6:7.1f} µs  ({t_where/t_total*100:5.1f}%)")
print(f"  TOTAL:                  {t_total/n_perms*1e6:7.1f} µs")
print()

# Now profile sub-components of elementwise chain
print("Sub-breakdown of elementwise chain:")
t_ops = [0.0] * 8
op_names = [
    "mean_s /= n_samp",
    "mean_s * mean_s → denom",
    "denom *= -n_samp",
    "denom += sum_sq",
    "maximum(denom, 0)",
    "sqrt(denom)",
    "mean_s / denom → t_buf",
    "t_buf *= sqrt_n_nm1",
]

for order in orders:
    signs_1d = 2.0 * order - 1.0
    np.dot(signs_1d, X, out=_mean_s)

    t0 = time.perf_counter()
    _mean_s /= n_samp
    t1 = time.perf_counter()
    np.multiply(_mean_s, _mean_s, out=_denom_sq)
    t2 = time.perf_counter()
    _denom_sq *= -n_samp
    t3 = time.perf_counter()
    _denom_sq += _sum_sq
    t4 = time.perf_counter()
    np.maximum(_denom_sq, 0, out=_denom_sq)
    t5 = time.perf_counter()
    np.sqrt(_denom_sq, out=_denom_sq)
    t6 = time.perf_counter()
    np.divide(_mean_s, _denom_sq, out=_t_buf)
    t7 = time.perf_counter()
    _t_buf *= _sqrt_n_nm1
    t8 = time.perf_counter()

    times = [t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6, t8-t7]
    for i, dt in enumerate(times):
        t_ops[i] += dt

total_elem = sum(t_ops)
for name, t in zip(op_names, t_ops):
    print(f"  {name:30s}: {t/n_perms*1e6:6.1f} µs  ({t/total_elem*100:5.1f}%)")
print(f"  {'TOTAL':30s}: {total_elem/n_perms*1e6:6.1f} µs")

# Memory analysis
print(f"\n\nMemory analysis:")
bytes_per_elem = 8  # float64
print(f"  X array: {n_samp} × {n_vars:,} × 8 = {n_samp * n_vars * bytes_per_elem / 1e6:.1f} MB")
print(f"  Per-perm arrays: 3 × {n_vars:,} × 8 = {3 * n_vars * bytes_per_elem / 1e6:.1f} MB")
print(f"  dot reads: X ({n_samp * n_vars * bytes_per_elem / 1e6:.1f} MB) + signs ({n_samp * 8} B)")
print(f"  Each elementwise op reads+writes: {n_vars * bytes_per_elem / 1e6:.1f} MB × 2 = {n_vars * bytes_per_elem * 2 / 1e6:.1f} MB")
print(f"  Total elementwise memory traffic: 8 ops × {n_vars * bytes_per_elem * 2 / 1e6:.1f} MB = {8 * n_vars * bytes_per_elem * 2 / 1e6:.1f} MB")
print(f"  Fused would do: 1 read ({n_samp * n_vars * bytes_per_elem / 1e6:.1f} MB) + 1 write ({n_vars * bytes_per_elem / 1e6:.1f} MB)")
print(f"  Memory reduction: {(n_samp * n_vars * bytes_per_elem + 8 * n_vars * bytes_per_elem * 2) / (n_samp * n_vars * bytes_per_elem + n_vars * bytes_per_elem):.1f}x")

# Test: What does a Numba fused version look like?
print("\n\n--- Numba fused ttest prototype ---")
try:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _fused_ttest_threshold(X_T, signs, sum_sq, inv_n, neg_n, sqrt_n_nm1,
                                threshold, t_buf):
        """Fused t-test + threshold in one memory pass.

        X_T: (n_vars, n_samp) contiguous — transposed for row-major access.
        signs: (n_samp,) float64 — ±1
        sum_sq: (n_vars,) float64 — precomputed sum(X²) per variable
        Returns n_active (number of supra-threshold indices).
        Active indices are written to t_buf (reused as output).
        t-statistics are written in-place.
        """
        n_vars = X_T.shape[0]
        n_samp = X_T.shape[1]
        n_active = 0  # will be overwritten by reduction
        for j in prange(n_vars):
            # Dot product: signs @ X_T[j, :]
            s = 0.0
            for i in range(n_samp):
                s += signs[i] * X_T[j, i]
            # mean
            mean_s = s * inv_n
            # variance denominator
            denom_sq = mean_s * mean_s * neg_n + sum_sq[j]
            if denom_sq < 0.0:
                denom_sq = 0.0
            denom = denom_sq ** 0.5
            # t-statistic
            if denom > 0.0:
                t = mean_s / denom * sqrt_n_nm1
            else:
                t = 0.0
            t_buf[j] = t
        return 0  # placeholder

    @njit(parallel=True, cache=True)
    def _fused_ttest_with_abs(X_T, signs, sum_sq, inv_n, neg_n, sqrt_n_nm1, t_buf):
        """Fused t-test computing abs(t) for tail=0 two-tailed case."""
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

    # Prepare transposed X for contiguous row access
    X_T = np.ascontiguousarray(X.T)
    inv_n = 1.0 / n_samp
    neg_n = -float(n_samp)

    # Warmup
    print("  Warming up Numba JIT...")
    for _ in range(3):
        signs_1d = 2.0 * orders[0] - 1.0
        _fused_ttest_with_abs(X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1, _t_buf)

    # Benchmark fused ttest only (no threshold/where yet)
    t_fused = 0.0
    for order in orders:
        signs_1d = 2.0 * order - 1.0
        t0 = time.perf_counter()
        _fused_ttest_with_abs(X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1, _t_buf)
        t1 = time.perf_counter()
        t_fused += t1 - t0

    print(f"  Fused ttest (Numba prange): {t_fused/n_perms*1e6:.1f} µs/perm")
    print(f"  vs numpy dot+elem:          {(t_dot+t_elem)/n_perms*1e6:.1f} µs/perm")
    print(f"  Speedup:                    {(t_dot+t_elem)/t_fused:.2f}x")

    # Verify correctness
    signs_1d = 2.0 * orders[0] - 1.0
    # NumPy reference
    np.dot(signs_1d, X, out=_mean_s)
    _mean_s_ref = _mean_s.copy()
    _mean_s_ref /= n_samp
    _denom_sq_ref = _mean_s_ref * _mean_s_ref
    _denom_sq_ref *= -n_samp
    _denom_sq_ref += _sum_sq
    np.maximum(_denom_sq_ref, 0, out=_denom_sq_ref)
    np.sqrt(_denom_sq_ref, out=_denom_sq_ref)
    t_ref = np.where(_denom_sq_ref > 0, _mean_s_ref / _denom_sq_ref * _sqrt_n_nm1, 0.0)

    # Numba
    _fused_ttest_with_abs(X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1, _t_buf)
    max_err = np.max(np.abs(t_ref - _t_buf))
    print(f"  Max error vs NumPy: {max_err:.2e}")

    # Also benchmark: fused ttest + threshold + where combined
    print("\n  --- Fused ttest + threshold + np.where ---")
    t_fused_full = 0.0
    for order in orders:
        signs_1d = 2.0 * order - 1.0
        t0 = time.perf_counter()
        _fused_ttest_with_abs(X_T, signs_1d, _sum_sq, inv_n, neg_n, _sqrt_n_nm1, _t_buf)
        x_in = _t_buf > threshold
        act = np.where(x_in)[0]
        t1 = time.perf_counter()
        t_fused_full += t1 - t0

    print(f"  Fused ttest + thresh + where: {t_fused_full/n_perms*1e6:.1f} µs/perm")
    print(f"  vs numpy all-in:              {t_total/n_perms*1e6:.1f} µs/perm")
    print(f"  Speedup:                      {t_total/t_fused_full:.2f}x")

except ImportError:
    print("  Numba not available, skipping fused benchmark")
