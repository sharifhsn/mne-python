#!/usr/bin/env python
"""Benchmark v14: CSR deduplication + vectorized _get_1samp_orders.

A/B comparison: v14 (precomputed CSR threaded through, vectorized orders)
vs v13 (CSR rebuilt 3x, list-comprehension binary_repr).

The key changes:
1. _setup_adjacency now returns (adjacency, csr_data) where csr_data is
   (indptr, indices, n_src). This avoids rebuilding CSR from list in
   _find_clusters_1dir and _do_1samp_permutations.
2. _get_1samp_orders uses vectorized bit-shifting instead of
   np.fromiter(np.binary_repr(...)) loop.
"""

import time

import numpy as np

import mne
from mne.stats import spatio_temporal_cluster_1samp_test
from mne.stats.cluster_level import (
    _find_clusters,
    _get_1samp_orders,
    _setup_adjacency,
    has_numba,
    ttest_1samp_no_p,
)


# ── Data setup ──────────────────────────────────────────────────────────────
print("=" * 70)
print("v14: CSR deduplication + vectorized orders benchmark")
print("=" * 70)

subjects_dir = mne.datasets.sample.data_path() / "subjects"
src = mne.setup_source_space(
    "fsaverage", spacing="ico5", subjects_dir=subjects_dir, add_dist=False
)
adjacency = mne.spatial_src_adjacency(src)
n_src = adjacency.shape[0]
print(f"Source space: {n_src} vertices")

n_subjects, n_times = 15, 15
n_tests = n_src * n_times
print(f"Tests: {n_tests} ({n_src} vertices × {n_times} times)")
rng = np.random.default_rng(42)
X = rng.standard_normal((n_subjects, n_times, n_src))
sig_verts = np.arange(100, 200)
sig_times = np.arange(5, 10)
X[:, sig_times[:, None], sig_verts[None, :]] += 2.0

threshold = 1.67
seed = 42


# ── Micro-benchmark: _setup_adjacency CSR deduplication ──────────────────
print("\n--- Micro-benchmark: _setup_adjacency ---")

# Current v14: returns (adjacency, csr_data)
times_v14_setup = []
for _ in range(20):
    adj_copy = adjacency.copy()
    t0 = time.perf_counter()
    adj_list, csr_data = _setup_adjacency(adj_copy, n_tests, n_times)
    times_v14_setup.append(time.perf_counter() - t0)
med_setup = np.median(times_v14_setup) * 1000
print(f"  _setup_adjacency: {med_setup:.1f}ms (returns list + CSR)")
print(f"  CSR data: indptr({csr_data[0].shape}), indices({csr_data[1].shape}), n_src={csr_data[2]}")


# ── Micro-benchmark: CSR rebuild cost (what v14 avoids) ──────────────────
print("\n--- Micro-benchmark: CSR rebuild from list (what v14 avoids) ---")

times_csr_rebuild = []
for _ in range(20):
    t0 = time.perf_counter()
    _lengths = np.array([len(a) for a in adj_list])
    _indptr = np.zeros(len(adj_list) + 1, dtype=np.intp)
    np.cumsum(_lengths, out=_indptr[1:])
    _indices = np.concatenate(adj_list).astype(np.intp)
    times_csr_rebuild.append(time.perf_counter() - t0)
med_rebuild = np.median(times_csr_rebuild) * 1000
print(f"  CSR rebuild from list: {med_rebuild:.1f}ms")
print(f"  v14 avoids this 2x (in _find_clusters_1dir + _do_1samp_permutations)")
print(f"  Estimated savings: {2 * med_rebuild:.1f}ms")

# Verify CSR data matches rebuild
assert np.array_equal(csr_data[0], _indptr), "indptr mismatch"
assert np.array_equal(csr_data[1], _indices), "indices mismatch"
assert csr_data[2] == len(adj_list), "n_src mismatch"
print("  CSR parity: PASS")


# ── Micro-benchmark: _get_1samp_orders vectorization ────────────────────
print("\n--- Micro-benchmark: _get_1samp_orders ---")

# Old way: list comprehension with np.fromiter(np.binary_repr(...))
def _get_1samp_orders_old(n_samples, n_permutations, tail, rng):
    max_perms = 2 ** (n_samples - (tail == 0)) - 1
    extra = ""
    n_permutations = int(n_permutations)
    if n_samples <= 20:
        orders = rng.choice(max_perms, n_permutations - 1, replace=False)
        orders = [
            np.fromiter(np.binary_repr(s + 1, n_samples), dtype=int) for s in orders
        ]
    return orders, n_permutations, extra


n_perms_test = 2048
times_old_orders = []
for _ in range(20):
    _rng = np.random.default_rng(42)
    t0 = time.perf_counter()
    orders_old, _, _ = _get_1samp_orders_old(n_subjects, n_perms_test, 1, _rng)
    times_old_orders.append(time.perf_counter() - t0)
med_old_orders = np.median(times_old_orders) * 1000

times_new_orders = []
for _ in range(20):
    _rng = np.random.default_rng(42)
    t0 = time.perf_counter()
    orders_new, _, _ = _get_1samp_orders(n_subjects, n_perms_test, 1, _rng)
    times_new_orders.append(time.perf_counter() - t0)
med_new_orders = np.median(times_new_orders) * 1000

# Parity check
old_arr = np.array(orders_old) if isinstance(orders_old, list) else orders_old
new_arr = np.array(orders_new) if isinstance(orders_new, list) else orders_new
orders_parity = "PASS" if np.array_equal(old_arr, new_arr) else "FAIL"
print(f"  Old (list comprehension): {med_old_orders:.2f}ms")
print(f"  New (vectorized):         {med_new_orders:.2f}ms")
print(f"  Speedup: {med_old_orders/med_new_orders:.1f}x, parity={orders_parity}")


# ── Micro-benchmark: _find_clusters with/without CSR data ───────────────
print("\n--- Micro-benchmark: initial clustering with/without precomputed CSR ---")

X_2d = X.reshape(n_subjects, n_tests)
t_obs = ttest_1samp_no_p(X_2d)

# With precomputed CSR (v14)
times_with_csr = []
for _ in range(20):
    t0 = time.perf_counter()
    clusters_v14, sums_v14 = _find_clusters(
        t_obs, threshold, 1, adj_list, max_step=1,
        include=None, partitions=None, t_power=1, show_info=False,
        _csr_data=csr_data,
    )
    times_with_csr.append(time.perf_counter() - t0)
med_with_csr = np.median(times_with_csr) * 1000

# Without precomputed CSR (v13 behavior)
times_without_csr = []
for _ in range(20):
    t0 = time.perf_counter()
    clusters_v13, sums_v13 = _find_clusters(
        t_obs, threshold, 1, adj_list, max_step=1,
        include=None, partitions=None, t_power=1, show_info=False,
        _csr_data=None,
    )
    times_without_csr.append(time.perf_counter() - t0)
med_without_csr = np.median(times_without_csr) * 1000

init_parity = "PASS" if (
    len(clusters_v14) == len(clusters_v13) and
    np.allclose(sums_v14, sums_v13)
) else "FAIL"
print(f"  Without CSR (v13): {med_without_csr:.1f}ms ({len(clusters_v13)} clusters)")
print(f"  With CSR (v14):    {med_with_csr:.1f}ms ({len(clusters_v14)} clusters)")
print(f"  Savings: {med_without_csr - med_with_csr:.1f}ms, parity={init_parity}")


# ── End-to-end A/B benchmark ───────────────────────────────────────────────
print("\n--- End-to-end A/B benchmark ---")
print(f"Sweeping n_perms = [256, 512, 1024, 2048, 4096]")
print(f"Runs: 5 each, median wall time\n")

import mne.stats.cluster_level as cl

perm_counts = [256, 512, 1024, 2048, 4096]
results_v14 = {}
results_v13 = {}

# Save original functions
_orig_setup_adjacency = cl._setup_adjacency
_orig_get_1samp_orders = cl._get_1samp_orders


def _v13_setup_adjacency(adjacency, n_tests, n_times):
    """v13 version: return adjacency with None CSR data to force rebuilds."""
    result, _ = _orig_setup_adjacency(adjacency, n_tests, n_times)
    return result, None  # Return tuple but CSR is None → forces rebuild


def _v13_get_1samp_orders(n_samples, n_permutations, tail, rng):
    """v13 version: use list comprehension for binary_repr."""
    max_perms = 2 ** (n_samples - (tail == 0)) - 1
    extra = ""
    if isinstance(n_permutations, str):
        if n_permutations != "all":
            raise ValueError('n_permutations as a string must be "all"')
        n_permutations = max_perms
    n_permutations = int(n_permutations)
    if max_perms < n_permutations:
        extra = " (exact test)"
        orders = cl.bin_perm_rep(n_samples)[1 : max_perms + 1]
    elif n_samples <= 20:
        orders = rng.choice(max_perms, n_permutations - 1, replace=False)
        orders = [
            np.fromiter(np.binary_repr(s + 1, n_samples), dtype=int) for s in orders
        ]
    else:
        orders = np.zeros((n_permutations - 1, n_samples), int)
        hashes = {}
        ii = 0
        use_samples = n_samples - (tail == 0)
        while ii < n_permutations - 1:
            signs = tuple((rng.uniform(size=use_samples) < 0.5).astype(int))
            if signs not in hashes:
                orders[ii, :use_samples] = signs
                if tail == 0 and rng.uniform() < 0.5:
                    orders[ii] = 1 - orders[ii]
                hashes[signs] = None
                ii += 1
    return orders, n_permutations, extra


for n_perms in perm_counts:
    # ── v14 (current code) ──
    times_v14 = []
    for run in range(5):
        t0 = time.perf_counter()
        t_obs_v14, clusters_v14e, pv_v14, H0_v14 = (
            spatio_temporal_cluster_1samp_test(
                X,
                threshold=threshold,
                adjacency=adjacency,
                n_permutations=n_perms,
                tail=1,
                seed=seed,
                out_type="indices",
                verbose=False,
            )
        )
        times_v14.append(time.perf_counter() - t0)
    med_v14 = np.median(times_v14)
    results_v14[n_perms] = {
        "median": med_v14,
        "t_obs": t_obs_v14,
        "H0": H0_v14.copy(),
        "pv": pv_v14.copy(),
        "n_clusters": len(clusters_v14e),
    }

    # ── v13 (monkeypatch: force old behavior) ──
    # Force _setup_adjacency to return None CSR → forces downstream rebuilds
    cl._setup_adjacency = _v13_setup_adjacency
    # Force _get_1samp_orders to use old list comprehension
    cl._get_1samp_orders = _v13_get_1samp_orders

    times_v13 = []
    for run in range(5):
        t0 = time.perf_counter()
        t_obs_v13, clusters_v13e, pv_v13, H0_v13 = (
            spatio_temporal_cluster_1samp_test(
                X,
                threshold=threshold,
                adjacency=adjacency,
                n_permutations=n_perms,
                tail=1,
                seed=seed,
                out_type="indices",
                verbose=False,
            )
        )
        times_v13.append(time.perf_counter() - t0)
    med_v13 = np.median(times_v13)
    results_v13[n_perms] = {
        "median": med_v13,
        "t_obs": t_obs_v13,
        "H0": H0_v13.copy(),
        "pv": pv_v13.copy(),
        "n_clusters": len(clusters_v13e),
    }

    # Restore
    cl._setup_adjacency = _orig_setup_adjacency
    cl._get_1samp_orders = _orig_get_1samp_orders

    # Parity check
    t_match = np.allclose(t_obs_v14, t_obs_v13, atol=1e-10)
    h0_match = np.allclose(H0_v14, H0_v13, atol=1e-10)
    pv_match = np.allclose(pv_v14, pv_v13, atol=1e-10)
    nc_match = len(clusters_v14e) == len(clusters_v13e)
    all_pass = t_match and h0_match and pv_match and nc_match
    parity = "ALL PASS" if all_pass else "FAIL"

    print(
        f"  n_perms={n_perms:5d}: v14={med_v14:.3f}s  v13={med_v13:.3f}s  "
        f"speedup={med_v13/med_v14:.3f}x  parity={parity}"
    )


# ── Linear regression ───────────────────────────────────────────────────────
print("\n--- Linear regression (overhead + slope * n_perms) ---")
from numpy.polynomial.polynomial import polyfit

perms_arr = np.array(perm_counts, dtype=np.float64)
for label, results in [("v14", results_v14), ("v13", results_v13)]:
    medians = np.array([results[p]["median"] for p in perm_counts])
    c = polyfit(perms_arr, medians, 1)
    overhead_ms = c[0] * 1000
    slope_ms = c[1] * 1000
    print(f"  {label}: overhead={overhead_ms:.1f}ms, slope={slope_ms:.3f}ms/perm")

v14_meds = np.array([results_v14[p]["median"] for p in perm_counts])
v13_meds = np.array([results_v13[p]["median"] for p in perm_counts])
c14 = polyfit(perms_arr, v14_meds, 1)
c13 = polyfit(perms_arr, v13_meds, 1)
overhead_saved_ms = (c13[0] - c14[0]) * 1000
print(f"\n  Overhead saved: {overhead_saved_ms:.1f}ms")
print(f"  Per-perm slope diff: {(c13[1] - c14[1])*1000:.3f}ms (should be ~0)")


# ── Multi-tail parity ─────────────────────────────────────────────────────
print("\n--- Multi-tail parity (2048 perms) ---")
for tail in [1, 0, -1]:
    thresh = threshold if tail >= 0 else -threshold
    t_v14, c_v14, pv_v14_mt, h0_v14_mt = spatio_temporal_cluster_1samp_test(
        X,
        threshold=thresh,
        adjacency=adjacency,
        n_permutations=2048,
        tail=tail,
        seed=seed,
        out_type="indices",
        verbose=False,
    )
    # v13 (old behavior)
    cl._setup_adjacency = _v13_setup_adjacency
    cl._get_1samp_orders = _v13_get_1samp_orders
    t_v13, c_v13, pv_v13_mt, h0_v13_mt = spatio_temporal_cluster_1samp_test(
        X,
        threshold=thresh,
        adjacency=adjacency,
        n_permutations=2048,
        tail=tail,
        seed=seed,
        out_type="indices",
        verbose=False,
    )
    cl._setup_adjacency = _orig_setup_adjacency
    cl._get_1samp_orders = _orig_get_1samp_orders

    t_ok = np.allclose(t_v14, t_v13, atol=1e-10)
    h0_ok = np.allclose(h0_v14_mt, h0_v13_mt, atol=1e-10)
    pv_ok = np.allclose(pv_v14_mt, pv_v13_mt, atol=1e-10)
    status = "ALL PASS" if (t_ok and h0_ok and pv_ok) else "FAIL"
    print(
        f"  tail={tail:+d}: t_obs={'PASS' if t_ok else 'FAIL'}, "
        f"H0={'PASS' if h0_ok else 'FAIL'}, "
        f"p-values={'PASS' if pv_ok else 'FAIL'} → {status}"
    )


print("\n" + "=" * 70)
print("Done.")
