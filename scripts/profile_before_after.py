"""Compare before/after JIT for both functions."""
import time
import sys
import numpy as np

sys.path.insert(0, "/Users/sharif/Code/mne-python")
import mne
from mne.forward._compute_forward import (
    _eeg_spherepot_coil, _do_eeg_spherepot,
    _prep_field_computation, _compute_forwards,
    _concatenate_coils,
)
from mne.forward._lead_dots import _get_legen_der, _get_legen_table
from mne.forward._make_forward import _prepare_for_forward
from mne.transforms import _get_trans
from copy import deepcopy

data_path = mne.datasets.sample.data_path()
raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
trans_fname = data_path / "MEG" / "sample" / "sample_audvis_raw-trans.fif"
src_fname = data_path / "subjects" / "sample" / "bem" / "sample-oct-6-src.fif"

info = mne.io.read_info(raw_fname)
src = mne.read_source_spaces(src_fname)
trans = mne.read_trans(trans_fname)
sphere = mne.make_sphere_model(r0="auto", head_radius="auto", info=info)
mri_head_t, _ = _get_trans(trans)

sensors, rr, info_used, update_kwargs, sphere_used = _prepare_for_forward(
    src, mri_head_t, info, sphere, 0.0, 1,
    bem_extra="sphere", trans=str(trans_fname), info_extra="info",
    meg=False, eeg=True, ignore_ref=False, on_inside="raise",
)

print(f"Source locations: {len(rr)}")

# ============================================================================
# Benchmark _eeg_spherepot_coil (current, JIT-compiled version)
# ============================================================================
sensors2 = deepcopy(sensors)
fwd_data = _prep_field_computation(sensors=sensors2, bem=sphere_used, n_jobs=1)
coils = sensors2['eeg']['defs']

# Warmup JIT
_ = _eeg_spherepot_coil(rr[:10], coils, sphere_used)

times_jit = []
for _ in range(5):
    t0 = time.perf_counter()
    result = _eeg_spherepot_coil(rr, coils, sphere_used)
    times_jit.append(time.perf_counter() - t0)

print(f"\n_eeg_spherepot_coil (JIT version, {len(rr)} sources):")
print(f"  Times: {[f'{t:.4f}' for t in times_jit]}")
print(f"  Mean: {np.mean(times_jit):.4f} sec")
print(f"  Min: {np.min(times_jit):.4f} sec")

# ============================================================================
# Benchmark _get_legen_der (current, JIT-compiled version)
# ============================================================================
xx = np.linspace(-1, 1, 20001)

# Warmup JIT
_ = _get_legen_der(xx[:100], n_coeff=100)

times_jit_legen = []
for _ in range(5):
    t0 = time.perf_counter()
    result = _get_legen_der(xx, n_coeff=100)
    times_jit_legen.append(time.perf_counter() - t0)

print(f"\n_get_legen_der (JIT version, 20001 pts):")
print(f"  Times: {[f'{t:.4f}' for t in times_jit_legen]}")
print(f"  Mean: {np.mean(times_jit_legen):.4f} sec")
print(f"  Min: {np.min(times_jit_legen):.4f} sec")

# ============================================================================
# Benchmark full forward with sphere model (current version)
# ============================================================================
times_full = []
for _ in range(3):
    sensors3 = deepcopy(sensors)
    t0 = time.perf_counter()
    fwds = _compute_forwards(rr, bem=sphere_used, sensors=sensors3, n_jobs=1)
    times_full.append(time.perf_counter() - t0)

print(f"\n_compute_forwards (sphere, EEG, {len(rr)} sources):")
print(f"  Times: {[f'{t:.4f}' for t in times_full]}")
print(f"  Mean: {np.mean(times_full):.4f} sec")

# ============================================================================
# Benchmark _get_legen_table (full pipeline including cache)
# ============================================================================
times_table_cached = []
for _ in range(3):
    t0 = time.perf_counter()
    lut, n_fact = _get_legen_table("meg", force_calc=False)
    times_table_cached.append(time.perf_counter() - t0)

times_table_compute = []
for _ in range(3):
    t0 = time.perf_counter()
    lut, n_fact = _get_legen_table("meg", force_calc=True)
    times_table_compute.append(time.perf_counter() - t0)

print(f"\n_get_legen_table('meg') from cache:")
print(f"  Times: {[f'{t:.4f}' for t in times_table_cached]}")
print(f"  Mean: {np.mean(times_table_cached):.4f} sec")
print(f"\n_get_legen_table('meg') force compute:")
print(f"  Times: {[f'{t:.4f}' for t in times_table_compute]}")
print(f"  Mean: {np.mean(times_table_compute):.4f} sec")

print("\n" + "=" * 70)
print("IMPACT ANALYSIS")
print("=" * 70)
eeg_fwd_time = np.mean(times_jit)
eeg_fwd_full = np.mean(times_full)
print(f"\n_eeg_spherepot_coil:")
print(f"  Current (JIT): {eeg_fwd_time:.4f} sec")
print(f"  If 6.4x slower (no JIT): ~{eeg_fwd_time * 6.4:.4f} sec")
print(f"  Time saved by JIT: ~{eeg_fwd_time * 5.4:.4f} sec")
print(f"  Fraction of _compute_forwards: {eeg_fwd_time/eeg_fwd_full*100:.1f}%")
print(f"  Forward total without JIT: ~{eeg_fwd_full + eeg_fwd_time*5.4:.4f} sec")
print(f"  End-to-end speedup from JIT: {(eeg_fwd_full + eeg_fwd_time*5.4)/eeg_fwd_full:.2f}x -> 1x")

legen_compute_time = np.mean(times_jit_legen)
print(f"\n_get_legen_der:")
print(f"  Current (JIT): {legen_compute_time:.4f} sec")
print(f"  Called once per session and cached to disk")
print(f"  Even without JIT: <0.1 sec overhead, once ever")
