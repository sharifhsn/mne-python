"""Profile the EEG sphere model forward solution to understand
where time is actually spent."""

import time
import cProfile
import pstats
import io
import sys
import numpy as np

# Make sure we use the local mne
sys.path.insert(0, "/Users/sharif/Code/mne-python")
import mne

print("MNE version:", mne.__version__)
print("=" * 70)

# ============================================================================
# Part 1: Setup - Load sample data
# ============================================================================
data_path = mne.datasets.sample.data_path()
subjects_dir = data_path / "subjects"
raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
trans_fname = data_path / "MEG" / "sample" / "sample_audvis_raw-trans.fif"
src_fname = data_path / "subjects" / "sample" / "bem" / "sample-oct-6-src.fif"
bem_fname = data_path / "subjects" / "sample" / "bem" / "sample-5120-5120-5120-bem-sol.fif"

info = mne.io.read_info(raw_fname)
src = mne.read_source_spaces(src_fname)
trans = mne.read_trans(trans_fname)
bem = mne.read_bem_solution(bem_fname)

# Count source points
n_src = sum(s["nuse"] for s in src)
print(f"\nNumber of source points in oct-6 source space: {n_src}")

# ============================================================================
# Part 2: Create a sphere model
# ============================================================================
sphere = mne.make_sphere_model(r0="auto", head_radius="auto", info=info)
print(f"\nSphere model: {sphere}")
print(f"Sphere model is_sphere: {sphere['is_sphere']}")
print(f"Number of layers: {len(sphere.get('layers', []))}")

# ============================================================================
# Part 3: Time the BEM forward solution (for comparison)
# ============================================================================
print("\n" + "=" * 70)
print("TIMING: BEM forward solution (EEG only)")
print("=" * 70)

t0 = time.perf_counter()
fwd_bem = mne.make_forward_solution(
    info, trans=trans, src=src, bem=bem,
    meg=False, eeg=True, n_jobs=1
)
t_bem = time.perf_counter() - t0
print(f"BEM forward solution time: {t_bem:.2f} seconds")
print(f"BEM forward shape: {fwd_bem['sol']['data'].shape}")

# ============================================================================
# Part 4: Time the sphere model forward solution
# ============================================================================
print("\n" + "=" * 70)
print("TIMING: Sphere model forward solution (EEG only)")
print("=" * 70)

t0 = time.perf_counter()
fwd_sphere = mne.make_forward_solution(
    info, trans=trans, src=src, bem=sphere,
    meg=False, eeg=True, n_jobs=1
)
t_sphere = time.perf_counter() - t0
print(f"Sphere model forward solution time: {t_sphere:.2f} seconds")
print(f"Sphere forward shape: {fwd_sphere['sol']['data'].shape}")

# ============================================================================
# Part 5: Profile the sphere model forward solution in detail
# ============================================================================
print("\n" + "=" * 70)
print("PROFILING: Sphere model forward solution (EEG only)")
print("=" * 70)

pr = cProfile.Profile()
pr.enable()
fwd_sphere2 = mne.make_forward_solution(
    info, trans=trans, src=src, bem=sphere,
    meg=False, eeg=True, n_jobs=1
)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(30)
print(s.getvalue())

# Also show by tottime
s2 = io.StringIO()
ps2 = pstats.Stats(pr, stream=s2).sort_stats("tottime")
ps2.print_stats(30)
print("\n--- Sorted by total time ---")
print(s2.getvalue())

# ============================================================================
# Part 6: Profile sphere model forward with MEG too
# ============================================================================
print("\n" + "=" * 70)
print("TIMING: Sphere model forward solution (MEG + EEG)")
print("=" * 70)

t0 = time.perf_counter()
fwd_sphere_meeg = mne.make_forward_solution(
    info, trans=trans, src=src, bem=sphere,
    meg=True, eeg=True, n_jobs=1
)
t_sphere_meeg = time.perf_counter() - t0
print(f"Sphere model forward (MEG+EEG) time: {t_sphere_meeg:.2f} seconds")

# ============================================================================
# Part 7: Profile the specific functions
# ============================================================================
print("\n" + "=" * 70)
print("DIRECT TIMING: _eeg_spherepot_coil and _do_eeg_spherepot")
print("=" * 70)

from mne.forward._compute_forward import (
    _eeg_spherepot_coil, _do_eeg_spherepot,
    _prep_field_computation, _compute_forwards_meeg, _compute_forwards,
    _concatenate_coils,
)
from mne.forward._make_forward import _prepare_for_forward

# Get the rr and setup exactly as make_forward_solution does
mri_head_t, trans_obj = mne._freesurfer.read_trans(trans_fname) if hasattr(mne, '_freesurfer') else (None, None)

# Re-do the preparation
from mne.transforms import _get_trans
mri_head_t, _ = _get_trans(trans)

# We need to replicate the setup
sensors, rr, info_used, update_kwargs, sphere_used = mne.forward._make_forward._prepare_for_forward(
    src, mri_head_t, info, sphere, 0.0, 1,
    bem_extra="sphere", trans=str(trans_fname), info_extra="info",
    meg=False, eeg=True, ignore_ref=False, on_inside="raise",
)

print(f"Number of source locations (rr): {len(rr)}")
print(f"EEG coils/electrodes: {len(sensors['eeg']['defs'])}")

# Now time _compute_forwards with sphere
t0 = time.perf_counter()
fwds = _compute_forwards(rr, bem=sphere_used, sensors=sensors, n_jobs=1)
t_compute = time.perf_counter() - t0
print(f"_compute_forwards time: {t_compute:.2f} seconds")

# Time just the _eeg_spherepot_coil part
from copy import deepcopy
sensors2 = deepcopy(sensors)
fwd_data = _prep_field_computation(sensors=sensors2, bem=sphere_used, n_jobs=1)

t0 = time.perf_counter()
Bs = _compute_forwards_meeg(rr, sensors=sensors2, fwd_data=fwd_data, n_jobs=1)
t_meeg = time.perf_counter() - t0
print(f"_compute_forwards_meeg time: {t_meeg:.2f} seconds")

# Time individual call to _eeg_spherepot_coil with a subset
coils = sensors2['eeg']['defs']
print(f"\nCoils type: {type(coils)}")

# Single call with 100 source points
subset_rr = rr[:100]
t0 = time.perf_counter()
result = _eeg_spherepot_coil(subset_rr, coils, sphere_used)
t_single = time.perf_counter() - t0
print(f"_eeg_spherepot_coil for 100 source points: {t_single:.4f} seconds")
print(f"Estimated for {len(rr)} source points: {t_single * len(rr) / 100:.2f} seconds")

# ============================================================================
# Part 8: Understand the Legendre table caching
# ============================================================================
print("\n" + "=" * 70)
print("LEGENDRE TABLE CACHING")
print("=" * 70)

from mne.forward._lead_dots import _get_legen_table, _get_legen_der

# Check if the cached file exists
import os
extra_data_path = mne.utils._get_extra_data_path()
tables_dir = os.path.join(extra_data_path, "tables")
print(f"Tables directory: {tables_dir}")
if os.path.isdir(tables_dir):
    for f in os.listdir(tables_dir):
        fpath = os.path.join(tables_dir, f)
        size = os.path.getsize(fpath)
        print(f"  {f}: {size / 1024:.1f} KB")
else:
    print("  No tables directory found")

# Time reading from cache vs computing
t0 = time.perf_counter()
lut, n_fact = _get_legen_table("meg", force_calc=False)
t_cached = time.perf_counter() - t0
print(f"\nReading cached Legendre derivative table: {t_cached:.4f} seconds")
print(f"LUT shape: {lut.shape}")

t0 = time.perf_counter()
lut2, n_fact2 = _get_legen_table("meg", force_calc=True)
t_computed = time.perf_counter() - t0
print(f"Computing Legendre derivative table: {t_computed:.4f} seconds")

t0 = time.perf_counter()
lut3, n_fact3 = _get_legen_table("eeg", force_calc=False)
t_eeg_cached = time.perf_counter() - t0
print(f"\nReading cached EEG Legendre table: {t_eeg_cached:.4f} seconds")
print(f"EEG LUT shape: {lut3.shape}")

t0 = time.perf_counter()
lut4, n_fact4 = _get_legen_table("eeg", force_calc=True)
t_eeg_computed = time.perf_counter() - t0
print(f"Computing EEG Legendre table: {t_eeg_computed:.4f} seconds")

# Time _get_legen_der directly
xx = np.linspace(-1, 1, 20001)
t0 = time.perf_counter()
result = _get_legen_der(xx, n_coeff=100)
t_legen_der = time.perf_counter() - t0
print(f"\n_get_legen_der for 20001 points: {t_legen_der:.4f} seconds")
print(f"Result shape: {result.shape}")

# ============================================================================
# Part 9: Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Source points: {len(rr)}")
print(f"EEG channels: {fwd_sphere['sol']['data'].shape[0]}")
print(f"")
print(f"BEM forward (EEG only): {t_bem:.2f} seconds")
print(f"Sphere forward (EEG only): {t_sphere:.2f} seconds")
print(f"Sphere forward (MEG+EEG): {t_sphere_meeg:.2f} seconds")
print(f"")
print(f"Of sphere forward, _compute_forwards: {t_compute:.2f} seconds")
print(f"Of sphere forward, _compute_forwards_meeg: {t_meeg:.2f} seconds")
print(f"")
print(f"Legendre table read (cached): {t_cached:.4f} seconds")
print(f"Legendre table compute: {t_computed:.4f} seconds")
print(f"_get_legen_der (20001 pts): {t_legen_der:.4f} seconds")
