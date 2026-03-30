"""Profile Part 2: Direct timing of _eeg_spherepot_coil and Legendre tables."""

import time
import os
import sys
import numpy as np

sys.path.insert(0, "/Users/sharif/Code/mne-python")
import mne
from mne.forward._compute_forward import (
    _eeg_spherepot_coil, _do_eeg_spherepot,
    _prep_field_computation, _compute_forwards_meeg, _compute_forwards,
    _concatenate_coils,
)
from mne.forward._make_forward import _prepare_for_forward
from mne.transforms import _get_trans
from copy import deepcopy

print("MNE version:", mne.__version__)

# Setup
data_path = mne.datasets.sample.data_path()
raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
trans_fname = data_path / "MEG" / "sample" / "sample_audvis_raw-trans.fif"
src_fname = data_path / "subjects" / "sample" / "bem" / "sample-oct-6-src.fif"

info = mne.io.read_info(raw_fname)
src = mne.read_source_spaces(src_fname)
trans = mne.read_trans(trans_fname)
sphere = mne.make_sphere_model(r0="auto", head_radius="auto", info=info)

mri_head_t, _ = _get_trans(trans)

# Prepare forward
sensors, rr, info_used, update_kwargs, sphere_used = _prepare_for_forward(
    src, mri_head_t, info, sphere, 0.0, 1,
    bem_extra="sphere", trans=str(trans_fname), info_extra="info",
    meg=False, eeg=True, ignore_ref=False, on_inside="raise",
)

print(f"\nNumber of source locations (rr): {len(rr)}")
print(f"EEG coils/electrodes: {len(sensors['eeg']['defs'])}")

# ============================================================================
# Time _compute_forwards with sphere
# ============================================================================
print("\n" + "=" * 70)
print("TIMING: _compute_forwards (sphere, EEG only)")
print("=" * 70)

t0 = time.perf_counter()
fwds = _compute_forwards(rr, bem=sphere_used, sensors=deepcopy(sensors), n_jobs=1)
t_compute = time.perf_counter() - t0
print(f"_compute_forwards time: {t_compute:.3f} seconds")

# ============================================================================
# Time _compute_forwards_meeg
# ============================================================================
print("\n" + "=" * 70)
print("TIMING: _compute_forwards_meeg (sphere, EEG only)")
print("=" * 70)

sensors2 = deepcopy(sensors)
fwd_data = _prep_field_computation(sensors=sensors2, bem=sphere_used, n_jobs=1)

t0 = time.perf_counter()
Bs = _compute_forwards_meeg(rr, sensors=sensors2, fwd_data=fwd_data, n_jobs=1)
t_meeg = time.perf_counter() - t0
print(f"_compute_forwards_meeg time: {t_meeg:.3f} seconds")

# ============================================================================
# Time _eeg_spherepot_coil for various chunk sizes
# ============================================================================
print("\n" + "=" * 70)
print("TIMING: _eeg_spherepot_coil for various source counts")
print("=" * 70)

coils = sensors2['eeg']['defs']
print(f"Coils type: {type(coils)}, length: {len(coils) if isinstance(coils, (list, tuple)) else 'tuple'}")

for n_src in [10, 50, 100, 500, 1000, len(rr)]:
    subset_rr = rr[:n_src]
    t0 = time.perf_counter()
    result = _eeg_spherepot_coil(subset_rr, coils, sphere_used)
    t_single = time.perf_counter() - t0
    print(f"  {n_src:5d} sources: {t_single:.4f} sec (shape: {result.shape})")

# ============================================================================
# Time _do_eeg_spherepot directly
# ============================================================================
print("\n" + "=" * 70)
print("TIMING: _do_eeg_spherepot internals")
print("=" * 70)

rmags, cosmags, ws, bins = coils if isinstance(coils, tuple) else _concatenate_coils(coils)
n_coils = bins[-1] + 1
r0 = sphere_used["r0"]
inner_rad = sphere_used["layers"][0]["rad"]
mu = sphere_used["mu"]
lambda_ = sphere_used["lambda"]
nfit = sphere_used["nfit"]

print(f"rmags shape: {rmags.shape}")
print(f"n_coils: {n_coils}")
print(f"inner_rad: {inner_rad}")
print(f"nfit (number of equiv sources): {nfit}")
print(f"mu: {mu}")
print(f"lambda_: {lambda_}")

for n_src in [100, 500, 1000, len(rr)]:
    subset_rr = rr[:n_src]
    t0 = time.perf_counter()
    result = _do_eeg_spherepot(subset_rr, rmags, ws, bins, n_coils, r0, inner_rad, mu, lambda_, nfit)
    t = time.perf_counter() - t0
    print(f"  _do_eeg_spherepot for {n_src:5d} sources: {t:.4f} sec")

# ============================================================================
# Legendre table analysis
# ============================================================================
print("\n" + "=" * 70)
print("LEGENDRE TABLE ANALYSIS")
print("=" * 70)

from mne.forward._lead_dots import _get_legen_table, _get_legen_der

# Check cached files
extra_data_path = mne.utils._get_extra_data_path()
tables_dir = os.path.join(extra_data_path, "tables")
print(f"Tables directory: {tables_dir}")
if os.path.isdir(tables_dir):
    for f in sorted(os.listdir(tables_dir)):
        fpath = os.path.join(tables_dir, f)
        size = os.path.getsize(fpath)
        print(f"  {f}: {size / 1024:.1f} KB")

# Time cached vs computed
t0 = time.perf_counter()
lut, n_fact = _get_legen_table("meg", force_calc=False)
t_cached = time.perf_counter() - t0
print(f"\nMEG Legendre table read (cached): {t_cached:.4f} sec, shape: {lut.shape}")

t0 = time.perf_counter()
lut2, n_fact2 = _get_legen_table("meg", force_calc=True)
t_computed = time.perf_counter() - t0
print(f"MEG Legendre table compute: {t_computed:.4f} sec")

t0 = time.perf_counter()
lut3, n_fact3 = _get_legen_table("eeg", force_calc=False)
t_eeg_cached = time.perf_counter() - t0
print(f"\nEEG Legendre table read (cached): {t_eeg_cached:.4f} sec, shape: {lut3.shape}")

t0 = time.perf_counter()
lut4, n_fact4 = _get_legen_table("eeg", force_calc=True)
t_eeg_computed = time.perf_counter() - t0
print(f"EEG Legendre table compute: {t_eeg_computed:.4f} sec")

# Time _get_legen_der directly
xx = np.linspace(-1, 1, 20001)
t0 = time.perf_counter()
result = _get_legen_der(xx, n_coeff=100)
t_legen_der = time.perf_counter() - t0
print(f"\n_get_legen_der (20001 pts, 100 coeffs): {t_legen_der:.4f} sec, shape: {result.shape}")

# When is _get_legen_table actually called during forward computation?
print("\n" + "=" * 70)
print("WHEN IS _get_legen_table/_get_legen_der CALLED?")
print("=" * 70)
print("_get_legen_der is called ONLY from _get_legen_table (for MEG Legendre tables)")
print("_get_legen_table is called from:")
print("  1. _setup_dots() in _field_interpolation.py - used by make_field_map()")
print("  2. That's it for production code.")
print("")
print("_get_legen_table is NOT called during make_forward_solution()!")
print("_get_legen_table is NOT called during fit_dipole()!")
print("")
print("The sphere forward (make_forward_solution with sphere model) uses")
print("_eeg_spherepot_coil -> _do_eeg_spherepot which does NOT use Legendre tables.")
print("It uses the equivalent source approach (Berg & Scherg parameters).")

# ============================================================================
# Profile _sphere_pot_or_field path
# ============================================================================
print("\n" + "=" * 70)
print("FULL BREAKDOWN: make_forward_solution with sphere")
print("=" * 70)

import cProfile
import pstats
import io

# Profile just the computation phase
sensors3 = deepcopy(sensors)

pr = cProfile.Profile()
pr.enable()
fwds = _compute_forwards(rr, bem=sphere_used, sensors=sensors3, n_jobs=1)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
ps.print_stats(20)
print(s.getvalue())

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"Source points used: {len(rr)}")
print(f"EEG channels: {n_coils}")
print(f"Equivalent sources (nfit): {nfit}")
print(f"")
print(f"_compute_forwards (sphere, EEG): {t_compute:.3f} sec")
print(f"  -> _compute_forwards_meeg: {t_meeg:.3f} sec")
print(f"     -> _eeg_spherepot_coil ({len(rr)} src): ~{t_meeg:.3f} sec")
print(f"")
print(f"For comparison, BEM forward (EEG) from part 1 was ~9 sec")
print(f"Sphere forward is already {9.0/t_compute:.1f}x faster than BEM")
print(f"")
print(f"_get_legen_der is used for Legendre TABLES (field interpolation),")
print(f"NOT for the sphere forward solution itself.")
print(f"Legendre table compute: {t_computed:.4f} sec")
print(f"Legendre table read (cached): {t_cached:.4f} sec")
print(f"The table is computed ONCE and then cached to disk forever.")
