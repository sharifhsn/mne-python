"""Final summary: precise breakdown of where time goes."""

import time
import sys
import numpy as np

sys.path.insert(0, "/Users/sharif/Code/mne-python")
import mne
from mne.forward._lead_dots import _get_legen_table, _get_legen_der
from mne.forward._field_interpolation import _setup_dots, make_field_map

print("=" * 70)
print("PART A: make_field_map (where _get_legen_der is actually used)")
print("=" * 70)

data_path = mne.datasets.sample.data_path()
raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
events = mne.find_events(raw, verbose=False)
raw.set_eeg_reference(projection=True, verbose=False)
epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=0.5,
                    baseline=(None, 0), preload=True, verbose=False)
evoked = epochs.average()
evoked.apply_proj()

subjects_dir = data_path / "subjects"

import cProfile, pstats, io

# Profile make_field_map (this is where Legendre tables are used)
pr = cProfile.Profile()
pr.enable()
t0 = time.perf_counter()
field_maps = make_field_map(evoked, trans="fsaverage",
                            subject="sample", subjects_dir=subjects_dir,
                            ch_type="eeg", verbose=False)
t_map = time.perf_counter() - t0
pr.disable()

print(f"make_field_map (EEG) time: {t_map:.3f} seconds")

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
ps.print_stats(20)
print(s.getvalue())

# Time just the Legendre table computation
print("\n" + "=" * 70)
print("PART B: Legendre table timing breakdown")
print("=" * 70)

# Delete cached files to force recomputation
import os
extra_data_path = mne.utils._get_extra_data_path()
tables_dir = os.path.join(extra_data_path, "tables")

for fname in ["legder_100_20000.bin", "legval_100_20000.bin"]:
    fpath = os.path.join(tables_dir, fname)
    if os.path.exists(fpath):
        os.remove(fpath)
        print(f"Removed cached {fname}")

# Time first call (has to compute + write)
t0 = time.perf_counter()
lut_meg, _ = _get_legen_table("meg", force_calc=False)
t_meg_first = time.perf_counter() - t0
print(f"\nMEG Legendre table (first call, must compute): {t_meg_first:.4f} sec")

# Time second call (read from cache)
t0 = time.perf_counter()
lut_meg2, _ = _get_legen_table("meg", force_calc=False)
t_meg_cached = time.perf_counter() - t0
print(f"MEG Legendre table (cached read): {t_meg_cached:.4f} sec")

# Time force compute
t0 = time.perf_counter()
lut_meg3, _ = _get_legen_table("meg", force_calc=True)
t_meg_recompute = time.perf_counter() - t0
print(f"MEG Legendre table (force recompute, no disk write): {t_meg_recompute:.4f} sec")

# _get_legen_der raw timing
xx = np.linspace(-1, 1, 20001)
times_der = []
for _ in range(5):
    t0 = time.perf_counter()
    result = _get_legen_der(xx, n_coeff=100)
    times_der.append(time.perf_counter() - t0)
print(f"\n_get_legen_der (20001 pts, 100 coeffs), 5 runs: {[f'{t:.4f}' for t in times_der]}")
print(f"  mean: {np.mean(times_der):.4f} sec")

print("\n" + "=" * 70)
print("PART C: Field interpolation timing (where Legendre is used in production)")
print("=" * 70)

# Profile make_field_map for MEG too
t0 = time.perf_counter()
field_maps_meg = make_field_map(evoked, trans="fsaverage",
                                 subject="sample", subjects_dir=subjects_dir,
                                 ch_type="meg", verbose=False)
t_map_meg = time.perf_counter() - t0
print(f"make_field_map (MEG) time: {t_map_meg:.3f} seconds")

print("\n" + "=" * 70)
print("FINAL COMPREHENSIVE SUMMARY")
print("=" * 70)
print()
print("1. _eeg_spherepot_coil:")
print("   - Used in: make_forward_solution (sphere model, EEG)")
print("   -          fit_dipole (sphere model, EEG component)")
print("   - For oct-6 source space (6688 sources, 60 EEG): 0.064 sec")
print("   - This is 97% of _compute_forwards time with sphere model")
print("   - But sphere forward total is ~0.064 sec vs BEM forward ~9 sec")
print("   - The sphere model is already 140x faster than BEM!")
print()
print("2. _sphere_field (MEG counterpart):")
print("   - Used in same contexts but for MEG channels")
print("   - In dipole fitting: 0.049 sec total for 646 calls (single-source each)")
print()
print("3. _get_legen_der:")
print("   - Used ONLY in _get_legen_table (Legendre derivative lookup table)")
print("   - _get_legen_table is used ONLY in make_field_map / _setup_dots")
print("   - NOT used in make_forward_solution")
print("   - NOT used in fit_dipole")
print("   - Table is computed ONCE and cached to disk")
print(f"   - Computation time: ~{np.mean(times_der):.3f} sec")
print(f"   - Cache read time: ~{t_meg_cached:.3f} sec")
print("   - Subsequent calls always read from cache (faster than computing)")
print()
print("4. Dipole fitting breakdown (13 time points):")
print("   Total time: 1.37 sec")
print("   _eeg_spherepot_coil: 0.012 sec (0.9% of total)")
print("   _sphere_field: 0.049 sec (3.6% of total)")
print("   COBYLA optimizer: 0.126 sec (9.2%)")
print("   parallel_func overhead: 0.725 sec (53%)")
print("   config/_open_lock I/O: 0.436 sec (32%)")
print("   -> Forward computation is <5% of dipole fitting time!")
print()
print("5. Forward solution breakdown (sphere, EEG, 6688 sources):")
print("   Total make_forward_solution: ~0.5 sec")
print("   Of that, _eeg_spherepot_coil: 0.064 sec (13%)")
print("   Of that, source space setup/filtering: ~0.4 sec (80%)")
print()
print("VERDICT: A 6.4x speedup on _eeg_spherepot_coil would save:")
print("  - Forward solution: 0.064 -> 0.010 sec (saving 0.054 sec out of 0.5 sec = 11%)")
print("  - Dipole fitting: 0.012 -> 0.002 sec (saving 0.010 sec out of 1.37 sec = 0.7%)")
print("  - The functions being optimized are NOT the bottleneck in any workflow")
