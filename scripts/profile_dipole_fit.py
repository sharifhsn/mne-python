"""Profile dipole fitting to understand where sphere forward is used."""

import time
import cProfile
import pstats
import io
import sys
import numpy as np

sys.path.insert(0, "/Users/sharif/Code/mne-python")
import mne

print("MNE version:", mne.__version__)

data_path = mne.datasets.sample.data_path()
raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
trans_fname = data_path / "MEG" / "sample" / "sample_audvis_raw-trans.fif"
subjects_dir = data_path / "subjects"

# Create evoked data from sample
raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
events = mne.find_events(raw, verbose=False)
raw.set_eeg_reference(projection=True, verbose=False)
epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=0.5,
                    baseline=(None, 0), preload=True, verbose=False)
evoked = epochs.average()
evoked.apply_proj()

# Pick a smaller time window for speed
evoked = evoked.crop(0.07, 0.09)
print(f"Number of time points: {len(evoked.times)}")
print(f"Number of channels: {len(evoked.ch_names)}")

# Create sphere model
sphere = mne.make_sphere_model(r0="auto", head_radius="auto", info=evoked.info)
# Create noise covariance
cov = mne.compute_covariance(epochs, tmax=0, method="empirical", verbose=False)

print("\n" + "=" * 70)
print("DIPOLE FITTING WITH SPHERE MODEL (most common use)")
print("=" * 70)

# Profile the dipole fit
pr = cProfile.Profile()
pr.enable()
t0 = time.perf_counter()
dip, residual = mne.fit_dipole(evoked, cov, sphere, verbose=False)
t_fit = time.perf_counter() - t0
pr.disable()

print(f"Dipole fit time: {t_fit:.2f} seconds")
print(f"Number of dipoles: {len(dip.times)}")

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
ps.print_stats(30)
print(s.getvalue())

# Show cumulative too
s2 = io.StringIO()
ps2 = pstats.Stats(pr, stream=s2).sort_stats("cumulative")
ps2.print_stats(30)
print("\n--- Sorted by cumulative time ---")
print(s2.getvalue())

# ============================================================================
# Check: how many forward evaluations does dipole fitting do?
# ============================================================================
print("\n" + "=" * 70)
print("ANALYSIS: Forward evaluations in dipole fitting")
print("=" * 70)
print(f"Dipole fitting calls _eeg_spherepot_coil or _sphere_field repeatedly")
print(f"for each time point, for the guess grid and then for optimization.")
print(f"With {len(evoked.times)} time points, the number of forward calls is high.")
print(f"Each call computes forward for 1 source location (during optimization)")
print(f"or for the full guess grid (once at the start).")

# Now try with BEM too
bem_fname = data_path / "subjects" / "sample" / "bem" / "sample-5120-5120-5120-bem-sol.fif"
bem = mne.read_bem_solution(bem_fname)
trans = mne.read_trans(trans_fname)

print("\n" + "=" * 70)
print("DIPOLE FITTING WITH BEM MODEL (less common, needs trans)")
print("=" * 70)

t0 = time.perf_counter()
dip_bem, residual_bem = mne.fit_dipole(evoked, cov, bem, trans=trans, verbose=False)
t_fit_bem = time.perf_counter() - t0
print(f"Dipole fit with BEM time: {t_fit_bem:.2f} seconds")

print(f"\nSphere fit: {t_fit:.2f} sec")
print(f"BEM fit: {t_fit_bem:.2f} sec")
print(f"BEM is {t_fit_bem/t_fit:.1f}x slower than sphere for dipole fitting")
