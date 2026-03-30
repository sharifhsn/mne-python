"""Benchmark WITHOUT numba to measure the actual JIT benefit."""
import time
import sys
import numpy as np

sys.path.insert(0, "/Users/sharif/Code/mne-python")
import mne
from mne.fixes import has_numba
print(f"has_numba: {has_numba}")

from mne.forward._compute_forward import (
    _eeg_spherepot_coil, _do_eeg_spherepot,
    _prep_field_computation, _compute_forwards,
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

# Benchmark _eeg_spherepot_coil
sensors2 = deepcopy(sensors)
fwd_data = _prep_field_computation(sensors=sensors2, bem=sphere_used, n_jobs=1)
coils = sensors2['eeg']['defs']

times_fwd = []
for _ in range(3):
    t0 = time.perf_counter()
    result = _eeg_spherepot_coil(rr, coils, sphere_used)
    times_fwd.append(time.perf_counter() - t0)

print(f"\n_eeg_spherepot_coil ({len(rr)} sources, numba={has_numba}):")
print(f"  Times: {[f'{t:.4f}' for t in times_fwd]}")
print(f"  Mean: {np.mean(times_fwd):.4f} sec")

# Benchmark _get_legen_der
xx = np.linspace(-1, 1, 20001)
times_ld = []
for _ in range(3):
    t0 = time.perf_counter()
    result = _get_legen_der(xx, n_coeff=100)
    times_ld.append(time.perf_counter() - t0)

print(f"\n_get_legen_der (20001 pts, numba={has_numba}):")
print(f"  Times: {[f'{t:.4f}' for t in times_ld]}")
print(f"  Mean: {np.mean(times_ld):.4f} sec")

# Full forward
times_full = []
for _ in range(3):
    sensors3 = deepcopy(sensors)
    t0 = time.perf_counter()
    fwds = _compute_forwards(rr, bem=sphere_used, sensors=sensors3, n_jobs=1)
    times_full.append(time.perf_counter() - t0)

print(f"\n_compute_forwards (sphere, EEG, {len(rr)} sources, numba={has_numba}):")
print(f"  Times: {[f'{t:.4f}' for t in times_full]}")
print(f"  Mean: {np.mean(times_full):.4f} sec")
