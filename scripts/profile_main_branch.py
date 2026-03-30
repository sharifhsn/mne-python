"""Benchmark the MAIN branch version of _eeg_spherepot_coil by simulating it."""
import time
import sys
import numpy as np

sys.path.insert(0, "/Users/sharif/Code/mne-python")
import mne
from mne.fixes import has_numba, bincount
from mne.forward._compute_forward import (
    _prep_field_computation, _concatenate_coils, _triage_coils,
)
from mne.forward._make_forward import _prepare_for_forward
from mne.transforms import _get_trans
from copy import deepcopy

print(f"has_numba: {has_numba}")

# Recreate the MAIN BRANCH version of _eeg_spherepot_coil (no JIT)
def _eeg_spherepot_coil_main(rrs, coils, sphere):
    """Main branch version: no JIT, dict access inside loop."""
    rmags, cosmags, ws, bins = _triage_coils(coils)
    n_coils = bins[-1] + 1
    del coils

    # Shift to the sphere model coordinates
    rrs = rrs - sphere["r0"]

    B = np.zeros((3 * len(rrs), n_coils))
    for ri, rr in enumerate(rrs):
        # Only process dipoles inside the innermost sphere
        if np.sqrt(np.dot(rr, rr)) >= sphere["layers"][0]["rad"]:
            continue
        # fwd_eeg_spherepot_vec
        vval_one = np.zeros((len(rmags), 3))

        # Make a weighted sum over the equivalence parameters
        for eq in range(sphere["nfit"]):
            # Scale the dipole position
            rd = sphere["mu"][eq] * rr
            rd2 = np.sum(rd * rd)
            rd2_inv = 1.0 / rd2
            # Go over all electrodes
            this_pos = rmags - sphere["r0"]

            # Vector from dipole to the field point
            a_vec = this_pos - rd

            # Compute the dot products needed
            a = np.sqrt(np.sum(a_vec * a_vec, axis=1))
            a3 = 2.0 / (a * a * a)
            r2 = np.sum(this_pos * this_pos, axis=1)
            r = np.sqrt(r2)
            rrd = np.sum(this_pos * rd, axis=1)
            ra = r2 - rrd
            rda = rrd - rd2

            # The main ingredients
            F = a * (r * a + ra)
            c1 = a3 * rda + 1.0 / a - 1.0 / r
            c2 = a3 + (a + r) / (r * F)

            # Mix them together and scale by lambda/(rd*rd)
            m1 = c1 - c2 * rrd
            m2 = c2 * rd2

            vval_one += (
                sphere["lambda"][eq]
                * rd2_inv
                * (m1[:, np.newaxis] * rd + m2[:, np.newaxis] * this_pos)
            )

            # compute total result
            xx = vval_one * ws[:, np.newaxis]
            zz = np.array([bincount(bins, x, bins[-1] + 1) for x in xx.T])
            B[3 * ri : 3 * ri + 3, :] = zz
    # finishing by scaling by 1/(4*M_PI)
    B *= 0.25 / np.pi
    return B


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

sensors2 = deepcopy(sensors)
fwd_data = _prep_field_computation(sensors=sensors2, bem=sphere_used, n_jobs=1)
coils = sensors2['eeg']['defs']

# Benchmark main branch version
times_main = []
for _ in range(3):
    t0 = time.perf_counter()
    result_main = _eeg_spherepot_coil_main(rr, coils, sphere_used)
    times_main.append(time.perf_counter() - t0)

print(f"\n_eeg_spherepot_coil MAIN BRANCH ({len(rr)} sources):")
print(f"  Times: {[f'{t:.4f}' for t in times_main]}")
print(f"  Mean: {np.mean(times_main):.4f} sec")

# Benchmark current branch version (with JIT)
from mne.forward._compute_forward import _eeg_spherepot_coil
# Warmup
_ = _eeg_spherepot_coil(rr[:10], coils, sphere_used)

times_jit = []
for _ in range(3):
    t0 = time.perf_counter()
    result_jit = _eeg_spherepot_coil(rr, coils, sphere_used)
    times_jit.append(time.perf_counter() - t0)

print(f"\n_eeg_spherepot_coil CURRENT BRANCH JIT ({len(rr)} sources):")
print(f"  Times: {[f'{t:.4f}' for t in times_jit]}")
print(f"  Mean: {np.mean(times_jit):.4f} sec")

# Verify results match
print(f"\nResults match: {np.allclose(result_main, result_jit, atol=1e-6)}")
if not np.allclose(result_main, result_jit, atol=1e-6):
    diff = np.abs(result_main - result_jit)
    print(f"  Max diff: {diff.max()}")
    print(f"  Mean diff: {diff.mean()}")

speedup = np.mean(times_main) / np.mean(times_jit)
print(f"\nSpeedup (main -> JIT): {speedup:.1f}x")
print(f"Time saved: {np.mean(times_main) - np.mean(times_jit):.4f} sec")
