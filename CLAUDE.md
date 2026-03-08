# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MNE-Python is an open-source Python package for exploring, visualizing, and analyzing human neurophysiological data (MEG, EEG, sEEG, ECoG, fNIRS). It requires Python >= 3.10 with core dependencies: NumPy, SciPy, Matplotlib, Pooch, tqdm.

## Common Commands

### Testing
```bash
# Run all tests (from repo root)
pytest mne/

# Run a single test file
pytest mne/tests/test_epochs.py

# Run a single test function
pytest mne/tests/test_epochs.py::test_event_repeated

# Run tests for a submodule
pytest mne/io/tests/
pytest mne/preprocessing/tests/

# Run with verbose debug output
pytest mne/tests/test_epochs.py -xvs

# Run only doctests in a module
pytest --doctest-modules mne/epochs.py

# Pytest is configured in pyproject.toml [tool.pytest.ini_options]
# Default: --capture=sys --tb=short --cov-branch -rfEXs with testpaths=["mne"]
```

### Linting and Style
```bash
# Run all pre-commit checks (ruff, codespell, rstcheck, yamllint, toml-sort)
make pre-commit
# or directly:
pre-commit run -a --show-diff-on-failure

# Ruff is the primary linter/formatter (configured in pyproject.toml [tool.ruff])
ruff check mne/
ruff format mne/

# Type checking (limited — many error codes disabled, see pyproject.toml [tool.mypy])
mypy

# Dead code detection
vulture mne/ tools/vulture_allowlist.py
```

### Building (uv)

This fork uses [uv](https://docs.astral.sh/uv/) for dependency management.
MNE's `pyproject.toml` uses standard `[dependency-groups]` which uv supports natively.

**Preferred: `uv sync` (lockfile-based, reproducible)**

```bash
# Create venv + install in dev mode with test deps (creates uv.lock)
uv sync --python 3.12 --group test

# Install with all optional dependencies (no Qt)
uv sync --python 3.12 --group test --extra full-no-qt

# Install with CuPy for GPU experiments (NVIDIA Linux only)
uv sync --python 3.12 --group test --group gpu

# Full setup: optional deps + GPU + tests
uv sync --python 3.12 --group test --group gpu --extra full-no-qt

# Verify installation
.venv/bin/python -c "import mne; print(mne.__version__)"
```

**Alternative: `uv pip install` (no lockfile, more manual)**

```bash
# Create venv manually first
uv venv --python 3.12 .venv

# Install in development mode with test dependencies
uv pip install -e . --group test

# Install with CuPy (NVIDIA Linux only)
uv pip install -e . --group test --group gpu
```

**Key notes:**
- MNE uses `[dependency-groups]` (PEP 735), not `[project.optional-dependencies]`
  for dev/test deps. Use `--group test` not `-e ".[test]"`.
- The `[full]`, `[full-no-qt]`, `[hdf5]` extras ARE in `[project.optional-dependencies]`
  and work with `--extra`.
- The `gpu` group (this fork only) installs `cupy-cuda12x` on Linux.
  CuPy wheels come from PyPI directly — no custom index URL needed.
- `uv sync` creates a `uv.lock` lockfile for reproducibility. The lockfile is
  gitignored since this is a fork.

## Architecture

### Lazy Loading System
- `mne/__init__.py` uses `lazy_loader.attach_stub()` for deferred imports
- `mne/__init__.pyi` defines the full public API for type checkers and IDE support
- Submodules also use lazy loading (e.g., `mne/io/__init__.py`)

### Core Class Hierarchy via Mixins
The main data containers share behavior through a mixin chain:

```
ProjMixin          → SSP projection handling
ContainsMixin      → Channel/coordinate queries
UpdateChannelsMixin → Channel type/name updates
SetChannelsMixin   → Channel property setting (includes MontageMixin)
ReferenceMixin     → EEG re-referencing (includes MontageMixin)
InterpolationMixin → Bad channel interpolation
FilterMixin        → Filtering and resampling
SizeMixin          → Memory estimation
SpectrumMixin      → Power spectrum computation
TimeMixin          → Time-based operations
GetEpochsMixin     → Epoch indexing/slicing
```

**Main data classes** compose these mixins:
- `BaseRaw` (in `mne/io/base.py`) — continuous data
- `BaseEpochs` (in `mne/epochs.py`) — segmented data (adds GetEpochsMixin)
- `Evoked` (in `mne/evoked.py`) — averaged data

### IO Subsystem (`mne/io/`)
- `BaseRaw` is the abstract base for all format readers
- `read_raw()` in `_read_raw.py` auto-detects format
- Each format has its own subdirectory (fiff/, ctf/, edf/, brainvision/, etc.) with a `Raw` subclass
- Reader pattern: `read_raw_xxx(fname, **kwargs)` returns a Raw instance

### Key Utility Patterns (`mne/utils/`)

**`@verbose` decorator** (`_logging.py`): Functions that log must accept `verbose=None` and be decorated with `@verbose`. This enables per-call log level control.

**`@fill_doc` decorator** (`docs.py`): Substitutes `%(param_name)s` placeholders in docstrings from a shared `docdict`. Use NumPy docstring format.

**Input validation** (`check.py`):
- `_validate_type(item, types, item_name)` — validates types; supports strings like `"path-like"`, `"numeric"`, `"array-like"`
- `_check_option(parameter, value, allowed_values)` — validates against allowed values
- `_ensure_int(x, name)` — integer validation (rejects booleans)
- `_check_fname(fname, overwrite, must_exist)` — file path validation
- `_check_preload(inst, msg)` — ensures data is loaded into memory

**Deprecation** (`docs.py`):
- `@deprecated("message")` — emits FutureWarning
- `@legacy(alt="new_way()")` — logs info message (less disruptive)

### Testing Infrastructure

**Pytest configuration** is in `pyproject.toml` and `mne/conftest.py`.

**Key fixtures** (from `conftest.py`):
- `raw`, `raw_orig`, `raw_ctf` — preloaded Raw objects
- `epochs`, `epochs_unloaded`, `epochs_full` — Epochs objects
- `events` — events array
- `verbose_debug` — runs test at DEBUG log level
- Auto-use: `close_all` (closes matplotlib), `check_verbose` (restores log level)

**Test markers**: `@pytest.mark.slowtest`, `@pytest.mark.ultraslowtest`

**Test data paths** defined in `conftest.py`: `fname_raw_io`, `fname_event_io`, `fname_evoked`, `fname_cov`, etc.

**Testing patterns**:
```python
# Test expected errors
with pytest.raises(ValueError, match="some pattern"):
    function_that_should_fail()

# Test expected warnings
with pytest.warns(RuntimeWarning, match="some pattern"):
    function_that_warns()

# Test logging output
from mne.utils import catch_logging
with catch_logging("info") as log:
    function_that_logs()
    assert "expected text" in log.getvalue()
```

## Changelog (Towncrier)

PRs need a changelog fragment in `doc/changes/dev/`. Format: `{PR_NUMBER}.{TYPE}.rst`

Types: `newfeature`, `bugfix`, `apichange`, `dependency`, `notable`, `other`

Example content (`13714.bugfix.rst`):
```
Fix bug in 3D overlay compositing that could produce NaN RGBA values when the resulting alpha is zero, by `Pragnya Khandelwal`_.
```

Contributor names use Sphinx roles referencing `doc/changes/names.inc`.

## Style Conventions

- **Ruff** rules: A, B006, D, E, F, I, UP, W (see `pyproject.toml` for ignores)
- **Docstrings**: NumPy convention via `numpydoc`
- `__init__.py`, `constants.py`, `resources.py` are excluded from ruff
- Imports sorted with isort (via ruff `I` rule)
- Use `warn()` from `mne.utils` (not `warnings.warn` directly) — it finds the user's frame in the call stack

## GPU Acceleration Work (this fork)

This fork explores GPU-accelerating the permutation cluster test bottleneck.
All work lives in `gpu_accel/`. See `gpu_accel/FEASIBILITY.md` for full context.

### The Bottleneck

`mne.stats.spatio_temporal_cluster_1samp_test` is the #1 performance bottleneck
for researchers. For source-space data (fsaverage ico-5, ~20K vertices), 5,000
permutations takes ~10-20 minutes. **~97% of time is in connected-component
labeling** on a sparse adjacency graph (`mne/stats/cluster_level.py::_get_components`),
which calls `scipy.sparse.csgraph.connected_components` (single-threaded CPU BFS).

### Key Files

- `mne/stats/cluster_level.py` — The permutation test implementation
  - `_get_components()` (line ~289) — The CCL bottleneck (uses `scipy.sparse.csgraph`)
  - `_get_clusters_st()` (line ~260) — Spatio-temporal clustering with Numba JIT
  - `_do_1samp_permutations()` (line ~723) — Inner permutation loop
  - `_find_clusters()` (line ~319) — Thresholding + cluster finding dispatch
  - `_permutation_cluster_test()` (line ~890) — Top-level orchestration
- `mne/stats/parametric.py` — `ttest_1samp_no_p()` (the stat function, ~3 lines of NumPy)
- `mne/stats/tests/test_cluster_level.py` — Comprehensive test suite

### Approach

1. **CuPy proof-of-concept** (`gpu_accel/patch_cupy_poc.py`): Drop-in GPU
   `connected_components` via CuPy. NVIDIA-only. Validates the speedup.
2. **wgpu + Rust** (future): Cross-platform GPU via compute shaders (WGSL).
   Works on NVIDIA/AMD/Apple Silicon/Intel via Vulkan/Metal/DX12.
3. **Fused pipeline** (aspirational): Entire permutation loop on GPU.

### Running Benchmarks

```bash
# Download sample dataset (~1.5 GB, one-time)
.venv/bin/python -c "import mne; mne.datasets.sample.data_path()"

# CPU baseline
.venv/bin/python gpu_accel/benchmark_cluster_cpu.py

# GPU (CuPy) — requires NVIDIA GPU + cupy-cuda12x
.venv/bin/python gpu_accel/patch_cupy_poc.py
```

### Numba JIT Functions (28 total across 9 files)

These are the only custom compute kernels in MNE (everything else delegates
to NumPy/SciPy). They are candidates for Rust replacement:

- `mne/surface.py` (8 fns): Mesh geometry, triangle nearest-point (only `prange` fn)
- `mne/forward/_compute_forward.py` (5 fns): BEM/sphere forward modeling
- `mne/stats/cluster_level.py` (6 fns): Cluster neighbor finding
- `mne/chpi.py` (4 fns): cHPI head position fitting
- `mne/transforms.py` (3 fns): Quaternion algebra
- `mne/decoding/time_delaying_ridge.py` (2 fns): Toeplitz matrix ops

### Performance Optimization Progress

See `gpu_accel/perf-log.md` for detailed benchmark tables and results.

**Completed optimizations** (cumulative 22.4x on 128-perm fsaverage test):

1. **Vectorize loop** (v1): Replaced Python for-loop in `_get_components` with
   `bincount`/`argsort`/`split`. Eliminated per-vertex Python overhead.
2. **Reindex active-only** (v2): Build compact graph of only supra-threshold
   vertices before `connected_components`. Shrinks CCL input from ~20K to ~1K
   vertices. Also eliminates `has_sig` filtering (all compact-graph components
   are valid). This was the biggest single win (~5x).
3. **Vectorize cluster sums** (v3): Replaced `[_masked_sum(x, c) for c in clusters]`
   loop (~1500 calls) with single `np.add.reduceat` in `_find_clusters_1dir`.
4. **Skip np.split in permutation loop** (v4): Added `_sums_only` fast path.
   During permutations, `_get_components(return_labels=True)` returns raw
   `(idx, components)`, and sums are computed via `np.bincount` — skips
   `argsort`/`split`/`concatenate` entirely.
5. **Numba union-find + precomputed t-test** (v5): Two optimizations:
   - `_fused_ccl` Numba JIT replaces scipy `connected_components` in the
     permutation loop (3.5-11x faster, avoids sparse matrix construction).
   - Precomputed `sum(X²)` exploits s²=1 identity for sign-flip permutations,
     replacing two full-array multiplies + `np.var` with a matrix-vector multiply.

**Current per-permutation breakdown** (fsaverage ico-5, 20K vertices, ~0.3ms):
- Numba UF edge iteration: ~50%
- Threshold comparison + np.where: ~20%
- Precomputed ttest (signs @ X): ~20%
- bincount sums: ~10%

**Remaining opportunities**:
- Fused GPU pipeline (sign-flip + ttest + threshold + CCL + reduce all on GPU)
- The standard spatio-temporal path uses `_get_clusters_st` (Numba BFS), not
  `_get_components` — optimizing that path is separate

**Optimization workflow** (follow for every optimization):
1. Profile to find the bottleneck (instrument with `time.perf_counter()`)
2. Implement the fix
3. **Adversarial code review**: edge cases, type variations (bool vs int `x_in`),
   downstream consumers, view-vs-copy safety, ordering invariants
4. **Parity + performance benchmarks**: verify identical output across varying
   density, graph size, edge cases, real fsaverage data, end-to-end permutation
   test. Use the old implementation as reference.
5. **Record in `gpu_accel/perf-log.md`**: what changed, why, benchmark tables

### Relevant MNE Issues/PRs

- [#13002](https://github.com/mne-tools/mne-python/pull/13002) — CUDA zero-copy (closed)
- [#12609](https://github.com/mne-tools/mne-python/pull/12609) — TFCE optimization (merged)
- [#7784](https://github.com/mne-tools/mne-python/pull/7784) — Permutation speedup (merged, 2-5%)
- [#8095](https://github.com/mne-tools/mne-python/pull/8095) — Numba cluster stc (merged, 35%)
- [#5439](https://github.com/mne-tools/mne-python/issues/5439) — Use CuPy (open, stalled)
- [#13175](https://github.com/mne-tools/mne-python/issues/13175) — uv support (open, maintainers already use it)
