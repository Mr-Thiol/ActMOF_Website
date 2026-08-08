# ActMOF Website — MOF Bayesian Optimization Student App

A local desktop GUI for **active-learning synthesis optimization** of Metal-Organic Frameworks (MOFs).

## Files

| File | Description |
|------|-------------|
| `student_bo_app_v109.py` | Main student-facing GUI app (Tkinter + NumPy-only GP) |
| `one_click_build_mof_bo_student_app_v109.py` | One-click Windows `.exe` builder (PyInstaller) |

## Features (v1.0.9)

- **Lightweight Gaussian Process** — pure NumPy implementation with Matérn 3/2 or 5/2 kernels; no heavy GP libraries required
- **Acquisition functions** — Expected Improvement (EI) or Probability of Improvement (PI)
- **Batch suggestions** — recommends the next set of synthesis experiment parameters per configurable batch size
- **Project planning wizard** — first-time setup: input experiments-per-batch and estimated iteration range; app previews total experiment count and calibrated-transfer window M
- **Persistent config** — settings saved to `project_config.json` and restored on reload

## Requirements

```
numpy
pandas
matplotlib
tkinter  # bundled with Python on Windows
```

## Usage

### Run the GUI directly

```bash
python student_bo_app_v109.py
```

### Build a standalone Windows executable

```bash
python one_click_build_mof_bo_student_app_v109.py
```

The script bundles the app into a single `.exe` via PyInstaller.
