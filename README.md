---
title: ActMOF - MOF Bayesian Optimization
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: Active-learning Bayesian optimization workflow for MOF synthesis.
---

# ActMOF — MOF Bayesian Optimization Web Application

An active-learning synthesis optimization web platform for Metal-Organic Frameworks (MOFs), modeled after the **EDBO+** interface (`doyle-lab-ucla/edboplus`).

Ready for **Hugging Face Spaces** using the Streamlit SDK. Spaces installs runtime dependencies from `requirements.txt`; local development can still use `uv` with `pyproject.toml` and `uv.lock`.

---

## 🚀 Quickstart with `uv`

### 1. Synchronize dependencies from `uv.lock`

```bash
# Sync environment from pyproject.toml and uv.lock
uv sync
```

### 2. Launch the Streamlit Web Application

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Hugging Face Spaces Deployment

This repository is configured for a Streamlit Space synced from GitHub:

1. Create a new Hugging Face Space with **SDK = Streamlit**.
2. Connect or push this GitHub repository to the Space.
3. Hugging Face reads the README YAML block, launches `app.py`, and installs packages from `requirements.txt`.

Do not launch the hosted app with `uv run app.py`; Streamlit apps must be started through `streamlit run app.py`, which Hugging Face handles automatically from the Space metadata.

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `app.py` | **Main Streamlit Web Application** (EDBO+-inspired UI workflow) |
| `bo_engine.py` | Standalone BO & GP engine (Matern 3/2 & 5/2, Calibrated Transfer Prior, Matplotlib plotting) |
| `requirements.txt` | Hugging Face Spaces dependency install file |
| `.streamlit/config.toml` | Streamlit runtime configuration for hosted execution |
| `pyproject.toml` | Local project configuration and dependency specifications |
| `uv.lock` | Lockfile for reproducible local `uv` environments |
| `student_bo_app_v109.py` | Desktop GUI version (Tkinter) |
| `one_click_build_mof_bo_student_app_v109.py` | PyInstaller Windows `.exe` builder |

---

## ✨ Features (v1.0.9)

- **EDBO+-Style Interactive Workflow:**
  - **📊 Dashboard & Visual Diagnostics:** Live metric summaries, model status banner, and embedded 6-panel Matplotlib diagnostics (Best $q$ progress, $q$ trajectory, Observed vs Predicted parity, Uncertainty landscape, Acquisition landscape, and 2D Parameter map).
  - **🧪 Batch Recommendations:** Run BO optimization, view proposed synthesis conditions, and instantly export results as `.csv`.
  - **📝 Experiment Data Entry:** Upload prior `experiments.csv` files, interactively edit measurement data (`intensity`, `fwhm`, `q`), and download updated logs.
  - **🧙 Project Planning Wizard:** Configure iteration bounds and compute the calibrated-transfer prior window $M = \lfloor \text{mean\_range} \times \text{fraction} \rfloor$.
- **Zero Heavy ML Dependencies:** Pure NumPy Gaussian Process engine with Matérn kernels and grid-searched hyperparameter tuning.
- **Calibrated Transfer Learning (Mode C):** Leverages built-in benchmark data to accelerate early-round optimization before student data accumulates.
