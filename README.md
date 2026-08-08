---
title: ActMOF - MOF Bayesian Optimization
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
---

# ActMOF — MOF Bayesian Optimization Web Application

An active-learning synthesis optimization web platform for Metal-Organic Frameworks (MOFs), modeled after the **EDBO+** interface (`doyle-lab-ucla/edboplus`).

Hosted directly on **Hugging Face Spaces** using Streamlit SDK and managed with `uv` (`pyproject.toml` and `uv.lock`).

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

## 📁 Repository Structure

| File | Description |
|------|-------------|
| [`app.py`](file:///home/reptiman/research/ActMOF_Website/app.py) | **Main Streamlit Web Application** (EDBO+-inspired UI workflow) |
| [`bo_engine.py`](file:///home/reptiman/research/ActMOF_Website/bo_engine.py) | Standalone BO & GP engine (Matérn 3/2 & 5/2, Calibrated Transfer Prior, Matplotlib plotting) |
| [`pyproject.toml`](file:///home/reptiman/research/ActMOF_Website/pyproject.toml) | Project configuration and dependency specifications |
| [`uv.lock`](file:///home/reptiman/research/ActMOF_Website/uv.lock) | Lockfile for reproducible environment installations |
| [`student_bo_app_v109.py`](file:///home/reptiman/research/ActMOF_Website/student_bo_app_v109.py) | Desktop GUI version (Tkinter) |
| [`one_click_build_mof_bo_student_app_v109.py`](file:///home/reptiman/research/ActMOF_Website/one_click_build_mof_bo_student_app_v109.py) | PyInstaller Windows `.exe` builder |

---

## ✨ Features (v1.0.9)

- **EDBO+-Style Interactive Workflow:**
  - **📊 Dashboard & Visual Diagnostics:** Live metric summaries, model status banner, and embedded 6-panel Matplotlib diagnostics (Best $q$ progress, $q$ trajectory, Observed vs Predicted parity, Uncertainty landscape, Acquisition landscape, and 2D Parameter map).
  - **🧪 Batch Recommendations:** Run BO optimization, view proposed synthesis conditions, and instantly export results as `.csv`.
  - **📝 Experiment Data Entry:** Upload prior `experiments.csv` files, interactively edit measurement data (`intensity`, `fwhm`, `q`), and download updated logs.
  - **🧙 Project Planning Wizard:** Configure iteration bounds and compute the calibrated-transfer prior window $M = \lfloor \text{mean\_range} \times \text{fraction} \rfloor$.
- **Zero Heavy ML Dependencies:** Pure NumPy Gaussian Process engine with Matérn kernels and grid-searched hyperparameter tuning.
- **Calibrated Transfer Learning (Mode C):** Leverages built-in benchmark data to accelerate early-round optimization before student data accumulates.
