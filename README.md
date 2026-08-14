
# ActMOF — MOF Bayesian Optimization Web Application

An active-learning synthesis optimization web platform for Metal-Organic Frameworks (MOFs), modeled after the **EDBO+** interface (`doyle-lab-ucla/edboplus`).

Ready for **Streamlit Community Cloud** using `uv`. Community Cloud detects `uv.lock`, installs the locked environment, and runs `app.py` with `streamlit run`.

📖 **New here?** See [`HELP.md`](HELP.md) for a full user guide (or the in-app **❓ Help & Guide** tab), covering the workflow, the `q` score, how recommendations are chosen, and the calibrated transfer prior.

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

## Streamlit Community Cloud Deployment

This repository is configured for Streamlit Community Cloud from GitHub:

1. Push the repository to GitHub with `app.py`, `pyproject.toml`, and `uv.lock` at the repo root.
2. Go to `share.streamlit.io` and create a new app from the GitHub repository.
3. Set the entrypoint file to `app.py`.
4. In Advanced settings, choose a Python version compatible with `pyproject.toml` (`>=3.9`; Python 3.10 or 3.11 is a conservative choice).
5. Deploy. Community Cloud will use `uv.lock` as the first-priority dependency file.

For local testing, run the same app shape from the repository root:

```bash
uv sync --frozen
uv run streamlit run app.py
```

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `app.py` | **Main Streamlit Web Application** (EDBO+-inspired UI workflow) |
| `bo_engine.py` | Standalone BO & GP engine (Matern 3/2 & 5/2, Calibrated Transfer Prior, Matplotlib plotting) |
| `HELP.md` | Full user guide (also mirrored in the app's ❓ Help & Guide tab) |
| `.streamlit/config.toml` | Streamlit runtime configuration for Community Cloud |
| `pyproject.toml` | Project metadata and direct dependency specifications |
| `uv.lock` | Streamlit Community Cloud and local `uv` dependency lockfile |
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
