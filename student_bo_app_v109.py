# -*- coding: utf-8 -*-
"""
MOF Bayesian Optimization Student App

Version 1.0.9 adds a first-time project planning wizard. Students enter the
experiments-per-batch value and an expected iteration range. The app previews
the total experiment range and computes the calibrated-transfer window from
the average iteration estimate, using floor(mean_range * fraction).

A local desktop GUI for active-learning synthesis optimization.
The app suggests configurable experiment batches with a Gaussian Process model,
Matérn 3/2 or Matérn 5/2 kernel, and EI or PI acquisition.
"""

from __future__ import annotations

import json
import re
import math
import os
import platform
import subprocess
import sys
import traceback
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    # This fallback only matters for non-GUI test environments.
    pass
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure




def normal_pdf(z: np.ndarray) -> np.ndarray:
    """Standard normal probability density function implemented directly."""
    z = np.asarray(z, dtype=float)
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def normal_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal cumulative distribution function using math.erf.

    This direct implementation keeps the packaged GUI smaller and more reliable.
    """
    z = np.asarray(z, dtype=float)
    erf_vec = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))



class LightweightGaussianProcess:
    """Small NumPy-only Gaussian Process for the student app.

    This class avoids heavy external GP libraries in the packaged GUI. The project
    only needs a compact GP for small active-learning histories, so a direct
    Cholesky implementation is sufficient and keeps the Windows executable much
    easier to copy and distribute.
    """

    def __init__(self, kernel_name: str = "matern52", random_state: int = 42):
        self.kernel_name = str(kernel_name).lower()
        self.random_state = int(random_state)

    def _kernel(self, Xa: np.ndarray, Xb: np.ndarray, length_scale: float, signal_variance: float = 1.0) -> np.ndarray:
        Xa = np.asarray(Xa, dtype=float)
        Xb = np.asarray(Xb, dtype=float)
        ell = max(float(length_scale), 1e-6)
        diff = (Xa[:, None, :] - Xb[None, :, :]) / ell
        r = np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))
        if self.kernel_name == "matern32":
            a = math.sqrt(3.0)
            k = (1.0 + a * r) * np.exp(-a * r)
        else:
            a = math.sqrt(5.0)
            k = (1.0 + a * r + (5.0 / 3.0) * r * r) * np.exp(-a * r)
        return float(signal_variance) * k

    def _negative_log_marginal_likelihood(self, X: np.ndarray, y: np.ndarray, length_scale: float, noise: float) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
        n = X.shape[0]
        K = self._kernel(X, X, length_scale, 1.0)
        K[np.diag_indices_from(K)] += float(noise)
        jitter_values = [0.0, 1e-10, 1e-8, 1e-6, 1e-4]
        for jitter in jitter_values:
            try:
                Kj = K.copy()
                Kj[np.diag_indices_from(Kj)] += jitter
                L = np.linalg.cholesky(Kj)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                nll = 0.5 * float(y @ alpha) + float(np.sum(np.log(np.diag(L)))) + 0.5 * n * math.log(2.0 * math.pi)
                if np.isfinite(nll):
                    return nll, L, alpha
            except np.linalg.LinAlgError:
                continue
        return float("inf"), None, None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or len(y) != X.shape[0] or len(y) == 0:
            raise ValueError("Invalid GP training data.")

        self.X_train_ = X.copy()
        self.y_mean_ = float(np.mean(y))
        self.y_std_ = float(np.std(y, ddof=1)) if len(y) > 1 else 1.0
        if not np.isfinite(self.y_std_) or self.y_std_ < 1e-12:
            self.y_std_ = 1.0
        y_norm = (y - self.y_mean_) / self.y_std_

        # A small hyperparameter search is enough for this teaching app. It
        # keeps the behavior GP-like without needing an external optimizer.
        length_candidates = [0.06, 0.09, 0.12, 0.18, 0.25, 0.35, 0.50, 0.75, 1.00]
        if len(X) >= 3:
            diffs = []
            for i in range(len(X)):
                for j in range(i + 1, len(X)):
                    d = float(np.linalg.norm(X[i] - X[j]))
                    if d > 1e-12:
                        diffs.append(d)
            if diffs:
                med = float(np.median(diffs))
                length_candidates.extend([max(0.05, min(1.25, med / 2.0)), max(0.05, min(1.25, med))])
        length_candidates = sorted(set(round(float(x), 6) for x in length_candidates))
        noise_candidates = [1e-7, 1e-5, 1e-4, 1e-3, 1e-2]

        best = (float("inf"), None, None, None, None)
        for length_scale in length_candidates:
            for noise in noise_candidates:
                nll, L, alpha = self._negative_log_marginal_likelihood(X, y_norm, length_scale, noise)
                if nll < best[0]:
                    best = (nll, length_scale, noise, L, alpha)

        if best[3] is None or best[4] is None:
            raise RuntimeError("The lightweight GP could not build a stable covariance matrix.")

        self.length_scale_ = float(best[1])
        self.noise_ = float(best[2])
        self.L_ = best[3]
        self.alpha_ = best[4]
        return self

    def predict(self, X: np.ndarray, return_std: bool = True):
        X = np.asarray(X, dtype=float)
        K_trans = self._kernel(X, self.X_train_, self.length_scale_, 1.0)
        mu_norm = K_trans @ self.alpha_
        mu = mu_norm * self.y_std_ + self.y_mean_
        if not return_std:
            return mu
        v = np.linalg.solve(self.L_, K_trans.T)
        var_norm = np.maximum(1.0 - np.sum(v * v, axis=0), 1e-12)
        sd = np.sqrt(var_norm) * self.y_std_
        return mu, sd

APP_NAME = "MOF BO Student App"
APP_VERSION = "1.0.9"

FEATURES = [
    "metal_amount",
    "modulator",
    "add_solvent",
    "reaction_time",
    "reaction_temperature",
]

BOUNDS: Dict[str, Tuple[int, int, int]] = {
    "metal_amount": (5, 75, 1),
    "modulator": (5, 15, 1),
    "add_solvent": (0, 30, 1),
    "reaction_time": (1, 12, 1),
    "reaction_temperature": (10, 30, 1),
}

EXPERIMENT_COLUMNS = [
    "record_id",
    "round",
    "batch_position",
    "status",
    *FEATURES,
    "intensity",
    "fwhm",
    "q",
    "predicted_q_mean",
    "predicted_q_sd",
    "acquisition_value",
    "notes",
    "created_at",
    "updated_at",
]

DEFAULT_CONFIG = {
    "project_name": "Untitled Project",
    "kernel": "matern52",
    "acquisition": "ei",
    "batch_size": 3,
    "initial_samples": 3,
    "random_candidate_count": 15000,
    "diversity_lambda": 0.03,
    "use_log1p_target": True,
    "use_reference_prior": False,
    "planned_iteration_min": 10,
    "planned_iteration_max": 20,
    "planned_total_batches": 15,
    "estimated_total_experiments_min": 30,
    "estimated_total_experiments_max": 60,
    "transfer_prior_fraction": 0.30,
    "transfer_rounds_mode": "auto",
    "transfer_prior_rounds": 4,
    "candidate_mode": "random",
    "objective": "maximize_q",
    "seed": 42,
    "created_at": "",
    "updated_at": "",
}

# Built-in reference data from the benchmark background.
# These rows are optional reference data for calibrated transfer. They are not shown as student runs.
REFERENCE_X = np.array([
    [35,5,22,9,14],[68,7,15,12,24],[12,9,1,7,29],[36,10,24,3,18],[55,11,8,2,15],
    [69,6,23,1,20],[14,12,2,11,15],[42,7,27,6,17],[9,14,7,4,11],[45,6,11,8,28],
    [27,12,1,2,25],[41,14,5,3,16],
    [5,5,0,1,30],[5,15,0,1,30],[75,15,0,1,30],
    [16,7,22,3,20],[16,9,23,4,21],[15,7,12,11,19],
    [6,5,29,11,27],[62,7,26,11,26],[5,15,2,11,16],
    [33,10,7,11,18],[9,6,1,12,19],[48,14,3,12,23],
    [19,8,12,11,16],[15,7,14,11,24],[12,8,10,11,18],
    [15,7,13,11,18],[14,7,11,11,20],[7,6,20,11,15],
    [15,7,14,11,24],[15,6,10,11,14],[15,7,10,11,12],
    [15,8,12,11,19],[9,8,13,7,20],[21,6,13,12,19],
    [6,6,12,11,20],[15,8,9,12,19],[75,12,6,11,18],
    [25,13,14,7,24],[17,13,16,12,24],[38,13,16,8,24],
    [65,5,0,12,24],[74,7,23,1,16],[5,7,21,12,20],
    [15,7,16,12,29],[16,7,15,11,18],[62,10,12,10,17],
    [11,7,12,11,24],[17,7,16,11,27],[13,7,10,12,25],
    [13,6,13,11,24],[15,5,0,12,24],[21,5,26,11,23],
    [16,7,8,11,24],[28,7,0,12,30],[10,7,0,12,24],
    [23,7,30,11,24],[20,7,30,11,29],[23,7,14,11,25],
    [16,7,18,12,25],[17,7,18,12,24],[24,7,18,12,24],
    [16,7,14,11,30],[15,7,13,9,23],[15,7,7,7,26],
    [13,6,25,12,24],[13,7,11,11,23],[14,7,7,11,26],
    [15,6,23,12,24],[15,5,8,12,23],[15,6,14,12,21],
    [13,5,10,11,24],[17,6,11,11,22],[19,5,2,12,24],
    [6,6,19,12,30],[25,7,18,11,24],[27,7,27,12,26],
    [15,5,8,11,23],[15,5,11,10,21],[24,5,30,12,24],
    [13,7,15,12,25],[16,7,11,12,24],[14,6,15,12,26],
    [14,7,15,11,26],[18,7,10,12,23],[10,7,13,12,29],
    [23,7,2,12,19],[18,7,1,4,19],[19,6,4,12,25],
    [17,7,10,12,25],[16,6,6,5,23],[20,6,0,12,23],
    [17,7,20,11,21],[24,7,22,8,21],[25,7,19,10,27]
], dtype=float)

REFERENCE_INTENSITY = np.array([
    0,0,785,0,0,605,0,0,182,0,880,0,
    1457,0,0,
    2345,960,16751,
    7575,0,0,
    8060,1318,920,
    0,23969,7305,
    7703,12434,0,
    23969,0,0,
    1103,0,17269,
    1077,903,0,
    0,0,0,
    0,0,0,
    31082,12791,0,
    28179,29258,24240,
    37307,15489,9906,
    21878,11411,688,
    18169,10048,12005,
    35900,35967,13410,
    2331,23301,11663,
    23658,17558,7418,
    32110,20121,19118,
    34595,17880,6978,
    0,8241,7910,
    26277,18112,4394,
    27452,16348,30033,
    28764,28299,1082,
    20334,19300,12874,
    28836,16918,11668,
    26346,17895,11657
], dtype=float)

REFERENCE_FWHM = np.array([
    30,30,2.2,30,30,30,30,30,30,30,3.9,30,
    0.32,30,30,
    0.37,1.1,0.22,
    0.18,30,30,
    0.25,0.40,4.6,
    30,0.21,0.15,
    0.27,0.24,30,
    0.21,30,30,
    0.8,30,0.21,
    0.48,2,30,
    30,30,30,
    30,30,30,
    0.2,0.3,30,
    0.26,0.29,0.24,
    0.18,0.28,0.26,
    0.21,0.22,0.6,
    0.2,0.22,0.24,
    0.22,0.21,0.22,
    0.3,0.29,0.23,
    0.28,0.26,0.29,
    0.3,0.21,0.25,
    0.14,0.22,0.16,
    30,0.26,0.23,
    0.22,0.29,0.31,
    0.24,0.28,0.23,
    0.29,0.22,0.41,
    0.27,0.32,0.21,
    0.26,0.31,0.30,
    0.28,0.31,0.34
], dtype=float)


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")



def sanitize_project_folder_name(name: str) -> str:
    """Return a short Windows-safe folder name for a project."""
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", str(name).strip())
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._-")
    return cleaned[:60] or "MOF_BO_Project"


def unique_project_folder(parent: Path, project_name: str) -> Path:
    """Create a unique project folder path under a parent workspace folder."""
    parent = Path(parent)
    stem = sanitize_project_folder_name(project_name)
    base = parent / stem
    if not base.exists():
        return base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = parent / f"{stem}_{stamp}"
    if not candidate.exists():
        return candidate
    counter = 2
    while True:
        candidate = parent / f"{stem}_{stamp}_{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


def calc_q(intensity: object, fwhm: object, q_value: object = None) -> float:
    """Calculate q from intensity and FWHM. A directly entered q value has priority."""
    q_parsed = parse_float_or_nan(q_value)
    if np.isfinite(q_parsed):
        return float(q_parsed)
    intensity_parsed = parse_float_or_nan(intensity)
    fwhm_parsed = parse_float_or_nan(fwhm)
    if not np.isfinite(intensity_parsed) or not np.isfinite(fwhm_parsed):
        return np.nan
    if fwhm_parsed == 30 or fwhm_parsed <= 0 or intensity_parsed <= 0:
        return 0.0
    return float(int(round(float(intensity_parsed) / float(fwhm_parsed))))


def parse_float_or_nan(value: object) -> float:
    if value is None:
        return np.nan
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return np.nan
    try:
        return float(text)
    except Exception:
        return np.nan


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except Exception:
        return default


def normalize_fraction(value: object, default: float = 0.30) -> float:
    """Parse either a fraction such as 0.30 or a percent such as 30."""
    fraction = parse_float_or_nan(value)
    if not np.isfinite(fraction) or fraction <= 0:
        fraction = default
    if fraction > 1.0 and fraction <= 100.0:
        fraction = fraction / 100.0
    return min(max(float(fraction), 0.01), 1.0)


def scale_features(X: np.ndarray) -> np.ndarray:
    """Scale original integer features to [0, 1] using the fixed synthesis bounds."""
    X = np.asarray(X, dtype=float)
    out = np.zeros_like(X, dtype=float)
    for j, feature in enumerate(FEATURES):
        lo, hi, _step = BOUNDS[feature]
        out[:, j] = (X[:, j] - lo) / max(hi - lo, 1e-12)
    return np.clip(out, 0.0, 1.0)


def unscale_features(X_scaled: np.ndarray) -> np.ndarray:
    X_scaled = np.asarray(X_scaled, dtype=float)
    out = np.zeros_like(X_scaled, dtype=float)
    for j, feature in enumerate(FEATURES):
        lo, hi, step = BOUNDS[feature]
        raw = lo + X_scaled[:, j] * (hi - lo)
        raw = np.round(raw / step) * step
        out[:, j] = np.clip(raw, lo, hi)
    return out.astype(int)


def condition_tuple(row: object) -> Tuple[int, int, int, int, int]:
    return tuple(int(float(row[f])) for f in FEATURES)


def valid_condition_values(values: Dict[str, object]) -> Tuple[bool, str]:
    for feature in FEATURES:
        lo, hi, step = BOUNDS[feature]
        try:
            raw = float(values[feature])
        except Exception:
            return False, f"{feature} must be a number."
        if not raw.is_integer():
            return False, f"{feature} must be an integer."
        val = int(raw)
        if val < lo or val > hi:
            return False, f"{feature} must be between {lo} and {hi}."
        if (val - lo) % step != 0:
            return False, f"{feature} must use step size {step}."
    return True, ""


def make_reference_df() -> pd.DataFrame:
    df = pd.DataFrame(REFERENCE_X, columns=FEATURES)
    df[FEATURES] = df[FEATURES].astype(int)
    df["intensity"] = REFERENCE_INTENSITY
    df["fwhm"] = REFERENCE_FWHM
    df["q"] = [calc_q(i, f) for i, f in zip(df["intensity"], df["fwhm"])]
    # Average duplicate reference conditions.
    grouped = df.groupby(FEATURES, as_index=False).agg(
        intensity=("intensity", "mean"),
        fwhm=("fwhm", "mean"),
        q=("q", "mean"),
    )
    return grouped


REFERENCE_DF = make_reference_df()


class BOEngine:
    """Small local Bayesian Optimization engine for the student GUI."""

    def __init__(self, config: Dict[str, object]):
        self.config = config
        self.rng = np.random.default_rng(int(config.get("seed", 42)))

    def _target_to_model(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if bool(self.config.get("use_log1p_target", True)):
            return np.log1p(np.maximum(y, 0.0))
        return y

    def _target_from_model(self, y_model: np.ndarray) -> np.ndarray:
        y_model = np.asarray(y_model, dtype=float)
        if bool(self.config.get("use_log1p_target", True)):
            return np.expm1(y_model)
        return y_model

    def _fit_model(self, X_original: np.ndarray, y_raw: np.ndarray) -> LightweightGaussianProcess:
        X_scaled = scale_features(X_original)
        y_model = self._target_to_model(y_raw)
        model = LightweightGaussianProcess(
            kernel_name=str(self.config.get("kernel", "matern52")),
            random_state=int(self.config.get("seed", 42)),
        )
        model.fit(X_scaled, y_model)
        return model

    def _acquisition(self, mu: np.ndarray, sd: np.ndarray, best: float) -> np.ndarray:
        sd = np.maximum(np.asarray(sd, dtype=float), 1e-9)
        mu = np.asarray(mu, dtype=float)
        xi = 0.01
        z = (mu - best - xi) / sd
        acq_name = str(self.config.get("acquisition", "ei")).lower()
        if acq_name == "pi":
            return normal_cdf(z)
        improvement = mu - best - xi
        return improvement * normal_cdf(z) + sd * normal_pdf(z)

    def _next_batch_count(self, all_rows: Optional[pd.DataFrame]) -> int:
        """Return the 1-based number of the batch that is about to be suggested."""
        if all_rows is None or len(all_rows) == 0 or "round" not in all_rows.columns:
            return 1
        rounds = pd.to_numeric(all_rows["round"], errors="coerce").dropna()
        if len(rounds) == 0:
            return 1
        # Project rounds are stored as 0, 1, 2, ... in experiments.csv.
        # The next batch count is therefore max_round + 2 in human 1-based language.
        return int(rounds.max()) + 2

    def _planned_iteration_bounds(self) -> Tuple[int, int]:
        """Return the saved estimated BO iteration range.

        Older projects may only have planned_total_batches. In that case, use
        that value as both the lower and upper estimate.
        """
        legacy_n = max(1, safe_int(self.config.get("planned_total_batches", 15), 15))
        low = max(1, safe_int(self.config.get("planned_iteration_min", legacy_n), legacy_n))
        high = max(1, safe_int(self.config.get("planned_iteration_max", legacy_n), legacy_n))
        if high < low:
            low, high = high, low
        return int(low), int(high)

    def _planned_iteration_range(self) -> Tuple[int, int, float]:
        """Return the lower estimate, upper estimate, and average number of batches."""
        low, high = self._planned_iteration_bounds()
        return low, high, (float(low) + float(high)) / 2.0

    def _normalized_transfer_fraction(self) -> float:
        """Return the transfer fraction after accepting either 0.30 or 30 as 30 percent."""
        return normalize_fraction(self.config.get("transfer_prior_fraction", 0.30), 0.30)

    def _auto_transfer_prior_rounds(self) -> int:
        """Calculate M from the average of the planned iteration range.

        Auto rule:
            M = floor(mean(iteration_min, iteration_max) * transfer_fraction)

        The result is clipped to the planned range and kept at least 1 for a
        normal transfer-enabled project.
        """
        low, high = self._planned_iteration_bounds()
        avg_batches = (float(low) + float(high)) / 2.0
        fraction = parse_float_or_nan(self.config.get("transfer_prior_fraction", 0.30))
        if not np.isfinite(fraction) or fraction <= 0:
            fraction = 0.30
        if fraction > 1.0 and fraction <= 100.0:
            fraction = fraction / 100.0
        fraction = min(max(float(fraction), 0.01), 1.0)
        auto_m = int(math.floor(avg_batches * fraction))
        return max(1, min(high, auto_m))

    def _transfer_prior_rounds(self) -> int:
        """Number of first suggestion batches that may use the calibrated transfer prior."""
        mode = str(self.config.get("transfer_rounds_mode", "auto")).strip().lower()
        auto_m = self._auto_transfer_prior_rounds()
        _low, high = self._planned_iteration_bounds()
        if mode == "manual":
            manual_m = max(0, safe_int(self.config.get("transfer_prior_rounds", auto_m), auto_m))
            return min(high, manual_m)
        return auto_m

    def _transfer_active(self, all_rows: Optional[pd.DataFrame]) -> bool:
        """Check whether calibrated transfer should be used for the next suggestion."""
        if not bool(self.config.get("use_reference_prior", False)):
            return False
        m_batches = self._transfer_prior_rounds()
        if m_batches <= 0:
            return False
        return self._next_batch_count(all_rows) <= m_batches

    def _fit_reference_model(self) -> LightweightGaussianProcess:
        """Fit a reference GP on the built-in benchmark data only."""
        X_ref = REFERENCE_DF[FEATURES].to_numpy(dtype=float)
        y_ref = REFERENCE_DF["q"].to_numpy(dtype=float)
        y_ref_model = self._target_to_model(y_ref)
        model = LightweightGaussianProcess(
            kernel_name=str(self.config.get("kernel", "matern52")),
            random_state=int(self.config.get("seed", 42)) + 1009,
        )
        model.fit(scale_features(X_ref), y_ref_model)
        return model

    def _fit_calibrated_transfer(self, completed_rows: pd.DataFrame) -> Dict[str, object]:
        """Fit Mode C calibrated transfer prior plus an optional residual GP.

        The reference GP gives mu_ref(x). Student results calibrate it as
        y_student ~= a + b * mu_ref(x). A second GP is fitted to the residuals
        when enough current-project data are available.
        """
        completed = completed_rows.copy()
        if len(completed) > 0:
            completed = completed[pd.to_numeric(completed["q"], errors="coerce").notna()].copy()

        ref_model = self._fit_reference_model()
        out: Dict[str, object] = {
            "ref_model": ref_model,
            "a": 0.0,
            "b": 1.0,
            "residual_model": None,
            "residual_sd": 0.35,
            "n_student": int(len(completed)),
            "calibration_mode": "reference only; no student calibration yet",
        }

        if len(completed) == 0:
            return out

        X_student = completed[FEATURES].to_numpy(dtype=float)
        y_student = completed["q"].to_numpy(dtype=float)
        y_student_model = self._target_to_model(y_student)
        mu_ref_student, _sd_ref_student = ref_model.predict(scale_features(X_student), return_std=True)
        mu_ref_student = np.asarray(mu_ref_student, dtype=float).reshape(-1)

        if len(completed) == 1:
            # With one point, estimate only an offset. The scale term b cannot be
            # identified yet, so it stays at 1 until more feedback arrives.
            a = float(y_student_model[0] - mu_ref_student[0])
            b = 1.0
            calibration_mode = "offset-only calibration from one student point"
        else:
            # Ridge-calibrated line: y_student ~= a + b * mu_ref.
            # The small ridge term shrinks b toward 1 in very early rounds, which
            # avoids an unstable scale estimate from only a few noisy points.
            A = np.column_stack([np.ones(len(mu_ref_student)), mu_ref_student])
            ridge = max(0.05, 1.0 / max(len(mu_ref_student), 1))
            A_aug = np.vstack([A, [0.0, math.sqrt(ridge)]])
            y_aug = np.concatenate([y_student_model, [math.sqrt(ridge) * 1.0]])
            coef, *_ = np.linalg.lstsq(A_aug, y_aug, rcond=None)
            a = float(coef[0])
            b = float(np.clip(coef[1], -2.0, 3.0))
            calibration_mode = f"linear calibration from {len(completed)} student points"

        prior_student = a + b * mu_ref_student
        residual = y_student_model - prior_student
        residual_sd = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.35
        if not np.isfinite(residual_sd) or residual_sd < 1e-6:
            residual_sd = 0.10

        residual_model = None
        if len(completed) >= 3 and np.nanstd(residual) > 1e-9:
            residual_model = LightweightGaussianProcess(
                kernel_name=str(self.config.get("kernel", "matern52")),
                random_state=int(self.config.get("seed", 42)) + 2027,
            )
            residual_model.fit(scale_features(X_student), residual)

        out.update({
            "a": a,
            "b": b,
            "residual_model": residual_model,
            "residual_sd": residual_sd,
            "n_student": int(len(completed)),
            "calibration_mode": calibration_mode,
        })
        return out

    def _predict_calibrated_transfer(self, transfer: Dict[str, object], X_original: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict model-scale mean and standard deviation from the calibrated transfer model."""
        X_scaled = scale_features(np.asarray(X_original, dtype=float))
        ref_model = transfer["ref_model"]
        mu_ref, sd_ref = ref_model.predict(X_scaled, return_std=True)  # type: ignore[union-attr]
        mu_ref = np.asarray(mu_ref, dtype=float)
        sd_ref = np.maximum(np.asarray(sd_ref, dtype=float), 1e-9)
        a = float(transfer.get("a", 0.0))
        b = float(transfer.get("b", 1.0))
        mu_prior = a + b * mu_ref
        sd_prior = abs(b) * sd_ref

        residual_model = transfer.get("residual_model", None)
        if residual_model is not None:
            mu_resid, sd_resid = residual_model.predict(X_scaled, return_std=True)  # type: ignore[union-attr]
            mu_resid = np.asarray(mu_resid, dtype=float)
            sd_resid = np.maximum(np.asarray(sd_resid, dtype=float), 1e-9)
        else:
            mu_resid = np.zeros(len(X_scaled), dtype=float)
            sd_resid = np.full(len(X_scaled), float(transfer.get("residual_sd", 0.35)), dtype=float)

        # Reference uncertainty is included but damped. The current-project residual
        # model is the main uncertainty source after student feedback is available.
        reference_weight = 0.35
        calibration_noise = float(transfer.get("residual_sd", 0.35)) * 0.25
        sd = np.sqrt(sd_resid ** 2 + (reference_weight * sd_prior) ** 2 + calibration_noise ** 2)
        mu = mu_prior + mu_resid
        return mu, np.maximum(sd, 1e-9)

    def _transfer_message(self, transfer: Dict[str, object], all_rows: Optional[pd.DataFrame]) -> str:
        next_batch = self._next_batch_count(all_rows)
        m_batches = self._transfer_prior_rounds()
        low, high = self._planned_iteration_bounds()
        avg_batches = (float(low) + float(high)) / 2.0
        fraction = parse_float_or_nan(self.config.get("transfer_prior_fraction", 0.30))
        if not np.isfinite(fraction) or fraction <= 0:
            fraction = 0.30
        if fraction > 1.0 and fraction <= 100.0:
            fraction = fraction / 100.0
        mode = str(self.config.get("transfer_rounds_mode", "auto")).strip().lower()
        schedule_text = f"M={m_batches} from {low}-{high} planned batch(es), mean={avg_batches:.1f}, fraction={fraction:.2g}, mode={mode}"
        return (
            f"Calibrated transfer prior is active for batch {next_batch} of the first {m_batches} batch(es) ({schedule_text}). "
            f"Calibration: y_student ≈ a + b * mu_ref(x), with a={float(transfer.get('a', 0.0)):.3g}, "
            f"b={float(transfer.get('b', 1.0)):.3g}, n_student={int(transfer.get('n_student', 0))}. "
            f"Mode: {transfer.get('calibration_mode', '')}. "
            "EI/PI improvement is referenced to the best current-project result when available."
        )

    def _generate_random_candidates(self, n: int, tried: set, extra_rows: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        rows: List[Tuple[int, int, int, int, int]] = []
        seen = set(tried)

        # Add local candidates around the best or recent experiments when available.
        if extra_rows is not None and len(extra_rows) > 0:
            sorted_rows = extra_rows.copy()
            if "q" in sorted_rows.columns:
                sorted_rows = sorted_rows.sort_values("q", ascending=False)
            for _idx, row in sorted_rows.head(8).iterrows():
                base = np.array([int(row[f]) for f in FEATURES], dtype=int)
                for _ in range(300):
                    offsets = self.rng.integers(-3, 4, size=len(FEATURES))
                    cand = base + offsets
                    for j, feature in enumerate(FEATURES):
                        lo, hi, step = BOUNDS[feature]
                        cand[j] = int(np.clip(round(cand[j] / step) * step, lo, hi))
                    tup = tuple(int(v) for v in cand)
                    if tup not in seen:
                        rows.append(tup)
                        seen.add(tup)
                    if len(rows) >= n // 3:
                        break
                if len(rows) >= n // 3:
                    break

        max_attempts = max(2000, n * 20)
        attempts = 0
        while len(rows) < n and attempts < max_attempts:
            attempts += 1
            cand = []
            for feature in FEATURES:
                lo, hi, step = BOUNDS[feature]
                choices = np.arange(lo, hi + 1, step, dtype=int)
                cand.append(int(self.rng.choice(choices)))
            tup = tuple(cand)
            if tup in seen:
                continue
            seen.add(tup)
            rows.append(tup)

        return pd.DataFrame(rows, columns=FEATURES)

    def _prepare_candidate_pool(self, candidate_pool: Optional[pd.DataFrame], tried: set, completed_rows: pd.DataFrame) -> pd.DataFrame:
        mode = str(self.config.get("candidate_mode", "random"))
        if mode == "csv" and candidate_pool is not None and len(candidate_pool) > 0:
            candidates = candidate_pool[FEATURES].copy()
            for feature in FEATURES:
                candidates[feature] = pd.to_numeric(candidates[feature], errors="coerce")
            candidates = candidates.dropna(subset=FEATURES)
            candidates[FEATURES] = candidates[FEATURES].astype(int)
            candidates = candidates.drop_duplicates(subset=FEATURES)
            candidates["_tuple"] = candidates.apply(condition_tuple, axis=1)
            candidates = candidates[~candidates["_tuple"].isin(tried)].drop(columns=["_tuple"])
            if len(candidates) > 0:
                return candidates.reset_index(drop=True)

        n = int(self.config.get("random_candidate_count", 15000))
        return self._generate_random_candidates(n, tried, completed_rows).reset_index(drop=True)

    def _select_diverse_batch(
        self,
        candidates: pd.DataFrame,
        acq: np.ndarray,
        pred_raw: np.ndarray,
        pred_sd_raw: np.ndarray,
        k: int,
    ) -> pd.DataFrame:
        if len(candidates) == 0:
            return pd.DataFrame(columns=FEATURES)
        acq = np.asarray(acq, dtype=float)
        if len(candidates) <= k:
            out = candidates.copy()
            out["predicted_q_mean"] = pred_raw[:len(out)]
            out["predicted_q_sd"] = pred_sd_raw[:len(out)]
            out["acquisition_value"] = acq[:len(out)]
            return out

        acq_safe = np.where(np.isfinite(acq), acq, -np.inf)
        std = np.nanstd(acq_safe[np.isfinite(acq_safe)]) if np.any(np.isfinite(acq_safe)) else 1.0
        mean = np.nanmean(acq_safe[np.isfinite(acq_safe)]) if np.any(np.isfinite(acq_safe)) else 0.0
        acq_z = (acq_safe - mean) / max(std, 1e-12)

        X_scaled = scale_features(candidates[FEATURES].to_numpy(dtype=float))
        chosen: List[int] = []
        remaining = list(range(len(candidates)))
        diversity_lambda = float(self.config.get("diversity_lambda", 0.03))

        for _ in range(k):
            if not remaining:
                break
            if not chosen or diversity_lambda <= 0:
                best_local = max(remaining, key=lambda i: acq_z[i])
            else:
                chosen_x = X_scaled[chosen]
                scores = []
                for i in remaining:
                    d = np.sqrt(((chosen_x - X_scaled[i]) ** 2).sum(axis=1)).min()
                    scores.append(acq_z[i] + diversity_lambda * d)
                best_local = remaining[int(np.argmax(scores))]
            chosen.append(best_local)
            remaining.remove(best_local)

        out = candidates.iloc[chosen].copy().reset_index(drop=True)
        out["predicted_q_mean"] = pred_raw[chosen]
        out["predicted_q_sd"] = pred_sd_raw[chosen]
        out["acquisition_value"] = acq[chosen]
        return out

    def _training_frame(self, completed_rows: pd.DataFrame) -> pd.DataFrame:
        """Build the current-project training frame.

        The reference benchmark is no longer mixed into this table
        as raw pseudo-observations. Reference data are handled by Mode C
        calibrated transfer prior instead.
        """
        completed = completed_rows.copy()
        if len(completed) > 0:
            completed = completed[pd.to_numeric(completed["q"], errors="coerce").notna()]
            completed = completed[FEATURES + ["q"]].copy()
            completed = completed.dropna(subset=FEATURES + ["q"])
        else:
            completed = pd.DataFrame(columns=FEATURES + ["q"])
        return completed

    def _condition_set(self, rows: pd.DataFrame) -> set:
        tried = set()
        if rows is not None and len(rows) > 0:
            for _idx, row in rows.dropna(subset=FEATURES).iterrows():
                try:
                    tried.add(condition_tuple(row))
                except Exception:
                    pass
        return tried

    def suggest(
        self,
        completed_rows: pd.DataFrame,
        all_rows: pd.DataFrame,
        candidate_pool: Optional[pd.DataFrame],
        batch_size: int,
    ) -> Tuple[pd.DataFrame, str]:
        """Return a batch of recommended conditions and a status message."""
        tried = self._condition_set(all_rows)
        completed = completed_rows.copy()
        completed = completed[pd.to_numeric(completed["q"], errors="coerce").notna()]
        train_df = self._training_frame(completed)
        transfer_active = self._transfer_active(all_rows)

        candidates = self._prepare_candidate_pool(candidate_pool, tried, completed)
        if len(candidates) == 0:
            raise RuntimeError("No valid candidate is available. Check the candidate CSV or project history.")

        if transfer_active:
            transfer = self._fit_calibrated_transfer(completed)
            X_cand_original = candidates[FEATURES].to_numpy(dtype=float)
            mu_model, sd_model = self._predict_calibrated_transfer(transfer, X_cand_original)
            if len(train_df) > 0:
                best_model = float(np.max(self._target_to_model(train_df["q"].to_numpy(dtype=float))))
            else:
                # Before any current-project results exist, EI/PI needs a
                # reference threshold. A high reference percentile produces a
                # useful mix of reference-guided exploitation and uncertainty.
                best_model = float(np.nanpercentile(mu_model, 75.0))
            acq = self._acquisition(mu_model, sd_model, best_model)
            pred_raw = self._target_from_model(mu_model)
            upper_raw = self._target_from_model(mu_model + sd_model)
            lower_raw = self._target_from_model(mu_model - sd_model)
            pred_sd_raw = np.maximum((upper_raw - lower_raw) / 2.0, 0.0)
            selected = self._select_diverse_batch(candidates, acq, pred_raw, pred_sd_raw, batch_size)
            return selected, self._transfer_message(transfer, all_rows)

        if len(train_df) < 3:
            sample_n = min(batch_size, len(candidates))
            pick = candidates.sample(n=sample_n, random_state=int(self.config.get("seed", 42))).copy().reset_index(drop=True)
            pick["predicted_q_mean"] = np.nan
            pick["predicted_q_sd"] = np.nan
            pick["acquisition_value"] = np.nan
            return pick, "Initial sampling was used because the project has fewer than three completed current-project q values."

        if train_df["q"].nunique(dropna=True) < 2:
            sample_n = min(batch_size, len(candidates))
            pick = candidates.sample(n=sample_n, random_state=int(self.config.get("seed", 42))).copy().reset_index(drop=True)
            constant_q = float(train_df["q"].iloc[0]) if len(train_df) else np.nan
            pick["predicted_q_mean"] = constant_q
            pick["predicted_q_sd"] = np.nan
            pick["acquisition_value"] = np.nan
            return pick, "Exploratory sampling was used because all completed q values are identical. BO will become model-driven once the results contain variation."

        X_train = train_df[FEATURES].to_numpy(dtype=float)
        y_train = train_df["q"].to_numpy(dtype=float)
        y_train_model = self._target_to_model(y_train)
        model = self._fit_model(X_train, y_train)

        X_cand = scale_features(candidates[FEATURES].to_numpy(dtype=float))
        mu_model, sd_model = model.predict(X_cand, return_std=True)
        best_model = float(np.max(y_train_model))
        acq = self._acquisition(mu_model, sd_model, best_model)

        pred_raw = self._target_from_model(mu_model)
        upper_raw = self._target_from_model(mu_model + sd_model)
        lower_raw = self._target_from_model(mu_model - sd_model)
        pred_sd_raw = np.maximum((upper_raw - lower_raw) / 2.0, 0.0)

        selected = self._select_diverse_batch(candidates, acq, pred_raw, pred_sd_raw, batch_size)
        message = (
            f"Fitted student-only GP with {len(train_df)} current-project training rows. "
            f"Calibrated transfer prior is off or outside the first {self._transfer_prior_rounds()} batch(es). "
            f"Suggestions were ranked by {str(self.config.get('acquisition', 'ei')).upper()} and batch diversity."
        )
        return selected, message

    def diagnostics(
        self,
        completed_rows: pd.DataFrame,
        all_rows: pd.DataFrame,
        candidate_pool: Optional[pd.DataFrame],
        max_candidates: int = 4000,
    ) -> Dict[str, object]:
        """Compute model-fit and acquisition diagnostics for visualization."""
        completed = completed_rows.copy()
        completed = completed[pd.to_numeric(completed["q"], errors="coerce").notna()]
        train_df = self._training_frame(completed)
        result: Dict[str, object] = {
            "model_ready": False,
            "status": "Not enough completed q values to fit a GP model.",
            "train_df": train_df,
            "completed_pred": pd.DataFrame(),
            "candidates": pd.DataFrame(),
        }

        transfer_active = self._transfer_active(all_rows)

        completed_tried = self._condition_set(completed)
        plot_config = dict(self.config)
        plot_config["random_candidate_count"] = min(int(plot_config.get("random_candidate_count", 15000)), int(max_candidates))
        tmp_engine = BOEngine(plot_config)
        candidates = tmp_engine._prepare_candidate_pool(candidate_pool, completed_tried, completed)
        if len(candidates) > max_candidates:
            candidates = candidates.sample(n=max_candidates, random_state=int(self.config.get("seed", 42))).reset_index(drop=True)

        open_suggestions = all_rows[all_rows["status"].astype(str) == "suggested"].copy() if all_rows is not None and len(all_rows) else pd.DataFrame()
        if len(open_suggestions) > 0:
            sug_features = open_suggestions[FEATURES].copy()
            candidates = pd.concat([candidates[FEATURES], sug_features], ignore_index=True).drop_duplicates(subset=FEATURES).reset_index(drop=True)

        if transfer_active:
            transfer = self._fit_calibrated_transfer(completed)
            if len(completed) > 0:
                X_completed = completed[FEATURES].to_numpy(dtype=float)
                mu_c, sd_c = self._predict_calibrated_transfer(transfer, X_completed)
                comp_pred = completed.copy().reset_index(drop=True)
                comp_pred["model_predicted_q"] = self._target_from_model(mu_c)
                comp_pred["model_predicted_sd"] = np.maximum((self._target_from_model(mu_c + sd_c) - self._target_from_model(mu_c - sd_c)) / 2.0, 0.0)
            else:
                comp_pred = pd.DataFrame()

            if len(candidates) > 0:
                X_cand_original = candidates[FEATURES].to_numpy(dtype=float)
                mu_model, sd_model = self._predict_calibrated_transfer(transfer, X_cand_original)
                if len(train_df) > 0:
                    best_model = float(np.max(self._target_to_model(train_df["q"].to_numpy(dtype=float))))
                else:
                    best_model = float(np.nanpercentile(mu_model, 75.0))
                acq = self._acquisition(mu_model, sd_model, best_model)
                candidates = candidates.copy().reset_index(drop=True)
                candidates["predicted_q_mean"] = self._target_from_model(mu_model)
                candidates["predicted_q_sd"] = np.maximum((self._target_from_model(mu_model + sd_model) - self._target_from_model(mu_model - sd_model)) / 2.0, 0.0)
                candidates["acquisition_value"] = acq
                suggestion_set = self._condition_set(open_suggestions)
                candidates["is_open_suggestion"] = [condition_tuple(row) in suggestion_set for _idx, row in candidates.iterrows()]

            result.update({
                "model_ready": True,
                "status": self._transfer_message(transfer, all_rows) + f" Diagnostics use {len(candidates)} candidate points.",
                "completed_pred": comp_pred,
                "candidates": candidates,
                "model": transfer,
            })
            return result

        if len(train_df) < 3:
            result["status"] = "At least three completed current-project q values are needed before the student-only GP diagnostics are shown."
            return result
        if train_df["q"].nunique(dropna=True) < 2:
            result["status"] = "The student-only GP is not informative yet because all completed q values are identical."
            return result

        X_train = train_df[FEATURES].to_numpy(dtype=float)
        y_train = train_df["q"].to_numpy(dtype=float)
        y_train_model = self._target_to_model(y_train)
        model = self._fit_model(X_train, y_train)

        if len(completed) > 0:
            X_completed = scale_features(completed[FEATURES].to_numpy(dtype=float))
            mu_c, sd_c = model.predict(X_completed, return_std=True)
            comp_pred = completed.copy().reset_index(drop=True)
            comp_pred["model_predicted_q"] = self._target_from_model(mu_c)
            comp_pred["model_predicted_sd"] = np.maximum((self._target_from_model(mu_c + sd_c) - self._target_from_model(mu_c - sd_c)) / 2.0, 0.0)
        else:
            comp_pred = pd.DataFrame()

        if len(candidates) > 0:
            X_cand = scale_features(candidates[FEATURES].to_numpy(dtype=float))
            mu_model, sd_model = model.predict(X_cand, return_std=True)
            best_model = float(np.max(y_train_model))
            acq = self._acquisition(mu_model, sd_model, best_model)
            candidates = candidates.copy().reset_index(drop=True)
            candidates["predicted_q_mean"] = self._target_from_model(mu_model)
            candidates["predicted_q_sd"] = np.maximum((self._target_from_model(mu_model + sd_model) - self._target_from_model(mu_model - sd_model)) / 2.0, 0.0)
            candidates["acquisition_value"] = acq
            suggestion_set = self._condition_set(open_suggestions)
            candidates["is_open_suggestion"] = [condition_tuple(row) in suggestion_set for _idx, row in candidates.iterrows()]

        result.update({
            "model_ready": True,
            "status": f"Student-only GP diagnostics use {len(train_df)} current-project training rows and {len(candidates)} candidate points.",
            "completed_pred": comp_pred,
            "candidates": candidates,
            "model": model,
        })
        return result




class RowEditor(tk.Toplevel):
    """Dialog for adding or editing one experiment row."""

    def __init__(
        self,
        master,
        title: str,
        initial: Optional[Dict[str, object]] = None,
        completed_default: bool = True,
        result_mode: bool = False,
    ):
        super().__init__(master)
        self.title(title)
        self.resizable(False, False)
        self.result: Optional[Dict[str, object]] = None
        self.initial = initial or {}
        self.result_mode = bool(result_mode)

        self.vars: Dict[str, tk.StringVar] = {}
        form = ttk.Frame(self, padding=12)
        form.grid(row=0, column=0, sticky="nsew")
        form.columnconfigure(1, weight=1)

        row = 0
        for feature in FEATURES:
            lo, hi, step = BOUNDS[feature]
            ttk.Label(form, text=f"{feature} ({lo} to {hi}, step {step})").grid(row=row, column=0, sticky="w", pady=3)
            var = tk.StringVar(value=str(self.initial.get(feature, "")))
            self.vars[feature] = var
            ttk.Entry(form, textvariable=var, width=26).grid(row=row, column=1, sticky="ew", pady=3)
            row += 1

        ttk.Separator(form, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 8))
        row += 1

        for label, key in [("Intensity", "intensity"), ("FWHM", "fwhm")]:
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=3)
            var = tk.StringVar(value="" if pd.isna(self.initial.get(key, "")) else str(self.initial.get(key, "")))
            self.vars[key] = var
            ttk.Entry(form, textvariable=var, width=26).grid(row=row, column=1, sticky="ew", pady=3)
            row += 1

        current_q = parse_float_or_nan(self.initial.get("q", ""))
        auto_from_current = calc_q(self.initial.get("intensity", ""), self.initial.get("fwhm", ""), None)
        q_is_manual = bool(np.isfinite(current_q) and (not np.isfinite(auto_from_current) or abs(current_q - auto_from_current) > 1e-9))
        self.override_q_var = tk.BooleanVar(value=q_is_manual)

        ttk.Label(form, text="q").grid(row=row, column=0, sticky="w", pady=3)
        q_initial = "" if not np.isfinite(current_q) else f"{current_q:.12g}"
        if not q_initial and np.isfinite(auto_from_current):
            q_initial = f"{auto_from_current:.12g}"
        self.vars["q"] = tk.StringVar(value=q_initial)
        self.q_entry = ttk.Entry(form, textvariable=self.vars["q"], width=26)
        self.q_entry.grid(row=row, column=1, sticky="ew", pady=3)
        row += 1

        self.auto_q_label_var = tk.StringVar(value="")
        ttk.Checkbutton(
            form,
            text="Manually override q",
            variable=self.override_q_var,
            command=self._update_auto_q,
        ).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Label(form, textvariable=self.auto_q_label_var, style="Hint.TLabel").grid(row=row, column=1, sticky="w", pady=3)
        row += 1

        ttk.Label(form, text="Status").grid(row=row, column=0, sticky="w", pady=3)
        if self.result_mode:
            status_default = "completed"
        else:
            status_default = self.initial.get("status", "completed" if completed_default else "suggested")
        self.vars["status"] = tk.StringVar(value=str(status_default))
        status_box = ttk.Combobox(form, textvariable=self.vars["status"], values=["suggested", "completed"], state="readonly", width=23)
        status_box.grid(row=row, column=1, sticky="ew", pady=3)
        row += 1

        ttk.Label(form, text="Notes").grid(row=row, column=0, sticky="w", pady=3)
        notes_default = "" if pd.isna(self.initial.get("notes", "")) else str(self.initial.get("notes", ""))
        self.vars["notes"] = tk.StringVar(value=notes_default)
        ttk.Entry(form, textvariable=self.vars["notes"], width=26).grid(row=row, column=1, sticky="ew", pady=3)
        row += 1

        info = ttk.Label(
            form,
            text=(
                "q is calculated automatically as round(Intensity / FWHM). "
                "Turn on manual override only if you need to enter q directly. "
                "Rows with a saved q value are treated as completed data for the next BO step."
            ),
            wraplength=420,
            foreground="#555555",
        )
        info.grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 3))
        row += 1

        buttons = ttk.Frame(form)
        buttons.grid(row=row, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="Cancel", command=self.destroy).grid(row=0, column=0, padx=4)
        ttk.Button(buttons, text="Save", command=self._save).grid(row=0, column=1, padx=4)

        self.vars["intensity"].trace_add("write", lambda *_args: self._update_auto_q())
        self.vars["fwhm"].trace_add("write", lambda *_args: self._update_auto_q())
        self._update_auto_q()

        self.bind("<Return>", lambda _event: self._save())
        self.bind("<Escape>", lambda _event: self.destroy())
        self.grab_set()
        self.transient(master)
        self.wait_visibility()
        self.focus_force()

    def _update_auto_q(self):
        q_auto = calc_q(self.vars["intensity"].get(), self.vars["fwhm"].get(), None)
        if np.isfinite(q_auto):
            self.auto_q_label_var.set(f"Auto q = {q_auto:.12g}")
        else:
            self.auto_q_label_var.set("Auto q is not available yet")

        if self.override_q_var.get():
            self.q_entry.configure(state="normal")
        else:
            self.q_entry.configure(state="normal")
            self.vars["q"].set("" if not np.isfinite(q_auto) else f"{q_auto:.12g}")
            self.q_entry.configure(state="readonly")

    def _save(self):
        values = {key: var.get().strip() for key, var in self.vars.items()}
        ok, msg = valid_condition_values(values)
        if not ok:
            messagebox.showerror("Invalid condition", msg, parent=self)
            return

        intensity = parse_float_or_nan(values.get("intensity"))
        fwhm = parse_float_or_nan(values.get("fwhm"))
        if self.override_q_var.get():
            q = calc_q(values.get("intensity"), values.get("fwhm"), values.get("q"))
        else:
            q = calc_q(values.get("intensity"), values.get("fwhm"), None)

        status = values.get("status", "completed")
        if self.result_mode and np.isfinite(q):
            status = "completed"
        if status == "suggested" and np.isfinite(q):
            status = "completed"
        if status == "completed" and not np.isfinite(q):
            messagebox.showerror("Missing result", "A completed row needs q, or intensity and FWHM so q can be calculated.", parent=self)
            return

        out = dict(self.initial)
        for feature in FEATURES:
            out[feature] = int(float(values[feature]))
        out["intensity"] = intensity if np.isfinite(intensity) else ""
        out["fwhm"] = fwhm if np.isfinite(fwhm) else ""
        out["q"] = q if np.isfinite(q) else ""
        out["status"] = status
        out["notes"] = values.get("notes", "")
        self.result = out
        self.destroy()


class NewProjectDialog(tk.Toplevel):
    """Dialog used to create a new project in its own folder."""

    def __init__(self, master):
        super().__init__(master)
        self.title("Create New Project")
        self.resizable(False, False)
        self.result = None

        frame = ttk.Frame(self, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)
        self.vars = {
            "project_name": tk.StringVar(value="MOF BO Project"),
            "kernel": tk.StringVar(value="matern52"),
            "acquisition": tk.StringVar(value="ei"),
            "batch_size": tk.StringVar(value="3"),
            "iteration_min": tk.StringVar(value="10"),
            "iteration_max": tk.StringVar(value="20"),
            "candidate_mode": tk.StringVar(value="random"),
            "use_reference_prior": tk.BooleanVar(value=False),
            "transfer_prior_fraction": tk.StringVar(value="0.30"),
            "manual_transfer_rounds": tk.BooleanVar(value=False),
            "transfer_prior_rounds": tk.StringVar(value="4"),
            "parent_folder": tk.StringVar(value=str(Path.home() / "MOF_BO_Projects")),
        }

        title = ttk.Label(frame, text="Create a separate folder for this project", style="Header.TLabel")
        title.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 4))
        ttk.Label(
            frame,
            text=(
                "Choose a parent workspace folder. The app will create a new project subfolder inside it. "
                "The defaults match the teaching workflow: 3 experiments per batch and an estimated 10-20 BO iterations."
            ),
            wraplength=620,
            style="Hint.TLabel",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 10))

        row = 2
        ttk.Label(frame, text="Project name").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.vars["project_name"], width=38).grid(row=row, column=1, columnspan=3, sticky="ew", pady=3)
        row += 1

        ttk.Label(frame, text="Parent workspace folder").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.vars["parent_folder"], width=38).grid(row=row, column=1, columnspan=2, sticky="ew", pady=3)
        ttk.Button(frame, text="Browse", command=self._browse_parent).grid(row=row, column=3, padx=(6, 0), pady=3)
        row += 1

        ttk.Label(frame, text="Kernel").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Combobox(frame, textvariable=self.vars["kernel"], values=["matern52", "matern32"], state="readonly", width=18).grid(row=row, column=1, sticky="w", pady=3)
        ttk.Label(frame, text="Acquisition").grid(row=row, column=2, sticky="w", padx=(12, 0), pady=3)
        ttk.Combobox(frame, textvariable=self.vars["acquisition"], values=["ei", "pi"], state="readonly", width=10).grid(row=row, column=3, sticky="w", pady=3)
        row += 1

        ttk.Label(frame, text="Experiments per batch").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.vars["batch_size"], width=12).grid(row=row, column=1, sticky="w", pady=3)
        ttk.Label(frame, text="Default is 3 experiments per BO iteration", style="Hint.TLabel").grid(row=row, column=2, columnspan=2, sticky="w", padx=(8, 0), pady=3)
        row += 1

        ttk.Label(frame, text="Estimated BO iterations").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Label(frame, text="Lower").grid(row=row, column=1, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.vars["iteration_min"], width=8).grid(row=row, column=1, sticky="e", pady=3)
        ttk.Label(frame, text="Upper").grid(row=row, column=2, sticky="w", padx=(12, 0), pady=3)
        ttk.Entry(frame, textvariable=self.vars["iteration_max"], width=8).grid(row=row, column=2, sticky="e", pady=3)
        ttk.Label(frame, text="Default range is 10 to 20", style="Hint.TLabel").grid(row=row, column=3, sticky="w", padx=(8, 0), pady=3)
        row += 1

        self.plan_preview_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.plan_preview_var, style="Hint.TLabel", wraplength=620).grid(row=row, column=0, columnspan=4, sticky="w", pady=(2, 8))
        row += 1

        ttk.Label(frame, text="Candidate sampling mode").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Combobox(frame, textvariable=self.vars["candidate_mode"], values=["random", "csv"], state="readonly", width=18).grid(row=row, column=1, sticky="w", pady=3)
        row += 1

        ttk.Checkbutton(
            frame,
            text="Use calibrated transfer prior for early batches",
            variable=self.vars["use_reference_prior"],
            command=self._update_transfer_preview,
        ).grid(row=row, column=0, columnspan=4, sticky="w", pady=(8, 3))
        row += 1

        ttk.Label(
            frame,
            text=(
                "When checked, the app uses the built-in benchmark as a calibrated prior only for the early part of the project. "
                "The default M is suggested from the average of the iteration range: M = floor(mean(lower, upper) × 0.30)."
            ),
            wraplength=620,
            style="Hint.TLabel",
        ).grid(row=row, column=0, columnspan=4, sticky="w", pady=(2, 4))
        row += 1

        self.transfer_options_expanded = False
        self.transfer_options_button = ttk.Button(frame, text="▶ + Advanced transfer settings", command=self._toggle_transfer_options)
        self.transfer_options_button.grid(row=row, column=0, columnspan=4, sticky="w", pady=(2, 2))
        row += 1
        self.transfer_options_frame = ttk.Frame(frame, padding=(18, 4, 4, 4))
        ttk.Label(self.transfer_options_frame, text="Auto transfer fraction").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(self.transfer_options_frame, textvariable=self.vars["transfer_prior_fraction"], width=12).grid(row=0, column=1, sticky="w", padx=(8, 0), pady=3)
        ttk.Label(self.transfer_options_frame, text="Example: 0.30 or 30 means 30%", style="Hint.TLabel").grid(row=0, column=2, sticky="w", padx=(8, 0), pady=3)
        ttk.Checkbutton(self.transfer_options_frame, text="Manually override M", variable=self.vars["manual_transfer_rounds"], command=self._update_transfer_preview).grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(self.transfer_options_frame, textvariable=self.vars["transfer_prior_rounds"], width=12).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=3)
        self.transfer_preview_var = tk.StringVar(value="")
        ttk.Label(self.transfer_options_frame, textvariable=self.transfer_preview_var, wraplength=560, style="Hint.TLabel").grid(row=2, column=0, columnspan=3, sticky="w", pady=(2, 6))
        self.transfer_options_row = row
        row += 1

        self.preview_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.preview_var, style="Hint.TLabel", wraplength=620).grid(row=row, column=0, columnspan=4, sticky="w", pady=(0, 8))
        row += 1

        for key in ["project_name", "parent_folder"]:
            self.vars[key].trace_add("write", lambda *_args: self._update_preview())
        for key in ["batch_size", "iteration_min", "iteration_max", "transfer_prior_fraction", "transfer_prior_rounds"]:
            self.vars[key].trace_add("write", lambda *_args: self._update_plan_and_transfer_previews())
        self.vars["manual_transfer_rounds"].trace_add("write", lambda *_args: self._update_plan_and_transfer_previews())
        self.vars["use_reference_prior"].trace_add("write", lambda *_args: self._update_plan_and_transfer_previews())
        self._update_preview()
        self._update_plan_and_transfer_previews()

        buttons = ttk.Frame(frame)
        buttons.grid(row=row, column=0, columnspan=4, sticky="e")
        ttk.Button(buttons, text="Cancel", command=self.destroy).grid(row=0, column=0, padx=4)
        ttk.Button(buttons, text="Create Project Folder", style="Primary.TButton", command=self._create).grid(row=0, column=1, padx=4)

        self.bind("<Return>", lambda _event: self._create())
        self.bind("<Escape>", lambda _event: self.destroy())
        self.grab_set()
        self.transient(master)
        self.wait_visibility()
        self.focus_force()

    def _toggle_transfer_options(self):
        self.transfer_options_expanded = not self.transfer_options_expanded
        self.transfer_options_button.configure(text=("▼ + Advanced transfer settings" if self.transfer_options_expanded else "▶ + Advanced transfer settings"))
        if self.transfer_options_expanded:
            self.transfer_options_frame.grid(row=self.transfer_options_row, column=0, columnspan=4, sticky="ew", pady=(0, 6))
        else:
            self.transfer_options_frame.grid_remove()

    def _browse_parent(self):
        folder = filedialog.askdirectory(title="Choose parent workspace folder", parent=self)
        if folder:
            self.vars["parent_folder"].set(folder)

    def _update_preview(self):
        name = self.vars["project_name"].get().strip() or "MOF BO Project"
        parent = Path(self.vars["parent_folder"].get().strip() or str(Path.home() / "MOF_BO_Projects"))
        folder = unique_project_folder(parent, name)
        self.preview_var.set(f"New project folder preview: {folder}")

    def _calculate_plan(self) -> Tuple[int, int, int, float, int, int]:
        batch_size = max(1, safe_int(self.vars["batch_size"].get(), 3))
        low = max(1, safe_int(self.vars["iteration_min"].get(), 10))
        high = max(1, safe_int(self.vars["iteration_max"].get(), 20))
        if high < low:
            low, high = high, low
        avg_iterations = (float(low) + float(high)) / 2.0
        total_min = batch_size * low
        total_max = batch_size * high
        return batch_size, low, high, avg_iterations, total_min, total_max

    def _normalized_fraction(self) -> float:
        fraction = parse_float_or_nan(self.vars["transfer_prior_fraction"].get())
        if not np.isfinite(fraction) or fraction <= 0:
            fraction = 0.30
        if fraction > 1.0 and fraction <= 100.0:
            fraction = fraction / 100.0
        return min(max(float(fraction), 0.01), 1.0)

    def _calculate_transfer_schedule(self) -> Tuple[int, int, float, float, int, str]:
        _batch, low, high, avg_iterations, _total_min, _total_max = self._calculate_plan()
        fraction = self._normalized_fraction()
        auto_m = max(1, min(high, int(math.floor(avg_iterations * fraction))))
        if bool(self.vars["manual_transfer_rounds"].get()):
            manual_m = max(1, safe_int(self.vars["transfer_prior_rounds"].get(), auto_m))
            return low, high, avg_iterations, fraction, min(high, manual_m), "manual"
        return low, high, avg_iterations, fraction, auto_m, "auto"

    def _update_plan_and_transfer_previews(self):
        if hasattr(self, "plan_preview_var"):
            batch, low, high, avg_iterations, total_min, total_max = self._calculate_plan()
            _lo, _hi, _avg, fraction, m_batches, mode = self._calculate_transfer_schedule()
            transfer_state = "ON" if bool(self.vars["use_reference_prior"].get()) else "OFF unless checked"
            mode_text = "manual" if mode == "manual" else "auto"
            self.plan_preview_var.set(
                f"Estimated total experiments: {total_min} to {total_max} "
                f"({batch} experiment(s) per batch × {low}-{high} BO iteration(s); average = {avg_iterations:.1f} batches). "
                f"Calibrated transfer prior is {transfer_state}; {mode_text} M = {m_batches} "
                f"using fraction {fraction:.2g}."
            )
        self._update_transfer_preview()

    def _update_transfer_preview(self):
        if not hasattr(self, "transfer_preview_var"):
            return
        low, high, avg_iterations, fraction, m_batches, mode = self._calculate_transfer_schedule()
        transfer_on = bool(self.vars["use_reference_prior"].get())
        if mode == "auto":
            self.transfer_preview_var.set(
                f"Calibrated transfer prior is {'ON' if transfer_on else 'OFF'}. Auto suggestion: "
                f"M = floor(mean({low}, {high}) × {fraction:.2g}) = floor({avg_iterations:.1f} × {fraction:.2g}) = {m_batches} batch(es). "
                "Open this panel only if you need to change the fraction or manually override M."
            )
        else:
            self.transfer_preview_var.set(
                f"Calibrated transfer prior is {'ON' if transfer_on else 'OFF'}. Manual schedule: "
                f"transfer will be used for the first M = {m_batches} batch(es). "
                f"The auto calculation would use floor({avg_iterations:.1f} × {fraction:.2g})."
            )

    def _create(self):
        name = self.vars["project_name"].get().strip() or "MOF BO Project"
        batch_size, low, high, avg_iterations, total_min, total_max = self._calculate_plan()
        if batch_size < 1:
            messagebox.showerror("Invalid setting", "Experiments per batch must be at least 1.", parent=self)
            return
        if low < 1 or high < 1 or high < low:
            messagebox.showerror("Invalid setting", "The iteration range must be positive, with upper >= lower.", parent=self)
            return
        transfer_low, transfer_high, _avg, transfer_fraction, transfer_rounds, transfer_mode = self._calculate_transfer_schedule()
        if not (0 < transfer_fraction <= 1):
            messagebox.showerror("Invalid setting", "Transfer fraction must be between 0 and 1, or a percentage from 1 to 100.", parent=self)
            return
        if bool(self.vars["use_reference_prior"].get()) and transfer_rounds < 1:
            messagebox.showerror("Invalid setting", "Transfer prior batches M must be at least 1 when calibrated transfer is enabled.", parent=self)
            return
        summary = (
            f"Project: {name}\n"
            f"Experiments per batch: {batch_size}\n"
            f"Estimated BO iterations: {low}-{high}\n"
            f"Estimated total experiments: {total_min}-{total_max}\n"
            f"Calibrated transfer prior: {'on' if bool(self.vars['use_reference_prior'].get()) else 'off'}\n"
            f"Transfer M: {transfer_rounds} batch(es), mode={transfer_mode}\n\n"
            "Create this project?"
        )
        if not messagebox.askyesno("Confirm project plan", summary, parent=self):
            return
        parent = Path(self.vars["parent_folder"].get().strip() or str(Path.home() / "MOF_BO_Projects"))
        folder = unique_project_folder(parent, name)
        self.result = {
            "folder": str(folder),
            "parent_folder": str(parent),
            "project_name": name,
            "kernel": self.vars["kernel"].get(),
            "acquisition": self.vars["acquisition"].get(),
            "batch_size": batch_size,
            "initial_samples": batch_size,
            "candidate_mode": self.vars["candidate_mode"].get(),
            "use_reference_prior": bool(self.vars["use_reference_prior"].get()),
            "planned_iteration_min": int(low),
            "planned_iteration_max": int(high),
            "planned_total_batches": int(round(avg_iterations)),
            "estimated_total_experiments_min": int(total_min),
            "estimated_total_experiments_max": int(total_max),
            "transfer_prior_fraction": transfer_fraction,
            "transfer_rounds_mode": transfer_mode,
            "transfer_prior_rounds": int(transfer_rounds),
        }
        self.destroy()


class CollapsibleSection(ttk.Frame):
    """Small expandable panel used to keep advanced actions out of the main workflow."""

    def __init__(self, master, title: str, expanded: bool = False):
        super().__init__(master)
        self.title = title
        self.expanded = bool(expanded)
        self.header = ttk.Button(self, text="", command=self.toggle)
        self.header.pack(anchor="w", pady=(2, 2))
        self.body = ttk.Frame(self, padding=(18, 4, 4, 4))
        self._sync()

    def _sync(self):
        arrow = "▼" if self.expanded else "▶"
        self.header.configure(text=f"{arrow} {self.title}")
        if self.expanded:
            self.body.pack(fill="x")
        else:
            self.body.pack_forget()

    def toggle(self):
        self.expanded = not self.expanded
        self._sync()


class BOStudentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} {APP_VERSION}")
        self.geometry("1180x760")
        self.minsize(980, 640)

        self.project_path: Optional[Path] = None
        self.project_config: Dict[str, object] = DEFAULT_CONFIG.copy()
        self.experiments = pd.DataFrame(columns=EXPERIMENT_COLUMNS)
        self.candidate_pool: Optional[pd.DataFrame] = None
        self.current_canvas: Optional[FigureCanvasTkAgg] = None

        self._build_style()
        self._build_menu()
        self._build_layout()
        self._refresh_all()

    def _build_style(self):
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Hint.TLabel", foreground="#555555")
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 5))
        style.configure("Action.TButton", padding=(9, 4))
        style.configure("Warning.TLabel", foreground="#8a4b00", font=("Segoe UI", 9, "bold"))

    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New Project", command=self.new_project)
        file_menu.add_command(label="Load Project", command=self.load_project_dialog)
        file_menu.add_command(label="Save Project", command=self.save_project)
        file_menu.add_separator()
        file_menu.add_command(label="Import Experiments CSV", command=self.import_experiments_csv)
        file_menu.add_command(label="Export Experiments CSV", command=self.export_experiments_csv)
        file_menu.add_command(label="Import Candidate CSV", command=self.import_candidate_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Suggest New Experiments", command=self.suggest_next_batch)
        tools_menu.add_command(label="Open Project Folder", command=self.open_project_folder)
        tools_menu.add_command(label="Reload Project Files", command=self.reload_project)
        tools_menu.add_command(label="Recalculate q from Intensity/FWHM", command=self.recalculate_q_for_all)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        self.configure(menu=menubar)

    def _build_layout(self):
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="No project loaded.")
        self.project_label_var = tk.StringVar(value="Project: none")

        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text=APP_NAME, style="Title.TLabel").pack(side="left")
        ttk.Label(header, textvariable=self.project_label_var, style="Hint.TLabel").pack(side="right")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        self.project_tab = ttk.Frame(self.notebook, padding=10)
        self.suggest_tab = ttk.Frame(self.notebook, padding=10)
        self.data_tab = ttk.Frame(self.notebook, padding=10)
        self.viz_tab = ttk.Frame(self.notebook, padding=10)
        self.help_tab = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.project_tab, text="Project")
        self.notebook.add(self.suggest_tab, text="Suggestions")
        self.notebook.add(self.data_tab, text="Data Table")
        self.notebook.add(self.viz_tab, text="Visualization")
        self.notebook.add(self.help_tab, text="Help")

        self._build_project_tab()
        self._build_suggestion_tab()
        self._build_data_tab()
        self._build_visualization_tab()
        self._build_help_tab()

        status = ttk.Label(root, textvariable=self.status_var, anchor="w", relief="sunken", padding=(6, 3))
        status.pack(fill="x", pady=(8, 0))

    def _build_project_tab(self):
        top = ttk.Frame(self.project_tab)
        top.pack(fill="x")
        ttk.Button(top, text="New Project", command=self.new_project).pack(side="left", padx=4)
        ttk.Button(top, text="Load Project", command=self.load_project_dialog).pack(side="left", padx=4)
        ttk.Button(top, text="Save Project", command=self.save_project).pack(side="left", padx=4)
        ttk.Button(top, text="Open Project Folder", command=self.open_project_folder).pack(side="left", padx=4)
        ttk.Button(top, text="Reload Files", command=self.reload_project).pack(side="left", padx=4)

        settings = ttk.LabelFrame(self.project_tab, text="Project Settings", padding=10)
        settings.pack(fill="x", pady=12)
        self.setting_vars = {
            "project_name": tk.StringVar(),
            "kernel": tk.StringVar(),
            "acquisition": tk.StringVar(),
            "batch_size": tk.StringVar(),
            "initial_samples": tk.StringVar(),
            "candidate_mode": tk.StringVar(),
            "random_candidate_count": tk.StringVar(),
            "diversity_lambda": tk.StringVar(),
            "use_log1p_target": tk.BooleanVar(),
            "use_reference_prior": tk.BooleanVar(),
            "planned_iteration_min": tk.StringVar(),
            "planned_iteration_max": tk.StringVar(),
            "planned_total_batches": tk.StringVar(),
            "estimated_total_experiments_min": tk.StringVar(),
            "estimated_total_experiments_max": tk.StringVar(),
            "transfer_prior_fraction": tk.StringVar(),
            "transfer_rounds_mode": tk.StringVar(),
            "transfer_prior_rounds": tk.StringVar(),
        }

        rows = [
            ("Project name", "project_name", "entry"),
            ("Kernel", "kernel", "kernel"),
            ("Acquisition", "acquisition", "acquisition"),
            ("Experiments per batch", "batch_size", "entry"),
            ("Candidate mode", "candidate_mode", "candidate_mode"),
            ("Random candidates per BO step", "random_candidate_count", "entry"),
            ("Batch diversity weight", "diversity_lambda", "entry"),
        ]
        for i, (label, key, kind) in enumerate(rows):
            r = i // 2
            c = (i % 2) * 2
            ttk.Label(settings, text=label).grid(row=r, column=c, sticky="w", padx=(0, 8), pady=4)
            if kind == "kernel":
                widget = ttk.Combobox(settings, textvariable=self.setting_vars[key], values=["matern52", "matern32"], state="readonly", width=24)
            elif kind == "acquisition":
                widget = ttk.Combobox(settings, textvariable=self.setting_vars[key], values=["ei", "pi"], state="readonly", width=24)
            elif kind == "candidate_mode":
                widget = ttk.Combobox(settings, textvariable=self.setting_vars[key], values=["random", "csv"], state="readonly", width=24)
            else:
                widget = ttk.Entry(settings, textvariable=self.setting_vars[key], width=26)
            widget.grid(row=r, column=c + 1, sticky="ew", padx=(0, 18), pady=4)

        check_row = len(rows) // 2 + 1
        ttk.Checkbutton(settings, text="Use log1p(q) target transform", variable=self.setting_vars["use_log1p_target"]).grid(row=check_row, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Checkbutton(settings, text="Use calibrated transfer prior for early batches", variable=self.setting_vars["use_reference_prior"]).grid(row=check_row, column=2, columnspan=2, sticky="w", pady=4)

        transfer_advanced = CollapsibleSection(settings, "+ Experiment plan and calibrated transfer prior settings", expanded=False)
        transfer_advanced.grid(row=check_row + 1, column=0, columnspan=4, sticky="ew", pady=(2, 6))
        ttk.Label(transfer_advanced.body, text="Estimated iteration lower bound").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(transfer_advanced.body, textvariable=self.setting_vars["planned_iteration_min"], width=12).grid(row=0, column=1, sticky="w", padx=(8, 0), pady=3)
        ttk.Label(transfer_advanced.body, text="Estimated iteration upper bound").grid(row=0, column=2, sticky="w", padx=(14, 0), pady=3)
        ttk.Entry(transfer_advanced.body, textvariable=self.setting_vars["planned_iteration_max"], width=12).grid(row=0, column=3, sticky="w", padx=(8, 0), pady=3)
        ttk.Label(transfer_advanced.body, text="Auto transfer fraction").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(transfer_advanced.body, textvariable=self.setting_vars["transfer_prior_fraction"], width=12).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=3)
        ttk.Label(transfer_advanced.body, text="M calculation mode").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Combobox(transfer_advanced.body, textvariable=self.setting_vars["transfer_rounds_mode"], values=["auto", "manual"], state="readonly", width=10).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=3)
        ttk.Label(transfer_advanced.body, text="Manual M override").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Entry(transfer_advanced.body, textvariable=self.setting_vars["transfer_prior_rounds"], width=12).grid(row=3, column=1, sticky="w", padx=(8, 0), pady=3)
        ttk.Label(
            transfer_advanced.body,
            text=(
                "Auto mode uses M = floor(mean(iteration lower, iteration upper) × fraction). "
                "With the default 10-20 iteration range and fraction 0.30, M = floor(15 × 0.30) = 4. "
                "The total experiment range is experiments per batch × the iteration range."
            ),
            style="Hint.TLabel",
            wraplength=820,
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(2, 4))

        ttk.Button(settings, text="Save Settings", command=self.apply_settings).grid(row=check_row + 2, column=3, sticky="e", pady=(8, 0))

        summary = ttk.LabelFrame(self.project_tab, text="Dashboard", padding=10)
        summary.pack(fill="both", expand=True, pady=(0, 8))
        self.dashboard_text = tk.Text(summary, height=18, wrap="word")
        self.dashboard_text.pack(fill="both", expand=True)
        self.dashboard_text.configure(state="disabled")

    def _build_suggestion_tab(self):
        intro = ttk.LabelFrame(self.suggest_tab, text="Guided workflow", padding=10)
        intro.pack(fill="x", pady=(0, 8))
        self.suggestion_hint_var = tk.StringVar(value="Create or load a project to begin.")
        ttk.Label(intro, textvariable=self.suggestion_hint_var, wraplength=1040, justify="left").pack(fill="x")

        controls = ttk.Frame(self.suggest_tab)
        controls.pack(fill="x", pady=(0, 8))
        self.initialize_button = ttk.Button(
            controls,
            text="1. Initialize Suggestions",
            style="Primary.TButton",
            command=self.initialize_suggestions,
        )
        self.initialize_button.pack(side="left", padx=4)
        self.enter_results_button = ttk.Button(
            controls,
            text="2. Enter Results for Selected",
            style="Action.TButton",
            command=lambda: self.edit_selected_row(result_mode=True),
        )
        self.enter_results_button.pack(side="left", padx=4)
        self.suggest_button = ttk.Button(
            controls,
            text="3. Suggest New Experiments",
            style="Action.TButton",
            command=self.suggest_next_batch,
        )
        self.suggest_button.pack(side="left", padx=4)

        advanced = CollapsibleSection(self.suggest_tab, "+ Advanced suggestion options", expanded=False)
        advanced.pack(fill="x", pady=(0, 8))
        ttk.Button(advanced.body, text="Import Candidate CSV", command=self.import_candidate_csv).pack(side="left", padx=4)
        ttk.Button(advanced.body, text="Export Current Suggestions", command=self.export_suggestions_csv).pack(side="left", padx=4)
        ttk.Button(advanced.body, text="Open Project Folder", command=self.open_project_folder).pack(side="left", padx=4)
        ttk.Button(advanced.body, text="Reload Project Files", command=self.reload_project).pack(side="left", padx=4)

        self.suggestion_tree = self._create_tree(self.suggest_tab, height=18)

    def _build_data_tab(self):
        controls = ttk.Frame(self.data_tab)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Button(controls, text="Add Completed Row", command=self.add_completed_row).pack(side="left", padx=4)
        ttk.Button(controls, text="Edit Selected Row", command=self.edit_selected_row).pack(side="left", padx=4)
        ttk.Button(controls, text="Recalculate q", command=self.recalculate_q_for_all).pack(side="left", padx=4)
        ttk.Button(controls, text="Delete Selected Row", command=self.delete_selected_row).pack(side="left", padx=4)

        advanced = CollapsibleSection(self.data_tab, "+ Import, export, and external editing", expanded=False)
        advanced.pack(fill="x", pady=(0, 8))
        ttk.Button(advanced.body, text="Import Experiments CSV", command=self.import_experiments_csv).pack(side="left", padx=4)
        ttk.Button(advanced.body, text="Export Experiments CSV", command=self.export_experiments_csv).pack(side="left", padx=4)
        ttk.Button(advanced.body, text="Open CSV in Default App", command=self.open_experiments_csv).pack(side="left", padx=4)
        ttk.Button(advanced.body, text="Open Project Folder", command=self.open_project_folder).pack(side="left", padx=4)

        self.data_tree = self._create_tree(self.data_tab, height=22)

    def _build_visualization_tab(self):
        controls = ttk.Frame(self.viz_tab)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Button(controls, text="Refresh Plots", command=self.refresh_plots).pack(side="left", padx=4)
        ttk.Button(controls, text="Save Plots as PNG", command=self.save_plots_png).pack(side="left", padx=4)
        ttk.Label(
            controls,
            text="Plots update after Save, Suggest, Reload, or Refresh.",
            style="Hint.TLabel",
        ).pack(side="left", padx=12)

        summary_frame = ttk.LabelFrame(self.viz_tab, text="Model and Acquisition Summary", padding=6)
        summary_frame.pack(fill="x", pady=(0, 8))
        self.viz_summary_var = tk.StringVar(value="No model diagnostics yet.")
        ttk.Label(summary_frame, textvariable=self.viz_summary_var, style="Hint.TLabel", wraplength=1040, justify="left").pack(fill="x")

        self.plot_frame = ttk.Frame(self.viz_tab)
        self.plot_frame.pack(fill="both", expand=True)

    def _build_help_tab(self):
        help_text = tk.Text(self.help_tab, wrap="word")
        help_text.pack(fill="both", expand=True)
        help_text.insert("1.0", self._help_text())
        help_text.configure(state="disabled")

    def _create_tree(self, parent, height: int) -> ttk.Treeview:
        display_columns = [
            "record_id",
            "round",
            "batch_position",
            "status",
            *FEATURES,
            "intensity",
            "fwhm",
            "q",
            "predicted_q_mean",
            "predicted_q_sd",
            "acquisition_value",
            "notes",
        ]
        container = ttk.Frame(parent)
        tree = ttk.Treeview(container, columns=display_columns, show="headings", height=height, selectmode="browse")
        yscroll = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        xscroll = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.pack(fill="both", expand=True)

        widths = {
            "record_id": 90,
            "round": 55,
            "batch_position": 105,
            "status": 90,
            "notes": 190,
        }
        for col in display_columns:
            tree.heading(col, text=col)
            tree.column(col, width=widths.get(col, 110), minwidth=55, anchor="center")
        tree.bind("<Double-1>", lambda _event: self.edit_selected_row())
        return tree

    def _help_text(self) -> str:
        return f"""
{APP_NAME} {APP_VERSION}

Purpose
This app helps students run a local closed-loop Bayesian Optimization workflow for MOF synthesis conditions. The objective is to maximize q, where q is calculated from intensity and FWHM. A higher q means a stronger and sharper primary peak.

Workflow
1. Create a new project.
2. Choose one of four BO settings:
   - Matérn 5/2 + EI
   - Matérn 5/2 + PI
   - Matérn 3/2 + EI
   - Matérn 3/2 + PI
3. On a new project, click "Initialize Suggestions" to get the first suggested batch. The default batch size is three experiments, but the project can save a different batch size.
4. Run the reactions in the lab.
5. Select each suggested row and click "Enter Results for Selected". Enter intensity and FWHM; q is calculated automatically.
6. Use "Manually override q" only when you need to enter q directly.
7. After all results are entered, click the highlighted "Suggest New Experiments" button.

Variables
The five variables are fixed to the benchmark search space:
- metal_amount: 5 to 75, step 1
- modulator: 5 to 15, step 1
- add_solvent: 0 to 30, step 1
- reaction_time: 1 to 12, step 1
- reaction_temperature: 10 to 30, step 1

Objective
By default, the app calculates q automatically when intensity and FWHM are entered:
q = round(intensity / FWHM)
If FWHM is 30, FWHM is non-positive, or intensity is zero, q is set to 0.

Matérn 3/2 vs Matérn 5/2
Matérn 5/2 assumes a smoother response surface. Matérn 3/2 allows a rougher response surface. Both are common choices for experimental Bayesian Optimization.

EI vs PI
EI means Expected Improvement. It values both the chance of improvement and the possible size of the improvement. PI means Probability of Improvement. It focuses on the chance that a candidate beats the current best result.

Calibrated transfer prior, Mode C
This app can use the built-in benchmark as an early-round prior, but it does not directly mix old raw q values with new project q values. This matters because a new PXRD instrument or a new compound can have a different intensity scale.

The transfer model works in three steps.

Step 1: Fit a reference GP from the built-in benchmark:
mu_ref(x) = reference GP predicted mean at condition x

Step 2: After the student enters results for the current project, calibrate the reference prediction to the current scale:
y_student_i ≈ a + b * mu_ref(x_i)

Here y_student_i is the model-scale target, usually log(1 + q_i). The constants a and b are estimated from the current project results. If the new PXRD intensity scale is lower or higher, a and b can shift and rescale the reference prediction.

Step 3: Fit a residual GP on what the calibrated reference model still misses:
r_i = y_student_i - (a + b * mu_ref(x_i))

The final prediction used by EI or PI is:
mu_final(x) = a + b * mu_ref(x) + mu_residual(x)

The uncertainty combines residual uncertainty with a damped reference uncertainty:
sigma_final(x) ≈ sqrt(sigma_residual(x)^2 + damped_reference_uncertainty(x)^2)

EI or PI uses the best current-project result as the improvement target once any student result exists. This keeps the reference benchmark from setting an unrealistic improvement threshold when q scales differ.

Planning wizard and early-batch limit M
When a new project is created, the app asks for the estimated project size:
- experiments per batch, default = 3
- estimated BO iteration range, default = 10 to 20 iterations

One BO iteration means one suggested batch. With the defaults, the project plan is:
3 experiments per batch × 10-20 iterations = 30-60 total experiments.

If calibrated transfer is turned on, the app suggests the early transfer window M from the average of the iteration range:

M = floor(mean(iteration_lower, iteration_upper) × transfer_fraction)

The default transfer_fraction is 0.30. With the default 10-20 iteration range:
mean(10, 20) = 15
M = floor(15 × 0.30) = 4

So transfer is used only for the first 4 suggestion batches by default, then the app switches to current-project student data only. This is useful for common 10-20 iteration teaching projects because transfer helps the early stage but does not dominate the later rounds. To override M manually, open '+ Experiment plan and calibrated transfer prior settings', set the mode to manual, and enter your own M.

These planning values are saved inside project_config.json. If a student closes the app after one round and later loads the same project folder, the batch size, iteration range, total experiment estimate, transfer fraction, and M setting are remembered.

Random sampling vs CSV sampling
Random mode samples candidates from the full integer search space. CSV mode suggests only from an imported candidate pool. A candidate CSV must include these columns:
{', '.join(FEATURES)}

Project files
Each project folder stores:
- project_config.json
- experiments.csv
- candidate_pool.csv, if imported
- exports folder, for exported files
- plots folder, for saved visualization PNG files

Visualization
The Visualization page includes progress plots, model-fit diagnostics, uncertainty plots, acquisition-score plots, and an experiment map. Open suggestions are marked so students can see how the proposed batch relates to predicted q, model uncertainty, and EI or PI. If you try to request a new batch while suggested rows still have no result, the app asks for confirmation. The Suggestions page keeps showing the latest batch even after results are entered, so the workflow state is easier to see.

Notes
This app runs locally on the student's computer. It does not call any API and does not need an API key. Suggested conditions should still be checked by the instructor for lab safety and practical constraints.
""".strip()

    # -----------------------------
    # Project file management
    # -----------------------------
    def _ensure_project_structure(self):
        """Create the project folder and standard subfolders."""
        if self.project_path is None:
            return
        self.project_path.mkdir(parents=True, exist_ok=True)
        for subfolder in ["exports", "plots", "archive"]:
            (self.project_path / subfolder).mkdir(parents=True, exist_ok=True)

    def new_project(self):
        dialog = NewProjectDialog(self)
        self.wait_window(dialog)
        if not dialog.result:
            return
        folder = Path(dialog.result["folder"])
        self.project_path = folder
        self._ensure_project_structure()
        self.project_config = DEFAULT_CONFIG.copy()
        self.project_config.update({k: v for k, v in dialog.result.items() if k != "folder"})
        self.project_config["created_at"] = now_text()
        self.project_config["updated_at"] = now_text()
        self.experiments = pd.DataFrame(columns=EXPERIMENT_COLUMNS)
        self.candidate_pool = None
        self.save_project()
        self.notebook.select(self.suggest_tab)
        self._set_status(f"Created project folder: {folder}")
        self._refresh_all()
        messagebox.showinfo(
            "Project created",
            "The project has been created in its own folder.\n\n"
            "Next step: open the Suggestions page and click 'Initialize Suggestions' to get the first batch.",
        )

    def load_project_dialog(self):
        folder = filedialog.askdirectory(title="Choose a project folder containing project_config.json")
        if not folder:
            return
        self.load_project(Path(folder))

    def load_project(self, folder: Path):
        try:
            folder = Path(folder)
            self.project_path = folder
            self._ensure_project_structure()
            config_path = folder / "project_config.json"
            if config_path.exists():
                loaded = json.loads(config_path.read_text(encoding="utf-8"))
                self.project_config = DEFAULT_CONFIG.copy()
                self.project_config.update(loaded)
            else:
                self.project_config = DEFAULT_CONFIG.copy()
                self.project_config["project_name"] = folder.name
            exp_path = folder / "experiments.csv"
            if exp_path.exists():
                self.experiments = pd.read_csv(exp_path)
                self.experiments = self._normalize_experiments(self.experiments)
            else:
                self.experiments = pd.DataFrame(columns=EXPERIMENT_COLUMNS)
            cand_path = folder / "candidate_pool.csv"
            if cand_path.exists():
                self.candidate_pool = self._read_candidate_csv(cand_path)
            else:
                self.candidate_pool = None
            self.notebook.select(self.suggest_tab)
            self._set_status(f"Loaded project from {folder}")
            self._refresh_all()
        except Exception as exc:
            messagebox.showerror("Load failed", f"Could not load project.\n\n{exc}")

    def reload_project(self):
        if self.project_path is None:
            messagebox.showinfo("No project", "Load or create a project first.")
            return
        self.load_project(self.project_path)

    def save_project(self):
        if self.project_path is None:
            messagebox.showinfo("No project", "Create or load a project first.")
            return
        try:
            self._ensure_project_structure()
            self.project_config["updated_at"] = now_text()
            (self.project_path / "project_config.json").write_text(json.dumps(self.project_config, indent=2), encoding="utf-8")
            self.experiments = self._normalize_experiments(self.experiments)
            self.experiments.to_csv(self.project_path / "experiments.csv", index=False)
            if self.candidate_pool is not None and len(self.candidate_pool) > 0:
                self.candidate_pool.to_csv(self.project_path / "candidate_pool.csv", index=False)
            self._set_status("Project saved.")
            self._refresh_all()
        except Exception as exc:
            messagebox.showerror("Save failed", f"Could not save project.\n\n{exc}")

    def apply_settings(self):
        try:
            self.project_config["project_name"] = self.setting_vars["project_name"].get().strip() or "Untitled Project"
            self.project_config["kernel"] = self.setting_vars["kernel"].get()
            self.project_config["acquisition"] = self.setting_vars["acquisition"].get()
            batch_size = max(1, safe_int(self.setting_vars["batch_size"].get(), 3))
            self.project_config["batch_size"] = batch_size
            self.project_config["initial_samples"] = batch_size
            self.project_config["candidate_mode"] = self.setting_vars["candidate_mode"].get()
            self.project_config["random_candidate_count"] = max(100, safe_int(self.setting_vars["random_candidate_count"].get(), 15000))
            self.project_config["diversity_lambda"] = float(self.setting_vars["diversity_lambda"].get())
            self.project_config["use_log1p_target"] = bool(self.setting_vars["use_log1p_target"].get())
            self.project_config["use_reference_prior"] = bool(self.setting_vars["use_reference_prior"].get())
            low = max(1, safe_int(self.setting_vars["planned_iteration_min"].get(), 10))
            high = max(1, safe_int(self.setting_vars["planned_iteration_max"].get(), 20))
            if high < low:
                low, high = high, low
            avg_iterations = (float(low) + float(high)) / 2.0
            self.project_config["planned_iteration_min"] = int(low)
            self.project_config["planned_iteration_max"] = int(high)
            self.project_config["planned_total_batches"] = int(round(avg_iterations))
            self.project_config["estimated_total_experiments_min"] = int(batch_size * low)
            self.project_config["estimated_total_experiments_max"] = int(batch_size * high)
            fraction = parse_float_or_nan(self.setting_vars["transfer_prior_fraction"].get())
            if not np.isfinite(fraction) or fraction <= 0:
                fraction = 0.30
            if fraction > 1.0 and fraction <= 100.0:
                fraction = fraction / 100.0
            self.project_config["transfer_prior_fraction"] = min(max(float(fraction), 0.01), 1.0)
            mode = self.setting_vars["transfer_rounds_mode"].get() or "auto"
            self.project_config["transfer_rounds_mode"] = mode
            auto_m = max(1, min(int(high), int(math.floor(avg_iterations * self.project_config["transfer_prior_fraction"]))))
            if mode == "manual":
                self.project_config["transfer_prior_rounds"] = max(0, min(int(high), safe_int(self.setting_vars["transfer_prior_rounds"].get(), auto_m)))
            else:
                self.project_config["transfer_prior_rounds"] = auto_m
            self.save_project()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))

    def _normalize_experiments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a clean experiments table and auto-calculate q where possible."""
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=EXPERIMENT_COLUMNS)
        out = df.copy()
        for col in EXPERIMENT_COLUMNS:
            if col not in out.columns:
                out[col] = ""
        for feature in FEATURES:
            out[feature] = pd.to_numeric(out[feature], errors="coerce")
        out = out.dropna(subset=FEATURES)
        for feature in FEATURES:
            out[feature] = out[feature].astype(int)
        for col in ["round", "batch_position"]:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
        for col in ["intensity", "fwhm", "q", "predicted_q_mean", "predicted_q_sd", "acquisition_value"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        out["status"] = out["status"].fillna("suggested").replace("", "suggested")
        out["notes"] = out["notes"].fillna("")
        out["record_id"] = out["record_id"].fillna("")

        # Auto-calculate q only when q is blank. A manually entered q is preserved.
        for i in range(len(out)):
            q_current = parse_float_or_nan(out.iloc[i]["q"])
            if not np.isfinite(q_current):
                q_auto = calc_q(out.iloc[i]["intensity"], out.iloc[i]["fwhm"], None)
                if np.isfinite(q_auto):
                    out.iat[i, out.columns.get_loc("q")] = q_auto
                    q_current = q_auto
            status_text = str(out.iloc[i]["status"]).strip().lower()
            if np.isfinite(q_current) and status_text in {"", "suggested", "pending"}:
                out.iat[i, out.columns.get_loc("status")] = "completed"

        for i in range(len(out)):
            if str(out.iloc[i]["record_id"]).strip() == "":
                out.iat[i, out.columns.get_loc("record_id")] = self._next_record_id(existing=out["record_id"].tolist())
        out = out[EXPERIMENT_COLUMNS]
        return out.reset_index(drop=True)

    def _read_candidate_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Candidate CSV is missing columns: {missing}")
        df = df[FEATURES].copy()
        for feature in FEATURES:
            df[feature] = pd.to_numeric(df[feature], errors="coerce")
        df = df.dropna(subset=FEATURES)
        df[FEATURES] = df[FEATURES].astype(int)
        valid_rows = []
        for _idx, row in df.iterrows():
            values = {f: row[f] for f in FEATURES}
            ok, _msg = valid_condition_values(values)
            if ok:
                valid_rows.append(values)
        return pd.DataFrame(valid_rows).drop_duplicates(subset=FEATURES).reset_index(drop=True)

    def open_project_folder(self):
        if self.project_path is None:
            messagebox.showinfo("No project", "Create or load a project first.")
            return
        path = str(self.project_path)
        try:
            if platform.system() == "Windows":
                os.startfile(path)  # type: ignore[attr-defined]
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:
            webbrowser.open(Path(path).as_uri())

    # -----------------------------
    # Data operations
    # -----------------------------
    def add_completed_row(self):
        if self.project_path is None:
            messagebox.showinfo("No project", "Create or load a project first.")
            return
        dialog = RowEditor(self, "Add Completed Experiment", completed_default=True)
        self.wait_window(dialog)
        if not dialog.result:
            return
        row = dialog.result
        row["record_id"] = self._next_record_id()
        row["round"] = self._next_round_number(add_new=True)
        row["batch_position"] = 1
        row["created_at"] = now_text()
        row["updated_at"] = now_text()
        self.experiments = pd.concat([self.experiments, pd.DataFrame([row])], ignore_index=True)
        self.save_project()

    def edit_selected_row(self, result_mode: Optional[bool] = None):
        tree = self._active_tree()
        selection = tree.selection()
        if not selection:
            messagebox.showinfo("No row selected", "Select a row first.")
            return
        item = selection[0]
        record_id = tree.item(item, "values")[0]
        idx_matches = self.experiments.index[self.experiments["record_id"].astype(str) == str(record_id)].tolist()
        if not idx_matches:
            messagebox.showerror("Row not found", "The selected row was not found in experiments.csv.")
            return
        idx = idx_matches[0]
        initial = self.experiments.loc[idx].to_dict()
        if result_mode is None:
            tab = self.notebook.select()
            tab_text = self.notebook.tab(tab, "text") if tab else ""
            result_mode = tab_text == "Suggestions"
        dialog_title = "Enter Reaction Result" if result_mode else "Edit Experiment"
        dialog = RowEditor(
            self,
            dialog_title,
            initial=initial,
            completed_default=True if result_mode else (initial.get("status") == "completed"),
            result_mode=bool(result_mode),
        )
        self.wait_window(dialog)
        if not dialog.result:
            return
        updated = initial.copy()
        updated.update(dialog.result)
        updated["updated_at"] = now_text()
        for col in EXPERIMENT_COLUMNS:
            if col not in updated:
                updated[col] = ""
        self.experiments.loc[idx, EXPERIMENT_COLUMNS] = [updated[col] for col in EXPERIMENT_COLUMNS]
        self.save_project()
        self._refresh_all()

    def delete_selected_row(self):
        tree = self._active_tree()
        selection = tree.selection()
        if not selection:
            messagebox.showinfo("No row selected", "Select a row first.")
            return
        item = selection[0]
        record_id = tree.item(item, "values")[0]
        if not messagebox.askyesno("Delete row", f"Delete row {record_id}?"):
            return
        self.experiments = self.experiments[self.experiments["record_id"].astype(str) != str(record_id)].reset_index(drop=True)
        self.save_project()

    def recalculate_q_for_all(self):
        if self.project_path is None:
            messagebox.showinfo("No project", "Create or load a project first.")
            return
        if self.experiments is None or len(self.experiments) == 0:
            messagebox.showinfo("No rows", "There are no experiment rows to update.")
            return
        overwrite = messagebox.askyesno(
            "Recalculate q",
            "Choose Yes to overwrite all q values from Intensity and FWHM.\n\nChoose No to fill only blank q values.",
        )
        updated = 0
        df = self.experiments.copy()
        for i in range(len(df)):
            q_current = parse_float_or_nan(df.loc[i, "q"])
            q_auto = calc_q(df.loc[i, "intensity"], df.loc[i, "fwhm"], None)
            if not np.isfinite(q_auto):
                continue
            if overwrite or not np.isfinite(q_current):
                df.loc[i, "q"] = q_auto
                if str(df.loc[i, "status"]).strip().lower() in {"", "suggested", "pending"}:
                    df.loc[i, "status"] = "completed"
                df.loc[i, "updated_at"] = now_text()
                updated += 1
        self.experiments = self._normalize_experiments(df)
        self.save_project()
        self._set_status(f"Recalculated q for {updated} rows.")

    def import_experiments_csv(self):
        if self.project_path is None:
            messagebox.showinfo("No project", "Create or load a project first.")
            return
        path = filedialog.askopenfilename(
            title="Import experiments CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
            missing = [c for c in FEATURES if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            for col in EXPERIMENT_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            df = self._normalize_experiments(df)
            # Recalculate q if intensity and FWHM were supplied and q is blank.
            for i in range(len(df)):
                q_current = parse_float_or_nan(df.loc[i, "q"])
                if not np.isfinite(q_current):
                    df.loc[i, "q"] = calc_q(df.loc[i, "intensity"], df.loc[i, "fwhm"])
            if messagebox.askyesno("Import mode", "Append imported rows to the current project?\n\nChoose No to replace the current table."):
                self.experiments = pd.concat([self.experiments, df], ignore_index=True)
            else:
                self.experiments = df
            self.experiments = self._normalize_experiments(self.experiments)
            self.save_project()
        except Exception as exc:
            messagebox.showerror("Import failed", str(exc))

    def export_experiments_csv(self):
        if self.project_path is None:
            messagebox.showinfo("No project", "Create or load a project first.")
            return
        self._ensure_project_structure()
        path = filedialog.asksaveasfilename(
            title="Export experiments CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialdir=str(self.project_path / "exports"),
            initialfile="experiments_export.csv",
        )
        if not path:
            return
        self.experiments.to_csv(path, index=False)
        self._set_status(f"Exported experiments to {path}")

    def export_suggestions_csv(self):
        if self.project_path is None:
            messagebox.showinfo("No project", "Create or load a project first.")
            return
        suggestions = self._suggestion_panel_rows().copy()
        if len(suggestions) == 0:
            messagebox.showinfo("No suggestions", "There are no suggestion-panel rows to export.")
            return
        self._ensure_project_structure()
        path = filedialog.asksaveasfilename(
            title="Export suggestions CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialdir=str(self.project_path / "exports"),
            initialfile="suggestions_export.csv",
        )
        if not path:
            return
        suggestions.to_csv(path, index=False)
        self._set_status(f"Exported suggestions to {path}")

    def import_candidate_csv(self):
        if self.project_path is None:
            messagebox.showinfo("No project", "Create or load a project first.")
            return
        path = filedialog.askopenfilename(
            title="Import candidate CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.candidate_pool = self._read_candidate_csv(Path(path))
            self.project_config["candidate_mode"] = "csv"
            self.save_project()
            messagebox.showinfo("Candidate CSV imported", f"Imported {len(self.candidate_pool)} candidate rows.")
        except Exception as exc:
            messagebox.showerror("Import failed", str(exc))

    def open_experiments_csv(self):
        if self.project_path is None:
            messagebox.showinfo("No project", "Create or load a project first.")
            return
        self.save_project()
        path = self.project_path / "experiments.csv"
        try:
            if platform.system() == "Windows":
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
            messagebox.showinfo("External editing", "After editing the CSV externally, save it and click Reload Files in this app.")
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))

    # -----------------------------
    # BO suggestions
    # -----------------------------
    def _pending_suggestion_rows(self) -> pd.DataFrame:
        """Return suggested rows that still do not have a usable q value."""
        if self.experiments is None or len(self.experiments) == 0:
            return pd.DataFrame(columns=EXPERIMENT_COLUMNS)
        df = self._normalize_experiments(self.experiments.copy())
        mask = (
            df["status"].astype(str).str.strip().str.lower().isin({"suggested", "pending", ""})
            & pd.to_numeric(df["q"], errors="coerce").isna()
        )
        return df[mask].copy()

    def _confirm_if_pending_suggestions(self) -> bool:
        """Ask for confirmation before creating another batch with open suggestions."""
        pending = self._pending_suggestion_rows()
        if len(pending) == 0:
            return True
        pending_rounds = sorted(set(str(int(x)) for x in pd.to_numeric(pending["round"], errors="coerce").dropna().tolist()))
        round_text = ", ".join(pending_rounds[:8]) if pending_rounds else "unknown"
        msg = (
            f"There are {len(pending)} suggested experiment(s) without saved results.\n\n"
            f"Open round(s): {round_text}\n\n"
            "If you continue, the app will create another suggested batch before these results are entered. "
            "This may make the active-learning history harder to interpret.\n\n"
            "Do you still want to suggest a new batch?"
        )
        proceed = messagebox.askyesno("Open suggestions without results", msg)
        if not proceed:
            self.notebook.select(self.suggest_tab)
            self._set_status("Suggestion cancelled because open suggested experiments still need results.")
        return bool(proceed)

    def initialize_suggestions(self):
        """Generate the first project batch with a first-time-user confirmation."""
        if self.project_path is None:
            messagebox.showinfo("No project", "Create or load a project first.")
            return
        if self.experiments is not None and len(self.experiments) > 0:
            if not messagebox.askyesno(
                "Project already has rows",
                "This project already contains experiment rows.\n\n"
                "Do you still want to create another suggestion batch?",
            ):
                return
        self._generate_suggestion_batch(initial_request=True)

    def suggest_next_batch(self):
        """Generate a new BO suggestion batch after the current results are entered."""
        if self.project_path is None:
            messagebox.showinfo("No project", "Create or load a project first.")
            return
        if self.experiments is None or len(self.experiments) == 0:
            if messagebox.askyesno(
                "Initialize first",
                "This project has no suggestions yet.\n\n"
                "Click Yes to initialize the first suggestion batch now.",
            ):
                self._generate_suggestion_batch(initial_request=True)
            return
        self._generate_suggestion_batch(initial_request=False)

    def _generate_suggestion_batch(self, initial_request: bool = False):
        if self.project_path is None:
            messagebox.showinfo("No project", "Create or load a project first.")
            return
        self.experiments = self._normalize_experiments(self.experiments)
        if not initial_request and not self._confirm_if_pending_suggestions():
            self._refresh_all()
            return
        try:
            self.apply_settings_without_save_dialog()
            completed = self.experiments[
                (self.experiments["status"].astype(str) == "completed") &
                (pd.to_numeric(self.experiments["q"], errors="coerce").notna())
            ].copy()
            batch_size = int(self.project_config.get("batch_size", 3))
            engine = BOEngine(self.project_config)
            if len(completed) < 3 and not engine._transfer_active(self.experiments):
                batch_size = int(self.project_config.get("initial_samples", 3))
            suggestions, message = engine.suggest(completed, self.experiments, self.candidate_pool, batch_size)
            if len(suggestions) == 0:
                messagebox.showwarning("No suggestions", "The optimizer did not find valid suggestions.")
                return
            round_number = self._next_round_number(add_new=True)
            rows = []
            for pos, (_idx, row) in enumerate(suggestions.iterrows(), start=1):
                rec = {col: "" for col in EXPERIMENT_COLUMNS}
                rec["record_id"] = self._next_record_id(existing=list(self.experiments["record_id"].astype(str)) + [r.get("record_id", "") for r in rows])
                rec["round"] = round_number
                rec["batch_position"] = pos
                rec["status"] = "suggested"
                for feature in FEATURES:
                    rec[feature] = int(row[feature])
                rec["predicted_q_mean"] = row.get("predicted_q_mean", np.nan)
                rec["predicted_q_sd"] = row.get("predicted_q_sd", np.nan)
                rec["acquisition_value"] = row.get("acquisition_value", np.nan)
                rec["notes"] = message
                rec["created_at"] = now_text()
                rec["updated_at"] = now_text()
                rows.append(rec)
            self.experiments = pd.concat([self.experiments, pd.DataFrame(rows)], ignore_index=True)
            self.save_project()
            self.notebook.select(self.suggest_tab)
            self._select_first_pending_suggestion()
            if initial_request:
                messagebox.showinfo(
                    "Initial suggestions ready",
                    "The first suggestion batch is ready.\n\n"
                    f"Run the {len(rows)} reaction(s), then select each row and click 'Enter Results for Selected'.\n"
                    "After all results are entered, the 'Suggest New Experiments' button will be highlighted.",
                )
            else:
                messagebox.showinfo(
                    "New suggestions ready",
                    "A new suggestion batch is ready.\n\n"
                    "Enter Intensity and FWHM for each suggested experiment. q will be calculated automatically.",
                )
            self._set_status(f"Suggested {len(rows)} experiments. {message}")
        except Exception as exc:
            messagebox.showerror("Suggestion failed", f"{exc}\n\n{traceback.format_exc(limit=2)}")

    def apply_settings_without_save_dialog(self):
        try:
            self.project_config["project_name"] = self.setting_vars["project_name"].get().strip() or self.project_config.get("project_name", "Untitled Project")
            self.project_config["kernel"] = self.setting_vars["kernel"].get() or self.project_config.get("kernel", "matern52")
            self.project_config["acquisition"] = self.setting_vars["acquisition"].get() or self.project_config.get("acquisition", "ei")
            batch_size = max(1, safe_int(self.setting_vars["batch_size"].get(), int(self.project_config.get("batch_size", 3))))
            self.project_config["batch_size"] = batch_size
            self.project_config["initial_samples"] = batch_size
            self.project_config["candidate_mode"] = self.setting_vars["candidate_mode"].get() or self.project_config.get("candidate_mode", "random")
            self.project_config["random_candidate_count"] = max(100, safe_int(self.setting_vars["random_candidate_count"].get(), int(self.project_config.get("random_candidate_count", 15000))))
            self.project_config["diversity_lambda"] = float(self.setting_vars["diversity_lambda"].get() or self.project_config.get("diversity_lambda", 0.03))
            self.project_config["use_log1p_target"] = bool(self.setting_vars["use_log1p_target"].get())
            self.project_config["use_reference_prior"] = bool(self.setting_vars["use_reference_prior"].get())
            low = max(1, safe_int(self.setting_vars["planned_iteration_min"].get(), int(self.project_config.get("planned_iteration_min", 10))))
            high = max(1, safe_int(self.setting_vars["planned_iteration_max"].get(), int(self.project_config.get("planned_iteration_max", 20))))
            if high < low:
                low, high = high, low
            avg_iterations = (float(low) + float(high)) / 2.0
            self.project_config["planned_iteration_min"] = int(low)
            self.project_config["planned_iteration_max"] = int(high)
            self.project_config["planned_total_batches"] = int(round(avg_iterations))
            self.project_config["estimated_total_experiments_min"] = int(batch_size * low)
            self.project_config["estimated_total_experiments_max"] = int(batch_size * high)
            fraction = parse_float_or_nan(self.setting_vars["transfer_prior_fraction"].get() or self.project_config.get("transfer_prior_fraction", 0.30))
            if not np.isfinite(fraction) or fraction <= 0:
                fraction = 0.30
            if fraction > 1.0 and fraction <= 100.0:
                fraction = fraction / 100.0
            self.project_config["transfer_prior_fraction"] = min(max(float(fraction), 0.01), 1.0)
            mode = self.setting_vars["transfer_rounds_mode"].get() or self.project_config.get("transfer_rounds_mode", "auto")
            self.project_config["transfer_rounds_mode"] = mode
            auto_m = max(1, min(int(high), int(math.floor(avg_iterations * self.project_config["transfer_prior_fraction"]))))
            if mode == "manual":
                self.project_config["transfer_prior_rounds"] = max(0, min(int(high), safe_int(self.setting_vars["transfer_prior_rounds"].get(), int(self.project_config.get("transfer_prior_rounds", auto_m)))))
            else:
                self.project_config["transfer_prior_rounds"] = auto_m
        except Exception:
            pass

    # -----------------------------
    # Visualization
    # -----------------------------
    def refresh_plots(self):
        for child in self.plot_frame.winfo_children():
            child.destroy()
        completed = self.experiments[
            (self.experiments["status"].astype(str) == "completed") &
            (pd.to_numeric(self.experiments["q"], errors="coerce").notna())
        ].copy()
        fig, summary = self._make_figure(completed)
        if hasattr(self, "viz_summary_var"):
            self.viz_summary_var.set(summary)
        self.current_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _make_figure(self, completed: pd.DataFrame) -> Tuple[Figure, str]:
        fig = Figure(figsize=(11.5, 8.2), dpi=100)
        axes = [fig.add_subplot(3, 2, i + 1) for i in range(6)]
        for ax in axes:
            ax.grid(True, alpha=0.25)

        if len(completed) == 0:
            axes[0].text(0.5, 0.5, "No completed experiments yet.", ha="center", va="center")
            for ax in axes[1:]:
                ax.axis("off")
            fig.tight_layout()
            return fig, "No completed experiments yet. Enter reaction results to enable BO progress and model diagnostics."

        completed = completed.sort_values(["round", "batch_position"]).reset_index(drop=True)
        q = completed["q"].astype(float).to_numpy()
        x = np.arange(1, len(q) + 1)
        best = np.maximum.accumulate(q)

        axes[0].plot(x, best, marker="o")
        axes[0].set_title("Best q so far")
        axes[0].set_xlabel("Completed experiment")
        axes[0].set_ylabel("Best q")

        axes[1].plot(x, q, marker="o")
        axes[1].set_title("q by completed experiment")
        axes[1].set_xlabel("Completed experiment")
        axes[1].set_ylabel("q")

        engine = BOEngine(self.project_config)
        diagnostics = engine.diagnostics(completed, self.experiments, self.candidate_pool, max_candidates=4000)
        summary = str(diagnostics.get("status", ""))

        comp_pred = diagnostics.get("completed_pred", pd.DataFrame())
        candidates = diagnostics.get("candidates", pd.DataFrame())
        model_ready = bool(diagnostics.get("model_ready", False))

        if model_ready and isinstance(comp_pred, pd.DataFrame) and len(comp_pred) > 0:
            obs = comp_pred["q"].astype(float).to_numpy()
            pred = comp_pred["model_predicted_q"].astype(float).to_numpy()
            sd = comp_pred["model_predicted_sd"].astype(float).to_numpy()
            axes[2].errorbar(obs, pred, yerr=sd, fmt="o", alpha=0.75)
            lo = float(np.nanmin([np.nanmin(obs), np.nanmin(pred)]))
            hi = float(np.nanmax([np.nanmax(obs), np.nanmax(pred)]))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                axes[2].plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
            axes[2].set_title("Model fit: observed q vs predicted q")
            axes[2].set_xlabel("Observed q")
            axes[2].set_ylabel("Predicted q")
        else:
            axes[2].text(0.5, 0.5, "Model fit appears after at least\n3 completed results with variation.", ha="center", va="center")
            axes[2].set_title("Model fit")

        if model_ready and isinstance(candidates, pd.DataFrame) and len(candidates) > 0:
            cand = candidates.copy()
            sug = cand[cand.get("is_open_suggestion", False).astype(bool)] if "is_open_suggestion" in cand.columns else pd.DataFrame()
            axes[3].scatter(cand["predicted_q_mean"], cand["predicted_q_sd"], s=12, alpha=0.35, label="candidate sample")
            if len(sug) > 0:
                axes[3].scatter(sug["predicted_q_mean"], sug["predicted_q_sd"], s=70, marker="*", label="open suggestions")
            axes[3].set_title("Uncertainty landscape")
            axes[3].set_xlabel("Predicted q mean")
            axes[3].set_ylabel("Predicted q standard deviation")
            axes[3].legend(fontsize=8)

            axes[4].scatter(cand["predicted_q_mean"], cand["acquisition_value"], s=12, alpha=0.35, label="candidate sample")
            if len(sug) > 0:
                axes[4].scatter(sug["predicted_q_mean"], sug["acquisition_value"], s=70, marker="*", label="open suggestions")
            axes[4].set_title("How suggestions are proposed")
            axes[4].set_xlabel("Predicted q mean")
            axes[4].set_ylabel(str(self.project_config.get("acquisition", "ei")).upper())
            axes[4].legend(fontsize=8)
        else:
            axes[3].text(0.5, 0.5, "No GP uncertainty yet.", ha="center", va="center")
            axes[3].set_title("Uncertainty landscape")
            axes[4].text(0.5, 0.5, "No acquisition scores yet.", ha="center", va="center")
            axes[4].set_title("How suggestions are proposed")

        axes[5].scatter(completed["metal_amount"].astype(float), completed["reaction_temperature"].astype(float), s=45, c=q, alpha=0.8)
        suggested = self.experiments[self.experiments["status"].astype(str) == "suggested"].copy()
        if len(suggested) > 0:
            axes[5].scatter(
                suggested["metal_amount"].astype(float),
                suggested["reaction_temperature"].astype(float),
                s=90,
                marker="*",
                label="open suggestions",
            )
        axes[5].set_title("Experiment map")
        axes[5].set_xlabel("metal_amount")
        axes[5].set_ylabel("reaction_temperature")
        if len(suggested) > 0:
            axes[5].legend(fontsize=8)

        if model_ready and isinstance(candidates, pd.DataFrame) and len(candidates) > 0:
            top = candidates.sort_values("acquisition_value", ascending=False).head(3)
            if len(top) > 0:
                top_lines = []
                for pos, (_idx, row) in enumerate(top.iterrows(), start=1):
                    cond = ", ".join(f"{f}={int(row[f])}" for f in FEATURES)
                    top_lines.append(f"Top acquisition {pos}: {cond}; mean={row['predicted_q_mean']:.3g}; sd={row['predicted_q_sd']:.3g}; acq={row['acquisition_value']:.3g}")
                summary = summary + " " + " | ".join(top_lines)

        fig.tight_layout()
        return fig, summary

    def save_plots_png(self):
        if self.project_path is None:
            messagebox.showinfo("No project", "Create or load a project first.")
            return
        completed = self.experiments[
            (self.experiments["status"].astype(str) == "completed") &
            (pd.to_numeric(self.experiments["q"], errors="coerce").notna())
        ].copy()
        fig, summary = self._make_figure(completed)
        self._ensure_project_structure()
        out = self.project_path / "plots" / "bo_progress_model_plots.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        if hasattr(self, "viz_summary_var"):
            self.viz_summary_var.set(summary)
        self._set_status(f"Saved plot to {out}")
        messagebox.showinfo("Plot saved", f"Saved plot to:\n{out}")

    # -----------------------------
    # Refresh helpers
    # -----------------------------
    def _refresh_all(self):
        self._refresh_settings_form()
        self._refresh_dashboard()
        self._refresh_tables()
        self.refresh_plots()
        name = self.project_config.get("project_name", "Untitled Project")
        path = str(self.project_path) if self.project_path else "none"
        self.project_label_var.set(f"Project: {name} | Folder: {path}")

    def _refresh_settings_form(self):
        if not hasattr(self, "setting_vars"):
            return
        if "planned_iteration_min" not in self.project_config or "planned_iteration_max" not in self.project_config:
            legacy_n = max(1, safe_int(self.project_config.get("planned_total_batches", 15), 15))
            self.project_config.setdefault("planned_iteration_min", legacy_n)
            self.project_config.setdefault("planned_iteration_max", legacy_n)
        batch_size = max(1, safe_int(self.project_config.get("batch_size", 3), 3))
        low = max(1, safe_int(self.project_config.get("planned_iteration_min", 10), 10))
        high = max(1, safe_int(self.project_config.get("planned_iteration_max", 20), 20))
        if high < low:
            low, high = high, low
        self.project_config["estimated_total_experiments_min"] = int(batch_size * low)
        self.project_config["estimated_total_experiments_max"] = int(batch_size * high)
        for key, var in self.setting_vars.items():
            value = self.project_config.get(key, DEFAULT_CONFIG.get(key, ""))
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            else:
                var.set(str(value))

    def _suggestion_panel_rows(self) -> pd.DataFrame:
        """Rows shown in the Suggestions tab.

        The panel shows all pending suggestions. If a batch has just been completed,
        it keeps showing the latest round so the table does not suddenly go blank.
        """
        if self.experiments is None or len(self.experiments) == 0:
            return pd.DataFrame(columns=EXPERIMENT_COLUMNS)
        df = self._normalize_experiments(self.experiments.copy())
        pending_mask = (
            df["status"].astype(str).str.strip().str.lower().isin({"suggested", "pending", ""})
            & pd.to_numeric(df["q"], errors="coerce").isna()
        )
        if pending_mask.any():
            rounds = sorted(set(pd.to_numeric(df.loc[pending_mask, "round"], errors="coerce").dropna().astype(int).tolist()))
            return df[df["round"].astype(int).isin(rounds)].copy()
        max_round = self._max_round_number()
        if max_round >= 0:
            latest = df[df["round"].astype(int) == int(max_round)].copy()
            if len(latest) > 0:
                return latest
        return pd.DataFrame(columns=EXPERIMENT_COLUMNS)

    def _refresh_suggestion_action_state(self):
        if not hasattr(self, "suggestion_hint_var"):
            return
        has_project = self.project_path is not None
        total_rows = 0 if self.experiments is None else len(self.experiments)
        pending = self._pending_suggestion_rows()
        panel_rows = self._suggestion_panel_rows()
        panel_q = pd.to_numeric(panel_rows.get("q", pd.Series(dtype=float)), errors="coerce") if len(panel_rows) else pd.Series(dtype=float)
        completed_in_panel = int(panel_q.notna().sum()) if len(panel_rows) else 0
        total_in_panel = int(len(panel_rows))

        if not has_project:
            self.suggestion_hint_var.set("Create or load a project first. Each project is stored in its own folder with its own experiments.csv and settings.")
            self.initialize_button.configure(state="disabled", style="Action.TButton")
            self.enter_results_button.configure(state="disabled")
            self.suggest_button.configure(state="disabled", style="Action.TButton")
            return

        self.enter_results_button.configure(state="normal")
        if total_rows == 0:
            self.suggestion_hint_var.set(
                "Start here: click 'Initialize Suggestions' to create the first batch of three experiments. "
                "After running them, enter Intensity and FWHM for each row."
            )
            self.initialize_button.configure(state="normal", style="Primary.TButton")
            self.suggest_button.configure(state="disabled", style="Action.TButton")
            return

        self.initialize_button.configure(state="disabled", style="Action.TButton")
        self.suggest_button.configure(state="normal")
        if len(pending) > 0:
            self.suggest_button.configure(style="Action.TButton")
            self.suggestion_hint_var.set(
                f"Current batch progress: {completed_in_panel}/{total_in_panel} result(s) entered. "
                "Select each suggested row and click 'Enter Results for Selected'. "
                "If you click 'Suggest New Experiments' before all results are entered, the app will ask for confirmation."
            )
        else:
            self.suggest_button.configure(style="Primary.TButton")
            self.suggestion_hint_var.set(
                "All visible suggestions have saved results. The next recommended step is highlighted: "
                "click 'Suggest New Experiments' to create the next batch."
            )

    def _select_first_pending_suggestion(self):
        if not hasattr(self, "suggestion_tree"):
            return
        for item in self.suggestion_tree.get_children():
            values = self.suggestion_tree.item(item, "values")
            if len(values) >= 4 and str(values[3]).strip().lower() in {"suggested", "pending", ""}:
                self.suggestion_tree.selection_set(item)
                self.suggestion_tree.focus(item)
                self.suggestion_tree.see(item)
                return

    def _refresh_dashboard(self):
        if not hasattr(self, "dashboard_text"):
            return
        completed = self.experiments[
            (self.experiments["status"].astype(str) == "completed") &
            (pd.to_numeric(self.experiments["q"], errors="coerce").notna())
        ].copy()
        suggested = self._pending_suggestion_rows()
        best_line = "No completed results yet."
        if len(completed) > 0:
            best_idx = completed["q"].astype(float).idxmax()
            best = completed.loc[best_idx]
            cond = ", ".join(f"{f}={int(best[f])}" for f in FEATURES)
            best_line = f"Best q: {float(best['q']):.4g}\nBest condition: {cond}\nRecord: {best['record_id']}"

        candidate_line = "Candidate mode: random"
        if self.candidate_pool is not None and len(self.candidate_pool) > 0:
            candidate_line = f"Candidate mode: {self.project_config.get('candidate_mode', 'random')} | Candidate CSV rows: {len(self.candidate_pool)}"

        engine_for_status = BOEngine(self.project_config)
        transfer_enabled = bool(self.project_config.get("use_reference_prior", False))
        transfer_active = engine_for_status._transfer_active(self.experiments)
        plan_low, plan_high = engine_for_status._planned_iteration_bounds()
        avg_plan = (float(plan_low) + float(plan_high)) / 2.0
        transfer_fraction = self.project_config.get("transfer_prior_fraction", 0.30)
        transfer_mode = self.project_config.get("transfer_rounds_mode", "auto")
        transfer_m = engine_for_status._transfer_prior_rounds()
        batch_size_for_plan = max(1, safe_int(self.project_config.get("batch_size", 3), 3))
        total_min = batch_size_for_plan * plan_low
        total_max = batch_size_for_plan * plan_high
        transfer_line = (
            f"Calibrated transfer prior: {'on' if transfer_enabled else 'off'} | "
            f"planned iterations = {plan_low}-{plan_high}, total experiments = {total_min}-{total_max}, "
            f"mean = {avg_plan:.1f}, fraction = {transfer_fraction}, M = {transfer_m} ({transfer_mode}) | "
            f"active for next suggestion: {'yes' if transfer_active else 'no'}"
        )

        if self.project_path is None:
            quick_actions = (
                "1. Click New Project.\n"
                "2. The app will create a separate folder for that project.\n"
                "3. Go to Suggestions and click Initialize Suggestions."
            )
        elif len(self.experiments) == 0:
            quick_actions = (
                "1. Go to the Suggestions page.\n"
                "2. Click Initialize Suggestions to create the first batch.\n"
                "3. Run those reactions and enter Intensity and FWHM."
            )
        elif len(suggested) > 0:
            quick_actions = (
                "1. Select each pending suggested row.\n"
                "2. Click Enter Results for Selected and enter Intensity and FWHM.\n"
                "3. q is calculated automatically.\n"
                "4. When all suggested rows have results, Suggest New Experiments will be highlighted."
            )
        else:
            quick_actions = (
                "1. The latest batch has results.\n"
                "2. Go to Suggestions and click the highlighted Suggest New Experiments button.\n"
                "3. Repeat the closed-loop workflow."
            )

        text = f"""
Project summary
---------------
Project name: {self.project_config.get('project_name', 'Untitled Project')}
Project folder: {self.project_path if self.project_path else 'none'}
Kernel: {self.project_config.get('kernel', 'matern52')}
Acquisition: {self.project_config.get('acquisition', 'ei')}
Batch size: {self.project_config.get('batch_size', 3)}
{transfer_line}
{candidate_line}

Progress
--------
Completed experiments: {len(completed)}
Pending suggested experiments: {len(suggested)}
Total project rows: {len(self.experiments)}
Current max round: {self._max_round_number()}

{best_line}

Next recommended action
-----------------------
{quick_actions}
""".strip()
        self.dashboard_text.configure(state="normal")
        self.dashboard_text.delete("1.0", "end")
        self.dashboard_text.insert("1.0", text)
        self.dashboard_text.configure(state="disabled")

    def _refresh_tables(self):
        self.experiments = self._normalize_experiments(self.experiments)
        self._populate_tree(self.data_tree, self.experiments)
        suggestions = self._suggestion_panel_rows()
        self._populate_tree(self.suggestion_tree, suggestions)
        self._refresh_suggestion_action_state()
        self._select_first_pending_suggestion()

    def _populate_tree(self, tree: ttk.Treeview, df: pd.DataFrame):
        for item in tree.get_children():
            tree.delete(item)
        display_columns = list(tree["columns"])
        if df is None or len(df) == 0:
            return
        df = df.copy().sort_values(["round", "batch_position", "record_id"])
        for _idx, row in df.iterrows():
            values = []
            for col in display_columns:
                val = row.get(col, "")
                if isinstance(val, float):
                    if np.isnan(val):
                        val = ""
                    else:
                        val = f"{val:.6g}"
                values.append(val)
            tree.insert("", "end", values=values)

    def _active_tree(self) -> ttk.Treeview:
        tab = self.notebook.select()
        tab_text = self.notebook.tab(tab, "text")
        if tab_text == "Suggestions":
            return self.suggestion_tree
        return self.data_tree

    def _set_status(self, text: str):
        self.status_var.set(text)

    def _next_record_id(self, existing: Optional[List[str]] = None) -> str:
        existing_set = set(str(x) for x in (existing if existing is not None else self.experiments.get("record_id", [])))
        n = 1
        while True:
            rid = f"R{n:05d}"
            if rid not in existing_set:
                return rid
            n += 1

    def _max_round_number(self) -> int:
        if self.experiments is None or len(self.experiments) == 0 or "round" not in self.experiments.columns:
            return 0
        return int(pd.to_numeric(self.experiments["round"], errors="coerce").fillna(0).max())

    def _next_round_number(self, add_new: bool = True) -> int:
        if len(self.experiments) == 0:
            return 0
        return self._max_round_number() + (1 if add_new else 0)


def main():
    app = BOStudentApp()
    app.mainloop()


if __name__ == "__main__":
    main()
