# -*- coding: utf-8 -*-
"""
bo_engine.py - Core Bayesian Optimization Engine for MOF Active Learning.

Decoupled core module containing Gaussian Process, calibrated transfer learning,
candidate evaluation, acquisition scoring, and Matplotlib figure generation.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

APP_NAME = "MOF BO Web App"
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

DEFAULT_CONFIG: Dict[str, object] = {
    "project_name": "MOF Synthesis Optimization",
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


def normal_pdf(z: np.ndarray) -> np.ndarray:
    """Standard normal probability density function implemented directly."""
    z = np.asarray(z, dtype=float)
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def normal_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal cumulative distribution function using math.erf."""
    z = np.asarray(z, dtype=float)
    erf_vec = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))


class LightweightGaussianProcess:
    """Small NumPy-only Gaussian Process for active learning."""

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
    [6,6,19,12,30],[15,8,9,12,19],[75,12,6,11,18],
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


def calc_q(intensity: object, fwhm: object, q_value: object = None) -> float:
    """Calculate the reaction quality score q = intensity / fwhm."""
    if q_value is not None:
        try:
            fv = float(q_value)
            if np.isfinite(fv):
                return float(np.clip(fv, 0.0, 1e7))
        except Exception:
            pass

    try:
        int_val = float(intensity)
        fwhm_val = float(fwhm)
        if not np.isfinite(int_val) or not np.isfinite(fwhm_val):
            return 0.0
        if int_val <= 0.0 or fwhm_val <= 0.0:
            return 0.0
        q = int_val / fwhm_val
        return float(np.clip(q, 0.0, 1e7))
    except Exception:
        return 0.0


def parse_float_or_nan(value: object) -> float:
    try:
        val = float(value)
        return val if np.isfinite(val) else np.nan
    except Exception:
        return np.nan


def safe_int(value: object, default: int = 0) -> int:
    try:
        val = int(value)
        return val
    except Exception:
        return default


def normalize_fraction(value: object, default: float = 0.30) -> float:
    val = parse_float_or_nan(value)
    if not np.isfinite(val) or val <= 0:
        return default
    if val > 1.0 and val <= 100.0:
        val = val / 100.0
    return min(max(float(val), 0.01), 1.0)


def scale_features(X: np.ndarray) -> np.ndarray:
    """Scale discrete synthesis features to [0, 1] range."""
    X = np.asarray(X, dtype=float)
    X_scaled = np.zeros_like(X, dtype=float)
    for j, feature in enumerate(FEATURES):
        lo, hi, _step = BOUNDS[feature]
        span = max(hi - lo, 1)
        X_scaled[:, j] = (X[:, j] - lo) / span
    return X_scaled


def unscale_features(X_scaled: np.ndarray) -> np.ndarray:
    """Unscale [0, 1] values back to discrete parameter space."""
    X_scaled = np.asarray(X_scaled, dtype=float)
    X = np.zeros_like(X_scaled, dtype=float)
    for j, feature in enumerate(FEATURES):
        lo, hi, step = BOUNDS[feature]
        span = max(hi - lo, 1)
        val = lo + X_scaled[:, j] * span
        X[:, j] = np.clip(np.round(val / step) * step, lo, hi)
    return X


def condition_tuple(row: object) -> Tuple[int, int, int, int, int]:
    return tuple(int(round(float(row[f]))) for f in FEATURES)


def valid_condition_values(values: Dict[str, object]) -> Tuple[bool, str]:
    for feature in FEATURES:
        lo, hi, _step = BOUNDS[feature]
        try:
            val = float(values[feature])
            if not np.isfinite(val) or val < lo or val > hi:
                return False, f"{feature} must be between {lo} and {hi}."
        except Exception:
            return False, f"Invalid numeric value for {feature}."
    return True, ""


def make_reference_df() -> pd.DataFrame:
    df = pd.DataFrame(REFERENCE_X, columns=FEATURES)
    df["intensity"] = REFERENCE_INTENSITY
    df["fwhm"] = REFERENCE_FWHM
    df["q"] = [calc_q(i, f) for i, f in zip(REFERENCE_INTENSITY, REFERENCE_FWHM)]
    return df


REFERENCE_DF = make_reference_df()


class BOEngine:
    """Bayesian Optimization Engine for MOF active learning synthesis."""

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
        if all_rows is None or len(all_rows) == 0 or "round" not in all_rows.columns:
            return 1
        rounds = pd.to_numeric(all_rows["round"], errors="coerce").dropna()
        if len(rounds) == 0:
            return 1
        return int(rounds.max()) + 2

    def _planned_iteration_bounds(self) -> Tuple[int, int]:
        legacy_n = max(1, safe_int(self.config.get("planned_total_batches", 15), 15))
        low = max(1, safe_int(self.config.get("planned_iteration_min", legacy_n), legacy_n))
        high = max(1, safe_int(self.config.get("planned_iteration_max", legacy_n), legacy_n))
        if high < low:
            low, high = high, low
        return int(low), int(high)

    def _planned_iteration_range(self) -> Tuple[int, int, float]:
        low, high = self._planned_iteration_bounds()
        return low, high, (float(low) + float(high)) / 2.0

    def _normalized_transfer_fraction(self) -> float:
        return normalize_fraction(self.config.get("transfer_prior_fraction", 0.30), 0.30)

    def _auto_transfer_prior_rounds(self) -> int:
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
        mode = str(self.config.get("transfer_rounds_mode", "auto")).strip().lower()
        auto_m = self._auto_transfer_prior_rounds()
        _low, high = self._planned_iteration_bounds()
        if mode == "manual":
            manual_m = max(0, safe_int(self.config.get("transfer_prior_rounds", auto_m), auto_m))
            return min(high, manual_m)
        return auto_m

    def _transfer_active(self, all_rows: Optional[pd.DataFrame]) -> bool:
        if not bool(self.config.get("use_reference_prior", False)):
            return False
        m_batches = self._transfer_prior_rounds()
        if m_batches <= 0:
            return False
        return self._next_batch_count(all_rows) <= m_batches

    def _fit_reference_model(self) -> LightweightGaussianProcess:
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
            a = float(y_student_model[0] - mu_ref_student[0])
            b = 1.0
            calibration_mode = "offset-only calibration from one student point"
        else:
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

    def _condition_set(self, df: pd.DataFrame) -> Set[Tuple[int, int, int, int, int]]:
        if df is None or len(df) == 0:
            return set()
        return {condition_tuple(row) for _idx, row in df.iterrows()}

    def _training_frame(self, completed_rows: pd.DataFrame) -> pd.DataFrame:
        if completed_rows is None or len(completed_rows) == 0:
            return pd.DataFrame(columns=FEATURES + ["q"])
        df = completed_rows.copy()
        df["q"] = pd.to_numeric(df["q"], errors="coerce")
        df = df.dropna(subset=FEATURES + ["q"])
        return df

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
            if not chosen:
                best_idx = int(remaining[np.argmax(acq_z[remaining])])
                chosen.append(best_idx)
                remaining.remove(best_idx)
                continue

            chosen_X = X_scaled[chosen]
            scores = []
            for idx in remaining:
                dists = np.linalg.norm(chosen_X - X_scaled[idx], axis=1)
                min_dist = float(np.min(dists))
                score = float(acq_z[idx] + diversity_lambda * min_dist)
                scores.append(score)

            best_r_idx = int(np.argmax(scores))
            best_idx = remaining[best_r_idx]
            chosen.append(best_idx)
            remaining.remove(best_idx)

        out = candidates.iloc[chosen].copy().reset_index(drop=True)
        out["predicted_q_mean"] = pred_raw[chosen]
        out["predicted_q_sd"] = pred_sd_raw[chosen]
        out["acquisition_value"] = acq[chosen]
        return out

    def suggest_batch(
        self,
        completed_rows: pd.DataFrame,
        all_rows: Optional[pd.DataFrame] = None,
        candidate_pool: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, str]:
        batch_size = int(self.config.get("batch_size", 3))
        completed = completed_rows.copy() if completed_rows is not None and len(completed_rows) else pd.DataFrame()
        if len(completed) > 0:
            completed = completed[pd.to_numeric(completed["q"], errors="coerce").notna()].copy()

        tried = self._condition_set(all_rows if all_rows is not None and len(all_rows) else completed)
        candidates = self._prepare_candidate_pool(candidate_pool, tried, completed)

        if len(candidates) == 0:
            return pd.DataFrame(columns=FEATURES), "No valid un-tried candidate conditions remained in the target space."

        transfer_active = self._transfer_active(all_rows)

        if transfer_active:
            transfer = self._fit_calibrated_transfer(completed)
            X_cand_original = candidates[FEATURES].to_numpy(dtype=float)
            mu_model, sd_model = self._predict_calibrated_transfer(transfer, X_cand_original)

            train_df = self._training_frame(completed)
            if len(train_df) > 0:
                best_model = float(np.max(self._target_to_model(train_df["q"].to_numpy(dtype=float))))
            else:
                best_model = float(np.nanpercentile(mu_model, 75.0))

            acq = self._acquisition(mu_model, sd_model, best_model)

            pred_raw = self._target_from_model(mu_model)
            upper_raw = self._target_from_model(mu_model + sd_model)
            lower_raw = self._target_from_model(mu_model - sd_model)
            pred_sd_raw = np.maximum((upper_raw - lower_raw) / 2.0, 0.0)

            selected = self._select_diverse_batch(candidates, acq, pred_raw, pred_sd_raw, batch_size)
            return selected, self._transfer_message(transfer, all_rows)

        train_df = self._training_frame(completed)

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
        completed = completed_rows.copy() if completed_rows is not None and len(completed_rows) else pd.DataFrame()
        if len(completed) > 0:
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


def make_figure(
    completed: pd.DataFrame,
    experiments: Optional[pd.DataFrame] = None,
    candidate_pool: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, object]] = None,
) -> Tuple[Figure, str]:
    """Render 6-panel diagnostic Matplotlib figure."""
    if config is None:
        config = DEFAULT_CONFIG.copy()
    if experiments is None:
        experiments = pd.DataFrame(columns=EXPERIMENT_COLUMNS)

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

    comp_sorted = completed.sort_values(["round", "batch_position"]).reset_index(drop=True) if "round" in completed.columns else completed.reset_index(drop=True)
    q = comp_sorted["q"].astype(float).to_numpy()
    x = np.arange(1, len(q) + 1)
    best = np.maximum.accumulate(q)

    axes[0].plot(x, best, marker="o", color="#1f77b4")
    axes[0].set_title("Best q so far", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Completed experiment")
    axes[0].set_ylabel("Best q")

    axes[1].plot(x, q, marker="o", color="#2ca02c")
    axes[1].set_title("q by completed experiment", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Completed experiment")
    axes[1].set_ylabel("q")

    engine = BOEngine(config)
    diagnostics = engine.diagnostics(completed, experiments, candidate_pool, max_candidates=4000)
    summary = str(diagnostics.get("status", ""))

    comp_pred = diagnostics.get("completed_pred", pd.DataFrame())
    candidates = diagnostics.get("candidates", pd.DataFrame())
    model_ready = bool(diagnostics.get("model_ready", False))

    if model_ready and isinstance(comp_pred, pd.DataFrame) and len(comp_pred) > 0:
        obs = comp_pred["q"].astype(float).to_numpy()
        pred = comp_pred["model_predicted_q"].astype(float).to_numpy()
        sd = comp_pred["model_predicted_sd"].astype(float).to_numpy()
        axes[2].errorbar(obs, pred, yerr=sd, fmt="o", alpha=0.75, color="#ff7f0e")
        lo = float(np.nanmin([np.nanmin(obs), np.nanmin(pred)]))
        hi = float(np.nanmax([np.nanmax(obs), np.nanmax(pred)]))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            axes[2].plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="gray")
        axes[2].set_title("Model fit: observed q vs predicted q", fontsize=10, fontweight="bold")
        axes[2].set_xlabel("Observed q")
        axes[2].set_ylabel("Predicted q")
    else:
        axes[2].text(0.5, 0.5, "Model fit appears after at least\n3 completed results with variation.", ha="center", va="center")
        axes[2].set_title("Model fit", fontsize=10, fontweight="bold")

    if model_ready and isinstance(candidates, pd.DataFrame) and len(candidates) > 0:
        cand = candidates.copy()
        sug = cand[cand.get("is_open_suggestion", False).astype(bool)] if "is_open_suggestion" in cand.columns else pd.DataFrame()
        axes[3].scatter(cand["predicted_q_mean"], cand["predicted_q_sd"], s=12, alpha=0.35, color="#9467bd", label="candidate sample")
        if len(sug) > 0:
            axes[3].scatter(sug["predicted_q_mean"], sug["predicted_q_sd"], s=70, marker="*", color="#d62728", label="open suggestions")
        axes[3].set_title("Uncertainty landscape", fontsize=10, fontweight="bold")
        axes[3].set_xlabel("Predicted q mean")
        axes[3].set_ylabel("Predicted q standard deviation")
        axes[3].legend(fontsize=8)

        axes[4].scatter(cand["predicted_q_mean"], cand["acquisition_value"], s=12, alpha=0.35, color="#8c564b", label="candidate sample")
        if len(sug) > 0:
            axes[4].scatter(sug["predicted_q_mean"], sug["acquisition_value"], s=70, marker="*", color="#d62728", label="open suggestions")
        axes[4].set_title("How suggestions are proposed", fontsize=10, fontweight="bold")
        axes[4].set_xlabel("Predicted q mean")
        axes[4].set_ylabel(str(config.get("acquisition", "ei")).upper())
        axes[4].legend(fontsize=8)
    else:
        axes[3].text(0.5, 0.5, "No GP uncertainty yet.", ha="center", va="center")
        axes[3].set_title("Uncertainty landscape", fontsize=10, fontweight="bold")
        axes[4].text(0.5, 0.5, "No acquisition scores yet.", ha="center", va="center")
        axes[4].set_title("How suggestions are proposed", fontsize=10, fontweight="bold")

    axes[5].scatter(comp_sorted["metal_amount"].astype(float), comp_sorted["reaction_temperature"].astype(float), s=45, c=q, cmap="viridis", alpha=0.8)
    suggested = experiments[experiments["status"].astype(str) == "suggested"].copy() if experiments is not None and len(experiments) else pd.DataFrame()
    if len(suggested) > 0:
        axes[5].scatter(
            suggested["metal_amount"].astype(float),
            suggested["reaction_temperature"].astype(float),
            s=90,
            marker="*",
            color="#d62728",
            label="open suggestions",
        )
    axes[5].set_title("Experiment map", fontsize=10, fontweight="bold")
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
            summary = summary + " | " + " | ".join(top_lines)

    fig.tight_layout()
    return fig, summary
