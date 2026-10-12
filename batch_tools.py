# -*- coding: utf-8 -*-
"""
batch_tools.py - Helpers for entering and correcting the results of a batch

- apply_measurement_rules: fills in what follows from the entered values
  (intensity 0 -> FWHM placeholder 30; intensity and FWHM -> q).
- missing_measurements: lists the cells that are still empty.
- pop_last_batch: takes the most recently saved batch out of the experiment table
  so that it can be corrected and saved again.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from bo_engine import FEATURES, calc_q

NO_PRODUCT_FWHM = 30.0  # FWHM recorded for a pattern without a diffraction peak (q = 0)
REFERENCE_NOTE = "Reference benchmark prior point"
MEASUREMENT_COLUMNS = ["intensity", "fwhm", "q"]


def apply_measurement_rules(table: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Fill in the values that follow from what was entered.

    1. intensity = 0 and FWHM empty: FWHM is set to the no-product placeholder 30.
    2. intensity and FWHM both entered: q is calculated from them (replacing a different q).

    Returns the table with numeric measurement columns and one message per change;
    an empty list means nothing had to be filled in.
    """
    out = table.copy()
    for col in MEASUREMENT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce") if col in out.columns else np.nan
    messages: List[str] = []
    for label in out.index:
        intensity, fwhm, q = (float(out.at[label, c]) for c in MEASUREMENT_COLUMNS)
        if np.isfinite(intensity) and intensity == 0 and not np.isfinite(fwhm):
            fwhm = NO_PRODUCT_FWHM
            out.at[label, "fwhm"] = fwhm
            messages.append(f"{label}: intensity is 0, FWHM set to the placeholder {NO_PRODUCT_FWHM:g}")
        if np.isfinite(intensity) and np.isfinite(fwhm):
            q_new = float(calc_q(intensity, fwhm))
            if not np.isfinite(q) or abs(q - q_new) > 1e-9:
                out.at[label, "q"] = q_new
                messages.append(f"{label}: q = {q_new:g}")
    return out, messages


def missing_measurements(table: pd.DataFrame) -> List[str]:
    """Rows whose result is incomplete, as 'label (empty cells)'.

    A row is complete when intensity and FWHM are both entered, or when q is entered directly.
    """
    missing: List[str] = []
    for label, row in table.iterrows():
        values = {c: pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0] for c in MEASUREMENT_COLUMNS}
        has_pair = np.isfinite(values["intensity"]) and np.isfinite(values["fwhm"])
        if has_pair or np.isfinite(values["q"]):
            continue
        empty = [c for c in ("intensity", "fwhm") if not np.isfinite(values[c])]
        missing.append(f"{label} ({', '.join(empty)})")
    return missing


def pop_last_batch(experiments: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Split the experiment table into (everything before the last saved batch, that batch).

    The last batch is the block of rows at the end of the table that share the round of the last
    row. Returns (experiments, None) when there is no batch to reopen: an empty table, a table
    without a `round` column, or one that ends with the built-in reference data.
    """
    if experiments is None or len(experiments) == 0 or "round" not in experiments.columns:
        return experiments, None
    table = experiments.reset_index(drop=True)
    rounds = pd.to_numeric(table["round"], errors="coerce")
    last_round = rounds.iloc[-1]
    if not np.isfinite(last_round):
        return experiments, None
    start = len(table)
    while start > 0 and rounds.iloc[start - 1] == last_round:
        start -= 1
    batch = table.iloc[start:].copy().reset_index(drop=True)
    if "notes" in batch.columns and (batch["notes"].astype(str) == REFERENCE_NOTE).any():
        return experiments, None
    if any(f not in batch.columns for f in FEATURES):
        return experiments, None

    keep = [c for c in ["batch_position", *FEATURES, *MEASUREMENT_COLUMNS,
                        "predicted_q_mean", "predicted_q_sd", "acquisition_value", "notes"] if c in batch.columns]
    batch = batch[keep]
    for col in MEASUREMENT_COLUMNS:
        batch[col] = pd.to_numeric(batch[col], errors="coerce") if col in batch.columns else np.nan
    batch["notes"] = batch["notes"].fillna("").astype(str) if "notes" in batch.columns else ""
    return table.iloc[:start].copy().reset_index(drop=True), batch
