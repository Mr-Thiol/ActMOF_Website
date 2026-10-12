# -*- coding: utf-8 -*-
"""
project_io.py - Saving and loading of ActMOF projects and candidate lists

A web session keeps its data only in the browser session, so a project is moved in and out
of the app as one .zip file. The archive holds the same files as a project folder of the
desktop app (project_config.json, experiments.csv and, if a custom list is used,
candidate_pool.csv), plus active_suggestions.csv for a batch that is proposed but not yet
measured. An unzipped project file can therefore be opened in the desktop app, and a zipped
desktop project folder can be loaded here.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from bo_engine import BOUNDS, DEFAULT_CONFIG, EXPERIMENT_COLUMNS, FEATURES, calc_q

CONFIG_FILE = "project_config.json"
EXPERIMENTS_FILE = "experiments.csv"
CANDIDATES_FILE = "candidate_pool.csv"
SUGGESTIONS_FILE = "active_suggestions.csv"

MAX_MEMBER_BYTES = 200 * 1024 * 1024  # refuse archive members that unpack to more than this


def normalize_experiments(df: pd.DataFrame) -> pd.DataFrame:
    """Check an experiments table and fill in the columns the app can derive.

    The five condition columns are required. A missing `status` is set to `completed`;
    a missing `q` is computed from `intensity` and `fwhm` when both are present.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"The experiments table must contain the columns {FEATURES}; missing: {missing}")
    if "status" not in df.columns:
        df["status"] = "completed"
    if "q" not in df.columns and "intensity" in df.columns and "fwhm" in df.columns:
        df["q"] = [calc_q(i, f) for i, f in zip(df["intensity"], df["fwhm"])]
    return df


def read_candidate_csv(source) -> Tuple[pd.DataFrame, str]:
    """Read a list of candidate conditions.

    Returns the cleaned table (five integer condition columns, no duplicates, all values
    inside the allowed ranges) and a one-line report of what was kept and dropped.
    """
    raw = pd.read_csv(source)
    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [f for f in FEATURES if f not in raw.columns]
    if missing:
        raise ValueError(f"The candidate CSV must contain the columns {FEATURES}; missing: {missing}")

    n_rows = len(raw)
    values = raw[FEATURES].apply(pd.to_numeric, errors="coerce")
    is_integer = values.notna().all(axis=1) & (values.round() == values).all(axis=1)
    values = values[is_integer]
    n_not_integer = n_rows - len(values)

    in_range = pd.Series(True, index=values.index)
    for feature in FEATURES:
        lo, hi, _step = BOUNDS[feature]
        in_range &= values[feature].between(lo, hi)
    values = values[in_range]
    n_out_of_range = n_rows - n_not_integer - len(values)

    n_before = len(values)
    values = values.drop_duplicates(subset=FEATURES).astype(int).reset_index(drop=True)
    n_duplicates = n_before - len(values)

    if len(values) == 0:
        raise ValueError("No usable candidate condition was found in the CSV (all rows were empty, "
                         "non-integer or outside the allowed ranges).")

    dropped = []
    if n_not_integer:
        dropped.append(f"{n_not_integer} with missing or non-integer values")
    if n_out_of_range:
        dropped.append(f"{n_out_of_range} outside the allowed ranges")
    if n_duplicates:
        dropped.append(f"{n_duplicates} duplicates")
    report = f"{len(values):,} candidate conditions loaded"
    if dropped:
        report += " (dropped: " + ", ".join(dropped) + ")"
    return values, report + "."


def candidate_template_csv() -> bytes:
    """A small example of a candidate list (header and three rows)."""
    rows = [[BOUNDS[f][0] for f in FEATURES],
            [int(round((BOUNDS[f][0] + BOUNDS[f][1]) / 2)) for f in FEATURES],
            [BOUNDS[f][1] for f in FEATURES]]
    return pd.DataFrame(rows, columns=FEATURES).to_csv(index=False).encode("utf-8")


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def build_project_zip(
    config: Dict[str, object],
    experiments: Optional[pd.DataFrame],
    candidate_pool: Optional[pd.DataFrame] = None,
    suggestions: Optional[pd.DataFrame] = None,
) -> bytes:
    """Pack the current project into the bytes of a .zip archive."""
    if experiments is None:
        experiments = pd.DataFrame(columns=EXPERIMENT_COLUMNS)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(CONFIG_FILE, json.dumps(dict(config), indent=2, default=_json_default))
        archive.writestr(EXPERIMENTS_FILE, experiments.to_csv(index=False))
        if candidate_pool is not None and len(candidate_pool) > 0:
            archive.writestr(CANDIDATES_FILE, candidate_pool[FEATURES].to_csv(index=False))
        if suggestions is not None and len(suggestions) > 0:
            archive.writestr(SUGGESTIONS_FILE, suggestions.to_csv(index=False))
    return buffer.getvalue()


def load_project_zip(data: bytes) -> Dict[str, object]:
    """Unpack a project archive.

    Returns a dict with the keys `config`, `experiments`, `candidate_pool` (None if the
    project has no custom list), `suggestions` (None if no batch was active) and `summary`.
    Files are found by name, so an archive made by zipping a project folder of the desktop
    app (files inside one top-level folder) is accepted as well.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("The uploaded file is not a .zip project file.") from exc

    with archive:
        members: Dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            name = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
            if name in (CONFIG_FILE, EXPERIMENTS_FILE, CANDIDATES_FILE, SUGGESTIONS_FILE) and name not in members:
                if info.file_size > MAX_MEMBER_BYTES:
                    raise ValueError(f"{name} in the project file is too large.")
                members[name] = info
        if CONFIG_FILE not in members and EXPERIMENTS_FILE not in members:
            raise ValueError(f"The project file contains neither {CONFIG_FILE} nor {EXPERIMENTS_FILE}.")

        def read_csv(name: str) -> pd.DataFrame:
            return pd.read_csv(io.BytesIO(archive.read(members[name])))

        config = DEFAULT_CONFIG.copy()
        if CONFIG_FILE in members:
            stored = json.loads(archive.read(members[CONFIG_FILE]).decode("utf-8"))
            if not isinstance(stored, dict):
                raise ValueError(f"{CONFIG_FILE} does not contain a settings dictionary.")
            config.update(stored)

        experiments = pd.DataFrame(columns=EXPERIMENT_COLUMNS)
        if EXPERIMENTS_FILE in members:
            table = read_csv(EXPERIMENTS_FILE)
            if len(table) > 0:
                experiments = normalize_experiments(table)

        candidate_pool = None
        if CANDIDATES_FILE in members:
            candidate_pool, _report = read_candidate_csv(io.BytesIO(archive.read(members[CANDIDATES_FILE])))
        config["candidate_mode"] = "csv" if candidate_pool is not None else "random"

        suggestions = None
        if SUGGESTIONS_FILE in members:
            table = read_csv(SUGGESTIONS_FILE)
            if len(table) > 0 and all(f in table.columns for f in FEATURES):
                for col in ["intensity", "fwhm", "q"]:
                    if col not in table.columns:
                        table[col] = np.nan
                table["notes"] = table["notes"].fillna("").astype(str) if "notes" in table.columns else ""
                suggestions = table

    n_completed = int((experiments["status"].astype(str) == "completed").sum()) if len(experiments) else 0
    parts = [f"{len(experiments)} experiment row(s), {n_completed} completed"]
    if candidate_pool is not None:
        parts.append(f"{len(candidate_pool):,} custom candidate conditions")
    if suggestions is not None:
        parts.append(f"an active batch of {len(suggestions)} suggestion(s)")
    summary = f"Project '{config.get('project_name', '')}' loaded: " + "; ".join(parts) + "."
    return {"config": config, "experiments": experiments, "candidate_pool": candidate_pool,
            "suggestions": suggestions, "summary": summary}


def figure_png_bytes(fig, dpi: int = 200) -> bytes:
    """Render a Matplotlib figure to PNG bytes."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    return buffer.getvalue()
