# -*- coding: utf-8 -*-
"""
app.py - Streamlit Web Application for ActMOF Bayesian Optimization

Multi-tab workflow based on EDBO+ user interaction choices:
Tab 1: Dashboard & Visual Diagnostics
Tab 2: Batch Recommendations & Results Entry (Initialize BO + Automatic Model Rerun)
Tab 3: Experiment Data & History
Tab 4: Project Planning Wizard
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from rasx_process import load_and_calc_q

from bo_engine import (
    APP_NAME,
    APP_VERSION,
    BOUNDS,
    DEFAULT_CONFIG,
    EXPERIMENT_COLUMNS,
    FEATURES,
    REFERENCE_DF,
    BOEngine,
    calc_q,
    condition_tuple,
    format_record_id,
    make_figure,
    now_text,
    valid_condition_values,
)

# Set page config
st.set_page_config(
    page_title="ActMOF — MOF Bayesian Optimization WebApp",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for EDBO+-style branding and clean card layouts
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563EB;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize persistent Streamlit session state variables."""
    if "config" not in st.session_state:
        st.session_state.config = DEFAULT_CONFIG.copy()
    if "experiments" not in st.session_state:
        st.session_state.experiments = pd.DataFrame(columns=EXPERIMENT_COLUMNS)
    if "candidate_pool" not in st.session_state:
        st.session_state.candidate_pool = None
    if "last_suggestions" not in st.session_state:
        st.session_state.last_suggestions = None
    if "last_message" not in st.session_state:
        st.session_state.last_message = ""
    if "uploaded_file_signature" not in st.session_state:
        st.session_state.uploaded_file_signature = None
    if "suggestions_editor_version" not in st.session_state:
        st.session_state.suggestions_editor_version = 0


init_session_state()

# Header & Branding
col_logo, col_header = st.columns([1, 6])
with col_logo:
    st.markdown("## 🧪")
with col_header:
    st.markdown('<div class="main-title">ActMOF — MOF Bayesian Optimization</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Active-learning synthesis optimization web platform based on EDBO+ workflow.</div>',
        unsafe_allow_html=True,
    )

# Sidebar - Project Setup & BO Hyperparameters (EDBO+ Controls)
with st.sidebar:
    st.header("⚙️ Optimization Setup")

    with st.expander("📌 Project Information", expanded=True):
        st.session_state.config["project_name"] = st.text_input(
            "Project Name", value=str(st.session_state.config.get("project_name", "MOF Synthesis"))
        )
        st.session_state.config["batch_size"] = st.number_input(
            "Experiments per Batch (k)", min_value=1, max_value=20, value=int(st.session_state.config.get("batch_size", 3))
        )

    with st.expander("🧠 Model & Acquisition", expanded=True):
        st.session_state.config["kernel"] = st.selectbox(
            "GP Kernel",
            options=["matern52", "matern32"],
            index=0 if st.session_state.config.get("kernel") == "matern52" else 1,
            help="Matérn 5/2 is recommended for smooth physical synthesis surfaces.",
        )
        acq_choice = st.selectbox(
            "Acquisition Function",
            options=["Expected Improvement (EI)", "Probability of Improvement (PI)"],
            index=0 if st.session_state.config.get("acquisition") == "ei" else 1,
        )
        st.session_state.config["acquisition"] = "ei" if "Expected" in acq_choice else "pi"

        st.session_state.config["seed"] = st.number_input(
            "Random Seed", min_value=1, max_value=999999, value=int(st.session_state.config.get("seed", 42))
        )
        st.session_state.config["use_log1p_target"] = st.checkbox(
            "Log1p Target Scaling (y = ln(1+q))", value=bool(st.session_state.config.get("use_log1p_target", True))
        )

    with st.expander("🔄 Calibrated Transfer Prior", expanded=False):
        st.session_state.config["use_reference_prior"] = st.checkbox(
            "Enable Calibrated Transfer Prior", value=bool(st.session_state.config.get("use_reference_prior", False))
        )
        frac = st.slider(
            "Transfer Fraction",
            min_value=0.05,
            max_value=1.00,
            value=float(st.session_state.config.get("transfer_prior_fraction", 0.30)),
            step=0.05,
        )
        st.session_state.config["transfer_prior_fraction"] = frac
        st.session_state.config["transfer_rounds_mode"] = st.radio(
            "Transfer Rounds Mode", options=["auto", "manual"], index=0
        )
        if st.session_state.config["transfer_rounds_mode"] == "manual":
            st.session_state.config["transfer_prior_rounds"] = st.number_input(
                "Transfer Prior Batches (M)", min_value=1, max_value=50, value=int(st.session_state.config.get("transfer_prior_rounds", 4))
            )

    st.markdown("---")
    st.subheader("📥 Reset & Presets")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("Load Benchmark Data", use_container_width=True):
            ref_rows = []
            for idx, row in REFERENCE_DF.iterrows():
                ref_rows.append({
                    "record_id": format_record_id(idx + 1),
                    "round": 0,
                    "batch_position": idx + 1,
                    "status": "completed",
                    "metal_amount": int(row["metal_amount"]),
                    "modulator": int(row["modulator"]),
                    "add_solvent": int(row["add_solvent"]),
                    "reaction_time": int(row["reaction_time"]),
                    "reaction_temperature": int(row["reaction_temperature"]),
                    "intensity": float(row["intensity"]),
                    "fwhm": float(row["fwhm"]),
                    "q": float(row["q"]),
                    "predicted_q_mean": np.nan,
                    "predicted_q_sd": np.nan,
                    "acquisition_value": np.nan,
                    "notes": "Reference benchmark prior point",
                    "created_at": now_text(),
                    "updated_at": now_text(),
                })
            st.session_state.experiments = pd.DataFrame(ref_rows)
            st.success(f"Loaded {len(ref_rows)} reference benchmark rows!")
            st.rerun()
    with col_r2:
        if st.button("Clear All Data", use_container_width=True):
            st.session_state.experiments = pd.DataFrame(columns=EXPERIMENT_COLUMNS)
            st.session_state.last_suggestions = None
            st.session_state.last_message = ""
            st.warning("Cleared active experiments dataset.")
            st.rerun()


# EDBO+ Main Workflow Tabs
tab_dash, tab_recommend, tab_data, tab_wizard, tab_help = st.tabs([
    "📊 Dashboard & Diagnostics",
    "🧪 Batch Recommendations",
    "📝 Experiment Entry & Data",
    "🧙 Planning Wizard",
    "❓ Help & Guide",
])

# -----------------------------------------------------------------------------
# TAB 1: Dashboard & Diagnostics
# -----------------------------------------------------------------------------
with tab_dash:
    df_exp = st.session_state.experiments
    completed_df = df_exp[df_exp["status"].astype(str) == "completed"].copy() if len(df_exp) else pd.DataFrame()

    # Metric Cards Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Experiments", len(df_exp))
    with m2:
        st.metric("Completed Runs", len(completed_df))
    with m3:
        best_q = float(completed_df["q"].max()) if len(completed_df) and "q" in completed_df.columns and completed_df["q"].notna().any() else 0.0
        st.metric("Best q Score", f"{best_q:.1f}" if best_q > 0 else "N/A")
    with m4:
        engine_tmp = BOEngine(st.session_state.config)
        next_b = engine_tmp._next_batch_count(df_exp)
        st.metric("Next Batch #", f"Batch {next_b}")

    st.markdown("### 📈 Visual Diagnostics")

    # Render 6-panel Figure
    fig, summary_msg = make_figure(
        completed=completed_df,
        experiments=df_exp,
        candidate_pool=st.session_state.candidate_pool,
        config=st.session_state.config,
    )
    st.pyplot(fig, use_container_width=True)

    if summary_msg:
        st.info(f"**Model Status:** {summary_msg}")


# -----------------------------------------------------------------------------
# TAB 2: Batch Recommendations & Results Entry (Initialize & Auto Rerun)
# -----------------------------------------------------------------------------
with tab_recommend:
    st.subheader("💡 Initialize BO & Enter Batch Measurement Results")
    st.markdown(
        "Click **Initialize BO Optimization** to generate the initial batch when starting a project. "
        "After conducting reactions, enter your measurement results (`intensity`, `fwhm`, or `q`) below and click "
        "**Update Model & Save Batch Results** to automatically re-run the model and receive the next batch suggestions!"
    )

    df_exp = st.session_state.experiments
    has_exp_data = len(df_exp) > 0
    has_session_prediction = st.session_state.last_suggestions is not None and len(st.session_state.last_suggestions) > 0
    can_initialize_prediction = not has_session_prediction

    col_btn, col_info = st.columns([1.5, 3])
    with col_btn:
        init_clicked = st.button(
            "⚡ Initialize BO Optimization",
            type="primary",
            disabled=not can_initialize_prediction,
            use_container_width=True,
            help="Enabled for the first prediction in this web session, including after uploading completed experimental data.",
        )
        if init_clicked:
            with st.spinner("Evaluating initial candidate space..."):
                engine = BOEngine(st.session_state.config)
                completed_rows = df_exp.copy()
                if has_exp_data and "q" in completed_rows.columns:
                    completed_rows = completed_rows[pd.to_numeric(completed_rows["q"], errors="coerce").notna()].copy()
                sug_df, msg = engine.suggest_batch(
                    completed_rows=completed_rows,
                    all_rows=df_exp.copy() if has_exp_data else pd.DataFrame(),
                    candidate_pool=st.session_state.candidate_pool,
                )
                for col in ["intensity", "fwhm", "q", "notes"]:
                    if col not in sug_df.columns:
                        sug_df[col] = np.nan if col != "notes" else ""
                st.session_state.last_suggestions = sug_df
                st.session_state.last_message = msg
                st.session_state.suggestions_editor_version += 1

    if has_session_prediction:
        st.caption("Initialize BO Optimization is disabled because this session already has active suggestions. Enter results below and click 'Update Model' to re-run.")

    if st.session_state.last_message:
        st.success(f"**Engine Status:** {st.session_state.last_message}")

    if st.session_state.last_suggestions is not None and len(st.session_state.last_suggestions) > 0:
        sug_df = st.session_state.last_suggestions
        st.markdown("#### Proposed Batch Conditions & Experimental Measurement Feedback")
        st.caption("Enter measured Intensity and FWHM (or q) directly into the spreadsheet table below:")

        display_cols = [f for f in FEATURES if f in sug_df.columns]
        measurement_cols = [c for c in ["intensity", "fwhm", "q"] if c in sug_df.columns]
        extra_cols = [c for c in ["predicted_q_mean", "predicted_q_sd", "acquisition_value"] if c in sug_df.columns]
        feedback_cols = measurement_cols + ["notes"]

        # Column order matches experiments.csv: features, then intensity/fwhm/q,
        # then the model's predicted stats, then notes.
        full_cols = display_cols + measurement_cols + extra_cols + ["notes"]
        show_cols = [c for c in full_cols if c in sug_df.columns]

        # Preview record IDs (R00016, R00017, ...) matching what will be assigned
        # when this batch is saved, so the leftmost row label reads like experiments.csv
        # instead of a bare 0, 1, 2, ... index.
        next_start = len(st.session_state.experiments) + 1
        preview_ids = [format_record_id(next_start + i) for i in range(len(sug_df))]
        display_sug_df = sug_df[show_cols].copy()
        display_sug_df.index = pd.Index(preview_ids, name="record_id")

        edited_sug_df = st.data_editor(
            display_sug_df,
            num_rows="fixed",
            use_container_width=True,
            key=f"suggestions_editor_{st.session_state.suggestions_editor_version}",
            column_config={
                "intensity": st.column_config.NumberColumn("Intensity (Measured)", min_value=0.0),
                "fwhm": st.column_config.NumberColumn("FWHM (Measured)", min_value=0.01),
                "q": st.column_config.NumberColumn("q Score", min_value=0.0),
                "notes": st.column_config.TextColumn("Notes"),
            },
        )

        st.markdown("#### Calculate Measurements from .rasx")
        rasx_file = st.file_uploader(
            "Drag and drop a .rasx file to calculate intensity, FWHM, and q Score",
            type=["rasx"],
            key="rasx_measurement_upload",
        )
        if rasx_file is not None:
            try:
                rasx_result = load_and_calc_q(io.BytesIO(rasx_file.getvalue()))
                calculated_intensity = float(rasx_result["peak_intensity"])
                calculated_fwhm = float(rasx_result["half_width"])
                calculated_q = float(rasx_result["q"])

                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Calculated Intensity", f"{calculated_intensity:.4g}")
                m_col2.metric("Calculated FWHM", f"{calculated_fwhm:.4g}")
                m_col3.metric("Calculated q Score", f"{calculated_q:.4g}")
                st.pyplot(rasx_result["fig"], use_container_width=True)

                measurement_cols = [c for c in ["intensity", "fwhm", "q"] if c in edited_sug_df.columns]
                if measurement_cols:
                    numeric_measurements = edited_sug_df[measurement_cols].apply(pd.to_numeric, errors="coerce")
                    rows_with_empty_measurement = numeric_measurements.isna().any(axis=1).tolist()
                    default_row = next((i for i, is_empty in enumerate(rows_with_empty_measurement) if is_empty), 0)
                else:
                    default_row = 0

                row_options = list(range(len(edited_sug_df)))

                def format_feedback_row(row_pos: int) -> str:
                    row = edited_sug_df.iloc[row_pos]
                    batch_pos = row.get("batch_position", row_pos + 1)
                    condition = ", ".join(f"{feature}={row.get(feature)}" for feature in FEATURES if feature in row)
                    return f"Row {row_pos + 1} / batch position {batch_pos}: {condition}"

                selected_row_pos = st.selectbox(
                    "Choose the proposed-batch row to fill",
                    options=row_options,
                    index=default_row,
                    format_func=format_feedback_row,
                    key=f"rasx_fill_row_{st.session_state.suggestions_editor_version}",
                )
                if st.button("Fill Selected Row with Calculated Measurements", use_container_width=True):
                    updated_suggestions = st.session_state.last_suggestions.copy().reset_index(drop=True)
                    edited_rows = edited_sug_df.reset_index(drop=True)
                    for col in show_cols:
                        if col in updated_suggestions.columns and col in edited_rows.columns:
                            updated_suggestions[col] = edited_rows[col]
                    updated_suggestions.loc[selected_row_pos, "intensity"] = calculated_intensity
                    updated_suggestions.loc[selected_row_pos, "fwhm"] = calculated_fwhm
                    updated_suggestions.loc[selected_row_pos, "q"] = calculated_q
                    st.session_state.last_suggestions = updated_suggestions
                    st.session_state.suggestions_editor_version += 1
                    st.success(f"Filled row {selected_row_pos + 1} with measurements from {rasx_file.name}.")
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not process .rasx file: {exc}")

        col_update, col_dl = st.columns([2, 1])
        with col_update:
            if st.button("🔄 Update Model & Save Batch Results", type="primary", use_container_width=True):
                engine_tmp = BOEngine(st.session_state.config)
                batch_no = engine_tmp._next_batch_count(df_exp)
                new_rows = []

                for pos, (_idx, row) in enumerate(edited_sug_df.iterrows(), start=1):
                    rec_id = format_record_id(len(st.session_state.experiments) + pos)
                    raw_int = row.get("intensity")
                    raw_fwhm = row.get("fwhm")
                    raw_q = row.get("q")

                    computed_q = calc_q(raw_int, raw_fwhm, raw_q)
                    status_val = "completed" if computed_q > 0 or pd.notna(raw_q) else "suggested"

                    new_rows.append({
                        "record_id": rec_id,
                        "round": batch_no - 1,
                        "batch_position": pos,
                        "status": status_val,
                        "metal_amount": int(row["metal_amount"]),
                        "modulator": int(row["modulator"]),
                        "add_solvent": int(row["add_solvent"]),
                        "reaction_time": int(row["reaction_time"]),
                        "reaction_temperature": int(row["reaction_temperature"]),
                        "intensity": float(raw_int) if pd.notna(raw_int) else np.nan,
                        "fwhm": float(raw_fwhm) if pd.notna(raw_fwhm) else np.nan,
                        "q": float(computed_q) if computed_q > 0 else (float(raw_q) if pd.notna(raw_q) else np.nan),
                        "predicted_q_mean": float(row.get("predicted_q_mean", np.nan)),
                        "predicted_q_sd": float(row.get("predicted_q_sd", np.nan)),
                        "acquisition_value": float(row.get("acquisition_value", np.nan)),
                        "notes": str(row.get("notes", "")) or f"Batch {batch_no} BO suggestion",
                        "created_at": now_text(),
                        "updated_at": now_text(),
                    })

                # Automatically append batch results into active experiment history
                updated_exp = pd.concat([st.session_state.experiments, pd.DataFrame(new_rows)], ignore_index=True)
                st.session_state.experiments = updated_exp

                # Automatically re-run the BO optimization model to generate next batch suggestions immediately
                with st.spinner("Re-fitting model and generating next batch suggestions..."):
                    engine = BOEngine(st.session_state.config)
                    completed_df = updated_exp[updated_exp["status"].astype(str) == "completed"].copy() if len(updated_exp) else pd.DataFrame()
                    new_sug_df, msg = engine.suggest_batch(
                        completed_rows=completed_df,
                        all_rows=updated_exp,
                        candidate_pool=st.session_state.candidate_pool,
                    )
                    for col in ["intensity", "fwhm", "q", "notes"]:
                        if col not in new_sug_df.columns:
                            new_sug_df[col] = np.nan if col != "notes" else ""
                    st.session_state.last_suggestions = new_sug_df
                    st.session_state.last_message = f"Batch results saved! Model re-fitted with {len(completed_df)} completed run(s). {msg}"
                    st.session_state.suggestions_editor_version += 1

                st.rerun()

        with col_dl:
            csv_data = edited_sug_df[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Suggestions CSV",
                data=csv_data,
                file_name=f"mof_bo_batch_suggestions_{now_text()[:10]}.csv",
                mime="text/csv",
                use_container_width=True,
            )


# -----------------------------------------------------------------------------
# TAB 3: Experiment Data & History
# -----------------------------------------------------------------------------
with tab_data:
    st.subheader("📝 Complete Experiment History & Data Entry")
    st.markdown("Upload prior experimental results or directly inspect/edit all project runs.")

    col_up, col_dl_all = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader("Upload Experiments CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                if all(f in uploaded_df.columns for f in FEATURES):
                    if "status" not in uploaded_df.columns:
                        uploaded_df["status"] = "completed"
                    if "q" not in uploaded_df.columns and "intensity" in uploaded_df.columns and "fwhm" in uploaded_df.columns:
                        uploaded_df["q"] = [calc_q(i, f) for i, f in zip(uploaded_df["intensity"], uploaded_df["fwhm"])]
                    upload_signature = (uploaded_file.name, uploaded_file.size)
                    if st.session_state.uploaded_file_signature != upload_signature:
                        st.session_state.last_suggestions = None
                        st.session_state.last_message = ""
                        st.session_state.uploaded_file_signature = upload_signature
                    st.session_state.experiments = uploaded_df
                    st.success(f"Successfully loaded {len(uploaded_df)} rows from uploaded file!")
                else:
                    st.error(f"Uploaded CSV must contain feature columns: {FEATURES}")
            except Exception as e:
                st.error(f"Error parsing uploaded CSV: {e}")

    with col_dl_all:
        if len(st.session_state.experiments) > 0:
            all_csv = st.session_state.experiments.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export All Experiments CSV",
                data=all_csv,
                file_name=f"mof_bo_all_experiments_{now_text()[:10]}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.markdown("#### Interactive History Table")
    if len(st.session_state.experiments) > 0:
        # Show record_id (R00001, R00002, ...) as the leftmost row label instead
        # of a bare 0, 1, 2, ... index, matching experiments.csv.
        display_hist_df = st.session_state.experiments.copy()
        if "record_id" in display_hist_df.columns:
            display_hist_df = display_hist_df.set_index("record_id")

        edited_hist_display = st.data_editor(
            display_hist_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "status": st.column_config.SelectboxColumn("Status", options=["completed", "suggested", "cancelled"]),
                "intensity": st.column_config.NumberColumn("Intensity", min_value=0.0),
                "fwhm": st.column_config.NumberColumn("FWHM", min_value=0.01),
                "q": st.column_config.NumberColumn("q Score", min_value=0.0),
            },
        )
        # Restore record_id as a normal leading column (matching experiments.csv)
        # before this data is used or persisted further.
        edited_df = edited_hist_display.reset_index() if edited_hist_display.index.name == "record_id" else edited_hist_display
        if st.button("💾 Save Table Edits", type="primary"):
            for idx, row in edited_df.iterrows():
                if pd.notna(row.get("intensity")) and pd.notna(row.get("fwhm")) and float(row["fwhm"]) > 0:
                    computed_q = calc_q(row["intensity"], row["fwhm"])
                    edited_df.at[idx, "q"] = computed_q
                    edited_df.at[idx, "status"] = "completed"
            st.session_state.experiments = edited_df
            st.success("Updated experiment table and recalculated q scores!")
            st.rerun()
    else:
        st.info("No experiment rows in history. Click 'Load Benchmark Data' in the sidebar or upload a CSV to start.")


# -----------------------------------------------------------------------------
# TAB 4: Planning Wizard (EDBO+ Project Setup)
# -----------------------------------------------------------------------------
with tab_wizard:
    st.subheader("🧙 Project Planning & Calibrated Transfer Wizard")
    st.markdown(
        "Estimate your total lab capacity and iteration range to compute the optimal "
        r"calibrated-transfer window $M = \lfloor \text{mean\_range} \times \text{fraction} \rfloor$."
    )

    w1, w2 = st.columns(2)
    with w1:
        exp_per_batch = st.number_input("Experiments per Batch", min_value=1, max_value=20, value=3)
        iter_min = st.number_input("Estimated Min Iterations", min_value=1, max_value=100, value=10)
        iter_max = st.number_input("Estimated Max Iterations", min_value=1, max_value=100, value=20)
        transfer_pct = st.slider("Transfer Fraction (%)", min_value=5, max_value=100, value=30, step=5)

    with w2:
        avg_iter = (iter_min + iter_max) / 2.0
        m_calc = int(np.floor(avg_iter * (transfer_pct / 100.0)))
        total_exp_min = exp_per_batch * iter_min
        total_exp_max = exp_per_batch * iter_max

        st.markdown("#### Wizard Summary")
        st.info(
            f"• **Total Experiments Range:** {total_exp_min} to {total_exp_max} synthesis runs\n"
            f"• **Average Planned Iterations:** {avg_iter:.1f} batches\n"
            f"• **Calibrated-Transfer Prior Window (M):** First **{m_calc}** batch(es)"
        )

        if st.button("Apply Wizard Settings to Active Project", type="primary"):
            st.session_state.config["batch_size"] = int(exp_per_batch)
            st.session_state.config["planned_iteration_min"] = int(iter_min)
            st.session_state.config["planned_iteration_max"] = int(iter_max)
            st.session_state.config["transfer_prior_fraction"] = float(transfer_pct / 100.0)
            st.session_state.config["transfer_prior_rounds"] = int(m_calc)
            st.success("Successfully applied wizard settings to project configuration!")
            st.rerun()


# -----------------------------------------------------------------------------
# TAB 5: Help & Guide
# -----------------------------------------------------------------------------
with tab_help:
    st.subheader("❓ How to Use ActMOF")
    st.markdown(
        "Each section below gives a **plain-language** explanation first, followed by the "
        "**rigorous** technical definition — skim for \"what do I click\" or expand for \"what is "
        "the app actually computing.\" The full write-up also lives in `HELP.md` in the repository."
    )

    st.markdown("### 🚀 Quickstart")
    st.markdown(
        """
1. *(Optional)* **Planning Wizard tab** — enter experiments per batch and a rough min–max estimate of
   how many batches your project will take, then click **Apply Wizard Settings**.
2. **Sidebar → Optimization Setup** — set project name, batch size, kernel, and acquisition function.
   Defaults work well for most projects.
3. *(Optional)* **Sidebar → Calibrated Transfer Prior** — turn on if you want your first few batches
   to be informed by the built-in benchmark data instead of pure random exploration.
4. **Batch Recommendations tab → ⚡ Initialize BO Optimization** — generates your first batch of
   conditions to try.
5. Run the reactions. Enter `intensity` + `fwhm` directly, **or** upload the `.rasx` XRD file for a
   row and click **Fill Selected Row with Calculated Measurements**.
6. Click **🔄 Update Model & Save Batch Results** — saves your results, re-fits the model, and
   proposes the next batch. Repeat from step 5.
7. Check the **Dashboard tab** anytime for progress plots and model diagnostics.
        """
    )

    with st.expander("🧪 What does the app optimize? (the q score)", expanded=False):
        st.markdown(
            "**Plain language:** After a reaction, you scan the powder with XRD. A well-crystallized "
            "MOF gives one tall, narrow diffraction peak. `q` rewards *tall and narrow* — bigger q "
            "means a sharper, more crystalline product. A failed/amorphous reaction gets `q = 0`.\n\n"
            "**Rigorously:** `q = peak_intensity / FWHM`, where FWHM (full width at half maximum) is "
            "found by linear interpolation between the two points nearest the peak on the intensity "
            "trace, on each side of the peak. Uploading a `.rasx` file computes this automatically; "
            "you can also type `intensity`/`fwhm`, or `q` itself, directly. The app **maximizes** q — "
            "it is chasing crystallinity/phase purity, not yield or particle size."
        )

    with st.expander("📦 Rounds, batches, and status", expanded=False):
        st.markdown(
            "Every proposed condition becomes one row in your experiment log. A **batch** is a set of "
            "`k` conditions proposed together (`k` = *Experiments per Batch*). A **round** is the "
            "0-indexed counter for batches (`round = batch_number - 1`). Row `status` is `suggested` "
            "(proposed, no result yet), `completed` (has a valid measurement, counts as training "
            "data), or `cancelled` (excluded from the model without deleting the row). Record IDs are "
            "formatted `R00001`, `R00002`, … in creation order."
        )

    with st.expander("🧠 How are the next conditions chosen?", expanded=False):
        st.markdown(
            "**Plain language:** With fewer than 3 completed results (and transfer prior off), the "
            "app samples spread-out random conditions — there isn't enough data yet to learn a trend. "
            "Once you have 3+ results with some variation in q, it proposes the batch of conditions "
            "that best balances *predicted high q* against *genuine uncertainty*, while keeping the "
            "batch spread out so you're not wasting reactions on near-duplicates.\n\n"
            "**Rigorously:** A NumPy-only Gaussian Process (Matérn 3/2 or 5/2 kernel) is fit to "
            "`(scaled condition) → target`, where target is optionally `ln(1+q)` (default on, dampens "
            "outlier-large q values). Length-scale and noise are chosen by grid search minimizing the "
            "negative log marginal likelihood, with Cholesky-jitter fallback for stability. ~15,000 "
            "random candidates are scored with Expected Improvement or Probability of Improvement "
            "against the best completed result, and a batch of `k` is picked greedily, trading "
            "acquisition score against minimum distance to already-chosen points in the batch."
        )

    with st.expander("🔄 What is the Calibrated Transfer Prior?", expanded=False):
        st.markdown(
            "**Plain language:** A brand-new project has nothing to learn from for its first few "
            "batches, so early suggestions would otherwise be random guesses. This app ships with a "
            "built-in reference dataset of prior MOF syntheses. Turning this on lets the app borrow "
            "patterns from that data for your first `M` batches, then automatically hands control "
            "over to your own data as it accumulates.\n\n"
            "**Rigorously:** A reference GP is fit once on the built-in 96-point benchmark set. For "
            "each of your completed points, its reference prediction `mu_ref(x)` is linearly "
            "calibrated against your real `y = ln(1+q)` via ridge-regularized least squares "
            r"($y \approx a + b \cdot \mathrm{mu\_ref}(x)$), regularized toward `b=1` in proportion to "
            "`1/n_student` (with `n_student=0` giving the pure-reference `a=0, b=1`). With 3+ points, "
            "a second GP models the residuals between your results and the calibrated prediction. "
            "The prior is used only through batch `M`, where "
            r"$M = \lfloor \text{mean(iteration range)} \times \text{transfer fraction} \rfloor$ "
            "(or set manually) — after that the app switches to a purely student-data GP."
        )

    with st.expander("📈 Reading the Dashboard panels", expanded=False):
        st.markdown(
            """
| Panel | What it tells you |
|---|---|
| Best q so far | Running best q across completed experiments — should trend up. |
| q by completed experiment | Raw q per experiment, in run order — spot noisy batches. |
| Observed vs predicted q | Model fit quality; points near the diagonal = good fit. |
| Uncertainty landscape | Where the model is confident vs unsure among candidates. |
| How suggestions are proposed | Acquisition score vs predicted q; stars = your open suggestions. |
| Experiment map | `metal_amount` vs `reaction_temperature`, colored by q. |
            """
        )

    with st.expander("📄 CSV import/export format", expanded=False):
        st.markdown(
            "Uploaded CSVs must contain the five feature columns (`metal_amount`, `modulator`, "
            "`add_solvent`, `reaction_time`, `reaction_temperature`). If `status` is missing it "
            "defaults to `completed`; if `q` is missing but `intensity`/`fwhm` are present, `q` is "
            "computed for you. Exports always use the full schema (`record_id, round, batch_position, "
            "status, <features>, intensity, fwhm, q, predicted_q_mean, predicted_q_sd, "
            "acquisition_value, notes, created_at, updated_at`) — see `experiments.csv` in the "
            "repository for a worked example."
        )

    with st.expander("🛠️ Troubleshooting / FAQ", expanded=False):
        st.markdown(
            """
- **"Initialize BO Optimization" is greyed out.** It's disabled once you already have an active batch
  of suggestions in this session. Save results with *Update Model*, or upload a new CSV, to re-enable
  it.
- **Dashboard says "No completed experiments yet."** It only reflects *completed* rows. A freshly
  initialized batch is `suggested` until you enter results and click *Update Model & Save Batch
  Results*.
- **Predicted q looks huge compared to any q I've measured.** Most visible in the reference-only phase
  of the transfer prior (`n_student=0`), where the model is extrapolating purely from the built-in
  benchmark GP. It pulls toward realistic values once a few of your own completed results calibrate it.
- **The status message says "Initial sampling was used…" or "Exploratory sampling…".** Either you have
  under 3 completed current-project points with transfer prior off, or your q values so far are all
  identical — the app deliberately explores rather than exploits until there's a real trend to model.
- **I want to reset everything.** Sidebar → *Clear All Data* clears the experiment log and any active
  suggestions in this session (it does not affect anything already exported).
            """
        )

    st.caption(f"{APP_NAME} · v{APP_VERSION}")
