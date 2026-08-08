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
                    "record_id": idx + 1,
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
tab_dash, tab_recommend, tab_data, tab_wizard = st.tabs([
    "📊 Dashboard & Diagnostics",
    "🧪 Batch Recommendations",
    "📝 Experiment Entry & Data",
    "🧙 Planning Wizard",
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

    if has_session_prediction:
        st.caption("Initialize BO Optimization is disabled because this session already has active suggestions. Enter results below and click 'Update Model' to re-run.")

    if st.session_state.last_message:
        st.success(f"**Engine Status:** {st.session_state.last_message}")

    if st.session_state.last_suggestions is not None and len(st.session_state.last_suggestions) > 0:
        sug_df = st.session_state.last_suggestions
        st.markdown("#### Proposed Batch Conditions & Experimental Measurement Feedback")
        st.caption("Enter measured Intensity and FWHM (or q) directly into the spreadsheet table below:")

        display_cols = [f for f in FEATURES if f in sug_df.columns]
        extra_cols = [c for c in ["predicted_q_mean", "predicted_q_sd", "acquisition_value"] if c in sug_df.columns]
        feedback_cols = ["intensity", "fwhm", "q", "notes"]

        full_cols = display_cols + extra_cols + feedback_cols
        show_cols = [c for c in full_cols if c in sug_df.columns]

        edited_sug_df = st.data_editor(
            sug_df[show_cols],
            num_rows="fixed",
            use_container_width=True,
            column_config={
                "intensity": st.column_config.NumberColumn("Intensity (Measured)", min_value=0.0),
                "fwhm": st.column_config.NumberColumn("FWHM (Measured)", min_value=0.01),
                "q": st.column_config.NumberColumn("q Score", min_value=0.0),
                "notes": st.column_config.TextColumn("Notes"),
            },
        )

        col_update, col_dl = st.columns([2, 1])
        with col_update:
            if st.button("🔄 Update Model & Save Batch Results", type="primary", use_container_width=True):
                engine_tmp = BOEngine(st.session_state.config)
                batch_no = engine_tmp._next_batch_count(df_exp)
                new_rows = []

                for pos, (_idx, row) in enumerate(edited_sug_df.iterrows(), start=1):
                    rec_id = len(st.session_state.experiments) + pos
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
        edited_df = st.data_editor(
            st.session_state.experiments,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "status": st.column_config.SelectboxColumn("Status", options=["completed", "suggested", "cancelled"]),
                "intensity": st.column_config.NumberColumn("Intensity", min_value=0.0),
                "fwhm": st.column_config.NumberColumn("FWHM", min_value=0.01),
                "q": st.column_config.NumberColumn("q Score", min_value=0.0),
            },
        )
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
