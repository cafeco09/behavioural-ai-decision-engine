# app.py
# Behavioural AI Decision Explorer — Retail Loyalty
# Drop this file in your repo as app.py, then run: streamlit run app.py

# app.py — Behavioural AI Decision Engine (Streamlit, backwards-compatible)

# app.py
# Behavioural AI Decision Engine (Streamlit older-version friendly)
# - No st.divider()
# - No st.button(type=...)
# - No st.dataframe(use_container_width=...)
#
# Goal: simulate decision-making with/without AI, measure influence + outcomes,
# and provide a dashboard + runs logging.

# app.py
# Behavioural AI Decision Engine (older Streamlit compatible)
# Fixes:
# - Saving works (no "save button inside run" trap)
# - Exports arXiv plots to outputs/ and zips them for download
# - Shows absolute paths so you can verify where files are being written

import os
import math
import io
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# -----------------------------
# Compatibility helpers
# -----------------------------
def safe_set_page_config():
    try:
        st.set_page_config(page_title="Behavioural AI Decision Engine", layout="wide")
    except Exception:
        pass


def hr():
    st.markdown("---")


def has_download_button():
    return hasattr(st, "download_button")


def download_bytes_ui(label: str, data: bytes, filename: str, mime: str):
    """
    Download helper with fallback for old Streamlit.
    """
    if has_download_button():
        st.download_button(label, data=data, file_name=filename, mime=mime)
    else:
        # Old Streamlit fallback: write to disk and show path (reliable),
        # because large base64 data-URIs can break.
        st.info("Your Streamlit version doesn't support download buttons. "
                "Files are saved locally; use the path shown below.")


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# -----------------------------
# Simulation primitives
# -----------------------------
ACTIONS = ["defer", "act", "act_plus"]


def sample_user_state(rng: np.random.Generator, user_distribution: str):
    if user_distribution == "Near threshold":
        urgency = float(np.clip(rng.normal(0.55, 0.12), 0.0, 1.0))
        effort_cost = float(np.clip(rng.normal(0.45, 0.12), 0.0, 1.0))
        ambiguity = float(np.clip(rng.normal(0.50, 0.15), 0.0, 1.0))
    elif user_distribution == "Below threshold":
        urgency = float(np.clip(rng.normal(0.35, 0.12), 0.0, 1.0))
        effort_cost = float(np.clip(rng.normal(0.55, 0.12), 0.0, 1.0))
        ambiguity = float(np.clip(rng.normal(0.55, 0.15), 0.0, 1.0))
    elif user_distribution == "Above threshold":
        urgency = float(np.clip(rng.normal(0.70, 0.12), 0.0, 1.0))
        effort_cost = float(np.clip(rng.normal(0.35, 0.12), 0.0, 1.0))
        ambiguity = float(np.clip(rng.normal(0.45, 0.15), 0.0, 1.0))
    else:  # Mixed
        urgency = float(rng.uniform(0.0, 1.0))
        effort_cost = float(rng.uniform(0.0, 1.0))
        ambiguity = float(rng.uniform(0.0, 1.0))

    return {"urgency": urgency, "effort_cost": effort_cost, "ambiguity": ambiguity}


def baseline_choice(state: dict, decision_maker: str, rng: np.random.Generator):
    u, c, a = state["urgency"], state["effort_cost"], state["ambiguity"]

    if decision_maker == "Rational":
        score_act = 1.2 * u - 0.9 * c - 0.4 * a
        score_plus = 1.0 * u - 1.1 * c - 0.6 * a
    else:  # Behavioural
        score_act = 1.1 * u - 1.2 * c - 0.8 * a - 0.15
        score_plus = 0.9 * u - 1.4 * c - 0.9 * a - 0.25

    noise = float(rng.normal(0.0, 0.25))
    score_act += noise
    score_plus += 0.6 * noise

    if score_plus > 0.35:
        return "act_plus"
    if score_act > 0.10:
        return "act"
    return "defer"


def ai_recommendation(state: dict):
    u, a, c = state["urgency"], state["ambiguity"], state["effort_cost"]
    if u > 0.68 and c < 0.55:
        return "act_plus" if a > 0.55 else "act"
    if u > 0.52 and c < 0.62:
        return "act"
    return "defer"


def compute_outcomes(state: dict, action: str):
    u, c, a = state["urgency"], state["effort_cost"], state["ambiguity"]

    if action == "defer":
        p_complete = max(0.0, 0.20 + 0.35 * u - 0.40 * a - 0.20 * c)
        regret = 1.2 * u + 0.4 * a
    elif action == "act":
        p_complete = max(0.0, 0.35 + 0.55 * u - 0.25 * c - 0.15 * a)
        regret = 0.55 * (1.0 - u) + 0.25 * c
    else:  # act_plus
        p_complete = max(0.0, 0.40 + 0.62 * u - 0.35 * c - 0.10 * a)
        regret = 0.70 * (1.0 - u) + 0.35 * c + 0.15 * a

    p_complete = float(np.clip(p_complete, 0.0, 0.98))
    regret = float(max(0.0, regret))
    return p_complete, regret


def choose_with_ai(
    fallback_action: str,
    rec_action: str,
    reactance: float,
    trust_ai: float,
    explainability: float,
    ai_strength: float,
    choice_preserving: bool,
    rng: np.random.Generator,
):
    disagree = (fallback_action != rec_action)

    reactance_multiplier = (1.0 - 0.55 * float(np.clip(explainability, 0.0, 1.0)))
    if choice_preserving:
        reactance_multiplier *= 0.75
    eff_reactance = float(np.clip(reactance * reactance_multiplier, 0.0, 2.0))

    p_consider = sigmoid(-0.25 + 1.35 * trust_ai + 1.15 * explainability + 0.80 * ai_strength - 1.10 * eff_reactance)
    considered = bool(rng.random() < p_consider)

    if not disagree:
        return fallback_action, considered, False, False

    base = -0.60 + 1.10 * trust_ai + 0.95 * explainability + 1.00 * ai_strength - 1.35 * eff_reactance
    if considered:
        base += 0.35

    p_follow = sigmoid(base)
    follow = bool(rng.random() < p_follow)

    final_action = rec_action if follow else fallback_action
    influenced = follow
    return final_action, considered, influenced, follow


def simulate_population(
    *,
    n_users: int,
    seed: int,
    decision_maker: str,
    user_distribution: str,
    reactance: float,
    trust_ai: float,
    explainability: float,
    ai_strength: float,
    choice_preserving: bool,
    ai_enabled: bool,
):
    rng = np.random.default_rng(int(seed))
    rows = []

    for i in range(int(n_users)):
        user_seed = int((seed * 10_000_019 + i * 1_000_003) % 2_147_483_647)
        urng = np.random.default_rng(user_seed)

        state = sample_user_state(urng, user_distribution)
        fallback = baseline_choice(state, decision_maker, urng)
        rec = ai_recommendation(state)

        if ai_enabled:
            final, considered, influenced, _ = choose_with_ai(
                fallback_action=fallback,
                rec_action=rec,
                reactance=reactance,
                trust_ai=trust_ai,
                explainability=explainability,
                ai_strength=ai_strength,
                choice_preserving=choice_preserving,
                rng=urng,
            )
        else:
            final, considered, influenced = fallback, False, False

        p_complete, regret = compute_outcomes(state, final)
        completed = bool(urng.random() < p_complete)

        tier_up = bool(final in ("act", "act_plus") and completed)
        disagree = (fallback != rec)
        follow_given_disagree = bool(disagree and final == rec)
        ai_influenced = bool(ai_enabled and disagree and follow_given_disagree and influenced)

        rows.append(
            dict(
                user_id=i,
                urgency=state["urgency"],
                effort_cost=state["effort_cost"],
                ambiguity=state["ambiguity"],
                fallback=fallback,
                rec=rec,
                final=final,
                completed=completed,
                tier_up=tier_up,
                regret=regret,
                considered=considered,
                disagree=disagree,
                follow_given_disagree=follow_given_disagree,
                ai_influenced=ai_influenced,
            )
        )

    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, ai_enabled: bool):
    completion_rate = float(df["completed"].mean())
    tier_up_rate = float(df["tier_up"].mean())
    mean_regret = float(df["regret"].mean())
    ai_influence_rate = float(df["ai_influenced"].mean()) if ai_enabled else 0.0
    consider_rate = float(df["considered"].mean()) if ai_enabled else 0.0
    disagree_rate = float(df["disagree"].mean()) if ai_enabled else 0.0
    if df["disagree"].any():
        follow_given_disagree = float(df.loc[df["disagree"], "follow_given_disagree"].mean())
    else:
        follow_given_disagree = float("nan")

    return dict(
        completion_rate=completion_rate,
        tier_up_rate=tier_up_rate,
        mean_regret=mean_regret,
        ai_influence_rate=ai_influence_rate,
        consideration_rate=consider_rate,
        disagree_rate=disagree_rate,
        follow_given_disagree=follow_given_disagree,
    )


def simulate_run(params: dict):
    off_df = simulate_population(ai_enabled=False, **params)
    on_df = simulate_population(ai_enabled=True, **params)

    off = summarise(off_df, ai_enabled=False)
    on = summarise(on_df, ai_enabled=True)

    return dict(
        off=off,
        on=on,
        delta_completion=on["completion_rate"] - off["completion_rate"],
        delta_tier_up=on["tier_up_rate"] - off["tier_up_rate"],
        delta_regret=on["mean_regret"] - off["mean_regret"],
        on_df=on_df,
        off_df=off_df,
    )


# -----------------------------
# File paths + runs.csv
# -----------------------------
RUN_COLUMNS = [
    "timestamp_utc",
    "decision_maker",
    "user_distribution",
    "reactance",
    "trust_ai",
    "explainability",
    "ai_strength",
    "choice_preserving",
    "n_users",
    "seed",
    "completion_rate_on",
    "tier_up_rate_on",
    "ai_influence_rate_on",
    "mean_regret_on",
    "consideration_rate_on",
    "follow_given_disagree_on",
    "completion_rate_off",
    "tier_up_rate_off",
    "mean_regret_off",
    "delta_completion",
    "delta_tier_up",
    "delta_regret",
]


def project_paths():
    base = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    outputs = os.path.join(base, "outputs")
    os.makedirs(outputs, exist_ok=True)
    runs_csv = os.path.join(base, "runs.csv")
    return base, outputs, runs_csv


def load_runs(runs_csv: str) -> pd.DataFrame:
    if not os.path.exists(runs_csv):
        return pd.DataFrame(columns=RUN_COLUMNS)
    df = pd.read_csv(runs_csv)
    for c in RUN_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[RUN_COLUMNS]


def append_run(runs_csv: str, record: dict):
    df = load_runs(runs_csv)
    row = {c: record.get(c, np.nan) for c in RUN_COLUMNS}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(runs_csv, index=False)


# -----------------------------
# Plot export
# -----------------------------
def save_fig(fig, path_png: str, path_pdf: str = None, dpi: int = 300):
    # Increased DPI to 300 for high-quality print
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    if path_pdf:
        # PDFs are vector-based; perfect for arXiv/LaTeX
        fig.savefig(path_pdf, bbox_inches="tight")


# --- ACADEMIC PLOTTING REFINEMENT ---
import matplotlib as mpl

# Set academic defaults
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.constrained_layout.use": True
})

def fig_metric_vs_reactance(df_sum: pd.DataFrame, y: str, title: str, ylabel: str):
    """Generates a journal-quality plot with error bars."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    
    ax.errorbar(
        df_sum["reactance"], 
        df_sum[y], 
        yerr=df_sum[y + "_se"], 
        marker="s",          # Professional square markers
        markersize=5,
        linestyle="-", 
        linewidth=1.2,
        capsize=3,           # Caps on error bars
        color="#1f77b4",     # Academic blue
        label="Simulated Population"
    )
    
    # Use mathematical notation for Reactance (Lambda)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(r"Reactance Level ($\lambda$)")
    ax.set_ylabel(ylabel)
    
    # Clean up spines for a modern look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig


def run_reactance_sweep(params_base: dict, reactance_grid, replicates: int, outputs_dir: str, tag: str):
    rows = []
    for r in reactance_grid:
        vals = {
            "ai_influence_rate": [],
            "consideration_rate": [],
            "follow_given_disagree": [],
            "delta_completion": [],
            "delta_regret": [],
            "strict_influence_rate": [],  # here: follow|disagree (same signal but conditional)
        }
        for k in range(replicates):
            p = dict(params_base)
            p["reactance"] = float(r)
            p["seed"] = int(params_base["seed"] + 10_000 * k + int(100 * r))
            m = simulate_run(p)
            vals["ai_influence_rate"].append(m["on"]["ai_influence_rate"])
            vals["consideration_rate"].append(m["on"]["consideration_rate"])
            vals["follow_given_disagree"].append(m["on"]["follow_given_disagree"])
            vals["strict_influence_rate"].append(m["on"]["follow_given_disagree"])
            vals["delta_completion"].append(m["delta_completion"])
            vals["delta_regret"].append(m["delta_regret"])

        def mean_se(x):
            x = np.array(x, dtype=float)
            mu = float(np.nanmean(x))
            se = float(np.nanstd(x, ddof=1) / math.sqrt(max(1, np.sum(~np.isnan(x))))) if np.sum(~np.isnan(x)) > 1 else 0.0
            return mu, se

        row = {"reactance": float(r)}
        for key in vals:
            mu, se = mean_se(vals[key])
            row[key] = mu
            row[key + "_se"] = se
        rows.append(row)

    df_sum = pd.DataFrame(rows).sort_values("reactance").reset_index(drop=True)

    # Save the sweep table
    table_path = os.path.join(outputs_dir, f"table_summary_{tag}.csv")
    df_sum.to_csv(table_path, index=False)

    figs = []

    figs.append(("fig_ai_influence_" + tag,
                 fig_metric_vs_reactance(df_sum, "ai_influence_rate",
                                         "AI influence rate (AI ON) vs reactance",
                                         "AI influence rate (AI ON)")))

    figs.append(("fig_strict_influence_" + tag,
                 fig_metric_vs_reactance(df_sum, "strict_influence_rate",
                                         "Strict influence rate vs reactance",
                                         "Strict influence rate")))

    figs.append(("fig_consideration_" + tag,
                 fig_metric_vs_reactance(df_sum, "consideration_rate",
                                         "Consideration rate vs reactance",
                                         "Consideration rate")))

    figs.append(("fig_follow_given_disagree_" + tag,
                 fig_metric_vs_reactance(df_sum, "follow_given_disagree",
                                         "Follow rate | disagreement vs reactance",
                                         "Follow rate | disagreement")))

    figs.append(("fig_delta_completion_" + tag,
                 fig_metric_vs_reactance(df_sum, "delta_completion",
                                         "Δ completion (AI ON − AI OFF) vs reactance",
                                         "Δ completion")))

    figs.append(("fig_delta_regret_" + tag,
                 fig_metric_vs_reactance(df_sum, "delta_regret",
                                         "Δ mean regret (AI ON − AI OFF) vs reactance",
                                         "Δ mean regret")))

    saved_files = [table_path]
    for name, fig in figs:
        png = os.path.join(outputs_dir, f"{name}.png")
        pdf = os.path.join(outputs_dir, f"{name}.pdf")
        save_fig(fig, png, pdf)
        plt.close(fig)
        saved_files.extend([png, pdf])

    # Zip everything
    zip_path = os.path.join(outputs_dir, f"arxiv_plots_{tag}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fp in saved_files:
            z.write(fp, arcname=os.path.basename(fp))

    return df_sum, zip_path


# -----------------------------
# UI
# -----------------------------
safe_set_page_config()
base_dir, outputs_dir, runs_csv = project_paths()
runs_df = load_runs(runs_csv)

st.title("Behavioural AI Decision Engine")
st.caption("Simulates decision-making with an AI adviser. Older Streamlit compatible.")

# IMPORTANT: show absolute paths so you can verify where things are being written
st.markdown(f"**Project folder:** `{base_dir}`")
st.markdown(f"**outputs/**: `{outputs_dir}`")
st.markdown(f"**runs.csv:** `{runs_csv}`")

hr()

page = st.radio("View", ["Dashboard", "Saved runs", "Export arXiv plots"], index=0)

st.sidebar.header("Simulation controls")
decision_maker = st.sidebar.selectbox("Decision-maker model", ["Behavioural", "Rational"], index=0)
user_distribution = st.sidebar.selectbox("User distribution", ["Near threshold", "Below threshold", "Above threshold", "Mixed"], index=0)

reactance = st.sidebar.slider("Reactance", 0.0, 1.5, 1.25, 0.05)
trust_ai = st.sidebar.slider("Trust in AI", 0.0, 1.0, 0.55, 0.05)
explainability = st.sidebar.slider("Explainability / transparency", 0.0, 1.0, 0.70, 0.05)
ai_strength = st.sidebar.slider("AI persuasiveness (strength)", 0.0, 1.0, 0.65, 0.05)
choice_preserving = st.sidebar.checkbox("Choice-preserving framing (reduces reactance)", value=True)

n_users = int(st.sidebar.number_input("Users", min_value=200, max_value=20000, value=1200, step=100))
seed = int(st.sidebar.number_input("Seed", min_value=0, max_value=10_000_000, value=42, step=1))

params = dict(
    n_users=n_users,
    seed=seed,
    decision_maker=decision_maker,
    user_distribution=user_distribution,
    reactance=float(reactance),
    trust_ai=float(trust_ai),
    explainability=float(explainability),
    ai_strength=float(ai_strength),
    choice_preserving=bool(choice_preserving),
)

if page == "Dashboard":
    st.subheader("Run a paired simulation (AI OFF vs AI ON)")

    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button("Run simulation", key="run_btn")
    with col2:
        save_btn = st.button("Run + Save (logs run to runs.csv)", key="run_save_btn")

    if run_btn or save_btn:
        try:
            with st.spinner("Simulating..."):
                metrics = simulate_run(params)
        except Exception:
            # spinner isn't critical
            metrics = simulate_run(params)

        off, on = metrics["off"], metrics["on"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Completion (AI ON)", f"{on['completion_rate']*100:.1f}%", f"{metrics['delta_completion']*100:+.1f} pp")
        c2.metric("Tier-up (AI ON)", f"{on['tier_up_rate']*100:.1f}%", f"{metrics['delta_tier_up']*100:+.1f} pp")
        c3.metric("AI influence (AI ON)", f"{on['ai_influence_rate']*100:.1f}%")
        c4.metric("Mean regret (AI ON)", f"{on['mean_regret']:.3f}", f"{metrics['delta_regret']:+.3f}")
        fgd = on.get("follow_given_disagree", float("nan"))
        c5.metric("Follow | disagree", "n/a" if (isinstance(fgd, float) and math.isnan(fgd)) else f"{fgd*100:.1f}%")

        hr()
        st.write("**Sanity checks**")
        st.write(f"- Disagreement rate (AI ON): **{on['disagree_rate']*100:.1f}%**")
        st.write(f"- Consideration rate (AI ON): **{on['consideration_rate']*100:.1f}%**")

        hr()
        st.subheader("Quick plots")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(["AI OFF", "AI ON"], [off["completion_rate"], on["completion_rate"]])
        ax.set_title("Completion rate")
        st.pyplot(fig)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(["AI OFF", "AI ON"], [off["mean_regret"], on["mean_regret"]])
        ax.set_title("Mean regret")
        st.pyplot(fig)
        plt.close(fig)

        if save_btn:
            record = dict(
                timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                decision_maker=decision_maker,
                user_distribution=user_distribution,
                reactance=float(reactance),
                trust_ai=float(trust_ai),
                explainability=float(explainability),
                ai_strength=float(ai_strength),
                choice_preserving=bool(choice_preserving),
                n_users=int(n_users),
                seed=int(seed),
                completion_rate_on=float(on["completion_rate"]),
                tier_up_rate_on=float(on["tier_up_rate"]),
                ai_influence_rate_on=float(on["ai_influence_rate"]),
                mean_regret_on=float(on["mean_regret"]),
                consideration_rate_on=float(on["consideration_rate"]),
                follow_given_disagree_on=float(on.get("follow_given_disagree", np.nan)),
                completion_rate_off=float(off["completion_rate"]),
                tier_up_rate_off=float(off["tier_up_rate"]),
                mean_regret_off=float(off["mean_regret"]),
                delta_completion=float(metrics["delta_completion"]),
                delta_tier_up=float(metrics["delta_tier_up"]),
                delta_regret=float(metrics["delta_regret"]),
            )
            append_run(runs_csv, record)
            st.success("Saved run to runs.csv. Switch to 'Saved runs' to view it.")

            # also offer direct download of runs.csv
            try:
                runs_bytes = open(runs_csv, "rb").read()
                download_bytes_ui("Download runs.csv", runs_bytes, "runs.csv", "text/csv")
            except Exception:
                st.warning("Could not read runs.csv for download, but it should be saved on disk. Check the path shown above.")

    hr()
    st.caption("Note: saving works via 'Run + Save' because Streamlit reruns on button clicks; separating Run then Save can lose state.")

elif page == "Saved runs":
    st.subheader("Saved runs (runs.csv)")
    runs_df = load_runs(runs_csv)

    if runs_df.empty:
        st.info("No saved runs yet. Use 'Dashboard' → 'Run + Save'.")
    else:
        st.write(f"Found **{len(runs_df)}** runs.")
        st.table(runs_df.tail(25))

        # download runs.csv
        try:
            runs_bytes = open(runs_csv, "rb").read()
            download_bytes_ui("Download runs.csv", runs_bytes, "runs.csv", "text/csv")
        except Exception:
            st.warning("Could not read runs.csv for download. Use the file path shown at the top.")

elif page == "Export arXiv plots":
    st.subheader("Export arXiv-ready plots (reactance sweep)")

    st.write(
        "This will run a sweep over reactance values and export:\n"
        "- PNG + PDF figures\n"
        "- table_summary CSV\n"
        "- a ZIP bundle in outputs/"
    )

    default_grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.2, 1.25, 1.5]
    react_grid = st.text_input("Reactance grid (comma-separated)", ",".join(str(x) for x in default_grid))
    replicates = int(st.number_input("Replicates per reactance", min_value=1, max_value=10, value=5, step=1))

    export_btn = st.button("Generate + Export arXiv plots (ZIP)", key="export_arxiv_btn")

    if export_btn:
        try:
            grid = [float(x.strip()) for x in react_grid.split(",") if x.strip() != ""]
        except Exception:
            st.error("Could not parse the reactance grid. Use comma-separated numbers like: 0,0.25,0.5,1.0")
            st.stop()

        tag = f"{user_distribution.replace(' ', '_').lower()}_{decision_maker.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            with st.spinner("Running sweep..."):
                df_sum, zip_path = run_reactance_sweep(params, grid, replicates, outputs_dir, tag)
        except Exception:
            df_sum, zip_path = run_reactance_sweep(params, grid, replicates, outputs_dir, tag)

        st.success("Export complete.")
        st.markdown(f"**ZIP saved to:** `{zip_path}`")
        st.markdown(f"**Also saved in:** `{outputs_dir}`")
        st.table(df_sum)

        # Download ZIP if supported
        try:
            zip_bytes = open(zip_path, "rb").read()
            download_bytes_ui("Download arXiv plots ZIP", zip_bytes, os.path.basename(zip_path), "application/zip")
        except Exception:
            st.warning("Could not read ZIP for download. Use the file path shown above and open it locally.")

