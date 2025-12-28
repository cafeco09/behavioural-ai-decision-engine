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

# app.py
# Behavioural AI Decision Engine (Streamlit, older-version friendly)
# Companion code for: "Mitigating Psychological Reactance in AI-Assisted Decision-Making"
#
# Notes:
# - No st.divider()
# - No st.button(type=...)
# - No st.dataframe(use_container_width=...)
# - Adds "Paper mode" to lock the configuration used for paper figures
# - Saves runs to runs.csv
# - Exports arXiv-ready plots to outputs/ with stable filenames + ZIP bundle

# app.py
# Behavioural AI Decision Engine (Streamlit, older-version friendly)
# Companion code for: "Mitigating Psychological Reactance in AI-Assisted Decision-Making"
#
# Key additions in this version:
# 1) "Paper mode" locks the exact configuration used to generate paper figures.
# 2) Export arXiv plots produces stable filenames that match your LaTeX includes.
# 3) Export also writes outputs/paper_config.json so anyone can reproduce your plots exactly.
# 4) Runs can be logged to runs.csv.
# app.py
# Behavioural AI Decision Engine (Streamlit, older-version friendly)
# Companion code for: "Mitigating Psychological Reactance in AI-Assisted Decision-Making"
#
# What this script guarantees:
# - Paper mode locks the export configuration used for arXiv figures.
# - Export generates stable figure filenames that match your LaTeX includes:
#     fig_consideration_high/low(.pdf/.png)
#     fig_follow_disagree_high/low(.pdf/.png)
#     fig_delta_completion_high/low(.pdf/.png)
#     fig_delta_regret_high(.pdf/.png)
# - Export writes outputs/paper_config.json for exact reproduction.
# - Optional error bars (SE across seeds) can be included or omitted.
# - Compatible with older Streamlit versions (no st.divider, no button(type=...), etc.)

# app.py
# Behavioural AI Decision Engine — journal-robust exports
# Compatible with older Streamlit:
# - No st.divider()
# - No st.button(type=...)
# - No st.dataframe(use_container_width=...)
# Exports:
# - sweep_runs_*.csv (per-seed audit trail)
# - table_summary_*.csv (mean + SE)
# - table_summary_*.tex (booktabs LaTeX)
# - figures (.png + .pdf)
# - zipped bundle for arXiv

import os
import io
import math
import zipfile
from datetime import datetime, timezone
from typing import Dict, Tuple, List

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


def supports_download_button() -> bool:
    return hasattr(st, "download_button")


def download_bytes(label: str, data: bytes, filename: str, mime: str):
    if supports_download_button():
        st.download_button(label, data=data, file_name=filename, mime=mime)
    else:
        st.info("Your Streamlit version doesn't support download buttons. Use the saved file paths shown above.")


def safe_columns(n: int):
    # Older Streamlit should still have st.columns, but add fallback.
    if hasattr(st, "columns"):
        try:
            return st.columns(n)
        except Exception:
            pass
    return [st] * n


def safe_metric(label: str, value: str, delta: str = None):
    if hasattr(st, "metric"):
        try:
            st.metric(label, value, delta)
            return
        except Exception:
            pass
    # Fallback
    st.write(f"**{label}:** {value}" + (f" ({delta})" if delta else ""))


def sigmoid(x: float) -> float:
    # numerically stable
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# -----------------------------
# Paths
# -----------------------------
def project_paths() -> Tuple[str, str, str]:
    base = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    outputs = os.path.join(base, "outputs")
    os.makedirs(outputs, exist_ok=True)
    runs_csv = os.path.join(base, "runs.csv")
    return base, outputs, runs_csv


# -----------------------------
# Simulation model (lightweight but coherent)
# -----------------------------
ACTIONS = ("defer", "act", "act_plus")


def sample_user_state(rng: np.random.Generator, distribution: str) -> Dict[str, float]:
    # u = urgency, c = effort cost, a = ambiguity
    distribution = (distribution or "Near threshold").strip()

    if distribution == "Near threshold":
        u = float(np.clip(rng.normal(0.55, 0.12), 0, 1))
        c = float(np.clip(rng.normal(0.45, 0.12), 0, 1))
        a = float(np.clip(rng.normal(0.50, 0.15), 0, 1))
    elif distribution == "Below threshold":
        u = float(np.clip(rng.normal(0.35, 0.12), 0, 1))
        c = float(np.clip(rng.normal(0.55, 0.12), 0, 1))
        a = float(np.clip(rng.normal(0.55, 0.15), 0, 1))
    elif distribution == "Above threshold":
        u = float(np.clip(rng.normal(0.70, 0.12), 0, 1))
        c = float(np.clip(rng.normal(0.35, 0.12), 0, 1))
        a = float(np.clip(rng.normal(0.45, 0.15), 0, 1))
    else:  # Mixed
        u = float(rng.uniform(0, 1))
        c = float(rng.uniform(0, 1))
        a = float(rng.uniform(0, 1))

    return {"urgency": u, "effort_cost": c, "ambiguity": a}


def baseline_choice(state: Dict[str, float], decision_maker: str, rng: np.random.Generator) -> str:
    u, c, a = state["urgency"], state["effort_cost"], state["ambiguity"]
    decision_maker = (decision_maker or "Behavioural").strip()

    if decision_maker == "Rational":
        s_act = 1.25 * u - 0.95 * c - 0.35 * a
        s_plus = 1.05 * u - 1.10 * c - 0.55 * a
    else:  # Behavioural
        s_act = 1.10 * u - 1.20 * c - 0.80 * a - 0.15
        s_plus = 0.90 * u - 1.40 * c - 0.95 * a - 0.25

    eps = float(rng.normal(0.0, 0.25))
    s_act += eps
    s_plus += 0.6 * eps

    if s_plus > 0.35:
        return "act_plus"
    if s_act > 0.10:
        return "act"
    return "defer"


def ai_recommendation(state: Dict[str, float]) -> str:
    u, c, a = state["urgency"], state["effort_cost"], state["ambiguity"]
    # Simple policy that can disagree with baseline in meaningful regions
    if u > 0.68 and c < 0.55:
        return "act_plus" if a > 0.55 else "act"
    if u > 0.52 and c < 0.62:
        return "act"
    return "defer"


def compute_outcomes(state: Dict[str, float], action: str) -> Tuple[float, float]:
    # completion probability + regret proxy
    u, c, a = state["urgency"], state["effort_cost"], state["ambiguity"]

    if action == "defer":
        p_complete = max(0.0, 0.20 + 0.35 * u - 0.40 * a - 0.20 * c)
        regret = 1.20 * u + 0.40 * a
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
    fallback: str,
    rec: str,
    reactance: float,
    trust_ai: float,
    explainability: float,
    ai_strength: float,
    choice_preserving: bool,
    rng: np.random.Generator,
) -> Tuple[str, bool, bool]:
    """
    Two-stage response:
      - consider advice
      - if disagree, follow (switch)
    Returns:
      final_action, considered, followed_when_disagree
    """
    disagree = (fallback != rec)

    # explainability + choice-preserving reduce effective reactance
    react_mult = (1.0 - 0.55 * float(np.clip(explainability, 0.0, 1.0)))
    if choice_preserving:
        react_mult *= 0.75
    eff_react = float(np.clip(reactance * react_mult, 0.0, 2.0))

    # Consideration
    p_consider = sigmoid(-0.25 + 1.35 * trust_ai + 1.15 * explainability + 0.80 * ai_strength - 1.10 * eff_react)
    considered = bool(rng.random() < p_consider)

    if not disagree:
        return fallback, considered, False

    # Follow given disagreement (higher if considered)
    base = -0.60 + 1.10 * trust_ai + 0.95 * explainability + 1.00 * ai_strength - 1.35 * eff_react
    if considered:
        base += 0.35
    p_follow = sigmoid(base)
    follow = bool(rng.random() < p_follow)

    final = rec if follow else fallback
    return final, considered, follow


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
) -> pd.DataFrame:
    rows = []
    base_seed = int(seed)

    for i in range(int(n_users)):
        # deterministic per-user RNG fork
        user_seed = int((base_seed * 10_000_019 + i * 1_000_003) % 2_147_483_647)
        rng = np.random.default_rng(user_seed)

        stt = sample_user_state(rng, user_distribution)
        fallback = baseline_choice(stt, decision_maker, rng)
        rec = ai_recommendation(stt)

        if ai_enabled:
            final, considered, follow = choose_with_ai(
                fallback=fallback,
                rec=rec,
                reactance=float(reactance),
                trust_ai=float(trust_ai),
                explainability=float(explainability),
                ai_strength=float(ai_strength),
                choice_preserving=bool(choice_preserving),
                rng=rng,
            )
        else:
            final, considered, follow = fallback, False, False

        p_complete, regret = compute_outcomes(stt, final)
        completed = bool(rng.random() < p_complete)

        disagree = (fallback != rec)
        ai_influenced = bool(ai_enabled and disagree and follow and (final == rec))

        rows.append(
            {
                "user_id": i,
                "urgency": stt["urgency"],
                "effort_cost": stt["effort_cost"],
                "ambiguity": stt["ambiguity"],
                "fallback": fallback,
                "rec": rec,
                "final": final,
                "completed": completed,
                "regret": float(regret),
                "considered": bool(considered),
                "disagree": bool(disagree),
                "follow_given_disagree": bool(disagree and (final == rec)),
                "ai_influenced": bool(ai_influenced),
            }
        )

    return pd.DataFrame(rows)


def summarise_condition(df: pd.DataFrame, ai_enabled: bool) -> Dict[str, float]:
    completion_rate = float(df["completed"].mean())
    mean_regret = float(df["regret"].mean())

    if not ai_enabled:
        return {
            "completion_rate": completion_rate,
            "mean_regret": mean_regret,
            "consideration_rate": 0.0,
            "disagree_rate": 0.0,
            "follow_given_disagree": float("nan"),
            "ai_influence_rate": 0.0,
            "strict_influence_rate": float("nan"),
        }

    consideration_rate = float(df["considered"].mean())
    disagree_rate = float(df["disagree"].mean())

    if df["disagree"].any():
        follow_given_disagree = float(df.loc[df["disagree"], "follow_given_disagree"].mean())
    else:
        follow_given_disagree = float("nan")

    ai_influence_rate = float(df["ai_influenced"].mean())

    # "strict influence": among those who considered, how often did they switch under disagreement?
    considered_mask = df["considered"].values.astype(bool)
    if considered_mask.any():
        strict_influence_rate = float((df["ai_influenced"].values.astype(bool) & considered_mask).mean() / considered_mask.mean())
    else:
        strict_influence_rate = float("nan")

    return {
        "completion_rate": completion_rate,
        "mean_regret": mean_regret,
        "consideration_rate": consideration_rate,
        "disagree_rate": disagree_rate,
        "follow_given_disagree": follow_given_disagree,
        "ai_influence_rate": ai_influence_rate,
        "strict_influence_rate": strict_influence_rate,
    }


def simulate_run(params: Dict) -> Dict:
    # Pair AI OFF vs AI ON using same population + seed for baseline, and seed+999 for AI ON to reduce shared noise
    p_off = dict(params)
    p_on = dict(params)
    p_on["seed"] = int(params["seed"]) + 999

    off_df = simulate_population(ai_enabled=False, **p_off)
    on_df = simulate_population(ai_enabled=True, **p_on)

    off = summarise_condition(off_df, ai_enabled=False)
    on = summarise_condition(on_df, ai_enabled=True)

    return {
        "off": off,
        "on": on,
        "delta_completion": float(on["completion_rate"] - off["completion_rate"]),
        "delta_regret": float(on["mean_regret"] - off["mean_regret"]),
        "on_df": on_df,
        "off_df": off_df,
    }


# -----------------------------
# Runs persistence
# -----------------------------
RUN_COLS = [
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
    "mean_regret_on",
    "consideration_rate_on",
    "disagree_rate_on",
    "follow_given_disagree_on",
    "ai_influence_rate_on",
    "strict_influence_rate_on",
    "completion_rate_off",
    "mean_regret_off",
    "delta_completion",
    "delta_regret",
]


def load_runs(runs_csv: str) -> pd.DataFrame:
    if not os.path.exists(runs_csv):
        return pd.DataFrame(columns=RUN_COLS)
    df = pd.read_csv(runs_csv)
    for c in RUN_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[RUN_COLS]


def append_run(runs_csv: str, record: Dict):
    df = load_runs(runs_csv)
    row = {c: record.get(c, np.nan) for c in RUN_COLS}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(runs_csv, index=False)


# -----------------------------
# Journal-robust sweep exports (per-seed audit trail + summary)
# -----------------------------
SWEEP_RUN_COLS = [
    "reactance",
    "explainability",
    "seed",
    "decision_maker",
    "user_distribution",
    "trust_ai",
    "ai_strength",
    "choice_preserving",
    "n_users",
    "completion_rate_on",
    "mean_regret_on",
    "consideration_rate_on",
    "disagree_rate_on",
    "follow_given_disagree_on",
    "ai_influence_rate_on",
    "strict_influence_rate_on",
    "completion_rate_off",
    "mean_regret_off",
    "delta_completion",
    "delta_regret",
]


def export_sweep_runs(
    params_base: Dict,
    reactance_grid: List[float],
    seeds: List[int],
    outputs_dir: str,
    explainability: float,
    tag: str,
) -> Tuple[pd.DataFrame, str]:
    rows = []
    for lam in reactance_grid:
        for sd in seeds:
            p = dict(params_base)
            p["reactance"] = float(lam)
            p["explainability"] = float(explainability)
            p["seed"] = int(sd)

            m = simulate_run(p)

            row = {
                "reactance": float(lam),
                "explainability": float(explainability),
                "seed": int(sd),
                "decision_maker": p["decision_maker"],
                "user_distribution": p["user_distribution"],
                "trust_ai": float(p["trust_ai"]),
                "ai_strength": float(p["ai_strength"]),
                "choice_preserving": bool(p["choice_preserving"]),
                "n_users": int(p["n_users"]),
                "completion_rate_on": float(m["on"]["completion_rate"]),
                "mean_regret_on": float(m["on"]["mean_regret"]),
                "consideration_rate_on": float(m["on"]["consideration_rate"]),
                "disagree_rate_on": float(m["on"]["disagree_rate"]),
                "follow_given_disagree_on": float(m["on"]["follow_given_disagree"]) if not math.isnan(m["on"]["follow_given_disagree"]) else np.nan,
                "ai_influence_rate_on": float(m["on"]["ai_influence_rate"]),
                "strict_influence_rate_on": float(m["on"]["strict_influence_rate"]) if not math.isnan(m["on"]["strict_influence_rate"]) else np.nan,
                "completion_rate_off": float(m["off"]["completion_rate"]),
                "mean_regret_off": float(m["off"]["mean_regret"]),
                "delta_completion": float(m["delta_completion"]),
                "delta_regret": float(m["delta_regret"]),
            }
            rows.append(row)

    df_runs = pd.DataFrame(rows, columns=SWEEP_RUN_COLS)
    path = os.path.join(outputs_dir, f"sweep_runs_{tag}_E{explainability:.1f}.csv")
    df_runs.to_csv(path, index=False)
    return df_runs, path


def summarise_from_sweep(df_runs: pd.DataFrame) -> pd.DataFrame:
    def mean_se(series: pd.Series) -> Tuple[float, float]:
        x = pd.to_numeric(series, errors="coerce")
        n = int(x.notna().sum())
        if n == 0:
            return float("nan"), float("nan")
        mu = float(x.mean())
        if n == 1:
            return mu, 0.0
        se = float(x.std(ddof=1) / math.sqrt(n))
        return mu, se

    metrics = [
        "ai_influence_rate_on",
        "strict_influence_rate_on",
        "consideration_rate_on",
        "follow_given_disagree_on",
        "delta_completion",
        "delta_regret",
    ]

    out_rows = []
    for (E, lam), g in df_runs.groupby(["explainability", "reactance"], as_index=False):
        row = {"explainability": float(E), "reactance": float(lam), "n": int(len(g))}
        for col in metrics:
            mu, se = mean_se(g[col])
            row[col] = mu
            row[col + "_se"] = se
        out_rows.append(row)

    return pd.DataFrame(out_rows).sort_values(["explainability", "reactance"]).reset_index(drop=True)


def write_latex_table(df: pd.DataFrame, out_path: str, caption: str, label: str):
    # Keep it journal-friendly: booktabs, rounded, no index.
    df2 = df.copy()
    # Nicely formatted percentages for key rates; keep deltas numeric
    pct_cols = ["ai_influence_rate_on", "strict_influence_rate_on", "consideration_rate_on", "follow_given_disagree_on", "delta_completion"]
    for c in pct_cols:
        if c in df2.columns:
            df2[c] = (df2[c] * 100.0).round(1)
    if "delta_regret" in df2.columns:
        df2["delta_regret"] = df2["delta_regret"].round(3)

    # Select a compact set of columns for the paper table
    keep = [
        "explainability",
        "reactance",
        "ai_influence_rate_on",
        "consideration_rate_on",
        "follow_given_disagree_on",
        "delta_completion",
        "delta_regret",
    ]
    keep = [c for c in keep if c in df2.columns]
    df_out = df2[keep].copy()
    df_out.rename(
        columns={
            "explainability": "E",
            "reactance": r"$\lambda$",
            "ai_influence_rate_on": "Influence (\\%)",
            "consideration_rate_on": "Consideration (\\%)",
            "follow_given_disagree_on": "Follow$|$Disagree (\\%)",
            "delta_completion": r"$\Delta$Completion (pp)",
            "delta_regret": r"$\Delta$Regret",
        },
        inplace=True,
    )

    # delta completion currently in percent units; already converted to %
    # Interpret as percentage points (pp) in caption.
    latex = df_out.to_latex(
        index=False,
        escape=False,
        longtable=False,
        caption=caption,
        label=label,
        bold_rows=False,
        column_format="l" * len(df_out.columns),
        float_format="%.3f",
    )

    # add booktabs if not present (pandas usually does)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)


# -----------------------------
# Plotting + bundling
# -----------------------------
def save_fig(fig, png_path: str, pdf_path: str):
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")


def fig_metric(df_sum: pd.DataFrame, E: float, y: str, title: str, ylabel: str):
    sub = df_sum[df_sum["explainability"] == E].sort_values("reactance")
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    ax.errorbar(
        sub["reactance"].values,
        sub[y].values,
        yerr=sub[y + "_se"].values,
        marker="o",
        linestyle="-",
        capsize=3,
    )
    ax.set_title(title)
    ax.set_xlabel(r"Reactance ($\lambda$)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    return fig


def fig_overlay(df_sum: pd.DataFrame, y: str, title: str, ylabel: str, E_high: float, E_low: float):
    a = df_sum[df_sum["explainability"] == E_high].sort_values("reactance")
    b = df_sum[df_sum["explainability"] == E_low].sort_values("reactance")
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    ax.errorbar(a["reactance"], a[y], yerr=a[y + "_se"], marker="o", linestyle="-", capsize=3, label=f"E={E_high:.1f}")
    ax.errorbar(b["reactance"], b[y], yerr=b[y + "_se"], marker="o", linestyle="-", capsize=3, label=f"E={E_low:.1f}")
    ax.set_title(title)
    ax.set_xlabel(r"Reactance ($\lambda$)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig


def make_zip(file_paths: List[str], zip_path: str):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fp in file_paths:
            if os.path.exists(fp):
                z.write(fp, arcname=os.path.basename(fp))


# -----------------------------
# Streamlit app
# -----------------------------
safe_set_page_config()
base_dir, outputs_dir, runs_csv = project_paths()

st.title("Behavioural AI Decision Engine")
st.caption("Journal-robust arXiv exports (per-seed audit trail + SE), compatible with older Streamlit.")

st.markdown(f"**Project folder:** `{base_dir}`")
st.markdown(f"**outputs/**: `{outputs_dir}`")
st.markdown(f"**runs.csv:** `{runs_csv}`")
hr()

page = st.radio("View", ["Dashboard", "Saved runs", "Export arXiv bundle"], index=0)

# Sidebar controls
st.sidebar.header("Controls")

decision_maker = st.sidebar.selectbox("Decision-maker", ["Behavioural", "Rational"], index=0)
user_distribution = st.sidebar.selectbox("User distribution", ["Near threshold", "Below threshold", "Above threshold", "Mixed"], index=0)

reactance = st.sidebar.slider("Reactance (single run)", 0.0, 1.5, 1.25, 0.05)
trust_ai = st.sidebar.slider("Trust", 0.0, 1.0, 0.55, 0.05)
ai_strength = st.sidebar.slider("AI strength", 0.0, 1.0, 0.65, 0.05)
choice_preserving = st.sidebar.checkbox("Choice-preserving framing", value=True)

n_users = int(st.sidebar.number_input("Users", min_value=200, max_value=20000, value=1200, step=100))
seed = int(st.sidebar.number_input("Seed (single run)", min_value=0, max_value=10_000_000, value=100, step=1))

# explainability used in single run UI
explainability_single = st.sidebar.slider("Explainability (single run)", 0.0, 1.0, 0.70, 0.05)

params_base = {
    "n_users": n_users,
    "seed": seed,
    "decision_maker": decision_maker,
    "user_distribution": user_distribution,
    "reactance": float(reactance),
    "trust_ai": float(trust_ai),
    "explainability": float(explainability_single),
    "ai_strength": float(ai_strength),
    "choice_preserving": bool(choice_preserving),
}

# -----------------------------
# Dashboard
# -----------------------------
if page == "Dashboard":
    st.subheader("Paired simulation (AI OFF vs AI ON)")

    c1, c2 = safe_columns(2)
    run_btn = c1.button("Run", key="run_btn")
    run_save_btn = c2.button("Run + Save", key="run_save_btn")

    if run_btn or run_save_btn:
        with st.spinner("Running simulation..."):
            metrics = simulate_run(params_base)

        off, on = metrics["off"], metrics["on"]

        k1, k2, k3, k4, k5 = safe_columns(5)
        k1.metric("Completion (AI ON)", f"{on['completion_rate']*100:.1f}%", f"{metrics['delta_completion']*100:+.1f} pp")
        k2.metric("Mean regret (AI ON)", f"{on['mean_regret']:.3f}", f"{metrics['delta_regret']:+.3f}")
        k3.metric("Consideration (AI ON)", f"{on['consideration_rate']*100:.1f}%")
        k4.metric("Follow | disagree", "n/a" if math.isnan(on["follow_given_disagree"]) else f"{on['follow_given_disagree']*100:.1f}%")
        k5.metric("AI influence (AI ON)", f"{on['ai_influence_rate']*100:.1f}%")

        hr()
        st.write("**Sanity checks**")
        st.write(f"- Disagreement rate (AI ON): **{on['disagree_rate']*100:.1f}%**")
        st.write(f"- Strict influence (among considered): " + ("n/a" if math.isnan(on["strict_influence_rate"]) else f"**{on['strict_influence_rate']*100:.1f}%**"))

        # quick plots
        fig = plt.figure(figsize=(6.2, 4.2))
        ax = fig.add_subplot(111)
        ax.bar(["AI OFF", "AI ON"], [off["completion_rate"], on["completion_rate"]])
        ax.set_title("Completion rate")
        ax.set_ylabel("Rate")
        st.pyplot(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(6.2, 4.2))
        ax = fig.add_subplot(111)
        ax.bar(["AI OFF", "AI ON"], [off["mean_regret"], on["mean_regret"]])
        ax.set_title("Mean regret")
        ax.set_ylabel("Regret")
        st.pyplot(fig)
        plt.close(fig)

        if run_save_btn:
            rec = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "decision_maker": decision_maker,
                "user_distribution": user_distribution,
                "reactance": float(reactance),
                "trust_ai": float(trust_ai),
                "explainability": float(explainability_single),
                "ai_strength": float(ai_strength),
                "choice_preserving": bool(choice_preserving),
                "n_users": int(n_users),
                "seed": int(seed),
                "completion_rate_on": float(on["completion_rate"]),
                "mean_regret_on": float(on["mean_regret"]),
                "consideration_rate_on": float(on["consideration_rate"]),
                "disagree_rate_on": float(on["disagree_rate"]),
                "follow_given_disagree_on": float(on["follow_given_disagree"]) if not math.isnan(on["follow_given_disagree"]) else np.nan,
                "ai_influence_rate_on": float(on["ai_influence_rate"]),
                "strict_influence_rate_on": float(on["strict_influence_rate"]) if not math.isnan(on["strict_influence_rate"]) else np.nan,
                "completion_rate_off": float(off["completion_rate"]),
                "mean_regret_off": float(off["mean_regret"]),
                "delta_completion": float(metrics["delta_completion"]),
                "delta_regret": float(metrics["delta_regret"]),
            }
            append_run(runs_csv, rec)
            st.success("Saved to runs.csv")

            # offer download runs.csv (if supported)
            try:
                data = open(runs_csv, "rb").read()
                download_bytes("Download runs.csv", data, "runs.csv", "text/csv")
            except Exception:
                st.warning("Could not read runs.csv for download. Use the path printed at the top.")

# -----------------------------
# Saved runs
# -----------------------------
elif page == "Saved runs":
    st.subheader("Saved runs")
    df = load_runs(runs_csv)

    if df.empty:
        st.info("No saved runs yet. Go to Dashboard → Run + Save.")
    else:
        st.write(f"Found **{len(df)}** runs.")
        st.table(df.tail(25))

        try:
            data = open(runs_csv, "rb").read()
            download_bytes("Download runs.csv", data, "runs.csv", "text/csv")
        except Exception:
            st.warning("Could not read runs.csv for download. Use the path printed at the top.")

# -----------------------------
# Export arXiv bundle (robust)
# -----------------------------
else:
    st.subheader("Export arXiv bundle (journal-robust)")

    st.write(
        "This export writes **per-seed rows** (`sweep_runs_*.csv`) and derives the summary table + error bars from them.\n\n"
        "Outputs saved to `outputs/`: CSV + LaTeX table + PNG/PDF figures + a ZIP bundle."
    )

    # Reactance grid + seeds
    default_grid = "0,0.25,0.5,0.75,1.0,1.2,1.25,1.5"
    grid_str = st.text_input("Reactance grid (comma-separated)", default_grid)
    seeds_str = st.text_input("Seeds (comma-separated)", "100,101,102,103,104")

    E_high = st.slider("High explainability (E_high)", 0.0, 1.0, 0.70, 0.05)
    E_low = st.slider("Low explainability (E_low)", 0.0, 1.0, 0.20, 0.05)

    export_btn = st.button("Generate arXiv bundle", key="export_btn")

    if export_btn:
        try:
            react_grid = [float(x.strip()) for x in grid_str.split(",") if x.strip() != ""]
            seeds = [int(x.strip()) for x in seeds_str.split(",") if x.strip() != ""]
        except Exception:
            st.error("Could not parse grid/seeds. Use comma-separated numbers.")
            st.stop()

        tag = f"{user_distribution.replace(' ', '_').lower()}_{decision_maker.lower()}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

        params_for_sweep = dict(params_base)
        # seed/reactance/explainability will be overridden per row
        # ensure single-run explainability slider doesn't leak into both conditions
        params_for_sweep["explainability"] = float(E_high)

        with st.spinner("Running sweep (this can take a bit)..."):
            df_high, path_high = export_sweep_runs(params_for_sweep, react_grid, seeds, outputs_dir, float(E_high), tag)
            df_low, path_low = export_sweep_runs(params_for_sweep, react_grid, seeds, outputs_dir, float(E_low), tag)

            df_runs = pd.concat([df_high, df_low], ignore_index=True)
            sweep_all_path = os.path.join(outputs_dir, f"sweep_runs_{tag}_ALL.csv")
            df_runs.to_csv(sweep_all_path, index=False)

            df_sum = summarise_from_sweep(df_runs)
            summary_path = os.path.join(outputs_dir, f"table_summary_{tag}.csv")
            df_sum.to_csv(summary_path, index=False)

            latex_path = os.path.join(outputs_dir, f"table_summary_{tag}.tex")
            write_latex_table(
                df_sum,
                latex_path,
                caption="Summary across reactance levels (mean; error bars computed as SE across seeds). Percentage columns are in \\%.",
                label=f"tab:summary_{tag}",
            )

            # Create figures (high/low panels + overlay for influence)
            files = [path_high, path_low, sweep_all_path, summary_path, latex_path]

            # Metric figures (two separate conditions)
            metrics = [
                ("consideration_rate_on", "Consideration rate vs reactance", "Consideration rate"),
                ("follow_given_disagree_on", "Follow rate | disagreement vs reactance", "Follow rate | disagreement"),
                ("ai_influence_rate_on", "AI influence rate vs reactance", "AI influence rate"),
                ("strict_influence_rate_on", "Strict influence (among considered) vs reactance", "Strict influence"),
                ("delta_completion", "Δ Completion (AI ON − AI OFF) vs reactance", "Δ Completion"),
                ("delta_regret", "Δ Regret (AI ON − AI OFF) vs reactance", "Δ Regret"),
            ]

            for col, title, ylabel in metrics:
                # High
                fig = fig_metric(df_sum, float(E_high), col, f"{title} (E={E_high:.1f})", ylabel)
                png = os.path.join(outputs_dir, f"fig_{col}_high_{tag}.png")
                pdf = os.path.join(outputs_dir, f"fig_{col}_high_{tag}.pdf")
                save_fig(fig, png, pdf)
                plt.close(fig)
                files.extend([png, pdf])

                # Low
                fig = fig_metric(df_sum, float(E_low), col, f"{title} (E={E_low:.1f})", ylabel)
                png = os.path.join(outputs_dir, f"fig_{col}_low_{tag}.png")
                pdf = os.path.join(outputs_dir, f"fig_{col}_low_{tag}.pdf")
                save_fig(fig, png, pdf)
                plt.close(fig)
                files.extend([png, pdf])

            # Overlay influence
            fig = fig_overlay(
                df_sum,
                "ai_influence_rate_on",
                "AI influence rate vs reactance (overlay)",
                "AI influence rate",
                float(E_high),
                float(E_low),
            )
            png = os.path.join(outputs_dir, f"fig_ai_influence_overlay_{tag}.png")
            pdf = os.path.join(outputs_dir, f"fig_ai_influence_overlay_{tag}.pdf")
            save_fig(fig, png, pdf)
            plt.close(fig)
            files.extend([png, pdf])

            # Zip bundle
            zip_path = os.path.join(outputs_dir, f"arxiv_bundle_{tag}.zip")
            make_zip(files, zip_path)

        st.success("Export complete.")
        st.markdown(f"**Saved sweep (high):** `{path_high}`")
        st.markdown(f"**Saved sweep (low):** `{path_low}`")
        st.markdown(f"**Saved sweep (ALL):** `{sweep_all_path}`")
        st.markdown(f"**Saved summary:** `{summary_path}`")
        st.markdown(f"**Saved LaTeX table:** `{latex_path}`")
        st.markdown(f"**ZIP bundle:** `{zip_path}`")

        hr()
        st.subheader("Preview: summary table")
        st.table(df_sum)

        # Download ZIP (if supported)
        try:
            data = open(zip_path, "rb").read()
            download_bytes("Download arXiv bundle (ZIP)", data, os.path.basename(zip_path), "application/zip")
        except Exception:
            st.warning("Could not read ZIP for download. Use the path printed above and open it locally.")

