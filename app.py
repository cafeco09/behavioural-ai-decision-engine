import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class Scenario:
    name: str
    autonomy_pressure: float
    overload: float


TIERS = ["Bronze", "Silver", "Gold"]
TIER_THRESHOLDS = {"Bronze": 0, "Silver": 500, "Gold": 1000}

ACTIONS = ["No_Action", "Small_Discount", "Discount", "Big_Offer"]

ACTION_EFFORT = {
    "No_Action": 0.0,
    "Small_Discount": 0.6,
    "Discount": 0.9,
    "Big_Offer": 1.2,
}

ACTION_VALUE = {
    "No_Action": 0.0,
    "Small_Discount": 0.8,
    "Discount": 1.0,
    "Big_Offer": 1.15,
}

ACTION_POINTS_GAIN = {
    "No_Action": 0,
    "Small_Discount": 120,
    "Discount": 170,
    "Big_Offer": 220,
}


def make_scenarios() -> List[Scenario]:
    return [
        Scenario(name="Threshold Sprint", autonomy_pressure=0.35, overload=0.25),
        Scenario(name="Offer Overload", autonomy_pressure=0.65, overload=1.00),
        Scenario(name="AI Suggestion vs Autonomy", autonomy_pressure=1.00, overload=0.40),
    ]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def tier_from_points(points: int) -> str:
    if points >= TIER_THRESHOLDS["Gold"]:
        return "Gold"
    if points >= TIER_THRESHOLDS["Silver"]:
        return "Silver"
    return "Bronze"


def sample_users(n: int, user_distribution: str, rng: np.random.Generator) -> pd.DataFrame:
    tiers = rng.choice(TIERS, size=n, p=[0.55, 0.33, 0.12])
    days = rng.integers(1, 31, size=n)

    if user_distribution == "Near threshold":
        points = []
        for t in tiers:
            if t == "Bronze":
                points.append(int(rng.integers(420, 510)))
            elif t == "Silver":
                points.append(int(rng.integers(920, 1010)))
            else:
                points.append(int(rng.integers(1050, 1300)))
        points = np.array(points, dtype=int)
        trust = rng.beta(4.0, 3.0, size=n)
    elif user_distribution == "Low trust":
        points = rng.integers(0, 1300, size=n)
        trust = rng.beta(1.3, 5.0, size=n)
    elif user_distribution == "High trust":
        points = rng.integers(0, 1300, size=n)
        trust = rng.beta(5.0, 1.5, size=n)
    else:
        points = rng.integers(0, 1300, size=n)
        trust = rng.beta(3.0, 3.0, size=n)

    autonomy = rng.beta(3.0, 3.0, size=n)

    df = pd.DataFrame(
        {
            "user_id": np.arange(n, dtype=int),
            "tier": [tier_from_points(int(p)) for p in points],
            "points": points.astype(int),
            "day": days.astype(int),
            "trust_ai": trust.astype(float),
            "autonomy_pref": autonomy.astype(float),
        }
    )
    return df


def advisor_recommendation(scenario: Scenario, tier: str, points: int) -> str:
    if scenario.name == "Offer Overload":
        return "Small_Discount"
    if scenario.name == "Threshold Sprint":
        if tier == "Bronze" and points < TIER_THRESHOLDS["Silver"]:
            return "Big_Offer" if (TIER_THRESHOLDS["Silver"] - points) <= 120 else "Discount"
        if tier == "Silver" and points < TIER_THRESHOLDS["Gold"]:
            return "Big_Offer" if (TIER_THRESHOLDS["Gold"] - points) <= 170 else "Discount"
        return "Small_Discount"
    return "Discount"


def fallback_choice(decision_maker: str, scenario: Scenario, tier: str, points: int, trust_ai: float, autonomy_pref: float, rng: np.random.Generator) -> str:
    near_silver = abs(TIER_THRESHOLDS["Silver"] - points) <= 120
    near_gold = abs(TIER_THRESHOLDS["Gold"] - points) <= 170
    near_threshold = near_silver or near_gold

    if decision_maker == "Rational":
        if scenario.overload >= 0.8:
            return "Small_Discount"
        if near_threshold:
            return "Discount"
        return "Small_Discount"

    if scenario.overload >= 0.8:
        if rng.random() < 0.55:
            return "No_Action"
        return "Small_Discount"

    if near_threshold and rng.random() < 0.7:
        return "Discount"

    if autonomy_pref > 0.7 and rng.random() < 0.35:
        return "No_Action"

    return "Small_Discount" if rng.random() < 0.7 else "Discount"


def act_probability(action: str, scenario: Scenario, trust_ai: float, autonomy_pref: float, ai_used: bool) -> float:
    effort = ACTION_EFFORT[action]
    base = 0.78
    base -= 0.22 * scenario.overload
    base -= 0.18 * effort
    if ai_used:
        base += 0.06 * trust_ai
        base -= 0.04 * autonomy_pref * scenario.autonomy_pressure
    return clamp(base, 0.05, 0.95)


def expected_utility(action: str, scenario: Scenario, tier: str, points: int, trust_ai: float, autonomy_pref: float, ai_used: bool) -> float:
    p_act = act_probability(action, scenario, trust_ai, autonomy_pref, ai_used)
    value = ACTION_VALUE[action]
    effort = ACTION_EFFORT[action]
    points_gain = ACTION_POINTS_GAIN[action]

    current_target = TIER_THRESHOLDS["Silver"] if tier == "Bronze" else TIER_THRESHOLDS["Gold"] if tier == "Silver" else TIER_THRESHOLDS["Gold"]
    dist = max(0, current_target - points)
    tier_bonus = 0.0
    if tier != "Gold":
        tier_bonus = 0.35 if dist <= points_gain else 0.10

    util = p_act * (value + tier_bonus) - 0.25 * effort
    if action == "No_Action":
        util -= 0.05
    return util


def choose_with_ai(
    scenario: Scenario,
    tier: str,
    points: int,
    trust_ai: float,
    autonomy_pref: float,
    reactance: float,
    fallback: str,
    recommended: str,
    rng: np.random.Generator,
) -> Tuple[str, bool]:
    if recommended is None:
        return fallback, False

    u_f = expected_utility(fallback, scenario, tier, points, trust_ai, autonomy_pref, ai_used=False)
    u_r = expected_utility(recommended, scenario, tier, points, trust_ai, autonomy_pref, ai_used=True)

    autonomy_cost = reactance * 0.25 * scenario.autonomy_pressure * (0.25 + autonomy_pref) * (1.0 - trust_ai)

    u_r_adj = u_r - autonomy_cost

    switch_margin = 0.12 + 0.08 * scenario.overload

    if trust_ai < 0.12 and scenario.overload >= 0.8 and rng.random() < 0.65:
        return fallback, False

    if (u_r_adj - u_f) >= switch_margin:
        return recommended, True
    return fallback, False


def simulate_once(
    decision_maker: str,
    scenario: Scenario,
    tier: str,
    points: int,
    trust_ai: float,
    autonomy_pref: float,
    reactance: float,
    ai_enabled: bool,
    seed_int: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed_int)

    rec = advisor_recommendation(scenario, tier, points) if ai_enabled else None
    fallback = fallback_choice(decision_maker, scenario, tier, points, trust_ai, autonomy_pref, rng)

    ai_aligned = False
    ai_influenced = False
    if ai_enabled and rec is not None:
        final, influenced = choose_with_ai(scenario, tier, points, trust_ai, autonomy_pref, reactance, fallback, rec, rng)
        ai_influenced = bool(influenced and (final == rec) and (fallback != rec))
        ai_aligned = bool(final == rec)
        ai_used = ai_aligned
    else:
        final = fallback
        ai_used = False

    p_act = act_probability(final, scenario, trust_ai, autonomy_pref, ai_used=ai_used)
    acted = bool(rng.random() < p_act) if final != "No_Action" else False

    gain = ACTION_POINTS_GAIN[final] if acted else 0
    new_points = int(points + gain)
    new_tier = tier_from_points(new_points)

    tier_up = (new_tier != tier) and (TIERS.index(new_tier) > TIERS.index(tier))
    tier_down = (new_tier != tier) and (TIERS.index(new_tier) < TIERS.index(tier))

    current_target = TIER_THRESHOLDS["Silver"] if tier == "Bronze" else TIER_THRESHOLDS["Gold"] if tier == "Silver" else TIER_THRESHOLDS["Gold"]
    dist = max(0, current_target - points)

    regret_action = 0.0
    regret_inaction = 0.0

    if acted:
        effort = ACTION_EFFORT[final]
        overshoot_penalty = 0.0
        if scenario.overload >= 0.8 and final in ["Discount", "Big_Offer"]:
            overshoot_penalty = 0.35
        regret_action = clamp(0.9 * effort + overshoot_penalty - 0.35 * (1.0 if tier_up else 0.0), 0.0, 5.0)
    else:
        near = 1.0 if dist <= 170 else 0.4 if dist <= 350 else 0.15
        regret_inaction = clamp(4.5 * near + 0.35 * scenario.overload, 0.0, 5.0)

    regret_total = float(regret_action + regret_inaction)

    return {
        "advisor_action": rec if rec is not None else "None",
        "fallback_action": fallback,
        "final_action": final,
        "ai_aligned": bool(ai_aligned),
        "ai_influenced": bool(ai_influenced),
        "acted": bool(acted),
        "new_tier": new_tier,
        "tier_up": bool(tier_up),
        "tier_down": bool(tier_down),
        "effort": float(ACTION_EFFORT[final]) if acted else 0.0,
        "regret_action": float(regret_action),
        "regret_inaction": float(regret_inaction),
        "regret_total": float(regret_total),
    }


def run_simulation(
    users_df: pd.DataFrame,
    scenarios: List[Scenario],
    decision_maker: str,
    ai_enabled: bool,
    reactance: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cf_rows = []

    for _, u in users_df.iterrows():
        uid = int(u["user_id"])
        tier = str(u["tier"])
        points = int(u["points"])
        day = int(u["day"])
        trust_ai = float(u["trust_ai"])
        autonomy_pref = float(u["autonomy_pref"])

        for s in scenarios:
            base_key = (seed * 10_000_000) + (uid * 10_000) + (hash(s.name) % 10_000)

            out_off = simulate_once(
                decision_maker=decision_maker,
                scenario=s,
                tier=tier,
                points=points,
                trust_ai=trust_ai,
                autonomy_pref=autonomy_pref,
                reactance=reactance,
                ai_enabled=False,
                seed_int=base_key + 13,
            )

            out_on = simulate_once(
                decision_maker=decision_maker,
                scenario=s,
                tier=tier,
                points=points,
                trust_ai=trust_ai,
                autonomy_pref=autonomy_pref,
                reactance=reactance,
                ai_enabled=ai_enabled,
                seed_int=base_key + 17,
            )

            row = {
                "scenario": s.name,
                "tier": tier,
                "points": points,
                "day": day,
                "trust_ai": trust_ai,
                "advisor_action": out_on["advisor_action"],
                "fallback_action": out_on["fallback_action"],
                "final_action": out_on["final_action"],
                "ai_aligned": out_on["ai_aligned"],
                "ai_influenced": out_on["ai_influenced"],
                "acted": out_on["acted"],
                "new_tier": out_on["new_tier"],
                "tier_up": out_on["tier_up"],
                "tier_down": out_on["tier_down"],
                "effort": out_on["effort"],
                "regret_action": out_on["regret_action"],
                "regret_inaction": out_on["regret_inaction"],
                "regret_total": out_on["regret_total"],
            }
            rows.append(row)

            cf = {
                "scenario": s.name,
                "tier": tier,
                "trust_ai": trust_ai,
                "delta_completion": int(out_on["acted"]) - int(out_off["acted"]),
                "delta_tier_up": int(out_on["tier_up"]) - int(out_off["tier_up"]),
                "delta_regret_action": float(out_on["regret_action"]) - float(out_off["regret_action"]),
                "delta_regret_inaction": float(out_on["regret_inaction"]) - float(out_off["regret_inaction"]),
            }
            cf["delta_regret_total"] = float(cf["delta_regret_action"] + cf["delta_regret_inaction"])
            cf_rows.append(cf)

    df = pd.DataFrame(rows)
    df_cf = pd.DataFrame(cf_rows)
    return df, df_cf


def main():
    st.set_page_config(page_title="Behavioural AI Decision Explorer", layout="wide")

    st.title("Behavioural AI Decision Explorer - Retail Loyalty")

    scenarios = make_scenarios()

    with st.sidebar:
        st.header("Controls")

        decision_maker = st.selectbox(
            "Decision-maker",
            ["Behavioural", "Rational"],
            index=0,
            key="decision_maker_select",
        )

        user_distribution = st.selectbox(
            "User distribution",
            ["Mixed", "Near threshold", "Low trust", "High trust"],
            index=1,
            key="user_distribution_select",
        )

        ai_enabled = st.checkbox(
            "AI advisor enabled",
            value=True,
            key="ai_enabled_checkbox",
        )

        reactance = st.slider(
            "Reactance (reject AI under overload)",
            0.0,
            1.5,
            1.2,
            0.05,
            key="reactance_slider",
            disabled=not ai_enabled,
        )

        n_users = st.slider(
            "Simulated users",
            200,
            6000,
            3200,
            100,
            key="n_users_slider",
        )

        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
            key="seed_number",
        )

        st.caption("This is a behavioural simulation and counterfactual lab, not a real retail prediction model.")

    rng = np.random.default_rng(int(seed))
    users_df = sample_users(int(n_users), str(user_distribution), rng)

    df, df_cf = run_simulation(
        users_df=users_df,
        scenarios=scenarios,
        decision_maker=str(decision_maker),
        ai_enabled=bool(ai_enabled),
        reactance=float(reactance),
        seed=int(seed),
    )

    st.subheader("Key Outcomes")
    c1, c2, c3, c4, c5 = st.columns(5)

    completion_rate = float(df["acted"].mean()) if len(df) else 0.0
    tier_up_rate = float(df["tier_up"].mean()) if len(df) else 0.0
    influence_rate = float(df["ai_influenced"].mean()) if (len(df) and ai_enabled) else 0.0
    regret_total = float(df["regret_total"].mean()) if len(df) else 0.0
    inaction_share = 1.0 - completion_rate

    c1.metric("Completion rate", f"{completion_rate:.2%}")
    c2.metric("Tier-up rate", f"{tier_up_rate:.2%}")
    c3.metric("AI influence rate", f"{influence_rate:.2%}" if ai_enabled else " - ")
    c4.metric("Mean regret (total)", f"{regret_total:.3f}")
    c5.metric("Inaction share", f"{inaction_share:.2%}")

    st.markdown("---")
    st.subheader("Counterfactual impact (same users, AI ON minus AI OFF)")

    cf_c1, cf_c2, cf_c3 = st.columns(3)
    d_comp = float(df_cf["delta_completion"].mean()) if len(df_cf) else 0.0
    d_tier = float(df_cf["delta_tier_up"].mean()) if len(df_cf) else 0.0
    d_reg = float(df_cf["delta_regret_total"].mean()) if len(df_cf) else 0.0

    cf_c1.metric("Delta completion", f"{d_comp:+.3f}")
    cf_c2.metric("Delta tier-up", f"{d_tier:+.3f}")
    cf_c3.metric("Delta regret (total)", f"{d_reg:+.3f}")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("Completion Rate by Scenario")
        st.bar_chart(df.groupby("scenario")["acted"].mean())

        st.subheader("Tier-up Rate by Scenario")
        st.bar_chart(df.groupby("scenario")["tier_up"].mean())

    with right:
        if ai_enabled:
            st.subheader("AI Influence Rate by Scenario")
            st.bar_chart(df.groupby("scenario")["ai_influenced"].mean())

        st.subheader("Regret by Scenario (total; lower is better)")
        st.bar_chart(df.groupby("scenario")["regret_total"].mean())

    st.markdown("---")
    st.subheader("Sample decisions (AI ON view if enabled)")

    show_cols = [
        "scenario",
        "tier",
        "points",
        "day",
        "trust_ai",
        "advisor_action",
        "fallback_action",
        "final_action",
        "ai_influenced",
        "acted",
        "new_tier",
        "effort",
        "regret_action",
        "regret_inaction",
        "regret_total",
    ]

    sample_df = df[show_cols].sample(min(30, len(df)), random_state=int(seed)).reset_index(drop=True)
    st.dataframe(sample_df)

    st.markdown("---")
    st.subheader("Sample counterfactual deltas (AI ON minus AI OFF)")

    cf_cols = [
        "scenario",
        "tier",
        "trust_ai",
        "delta_completion",
        "delta_tier_up",
        "delta_regret_action",
        "delta_regret_inaction",
        "delta_regret_total",
    ]
    sample_cf = df_cf[cf_cols].sample(min(20, len(df_cf)), random_state=int(seed)).reset_index(drop=True)
    st.dataframe(sample_cf)

    st.caption("Interpretation hint: positive delta completion with positive delta regret suggests AI increases action but not always quality.")


if __name__ == "__main__":
    main()

