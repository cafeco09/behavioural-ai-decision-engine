
"""
Behavioural AI Decision Explorer — Retail Loyalty (Streamlit)

Improvements implemented (before adding extra charts):
1) Probabilistic AI-follow via user-level AI trust (heterogeneous) + reactance + overload.
2) Separate regret into action regret vs inaction regret (and keep a total).
3) Tier-conditioned behaviour (different action propensity / reactance sensitivity by tier).
4) Overload increases noise (misperception + occasional “slip” to a random option under overload).
5) Within-user counterfactual: simulate AI OFF and AI ON for the same user+scenario, store deltas.
"""

import math
import random
from dataclasses import replace
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

from src.environment.scenarios import TierRules, Tier, UserState, make_scenarios
from src.agents.rational import RationalAgent
from src.agents.bounded import BoundedRationalAgent
from src.agents.behavioural import BehaviouralAgent
from src.agents.ai_response import BehaviouralAIResponse
from src.advisor.simple_advisor import SimpleAdvisor
from src.evaluation.metrics import simulate_outcome

# ---------------------------------------------------------------------
# Streamlit MUST be configured once, before any other Streamlit call.
# ---------------------------------------------------------------------
st.set_page_config(page_title="Behavioural AI Decision Explorer", layout="wide")


# -----------------------------
# Small utilities
# -----------------------------
def sigmoid(x: float) -> float:
    # numerically stable-ish for our range
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def copy_state(state: UserState) -> UserState:
    # Defensive copy in case downstream code ever mutates (shouldn’t, but safe).
    return UserState(
        tier=state.tier,
        day=state.day,
        points=state.points,
        last_window_points=state.last_window_points,
    )


def is_overload(scenario_name: str) -> bool:
    return "Overload" in scenario_name


def scenario_label(s: str) -> str:
    return s.replace("_", " ")


# -----------------------------
# Behavioural primitives (Fixes #1, #3, #4)
# -----------------------------
def sample_trust_ai(rng: random.Random, tier: Tier) -> float:
    """
    User-level AI trust (0..1). Tier-conditioned:
    - Bronze: slightly lower trust (newer users)
    - Silver: moderate
    - Gold: can be more sceptical / autonomy-driven
    """
    # Beta distribution via random.betavariate (built-in).
    if tier == Tier.BRONZE:
        return rng.betavariate(2.0, 3.2)
    if tier == Tier.SILVER:
        return rng.betavariate(2.5, 2.7)
    return rng.betavariate(2.0, 2.8)


def tier_behaviour_params(tier: Tier) -> Dict[str, float]:
    """
    Tier-conditioned behaviour knobs (Fix #3).
    """
    if tier == Tier.BRONZE:
        return {
            "base_act": 0.88,
            "effort_penalty": 0.28,
            "reactance_penalty": 0.18,
            "overload_penalty": 0.22,
            "urgency_boost": 0.22,
        }
    if tier == Tier.SILVER:
        return {
            "base_act": 0.85,
            "effort_penalty": 0.24,
            "reactance_penalty": 0.16,
            "overload_penalty": 0.26,
            "urgency_boost": 0.18,
        }
    # GOLD
    return {
        "base_act": 0.80,
        "effort_penalty": 0.22,
        "reactance_penalty": 0.20,
        "overload_penalty": 0.30,
        "urgency_boost": 0.12,
    }


def distance_to_next_tier(state: UserState, rules: TierRules) -> int:
    if state.tier == Tier.BRONZE:
        return max(0, rules.silver_threshold - state.points)
    if state.tier == Tier.SILVER:
        return max(0, rules.gold_threshold - state.points)
    return 999999  # no next tier beyond Gold in this toy setting


def decides_to_act(
    state: UserState,
    scenario_name: str,
    rules: TierRules,
    effort: float,
    reactance: float,
    rng: random.Random,
) -> bool:
    """
    Explicit procrastination / inaction (Fix #0 you already did),
    now tier-conditioned + urgency + distance-to-threshold (Fix #3).
    """
    p = tier_behaviour_params(state.tier)

    days_left = max(0, rules.window_days - state.day)
    dist = distance_to_next_tier(state, rules)

    # urgency increases action as window closes
    urgency = 1.0 - (days_left / max(1, rules.window_days))
    urgency_term = p["urgency_boost"] * urgency

    # being close to threshold increases motivation
    # (closeness ~1 when dist small)
    closeness = 1.0 - clamp(dist / 400.0, 0.0, 1.0)
    closeness_term = 0.18 * closeness

    p_act = (
        p["base_act"]
        + urgency_term
        + closeness_term
        - p["effort_penalty"] * float(effort)
        - p["reactance_penalty"] * float(reactance)
        - (p["overload_penalty"] if is_overload(scenario_name) else 0.0)
    )

    p_act = clamp(p_act, 0.05, 0.95)
    return rng.random() < p_act


def overload_noise_sigma(state: UserState, scenario_name: str) -> float:
    """
    Overload increases misperception noise (Fix #4).
    Also allow mild tier differences.
    """
    if not is_overload(scenario_name):
        return 0.05

    # Under overload, noise is materially higher; Gold users can still get “choice fatigue”.
    if state.tier == Tier.BRONZE:
        return 0.35
    if state.tier == Tier.SILVER:
        return 0.30
    return 0.33


def choose_with_noise(
    state: UserState,
    scenario,
    rules: TierRules,
    candidate_a,
    candidate_b,
    rational_eval: RationalAgent,
    rng: random.Random,
    sigma: float,
):
    """
    Compare two candidate offers by *perceived* utility (true utility + noise).
    (Fix #4: overload noise can flip preferences.)
    """
    u_a = rational_eval.utility(state, scenario, candidate_a, rules)
    u_b = rational_eval.utility(state, scenario, candidate_b, rules)

    perceived_a = u_a + rng.gauss(0.0, sigma)
    perceived_b = u_b + rng.gauss(0.0, sigma)

    return candidate_a if perceived_a >= perceived_b else candidate_b


def maybe_slip_to_random_offer(
    state: UserState,
    scenario,
    chosen_offer,
    rng: random.Random,
    sigma: float,
):
    """
    Under high overload/noise, occasionally the user ‘slips’ into a random option.
    This makes regret distributions less degenerate and more realistic (Fix #4).
    """
    offers = getattr(scenario, "offers", None)
    if not offers or len(offers) < 2:
        return chosen_offer  # nothing to slip to

    # Slip probability grows with sigma (overload) and is slightly higher for Bronze.
    base = 0.02
    if state.tier == Tier.BRONZE:
        base = 0.04

    slip_p = clamp(base + (sigma * 0.08), 0.0, 0.18)
    if rng.random() >= slip_p:
        return chosen_offer

    # Choose a different offer uniformly
    alternatives = [o for o in offers if getattr(o, "name", None) != getattr(chosen_offer, "name", None)]
    if not alternatives:
        return chosen_offer
    return rng.choice(alternatives)


# -----------------------------
# User state sampling (unchanged core, but returns trust too)
# -----------------------------
def sample_user_state(rng: random.Random, rules: TierRules, profile: str) -> UserState:
    tier = rng.choices(
        [Tier.BRONZE, Tier.SILVER, Tier.GOLD],
        weights=[0.45, 0.40, 0.15],
        k=1
    )[0]

    day = rng.randint(1, rules.window_days)

    def _clamp_int(x, lo, hi):
        return max(lo, min(hi, x))

    if tier == Tier.BRONZE:
        if profile == "Near threshold":
            points = rng.randint(max(0, rules.silver_threshold - 120), rules.silver_threshold - 1)
        elif profile == "Far from threshold":
            points = rng.randint(0, max(1, rules.silver_threshold - 250))
        else:
            points = rng.randint(0, rules.silver_threshold - 1)
        last_window_points = _clamp_int(points + rng.randint(-150, 120), 0, rules.silver_threshold - 1)

    elif tier == Tier.SILVER:
        if profile == "Near threshold":
            points = rng.randint(max(rules.silver_threshold, rules.gold_threshold - 140), rules.gold_threshold - 1)
        elif profile == "Far from threshold":
            points = rng.randint(
                rules.silver_threshold,
                max(rules.silver_threshold + 1, rules.gold_threshold - 300),
            )
        else:
            points = rng.randint(rules.silver_threshold, rules.gold_threshold - 1)
        last_window_points = _clamp_int(points + rng.randint(-220, 160), 0, rules.gold_threshold - 1)

    else:  # GOLD
        points = rng.randint(rules.gold_threshold, rules.gold_threshold + 400)
        if profile == "Near threshold":
            last_window_points = rng.randint(max(0, rules.gold_threshold - 250), rules.gold_threshold + 50)
        elif profile == "Far from threshold":
            last_window_points = rng.randint(rules.gold_threshold, rules.gold_threshold + 300)
        else:
            last_window_points = rng.randint(max(0, rules.gold_threshold - 350), rules.gold_threshold + 250)

    return UserState(
        tier=tier,
        day=day,
        points=points,
        last_window_points=last_window_points,
    )


# -----------------------------
# One trial runner (Fix #5 counterfactual within-user)
# -----------------------------
def run_one_condition(
    *,
    user_id: int,
    state: UserState,
    trust_ai: float,
    scenario,
    rules: TierRules,
    agent_type: str,
    ai_enabled: bool,
    reactance: float,
    rng_seed: int,
    advisor: SimpleAdvisor,
    behavioural_agent: BehaviouralAgent,
    bounded_agent: BoundedRationalAgent,
    rational_agent: RationalAgent,
    ai_response: BehaviouralAIResponse,
) -> Dict:
    """
    Run a single condition (AI on/off) for a user and scenario. Returns one row.
    """
    rng = random.Random(rng_seed)

    # Choose the decision-maker agent
    if agent_type == "Rational":
        agent = rational_agent
    elif agent_type == "Bounded-rational":
        agent = bounded_agent
    else:
        agent = behavioural_agent

    # Baseline offer (user-only) and AI recommendation
    fallback = agent.choose_offer(state, scenario, rules)
    rec = advisor.recommend(state, scenario, rules)

    sigma = overload_noise_sigma(state, scenario.name)

    # Decide final offer
    followed_ai = False
    final = fallback

    if ai_enabled:
        # Probabilistic following: trust vs reactance vs overload (Fix #1)
        overload_pen = 0.20 if is_overload(scenario.name) else 0.0
        tier_pen = 0.06 if state.tier == Tier.GOLD else 0.0  # gold can resist “help”
        follow_logit = (trust_ai * 2.2) - (reactance * 1.3) - overload_pen - tier_pen

        p_follow = clamp(sigmoid(follow_logit), 0.02, 0.98)
        attempt_follow = rng.random() < p_follow

        if agent_type == "Behavioural":
            # Behavioural path: compare advisor vs fallback with misperception noise (Fix #4)
            # If "attempt_follow" true, the AI pushes the comparison towards rec.
            if attempt_follow:
                final = choose_with_noise(
                    state, scenario, rules, rec, fallback, rational_agent, rng, sigma
                )
            else:
                final = choose_with_noise(
                    state, scenario, rules, fallback, rec, rational_agent, rng, sigma
                )
        else:
            # For non-behavioural demo, still allow probabilistic follow:
            final = rec if attempt_follow else fallback

        followed_ai = getattr(final, "name", None) == getattr(rec, "name", None)

    # Under overload, user can slip to random option (Fix #4)
    final = maybe_slip_to_random_offer(state, scenario, final, rng, sigma)

    # Action vs no-action gate (Fix #0 and improved Fix #3)
    effort_cost = float(getattr(final, "effort_cost", 1.0))
    acts = decides_to_act(
        state=state,
        scenario_name=scenario.name,
        rules=rules,
        effort=effort_cost,
        reactance=(reactance if ai_enabled else 0.0),
        rng=rng,
    )

    # Normative regret benchmark (rational optimum)
    optimal_offer = rational_agent.choose_offer(state, scenario, rules)
    optimal_u = rational_agent.utility(state, scenario, optimal_offer, rules)

    if acts:
        outcome = simulate_outcome(copy_state(state), scenario, final, rules)
        realised_u = rational_agent.utility(state, scenario, final, rules)

        regret_action = max(0.0, optimal_u - realised_u)  # Fix #2
        regret_inaction = 0.0
        regret_total = regret_action

        foregone_u = 0.0
    else:
        # No action: stay put
        class NoActionOutcome:
            acted = False
            points_delta = 0
            effort_cost = 0.0
            new_tier = state.tier.value

        outcome = NoActionOutcome()

        # Regret for inaction: foregone optimal utility (Fix #2)
        regret_action = 0.0
        regret_inaction = max(0.0, optimal_u)
        regret_total = regret_inaction

        foregone_u = optimal_u

    # Strict tier movement (Fix #3 clean outputs)
    tier_up = (
        (state.tier.value == "Bronze" and outcome.new_tier == "Silver")
        or (state.tier.value == "Silver" and outcome.new_tier == "Gold")
    )
    tier_down = (
        (state.tier.value == "Gold" and outcome.new_tier == "Silver")
        or (state.tier.value == "Silver" and outcome.new_tier == "Bronze")
    )

    return {
        "user_id": user_id,
        "scenario": scenario_label(scenario.name),
        "tier": state.tier.value,
        "points": state.points,
        "day": state.day,
        "trust_ai": float(trust_ai),
        "ai_enabled": bool(ai_enabled),
        "advisor_action": getattr(rec, "name", "rec"),
        "final_action": getattr(final, "name", "final"),
        "followed_ai": bool(followed_ai) if ai_enabled else False,
        "acted": bool(getattr(outcome, "acted", False)),
        "new_tier": getattr(outcome, "new_tier", state.tier.value),
        "tier_up": bool(tier_up),
        "tier_down": bool(tier_down),
        "effort": float(getattr(outcome, "effort_cost", 0.0)),
        "regret_action": float(regret_action),
        "regret_inaction": float(regret_inaction),
        "regret_total": float(regret_total),
        "foregone_u": float(foregone_u),
        "sigma": float(sigma),
        "seed": int(rng_seed),
    }


def main():
    st.title("Behavioural AI Decision Explorer — Retail Loyalty")

    # -----------------------
    # Sidebar controls (keys to avoid DuplicateWidgetID)
    # -----------------------
    st.sidebar.header("Controls")

    agent_type = st.sidebar.selectbox(
        "Decision-maker",
        ["Behavioural", "Bounded-rational", "Rational"],
        index=0,
        key="agent_type_select",
    )

    user_profile = st.sidebar.selectbox(
        "User distribution",
        ["Mixed", "Near threshold", "Far from threshold"],
        index=0,
        key="user_profile_select",
    )


    ai_enabled_ui = st.sidebar.checkbox(
        "AI advisor enabled",
        value=True,
        key="ai_enabled_checkbox",
    )
    ai_enabled = st.sidebar.checkbox(
    "AI advisor enabled",
    value=True,
    key="ai_enabled_checkbox"
)




    reactance = st.sidebar.slider(
        "Reactance (reject AI under overload)",
        0.0, 1.5, 0.6, 0.1,
        disabled=not ai_enabled_ui,
        key="reactance_slider",
    )

    n_users = st.sidebar.slider(
        "Simulated users",
        200, 6000, 1500, 200,
        key="n_users_slider",
    )


    ai_enabled = st.sidebar.checkbox("AI advisor enabled", value=True)
    reactance = st.sidebar.slider(
    "Reactance (reject AI under overload)",
    0.0, 1.5, 0.6, 0.1,
    disabled=not ai_enabled,
    key="reactance_slider"
)



    seed = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
        key="seed_input",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("This is a behavioural simulation (counterfactual lab), not a real retail prediction model.")

    # -----------------------
    # Setup
    # -----------------------
    rules = TierRules()
    scenarios = make_scenarios()
    rng = random.Random(int(seed))

    behavioural_agent = BehaviouralAgent()
    bounded_agent = BoundedRationalAgent()
    rational_agent = RationalAgent()

    advisor = SimpleAdvisor()
    ai_response = BehaviouralAIResponse(reactance=reactance)  # kept for compatibility; decisions handled above

    # -----------------------
    # Simulation (Fix #5: within-user counterfactual AI OFF vs AI ON)
    # -----------------------
    rows = []
    delta_rows = []

    user_counter = 0

    for scenario in scenarios:
        for _ in range(int(n_users)):
            user_counter += 1
            base_state = sample_user_state(rng, rules, user_profile)
            trust_ai = sample_trust_ai(rng, base_state.tier)

            # Shared base seed to keep randomness comparable across OFF/ON
            base_trial_seed = rng.randint(0, 10_000_000)

            row_off = run_one_condition(
                user_id=user_counter,
                state=copy_state(base_state),
                trust_ai=trust_ai,
                scenario=scenario,
                rules=rules,
                agent_type=agent_type,
                ai_enabled=False,
                reactance=reactance,
                rng_seed=base_trial_seed,
                advisor=advisor,
                behavioural_agent=behavioural_agent,
                bounded_agent=bounded_agent,
                rational_agent=rational_agent,
                ai_response=ai_response,
            )

            row_on = run_one_condition(
                user_id=user_counter,
                state=copy_state(base_state),
                trust_ai=trust_ai,
                scenario=scenario,
                rules=rules,
                agent_type=agent_type,
                ai_enabled=ai_enabled_ui,
                reactance=reactance,
                rng_seed=base_trial_seed,  # same seed for fairness
                advisor=advisor,
                behavioural_agent=behavioural_agent,
                bounded_agent=bounded_agent,
                rational_agent=rational_agent,
                ai_response=ai_response,
            )

            rows.append(row_off)
            rows.append(row_on)

            # Deltas (AI ON minus AI OFF) – Fix #5 output improvement
            delta_rows.append({
                "user_id": user_counter,
                "scenario": row_on["scenario"],
                "tier": row_on["tier"],
                "delta_acted": int(row_on["acted"]) - int(row_off["acted"]),
                "delta_tier_up": int(row_on["tier_up"]) - int(row_off["tier_up"]),
                "delta_regret_total": row_on["regret_total"] - row_off["regret_total"],
                "delta_regret_action": row_on["regret_action"] - row_off["regret_action"],
                "delta_regret_inaction": row_on["regret_inaction"] - row_off["regret_inaction"],
                "trust_ai": row_on["trust_ai"],
            })

    df = pd.DataFrame(rows)
    df_delta = pd.DataFrame(delta_rows)

    # Focus view: AI-ON if enabled, else show OFF
    df_view = df[df["ai_enabled"] == bool(ai_enabled_ui)].reset_index(drop=True)

    # -----------------------
    # KPIs
    # -----------------------
    st.subheader("Key Outcomes")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Completion rate", f"{df_view['acted'].mean():.2%}")
    c2.metric("Tier-up rate", f"{df_view['tier_up'].mean():.2%}")

    if ai_enabled_ui:
        c3.metric("AI follow rate", f"{df_view['followed_ai'].mean():.2%}")
    else:
        c3.metric("AI follow rate", "—")

    c4.metric("Mean regret (total)", f"{df_view['regret_total'].mean():.3f}")
    c5.metric("Inaction share", f"{(1.0 - df_view['acted'].mean()):.2%}")

    # Counterfactual KPI summary (Fix #5)
    st.markdown("---")
    st.subheader("Counterfactual impact of AI (same users, AI ON − AI OFF)")

    d1, d2, d3 = st.columns(3)
    d1.metric("Δ completion", f"{df_delta['delta_acted'].mean():+.3f}")
    d2.metric("Δ tier-up", f"{df_delta['delta_tier_up'].mean():+.3f}")
    d3.metric("Δ regret (total)", f"{df_delta['delta_regret_total'].mean():+.3f}")

    # -----------------------
    # Charts (kept simple; you asked “before extra charts”)
    # -----------------------
    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("Completion Rate by Scenario")
        st.bar_chart(df_view.groupby("scenario")["acted"].mean())

        st.subheader("Tier-up Rate by Scenario")
        st.bar_chart(df_view.groupby("scenario")["tier_up"].mean())

    with right:
        if ai_enabled_ui:
            st.subheader("AI Follow Rate by Scenario")
            st.bar_chart(df_view.groupby("scenario")["followed_ai"].mean())

        st.subheader("Regret by Scenario (total; lower is better)")
        st.bar_chart(df_view.groupby("scenario")["regret_total"].mean())

    # -----------------------
    # Tables (clean + interpretable outputs)
    # -----------------------
    st.markdown("---")
    st.subheader("Sample decisions (AI ON view if enabled)")

    st.dataframe(
        df_view[
            [
                "scenario", "tier", "points", "day",
                "trust_ai",
                "advisor_action", "final_action",
                "followed_ai", "acted", "new_tier",
                "effort",
                "regret_action", "regret_inaction", "regret_total",
            ]
        ]
        .sample(min(30, len(df_view)), random_state=int(seed))
        .reset_index(drop=True)
    )


    st.subheader("Sample counterfactual deltas (AI ON − AI OFF)")
    st.dataframe(
        df_delta[
            [
                "scenario", "tier", "trust_ai",
                "delta_acted", "delta_tier_up",
                "delta_regret_action", "delta_regret_inaction", "delta_regret_total",
            ]
        ]
        .sample(min(30, len(df_delta)), random_state=int(seed))
        .reset_index(drop=True)
    )

    st.caption(
        "Interpretation hint: positive delta completion with positive delta regret suggests AI induces action but not always quality."
    )


if __name__ == "__main__":
    main()


