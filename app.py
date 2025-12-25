import math
import random
import pandas as pd
import streamlit as st

from src.environment.scenarios import TierRules, Tier, UserState, make_scenarios
from src.agents.rational import RationalAgent
from src.agents.bounded import BoundedRationalAgent
from src.agents.behavioural import BehaviouralAgent
from src.advisor.simple_advisor import SimpleAdvisor
from src.evaluation.metrics import simulate_outcome

st.set_page_config(page_title="Behavioural AI Decision Explorer", layout="wide")


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def is_overload(name):
    return "Overload" in name


def label(name):
    return name.replace("_", " ")


def copy_state(s):
    return UserState(tier=s.tier, day=s.day, points=s.points, last_window_points=s.last_window_points)


def sample_user_state(rng, rules, profile):
    tier = rng.choices([Tier.BRONZE, Tier.SILVER, Tier.GOLD], weights=[0.45, 0.40, 0.15], k=1)[0]
    day = rng.randint(1, rules.window_days)

    def clamp_int(x, lo, hi):
        return max(lo, min(hi, x))

    if tier == Tier.BRONZE:
        if profile == "Near threshold":
            points = rng.randint(max(0, rules.silver_threshold - 120), rules.silver_threshold - 1)
        elif profile == "Far from threshold":
            points = rng.randint(0, max(1, rules.silver_threshold - 250))
        else:
            points = rng.randint(0, rules.silver_threshold - 1)
        last_window_points = clamp_int(points + rng.randint(-150, 120), 0, rules.silver_threshold - 1)

    elif tier == Tier.SILVER:
        if profile == "Near threshold":
            points = rng.randint(max(rules.silver_threshold, rules.gold_threshold - 140), rules.gold_threshold - 1)
        elif profile == "Far from threshold":
            points = rng.randint(rules.silver_threshold, max(rules.silver_threshold + 1, rules.gold_threshold - 300))
        else:
            points = rng.randint(rules.silver_threshold, rules.gold_threshold - 1)
        last_window_points = clamp_int(points + rng.randint(-220, 160), 0, rules.gold_threshold - 1)

    else:
        points = rng.randint(rules.gold_threshold, rules.gold_threshold + 400)
        if profile == "Near threshold":
            last_window_points = rng.randint(max(0, rules.gold_threshold - 250), rules.gold_threshold + 50)
        elif profile == "Far from threshold":
            last_window_points = rng.randint(rules.gold_threshold, rules.gold_threshold + 300)
        else:
            last_window_points = rng.randint(max(0, rules.gold_threshold - 350), rules.gold_threshold + 250)

    return UserState(tier=tier, day=day, points=points, last_window_points=last_window_points)


def sample_trust_ai(rng, tier):
    if tier == Tier.BRONZE:
        return rng.betavariate(2.0, 3.2)
    if tier == Tier.SILVER:
        return rng.betavariate(2.5, 2.7)
    return rng.betavariate(2.0, 2.8)


def tier_params(tier):
    if tier == Tier.BRONZE:
        return {"base_act": 0.88, "eff_pen": 0.28, "rea_pen": 0.18, "ovl_pen": 0.22, "urg_boost": 0.22}
    if tier == Tier.SILVER:
        return {"base_act": 0.85, "eff_pen": 0.24, "rea_pen": 0.16, "ovl_pen": 0.26, "urg_boost": 0.18}
    return {"base_act": 0.80, "eff_pen": 0.22, "rea_pen": 0.20, "ovl_pen": 0.30, "urg_boost": 0.12}


def dist_to_next_tier(state, rules):
    if state.tier == Tier.BRONZE:
        return max(0, rules.silver_threshold - state.points)
    if state.tier == Tier.SILVER:
        return max(0, rules.gold_threshold - state.points)
    return 999999


def decides_to_act(state, scenario_name, rules, effort, reactance, rng):
    p = tier_params(state.tier)
    days_left = max(0, rules.window_days - state.day)
    urgency = 1.0 - (days_left / max(1, rules.window_days))
    dist = dist_to_next_tier(state, rules)
    closeness = 1.0 - clamp(dist / 400.0, 0.0, 1.0)

    p_act = (
        p["base_act"]
        + p["urg_boost"] * urgency
        + 0.18 * closeness
        - p["eff_pen"] * float(effort)
        - p["rea_pen"] * float(reactance)
        - (p["ovl_pen"] if is_overload(scenario_name) else 0.0)
    )
    p_act = clamp(p_act, 0.05, 0.95)
    return rng.random() < p_act


def overload_sigma(state, scenario_name):
    if not is_overload(scenario_name):
        return 0.05
    if state.tier == Tier.BRONZE:
        return 0.35
    if state.tier == Tier.SILVER:
        return 0.30
    return 0.33


def choose_between_with_noise(state, scenario, rules, a, b, rational_eval, rng, sigma):
    ua = rational_eval.utility(state, scenario, a, rules)
    ub = rational_eval.utility(state, scenario, b, rules)
    pa = ua + rng.gauss(0.0, sigma)
    pb = ub + rng.gauss(0.0, sigma)
    return a if pa >= pb else b


def maybe_slip_random_offer(state, scenario, chosen, rng, sigma):
    offers = getattr(scenario, "offers", None)
    if not offers or len(offers) < 2:
        return chosen
    base = 0.04 if state.tier == Tier.BRONZE else 0.02
    slip_p = clamp(base + sigma * 0.08, 0.0, 0.18)
    if rng.random() >= slip_p:
        return chosen
    alts = [o for o in offers if getattr(o, "name", None) != getattr(chosen, "name", None)]
    return rng.choice(alts) if alts else chosen


def strict_tier_up(old_tier_value, new_tier_value):
    return (old_tier_value == "Bronze" and new_tier_value == "Silver") or (old_tier_value == "Silver" and new_tier_value == "Gold")


def strict_tier_down(old_tier_value, new_tier_value):
    return (old_tier_value == "Gold" and new_tier_value == "Silver") or (old_tier_value == "Silver" and new_tier_value == "Bronze")


def run_one(user_id, state, trust_ai, scenario, rules, agent_type, ai_enabled, reactance, trial_seed, advisor, behavioural_agent, bounded_agent, rational_agent):
    rng = random.Random(int(trial_seed))

    if agent_type == "Rational":
        agent = rational_agent
    elif agent_type == "Bounded-rational":
        agent = bounded_agent
    else:
        agent = behavioural_agent

    fallback = agent.choose_offer(state, scenario, rules)
    rec = advisor.recommend(state, scenario, rules)

    sigma = overload_sigma(state, scenario.name)

    final = fallback
    followed_ai = False

    if ai_enabled:
        overload_pen = 0.20 if is_overload(scenario.name) else 0.0
        gold_pen = 0.06 if state.tier == Tier.GOLD else 0.0
        follow_logit = (trust_ai * 2.2) - (reactance * 1.3) - overload_pen - gold_pen
        p_follow = clamp(sigmoid(follow_logit), 0.02, 0.98)
        attempt_follow = rng.random() < p_follow

        if agent_type == "Behavioural":
            if attempt_follow:
                final = choose_between_with_noise(state, scenario, rules, rec, fallback, rational_agent, rng, sigma)
            else:
                final = choose_between_with_noise(state, scenario, rules, fallback, rec, rational_agent, rng, sigma)
        else:
            final = rec if attempt_follow else fallback

        followed_ai = getattr(final, "name", None) == getattr(rec, "name", None)

    final = maybe_slip_random_offer(state, scenario, final, rng, sigma)

    effort_cost = float(getattr(final, "effort_cost", 1.0))
    acts = decides_to_act(state, scenario.name, rules, effort_cost, (reactance if ai_enabled else 0.0), rng)

    optimal_offer = rational_agent.choose_offer(state, scenario, rules)
    optimal_u = rational_agent.utility(state, scenario, optimal_offer, rules)

    if acts:
        outcome = simulate_outcome(copy_state(state), scenario, final, rules)
        realised_u = rational_agent.utility(state, scenario, final, rules)
        regret_action = max(0.0, optimal_u - realised_u)
        regret_inaction = 0.0
        regret_total = regret_action
        foregone_u = 0.0
    else:
        class NoActionOutcome:
            acted = False
            points_delta = 0
            effort_cost = 0.0
            new_tier = state.tier.value

        outcome = NoActionOutcome()
        regret_action = 0.0
        regret_inaction = max(0.0, optimal_u)
        regret_total = regret_inaction
        foregone_u = optimal_u

    tier_up = strict_tier_up(state.tier.value, outcome.new_tier)
    tier_down = strict_tier_down(state.tier.value, outcome.new_tier)

    return {
        "user_id": int(user_id),
        "scenario": label(scenario.name),
        "tier": state.tier.value,
        "points": int(state.points),
        "day": int(state.day),
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
        "seed": int(trial_seed),
    }


def main():
    st.title("Behavioural AI Decision Explorer - Retail Loyalty")

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

    seed = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
        key="seed_input",
    )

    rules = TierRules()
    scenarios = make_scenarios()
    rng = random.Random(int(seed))

    behavioural_agent = BehaviouralAgent()
    bounded_agent = BoundedRationalAgent()
    rational_agent = RationalAgent()
    advisor = SimpleAdvisor()

    rows = []
    delta_rows = []
    user_counter = 0

    for scenario in scenarios:
        for _ in range(int(n_users)):
            user_counter += 1
            base_state = sample_user_state(rng, rules, user_profile)
            trust_ai = sample_trust_ai(rng, base_state.tier)
            trial_seed = rng.randint(0, 10_000_000)

            row_off = run_one(
                user_id=user_counter,
                state=copy_state(base_state),
                trust_ai=trust_ai,
                scenario=scenario,
                rules=rules,
                agent_type=agent_type,
                ai_enabled=False,
                reactance=reactance,
                trial_seed=trial_seed,
                advisor=advisor,
                behavioural_agent=behavioural_agent,
                bounded_agent=bounded_agent,
                rational_agent=rational_agent,
            )

            row_on = run_one(
                user_id=user_counter,
                state=copy_state(base_state),
                trust_ai=trust_ai,
                scenario=scenario,
                rules=rules,
                agent_type=agent_type,
                ai_enabled=bool(ai_enabled_ui),
                reactance=reactance,
                trial_seed=trial_seed,
                advisor=advisor,
                behavioural_agent=behavioural_agent,
                bounded_agent=bounded_agent,
                rational_agent=rational_agent,
            )

            rows.append(row_off)
            rows.append(row_on)

            delta_rows.append({
                "user_id": int(user_counter),
                "scenario": row_on["scenario"],
                "tier": row_on["tier"],
                "trust_ai": row_on["trust_ai"],
                "delta_acted": int(row_on["acted"]) - int(row_off["acted"]),
                "delta_tier_up": int(row_on["tier_up"]) - int(row_off["tier_up"]),
                "delta_regret_total": float(row_on["regret_total"] - row_off["regret_total"]),
                "delta_regret_action": float(row_on["regret_action"] - row_off["regret_action"]),
                "delta_regret_inaction": float(row_on["regret_inaction"] - row_off["regret_inaction"]),
            })

    df = pd.DataFrame(rows)
    df_delta = pd.DataFrame(delta_rows)

    df_view = df[df["ai_enabled"] == bool(ai_enabled_ui)].reset_index(drop=True)

    st.subheader("Key Outcomes")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Completion rate", f"{df_view['acted'].mean():.2%}")
    c2.metric("Tier-up rate", f"{df_view['tier_up'].mean():.2%}")
    c3.metric("AI follow rate", f"{df_view['followed_ai'].mean():.2%}" if ai_enabled_ui else "-")
    c4.metric("Mean regret total", f"{df_view['regret_total'].mean():.3f}")
    c5.metric("Inaction share", f"{(1.0 - df_view['acted'].mean()):.2%}")

    st.markdown("---")
    st.subheader("Counterfactual impact (AI on minus AI off)")
    d1, d2, d3 = st.columns(3)
    d1.metric("Delta completion", f"{df_delta['delta_acted'].mean():+.3f}")
    d2.metric("Delta tier-up", f"{df_delta['delta_tier_up'].mean():+.3f}")
    d3.metric("Delta regret total", f"{df_delta['delta_regret_total'].mean():+.3f}")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("Completion rate by scenario")
        st.bar_chart(df_view.groupby("scenario")["acted"].mean())
        st.subheader("Tier-up rate by scenario")
        st.bar_chart(df_view.groupby("scenario")["tier_up"].mean())

    with right:
        if ai_enabled_ui:
            st.subheader("AI follow rate by scenario")
            st.bar_chart(df_view.groupby("scenario")["followed_ai"].mean())
        st.subheader("Regret total by scenario")
        st.bar_chart(df_view.groupby("scenario")["regret_total"].mean())

    st.markdown("---")
    st.subheader("Sample decisions")

    st.dataframe(
        df_view[
            [
                "scenario", "tier", "points", "day", "trust_ai",
                "advisor_action", "final_action",
                "followed_ai", "acted", "new_tier",
                "effort", "regret_action", "regret_inaction", "regret_total",
            ]
        ]
        .sample(min(30, len(df_view)), random_state=int(seed))
        .reset_index(drop=True)
    )

    st.subheader("Sample deltas (AI on minus AI off)")
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


if __name__ == "__main__":
    main()
