# behavioural-ai-decision-engine
Loyalty decisions under cognitive load (when AI advice helps vs harms)


# Behavioural AI Decision Engine (Retail Loyalty)

A simulation-based engine that models how consumers make loyalty and repeat-purchase decisions under cognitive bias — and evaluates when AI recommendations improve outcomes versus when they backfire.

## Why this matters
Retail loyalty systems increasingly use ML/AI to recommend offers, timing, and next-best actions. But consumers are not perfectly rational: they procrastinate, avoid effort, get overwhelmed by choices, and react differently to defaults and reminders. As a result, “better predictions” do not always translate into better decisions or higher follow-through.

This project makes that gap measurable by simulating decision-making for different agent types and scoring AI recommendations on decision quality, cognitive effort, and long-term value — not accuracy alone.

## What this engine does
1. Generates loyalty decision scenarios (e.g., redeem now vs later, pick an offer under overload, respond to reminders).
2. Simulates choices for:
   - Rational agent (utility maximiser)
   - Bounded-rational agent (limited attention + noise)
   - Behavioural agent (present bias, loss aversion, effort aversion, susceptibility to defaults)
3. Introduces an AI “advisor” that recommends actions (not behaviour predictions).
4. Evaluates outcomes using decision-focused metrics.

## Core scenarios (v1)
- Redeem now vs later (present bias / procrastination)
- Offer overload (decision fatigue / choice overload)
- AI suggestion vs autonomy (automation bias / reactance)

## Evaluation metrics
- Completion rate (did the user act?)
- Long-term value (loyalty value proxy across time)
- Regret (gap vs optimal policy)
- Cognitive effort (proxy cost of decision-making)
- Over-reliance / drop-off after recommendations

## Why this is different from typical ML demos
Most ML work optimises prediction metrics under an assumption of rational action. This project evaluates AI in the loop with biased humans, focusing on decision quality and real-world follow-through.

## Limitations (initial)
- Simulation assumptions simplify real behaviour and context.
- “Cognitive effort” is proxied, not measured via lab studies.
- The AI advisor is intentionally simple in v1 to isolate behavioural effects.

## Roadmap
- Add calibrated bias parameters from behavioural literature ranges.
- Add an “intervention layer” (defaults, framing, timing).
- Replace simple advisor with a learning policy and compare against human-centred metrics.
