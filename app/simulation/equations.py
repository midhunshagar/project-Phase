from .economy import EconomicState
from .policies import PolicyScenario


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def apply_policy(
    state: EconomicState,
    policy: PolicyScenario,
) -> EconomicState:

    s = state.copy()

    if policy.name == "income_tax":
        s.tax_rate = policy.counterfactual_value

    elif policy.name == "interest_rate":
        s.interest_rate = policy.counterfactual_value

    elif policy.name == "government_spending":
        s.government_spending = policy.counterfactual_value

    elif policy.name == "subsidy":
        # Subsidy is handled through investment support.
        pass

    else:
        raise ValueError(f"Unsupported policy: {policy.name}")

    return s


def step_economy(
    state: EconomicState,
    policy: PolicyScenario,
) -> EconomicState:

    s = apply_policy(state, policy)

    # --------------------------------------------------
    # Policy effects
    # --------------------------------------------------

    tax_effect = -0.003 * (s.tax_rate - 20.0)

    interest_effect = -0.0025 * (s.interest_rate - 6.0)

    spending_effect = (
    0.000003 * (s.government_spending - 250.0)
    )

    subsidy_effect = (
        0.002 * policy.counterfactual_value
        if policy.name == "subsidy"
        else 0.0
    )

    # --------------------------------------------------
    # Economic growth
    # --------------------------------------------------

    base_growth = 0.025

    growth_rate = (
        base_growth
        + tax_effect
        + interest_effect
        + spending_effect
        + subsidy_effect
    )

    # Keep annual growth within a sensible range.
    growth_rate = clamp(
        growth_rate,
        -0.05,
        0.08,
    )

    new_gdp = max(
        100.0,
        s.gdp * (1.0 + growth_rate),
    )

    # --------------------------------------------------
    # Consumption
    # --------------------------------------------------

    consumption_ratio = (
        0.60
        * (
            1.0
            - 0.015 * (s.tax_rate - 20.0)
        )
    )

    consumption = max(
        50.0,
        new_gdp * consumption_ratio,
    )

    # --------------------------------------------------
    # Investment
    # --------------------------------------------------

    investment_ratio = (
        0.20
        * (
            1.0
            - 0.02 * (s.interest_rate - 6.0)
        )
    )

    if policy.name == "subsidy":
        investment_ratio *= (
            1.0 + 0.008 * policy.counterfactual_value
        )

    investment = max(
        30.0,
        new_gdp * investment_ratio,
    )

    # --------------------------------------------------
    # Government spending
    # --------------------------------------------------

    government = max(
        0.0,
        s.government_spending,
    )

    # --------------------------------------------------
    # Inflation
    # --------------------------------------------------

    demand_pressure = (
        growth_rate * 100.0
    )

    new_inflation = clamp(
        0.75 * s.inflation
        + 0.25 * (
            5.0
            + 0.35 * demand_pressure
        ),
        1.0,
        15.0,
    )

    # --------------------------------------------------
    # Unemployment
    # --------------------------------------------------

    new_unemployment = clamp(
        s.unemployment
        - 0.20 * (growth_rate * 100.0)
        + 0.05 * (s.interest_rate - 6.0),
        2.5,
        15.0,
    )

    # --------------------------------------------------
    # Tax revenue
    # --------------------------------------------------

    tax_revenue = max(
        0.0,
        new_gdp
        * s.tax_rate
        / 100.0
        * 0.98,
    )

    # --------------------------------------------------
    # Emissions
    # --------------------------------------------------

    # Economic activity increases emissions.
    # Government spending provides a small
    # efficiency effect rather than a huge
    # negative multiplier.

    emissions_intensity = clamp(
        0.50
        * (
            1.0
            - 0.00002
            * max(
                0.0,
                s.government_spending - 250.0,
            )
        ),
        0.35,
        0.55,
    )

    new_emissions = max(
        50.0,
        new_gdp * emissions_intensity,
    )

    # --------------------------------------------------
    # Inequality
    # --------------------------------------------------

    inequality_change = (
        0.0008 * (s.tax_rate - 20.0)
        - 0.000001
        * max(
            0.0,
            s.government_spending - 250.0,
        )
    )

    new_inequality = clamp(
        s.inequality + inequality_change,
        0.25,
        0.65,
    )

    # --------------------------------------------------
    # Return new state
    # --------------------------------------------------

    return EconomicState(
        year=s.year + 1,
        gdp=new_gdp,
        inflation=new_inflation,
        unemployment=new_unemployment,
        consumption=consumption,
        investment=investment,
        government_spending=government,
        tax_rate=s.tax_rate,
        interest_rate=s.interest_rate,
        tax_revenue=tax_revenue,
        emissions=new_emissions,
        inequality=new_inequality,
    )