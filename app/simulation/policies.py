from dataclasses import dataclass
from typing import Literal

PolicyName = Literal["income_tax", "interest_rate", "government_spending", "subsidy"]

@dataclass
class PolicyScenario:
    name: PolicyName
    baseline_value: float
    counterfactual_value: float
    duration_years: int = 5

    @property
    def change(self) -> float:
        return self.counterfactual_value - self.baseline_value

    def validate(self) -> None:
        if self.duration_years < 1 or self.duration_years > 20:
            raise ValueError("duration_years must be between 1 and 20.")
        if self.name in {"income_tax", "interest_rate"}:
            if not 0 <= self.counterfactual_value <= 50:
                raise ValueError(f"{self.name} must be between 0 and 50.")
        if self.name == "government_spending" and self.counterfactual_value < 0:
            raise ValueError("Government spending cannot be negative.")
        if self.name == "subsidy" and not 0 <= self.counterfactual_value <= 30:
            raise ValueError("Subsidy rate must be between 0 and 30.")
