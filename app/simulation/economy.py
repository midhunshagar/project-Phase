from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class EconomicState:
    year: int = 2025
    gdp: float = 1000.0
    inflation: float = 5.0
    unemployment: float = 6.0
    consumption: float = 600.0
    investment: float = 200.0
    government_spending: float = 250.0
    tax_rate: float = 20.0
    interest_rate: float = 6.0
    tax_revenue: float = 200.0
    emissions: float = 500.0
    inequality: float = 0.35

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EconomicState":
        return cls(**data)

    def copy(self) -> "EconomicState":
        return EconomicState.from_dict(self.to_dict())
