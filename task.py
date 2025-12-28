# task.py
from dataclasses import dataclass, field
from typing import List
import random

@dataclass
class Task:
    id: str
    optimistic: float
    most_likely: float
    pessimistic: float
    base_accuracy: float
    token_cost: float
    depends_on: List[str] = field(default_factory=list)

    def sample_time(self, alpha):
        beta = random.betavariate(2 + 4*alpha, 2)
        return self.optimistic + beta * (self.pessimistic - self.optimistic)

    def accuracy(self, alpha):
        return min(1.0, self.base_accuracy + 0.4 * alpha)
