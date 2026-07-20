"""
Fuzzy logic controller for GA mutation rate.

Inputs (both already normalized to [0,1]):
  - diversity   : mean pairwise normalized Hamming distance in the population
  - stagnation  : min(generations_since_improvement, CAP) / CAP

Output:
  - mutation rate in {0.01 (Low), 0.10 (Medium), 0.30 (High)} blended by a
    weighted-average (zero-order Sugeno-style) defuzzification. This keeps
    every step of the inference legible enough to narrate in one sentence,
    which is the whole point of this project.
"""

TERMS = ("Low", "Medium", "High")
CONSEQUENT_VALUE = {"Low": 0.02, "Medium": 0.05, "High": 0.12}
STAGNATION_CAP = 20  # generations


def memberships(x):
    """Return {"Low": deg, "Medium": deg, "High": deg} for x in [0,1]."""
    x = max(0.0, min(1.0, x))
    low = max(0.0, min(1.0, (0.5 - x) / 0.5))
    med = max(0.0, 1.0 - abs(x - 0.5) / 0.5)
    high = max(0.0, min(1.0, (x - 0.5) / 0.5))
    return {"Low": low, "Medium": med, "High": high}