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

# Rule base: (diversity_term, stagnation_term) -> mutation_term
RULES = {
    ("Low", "Low"): "Medium",
    ("Low", "Medium"): "High",
    ("Low", "High"): "High",
    ("Medium", "Low"): "Low",
    ("Medium", "Medium"): "Medium",
    ("Medium", "High"): "High",
    ("High", "Low"): "Low",
    ("High", "Medium"): "Low",
    ("High", "High"): "Medium",
}

def infer(diversity, stagnation_norm):
    """Run the full rule base. Returns (mutation_rate, rule_trace).
    rule_trace is a list of dicts, one per fired rule (strength > 0),
    sorted by firing strength descending.
    """
    div_m = memberships(diversity)
    stag_m = memberships(stagnation_norm)

    fired = []
    num, den = 0.0, 0.0
    for (dt, st), cons in RULES.items():
        strength = min(div_m[dt], stag_m[st])  # Mamdani AND = min
        if strength > 0:
            fired.append({
                "diversity_term": dt, "diversity_deg": div_m[dt],
                "stagnation_term": st, "stagnation_deg": stag_m[st],
                "consequent": cons, "strength": strength,
            })
        num += strength * CONSEQUENT_VALUE[cons]
        den += strength
    mutation_rate = num / den if den > 0 else CONSEQUENT_VALUE["Medium"]
    fired.sort(key=lambda r: r["strength"], reverse=True)
    return mutation_rate, fired