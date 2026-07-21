"""
Runs the GA on FT06 under three mutation-control strategies:

  1. static  -- constant mutation rate (0.10)
  2. crisp   -- SAME rule table as the fuzzy controller, but with hard
                thresholds instead of fuzzy membership (no blending).
  3. fuzzy   -- the fuzzy logic controller (fuzzy_controller.py)

All three share identical GA machinery (selection, crossover, elitism),
differing only in how the mutation rate is chosen each generation.
"""
import numpy as np
from .core import make_instance, tournament_select, pox_crossover, swap_mutation, population_diversity
from . import fuzzy as fc

POP_SIZE = 60
ELITE = 2
STAGNATION_CAP = fc.STAGNATION_CAP


def crisp_rate(diversity, stagnation_norm):
    """Same 9-rule table as the fuzzy controller, but hard-thresholded."""
    def level(x):
        if x < 1 / 3:
            return "Low"
        elif x < 2 / 3:
            return "Medium"
        return "High"
    cons = fc.RULES[(level(diversity), level(stagnation_norm))]
    return fc.CONSEQUENT_VALUE[cons]