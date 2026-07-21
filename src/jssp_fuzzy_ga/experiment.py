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

def run_ga(method, seed, instance="ft06", n_generations=150, record_trace=False):
    """method in {"static", "crisp", "fuzzy"}. Returns dict with
    best_curve (len n_generations+1) and, if record_trace, a list of
    per-generation explanation strings (fuzzy method only)."""
    inst = make_instance(instance)
    rng = np.random.default_rng(seed)

    pop = np.stack([inst.random_chromosome(rng) for _ in range(POP_SIZE)])
    fitness = np.array([inst.decode(c) for c in pop])

    best_curve = [fitness.min()]
    best_so_far = fitness.min()
    gens_since_improve = 0
    trace = []

    for gen in range(1, n_generations + 1):
        diversity = population_diversity(pop)
        stagnation_norm = min(gens_since_improve, STAGNATION_CAP) / STAGNATION_CAP

        if method == "static":
            mut_rate = 0.05
        elif method == "crisp":
            mut_rate = crisp_rate(diversity, stagnation_norm)
        elif method == "fuzzy":
            mut_rate, fired = fc.infer(diversity, stagnation_norm)
            if record_trace:
                trace.append(fc.explain(gen, diversity, stagnation_norm, mut_rate, fired))
        else:
            raise ValueError(method)

        # elitism: carry over the best individuals unchanged
        order = np.argsort(fitness)
        new_pop = [pop[order[i]].copy() for i in range(ELITE)]

        while len(new_pop) < POP_SIZE:
            p1 = tournament_select(pop, fitness, rng)
            p2 = tournament_select(pop, fitness, rng)
            child = pox_crossover(p1, p2, inst.n_jobs, rng)
            child = swap_mutation(child, mut_rate, rng)
            new_pop.append(child)

        pop = np.stack(new_pop)
        fitness = np.array([inst.decode(c) for c in pop])

        gen_best = fitness.min()
        if gen_best < best_so_far:
            best_so_far = gen_best
            gens_since_improve = 0
        else:
            gens_since_improve += 1
        best_curve.append(best_so_far)

    result = {"best_curve": np.array(best_curve), "final": best_so_far}
    if record_trace:
        result["trace"] = trace
    return result