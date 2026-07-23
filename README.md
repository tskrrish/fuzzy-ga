# fuzzy-ga

Explainable adaptive mutation control in genetic algorithms: a fuzzy logic approach to job-shop scheduling.

This project explores whether a small, human-readable fuzzy rule base can control a genetic algorithm's mutation rate as well as (or better than) a fixed rate — while also producing a plain-English explanation for *why* the rate changed at every generation. It solves the Job-Shop Scheduling Problem (JSSP) as a test bed and compares three mutation strategies head-to-head.

## The idea

A genetic algorithm's mutation rate controls the tradeoff between exploration and exploitation. Too low, and the population converges prematurely; too high, and it never converges at all. The usual fix is to make the rate adaptive — but most adaptive schemes (simulated-annealing-style decay, 1/5-success rules, etc.) are opaque: you can't easily explain *why* the rate is 0.08 at generation 40.

This project uses a small **Mamdani-style fuzzy inference system** instead. It watches two signals every generation:

- **Diversity** — mean pairwise normalized Hamming distance across the population
- **Stagnation** — how long it's been since the best solution improved (capped and normalized)

...and fuzzifies each into `Low` / `Medium` / `High` using triangular membership functions. A 9-rule table (all combinations of the two inputs) is evaluated, the firing strengths are combined via min-AND, and the mutation rate is produced by a weighted-average (zero-order Sugeno-style) defuzzification. Because every rule is legible, the system can narrate its own decision in one sentence per generation, e.g.:

```
Gen 5: mutation set to 0.098 [raw diversity=0.20, stagnation=0.80] because
diversity is Low (0.60) AND stagnation is High (1.00) → mutation should be High
(rule strength 0.60); also, diversity is Medium (0.40) AND stagnation is High
(1.00) → mutation should be Medium (rule strength 0.40).
```

## Three strategies compared

All three share identical GA machinery (selection, crossover, elitism) and differ only in how the mutation rate is chosen each generation:

| Strategy | How the rate is chosen |
|---|---|
| **static** | Fixed rate (0.05), never changes |
| **crisp** | Same 9-rule table as `fuzzy`, but with hard thresholds instead of fuzzy membership (no blending) |
| **fuzzy** | Full fuzzy controller — fuzzification, rule firing, weighted-average defuzzification |

Comparing `crisp` against `fuzzy` isolates the effect of *fuzziness itself* (soft blending near category boundaries), since both use the exact same rule table.

## Benchmarks

Two classic JSSP instances from [JSPLIB](https://github.com/tamy0612/JSPLIB), both by Fisher & Thompson:

- **FT06** — 6 jobs × 6 machines, known optimal makespan = 55
- **FT10** — 10 jobs × 10 machines, known optimal makespan = 930

## Project layout

```
src/jssp_fuzzy_ga/
├── core.py        # JSSP instance representation, semi-active decoder, GA operators
│                   # (tournament selection, POX crossover, swap mutation, diversity metric)
├── fuzzy.py        # The fuzzy inference system: membership functions, rule base,
│                   # inference, and plain-language explanation generator
├── experiment.py   # run_ga() / run_many(): wires the GA loop together for each
│                   # of the three mutation strategies and collects results
└── plotting.py     # Convergence curves and final-makespan boxplots from
                     # experiment results

notebooks/
└── 01_explore.ipynb   # Step-by-step walkthrough: build an instance, inspect
                        # chromosomes, test each operator in isolation, run the
                        # fuzzy controller on hand-picked inputs, then run full
                        # experiments and plot the comparison

Fuzzy-schedule-mutationGA-document.pdf   # Write-up / report accompanying the project
```

## Installation

Requires Python ≥ 3.9.

```bash
pip install -e .
```

This installs the package (`jssp_fuzzy_ga`) along with its dependencies: `numpy`, `matplotlib`, `scipy`.

## Quick start

```python
from jssp_fuzzy_ga.experiment import run_ga

result = run_ga(method="fuzzy", seed=42, instance="ft06", n_generations=150, record_trace=True)

print("Final makespan:", result["final"])
for line in result["trace"][:5]:
    print(line)
```

To reproduce the full comparison across strategies and instances:

```python
from jssp_fuzzy_ga.experiment import run_many
from jssp_fuzzy_ga.plotting import plot_convergence, plot_final_boxplot

curves_static, finals_static = run_many("static", n_runs=15, instance="ft06", n_generations=150)
curves_crisp,  finals_crisp  = run_many("crisp",  n_runs=15, instance="ft06", n_generations=150)
curves_fuzzy,  finals_fuzzy  = run_many("fuzzy",  n_runs=15, instance="ft06", n_generations=150)

data = {
    "ft06": {
        "static": {"curves": curves_static, "finals": finals_static},
        "crisp":  {"curves": curves_crisp,  "finals": finals_crisp},
        "fuzzy":  {"curves": curves_fuzzy,  "finals": finals_fuzzy},
    }
}
plot_convergence(data, instances=("ft06",))
plot_final_boxplot(data, instance="ft06")
```

Or just open `notebooks/01_explore.ipynb`, which walks through every piece (chromosome encoding, each GA operator in isolation, the fuzzy controller's membership functions and rule firing, then full multi-run experiments with statistical comparison via `scipy.stats.ttest_ind`) and regenerates the plots in `notebooks/`.

## Results at a glance

Precomputed comparison plots are included in `notebooks/`:

- `convergence_comparison.png` — mean best-makespan-so-far per generation, both instances, all three strategies, shaded by standard error
- `ft06_final_boxplot.png` / `ft10_final_boxplot.png` — distribution of final makespans across runs, per strategy

## License

MIT — see `LICENSE`.
