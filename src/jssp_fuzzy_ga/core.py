"""
Core Job-Shop Scheduling Problem (JSSP) representation and GA operators.

Reference instance: FT06 (Fisher & Thompson 6x6), fetched from JSPLIB
(https://github.com/tamy0612/JSPLIB/blob/master/instances/ft06).
Known optimal makespan = 55.
"""
import numpy as np


FT06_JOBS = [
    [(2, 1), (0, 3), (1, 6), (3, 7), (5, 3), (4, 6)],
    [(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)],
    [(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)],
    [(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)],
    [(2, 9), (1, 3), (4, 5), (5, 4), (0, 3), (3, 1)],
    [(1, 3), (3, 3), (5, 9), (0, 10), (4, 4), (2, 1)],
]
FT06_OPTIMUM = 55

# Fisher & Thompson 10x10 instance (mt10), also from JSPLIB. Known optimum = 930.
FT10_JOBS = [
    [(0, 29), (1, 78), (2, 9), (3, 36), (4, 49), (5, 11), (6, 62), (7, 56), (8, 44), (9, 21)],
    [(0, 43), (2, 90), (4, 75), (9, 11), (3, 69), (1, 28), (6, 46), (5, 46), (7, 72), (8, 30)],
    [(1, 91), (0, 85), (3, 39), (2, 74), (8, 90), (5, 10), (7, 12), (6, 89), (9, 45), (4, 33)],
    [(1, 81), (2, 95), (0, 71), (4, 99), (6, 9), (8, 52), (7, 85), (3, 98), (9, 22), (5, 43)],
    [(2, 14), (0, 6), (1, 22), (5, 61), (3, 26), (4, 69), (8, 21), (7, 49), (9, 72), (6, 53)],
    [(2, 84), (1, 2), (5, 52), (3, 95), (8, 48), (9, 72), (0, 47), (6, 65), (4, 6), (7, 25)],
    [(1, 46), (0, 37), (3, 61), (2, 13), (6, 32), (5, 21), (9, 32), (8, 89), (7, 30), (4, 55)],
    [(2, 31), (0, 86), (1, 46), (5, 74), (4, 32), (6, 88), (8, 19), (9, 48), (7, 36), (3, 79)],
    [(0, 76), (1, 69), (3, 76), (5, 51), (2, 85), (9, 11), (6, 40), (7, 89), (4, 26), (8, 74)],
    [(1, 85), (0, 13), (2, 61), (6, 7), (8, 64), (9, 76), (5, 47), (3, 52), (4, 90), (7, 45)],
]
FT10_OPTIMUM = 930


class JSSPInstance:
    def __init__(self, jobs):
        self.jobs = jobs
        self.n_jobs = len(jobs)
        self.n_ops_per_job = [len(j) for j in jobs]
        self.chrom_len = sum(self.n_ops_per_job)
        self.n_machines = max(m for job in jobs for m, _ in job) + 1

    def random_chromosome(self, rng):
        genes = []
        for j, n_ops in enumerate(self.n_ops_per_job):
            genes += [j] * n_ops
        genes = np.array(genes, dtype=np.int32)
        rng.shuffle(genes)
        return genes

    def decode(self, chromosome):
        """Semi-active decode: returns makespan (int)."""
        next_op = [0] * self.n_jobs
        machine_free = [0] * self.n_machines
        job_free = [0] * self.n_jobs
        for gene in chromosome:
            j = int(gene)
            op_idx = next_op[j]
            m, dur = self.jobs[j][op_idx]
            start = max(machine_free[m], job_free[j])
            finish = start + dur
            machine_free[m] = finish
            job_free[j] = finish
            next_op[j] += 1
        return max(job_free)


def make_instance(name="ft06"):
    if name == "ft06":
        return JSSPInstance(FT06_JOBS)
    elif name == "ft10":
        return JSSPInstance(FT10_JOBS)
    raise ValueError(name)



# ---------------------------------------------------------------------------
# GA operators
# ---------------------------------------------------------------------------

def tournament_select(pop, fitness, rng, k=3):
    idx = rng.integers(0, len(pop), size=k)
    best = idx[0]
    for i in idx[1:]:
        if fitness[i] < fitness[best]:
            best = i
    return pop[best]


def pox_crossover(p1, p2, n_jobs, rng):
    """Precedence-preserving order-based crossover for operation-based encoding."""
    jobs = np.arange(n_jobs)
    rng.shuffle(jobs)
    split = rng.integers(1, n_jobs) if n_jobs > 1 else 1 "split used to break"
    j1 = set(jobs[:split].tolist())

    child = np.full_like(p1, -1)
    for i, g in enumerate(p1):
        if g in j1:
            child[i] = g
    fill_vals = [g for g in p2 if g not in j1]
    fi = 0
    for i in range(len(child)):
        if child[i] == -1:
            child[i] = fill_vals[fi]
            fi += 1
    return child


def swap_mutation(chrom, rate, rng):
    """Each pass, with probability `rate`, swap two random positions."""
    c = chrom.copy()
    if rng.random() < rate:
        i, j = rng.integers(0, len(c), size=2)
        c[i], c[j] = c[j], c[i]
    return c


def population_diversity(pop):
    """Mean pairwise normalized Hamming distance across the population, in [0,1]."""
    diffs = (pop[:, None, :] != pop[None, :, :]).sum(axis=2)
    n = len(pop)
    iu = np.triu_indices(n, k=1)
    return float(diffs[iu].mean()) / pop.shape[1]

