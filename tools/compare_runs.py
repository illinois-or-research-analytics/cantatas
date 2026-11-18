#!/usr/bin/env python3
"""Compare the macroscopic properties of two groups of simulation runs.

Each run is reduced first with `run_summary <run_dir>` (built with the
simulator), which writes <run_dir>/summary.json. Then:

    compare_runs.py --baseline old/rep*/summary.json --candidate new/rep*/summary.json

Runs are stochastic, so differences are judged against replicate-to-replicate
variation:
  * scalar metrics: the candidate mean must lie within 3 standard errors of the
    baseline mean (or within 2% when both groups barely vary);
  * distributions (in-degree, out-degree, citation age): the KS distance between
    the pooled groups must not exceed 1.5x the largest KS distance between two
    baseline replicates (with a floor of 0.01).
Exit status is 0 when everything passes.
"""
import argparse
import itertools
import json
import math
import sys

import numpy as np


def load(path):
    with open(path) as f:
        data = json.load(f)
    for key, value in list(data.items()):
        if isinstance(value, dict):
            data[key] = {int(k): int(v) for k, v in value.items()}
    return data


def pooled(hists):
    total = {}
    for h in hists:
        for k, v in h.items():
            total[k] = total.get(k, 0) + v
    return total


def ks_distance(h1, h2):
    support = sorted(set(h1) | set(h2))
    c1 = np.cumsum([h1.get(k, 0) for k in support], dtype=float)
    c2 = np.cumsum([h2.get(k, 0) for k in support], dtype=float)
    if c1[-1] == 0 or c2[-1] == 0:
        return float("nan")
    return float(np.max(np.abs(c1 / c1[-1] - c2 / c2[-1])))


def quantile(h, q):
    keys = sorted(h)
    counts = np.array([h[k] for k in keys], dtype=float)
    cum = np.cumsum(counts) / counts.sum()
    return keys[int(np.searchsorted(cum, q))]


def gini(h):
    keys = np.array(sorted(h), dtype=float)
    counts = np.array([h[int(k)] for k in keys], dtype=float)
    n = counts.sum()
    total = (keys * counts).sum()
    if n == 0 or total == 0:
        return 0.0
    # Gini from the sorted distribution: 1 - 2 * area under the Lorenz curve.
    cum_people = np.cumsum(counts) / n
    cum_wealth = np.cumsum(keys * counts) / total
    prev_people = np.concatenate([[0.0], cum_people[:-1]])
    prev_wealth = np.concatenate([[0.0], cum_wealth[:-1]])
    area = np.sum((cum_people - prev_people) * (cum_wealth + prev_wealth) / 2)
    return float(1 - 2 * area)


def scalars(run):
    age = {k: v for k, v in run["citation_age"].items() if k >= 0}
    age_total = sum(age.values())
    agents = max(run["agents"], 1)
    top = run["top_in_degrees"]
    return {
        "nodes": run["nodes"],
        "edges": run["edges"],
        "mean agent out-degree": run["agent_edges"] / agents,
        "share of agent citations to seeds": run["agent_edges_to_seeds"] / max(run["agent_edges"], 1),
        "intra-cluster citation share": run["agent_edges_intra_cluster"]
        / max(run["agent_edges_clustered_source"], 1),
        "same-year citation share": age.get(0, 0) / max(age_total, 1),
        "mean citation age": sum(k * v for k, v in age.items()) / max(age_total, 1),
        "median citation age": quantile(age, 0.5),
        "max in-degree": top[0] if top else 0,
        "mean top-10 in-degree": float(np.mean(top[:10])) if top else 0,
        "in-degree Gini (all)": gini(run["in_degree_all"]),
        "in-degree Gini (agents)": gini(run["in_degree_agents"]),
        "uncited agent share": run["in_degree_agents"].get(0, 0) / agents,
        "mean agent in-degree": sum(k * v for k, v in run["in_degree_agents"].items()) / agents,
        "in-degree p99 (all)": quantile(run["in_degree_all"], 0.99),
        "in-degree p99.9 (all)": quantile(run["in_degree_all"], 0.999),
        "in-degree p99.99 (all)": quantile(run["in_degree_all"], 0.9999),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--z", type=float, default=3.0, help="allowed standard errors for scalar metrics")
    args = parser.parse_args()

    base = [load(p) for p in args.baseline]
    cand = [load(p) for p in args.candidate]
    ok = True

    print(f"Baseline runs: {len(base)}   Candidate runs: {len(cand)}\n")
    print(f"{'metric':38s} {'baseline mean (sd)':>24s} {'candidate mean (sd)':>24s} {'z':>7s}  result")
    sb = [scalars(r) for r in base]
    sc = [scalars(r) for r in cand]
    for name in sb[0]:
        b = np.array([s[name] for s in sb], dtype=float)
        c = np.array([s[name] for s in sc], dtype=float)
        se = math.sqrt(b.var(ddof=1) / len(b) + c.var(ddof=1) / len(c)) if len(b) > 1 and len(c) > 1 else 0.0
        diff = c.mean() - b.mean()
        z = diff / se if se > 0 else (0.0 if diff == 0 else float("inf"))
        rel = abs(diff) / max(abs(b.mean()), 1e-12)
        passed = abs(z) <= args.z or rel <= 0.02
        ok &= passed
        print(f"{name:38s} {b.mean():14.5g} ({b.std(ddof=1) if len(b) > 1 else 0:8.3g}) "
              f"{c.mean():14.5g} ({c.std(ddof=1) if len(c) > 1 else 0:8.3g}) {z:7.2f}  {'ok' if passed else 'DIFFERS'}")

    print(f"\n{'distribution':38s} {'KS between groups':>18s} {'max KS within baseline':>24s}  result")
    for key in ["in_degree_all", "in_degree_agents", "in_degree_seeds", "out_degree_agents", "citation_age"]:
        between = ks_distance(pooled(r[key] for r in base), pooled(r[key] for r in cand))
        within = max((ks_distance(a[key], b[key]) for a, b in itertools.combinations(base, 2)), default=0.0)
        limit = max(1.5 * within, 0.01)
        passed = between <= limit
        ok &= passed
        print(f"{key:38s} {between:18.4f} {within:24.4f}  {'ok' if passed else 'DIFFERS'}")

    print("\nRESULT:", "all metrics within replicate variation" if ok else "some metrics differ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
