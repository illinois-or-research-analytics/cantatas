# CantataS: Community-Aware Agent-Based Citation Simulator

The **Community-aware agent-based citation Simulator (CantataS)** is a network growth model built on top of SASCA-ReSA. CantataS simulates large-scale scientific citation dynamics, author lifecycles, and research community formation over decades of simulated time.

Version 9.0 makes **LadyBugDB**, a compact bidirectional CSR graph store, the only graph the simulation reads; runs on any number of MPI ranks with **identical results for a given seed**; and restarts from **checksummed binary snapshots** in minutes, so long runs can be split into chained 4-hour jobs.

---

## Recent Changelogs

### 9.0.0 (single graph store, deterministic distributed runs, restartable)
- **One graph store.** The in-memory `Graph` (a vector per node) is gone. The model reads LadyBugDB: a bidirectional CSR (forward = papers cited, backward = citing papers) plus per-node attribute columns, a cluster index and an author registry. Every MPI rank holds an identical copy and applies every year's new nodes and edges in the same order. Fixes the 8.x multi-rank behavior where each rank simulated against, and rank 0 exported, only part of the new edges.
- **Deterministic runs.** Each agent draws from its own random stream derived from the run seed, so a run gives the same output on 1, 2 or 8 ranks and with any thread count. Set `random_seed`, or read it from the log.
- **Bounded 2-hop sampling.** When the 2-hop search reaches a neighbor with a long citer list, it now samples random positions instead of copying the whole list first. The sample is still uniform without replacement over unvisited papers, but per-agent cost no longer grows with hub size, which was making each simulated year ~30% slower than the previous one at production scale.
- **Snapshots and chained jobs.** At year boundaries rank 0 writes a binary snapshot (one file per column, checksummed, atomically published through a `LATEST` pointer). A job pauses before its Slurm deadline (`SLURM_JOB_END_TIME`), writes a snapshot and exits with code 3; the next job resumes with `--resume auto`. `tools/run_chained.sbatch` automates this. Resuming no longer redraws `num_authors`, no longer re-reads the community file, and continues to the original end year.
- **Faster ingest.** Seed CSVs are parsed in parallel from a memory map; nodes are renumbered contiguously (original IDs are kept for output).
- **Validated against 8.x.** 15 replicates per side on a 490K-paper seed (25 years): out-degree, citation-age and in-degree distributions, intra-cluster share and Gini match; the upper in-degree tail is about 5% thinner and not yet explained. See [Behavior that differs from 8.x](#behavior-that-differs-from-8x).
- **Removed:** cartels and planted agents (`planted_nodes` and `clonal_cartel_agent_file` must be empty), the unused ghost-cell / actor-exchange code, the delta journal, the `legacy` backend and the Eigen dependency.
- **Validation tools.** `run_summary` and `tools/compare_runs.py` compare degree distributions, citation ages and other macroscopic properties between groups of runs.

### 8.0.0
- LadyBugDB side store, MPI work partitioning by Leiden cluster, wall-clock supervisor.

### 7.0.4
- Fixes current year nodes leaking into the alpha intra-cluster citations.

### 7.0.3
- Fixes removal of duplicate edges within cluster.

### 7.0.2
- Fixes alpha value not showing correctly in auxiliary file.

### 7.0.1
- Fixes a memory management issue causing abrupt exit codes.

### 7.0.0
- Allows for cluster-based citations (CantataS).

---

## Architecture

```
            rank 0                                   every rank
  +--------------------------+        +-------------------------------------------+
  | AgentFactory             |        | LadyBugDB (identical copy on each rank)    |
  |  authors (Lotka's law),  |  bcast |  BidirectionalCSR  out- and in-citations   |
  |  generators, weights,    | -----> |  NodeColumns       year, cluster, author,  |
  |  same-year sources       |        |                    fitness, num_authors... |
  +--------------------------+        |  ClusterIndex      members per cluster     |
                                      |  AuthorRegistry    h-index per author      |
                                      +-------------------------------------------+
                                        | simulate this rank's agents (OpenMP)
                                        v
                         all-gather new citations -> apply on every rank -> snapshot
```

Each simulated year is bulk-synchronous:
1. Rank 0 creates the year's agents (authors, generators, preference weights, alpha, out-degree, fitness, same-year sources) from the year's random stream and broadcasts them.
2. Every rank computes this year's preference scores and simulates the agents of its clusters (`DistributedPartitioner` assigns clusters to ranks by size). Agents read the graph as it stood at the start of the year.
3. All new citations are gathered on every rank and applied in a fixed order: new forward rows are appended, new in-edges are merged into the backward CSR, the cluster index and author h-indexes are rebuilt.
4. Ranks agree whether to pause; rank 0 writes snapshots.

Because every agent has its own random stream, the assignment of agents to ranks and threads changes only speed, never results.

### Code map
| Path | Role |
|---|---|
| `includes/ladybug/ladybug_db.h` | LadyBugDB: CSR, node columns, cluster index, author registry, run metadata |
| `src/ladybug/ladybug_csr.cpp` | Bidirectional CSR: build, append rows, merge in-edges |
| `src/ladybug/ladybug_ingest.cpp` | Parallel seed CSV ingest |
| `src/ladybug/ladybug_snapshot.cpp` | Binary snapshots with checksums |
| `src/ladybug/ladybug_export.cpp` | `edgelist.csv` / `nodelist.csv` output |
| `src/ladybug/ladybug_authors.cpp` | Lotka author assignment, h-index |
| `src/agent_factory.cpp` | Rank 0: new agents' attributes, authors, generators |
| `src/neighborhood_search.cpp` | 1-hop / 2-hop neighborhoods, recency bins |
| `src/citation_engine.cpp` | Same-year, scored, uniform and fully random citations |
| `src/abm.cpp` | Yearly loop, checkpoints, export |
| `src/distributed/` | MPI coordination and cluster-to-rank assignment |
| `src/signal_supervisor.cpp` | Deadline and signal handling |

### Memory
Every rank holds the whole graph: about 8 bytes per edge (both directions) plus ~125 bytes per node. With ~19 citations per paper that is roughly 135 GB at 500M nodes and 280 GB at 1B nodes, with peaks about 40% higher while each year's in-edges are merged (~190 GB and ~400 GB). 1B-node runs therefore need 512 GB nodes.

---

## Dependencies

- **C++ compiler** with C++20 support: GCC >= 12 or Clang >= 15
- **OpenMP** >= 4.5
- **CMake** >= 3.23
- **MPI** (optional): Open MPI >= 4 or MPICH. Without MPI the build runs single-process.
- Header libraries installed by `setup.sh` into `external_libs/`: **PCG Random**, **inih**, **argparse**

---

## Build and test

```bash
./setup.sh                      # once: fetch header libraries into external_libs/
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

On the Illinois Campus Cluster load `gcc/13.3.0` and `openmpi/5.0.1-gcc-13.3.0` first. The binaries are `build/bin/abm` and `build/bin/run_summary`.

---

## Running

```bash
# Single process (threads = num_processors in the config)
./build/bin/abm --config docs/default_experiment.ini

# Several nodes, one rank per node
mpirun -n 4 --map-by ppr:1:node --bind-to none ./build/bin/abm --config config.ini
```

| Argument | Default | Description |
|---|---|---|
| `--config` | required | `.ini` configuration file |
| `--resume` | `config` | `auto`: resume if `checkpoint_dir` has a snapshot, else start; `yes`: must resume; `no`: start fresh; `config`: follow `start_from_checkpoint` |
| `--seed` | | Overrides `random_seed` |
| `--db-path`, `--storage-backend` | | Accepted for old job scripts; ignored |

Exit codes: **0** finished (CSV output written), **3** paused (snapshot written; run again with `--resume auto`), **1** error.

### Long runs: chained jobs
Copy `tools/run_chained.sbatch` into the run directory, set `ABM`, the `#SBATCH` resources and `num_processors`, and submit it once. Each job resumes from the newest snapshot, pauses before its time limit, and submits the next job until the simulation finishes. The deadline is read from `SLURM_JOB_END_TIME` (or `CANTATAS_DEADLINE_EPOCH`, or `wall_clock_limit`); before each year the run continues only if the remaining time covers max(`safety_margin_seconds`, 1.3 x the last year) plus the time to write a snapshot. `SIGTERM`, `SIGINT` and `SIGUSR1` request a pause at the next year boundary.

Keep `checkpoint_dir` on `/scratch` or `/projects`; node-local storage is wiped when a job ends.

---

## Configuration reference

Existing configuration files keep working. Example: [`docs/default_experiment.ini`](docs/default_experiment.ini).

```ini
[Environment]
nodelist=/path/to/seed_nodelist.csv                 ; node_id,year
edgelist=/path/to/seed_edgelist.csv                 ; source,target
community_assignment=/path/to/clusters.csv          ; node_id,cluster_id (optional)
recency_table=/path/to/recency_counts.csv           ; year difference,count
out_degree_bag=/path/to/out_degree_bag.csv          ; second column = out-degree
growth_rate=0.05                                    ; new papers per year, as a fraction
num_cycles=33                                       ; simulated years (counted from the seed)
recency_bins=1,2,3,4,5,10,20,30
planted_nodes=                                      ; must be empty (not supported)
start_from_checkpoint=false                         ; used by --resume config

[Agent]
alpha=-1                    ; share of neighborhood citations from distance 1 (-1: random per agent)
use_alpha=true
theta=50                    ; minimum cluster size for cluster-based citations
same_year_citations=0.12
fully_random_citations=0.05
preferential_weight=-1      ; -1 for all four weights: random per agent
fitness_weight=-1
num_authors_weight=-1
author_reputation_weight=-1
fitness_value_min=1
fitness_value_max=1000
fitness_lag_duration_min=0
fitness_lag_duration_max=0
fitness_peak_duration_min=1000
fitness_peak_duration_max=1000
neighborhood_sample=10000   ; cap per distance (-1: no cap)
in_degree_threshold=5
fitness_threshold=20
recency_threshold=10
non_random_generator_probability=0.0
author_max_lifetime=30
num_authors_bag=/path/to/num_authors.csv
clonal_cartel_agent_file=   ; must be empty (not supported)
attachment_exponent=3       ; new, optional: exponent applied to all four score components

[General]
output_file=./output/edgelist.csv
auxiliary_information_file=./output/nodelist.csv
log_file=./output/log.txt   ; ranks write log.txt_rank_N; timings go to <log>_timing
num_processors=32           ; OpenMP threads per rank
log_level=1                 ; 1 also writes the timing CSV
checkpoint_dir=             ; new: snapshot directory (default <output dir>/checkpoints)
checkpoint_interval_minutes=120 ; new: periodic snapshots; 0 = only when pausing
checkpoints_to_keep=2       ; new
random_seed=0               ; new: 0 = draw one (logged)
verify_snapshots=true       ; new: check checksums when resuming
wall_clock_limit=0          ; seconds after start; used only if Slurm gives no end time
safety_margin_seconds=1200  ; new
```

Ignored keys kept for compatibility: `storage_backend`, `db_path`, `use_distributed`, `ghost_hub_threshold`, `cartel_outdegree_proportion`, `null_cartel`.

---

## Output

At the end, rank 0 writes:
1. **`edgelist.csv`**: `source,target`, original node IDs (agents continue after the largest seed ID).
2. **`nodelist.csv`**: `node_id,type,year,alpha,pa_weight,fit_weight,num_authors_weight,author_reputation_weight,fit_lag_duration,fit_peak_value,fit_peak_duration,in_degree,out_degree,assigned_out_degree,planted_nodes_line_number,generator_node_string,sampled_neighborhood_size,fully_random_citations,author_id,num_authors,initial_author_reputation,final_author_reputation,cartel_id,cluster_id` (same columns as 8.x).
3. **`log.txt`** and **`log.txt_timing`** (`year,label,value`): per-stage wall time in ms, per-thread stage sums inside the agent loop, citation counts, cycle seconds and memory (`VmRSS`, `VmHWM`).

---

## Validating model changes

```bash
build/bin/run_summary runs/old/rep1          # writes runs/old/rep1/summary.json
python3 tools/compare_runs.py --baseline runs/old/rep*/summary.json --candidate runs/new/rep*/summary.json
```

`compare_runs.py` checks scalar metrics (mean out-degree, intra-cluster share, same-year share, citation age, top in-degrees, in-degree Gini and quantiles) against replicate variation, and compares in-degree, out-degree and citation-age distributions with the KS distance relative to the spread between baseline replicates.

### Behavior that differs from 8.x

Runs are not bit-identical to 8.x (different random streams), but they are meant to match it statistically.

**Known, unexplained difference.** The upper in-degree tail is thinner: in-degree p99.9 is 1,347 against 1,421 for 8.x (15 replicates each, z = -5.7), with about 5% fewer papers at 500-3,000 citations and slightly more at 30-200; it builds up over the first simulated years. Whole-distribution KS distances are 5-6 times smaller than the spread between 8.x replicates. The 8.x branch this code was ported from (17a41a6) matches 8.x, so the cause is in this rewrite. Ruled out so far: the scoring state (every node's in-degree, fitness, author count and author h-index at 1990 checked exactly against the outputs), per-agent random streams (per-thread streams give the same tail), candidate list order (shuffled lists give the same tail), the faster 2-hop sampling (`CANTATAS_EXACT_2HOP=1` gives the same tail), seed author order, and the alias sampler's in-order fallback (0.1% of picks).

These are the deliberate differences:

- **Seed author order.** Seed papers receive authors (Lotka's law) year by year; within a year the order is now a seeded shuffle. The order matters because an author who gets a second paper in the same year gets one from later in the sequence, so an order that tracks citation counts changes which papers share an author. 8.x used node-list order passed through an unstable sort, which behaves like a shuffle. Ascending original ID (tried during the rewrite) tracks citation counts and gave seed authors lower h-indices than 8.x; the shuffle matches 8.x.
- **`use_alpha=false`** treats distances 1 and 2 as one pool, as the first version of the model did. 8.x passed `alpha=-1` into the distance split, got a negative quota, and so made no neighborhood citations at all in that mode (everything fell through to uniform random citations).
- **`non_random_generator_probability > 0` with no eligible paper** falls back to a uniform generator instead of giving the agent no generator.
- **Model changes on resume** are refused unless `--allow-model-change` is passed (the snapshot stores a fingerprint of the model parameters).
- **`initial_author_reputation` of seeds** (output only) is the h-index right after the paper's year cohort; 8.x also counted the first paper of the next year when it went to the same author.
- Unchanged from 8.x: node-list papers without any citation edge are not added to the graph but still take part in author assignment.

`CANTATAS_EXACT_2HOP=1` forces the 8.x method for sampling long 2-hop neighbor lists (copy the list, then `std::sample`). It is a diagnostic switch for checking that the faster method does not change results.

---

## Citation Methodology

Each simulated paper fulfills its assigned out-degree quota across sequential phases:

1. **Quota Reservation**: Portions of the citation quota are reserved for same-year papers (`same_year_citations`) and globally uniform random citations (`fully_random_citations`).
2. **Primary Citations (1-Hop vs. Intra-Cluster)**: The `alpha` parameter governs primary citations. If the paper's community cluster has at least `theta` members, citations target nodes within that cluster. Otherwise, citations target the 1-hop neighborhood of the generator node.
3. **Secondary Citations (2-Hop)**: Remaining neighborhood quota is fulfilled by sampling from 2-hop neighbor candidates (outside the cluster when cluster citations are used).
4. **Deficit Compensation**: Any remaining deficit is fulfilled by uniform random selection from the global network.

Within each recency bin, candidates are drawn with probability proportional to the agent's weighted combination of preferential attachment, fitness, number of authors and author reputation (each raised to `attachment_exponent`), using Vose's alias method.

---

## Generating Community Clusters (Leiden Algorithm)

```bash
python leiden_clustering.py \
  --nodelist /path/to/seed_nodelist.csv \
  --edgelist /path/to/seed_edgelist.csv \
  --output /path/to/seed_field_labels.csv \
  --resolution 0.01 \
  --objective CPM
```

Point `community_assignment` in `[Environment]` to the generated output CSV to enable cluster-aware simulation dynamics.

---

## Citation & References

If you use CantataS in your research, please cite:

1. Chacko, George, Minhyuk Park, Vikram Ramavarapu, Ananth Grama, Pablo Robles-Granda, and Tandy Warnow. *"An agent-based model of citation behavior."* Applied Network Science 11, no. 1 (2026): 10.
