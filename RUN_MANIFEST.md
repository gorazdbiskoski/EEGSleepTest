# Optimizer Run Manifest

Generated **2026-09-07 15:44:13** - total wall clock **26.23h**, `--jobs 3`, 6 threads/worker.

A script counts as complete only when **both** dataset CSVs exist (`*_haaglanden.csv` and `*_sleep_edf.csv`). Exit code alone is not sufficient: a script can be killed after finishing one dataset and still report success.

| Status | Meaning |
|---|---|
| `COMPLETE` | exit 0 and both CSVs present |
| `FAILED` | non-zero exit code |
| `PARTIAL` | exit 0 but fewer than 2 CSVs - silent failure |
| `SKIPPED` | already had both CSVs before this run |

> **All scripts that ran completed successfully.**

## Status

| Script | Status | Exit | Duration | CSVs | Log |
|---|---|---|---|---|---|
| `ant_colony_optimization.py` | **COMPLETE** | 0 | 8.96h | 2/2 | `results/logs/ant_colony_optimization.log` |
| `bayesian_optimization_algorithm.py` | **COMPLETE** | 0 | 10.01h | 2/2 | `results/logs/bayesian_optimization_algorithm.log` |
| `genetic_algorithm.py` | **COMPLETE** | 0 | 9.88h | 2/2 | `results/logs/genetic_algorithm.log` |
| `grid_search.py` | **COMPLETE** | 0 | 2.94h | 2/2 | `results/logs/grid_search.log` |
| `hill_climbing.py` | **COMPLETE** | 0 | 6.34h | 2/2 | `results/logs/hill_climbing.log` |
| `hill_climbing_with_simulated_annealing.py` | **COMPLETE** | 0 | 7.35h | 2/2 | `results/logs/hill_climbing_with_simulated_annealing.log` |
| `pso_optimizer.py` | **COMPLETE** | 0 | 7.10h | 2/2 | `results/logs/pso_optimizer.log` |
| `random_search.py` | **COMPLETE** | 0 | 7.38h | 2/2 | `results/logs/random_search.log` |
| `simulated_annealing.py` | **COMPLETE** | 0 | 6.57h | 2/2 | `results/logs/simulated_annealing.log` |
| `tabu_search.py` | **COMPLETE** | 0 | 7.23h | 2/2 | `results/logs/tabu_search.log` |

**10 complete this run | 0 problem(s) | 0 skipped**
