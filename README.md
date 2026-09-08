# Hyperparameter Optimization for EEG Sleep-Stage Classification

Ten hyperparameter optimization algorithms, all searching the same CNN+LSTM
architecture on the same two EEG sleep datasets under the same evaluation
budget, so their results can be compared directly.

The model classifies each 30-second EEG epoch into one of five sleep stages
(W, N1, N2, N3, REM). The point of the repository is not the classifier — it is
the comparison between search strategies.

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [Model](#model)
- [Search Space and Evaluation Protocol](#search-space-and-evaluation-protocol)
- [The Ten Optimizers](#the-ten-optimizers)
- [Results](#results)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

Every optimizer searches the same six hyperparameters, spends the same ~100
model evaluations per dataset, scores candidates on the same held-out
validation split, and writes its results in the same schema. Each run records
validation loss plus macro-averaged accuracy, precision, recall and F1.

Selection is on `val_loss`. The other four metrics are recorded but not
optimized against — worth remembering when reading the results, since the two
criteria disagree substantially (see [Results](#results)).

## Datasets

Two datasets, deliberately different in recording setup and scoring convention:

| Dataset | Channels | Scoring | Source |
|---|---|---|---|
| Sleep-EDF Expanded (sleep-cassette) | `EEG Fpz-Cz`, `EEG Pz-Oz` | R&K | [physionet.org/content/sleep-edfx](https://physionet.org/content/sleep-edfx/1.0.0/) |
| Haaglanden Medisch Centrum (HMC) | `EEG C4-M1`, `EEG O2-M1` | AASM | PhysioNet HMC sleep-staging database |

Both are mapped onto the same five classes; R&K stages 3 and 4 are merged into
N3. Epochs with no usable stage (`Sleep stage ?`, movement time, lights on/off
markers) are excluded rather than silently shifted, since annotations are placed
by onset rather than appended in order.

Point the loaders at your local copies via `.env`:

```bash
cp .env.example .env
```

```ini
DATA_HAANGLANDEN='.../datasets/Haaglanden/recordings'
DATA_SLEEP_EDFX='.../datasets/EDF/sleep-cassette'
```

Both loaders fail with an explicit error if the variable is unset or does not
point at a directory.

## Installation

Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

## Usage

**1. Build the dataset cache once.** Decoding the EDF files takes hours; every
optimizer is a separate process, so without a cache each one would repeat that
work.

```bash
python -c "from data.global_data_loader import get_data_all_datasets; get_data_all_datasets()"
```

This writes `data/cache/haaglanden_8000_42.npz` and
`data/cache/sleep_edfx_8000_42.npz` (~190 MB each). Delete them to force a
re-decode — the cache key is only `(name, n_samples, random_state)` and does
**not** track channel selection or preprocessing, so a loader change requires
clearing it by hand.

**2. Run the optimizers.**

```bash
cd "optimization scripts"

python run_missing.py --jobs 3          # only what lacks results, 3 at a time
python run_missing.py --dry-run         # show the plan, run nothing
python run_missing.py --only pso_optimizer.py
python run_all.py                       # all ten, serially
```

`run_missing.py` treats a script as done only when **both** of its dataset CSVs
exist — a script can be killed after finishing one dataset and still exit 0. It
refuses to start if the cache is missing, tees each script to
`results/logs/<script>.log`, and writes `RUN_MANIFEST.md` with a per-script
COMPLETE / FAILED / PARTIAL table.

Run only one instance at a time. Two concurrent runners will both queue the same
outstanding scripts and race on the same output files.

**3. Plot the comparison.**

```bash
python visualizations/visualizations.py
```

Writes one dashboard per dataset to `visualizations/dashboards/`: validation-loss
ECDF by method, best macro F1 per method, random-forest feature importance, and
per-parameter correlation with loss.

## Data Pipeline

`data/edf_common.py` loads both datasets in two passes:

1. **Scan** — read only the EDF *header* for recording length and the annotation
   file for labels. No signal is decoded.
2. **Select** — pick a class-balanced subset of epochs across all recordings.
3. **Decode** — read only the files that contributed selected epochs, keeping
   only those epochs.

Signals are resampled to 100 Hz, cut into 30 s epochs (3000 samples), and
z-scored per epoch and channel. The default subset is 8000 epochs per dataset at
seed 42, giving `X` of shape `(8000, 2, 3000)`.

The earlier implementation decoded every recording and materialised the whole
dataset (~9 GB for sleep-cassette) before subsampling, which exceeded this
machine's commit limit.

## Model

`model/model_builder.py` — a CNN+LSTM classifier:

```
Conv1D(filters, kernel_size, stride 2, ReLU) → BatchNorm → MaxPool(2) → Dropout
Conv1D(filters, kernel_size, stride 2, ReLU) → BatchNorm → MaxPool(2) → Dropout
LSTM(lstm_units) → Dense(32, ReLU) → Dropout → Dense(5, softmax)
```

Adam, sparse categorical cross-entropy. Each evaluation trains at most 4 epochs
with `EarlyStopping(patience=2, restore_best_weights=True)`.

Because the callback restores the best-`val_loss` epoch's weights before `fit`
returns — whether or not it stopped early — a single `predict` after training
yields accuracy, precision, recall and F1 that all describe that same epoch,
rather than mixing `min(val_loss)` with a `max(val_accuracy)` from a different
one. Precision, recall and F1 are macro-averaged so that N1, the rarest and
hardest stage, is not drowned out by W and N2, and use `zero_division=0` so a
model that never predicts some stage yields 0 rather than NaN.

## Search Space and Evaluation Protocol

Seven optimizers search this discrete grid:

| Parameter | Values |
|---|---|
| `filters` | 16, 32, 64, 96, 128 |
| `kernel_size` | 2, 3, 4, 5 |
| `lstm_units` | 32, 64, 96, 128 |
| `dropout` | 0.1, 0.2, 0.3, 0.4, 0.5 |
| `learning_rate` | 1e-4, 5e-4, 1e-3, 5e-3, 1e-2 |
| `batch_size` | 16, 32, 48, 64 |

Grid search takes a coarser 96-point subset of it, every value drawn from the
same set. Bayesian optimization and PSO instead work on the continuous box
spanning the same ranges (`filters` 16–128, `kernel_size` 2–5, `lstm_units`
32–128, `dropout` 0.1–0.5, `learning_rate` 1e-4–1e-2, `batch_size` 16–64),
truncating to integers where required — so their best configurations can report
values off the grid, such as `filters=102, lstm_units=60, batch_size=23`.

**Splitting.** Every optimizer applies the same nested split at
`random_state=42`: 80/20 into train+val and test, then 80/20 within the first
part — 64% train, 16% validation, 20% test. Search is scored on the validation
split only. The test split is reserved and currently untouched by any script.

## The Ten Optimizers

All in `optimization scripts/`, all at ~100 evaluations per dataset.

| Script | Method | Budget | Key settings |
|---|---|---|---|
| `grid_search.py` | Grid search | 96 | exhaustive over a 2×2×2×2×3×2 grid |
| `random_search.py` | Random search | 100 | uniform over the grid |
| `hill_climbing.py` | Hill climbing | 100 | random restarts, ±1 step per parameter |
| `simulated_annealing.py` | Simulated annealing | 100 | T 1.0, cooling 0.95, T_min 1e-3 |
| `hill_climbing_with_simulated_annealing.py` | Basin hopping | 100 | inner hill climb, 2 dims kicked by 3 steps |
| `tabu_search.py` | Tabu search | 100 | tenure 10 |
| `genetic_algorithm.py` | Genetic algorithm | 100 | pop 10 × 10 gens, tournament 3, elitism 1, mutation 0.2 |
| `ant_colony_optimization.py` | Ant colony | 100 | 10 ants × 10 iters, evaporation 0.3, alpha 1.0 |
| `pso_optimizer.py` | Particle swarm | 100 | 10 particles × 10 iters, c1 0.5, c2 0.3, w 0.9 |
| `bayesian_optimization_algorithm.py` | Bayesian optimization | 100 | 20 random probes + 80 GP-guided |

Each writes `results/<method>_results_<dataset>.csv` with one row per evaluation,
appends its best row to the shared `results/best_results.csv`, and updates a
convergence plot in `optimization scripts/convergence plots/` after every
evaluation. ACO and the GA also emit a pheromone / population heatmap.

Because all ten append to one summary file whose header is written once,
`model/metrics.py` defines the metric columns and their order in a single place;
`optimization scripts/results_io.py` serializes the appends behind a file lock so
concurrent runs cannot interleave mid-row.

## Results

From the run committed in `results/` — 1992 evaluations, ~100 per method per
dataset. Best configuration per method, selected on `val_loss`:

**Sleep-EDF**

| Method | val_loss | accuracy | precision | recall | macro F1 |
|---|---|---|---|---|---|
| GA | 0.7847 | 0.6586 | 0.6492 | 0.6440 | 0.6301 |
| Hill Climbing | 0.7886 | 0.6711 | 0.6507 | 0.6557 | 0.6422 |
| ACO | 0.7922 | 0.6664 | 0.6663 | 0.6549 | 0.6549 |
| Tabu Search | 0.7958 | 0.6617 | 0.6485 | 0.6471 | 0.6386 |
| Basin Hopping | 0.7985 | 0.6664 | 0.6525 | 0.6511 | 0.6401 |
| Simulated Annealing | 0.8066 | 0.6687 | 0.6660 | 0.6589 | **0.6606** |
| Bayesian Optimization | 0.8189 | 0.6641 | 0.6393 | 0.6486 | 0.6352 |
| Random Search | 0.8526 | 0.6445 | 0.6408 | 0.6306 | 0.6210 |
| Grid Search | 0.8653 | 0.6242 | 0.6106 | 0.6071 | 0.5929 |
| PSO | 0.9272 | 0.5992 | 0.5841 | 0.5880 | 0.5847 |

**Haaglanden**

| Method | val_loss | accuracy | precision | recall | macro F1 |
|---|---|---|---|---|---|
| Basin Hopping | 1.0625 | 0.5430 | 0.5008 | 0.5361 | 0.4973 |
| Tabu Search | 1.0629 | 0.5312 | 0.5070 | 0.5291 | **0.5136** |
| Hill Climbing | 1.0644 | 0.5508 | 0.5141 | 0.5435 | 0.5127 |
| GA | 1.0748 | 0.5523 | 0.5231 | 0.5462 | 0.5119 |
| Simulated Annealing | 1.1044 | 0.5258 | 0.5019 | 0.5157 | 0.4880 |
| PSO | 1.1077 | 0.5188 | 0.4680 | 0.5145 | 0.4669 |
| ACO | 1.1366 | 0.5117 | 0.4978 | 0.5059 | 0.4756 |
| Grid Search | 1.1437 | 0.5078 | 0.4629 | 0.5043 | 0.4592 |
| Bayesian Optimization | 1.1513 | 0.5023 | 0.4946 | 0.5018 | 0.4896 |
| Random Search | 1.1573 | 0.5047 | 0.4801 | 0.5062 | 0.4827 |

Three things stand out:

**Haaglanden is the harder dataset** — median loss 1.303 against 1.017, best
1.062 against 0.785. Consistent across all ten methods.

**Loss and macro F1 rank the methods differently.** Simulated annealing is 6th by
loss on Sleep-EDF but 1st by F1; Bayesian optimization is 9th by loss on
Haaglanden but 5th by F1. Spearman correlation between the two rankings is −0.60
on Sleep-EDF and −0.69 on Haaglanden. Selecting on loss does not select for
balanced per-stage performance.

**`batch_size` dominates** — 0.44 relative importance on Sleep-EDF and 0.36 on
Haaglanden, against 0.27–0.28 for `learning_rate`, and correlating +0.62 / +0.55
with loss. Over half the best configurations sit at `batch_size=16`, the low end
of the range, so the optimum may lie below the search space.

> **Caveat on `execution_time`.** During the committed run up to six scripts ran
> concurrently rather than three, and contention varied over 24 hours. The column
> records real wall-clock time but is not a fair basis for comparing methods —
> the same tabu search took 5698 s in one instance and 9263 s in another.

## Project Structure

```
data/
  edf_common.py              two-pass EDF loading shared by both datasets
  data_loader_sleep_edfx.py  Sleep-EDF pairing and channel selection
  data_loader_haanglanden.py Haaglanden pairing and channel selection
  global_data_loader.py      .npz caching across optimizer processes
  subsample.py               class-balanced subsampling
  cache/                     generated, gitignored
model/
  model_builder.py           CNN+LSTM, and the evaluation returning all metrics
  metrics.py                 the metric columns and their order
optimization scripts/
  <ten optimizer scripts>
  results_io.py              lock-protected append to best_results.csv
  run_all.py                 run all ten serially
  run_missing.py             run only what is missing, in parallel, + manifest
  convergence plots/         one plot per method per dataset
results/
  <method>_results_<dataset>.csv   one row per evaluation
  best_results.csv                 best configuration per method per dataset
  logs/                            generated, gitignored
visualizations/
  visualizations.py          cross-method dashboards
  dashboards/                generated output
```

## License

Eclipse Public License 2.0 — see [LICENSE](LICENSE).
