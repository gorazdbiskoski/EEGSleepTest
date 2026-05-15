import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from model.data_loader import load_data
from model.model_builder import evaluate_model

X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results_log = []

PARAM_SPACE = {
    "filters": [16, 32, 64, 96, 128],
    "kernel_size": [2, 3, 4, 5],
    "lstm_units": [32, 64, 96, 128],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "batch_size": [16, 32, 48, 64]
}

pheromones = {
    key: np.ones(len(values))
    for key, values in PARAM_SPACE.items()
}

NUM_ANTS = 10
NUM_ITERATIONS = 10
EVAPORATION_RATE = 0.3
ALPHA = 1.0
Q = 1.0

# ── visualisation setup ────────────────────────────────────────────────────────
VIZ_DIR = os.path.join(os.path.dirname(__file__), 'convergence plots')
os.makedirs(VIZ_DIR, exist_ok=True)

fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
ax_conv.set_title('ACO – Convergence')
ax_conv.set_xlabel('Iteration')
ax_conv.set_ylabel('Best val_loss')
conv_x, conv_y = [], []
(conv_line,) = ax_conv.plot([], [], marker='o', color='royalblue')

param_names = list(PARAM_SPACE.keys())
_max_len = max(len(PARAM_SPACE[p]) for p in param_names)
_heat_matrix = np.full((len(param_names), _max_len), np.nan)
fig_heat, ax_heat = plt.subplots(figsize=(7, 4))
_im = ax_heat.imshow(_heat_matrix, aspect='auto', cmap='YlOrRd',
                     vmin=0, vmax=1, interpolation='nearest')
ax_heat.set_yticks(range(len(param_names)))
ax_heat.set_yticklabels(param_names)
ax_heat.set_xlabel('Parameter value index')
fig_heat.colorbar(_im, ax=ax_heat, label='Normalised pheromone')
fig_heat.tight_layout()


def _update_convergence(iteration, best_loss):
    conv_x.append(iteration)
    conv_y.append(best_loss)
    conv_line.set_data(conv_x, conv_y)
    ax_conv.relim()
    ax_conv.autoscale_view()
    fig_conv.savefig(os.path.join(VIZ_DIR, 'aco_convergence.png'), dpi=100)


def _update_pheromone_heatmap(iteration):
    for i, p in enumerate(param_names):
        vals = pheromones[p]
        _heat_matrix[i, :] = np.nan
        _heat_matrix[i, :len(vals)] = vals / vals.sum()
    _im.set_data(_heat_matrix)
    ax_heat.set_title(f'ACO – Pheromone heatmap (iter {iteration})')
    fig_heat.savefig(os.path.join(VIZ_DIR, 'aco_pheromone_heatmap.png'), dpi=100)
# ──────────────────────────────────────────────────────────────────────────────


def choose_parameter(param_name):
    pheromone = pheromones[param_name]
    probabilities = pheromone ** ALPHA
    probabilities = probabilities / probabilities.sum()
    index = np.random.choice(range(len(PARAM_SPACE[param_name])), p=probabilities)
    return PARAM_SPACE[param_name][index], index


def construct_solution():
    solution, indices = {}, {}
    for param in PARAM_SPACE:
        value, idx = choose_parameter(param)
        solution[param] = value
        indices[param] = idx
    return solution, indices


def evaluate_solution(solution):
    params = [
        int(solution["filters"]),
        int(solution["kernel_size"]),
        int(solution["lstm_units"]),
        float(solution["dropout"]),
        float(solution["learning_rate"]),
        int(solution["batch_size"])
    ]
    return evaluate_model(params, X_train, X_test, y_train, y_test)


def append_best_to_summary(best_solution, best_loss, elapsed):
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, 'best_results.csv')
    row = pd.DataFrame([{
        "filters":        best_solution["filters"],
        "kernel_size":    best_solution["kernel_size"],
        "lstm_units":     best_solution["lstm_units"],
        "dropout":        best_solution["dropout"],
        "learning_rate":  best_solution["learning_rate"],
        "batch_size":     best_solution["batch_size"],
        "val_loss":       best_loss,
        "method":         "ACO",
        "execution_time": round(elapsed, 4),
    }])
    write_header = not os.path.exists(summary_path)
    row.to_csv(summary_path, mode='a', header=write_header, index=False)


if __name__ == "__main__":
    best_loss = float("inf")
    best_solution = None

    start_time = time.perf_counter()

    for iteration in range(NUM_ITERATIONS):
        all_solutions = []

        for ant in range(NUM_ANTS):
            solution, indices = construct_solution()
            loss = evaluate_solution(solution)

            results_log.append({
                "filters": solution["filters"],
                "kernel_size": solution["kernel_size"],
                "lstm_units": solution["lstm_units"],
                "dropout": solution["dropout"],
                "learning_rate": solution["learning_rate"],
                "batch_size": solution["batch_size"],
                "val_loss": loss,
                "method": "ACO"
            })

            all_solutions.append((solution, indices, loss))

            if loss < best_loss:
                best_loss = loss
                best_solution = solution

        for param in pheromones:
            pheromones[param] *= (1 - EVAPORATION_RATE)

        for solution, indices, loss in all_solutions:
            for param in PARAM_SPACE:
                idx = indices[param]
                pheromones[param][idx] += Q / (loss + 1e-8)

        _update_convergence(iteration + 1, best_loss)
        _update_pheromone_heatmap(iteration + 1)

    elapsed = time.perf_counter() - start_time

    df = pd.DataFrame(results_log)
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, 'aco_results.csv'), index=False)

    append_best_to_summary(best_solution, best_loss, elapsed)

    plt.close('all')
    print(f"Visualisations saved to {VIZ_DIR}")