import os
import time
import math
import random

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

MAX_EVALS = 100
T_START = 1.0
COOLING_RATE = 0.95
T_MIN = 1e-3
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

VIZ_DIR = os.path.join(os.path.dirname(__file__), 'convergence plots')
os.makedirs(VIZ_DIR, exist_ok=True)

fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
ax_conv.set_title('Simulated Annealing - Convergence')
ax_conv.set_xlabel('Evaluation #')
ax_conv.set_ylabel('Best val_loss')
conv_x, conv_y = [], []
(conv_line,) = ax_conv.plot([], [], marker='o', color='teal')


def _update_convergence(eval_num, best_loss):
    conv_x.append(eval_num)
    conv_y.append(best_loss)
    conv_line.set_data(conv_x, conv_y)
    ax_conv.relim()
    ax_conv.autoscale_view()
    fig_conv.tight_layout()
    fig_conv.savefig(
        os.path.join(VIZ_DIR, 'hill_climbing_convergence.png'),
        dpi=100
    )


def random_individual():
    return {
        param: random.choice(values)
        for param, values in PARAM_SPACE.items()
    }


def random_neighbor(individual):
    neighbor = individual.copy()
    param = random.choice(list(PARAM_SPACE.keys()))
    values = PARAM_SPACE[param]

    current_idx = values.index(individual[param])
    step = random.choice([-1, 1])
    new_idx = current_idx + step

    if new_idx < 0:
        new_idx = 1
    elif new_idx >= len(values):
        new_idx = len(values) - 2

    neighbor[param] = values[new_idx]
    return neighbor


def evaluate_individual(individual):
    params = [
        int(individual["filters"]),
        int(individual["kernel_size"]),
        int(individual["lstm_units"]),
        float(individual["dropout"]),
        float(individual["learning_rate"]),
        int(individual["batch_size"])
    ]
    return evaluate_model(params, X_train, X_test, y_train, y_test)


def log_trial(individual, loss):
    results_log.append({
        "filters": individual["filters"],
        "kernel_size": individual["kernel_size"],
        "lstm_units": individual["lstm_units"],
        "dropout": individual["dropout"],
        "learning_rate": individual["learning_rate"],
        "batch_size": individual["batch_size"],
        "val_loss": loss,
        "method": "Simulated Annealing"
    })


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
        "method":         "Simulated Annealing",
        "execution_time": round(elapsed, 4),
    }])
    write_header = not os.path.exists(summary_path)
    row.to_csv(summary_path, mode='a', header=write_header, index=False)


if __name__ == "__main__":
    start_time = time.perf_counter()

    current = random_individual()
    current_loss = evaluate_individual(current)
    log_trial(current, current_loss)

    best_solution = current.copy()
    best_loss = current_loss

    _update_convergence(1, best_loss)
    print(f"Eval 1/{MAX_EVALS} | T={T_START:.4f} | Best Loss: {best_loss:.6f}")

    T = T_START

    for step in range(2, MAX_EVALS + 1):
        if T < T_MIN:
            print(f"Temperature below T_MIN ({T_MIN}), stopping early at eval {step - 1}.")
            break

        candidate = random_neighbor(current)
        candidate_loss = evaluate_individual(candidate)
        log_trial(candidate, candidate_loss)

        delta = candidate_loss - current_loss

        if delta < 0 or random.random() < math.exp(-delta / T):
            current = candidate
            current_loss = candidate_loss

            if current_loss < best_loss:
                best_loss = current_loss
                best_solution = current.copy()

        T *= COOLING_RATE

        _update_convergence(step, best_loss)

        print(
            f"Eval {step}/{MAX_EVALS} | T={T:.4f} | "
            f"Candidate Loss: {candidate_loss:.6f} | Best Loss: {best_loss:.6f}"
        )

    elapsed = time.perf_counter() - start_time

    df = pd.DataFrame(results_log)
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(
        os.path.join(results_dir, 'hill_climbing_results.csv'),
        index=False
    )

    append_best_to_summary(best_solution, best_loss, elapsed)

    plt.close('all')

    print(f"\nBest Solution: {best_solution}")
    print(f"Best Validation Loss: {best_loss:.6f}")
    print(f"Final Temperature: {T:.6f}")
    print(f"Execution Time: {elapsed:.2f} seconds")
    print(f"Visualisations saved to {VIZ_DIR}")
