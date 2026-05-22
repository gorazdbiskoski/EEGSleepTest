import os
import time
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
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

VIZ_DIR = os.path.join(os.path.dirname(__file__), 'convergence plots')
os.makedirs(VIZ_DIR, exist_ok=True)

fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
ax_conv.set_title('Hill Climbing (Random Restart) - Convergence')
ax_conv.set_xlabel('Evaluation #')
ax_conv.set_ylabel('Best val_loss')
conv_x, conv_y = [], []
(conv_line,) = ax_conv.plot([], [], marker='o', color='indigo')


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


def get_neighbors(individual):
    neighbors = []
    for param, values in PARAM_SPACE.items():
        current_idx = values.index(individual[param])
        for step in (-1, 1):
            new_idx = current_idx + step
            if 0 <= new_idx < len(values):
                neighbor = individual.copy()
                neighbor[param] = values[new_idx]
                neighbors.append(neighbor)
    return neighbors


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


def log_trial(individual, loss, restart_id):
    results_log.append({
        "filters": individual["filters"],
        "kernel_size": individual["kernel_size"],
        "lstm_units": individual["lstm_units"],
        "dropout": individual["dropout"],
        "learning_rate": individual["learning_rate"],
        "batch_size": individual["batch_size"],
        "val_loss": loss,
        "restart_id": restart_id,
        "method": "Hill Climbing"
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
        "method":         "Hill Climbing",
        "execution_time": round(elapsed, 4),
    }])
    write_header = not os.path.exists(summary_path)
    row.to_csv(summary_path, mode='a', header=write_header, index=False)


if __name__ == "__main__":
    start_time = time.perf_counter()

    evals_used = 0
    best_loss = float("inf")
    best_solution = None
    restart_id = 0

    while evals_used < MAX_EVALS:
        restart_id += 1

        current = random_individual()
        current_loss = evaluate_individual(current)
        evals_used += 1
        log_trial(current, current_loss, restart_id)

        if current_loss < best_loss:
            best_loss = current_loss
            best_solution = current.copy()

        _update_convergence(evals_used, best_loss)

        print(
            f"Restart {restart_id} (eval {evals_used}/{MAX_EVALS}) | "
            f"Start loss: {current_loss:.6f} | Best: {best_loss:.6f}"
        )

        # Inner first-improvement HC loop
        while evals_used < MAX_EVALS:
            neighbors = get_neighbors(current)
            random.shuffle(neighbors)

            improved = False
            for neighbor in neighbors:
                if evals_used >= MAX_EVALS:
                    break

                n_loss = evaluate_individual(neighbor)
                evals_used += 1
                log_trial(neighbor, n_loss, restart_id)

                if n_loss < best_loss:
                    best_loss = n_loss
                    best_solution = neighbor.copy()

                _update_convergence(evals_used, best_loss)

                if n_loss < current_loss:
                    current = neighbor
                    current_loss = n_loss
                    improved = True
                    print(
                        f"  HC step (eval {evals_used}/{MAX_EVALS}) | "
                        f"Loss: {current_loss:.6f} | Best: {best_loss:.6f}"
                    )
                    break

            if not improved:
                print(f"  Local min reached at loss {current_loss:.6f}")
                break

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
    print(f"Total Restarts: {restart_id}")
    print(f"Execution Time: {elapsed:.2f} seconds")
    print(f"Visualisations saved to {VIZ_DIR}")
