import os
import sys

# Allow running this script directly (outside PyCharm), which otherwise puts
# only this folder on sys.path and cannot import the repo-root `data`/`model`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data.global_data_loader import get_data_all_datasets
from model.metrics import metrics_from
from model.model_builder import evaluate_model
from results_io import append_best_row

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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
VIZ_DIR = os.path.join(os.path.dirname(__file__), 'convergence plots')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)


def run_hill_climbing(dataset_name, X, y):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # First split holds out a test set the search never sees; the second
    # carves off the validation set every trial is scored on. The six other
    # optimizers already did this -- tuning directly against X_test made
    # their val_loss figures incomparable with these.
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    results_log = []
    best_loss = float("inf")
    best_metrics = None
    best_solution = None
    conv_x, conv_y = [], []

    fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
    ax_conv.set_title(f'Hill Climbing (Random Restart) – Convergence ({dataset_name})')
    ax_conv.set_xlabel('Evaluation #')
    ax_conv.set_ylabel('Best val_loss')
    conv_line, = ax_conv.plot([], [], marker='o', color='indigo')

    def update_convergence(eval_num, current_best_loss):
        conv_x.append(eval_num)
        conv_y.append(current_best_loss)
        conv_line.set_data(conv_x, conv_y)
        ax_conv.relim()
        ax_conv.autoscale_view()
        fig_conv.tight_layout()
        filename = f"hill_climbing_convergence_{dataset_name.lower().replace('-', '_')}.png"
        fig_conv.savefig(os.path.join(VIZ_DIR, filename), dpi=100)

    def random_individual():
        return {param: random.choice(values) for param, values in PARAM_SPACE.items()}

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
        return evaluate_model(params, X_train, X_val, y_train, y_val)

    def log_trial(individual, result, restart_id, eval_num):
        results_log.append({
            "dataset": dataset_name,
            "eval": eval_num,
            "restart_id": restart_id,
            "filters": individual["filters"],
            "kernel_size": individual["kernel_size"],
            "lstm_units": individual["lstm_units"],
            "dropout": individual["dropout"],
            "learning_rate": individual["learning_rate"],
            "batch_size": individual["batch_size"],
            "val_loss": result["val_loss"],
            **metrics_from(result),
            "epochs_used": result["epochs"],
            "method": "Hill Climbing"
        })

    start_time = time.perf_counter()
    evals_used = 0
    restart_id = 0

    print(f"Starting Hill Climbing on {dataset_name}. Max evaluations: {MAX_EVALS}")

    while evals_used < MAX_EVALS:
        restart_id += 1
        current = random_individual()
        current_result = evaluate_individual(current)
        current_loss = current_result["val_loss"]
        evals_used += 1
        log_trial(current, current_result, restart_id, evals_used)

        if current_loss < best_loss:
            best_loss = current_loss
            best_metrics = metrics_from(current_result)
            best_solution = current.copy()

        update_convergence(evals_used, best_loss)
        print(f"Restart {restart_id} (eval {evals_used}/{MAX_EVALS}) | Start loss: {current_loss:.6f} | Best: {best_loss:.6f}")

        while evals_used < MAX_EVALS:
            neighbors = get_neighbors(current)
            random.shuffle(neighbors)
            improved = False

            for neighbor in neighbors:
                if evals_used >= MAX_EVALS:
                    break

                n_result = evaluate_individual(neighbor)
                n_loss = n_result["val_loss"]
                evals_used += 1
                log_trial(neighbor, n_result, restart_id, evals_used)

                if n_loss < best_loss:
                    best_loss = n_loss
                    best_metrics = metrics_from(n_result)
                    best_solution = neighbor.copy()

                update_convergence(evals_used, best_loss)

                if n_loss < current_loss:
                    current = neighbor
                    current_loss = n_loss
                    improved = True
                    print(f"  HC step (eval {evals_used}/{MAX_EVALS}) | Loss: {current_loss:.6f} | Best: {best_loss:.6f}")
                    break

            if not improved:
                print(f"  Local min reached at loss {current_loss:.6f}")
                break

    elapsed = time.perf_counter() - start_time
    plt.close(fig_conv)
    return (results_log, best_solution, best_loss, best_metrics, elapsed)


def append_best_to_summary(best_solution, best_loss, best_metrics, elapsed, dataset_name):
    summary_path = os.path.join(RESULTS_DIR, 'best_results.csv')
    row = pd.DataFrame([{
        "dataset": dataset_name,
        "filters": best_solution["filters"],
        "kernel_size": best_solution["kernel_size"],
        "lstm_units": best_solution["lstm_units"],
        "dropout": best_solution["dropout"],
        "learning_rate": best_solution["learning_rate"],
        "batch_size": best_solution["batch_size"],
        "val_loss": best_loss,
        **best_metrics,
        "method": "Hill Climbing",
        "execution_time": round(elapsed, 4)
    }])
    append_best_row(summary_path, row)


if __name__ == "__main__":
    haaglanden, sleep_edfx = get_data_all_datasets()
    datasets = {
        "Haaglanden": haaglanden,
        "Sleep-EDF": sleep_edfx
    }

    for dataset_name, dataset in datasets.items():
        X, y = dataset
        (results, best_solution, best_loss, best_metrics, elapsed) = run_hill_climbing(dataset_name, X, y)
        df = pd.DataFrame(results)
        filename = f"hill_climbing_results_{dataset_name.lower().replace('-', '_')}.csv"
        df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)
        append_best_to_summary(best_solution, best_loss, best_metrics, elapsed, dataset_name)