import os
import sys

# Allow running this script directly (outside PyCharm), which otherwise puts
# only this folder on sys.path and cannot import the repo-root `data`/`model`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import time
import math
import random
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data.global_data_loader import get_data_all_datasets
from model.model_builder import evaluate_model
from results_io import append_best_row

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PARAM_SPACE = {
    "filters": [16, 32, 64, 96, 128],
    "kernel_size": [2, 3, 4, 5],
    "lstm_units": [32, 64, 96, 128],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "batch_size": [16, 32, 48, 64],
}

MAX_EVALS = 100
T_START = 1.0
COOLING_RATE = 0.95
T_MIN = 1e-3
PERTURB_STRENGTH = 3
PERTURB_DIMS = 2
RANDOM_SEED = 42

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
VIZ_DIR = os.path.join(BASE_DIR, 'convergence plots')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)


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


def perturb(individual):
    perturbed = individual.copy()
    dims_to_kick = random.sample(list(PARAM_SPACE.keys()), PERTURB_DIMS)
    for dim in dims_to_kick:
        values = PARAM_SPACE[dim]
        current_idx = values.index(perturbed[dim])
        step = random.choice([-1, 1]) * PERTURB_STRENGTH
        new_idx = max(0, min(len(values) - 1, current_idx + step))
        perturbed[dim] = values[new_idx]
    return perturbed


def evaluate_individual(individual, X_train, X_val, y_train, y_val):
    params = [
        int(individual["filters"]),
        int(individual["kernel_size"]),
        int(individual["lstm_units"]),
        float(individual["dropout"]),
        float(individual["learning_rate"]),
        int(individual["batch_size"]),
    ]
    result = evaluate_model(params, X_train, X_val, y_train, y_val)
    return result["val_loss"], result["val_accuracy"], result["epochs"]


def create_visualisations(dataset_name):
    fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
    ax_conv.set_title(f'Basin Hopping – Convergence ({dataset_name})')
    ax_conv.set_xlabel('Evaluation #')
    ax_conv.set_ylabel('Best val_loss')
    conv_x, conv_y = [], []
    (conv_line,) = ax_conv.plot([], [], marker='o', color='darkred')
    return fig_conv, ax_conv, conv_x, conv_y, conv_line


def update_convergence(eval_num, best_loss, conv_x, conv_y, conv_line, ax_conv, fig_conv, dataset_name):
    conv_x.append(eval_num)
    conv_y.append(best_loss)
    conv_line.set_data(conv_x, conv_y)
    ax_conv.relim()
    ax_conv.autoscale_view()
    filename = f"basin_hopping_convergence_{dataset_name.lower().replace('-', '_')}.png"
    fig_conv.savefig(os.path.join(VIZ_DIR, filename), dpi=100)


def run_inner_hc(start, basin_id, state, X_train, X_val, y_train, y_val, results_log, dataset_name,
                 conv_x, conv_y, conv_line, ax_conv, fig_conv):
    current = start
    current_loss, current_acc, epochs = evaluate_individual(current, X_train, X_val, y_train, y_val)
    state["evals_used"] += 1

    results_log.append({
        "dataset": dataset_name,
        "filters": current["filters"],
        "kernel_size": current["kernel_size"],
        "lstm_units": current["lstm_units"],
        "dropout": current["dropout"],
        "learning_rate": current["learning_rate"],
        "batch_size": current["batch_size"],
        "val_loss": current_loss,
        "val_accuracy": current_acc,
        "epochs_used": epochs,
        "basin_id": basin_id,
        "phase": "hc_start",
        "method": "Basin Hopping",
    })

    if current_loss < state["best_loss"]:
        state["best_loss"] = current_loss
        state["best_accuracy"] = current_acc
        state["best_solution"] = current.copy()

    update_convergence(state["evals_used"], state["best_loss"], conv_x, conv_y, conv_line, ax_conv, fig_conv, dataset_name)

    while state["evals_used"] < MAX_EVALS:
        neighbors = get_neighbors(current)
        random.shuffle(neighbors)
        improved = False

        for neighbor in neighbors:
            if state["evals_used"] >= MAX_EVALS:
                return current, current_loss

            n_loss, n_acc, n_epochs = evaluate_individual(neighbor, X_train, X_val, y_train, y_val)
            state["evals_used"] += 1

            results_log.append({
                "dataset": dataset_name,
                "filters": neighbor["filters"],
                "kernel_size": neighbor["kernel_size"],
                "lstm_units": neighbor["lstm_units"],
                "dropout": neighbor["dropout"],
                "learning_rate": neighbor["learning_rate"],
                "batch_size": neighbor["batch_size"],
                "val_loss": n_loss,
                "val_accuracy": n_acc,
                "epochs_used": n_epochs,
                "basin_id": basin_id,
                "phase": "hc_step",
                "method": "Basin Hopping",
            })

            if n_loss < state["best_loss"]:
                state["best_loss"] = n_loss
                state["best_accuracy"] = n_acc
                state["best_solution"] = neighbor.copy()

            update_convergence(state["evals_used"], state["best_loss"], conv_x, conv_y, conv_line, ax_conv, fig_conv, dataset_name)

            if n_loss < current_loss:
                current = neighbor
                current_loss = n_loss
                improved = True
                break

        if not improved:
            return current, current_loss

    return current, current_loss


def run_basin_hopping(dataset_name, X, y):
    logger.info(f"Running Basin Hopping on {dataset_name}")

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    logger.info(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    results_log = []
    state = {
        "evals_used": 0,
        "best_loss": float("inf"),
        "best_accuracy": None,
        "best_solution": None,
    }

    fig_conv, ax_conv, conv_x, conv_y, conv_line = create_visualisations(dataset_name)

    start_time = time.perf_counter()
    basin_id = 0
    T = T_START

    # Initial HC
    basin_id += 1
    initial = random_individual()
    launch, launch_loss = run_inner_hc(
        initial, basin_id, state, X_train, X_val, y_train, y_val,
        results_log, dataset_name, conv_x, conv_y, conv_line, ax_conv, fig_conv
    )
    logger.info(
        f"Basin {basin_id} (eval {state['evals_used']}/{MAX_EVALS}) | "
        f"Launch loss: {launch_loss:.6f} | T={T:.4f} | Best: {state['best_loss']:.6f}"
    )

    while state["evals_used"] < MAX_EVALS:
        if T < T_MIN:
            logger.info(f"Temperature below T_MIN ({T_MIN}), stopping at eval {state['evals_used']}.")
            break

        basin_id += 1
        perturbed = perturb(launch)
        candidate, candidate_loss = run_inner_hc(
            perturbed, basin_id, state, X_train, X_val, y_train, y_val,
            results_log, dataset_name, conv_x, conv_y, conv_line, ax_conv, fig_conv
        )

        delta = candidate_loss - launch_loss
        accepted = delta < 0 or random.random() < math.exp(-delta / T)

        if accepted:
            launch = candidate
            launch_loss = candidate_loss

        logger.info(
            f"Basin {basin_id} (eval {state['evals_used']}/{MAX_EVALS}) | "
            f"Candidate: {candidate_loss:.6f} | delta={delta:+.4f} | T={T:.4f} | "
            f"{'accepted' if accepted else 'rejected'} | Best: {state['best_loss']:.6f}"
        )
        T *= COOLING_RATE

    elapsed = time.perf_counter() - start_time
    plt.close(fig_conv)

    return results_log, state["best_solution"], state["best_loss"], state["best_accuracy"], elapsed


def append_best_to_summary(best_solution, best_loss, best_accuracy, elapsed, dataset_name):
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
        "val_accuracy": best_accuracy,
        "method": "Basin Hopping",
        "execution_time": round(elapsed, 4),
    }])
    append_best_row(summary_path, row)


if __name__ == "__main__":
    sleep_edfx, haaglanden = get_data_all_datasets()
    datasets = {"Sleep-EDF": sleep_edfx, "Haaglanden": haaglanden}

    for dataset_name, dataset in datasets.items():
        X, y = dataset
        logger.info(f"{dataset_name}: X={X.shape}, y={y.shape}")

        results, best_solution, best_loss, best_accuracy, elapsed = run_basin_hopping(dataset_name, X, y)

        df = pd.DataFrame(results)
        filename = f"basin_hopping_results_{dataset_name.lower().replace('-', '_')}.csv"
        output_path = os.path.join(RESULTS_DIR, filename)
        df.to_csv(output_path, index=False)

        append_best_to_summary(best_solution, best_loss, best_accuracy, elapsed, dataset_name)

        logger.info(f"{dataset_name} FINISHED")
        logger.info(f"Best parameters: {best_solution}")
        logger.info(f"Best val_loss: {best_loss:.6f}")
        logger.info(f"Best val_accuracy: {best_accuracy:.6f}")
        logger.info(f"Execution time: {elapsed:.2f} s")
        logger.info(f"Results saved to: {output_path}")

    logger.info("ALL DATASETS FINISHED")