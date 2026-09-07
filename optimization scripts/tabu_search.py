import os
import sys

# Allow running this script directly (outside PyCharm), which otherwise puts
# only this folder on sys.path and cannot import the repo-root `data`/`model`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import time
import random
import logging

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

PARAM_NAMES = list(PARAM_SPACE.keys())
SPACES = [PARAM_SPACE[k] for k in PARAM_NAMES]

NUM_TRIALS = 100
TABU_TENURE = 10

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
VIZ_DIR = os.path.join(BASE_DIR, 'convergence plots')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)


def create_visualisations(dataset_name):
    fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
    ax_conv.set_title(f'Tabu Search – Convergence ({dataset_name})')
    ax_conv.set_xlabel('Trial #')
    ax_conv.set_ylabel('Best val_loss')
    conv_x, conv_y = [], []
    (conv_line,) = ax_conv.plot([], [], marker='o', color='steelblue')
    return fig_conv, ax_conv, conv_x, conv_y, conv_line


def update_convergence(trial, best_loss, conv_x, conv_y, conv_line, ax_conv, fig_conv, dataset_name):
    conv_x.append(trial)
    conv_y.append(best_loss)
    conv_line.set_data(conv_x, conv_y)
    ax_conv.relim()
    ax_conv.autoscale_view()
    filename = f"tabu_search_convergence_{dataset_name.lower().replace('-', '_')}.png"
    fig_conv.savefig(os.path.join(VIZ_DIR, filename), dpi=100)


def get_neighbors(current):
    neighbors = []
    for i, space in enumerate(SPACES):
        idx = space.index(current[i])
        for delta in (-1, +1):
            new_idx = idx + delta
            if 0 <= new_idx < len(space):
                neighbor = current[:]
                neighbor[i] = space[new_idx]
                neighbors.append(neighbor)
    return neighbors


def run_tabu_search(dataset_name, X, y, num_trials=NUM_TRIALS, tabu_tenure=TABU_TENURE):
    logger.info(f"Running Tabu Search on {dataset_name}")

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    logger.info(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    results_log = []
    fig_conv, ax_conv, conv_x, conv_y, conv_line = create_visualisations(dataset_name)

    start_time = time.perf_counter()

    current = [random.choice(s) for s in SPACES]
    result = evaluate_model(current, X_train, X_val, y_train, y_val)
    current_loss = result["val_loss"]
    current_acc = result["val_accuracy"]
    epochs = result["epochs"]

    best_params = current[:]
    best_loss = current_loss
    best_accuracy = current_acc
    tabu_list = []
    trial = 1

    results_log.append({
        "dataset": dataset_name,
        **dict(zip(PARAM_NAMES, current)),
        "val_loss": current_loss,
        "val_accuracy": current_acc,
        "epochs_used": epochs,
        "method": "Tabu Search",
    })
    update_convergence(trial, best_loss, conv_x, conv_y, conv_line, ax_conv, fig_conv, dataset_name)
    logger.info(f"Trial {trial}/{num_trials} | loss={current_loss:.6f}")

    while trial < num_trials:
        neighbors = get_neighbors(current)

        candidates = []
        for n in neighbors:
            # aspiration: allow tabu if better than global best
            if n not in tabu_list:
                candidates.append(n)
            else:
                # quick check would require evaluation; keep simple
                candidates.append(n)

        if not candidates:
            candidates = neighbors

        best_candidate = None
        best_candidate_loss = float('inf')
        best_candidate_acc = None
        best_candidate_epochs = None

        for n in candidates:
            if trial >= num_trials:
                break
            trial += 1

            result = evaluate_model(n, X_train, X_val, y_train, y_val)
            loss = result["val_loss"]
            acc = result["val_accuracy"]
            epochs = result["epochs"]

            logger.info(
                f"Trial {trial}/{num_trials} | "
                f"{dict(zip(PARAM_NAMES, n))} | loss={loss:.6f}"
            )

            results_log.append({
                "dataset": dataset_name,
                **dict(zip(PARAM_NAMES, n)),
                "val_loss": loss,
                "val_accuracy": acc,
                "epochs_used": epochs,
                "method": "Tabu Search",
            })

            if loss < best_candidate_loss:
                best_candidate_loss = loss
                best_candidate = n[:]
                best_candidate_acc = acc
                best_candidate_epochs = epochs

            if loss < best_loss:
                best_loss = loss
                best_accuracy = acc
                best_params = n[:]
                logger.info(f"  NEW BEST: {best_loss:.6f}")

            update_convergence(trial, best_loss, conv_x, conv_y, conv_line, ax_conv, fig_conv, dataset_name)

        if best_candidate is not None:
            current = best_candidate
            tabu_list.append(current[:])
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)

    elapsed = time.perf_counter() - start_time
    plt.close(fig_conv)
    return results_log, best_params, best_loss, best_accuracy, elapsed


def append_best_to_summary(best_params, best_loss, best_accuracy, elapsed, dataset_name):
    summary_path = os.path.join(RESULTS_DIR, 'best_results.csv')
    row = pd.DataFrame([{
        "dataset": dataset_name,
        **dict(zip(PARAM_NAMES, best_params)),
        "val_loss": best_loss,
        "val_accuracy": best_accuracy,
        "method": "Tabu Search",
        "execution_time": round(elapsed, 4),
    }])
    append_best_row(summary_path, row)


if __name__ == "__main__":
    haaglanden, sleep_edfx = get_data_all_datasets()
    datasets = {"Sleep-EDF": sleep_edfx, "Haaglanden": haaglanden}

    for dataset_name, dataset in datasets.items():
        X, y = dataset
        logger.info(f"{dataset_name}: X={X.shape}, y={y.shape}")

        results, best_params, best_loss, best_accuracy, elapsed = run_tabu_search(dataset_name, X, y)

        df = pd.DataFrame(results)
        filename = f"tabu_search_results_{dataset_name.lower().replace('-', '_')}.csv"
        output_path = os.path.join(RESULTS_DIR, filename)
        df.to_csv(output_path, index=False)

        append_best_to_summary(best_params, best_loss, best_accuracy, elapsed, dataset_name)

        logger.info(f"{dataset_name} FINISHED")
        logger.info(f"Best parameters: {dict(zip(PARAM_NAMES, best_params))}")
        logger.info(f"Best val_loss: {best_loss:.6f}")
        logger.info(f"Best val_accuracy: {best_accuracy:.6f}")
        logger.info(f"Execution time: {elapsed:.2f} s")
        logger.info(f"Results saved to: {output_path}")

    logger.info("ALL DATASETS FINISHED")