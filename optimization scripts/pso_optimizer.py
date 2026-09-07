import os
import sys

# Allow running this script directly (outside PyCharm), which otherwise puts
# only this folder on sys.path and cannot import the repo-root `data`/`model`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import time
import logging

import numpy as np
import pandas as pd
import pyswarms as ps
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

N_PARTICLES = 10
N_ITERS = 10

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
VIZ_DIR = os.path.join(BASE_DIR, 'convergence plots')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)


def create_visualisations(dataset_name):
    fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
    ax_conv.set_title(f'PSO – Convergence ({dataset_name})')
    ax_conv.set_xlabel('Iteration')
    ax_conv.set_ylabel('Best val_loss')
    conv_x, conv_y = [], []
    (conv_line,) = ax_conv.plot([], [], marker='o', color='darkorange')
    return fig_conv, ax_conv, conv_x, conv_y, conv_line


def update_convergence(iteration, best_loss, conv_x, conv_y, conv_line, ax_conv, fig_conv, dataset_name):
    conv_x.append(iteration)
    conv_y.append(best_loss)
    conv_line.set_data(conv_x, conv_y)
    ax_conv.relim()
    ax_conv.autoscale_view()
    filename = f"pso_convergence_{dataset_name.lower().replace('-', '_')}.png"
    fig_conv.savefig(os.path.join(VIZ_DIR, filename), dpi=100)


def run_pso(dataset_name, X, y):
    logger.info(f"Running PSO on {dataset_name}")

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    logger.info(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    results_log = []
    fig_conv, ax_conv, conv_x, conv_y, conv_line = create_visualisations(dataset_name)
    pso_iter = [0]
    global_best_loss = [float("inf")]
    global_best_acc = [None]
    global_best_pos = [None]

    def objective_function(particles):
        scores = []
        for p in particles:
            params = [int(p[0]), int(p[1]), int(p[2]), float(p[3]), float(p[4]), int(p[5])]
            result = evaluate_model(params, X_train, X_val, y_train, y_val)
            loss = result["val_loss"]
            acc = result["val_accuracy"]
            epochs = result["epochs"]

            results_log.append({
                "dataset": dataset_name,
                "filters": int(p[0]),
                "kernel_size": int(p[1]),
                "lstm_units": int(p[2]),
                "dropout": float(p[3]),
                "learning_rate": float(p[4]),
                "batch_size": int(p[5]),
                "val_loss": loss,
                "val_accuracy": acc,
                "epochs_used": epochs,
                "method": "PSO",
            })
            scores.append(loss)

            if loss < global_best_loss[0]:
                global_best_loss[0] = loss
                global_best_acc[0] = acc
                global_best_pos[0] = p.copy()

        best_this_iter = min(scores)
        current_best = min(conv_y + [best_this_iter]) if conv_y else best_this_iter
        pso_iter[0] += 1
        update_convergence(pso_iter[0], current_best, conv_x, conv_y, conv_line, ax_conv, fig_conv, dataset_name)
        logger.info(f"Iteration {pso_iter[0]}/{N_ITERS} | Best this iter: {best_this_iter:.6f} | Global best: {global_best_loss[0]:.6f}")
        return np.array(scores)

    bounds = (
        np.array([16, 2, 32, 0.1, 1e-4, 16]),
        np.array([128, 5, 128, 0.5, 1e-2, 64]),
    )

    optimizer = ps.single.GlobalBestPSO(
        n_particles=N_PARTICLES,
        dimensions=6,
        options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
        bounds=bounds,
    )

    start_time = time.perf_counter()
    best_cost, best_pos = optimizer.optimize(objective_function, iters=N_ITERS)
    elapsed = time.perf_counter() - start_time

    plt.close(fig_conv)

    # Ensure we have the tracked best
    if global_best_pos[0] is not None:
        best_pos = global_best_pos[0]
        best_cost = global_best_loss[0]
        best_acc = global_best_acc[0]
    else:
        best_acc = None

    return results_log, best_pos, best_cost, best_acc, elapsed


def append_best_to_summary(best_pos, best_loss, best_accuracy, elapsed, dataset_name):
    summary_path = os.path.join(RESULTS_DIR, 'best_results.csv')
    row = pd.DataFrame([{
        "dataset": dataset_name,
        "filters": int(best_pos[0]),
        "kernel_size": int(best_pos[1]),
        "lstm_units": int(best_pos[2]),
        "dropout": float(best_pos[3]),
        "learning_rate": float(best_pos[4]),
        "batch_size": int(best_pos[5]),
        "val_loss": best_loss,
        "val_accuracy": best_accuracy,
        "method": "PSO",
        "execution_time": round(elapsed, 4),
    }])
    append_best_row(summary_path, row)


if __name__ == "__main__":
    haaglanden, sleep_edfx = get_data_all_datasets()
    datasets = {"Sleep-EDF": sleep_edfx, "Haaglanden": haaglanden}

    for dataset_name, dataset in datasets.items():
        X, y = dataset
        logger.info(f"{dataset_name}: X={X.shape}, y={y.shape}")

        results, best_pos, best_loss, best_accuracy, elapsed = run_pso(dataset_name, X, y)

        df = pd.DataFrame(results)
        filename = f"pso_results_{dataset_name.lower().replace('-', '_')}.csv"
        output_path = os.path.join(RESULTS_DIR, filename)
        df.to_csv(output_path, index=False)

        append_best_to_summary(best_pos, best_loss, best_accuracy, elapsed, dataset_name)

        logger.info(f"{dataset_name} FINISHED")
        logger.info(f"Best parameters: {best_pos}")
        logger.info(f"Best val_loss: {best_loss:.6f}")
        logger.info(f"Best val_accuracy: {best_accuracy}")
        logger.info(f"Execution time: {elapsed:.2f} s")
        logger.info(f"Results saved to: {output_path}")

    logger.info("ALL DATASETS FINISHED")