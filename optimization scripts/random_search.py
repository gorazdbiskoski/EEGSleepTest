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

NUM_TRIALS = 100

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
VIZ_DIR = os.path.join(BASE_DIR, 'convergence plots')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)


def create_visualisations(dataset_name):
    fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
    ax_conv.set_title(f'Random Search – Convergence ({dataset_name})')
    ax_conv.set_xlabel('Trial #')
    ax_conv.set_ylabel('Best val_loss')
    conv_x, conv_y = [], []
    (conv_line,) = ax_conv.plot([], [], marker='o', color='crimson')
    return fig_conv, ax_conv, conv_x, conv_y, conv_line


def update_convergence(trial, best_loss, conv_x, conv_y, conv_line, ax_conv, fig_conv, dataset_name):
    conv_x.append(trial)
    conv_y.append(best_loss)
    conv_line.set_data(conv_x, conv_y)
    ax_conv.relim()
    ax_conv.autoscale_view()
    filename = f"random_search_convergence_{dataset_name.lower().replace('-', '_')}.png"
    fig_conv.savefig(os.path.join(VIZ_DIR, filename), dpi=100)


def run_random_search(dataset_name, X, y, num_trials=NUM_TRIALS):
    logger.info(f"Running Random Search on {dataset_name}")

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    logger.info(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    results_log = []
    best_loss = float('inf')
    best_accuracy = None
    best_params = None

    fig_conv, ax_conv, conv_x, conv_y, conv_line = create_visualisations(dataset_name)
    start_time = time.perf_counter()

    for i in range(num_trials):
        solution = {k: random.choice(v) for k, v in PARAM_SPACE.items()}
        params = [
            int(solution["filters"]),
            int(solution["kernel_size"]),
            int(solution["lstm_units"]),
            float(solution["dropout"]),
            float(solution["learning_rate"]),
            int(solution["batch_size"]),
        ]

        logger.info(f"Trial {i+1}/{num_trials} | {solution}")

        result = evaluate_model(params, X_train, X_val, y_train, y_val)
        loss = result["val_loss"]
        acc = result["val_accuracy"]
        epochs = result["epochs"]

        logger.info(f"  val_loss: {loss:.6f} | val_accuracy: {acc:.6f}")

        results_log.append({
            "dataset": dataset_name,
            "trial": i + 1,
            **solution,
            "val_loss": loss,
            "val_accuracy": acc,
            "epochs_used": epochs,
            "method": "Random Search",
        })

        if loss < best_loss:
            best_loss = loss
            best_accuracy = acc
            best_params = solution.copy()
            logger.info(f"  NEW BEST: {best_loss:.6f}")

        update_convergence(i + 1, best_loss, conv_x, conv_y, conv_line, ax_conv, fig_conv, dataset_name)

    elapsed = time.perf_counter() - start_time
    plt.close(fig_conv)
    return results_log, best_params, best_loss, best_accuracy, elapsed


def append_best_to_summary(best_params, best_loss, best_accuracy, elapsed, dataset_name):
    summary_path = os.path.join(RESULTS_DIR, 'best_results.csv')
    row = pd.DataFrame([{
        "dataset": dataset_name,
        **best_params,
        "val_loss": best_loss,
        "val_accuracy": best_accuracy,
        "method": "Random Search",
        "execution_time": round(elapsed, 4),
    }])
    write_header = not os.path.exists(summary_path)
    row.to_csv(summary_path, mode="a", header=write_header, index=False)


if __name__ == "__main__":
    sleep_edfx, haaglanden = get_data_all_datasets()
    datasets = {"Sleep-EDF": sleep_edfx, "Haaglanden": haaglanden}

    for dataset_name, dataset in datasets.items():
        X, y = dataset
        logger.info(f"{dataset_name}: X={X.shape}, y={y.shape}")

        results, best_params, best_loss, best_accuracy, elapsed = run_random_search(dataset_name, X, y)

        df = pd.DataFrame(results)
        filename = f"random_search_results_{dataset_name.lower().replace('-', '_')}.csv"
        output_path = os.path.join(RESULTS_DIR, filename)
        df.to_csv(output_path, index=False)

        append_best_to_summary(best_params, best_loss, best_accuracy, elapsed, dataset_name)

        logger.info(f"{dataset_name} FINISHED")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best val_loss: {best_loss:.6f}")
        logger.info(f"Best val_accuracy: {best_accuracy:.6f}")
        logger.info(f"Execution time: {elapsed:.2f} s")
        logger.info(f"Results saved to: {output_path}")

    logger.info("ALL DATASETS FINISHED")