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
RANDOM_SEED = 42

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
VIZ_DIR = os.path.join(BASE_DIR, 'convergence plots')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)


def random_individual():
    return {param: random.choice(values) for param, values in PARAM_SPACE.items()}


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
    ax_conv.set_title(f'Simulated Annealing – Convergence ({dataset_name})')
    ax_conv.set_xlabel('Evaluation #')
    ax_conv.set_ylabel('Best val_loss')
    conv_x, conv_y = [], []
    (conv_line,) = ax_conv.plot([], [], marker='o', color='teal')
    return fig_conv, ax_conv, conv_x, conv_y, conv_line


def update_convergence(eval_num, best_loss, conv_x, conv_y, conv_line, ax_conv, fig_conv, dataset_name):
    conv_x.append(eval_num)
    conv_y.append(best_loss)
    conv_line.set_data(conv_x, conv_y)
    ax_conv.relim()
    ax_conv.autoscale_view()
    filename = f"simulated_annealing_convergence_{dataset_name.lower().replace('-', '_')}.png"
    fig_conv.savefig(os.path.join(VIZ_DIR, filename), dpi=100)


def run_simulated_annealing(dataset_name, X, y):
    logger.info(f"Running Simulated Annealing on {dataset_name}")

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    logger.info(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    results_log = []
    fig_conv, ax_conv, conv_x, conv_y, conv_line = create_visualisations(dataset_name)

    start_time = time.perf_counter()

    current = random_individual()
    current_loss, current_acc, epochs = evaluate_individual(current, X_train, X_val, y_train, y_val)

    best_solution = current.copy()
    best_loss = current_loss
    best_accuracy = current_acc

    results_log.append({
        "dataset": dataset_name,
        **current,
        "val_loss": current_loss,
        "val_accuracy": current_acc,
        "epochs_used": epochs,
        "method": "Simulated Annealing",
    })
    update_convergence(1, best_loss, conv_x, conv_y, conv_line, ax_conv, fig_conv, dataset_name)
    logger.info(f"Eval 1/{MAX_EVALS} | T={T_START:.4f} | Best: {best_loss:.6f}")

    T = T_START

    for step in range(2, MAX_EVALS + 1):
        if T < T_MIN:
            logger.info(f"Temperature below T_MIN ({T_MIN}), stopping at eval {step-1}.")
            break

        candidate = random_neighbor(current)
        candidate_loss, candidate_acc, cand_epochs = evaluate_individual(candidate, X_train, X_val, y_train, y_val)

        results_log.append({
            "dataset": dataset_name,
            **candidate,
            "val_loss": candidate_loss,
            "val_accuracy": candidate_acc,
            "epochs_used": cand_epochs,
            "method": "Simulated Annealing",
        })

        delta = candidate_loss - current_loss
        if delta < 0 or random.random() < math.exp(-delta / T):
            current = candidate
            current_loss = candidate_loss
            if current_loss < best_loss:
                best_loss = current_loss
                best_accuracy = candidate_acc
                best_solution = current.copy()
                logger.info(f"  NEW BEST: {best_loss:.6f}")

        T *= COOLING_RATE
        update_convergence(step, best_loss, conv_x, conv_y, conv_line, ax_conv, fig_conv, dataset_name)
        logger.info(
            f"Eval {step}/{MAX_EVALS} | T={T:.4f} | "
            f"Candidate: {candidate_loss:.6f} | Best: {best_loss:.6f}"
        )

    elapsed = time.perf_counter() - start_time
    plt.close(fig_conv)
    return results_log, best_solution, best_loss, best_accuracy, elapsed


def append_best_to_summary(best_solution, best_loss, best_accuracy, elapsed, dataset_name):
    summary_path = os.path.join(RESULTS_DIR, 'best_results.csv')
    row = pd.DataFrame([{
        "dataset": dataset_name,
        **best_solution,
        "val_loss": best_loss,
        "val_accuracy": best_accuracy,
        "method": "Simulated Annealing",
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

        results, best_solution, best_loss, best_accuracy, elapsed = run_simulated_annealing(dataset_name, X, y)

        df = pd.DataFrame(results)
        filename = f"simulated_annealing_results_{dataset_name.lower().replace('-', '_')}.csv"
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