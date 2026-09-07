import os
import sys

# Allow running this script directly (outside PyCharm), which otherwise puts
# only this folder on sys.path and cannot import the repo-root `data`/`model`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import time

import tensorflow as tf
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from data.global_data_loader import get_data_all_datasets
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

# Start small to test the implementation.
# Once everything works, increase to 5/5 and eventually 10/10.
NUM_ANTS = 3
NUM_ITERATIONS = 2

EVAPORATION_RATE = 0.3
ALPHA = 1.0
Q = 1.0

BASE_DIR = os.path.dirname(__file__)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    '..',
    'results'
)

VIZ_DIR = os.path.join(
    BASE_DIR,
    'convergence plots'
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)


def choose_parameter(param_name, pheromones):
    pheromone = pheromones[param_name]

    probabilities = pheromone ** ALPHA
    probabilities = probabilities / probabilities.sum()

    index = np.random.choice(
        range(len(PARAM_SPACE[param_name])),
        p=probabilities
    )

    return PARAM_SPACE[param_name][index], index


def construct_solution(pheromones):
    solution = {}
    indices = {}

    for param in PARAM_SPACE:
        value, index = choose_parameter(param, pheromones)
        solution[param] = value
        indices[param] = index

    return solution, indices


def evaluate_solution(
    solution,
    X_train,
    X_val,
    y_train,
    y_val
):
    params = [
        int(solution["filters"]),
        int(solution["kernel_size"]),
        int(solution["lstm_units"]),
        float(solution["dropout"]),
        float(solution["learning_rate"]),
        int(solution["batch_size"])
    ]

    result = evaluate_model(
        params,
        X_train,
        X_val,
        y_train,
        y_val
    )

    return (
        result["val_loss"],
        result["val_accuracy"],
        result["epochs"]
    )


def create_visualisations(param_names):
    max_len = max(
        len(PARAM_SPACE[param])
        for param in param_names
    )

    heat_matrix = np.full(
        (len(param_names), max_len),
        np.nan
    )

    fig_conv, ax_conv = plt.subplots(figsize=(8, 4))

    ax_conv.set_title('ACO – Convergence')
    ax_conv.set_xlabel('Iteration')
    ax_conv.set_ylabel('Best val_loss')

    conv_x = []
    conv_y = []

    conv_line, = ax_conv.plot(
        [],
        [],
        marker='o',
        color='royalblue'
    )

    fig_heat, ax_heat = plt.subplots(figsize=(7, 4))

    image = ax_heat.imshow(
        heat_matrix,
        aspect='auto',
        cmap='YlOrRd',
        vmin=0,
        vmax=1,
        interpolation='nearest'
    )

    ax_heat.set_yticks(range(len(param_names)))
    ax_heat.set_yticklabels(param_names)
    ax_heat.set_xlabel('Parameter value index')

    fig_heat.colorbar(
        image,
        ax=ax_heat,
        label='Normalised pheromone'
    )

    fig_heat.tight_layout()

    return (
        fig_conv,
        ax_conv,
        conv_x,
        conv_y,
        conv_line,
        fig_heat,
        ax_heat,
        heat_matrix,
        image
    )


def update_convergence(
    iteration,
    best_loss,
    conv_x,
    conv_y,
    conv_line,
    ax_conv,
    fig_conv,
    dataset_name
):
    conv_x.append(iteration)
    conv_y.append(best_loss)

    conv_line.set_data(
        conv_x,
        conv_y
    )

    ax_conv.relim()
    ax_conv.autoscale_view()

    filename = (
        f"aco_convergence_"
        f"{dataset_name.lower().replace('-', '_')}.png"
    )

    fig_conv.savefig(
        os.path.join(
            VIZ_DIR,
            filename
        ),
        dpi=100
    )


def update_pheromone_heatmap(
    iteration,
    pheromones,
    param_names,
    heat_matrix,
    image,
    ax_heat,
    fig_heat,
    dataset_name
):
    for i, param in enumerate(param_names):
        values = pheromones[param]

        heat_matrix[i, :] = np.nan
        heat_matrix[i, :len(values)] = (
            values / values.sum()
        )

    image.set_data(heat_matrix)

    ax_heat.set_title(
        f'ACO – Pheromone heatmap '
        f'({dataset_name}, iter {iteration})'
    )

    filename = (
        f"aco_pheromone_heatmap_"
        f"{dataset_name.lower().replace('-', '_')}.png"
    )

    fig_heat.savefig(
        os.path.join(
            VIZ_DIR,
            filename
        ),
        dpi=100
    )


def run_aco(dataset_name, X, y):
    logger.info(f"Running ACO on {dataset_name}")

    # First split: keep test set completely separate.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42
    )

    logger.info(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    logger.info(f"Training samples:   {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Testing samples:    {len(X_test)}")

    pheromones = {
        key: np.ones(len(values))
        for key, values in PARAM_SPACE.items()
    }

    results_log = []

    best_loss = float("inf")
    best_solution = None
    best_accuracy = None

    param_names = list(PARAM_SPACE.keys())

    (
        fig_conv,
        ax_conv,
        conv_x,
        conv_y,
        conv_line,
        fig_heat,
        ax_heat,
        heat_matrix,
        image
    ) = create_visualisations(param_names)

    start_time = time.perf_counter()

    for iteration in range(NUM_ITERATIONS):
        all_solutions = []

        logger.info(f"Iteration "f"{iteration + 1}/{NUM_ITERATIONS}")

        for ant in range(NUM_ANTS):
            solution, indices = construct_solution(
                pheromones
            )

            logger.info(f"  Ant {ant + 1}/{NUM_ANTS}: "f"{solution}")

            ant_start = time.perf_counter()

            loss, accuracy, epochs_used = evaluate_solution(
                solution,
                X_train,
                X_val,
                y_train,
                y_val
            )

            ant_elapsed = (
                time.perf_counter() - ant_start
            )

            logger.info(f"val_loss: {loss:.6f}")

            logger.info(f"val_accuracy: {accuracy:.6f}")

            print(f"epochs used: {epochs_used}")
            print(f"time: {ant_elapsed:.2f} seconds")

            results_log.append({
                "dataset": dataset_name,
                "iteration": iteration + 1,
                "ant": ant + 1,
                "filters": solution["filters"],
                "kernel_size": solution["kernel_size"],
                "lstm_units": solution["lstm_units"],
                "dropout": solution["dropout"],
                "learning_rate": solution["learning_rate"],
                "batch_size": solution["batch_size"],
                "val_loss": loss,
                "val_accuracy": accuracy,
                "epochs_used": epochs_used,
                "method": "ACO"
            })

            all_solutions.append(
                (solution, indices, loss)
            )

            if loss < best_loss:
                best_loss = loss
                best_solution = solution.copy()
                best_accuracy = accuracy

                logger.info(f"NEW BEST: "f"{best_loss:.6f}")

            for param in pheromones:
                pheromones[param] *= (
                1 - EVAPORATION_RATE
            )

        for solution, indices, loss in all_solutions:
            pheromone_deposit = (
                Q / (loss + 1e-8)
            )

            for param in PARAM_SPACE:
                index = indices[param]

                pheromones[param][index] += (pheromone_deposit)

        update_convergence(
            iteration + 1,
            best_loss,
            conv_x,
            conv_y,
            conv_line,
            ax_conv,
            fig_conv,
            dataset_name
        )

        update_pheromone_heatmap(
            iteration + 1,
            pheromones,
            param_names,
            heat_matrix,
            image,
            ax_heat,
            fig_heat,
            dataset_name
        )

        logger.info(
            f"Best loss: "
            f"{best_loss:.6f}"
        )

        logger.info(
            f"Best accuracy: "
            f"{best_accuracy:.6f}"
        )

        logger.info(f"Best parameters: "
            f"{best_solution}"
        )

    elapsed = time.perf_counter() - start_time

    plt.close(fig_conv)
    plt.close(fig_heat)

    return (
        results_log,
        best_solution,
        best_loss,
        best_accuracy,
        elapsed
    )


def append_best_to_summary(
    best_solution,
    best_loss,
    best_accuracy,
    elapsed,
    dataset_name
):
    summary_path = os.path.join(
        RESULTS_DIR,
        'best_results.csv'
    )

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
        "method": "ACO",
        "execution_time": round(elapsed, 4)
    }])

    append_best_row(summary_path, row)


if __name__ == "__main__":
    haaglanden, sleep_edfx = get_data_all_datasets()

    datasets = {
        "Sleep-EDF": sleep_edfx,
        "Haaglanden": haaglanden,
    }

    for dataset_name, dataset in datasets.items():
        X, y = dataset

        logger.info(f"{dataset_name} data:")
        logger.info(f"X shape: {X.shape}")
        logger.info(f"y shape: {y.shape}")

        logger.info(f"{dataset_name}: X={X.shape}, y={y.shape}")

        (
            results,
            best_solution,
            best_loss,
            best_accuracy,
            elapsed
        ) = run_aco(
            dataset_name,
            X,
            y
        )

        df = pd.DataFrame(results)

        filename = (
            f"aco_results_"
            f"{dataset_name.lower().replace('-', '_')}.csv"
        )

        output_path = os.path.join(
            RESULTS_DIR,
            filename
        )

        df.to_csv(
            output_path,
            index=False
        )

        append_best_to_summary(
            best_solution,
            best_loss,
            best_accuracy,
            elapsed,
            dataset_name
        )

        logger.info(f"{dataset_name} FINISHED")
        logger.info(f"Best parameters: {best_solution}")
        logger.info(f"Best val_loss: {best_loss:.6f}")
        logger.info(f"Best val_accuracy: {best_accuracy:.6f}")
        logger.info(f"Execution time: {elapsed:.2f} seconds")
        logger.info(f"Results saved to: {output_path}")


    logger.info("ALL DATASETS FINISHED")
    logger.info(f"Results directory: {RESULTS_DIR}")
    logger.info(f"Visualisations directory: {VIZ_DIR}")