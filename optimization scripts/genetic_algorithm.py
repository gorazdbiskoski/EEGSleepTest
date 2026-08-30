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

POPULATION_SIZE = 10
NUM_GENERATIONS = 10
MUTATION_RATE = 0.2
TOURNAMENT_SIZE = 3
ELITISM = 1

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
VIZ_DIR = os.path.join(os.path.dirname(__file__), 'convergence plots')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)


def run_genetic_algorithm(dataset_name, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results_log = []
    best_loss = float("inf")
    best_accuracy = None
    best_solution = None
    conv_x = []
    conv_y = []

    fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
    ax_conv.set_title(f'GA – Convergence ({dataset_name})')
    ax_conv.set_xlabel('Generation')
    ax_conv.set_ylabel('Best val_loss')
    conv_line, = ax_conv.plot([], [], marker='o')

    param_names = list(PARAM_SPACE.keys())
    max_len = max(len(PARAM_SPACE[param]) for param in param_names)
    heat_matrix = np.full((len(param_names), max_len), np.nan)

    fig_heat, ax_heat = plt.subplots(figsize=(7, 4))
    image = ax_heat.imshow(heat_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    ax_heat.set_yticks(range(len(param_names)))
    ax_heat.set_yticklabels(param_names)
    ax_heat.set_xlabel('Parameter value index')
    fig_heat.colorbar(image, ax=ax_heat, label='Selection frequency')
    fig_heat.tight_layout()

    def update_convergence(generation, current_best_loss):
        conv_x.append(generation)
        conv_y.append(current_best_loss)
        conv_line.set_data(conv_x, conv_y)
        ax_conv.relim()
        ax_conv.autoscale_view()
        filename = f"ga_convergence_{dataset_name.lower().replace('-', '_')}.png"
        fig_conv.savefig(os.path.join(VIZ_DIR, filename), dpi=100)

    def update_population_heatmap(population, generation):
        heat_matrix[:] = np.nan
        for i, param in enumerate(param_names):
            values = PARAM_SPACE[param]
            counts = np.zeros(len(values))
            for individual in population:
                value = individual[param]
                index = values.index(value)
                counts[index] += 1
            if counts.sum() > 0:
                counts = counts / counts.sum()
            heat_matrix[i, :len(values)] = counts
        image.set_data(heat_matrix)
        ax_heat.set_title(f'GA – Population heatmap ({dataset_name}, gen {generation})')
        filename = f"ga_population_heatmap_{dataset_name.lower().replace('-', '_')}.png"
        fig_heat.savefig(os.path.join(VIZ_DIR, filename), dpi=100)

    def create_individual():
        return {param: random.choice(values) for param, values in PARAM_SPACE.items()}

    def create_population():
        return [create_individual() for _ in range(POPULATION_SIZE)]

    def evaluate_individual(individual):
        params = [
            int(individual["filters"]),
            int(individual["kernel_size"]),
            int(individual["lstm_units"]),
            float(individual["dropout"]),
            float(individual["learning_rate"]),
            int(individual["batch_size"])
        ]
        result = evaluate_model(params, X_train, X_test, y_train, y_test)
        return result["val_loss"], result["val_accuracy"], result["epochs"]

    def tournament_selection(population, fitnesses):
        selected = random.sample(list(zip(population, fitnesses)), TOURNAMENT_SIZE)
        selected.sort(key=lambda x: x[1])
        return selected[0][0]

    def crossover(parent1, parent2):
        child = {}
        for param in PARAM_SPACE:
            child[param] = random.choice([parent1[param], parent2[param]])
        return child

    def mutate(individual):
        mutated = individual.copy()
        for param, values in PARAM_SPACE.items():
            if random.random() < MUTATION_RATE:
                mutated[param] = random.choice(values)
        return mutated

    start_time = time.perf_counter()
    population = create_population()

    for generation in range(NUM_GENERATIONS):
        fitnesses = []
        for individual in population:
            loss, acc, epochs_used = evaluate_individual(individual)
            fitnesses.append(loss)
            results_log.append({
                "dataset": dataset_name,
                "generation": generation + 1,
                "filters": individual["filters"],
                "kernel_size": individual["kernel_size"],
                "lstm_units": individual["lstm_units"],
                "dropout": individual["dropout"],
                "learning_rate": individual["learning_rate"],
                "batch_size": individual["batch_size"],
                "val_loss": loss,
                "val_accuracy": acc,
                "epochs_used": epochs_used,
                "method": "GA"
            })
            if loss < best_loss:
                best_loss = loss
                best_accuracy = acc
                best_solution = individual.copy()

        sorted_population = [individual for _, individual in sorted(zip(fitnesses, population), key=lambda pair: pair[0])]
        new_population = sorted_population[:ELITISM]

        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population
        update_convergence(generation + 1, best_loss)
        update_population_heatmap(population, generation + 1)

    elapsed = time.perf_counter() - start_time
    plt.close(fig_conv)
    plt.close(fig_heat)
    return (results_log, best_solution, best_loss, best_accuracy, elapsed)


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
        "method": "GA",
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
        (results, best_solution, best_loss, best_accuracy, elapsed) = run_genetic_algorithm(dataset_name, X, y)
        df = pd.DataFrame(results)
        filename = f"ga_results_{dataset_name.lower().replace('-', '_')}.csv"
        df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)
        append_best_to_summary(best_solution, best_loss, best_accuracy, elapsed, dataset_name)