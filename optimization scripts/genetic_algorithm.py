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

POPULATION_SIZE = 10
NUM_GENERATIONS = 10
MUTATION_RATE = 0.2
TOURNAMENT_SIZE = 3
ELITISM = 1

VIZ_DIR = os.path.join(os.path.dirname(__file__), 'convergence plots')
os.makedirs(VIZ_DIR, exist_ok=True)

fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
ax_conv.set_title('GA – Convergence')
ax_conv.set_xlabel('Generation')
ax_conv.set_ylabel('Best val_loss')
conv_x, conv_y = [], []
(conv_line,) = ax_conv.plot([], [], marker='o')

param_names = list(PARAM_SPACE.keys())
_max_len = max(len(PARAM_SPACE[p]) for p in param_names)

_heat_matrix = np.full((len(param_names), _max_len), np.nan)

fig_heat, ax_heat = plt.subplots(figsize=(7, 4))
_im = ax_heat.imshow(
    _heat_matrix,
    aspect='auto',
    cmap='viridis',
    interpolation='nearest'
)

ax_heat.set_yticks(range(len(param_names)))
ax_heat.set_yticklabels(param_names)
ax_heat.set_xlabel('Parameter value index')

fig_heat.colorbar(_im, ax=ax_heat, label='Selection frequency')
fig_heat.tight_layout()


def _update_convergence(generation, best_loss):
    conv_x.append(generation)
    conv_y.append(best_loss)

    conv_line.set_data(conv_x, conv_y)

    ax_conv.relim()
    ax_conv.autoscale_view()

    fig_conv.savefig(
        os.path.join(VIZ_DIR, 'ga_convergence.png'),
        dpi=100
    )


def _update_population_heatmap(population, generation):
    global _heat_matrix

    _heat_matrix[:] = np.nan

    for i, param in enumerate(param_names):
        values = PARAM_SPACE[param]

        counts = np.zeros(len(values))

        for individual in population:
            value = individual[param]
            idx = values.index(value)
            counts[idx] += 1

        if counts.sum() > 0:
            counts = counts / counts.sum()

        _heat_matrix[i, :len(values)] = counts

    _im.set_data(_heat_matrix)

    ax_heat.set_title(f'GA – Population heatmap (gen {generation})')

    fig_heat.savefig(
        os.path.join(VIZ_DIR, 'ga_population_heatmap.png'),
        dpi=100
    )


def create_individual():
    return {
        param: random.choice(values)
        for param, values in PARAM_SPACE.items()
    }


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

    return evaluate_model(params, X_train, X_test, y_train, y_test)


def tournament_selection(population, fitnesses):
    selected = random.sample(
        list(zip(population, fitnesses)),
        TOURNAMENT_SIZE
    )

    selected.sort(key=lambda x: x[1])

    return selected[0][0]


def crossover(parent1, parent2):
    child = {}

    for param in PARAM_SPACE:
        child[param] = random.choice([
            parent1[param],
            parent2[param]
        ])

    return child


def mutate(individual):
    mutated = individual.copy()

    for param, values in PARAM_SPACE.items():
        if random.random() < MUTATION_RATE:
            mutated[param] = random.choice(values)

    return mutated


def append_best_to_summary(best_solution, best_loss, elapsed):
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, 'best_results.csv')

    row = pd.DataFrame([{
        "filters": best_solution["filters"],
        "kernel_size": best_solution["kernel_size"],
        "lstm_units": best_solution["lstm_units"],
        "dropout": best_solution["dropout"],
        "learning_rate": best_solution["learning_rate"],
        "batch_size": best_solution["batch_size"],
        "val_loss": best_loss,
        "method": "GA",
        "execution_time": round(elapsed, 4),
    }])

    write_header = not os.path.exists(summary_path)

    row.to_csv(
        summary_path,
        mode='a',
        header=write_header,
        index=False
    )


if __name__ == "__main__":
    best_loss = float("inf")
    best_solution = None

    start_time = time.perf_counter()

    population = create_population()

    for generation in range(NUM_GENERATIONS):
        fitnesses = []

        for individual in population:
            loss = evaluate_individual(individual)

            fitnesses.append(loss)

            results_log.append({
                "filters": individual["filters"],
                "kernel_size": individual["kernel_size"],
                "lstm_units": individual["lstm_units"],
                "dropout": individual["dropout"],
                "learning_rate": individual["learning_rate"],
                "batch_size": individual["batch_size"],
                "val_loss": loss,
                "method": "GA"
            })

            if loss < best_loss:
                best_loss = loss
                best_solution = individual.copy()

        sorted_population = [
            x for _, x in sorted(
                zip(fitnesses, population),
                key=lambda pair: pair[0]
            )
        ]

        new_population = sorted_population[:ELITISM]

        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            child = crossover(parent1, parent2)
            child = mutate(child)

            new_population.append(child)

        population = new_population

        _update_convergence(generation + 1, best_loss)
        _update_population_heatmap(population, generation + 1)

        print(
            f"Generation {generation + 1}/{NUM_GENERATIONS} "
            f"| Best Loss: {best_loss:.6f}"
        )

    elapsed = time.perf_counter() - start_time

    df = pd.DataFrame(results_log)

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    df.to_csv(
        os.path.join(results_dir, 'ga_results.csv'),
        index=False
    )

    append_best_to_summary(best_solution, best_loss, elapsed)

    plt.close('all')

    print(f"\nBest Solution: {best_solution}")
    print(f"Best Validation Loss: {best_loss:.6f}")
    print(f"Execution Time: {elapsed:.2f} seconds")
    print(f"Visualisations saved to {VIZ_DIR}")