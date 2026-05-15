import os
import time

import numpy as np
import pandas as pd
import pyswarms as ps

from model.data_loader import load_data
from model.model_builder import evaluate_model
from sklearn.model_selection import train_test_split

X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results_log = []


def objective_function(particles):
    global results_log
    scores = []

    for p in particles:
        params = [int(p[0]), int(p[1]), int(p[2]), float(p[3]), float(p[4]), int(p[5])]
        loss = evaluate_model(params, X_train, X_test, y_train, y_test)

        results_log.append({
            "filters": int(p[0]),
            "kernel_size": int(p[1]),
            "lstm_units": int(p[2]),
            "dropout": p[3],
            "learning_rate": p[4],
            "batch_size": int(p[5]),
            "val_loss": loss,
            "method": "PSO"
        })

        scores.append(loss)

    return np.array(scores)


bounds = (
    np.array([16, 2, 32, 0.1, 1e-4, 16]),
    np.array([128, 5, 128, 0.5, 1e-2, 64])
)

optimizer = ps.single.GlobalBestPSO(
    n_particles=10,
    dimensions=6,
    options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
    bounds=bounds
)


def append_best_to_summary(best_pos, best_cost, elapsed):
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, 'best_results.csv')

    row = pd.DataFrame([{
        "filters":        int(best_pos[0]),
        "kernel_size":    int(best_pos[1]),
        "lstm_units":     int(best_pos[2]),
        "dropout":        float(best_pos[3]),
        "learning_rate":  float(best_pos[4]),
        "batch_size":     int(best_pos[5]),
        "val_loss":       best_cost,
        "method":         "PSO",
        "execution_time": round(elapsed, 4),
    }])

    write_header = not os.path.exists(summary_path)
    row.to_csv(summary_path, mode='a', header=write_header, index=False)


if __name__ == "__main__":
    start_time = time.perf_counter()

    best_cost, best_pos = optimizer.optimize(objective_function, iters=10)

    elapsed = time.perf_counter() - start_time

    df = pd.DataFrame(results_log)
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, 'pso_results.csv'), index=False)

    append_best_to_summary(best_pos, best_cost, elapsed)
