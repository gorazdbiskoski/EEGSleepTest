import os
import time
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split

from model.data_loader import load_data
from model.model_builder import evaluate_model

X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results_log = []


def objective_function(
    filters,
    kernel_size,
    lstm_units,
    dropout,
    learning_rate,
    batch_size
):
    global results_log

    params = [
        int(filters),
        int(kernel_size),
        int(lstm_units),
        float(dropout),
        float(learning_rate),
        int(batch_size)
    ]

    loss = evaluate_model(
        params,
        X_train,
        X_test,
        y_train,
        y_test
    )

    results_log.append({
        "filters": int(filters),
        "kernel_size": int(kernel_size),
        "lstm_units": int(lstm_units),
        "dropout": float(dropout),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "val_loss": loss,
        "method": "Bayesian Optimization"
    })

    return -loss


pbounds = {
    "filters": (16, 128),
    "kernel_size": (2, 5),
    "lstm_units": (32, 128),
    "dropout": (0.1, 0.5),
    "learning_rate": (1e-4, 1e-2),
    "batch_size": (16, 64)
}


optimizer = BayesianOptimization(
    f=objective_function,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)


def append_best_to_summary(best_params, best_loss, elapsed):
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, 'best_results.csv')

    row = pd.DataFrame([{
        "filters":        int(best_params["filters"]),
        "kernel_size":    int(best_params["kernel_size"]),
        "lstm_units":     int(best_params["lstm_units"]),
        "dropout":        float(best_params["dropout"]),
        "learning_rate":  float(best_params["learning_rate"]),
        "batch_size":     int(best_params["batch_size"]),
        "val_loss":       best_loss,
        "method":         "Bayesian Optimization",
        "execution_time": round(elapsed, 4),
    }])

    write_header = not os.path.exists(summary_path)
    row.to_csv(summary_path, mode='a', header=write_header, index=False)


if __name__ == "__main__":
    start_time = time.perf_counter()

    optimizer.maximize(
        init_points=5,
        n_iter=10
    )

    elapsed = time.perf_counter() - start_time

    df = pd.DataFrame(results_log)

    results_dir = os.path.join(
        os.path.dirname(__file__),
        '..',
        'results'
    )

    os.makedirs(results_dir, exist_ok=True)

    df.to_csv(
        os.path.join(results_dir, 'bayesian_optimization_results.csv'),
        index=False
    )

    best_result = optimizer.max
    best_loss = -best_result["target"]
    best_params = best_result["params"]

    append_best_to_summary(best_params, best_loss, elapsed)
