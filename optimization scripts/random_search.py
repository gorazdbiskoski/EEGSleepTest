import os
import time
import random
import pandas as pd

from model.data_loader import load_data
from model.model_builder import evaluate_model
from sklearn.model_selection import train_test_split

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results_log = []

space_filters = [16, 32, 64, 128]
space_kernel = [2, 3, 4, 5]
space_lstm = [32, 64, 128]
space_dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
space_lr = [0.01, 0.005, 0.001, 0.0005, 0.0001]
space_batch = [16, 32, 64]


def run_random_search(num_trials=20):
    best_loss = float('inf')
    best_params = None

    for i in range(num_trials):
        f = random.choice(space_filters)
        k = random.choice(space_kernel)
        u = random.choice(space_lstm)
        d = random.choice(space_dropout)
        lr = random.choice(space_lr)
        b = random.choice(space_batch)

        params = [f, k, u, d, lr, b]

        print(f"Trial {i + 1}/{num_trials} | Testing: Filters={f}, Kernel={k}, LSTM={u}, Drop={d}, LR={lr}, Batch={b}")

        loss = evaluate_model(params, X_train, X_test, y_train, y_test)
        print(f"  -> Validation Loss: {loss:.4f}")

        results_log.append({
            "filters": f,
            "kernel_size": k,
            "lstm_units": u,
            "dropout": d,
            "learning_rate": lr,
            "batch_size": b,
            "val_loss": loss,
            "method": "Random Search"
        })

        if loss < best_loss:
            best_loss = loss
            best_params = params

    return best_loss, best_params


def append_best_to_summary(best_params, best_loss, elapsed):
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, 'best_results.csv')

    f, k, u, d, lr, b = best_params
    row = pd.DataFrame([{
        "filters":        f,
        "kernel_size":    k,
        "lstm_units":     u,
        "dropout":        d,
        "learning_rate":  lr,
        "batch_size":     b,
        "val_loss":       best_loss,
        "method":         "Random Search",
        "execution_time": round(elapsed, 4),
    }])

    write_header = not os.path.exists(summary_path)
    row.to_csv(summary_path, mode='a', header=write_header, index=False)


if __name__ == "__main__":
    start_time = time.perf_counter()

    best_cost, best_pos = run_random_search(num_trials=100)

    elapsed = time.perf_counter() - start_time

    df = pd.DataFrame(results_log)
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, 'random_search_results.csv'), index=False)

    append_best_to_summary(best_pos, best_cost, elapsed)
