import os
import time
import random
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

SPACES = [space_filters, space_kernel, space_lstm, space_dropout, space_lr, space_batch]
PARAM_NAMES = ["filters", "kernel_size", "lstm_units", "dropout", "learning_rate", "batch_size"]

VIZ_DIR = os.path.join(os.path.dirname(__file__), 'convergence plots')
os.makedirs(VIZ_DIR, exist_ok=True)

fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
ax_conv.set_title('Tabu Search – Convergence')
ax_conv.set_xlabel('Trial #')
ax_conv.set_ylabel('Best val_loss')
conv_x, conv_y = [], []
(conv_line,) = ax_conv.plot([], [], marker='o', color='steelblue')


def _update_convergence(trial, best_loss):
    conv_x.append(trial)
    conv_y.append(best_loss)
    conv_line.set_data(conv_x, conv_y)
    ax_conv.relim()
    ax_conv.autoscale_view()
    fig_conv.tight_layout()
    fig_conv.savefig(os.path.join(VIZ_DIR, 'tabu_search_convergence.png'), dpi=100)


def _get_neighbors(current):
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


def run_tabu_search(num_trials=100, tabu_tenure=10):
    current = [random.choice(s) for s in SPACES]
    current_loss = evaluate_model(current, X_train, X_test, y_train, y_test)

    best_params = current[:]
    best_loss = current_loss
    tabu_list = []
    trial = 1

    results_log.append({
        **dict(zip(PARAM_NAMES, current)),
        "val_loss": current_loss,
        "method": "Tabu Search"
    })
    _update_convergence(trial, best_loss)

    while trial < num_trials:
        neighbors = _get_neighbors(current)

        candidates = []
        for n in neighbors:
            if n not in tabu_list or evaluate_model(n, X_train, X_test, y_train, y_test) < best_loss:
                candidates.append(n)

        if not candidates:
            candidates = neighbors

        best_candidate = None
        best_candidate_loss = float('inf')

        for n in candidates:
            if trial >= num_trials:
                break
            trial += 1

            f, k, u, d, lr, b = n
            print(f"Trial {trial}/{num_trials} | "
                  f"Filters={f}, Kernel={k}, LSTM={u}, "
                  f"Drop={d}, LR={lr}, Batch={b}")

            loss = evaluate_model(n, X_train, X_test, y_train, y_test)
            print(f"  -> Validation Loss: {loss:.4f}")

            results_log.append({
                **dict(zip(PARAM_NAMES, n)),
                "val_loss": loss,
                "method": "Tabu Search"
            })

            if loss < best_candidate_loss:
                best_candidate_loss = loss
                best_candidate = n[:]

            if loss < best_loss:
                best_loss = loss
                best_params = n[:]

            _update_convergence(trial, best_loss)

        if best_candidate:
            current = best_candidate
            tabu_list.append(current[:])
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)

    return best_loss, best_params


def append_best_to_summary(best_params, best_loss, elapsed):
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, 'best_results.csv')
    f, k, u, d, lr, b = best_params
    row = pd.DataFrame([{
        "filters": f, "kernel_size": k, "lstm_units": u,
        "dropout": d, "learning_rate": lr, "batch_size": b,
        "val_loss": best_loss, "method": "Tabu Search",
        "execution_time": round(elapsed, 4),
    }])
    write_header = not os.path.exists(summary_path)
    row.to_csv(summary_path, mode='a', header=write_header, index=False)


if __name__ == "__main__":
    start_time = time.perf_counter()

    best_cost, best_pos = run_tabu_search(num_trials=100, tabu_tenure=10)

    elapsed = time.perf_counter() - start_time

    df = pd.DataFrame(results_log)
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, 'tabu_search_results.csv'), index=False)

    append_best_to_summary(best_pos, best_cost, elapsed)

    plt.close('all')
    print(f"Visualisations saved to {VIZ_DIR}")