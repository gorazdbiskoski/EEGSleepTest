import os
import sys

# Allow running this script directly (outside PyCharm), which otherwise puts
# only this folder on sys.path and cannot import the repo-root `data`/`model`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import time
import itertools
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data.global_data_loader import get_data_all_datasets
from model.metrics import metrics_from
from model.model_builder import evaluate_model
from results_io import append_best_row

# 2*2*2*2*3*2 = 96 combinations, in line with the ~100-evaluation budget the
# other optimizers use. Every value is drawn from the shared PARAM_SPACE the
# other eight scripts search, so results stay directly comparable. The previous
# grid was 486 combinations and was the only one offering kernel_size=1, a
# degenerate convolution no other optimizer could select.
PARAM_SPACE = {
    "filters": [16, 64],
    "kernel_size": [2, 5],
    "lstm_units": [32, 64],
    "dropout": [0.1, 0.3],
    "learning_rate": [0.01, 0.001, 0.0001],
    "batch_size": [16, 64]
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
VIZ_DIR = os.path.join(os.path.dirname(__file__), 'convergence plots')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)


def   run_grid_search(dataset_name, X, y):
    # First split holds out a test set the search never sees; the second
    # carves off the validation set every trial is scored on. The six other
    # optimizers already did this -- tuning directly against X_test made
    # their val_loss figures incomparable with these.
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    results_log = []
    best_loss = float('inf')
    best_metrics = None
    best_params = None
    conv_x, conv_y = [], []

    fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
    ax_conv.set_title(f'Grid Search – Convergence ({dataset_name})')
    ax_conv.set_xlabel('Trial #')
    ax_conv.set_ylabel('Best val_loss')
    conv_line, = ax_conv.plot([], [], color='mediumpurple')

    def update_convergence(trial, current_best_loss):
        conv_x.append(trial)
        conv_y.append(current_best_loss)
        conv_line.set_data(conv_x, conv_y)
        ax_conv.relim()
        ax_conv.autoscale_view()
        fig_conv.tight_layout()
        filename = f"grid_search_convergence_{dataset_name.lower().replace('-', '_')}.png"
        fig_conv.savefig(os.path.join(VIZ_DIR, filename), dpi=100)

    all_combinations = list(itertools.product(
        PARAM_SPACE["filters"],
        PARAM_SPACE["kernel_size"],
        PARAM_SPACE["lstm_units"],
        PARAM_SPACE["dropout"],
        PARAM_SPACE["learning_rate"],
        PARAM_SPACE["batch_size"]
    ))

    total_trials = len(all_combinations)
    print(f"Starting Grid Search on {dataset_name}. Total combinations: {total_trials}")

    start_time = time.perf_counter()

    for i, params in enumerate(all_combinations):
        f, k, u, d, lr, b = params
        print(f"Trial {i + 1}/{total_trials} | Filters={f}, Kernel={k}, LSTM={u}, Drop={d}, LR={lr}, Batch={b}")

        result = evaluate_model(params, X_train, X_val, y_train, y_val)
        loss = result["val_loss"]
        acc = result["val_accuracy"]
        print(f"  -> Validation Loss: {loss:.4f} | Accuracy: {acc:.4f}")

        results_log.append({
            "dataset": dataset_name,
            "trial": i + 1,
            "filters": f,
            "kernel_size": k,
            "lstm_units": u,
            "dropout": d,
            "learning_rate": lr,
            "batch_size": b,
            "val_loss": loss,
            **metrics_from(result),
            "epochs_used": result["epochs"],
            "method": "Grid Search"
        })

        if loss < best_loss:
            best_loss = loss
            best_metrics = metrics_from(result)
            best_params = {
                "filters": f,
                "kernel_size": k,
                "lstm_units": u,
                "dropout": d,
                "learning_rate": lr,
                "batch_size": b
            }

        update_convergence(i + 1, best_loss)

    elapsed = time.perf_counter() - start_time
    plt.close(fig_conv)
    return (results_log, best_params, best_loss, best_metrics, elapsed)


def append_best_to_summary(best_params, best_loss, best_metrics, elapsed, dataset_name):
    summary_path = os.path.join(RESULTS_DIR, 'best_results.csv')
    row = pd.DataFrame([{
        "dataset": dataset_name,
        "filters": best_params["filters"],
        "kernel_size": best_params["kernel_size"],
        "lstm_units": best_params["lstm_units"],
        "dropout": best_params["dropout"],
        "learning_rate": best_params["learning_rate"],
        "batch_size": best_params["batch_size"],
        "val_loss": best_loss,
        **best_metrics,
        "method": "Grid Search",
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
        (results, best_params, best_loss, best_metrics, elapsed) = run_grid_search(dataset_name, X, y)
        df = pd.DataFrame(results)
        filename = f"grid_search_results_{dataset_name.lower().replace('-', '_')}.csv"
        df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)
        append_best_to_summary(best_params, best_loss, best_metrics, elapsed, dataset_name)