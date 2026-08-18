import os
import time
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from data.global_data_loader import get_data_all_datasets
from model.model_builder import evaluate_model

PARAM_BOUNDS = {
    "filters": (16, 128),
    "kernel_size": (2, 5),
    "lstm_units": (32, 128),
    "dropout": (0.1, 0.5),
    "learning_rate": (1e-4, 1e-2),
    "batch_size": (16, 64)
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
VIZ_DIR = os.path.join(os.path.dirname(__file__), 'convergence plots')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)


def run_bayesian_optimization(dataset_name, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results_log = []
    conv_x = []
    conv_y = []
    probe_counter = [0]
    running_best = [float('inf')]

    fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
    ax_conv.set_title(f'Bayesian Optimisation – {dataset_name}')
    ax_conv.set_xlabel('Probe #')
    ax_conv.set_ylabel('Best val_loss')
    conv_line, = ax_conv.plot([], [], marker='o', color='seagreen')

    def update_convergence(probe_num, best_loss):
        conv_x.append(probe_num)
        conv_y.append(best_loss)
        conv_line.set_data(conv_x, conv_y)
        ax_conv.relim()
        ax_conv.autoscale_view()
        fig_conv.tight_layout()
        filename = f"bayesian_convergence_{dataset_name.lower().replace('-', '_')}.png"
        fig_conv.savefig(os.path.join(VIZ_DIR, filename), dpi=100)

    def objective_function(filters, kernel_size, lstm_units, dropout, learning_rate, batch_size):
        params = [
            int(filters),
            int(kernel_size),
            int(lstm_units),
            float(dropout),
            float(learning_rate),
            int(batch_size)
        ]
        loss = evaluate_model(params, X_train, X_test, y_train, y_test)

        results_log.append({
            "dataset": dataset_name,
            "probe": probe_counter[0] + 1,
            "filters": int(filters),
            "kernel_size": int(kernel_size),
            "lstm_units": int(lstm_units),
            "dropout": float(dropout),
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "val_loss": loss,
            "method": "Bayesian Optimization"
        })

        probe_counter[0] += 1
        if loss < running_best[0]:
            running_best[0] = loss
        update_convergence(probe_counter[0], running_best[0])
        return -loss

    optimizer = BayesianOptimization(f=objective_function, pbounds=PARAM_BOUNDS, random_state=42, verbose=0)

    start_time = time.perf_counter()
    optimizer.maximize(init_points=5, n_iter=10)
    elapsed = time.perf_counter() - start_time

    best_result = optimizer.max
    best_loss = -best_result["target"]
    best_params = best_result["params"]

    plt.close(fig_conv)
    return (results_log, best_params, best_loss, elapsed)


def append_best_to_summary(best_params, best_loss, elapsed, dataset_name):
    summary_path = os.path.join(RESULTS_DIR, 'best_results.csv')
    row = pd.DataFrame([{
        "dataset": dataset_name,
        "filters": int(best_params["filters"]),
        "kernel_size": int(best_params["kernel_size"]),
        "lstm_units": int(best_params["lstm_units"]),
        "dropout": float(best_params["dropout"]),
        "learning_rate": float(best_params["learning_rate"]),
        "batch_size": int(best_params["batch_size"]),
        "val_loss": best_loss,
        "method": "Bayesian Optimization",
        "execution_time": round(elapsed, 4)
    }])
    write_header = not os.path.exists(summary_path)
    row.to_csv(summary_path, mode='a', header=write_header, index=False)


if __name__ == "__main__":
    haaglanden, sleep_edfx = get_data_all_datasets()
    datasets = {
        "Haaglanden": haaglanden,
        "Sleep-EDF": sleep_edfx
    }

    for dataset_name, dataset in datasets.items():
        X, y = dataset
        (results, best_params, best_loss, elapsed) = run_bayesian_optimization(dataset_name, X, y)
        df = pd.DataFrame(results)
        filename = f"bayesian_optimization_results_{dataset_name.lower().replace('-', '_')}.csv"
        df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)
        append_best_to_summary(best_params, best_loss, elapsed, dataset_name)