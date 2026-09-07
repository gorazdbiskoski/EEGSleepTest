"""Cross-optimizer dashboard over the per-trial HPO results.

One dashboard per dataset. Every optimizer writes
``results/<method>_results_<dataset>.csv`` with a ``method`` column it fills in
itself, so both the file discovery and the method labels come from the run
rather than being hard-coded here -- the previous version looked for
``pso_results.csv`` and friends without the dataset suffix, matched nothing,
and always reported "No result files found".
"""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

FEATURES = ['filters', 'kernel_size', 'lstm_units', 'dropout', 'learning_rate', 'batch_size']

# Written by every optimizer since the metrics change; macro-averaged, so each
# sleep stage counts equally regardless of how rare it is.
METRICS = ['val_accuracy', 'precision', 'recall', 'f1']

DATASETS = {'haaglanden': 'Haaglanden', 'sleep_edf': 'Sleep-EDF'}


def load_results(results_dir, dataset_suffix):
    """Every optimizer's per-trial rows for one dataset, concatenated."""
    pattern = os.path.join(results_dir, f'*_results_{dataset_suffix}.csv')
    paths = sorted(glob.glob(pattern))

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if 'method' not in frame.columns:
            print(f"  skipping {os.path.basename(path)}: no 'method' column")
            continue
        frames.append(frame)

    if not frames:
        return None

    print(f"  loaded {len(frames)} file(s), {sum(len(f) for f in frames)} trials")
    return pd.concat(frames, ignore_index=True)


def plot_dashboard(results, dataset_label, output_dir):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    fig.suptitle(f'Hyperparameter search - {dataset_label}', fontsize=17, fontweight='bold')

    sns.ecdfplot(data=results, x='val_loss', hue='method', ax=axes[0], linewidth=2)
    axes[0].set_title('Search reliability (ECDF)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Validation loss (lower is better)')
    axes[0].set_ylabel('Proportion of trials')

    has_f1 = 'f1' in results.columns
    if has_f1:
        best = (results.groupby('method')['f1'].max().sort_values(ascending=False).reset_index())
        sns.barplot(data=best, x='f1', y='method', ax=axes[1], hue='method',
                    palette='crest', legend=False)
        axes[1].set_title('Best macro F1 per method', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Macro F1 (higher is better)')
        axes[1].set_ylabel('')
    else:
        axes[1].set_visible(False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(results[FEATURES], results['val_loss'])
    importances = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': model.feature_importances_,
    }).sort_values('Importance', ascending=False)

    sns.barplot(data=importances, x='Importance', y='Feature', ax=axes[2], hue='Feature',
                palette='magma', legend=False)
    axes[2].set_title('Which setting matters most?', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Relative importance')

    columns = FEATURES + ['val_loss'] + [m for m in METRICS if m in results.columns]
    corr = results[columns].corr()[['val_loss']].drop('val_loss')
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=axes[3], fmt=".2f")
    axes[3].set_title('Correlation with validation loss', fontsize=13, fontweight='bold')

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'hpo_dashboard_{dataset_label.lower().replace("-", "_")}.png')
    fig.savefig(out_path, dpi=110)
    print(f"  wrote {out_path}")
    return fig


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base_dir, '..', 'results')
    output_path = os.path.join(base_dir, 'dashboards')

    any_loaded = False
    for suffix, label in DATASETS.items():
        print(f"--- {label} ---")
        data = load_results(results_path, suffix)
        if data is None:
            print(f"  no result files matching *_results_{suffix}.csv -- run the optimizers first")
            continue
        plot_dashboard(data, label, output_path)
        any_loaded = True

    if any_loaded:
        plt.show()
