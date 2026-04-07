import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor


def load_and_preprocess_data(results_dir):
    files_to_load = {
        'PSO': 'pso_results.csv',
        'Random Search': 'random_search_results.csv',
        'Grid Search': 'grid_search_results.csv'
    }

    df_list = []
    for method, filename in files_to_load.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['method'] = method
            df_list.append(df)

    if not df_list:
        return None

    return pd.concat(df_list, ignore_index=True)


def plot_hpo_dashboard(all_results):
    """Generates the 3-panel dashboard for HPO analysis."""
    clean_results = all_results[all_results['val_loss'] <= 200].copy()

    features = ['filters', 'kernel_size', 'lstm_units', 'dropout', 'learning_rate', 'batch_size']

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # This shows the probability of finding a specific loss level
    sns.ecdfplot(data=clean_results, x='val_loss', hue='method', ax=axes[0], linewidth=3)
    axes[0].set_title('Search Reliability (ECDF)', fontsize=15, fontweight='bold')
    axes[0].set_xlabel('Validation MSE (Lower is Better)')
    axes[0].set_ylabel('Proportion of Trials')

    # We use a Random Forest to see which setting "moves the needle" the most
    X = all_results[features]
    y = all_results['val_loss']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    importances = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    sns.barplot(data=importances, x='Importance', y='Feature', ax=axes[1], palette='magma')
    axes[1].set_title('Which Setting Matters Most?', fontsize=15, fontweight='bold')
    axes[1].set_xlabel('Relative Importance (%)')

    # Shows if a parameter has a linear relationship with success
    corr = clean_results[features + ['val_loss']].corr()[['val_loss']].drop('val_loss')
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=axes[2], fmt=".2f")
    axes[2].set_title('Parameter Correlation with Loss', fontsize=15, fontweight='bold')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    results_path = os.path.join(current_dir, '..', 'results')

    print("--- Loading HPO Results ---")
    data = load_and_preprocess_data(results_path)

    if data is not None:
        print(f"Successfully loaded {len(data)} total trials.")
        plot_hpo_dashboard(data)
    else:
        print("Error: No result files found in the 'results' directory.")