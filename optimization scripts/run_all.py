import subprocess
import sys

OPTIMIZERS = [
    "ant_colony_optimization.py"
    "bayesian_optimization_algorithm.py",
    "grid_search.py",
    "pso_optimizer.py",
    "random_search.py",
]

if __name__ == "__main__":
    for script in OPTIMIZERS:
        result = subprocess.run([sys.executable, script], check=False)

        if result.returncode != 0:
            print(f"\n{script} exited with code {result.returncode}")
        else:
            print(f"\n{script} completed successfully")
