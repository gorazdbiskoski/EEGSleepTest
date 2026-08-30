import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

OPTIMIZERS = [
    "ant_colony_optimization.py",
    "bayesian_optimization_algorithm.py",
    "genetic_algorithm.py",
    "grid_search.py",
    "hill_climbing.py",
    "hill_climbing_with_simulated_annealing.py",
    "pso_optimizer.py",
    "random_search.py",
    "simulated_annealing.py",
    "tabu_search.py",
]

if __name__ == "__main__":
    # Children import `data` / `model` from the repo root, which is not on
    # sys.path when Python is launched from this folder.
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    failures = []

    for script in OPTIMIZERS:
        script_path = os.path.join(BASE_DIR, script)
        print(f"\n{'=' * 70}\nRunning {script}\n{'=' * 70}", flush=True)

        result = subprocess.run(
            [sys.executable, script_path],
            check=False,
            cwd=BASE_DIR,
            env=env,
        )

        if result.returncode != 0:
            print(f"\n{script} exited with code {result.returncode}")
            failures.append((script, result.returncode))
        else:
            print(f"\n{script} completed successfully")

    print(f"\n{'=' * 70}")
    if failures:
        print(f"{len(failures)}/{len(OPTIMIZERS)} optimizers FAILED:")
        for script, code in failures:
            print(f"  - {script} (exit {code})")
        sys.exit(1)

    print(f"All {len(OPTIMIZERS)} optimizers completed successfully")
