import multiprocessing
import subprocess
from itertools import product
import os
# Parameter values to sweep
SEEDS = [38 ,40 ,42 ,44, 46 ,48, 50 ,52 ,54 ,56]         # Add as many seeds as you want
EPS_VALUES = [0.9]  # Add as many epsilon values as you want
NOISE_LEVELS = [40 ,42, 44, 46, 48 ,50, 52, 54 ,56, 58, 60, 62, 64, 66, 68, 70]  # Add as many noise levels as you want
# NOISE_LEVELS = [0,10,15,20,25,30,35]  # Add as many noise levels as you want
# NOISE_LEVELS = [ 54 ]
N_Steps = 80
# Path to the evaluation script
EVAL_SCRIPT = "python scripts/eval_diffusion_maniskill.py"

def evaluate(params):
    seed, eps, noise_level = params
    print(f"Running with seed={seed}, noise_scale={eps}, noise_level={noise_level}")

    # Construct the command
    cmd = [
        "python", "scripts/eval_diffusion.py",
        f"seed={seed}",
        f"actor_kwargs.noise_scale={eps}",
        f"noise_level={noise_level}",
        f"n_steps={N_Steps}"
    ]

    # Run the command
    subprocess.run(cmd)

if __name__ == "__main__":
    # Generate combinations of parameters
    parameter_combinations = list(product(SEEDS, EPS_VALUES, NOISE_LEVELS))

    # Use multiprocessing to run evaluations in parallel
    print("Starting parallel evaluations...")
    max_processes = 16
    with multiprocessing.Pool(processes = max_processes) as pool:
        pool.map(evaluate, parameter_combinations)

    print("All parameter sweeps completed.")

