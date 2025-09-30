#!/usr/bin/env python3
"""
Run BasicSR on expressions and data sampled from the e2e generator.

Procedure:
- Generate 10 expressions via E2EDataGenerator (with default knobs).
- For each expression, use the generated fit data (or regenerate) and run BasicSR.
- Print final MSE and the best evolved expression.
"""
import os
import sys
import numpy as np


def _ensure_local_paths():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    e2e_dir = os.path.join(repo_dir, "e2e_sr")
    if e2e_dir not in sys.path:
        sys.path.insert(0, e2e_dir)


_ensure_local_paths()

from e2e_data_gen import E2EDataGenerator
from basic_sr import BasicSR


def main(n: int = 10, seed: int = 0):
    # Make sampling deterministic-ish
    np.random.seed(seed)

    # Build e2e generator; allow non-controller mode to vary ops/dims
    gen = E2EDataGenerator(env_overrides={
        "env_base_seed": seed,
        "use_controller": False,
        # Restrict to add/sub/mul only and target low complexity
        "allowed_binary_operators": "add,sub,mul",
        # "allowed_unary_operators": "abs,sqrt,tan,inv,sin,cos",
        "allowed_unary_operators": "",
        "complexity": 0.3,
    })

    # Iterate over expressions
    for i in range(n):
        sample = gen.generate_expression(train=True)
        expr_str = sample["expr_str"]
        tree = sample["tree"]

        # Use the first fit split; fall back to generating on-the-fly
        X_list = sample.get("X_to_fit", [])
        Y_list = sample.get("Y_to_fit", [])
        if X_list and Y_list:
            X = X_list[0]
            Y = Y_list[0]
        else:
            data = gen.generate_data_for_tree(tree=tree, n_input_points=64)
            X, Y = data["fit"]

        # Flatten Y to 1-D if needed
        Y = np.asarray(Y)
        if Y.ndim == 2 and Y.shape[1] == 1:
            y = Y[:, 0]
        elif Y.ndim == 1:
            y = Y
        else:
            # If multi-output, fit first output
            y = Y[:, 0]

        print(f"\n[{i+1:02d}] Target expr: {expr_str}")
        print(f"    Data: X={X.shape}, y-range=[{float(np.min(y)):.3g}, {float(np.max(y)):.3g}]")

        # Configure and run BasicSR
        model = BasicSR(
            population_size=20,
            num_generations=10000,
            max_depth=10,
            max_size=25,
            tournament_size=3,
            collect_trajectory=False,
            time_limit=None,
            early_stop=True,
            early_stop_threshold=3e-16,
            min_generations=2,
            binary_operators=['+', '-', '*'],
            # unary_operators=['abs', 'sqrt', 'cos', 'sin', 'inv', 'tan'],
            unary_operators=[],
            record_heritage=True,
        )

        model.fit(X, y, verbose=True)
        y_pred = model.predict(X)
        mse = float(np.mean((y - y_pred) ** 2))
        print(f"    Best MSE={mse:.3e}; Best Expr={model.best_model_}")

        # print out the heritage
        heritage = model.retrieve_heritage_of_best_expression()
        print([len(g) for g in heritage])


if __name__ == "__main__":
    main(n=10, seed=1)
