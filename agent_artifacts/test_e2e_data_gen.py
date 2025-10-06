#!/usr/bin/env python3
import os
import sys


def main():
    # Ensure the moved package is on sys.path
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    e2e_dir = os.path.join(repo_dir, "e2e_sr")
    if e2e_dir not in sys.path:
        sys.path.insert(0, e2e_dir)

    import numpy as np
    # Import using the original package/module names expected by the codebase
    import parsers
    from symbolicregression.envs import build_env

    # Build default params without relying on CLI args
    params = parsers.get_parser().parse_args([])

    # Build environment and RNG for data generation
    env = build_env(params)
    env.rng = np.random.RandomState(params.env_base_seed or 0)

    # Generate and print 10 expressions using the e2e data generation procedure
    n = 100
    print(f"Generating {n} expressions using e2e data generation...\n")
    for i in range(n):
        expr, _ = env.gen_expr(train=True)
        tree = expr["tree"]  # Node or NodeList
        # Print the infix form of the expression (Node.__str__ returns infix)
        print(f"{i+1:2d}. {tree}")


if __name__ == "__main__":
    main()
