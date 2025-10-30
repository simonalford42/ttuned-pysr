#!/usr/bin/env python3
"""
Generate synthetic expressions for training data using the e2e environment.

This module provides:
- E2EDataGenerator: Thin wrapper over the e2e FunctionEnvironment for sampling expressions and data
- Functions to generate and save expression datasets with configurable operators and complexity

It mirrors the knobs used by the e2e generator (operators, complexity,
dimensions, distributions, noise, etc.) by letting you override any of the
underlying environment parameters on initialization, and per-call overrides for
key sampling controls.
"""
import numpy as np
import json
import pickle
import gzip
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union, Iterable
from tqdm import tqdm


def _ensure_e2e_on_path():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    e2e_dir = os.path.join(repo_dir, "e2e_sr")
    if e2e_dir not in sys.path:
        sys.path.insert(0, e2e_dir)


_ensure_e2e_on_path()

# Import e2e modules after path setup
import parsers  # type: ignore
from symbolicregression.envs import build_env  # type: ignore
from symbolicregression.envs.generators import (
    Node,
    NodeList,
    operators_real,
    operators_extra,
)  # type: ignore


class E2EDataGenerator:
    """
    Thin wrapper over the e2e FunctionEnvironment for sampling expressions and data.

    Override any environment parameter via kwargs at init. Common useful knobs include:
    - operators_to_downsample, operators_to_not_repeat, required_operators
    - allowed_unary_operators, allowed_binary_operators, disabled_operators
    - extra_unary_operators, extra_binary_operators, extra_constants
    - min_input_dimension, max_input_dimension, min_output_dimension, max_output_dimension
    - min_binary_ops_per_dim, max_binary_ops_per_dim, min_unary_ops, max_unary_ops
    - complexity (0..1) to scale overall ops counts
    - use_sympy, use_abs, prob_const, prob_rand, max_int, max_len, min_len_per_dim
    - max_centroids, prediction_sigmas, max_trials, n_prediction_points
    - train_noise_gamma, eval_noise_gamma, env_base_seed
    """

    def __init__(self, env_overrides: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        # Build default params, then apply overrides
        self.params = parsers.get_parser().parse_args([])

        # Preprocess convenience knobs in env_overrides before building env
        allowed_unaries = None
        allowed_binaries = None
        disabled_ops = set()
        complexity = None

        if env_overrides:
            # Extract convenience keys if present
            def _coerce_list(x: Any) -> Optional[Iterable[str]]:
                if x is None:
                    return None
                if isinstance(x, str):
                    return [s.strip() for s in x.split(",") if s.strip()]
                if isinstance(x, Iterable):
                    return [str(s).strip() for s in x]
                return None

            if "allowed_unary_operators" in env_overrides:
                allowed_unaries = _coerce_list(env_overrides.get("allowed_unary_operators"))
                env_overrides.pop("allowed_unary_operators", None)
            if "allowed_binary_operators" in env_overrides:
                allowed_binaries = _coerce_list(env_overrides.get("allowed_binary_operators"))
                env_overrides.pop("allowed_binary_operators", None)
            if "disabled_operators" in env_overrides:
                disabled_ops = set(_coerce_list(env_overrides.get("disabled_operators")) or [])
                env_overrides.pop("disabled_operators", None)
            if "complexity" in env_overrides:
                complexity = float(env_overrides.get("complexity"))
                env_overrides.pop("complexity", None)

            # Apply remaining overrides directly onto argparse Namespace
            for k, v in env_overrides.items():
                if hasattr(self.params, k):
                    setattr(self.params, k, v)
                else:
                    raise ValueError(f"Unknown environment parameter: {k}")

        # Translate allowed_* and disabled_* into operators_to_downsample
        if allowed_unaries is not None or allowed_binaries is not None or disabled_ops:
            # Determine full sets
            all_unaries = [o for o, a in operators_real.items() if abs(a) == 1]
            all_binaries = [o for o, a in operators_real.items() if abs(a) == 2]
            # include extras
            all_binaries += [o for o, a in operators_extra.items() if abs(a) == 2]

            zero_ops = set()
            if allowed_unaries is not None:
                zero_ops.update(set(all_unaries) - set(allowed_unaries))
            if allowed_binaries is not None:
                zero_ops.update(set(all_binaries) - set(allowed_binaries))
            zero_ops.update(disabled_ops)

            # Avoid zeroing all unaries to keep probability vector valid; we will still set max_unary_ops=0
            # For safety, if zero_ops covers all unaries, drop one common unary from zeroing
            if set(all_unaries).issubset(zero_ops) and len(all_unaries) > 0:
                zero_ops.discard("sin")  # leave one with non-zero prob
                self.params.max_unary_ops = 0 # user likely wants no unaries

            downsample_pairs = [f"{op}_0" for op in sorted(zero_ops)]
            self.params.operators_to_downsample = ",".join(downsample_pairs)

        # Apply complexity scaling to ops counts if requested (0..1)
        if complexity is not None:
            c = max(0.0, min(1.0, float(complexity)))
            # Scale binary ops per dim and offset (base defaults ~4)
            base_cap = 4
            scaled = max(1, int(round(c * base_cap)))
            self.params.min_binary_ops_per_dim = 0
            self.params.max_binary_ops_per_dim = scaled
            self.params.max_binary_ops_offset = scaled
            # Scale unary ops upper bound
            self.params.min_unary_ops = getattr(self.params, "min_unary_ops", 0)
            self.params.max_unary_ops = int(round(c * max(1, getattr(self.params, "max_unary_ops", 4))))

        # Seed selection: prefer explicit seed argument, fallback to env_base_seed
        base_seed = seed if seed is not None else (self.params.env_base_seed or 0)

        # Build environment and RNG
        self.env = build_env(self.params)
        self.env.rng = np.random.RandomState(base_seed)

    # ---------------------- Expression Generation ----------------------
    def generate_expression(
        self,
        train: bool = True,
        nb_binary_ops: Optional[Union[int, Tuple[int, ...]]] = None,
        nb_unary_ops: Optional[Union[int, Tuple[int, ...]]] = None,
        input_dimension: Optional[int] = None,
        output_dimension: Optional[int] = None,
        n_input_points: Optional[int] = None,
        input_distribution_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Sample a single expression from the environment with optional per-call overrides.

        Returns a dict including:
        - tree: Node or NodeList (e2e expression tree)
        - expr_str: human-readable infix string
        - infos: metadata (ops counts, dims, length, distribution, etc.)
        - X_to_fit, Y_to_fit: lists of numpy arrays (slices by points kept)
        - x_to_predict_*, y_to_predict_*: optional prediction sets if enabled
        - tree_encoded, skeleton_tree, skeleton_tree_encoded: tokens and skeletons
        """
        expr, _errors = self.env.gen_expr(
            train=train,
            nb_binary_ops=nb_binary_ops,
            nb_unary_ops=nb_unary_ops,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            n_input_points=n_input_points,
            input_distribution_type=input_distribution_type,
        )
        tree = expr["tree"]
        expr_str = str(tree)
        # Attach a convenient field
        result = dict(expr)
        result["expr_str"] = expr_str
        return result

    # ---------------------- Data Generation ----------------------
    def generate_data_for_tree(
        self,
        tree: Union[Node, NodeList],
        n_input_points: int,
        n_prediction_points: int = 0,
        prediction_sigmas: Optional[str] = None,
        input_distribution_type: str = "gaussian",
        n_centroids: Optional[int] = None,
        rotate: bool = True,
        offset: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate data for a provided e2e tree using the e2e generator.

        - tree: Node/NodeList returned by generate_expression()
        - n_input_points: number of fit points to sample
        - n_prediction_points: number of prediction points per sigma
        - prediction_sigmas: comma-separated string of sigma multipliers (overrides env)
        - input_distribution_type: 'gaussian' or 'uniform'
        - n_centroids: number of mixture components for input distribution (overrides env)
        - rotate: rotate input distribution via random orthogonal matrices
        - offset: optional (mean, std) scaling applied to inputs

        Returns a dict like {'fit': (X_fit, Y_fit), 'predict_<sigma>': (X_pred, Y_pred), ...}
        """
        # Temporarily override env params for prediction points / sigmas
        orig_n_pred = self.params.n_prediction_points
        orig_sigmas = self.params.prediction_sigmas
        try:
            self.params.n_prediction_points = int(n_prediction_points)
            if prediction_sigmas is not None:
                self.params.prediction_sigmas = prediction_sigmas

            # Parse prediction sigmas for generator API
            if self.params.prediction_sigmas is None:
                sigmas = []
            else:
                sigmas = [float(s) for s in str(self.params.prediction_sigmas).split(",") if str(s).strip()]

            # Sample mixture count if not given
            if n_centroids is None:
                n_centroids = int(self.env.rng.randint(1, self.params.max_centroids))

            tree_out, datapoints = self.env.generator.generate_datapoints(
                tree=tree,
                rng=self.env.rng,
                input_dimension=self.env.generator.relabel_variables(tree),
                n_input_points=n_input_points,
                n_prediction_points=n_prediction_points,
                prediction_sigmas=sigmas,
                input_distribution_type=input_distribution_type,
                n_centroids=int(n_centroids),
                max_trials=self.params.max_trials,
                rotate=rotate,
                offset=offset,
            )
            if datapoints is None:
                raise RuntimeError("Data generation failed for the provided tree")
            return datapoints
        finally:
            self.params.n_prediction_points = orig_n_pred
            self.params.prediction_sigmas = orig_sigmas


def generate_e2e_expressions(
    n_expressions: int = 100,
    binary_ops: str = "add,sub,mul",
    unary_ops: str = "",
    complexity: float = 0.5,
    n_input_points: int = None,
    seed: int = 42,
    constants: List[float] = [1.0]
) -> List[Dict[str, Any]]:
    """
    Generate expressions using E2EDataGenerator.

    Args:
        n_expressions: Number of expressions to generate
        binary_ops: Comma-separated binary operators (e.g., "add,sub,mul")
        unary_ops: Comma-separated unary operators (e.g., "abs,sqrt,sin,cos,tan,inv")
        complexity: Target complexity for expression generation (0.0 to 1.0)
        n_input_points: Number of data points to generate per expression
        seed: Random seed
        constants: List of constants to use in expressions (default: [1.0])

    Returns:
        List of expression dicts with id, expression, X_data, y_data, metadata
    """
    np.random.seed(seed)

    # Build e2e generator with constant control
    env_overrides = {
        "env_base_seed": seed,
        "use_controller": False,
        "allowed_binary_operators": binary_ops,
        "allowed_unary_operators": unary_ops,
        "extra_constants": ",".join(str(c) for c in constants) if constants else None,
        "complexity": complexity,
    }

    # Control constant generation
    if not constants or len(constants) == 0:
        env_overrides["prob_const"] = 0.0  # Disable constant generation
    else:
        env_overrides["prob_const"] = 0.2  # Reasonable default for constant generation

    gen = E2EDataGenerator(env_overrides=env_overrides)

    expressions = []

    for i in tqdm(range(n_expressions)):
        # Generate expression with data
        sample = gen.generate_expression(train=True, n_input_points=n_input_points)
        expr_str = sample["expr_str"]
        tree = sample["tree"]

        # Get data
        X_list = sample.get("X_to_fit", [])
        Y_list = sample.get("Y_to_fit", [])

        if X_list and Y_list:
            X = X_list[0]
            Y = Y_list[0]
        else:
            assert 0, "n_input_points doesnt work here"
            # Generate fresh data if not provided
            data = gen.generate_data_for_tree(tree=tree, n_input_points=n_input_points)
            X, Y = data["fit"]

        # Flatten Y to 1-D if needed
        Y = np.asarray(Y)
        if Y.ndim == 2 and Y.shape[1] == 1:
            y = Y[:, 0]
        elif Y.ndim == 1:
            y = Y
        else:
            y = Y[:, 0]

        # Store expression with data (keep as numpy arrays for efficient pickle storage)
        expressions.append({
            'id': i,
            'expression': expr_str,
            'X_data': X.astype(np.float32),  # Use float32 for smaller size
            'y_data': y.astype(np.float32),
        })

    return expressions


def save_expressions(expressions: List[Dict], output_file: str, metadata: Dict):
    """Save expressions with X/y data to file with metadata using compressed pickle format"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    data = {
        'metadata': metadata,
        'expressions': expressions  # Now includes X_data, y_data, and per-expression metadata
    }

    # Use gzipped pickle for ~9x compression vs JSON
    with gzip.open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Saved {len(expressions)} expressions to {output_file} ({file_size_mb:.2f} MB)")
    return expressions


def generate_training_expressions(n_expressions: int = 100,
                                 binary_ops: str = "add,sub,mul",
                                 unary_ops: str = "",
                                 complexity: float = 0.5,
                                 n_input_points: int = None,
                                 seed: int = 42,
                                 constants: List[float] = [1.0],
                                 output_dir: str = "datasets/expressions",
                                 demo=False) -> str:
    """Generate training expressions with data and save to file"""

    print(f"Generating {n_expressions} training expressions...")
    print(f"Parameters: binary_ops={binary_ops}, unary_ops={unary_ops or 'none'}, complexity={complexity}")
    print(f"Constants: {constants if constants else 'none'}")
    print(f"Data points per expression: {n_input_points}")

    # Generate expressions
    expressions = generate_e2e_expressions(
        n_expressions=n_expressions,
        binary_ops=binary_ops,
        unary_ops=unary_ops,
        complexity=complexity,
        n_input_points=n_input_points,
        seed=seed,
        constants=constants
    )

    if demo:
        print("\nGenerated Expressions:")
        for i, expr in enumerate(expressions):
            print(f"{expr['expression']}")
        return

    # Create metadata
    const_str = ','.join(str(c) for c in constants) if constants else ''
    metadata = {
        'generation_command': f"python generate_expressions.py --n_expressions={n_expressions} --binary_ops={binary_ops} --unary_ops={unary_ops} --complexity={complexity} --n_input_points={n_input_points} --seed={seed} --constants={const_str}",
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'n_expressions': n_expressions,
            'binary_ops': binary_ops,
            'unary_ops': unary_ops,
            'complexity': complexity,
            'n_input_points': n_input_points,
            'seed': seed,
            'constants': constants
        },
        'generator_version': '2.0-e2e'
    }

    # Generate descriptive filename
    # Size: 1k, 10k, 100k, 1M, etc.
    if n_expressions >= 1_000_000:
        size_str = f"{n_expressions // 1_000_000}M"
    elif n_expressions >= 1_000:
        size_str = f"{n_expressions // 1_000}k"
    else:
        size_str = str(n_expressions)

    # Operators: use readable names
    # Either basic arithmetic (add,sub,mul) or full ops (add,sub,mul,div,pow + unary)
    if unary_ops:
        ops_str = "full"
    else:
        ops_str = "arith"

    # Complexity: c05, c10, etc.
    complexity_str = f"c{int(complexity * 10):02d}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"{output_dir}/train_{size_str}_{ops_str}_{complexity_str}_{timestamp}.pkl.gz"
    filename = f"{output_dir}/{ops_str}_{size_str}_{complexity_str}_{timestamp}.pkl.gz"

    # Save expressions
    save_expressions(expressions, filename, metadata)

    print(f"✓ Generated {len(expressions)} expressions with data")
    print(f"✓ Saved to {filename}")

    return filename


def demo_generate_n(n: int = 10) -> None:
    """Example: generate N expressions with data and print a brief summary."""
    gen = E2EDataGenerator(env_overrides={"env_base_seed": 0, "use_controller": False})

    print(f"Generating {n} expressions with data...\n")
    for i in range(n):
        sample = gen.generate_expression(
            train=True,
            # Per-sample overrides are optional; here we let the generator choose
            # nb_binary_ops=None,
            # nb_unary_ops=None,
            # input_dimension=None,
            # output_dimension=None,
            # n_input_points=None,
            # input_distribution_type=None,
        )
        expr_str = sample["expr_str"]
        tree = sample["tree"]
        X_fit_list = sample.get("X_to_fit", [])
        Y_fit_list = sample.get("Y_to_fit", [])
        X = X_fit_list[0] if X_fit_list else None
        Y = Y_fit_list[0] if Y_fit_list else None

        print(f"{i+1:2d}. {expr_str}")
        if X is not None and Y is not None:
            y_min = float(np.min(Y)) if Y.size else float("nan")
            y_max = float(np.max(Y)) if Y.size else float("nan")
            print(f"    X: {X.shape}, Y: {Y.shape}, y-range=[{y_min:.3g}, {y_max:.3g}]")
        else:
            # If you prefer fresh data using custom knobs, you can do:
            data = gen.generate_data_for_tree(
                tree=tree,
                n_input_points=50,
                n_prediction_points=0,
                input_distribution_type="gaussian",
            )
            X, Y = data["fit"]
            y_min = float(np.min(Y)) if Y.size else float("nan")
            y_max = float(np.max(Y)) if Y.size else float("nan")
            print(f"    X: {X.shape}, Y: {Y.shape}, y-range=[{y_min:.3g}, {y_max:.3g}]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic expressions using e2e")
    parser.add_argument("--n_expressions", type=int, default=100, help="Number of expressions to generate")
    parser.add_argument("--binary_ops", type=str, default="add,sub,mul", help="Comma-separated binary operators (e.g., add,sub,mul,div,pow)")
    parser.add_argument("--unary_ops", type=str, default="", help="Comma-separated unary operators (e.g., abs,sqrt,sin,cos,tan,inv)")
    parser.add_argument("--complexity", type=float, default=0.5, help="Complexity level (0.0 to 1.0)")
    parser.add_argument("--n_input_points", type=int, default=None, help="Number of data points per expression")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--demo", action="store_true", help="Instead of saving to file, print out the expressions and then finish")
    parser.add_argument("--output_dir", default="datasets/expressions", help="Output directory")
    parser.add_argument("--constants", type=str, default="1.0", help="Comma-separated list of constants (default: 1.0, empty string for no constants)")
    args = parser.parse_args()

    # Parse constants
    if args.constants.strip():
        constants_list = [float(c.strip()) for c in args.constants.split(',')]
    else:
        constants_list = []

    generate_training_expressions(
        n_expressions=args.n_expressions,
        binary_ops=args.binary_ops,
        unary_ops=args.unary_ops,
        complexity=args.complexity,
        n_input_points=args.n_input_points,
        seed=args.seed,
        constants=constants_list,
        output_dir=args.output_dir,
        demo=args.demo
    )
