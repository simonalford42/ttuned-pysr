#!/usr/bin/env python3
"""
End-to-end expression and data generator wrapper around the e2e environment.

This exposes a small, direct API to:
- generate expressions (as trees and readable strings)
- generate input/output data for those expressions

It mirrors the knobs used by the e2e generator (operators, complexity,
dimensions, distributions, noise, etc.) by letting you override any of the
underlying environment parameters on initialization, and per-call overrides for
key sampling controls.

Note: This does not yet integrate with generate_expressions.py or
generate_traces.py. It is designed to be compatible with a future connecting
API.
"""
import os
import sys
from typing import Any, Dict, Optional, Tuple, Union, Iterable

import numpy as np


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

            # Avoid zeroing all unaries to keep probability vector valid; we will still set max_unary_ops=0 if needed by user
            # For safety, if zero_ops covers all unaries, drop one common unary from zeroing
            if set(all_unaries).issubset(zero_ops) and len(all_unaries) > 0:
                zero_ops.discard("sin")  # leave one with non-zero prob

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
    # Minimal demo: print 10 sampled expressions and basic data stats
    demo_generate_n(10)
