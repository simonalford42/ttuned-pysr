"""
Generate synthetic expressions for training data using e2e_data_gen.
This module now wraps E2EDataGenerator to create expression datasets
with configurable operators and complexity levels.
"""
import numpy as np
import json
import pickle
import gzip
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from e2e_data_gen import E2EDataGenerator


def generate_e2e_expressions(
    n_expressions: int = 100,
    binary_ops: str = "add,sub,mul",
    unary_ops: str = "",
    complexity: float = 0.5,
    n_input_points: int = 64,
    seed: int = 42
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

    Returns:
        List of expression dicts with id, expression, X_data, y_data, metadata
    """
    np.random.seed(seed)

    # Build e2e generator
    gen = E2EDataGenerator(env_overrides={
        "env_base_seed": seed,
        "use_controller": False,
        "allowed_binary_operators": binary_ops,
        "allowed_unary_operators": unary_ops,
        "complexity": complexity,
    })

    expressions = []

    for i in range(n_expressions):
        # Generate expression with data
        sample = gen.generate_expression(train=True)
        expr_str = sample["expr_str"]
        tree = sample["tree"]

        # Get data
        X_list = sample.get("X_to_fit", [])
        Y_list = sample.get("Y_to_fit", [])

        if X_list and Y_list:
            X = X_list[0]
            Y = Y_list[0]
        else:
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
                                 n_input_points: int = 64,
                                 seed: int = 42,
                                 output_dir: str = "datasets/expressions") -> str:
    """Generate training expressions with data and save to file"""

    print(f"Generating {n_expressions} training expressions...")
    print(f"Parameters: binary_ops={binary_ops}, unary_ops={unary_ops or 'none'}, complexity={complexity}")
    print(f"Data points per expression: {n_input_points}")

    # Generate expressions
    expressions = generate_e2e_expressions(
        n_expressions=n_expressions,
        binary_ops=binary_ops,
        unary_ops=unary_ops,
        complexity=complexity,
        n_input_points=n_input_points,
        seed=seed
    )

    # Create metadata
    metadata = {
        'generation_command': f"python generate_expressions.py --n_expressions={n_expressions} --binary_ops={binary_ops} --unary_ops={unary_ops} --complexity={complexity} --n_input_points={n_input_points} --seed={seed}",
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'n_expressions': n_expressions,
            'binary_ops': binary_ops,
            'unary_ops': unary_ops,
            'complexity': complexity,
            'n_input_points': n_input_points,
            'seed': seed
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
    filename = f"{output_dir}/train_{size_str}_{ops_str}_{complexity_str}_{timestamp}.pkl.gz"

    # Save expressions
    save_expressions(expressions, filename, metadata)

    print(f"✓ Generated {len(expressions)} expressions with data")
    print(f"✓ Saved to {filename}")

    return filename


# Kept for backward compatibility - old sympy-based generator (legacy)
def generate_expressions_legacy(
    n_expressions: int = 100,
    max_size: int = 30,
    min_size: int = 5,
    seed: int = 42,
    operators: List[str] = ['+', '-', '*', '/'],
    constants: List[float] = [1.0, 2.0],
    n_variables: int = 3,
    max_depth: Optional[int] = None,
) -> List[str]:
    """
    Generate n expressions with sizes uniformly sampled in [min_size, max_size].

    - Respects operator arity (unary/binary) and builds valid SymPy expressions.
    - Avoids automatic evaluation to preserve intended tree structure.
    - Handles operators like 'exp', 'log_10', and power ('^'/'pow').

    Notes on size: we count AST nodes where a leaf (variable or constant) counts as 1,
    and an operator node counts as 1 plus its children. This ensures total node count
    equals the target size exactly.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    names = ' '.join(f'x{i}' for i in range(n_variables))
    # Set variables as real to avoid SymPy introducing re()/im() in simplifications
    syms = sp.symbols(names, real=True)
    # sp.symbols returns a single Symbol when n_variables == 1
    var_pool = list(syms) if isinstance(syms, (tuple, list)) else [syms]

    # Map operators to (arity, constructor). Use evaluate=False for arithmetic
    # to avoid SymPy auto-merging constants or flattening trees.
    op_registry = {
        '+': (2, lambda a, b: sp.Add(a, b, evaluate=False)),
        '-': (2, lambda a, b: sp.Add(a, sp.Mul(-1, b, evaluate=False), evaluate=False)),
        '*': (2, lambda a, b: sp.Mul(a, b, evaluate=False)),
        '/': (2, lambda a, b: sp.Mul(a, sp.Pow(b, -1, evaluate=False), evaluate=False)),
        '^': (2, lambda a, b: sp.Pow(a, b, evaluate=False)),
        'pow': (2, lambda a, b: sp.Pow(a, b, evaluate=False)),
        'exp': (1, lambda a: sp.exp(a)),
        'log': (1, lambda a: sp.log(a)),
        'log_10': (1, lambda a: sp.log(a, 10)),
        'sqrt': (1, lambda a: sp.sqrt(a)),
        'sin': (1, lambda a: sp.sin(a)),
        'cos': (1, lambda a: sp.cos(a)),
        'tan': (1, lambda a: sp.tan(a)),
        'abs': (1, lambda a: sp.Abs(a)),
    }

    # Filter to only the requested operators that we support
    unary_ops = []
    binary_ops = []
    for op in operators:
        if op not in op_registry:
            raise ValueError(f"Operator {op} not provided.")
        arity, fn = op_registry[op]
        if arity == 1:
            unary_ops.append((op, fn))
        elif arity == 2:
            binary_ops.append((op, fn))

    if not unary_ops and not binary_ops:
        raise ValueError("No valid operators provided.")

    # Constant pool as SymPy objects
    const_pool = [sp.Float(c) if not isinstance(c, (sp.Float, sp.Integer)) else c for c in constants]
    if not const_pool:
        const_pool = [sp.Integer(1)]

    # Depth feasibility helpers
    def max_nodes_possible(depth_left: int) -> int:
        if depth_left < 0:
            return 0
        if binary_ops:
            return (1 << (depth_left + 1)) - 1
        elif unary_ops:
            return depth_left + 1
        else:
            return 1

    def feasible(nodes: int, depth_left: Optional[int]) -> bool:
        if nodes < 1:
            return False
        if depth_left is None:
            return True
        return nodes <= max_nodes_possible(depth_left)

    # Use typing.Tuple for wider Python version compatibility
    from typing import Tuple

    def build_tree(target_nodes: int, must_use_var: bool, depth: int = 0) -> Tuple[sp.Expr, bool]:
        """Recursively build a tree with exactly target_nodes and report if it uses any variable."""
        assert target_nodes >= 1
        depth_left = None if max_depth is None else (max_depth - depth)
        if max_depth is not None and not feasible(target_nodes, depth_left):
            raise ValueError("Infeasible size with current depth budget")

        # Base case or depth cap
        if target_nodes == 1 or (max_depth is not None and depth >= max_depth):
            if must_use_var or rng.random() < 0.6:
                sym = rng.choice(var_pool)
                return sym, True
            else:
                c = rng.choice(const_pool)
                return c, False

        # Choose op type constrained by remaining budget
        choices = []
        # Unary option
        if unary_ops and target_nodes >= 2:
            if feasible(target_nodes - 1, None if max_depth is None else (max_depth - (depth + 1))):
                choices.append('unary')
        # Binary option
        feasible_splits = []
        if binary_ops and target_nodes >= 3:
            rest = target_nodes - 1
            for l_nodes in range(1, rest):
                r_nodes = rest - l_nodes
                if max_depth is None:
                    feasible_splits.append((l_nodes, r_nodes))
                else:
                    dl = max_depth - (depth + 1)
                    dr = dl
                    if feasible(l_nodes, dl) and feasible(r_nodes, dr):
                        feasible_splits.append((l_nodes, r_nodes))
        if feasible_splits:
            choices.append('binary')
        if not choices:
            raise ValueError("No feasible operator choice at this size/depth")

        kind = rng.choice(choices)
        if kind == 'unary':
            op, fn = rng.choice(unary_ops)
            child_nodes = target_nodes - 1
            child, used_var = build_tree(child_nodes, must_use_var, depth + 1)
            try:
                expr = fn(child)
            except Exception:
                expr = child
            return expr, used_var
        else:
            op, fn = rng.choice(binary_ops)
            if feasible_splits:
                l_nodes, r_nodes = rng.choice(feasible_splits)
            else:
                rest = target_nodes - 1
                l_nodes = rng.randint(1, rest - 1)
                r_nodes = rest - l_nodes

            left_must_var = must_use_var and rng.random() < 0.5
            right_must_var = must_use_var and not left_must_var

            left, l_used = build_tree(l_nodes, left_must_var, depth + 1)
            right, r_used = build_tree(r_nodes, right_must_var, depth + 1)

            # Bias exponents to small integers for pow
            if op in ('^', 'pow'):
                if rng.random() < 0.7:
                    choices_int = [-2, -1, 2, 3]
                    if rng.random() < 0.5:
                        right = sp.Integer(rng.choice(choices_int))
                    else:
                        left, right = right, sp.Integer(rng.choice(choices_int))

            try:
                expr = fn(left, right)
            except Exception:
                expr = left + right
            return expr, (l_used or r_used)

    results: List[str] = []

    def renumber_variables(expr: sp.Expr) -> sp.Expr:
        """Rename variables used in expr to contiguous names x0..x{k-1}.

        Ensures we never emit expressions that reference, e.g., x2 without x1.
        The order is based on the numeric suffix in each symbol's name.
        """
        if not expr.free_symbols:
            return expr

        def sym_index(s: sp.Symbol) -> int:
            name = s.name
            return int(name[1:]) if name.startswith('x') and name[1:].isdigit() else 10**9

        used = sorted(list(expr.free_symbols), key=sym_index)
        mapping = {}
        for i, old in enumerate(used):
            new = sp.Symbol(f"x{i}", real=True)
            if old != new:
                mapping[old] = new
        return expr.xreplace(mapping) if mapping else expr
    # Adjust feasible size band if depth-limited
    size_lo, size_hi = min_size, max_size
    if max_depth is not None:
        global_max = max_nodes_possible(max_depth)
        size_lo = max(1, size_lo)
        size_hi = min(size_hi, global_max)
        if size_lo > size_hi:
            raise ValueError(f"Requested size range [{min_size},{max_size}] infeasible for max_depth={max_depth}; max feasible={global_max}")

    # Uniformly sample sizes; keep trying until we collect n_expressions.
    # Guard with a maximum number of total attempts to avoid infinite loops for
    # pathological (size, depth, operator) combinations.
    total_attempts = 0
    max_total_attempts = max(200, 20 * n_expressions)
    while len(results) < n_expressions and total_attempts < max_total_attempts:
        total_attempts += 1
        size = rng.randint(size_lo, size_hi)
        try:
            expr, used_var = build_tree(size, must_use_var=True, depth=0)
            if not expr.free_symbols:
                expr = sp.Add(expr, var_pool[0], evaluate=False)
            expr = renumber_variables(expr)
            results.append(str(expr))
        except ValueError:
            continue

    # If still short (very strict constraints), backfill with trivial variable-based expressions
    # to meet the requested count without raising.
    while len(results) < n_expressions:
        v = rng.choice(var_pool)
        c = rng.choice([sp.Integer(1), sp.Integer(2)])
        expr = sp.Add(v, c, evaluate=False)
        expr = renumber_variables(expr)
        results.append(str(expr))

    return results


def create_test_expressions(n_expressions: int = 10,
                           seed: int = 42,
                           output_dir: str = "datasets/expressions") -> str:
    """Create a small test set of expressions"""
    print(f"Creating test set with {n_expressions} expressions...")

    return generate_training_expressions(
        n_expressions=n_expressions,
        binary_ops="add,sub,mul",
        unary_ops="",
        complexity=0.3,
        n_input_points=64,
        seed=seed,
        output_dir=output_dir
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic expressions using e2e")
    parser.add_argument("--n_expressions", type=int, default=100, help="Number of expressions to generate")
    parser.add_argument("--binary_ops", type=str, default="add,sub,mul", help="Comma-separated binary operators (e.g., add,sub,mul,div,pow)")
    parser.add_argument("--unary_ops", type=str, default="", help="Comma-separated unary operators (e.g., abs,sqrt,sin,cos,tan,inv)")
    parser.add_argument("--complexity", type=float, default=0.5, help="Complexity level (0.0 to 1.0)")
    parser.add_argument("--n_input_points", type=int, default=64, help="Number of data points per expression")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", default="datasets/expressions", help="Output directory")
    parser.add_argument("--test", action="store_true", help="Generate small test set")

    args = parser.parse_args()

    if args.test:
        filename = create_test_expressions(args.n_expressions, args.seed, args.output_dir)
    else:
        filename = generate_training_expressions(
            n_expressions=args.n_expressions,
            binary_ops=args.binary_ops,
            unary_ops=args.unary_ops,
            complexity=args.complexity,
            n_input_points=args.n_input_points,
            seed=args.seed,
            output_dir=args.output_dir
        )

    print(f"\n✓ Expression generation complete: {filename}")
