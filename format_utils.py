"""
Shared formatting utilities for neural SR training and inference.
Ensures consistent formatting between trajectory conversion and model usage.
"""
import numpy as np
import math
from typing import Optional, Dict, Any


def format_context(generation, num_variables, operators, constants, context_type='basic', data_stats=None):
    """Format context string for neural model input with different context levels

    Args:
        generation: Current generation number for the context
        num_variables: number of variables
        operators: List of operators (e.g., ['+', '-', '*', '/'])
        constants: List of constants (e.g., [1.0, 2.0])
        context_type: 'basic', 'rich', or 'superrich'
        data_stats: Dict with data statistics (for rich/superrich context)

    Returns:
        Formatted context string
    """
    var_str = f"{num_variables} variables"
    op_str = ",".join(operators)
    const_str = ",".join([str(c) for c in constants])

    basic_context = f"generation: {generation} | {var_str} | ops: {op_str} | constants: {const_str}"

    if context_type == 'basic':
        return basic_context
    elif context_type == 'rich':
        return format_rich_context(basic_context, data_stats)
    elif context_type == 'superrich':
        return format_superrich_context(basic_context, data_stats)
    else:
        raise ValueError(f"Unknown context_type: {context_type}")


def format_rich_context(basic_context, data_stats):
    """Format rich context with data statistics

    Args:
        basic_context: Basic context string
        data_stats: Dict with keys like 'y_mean', 'y_std', 'y_min', 'y_max', etc.

    Returns:
        Rich context string with data statistics
    """
    if data_stats is None:
        return basic_context

    # Extract statistics
    stats_parts = []

    # Min/Max
    if 'y_min' in data_stats and 'y_max' in data_stats:
        stats_parts.append(f"range=[{data_stats['y_min']:.3f},{data_stats['y_max']:.3f}]")

    # Moments (mean, variance, skewness)
    if 'y_mean' in data_stats:
        stats_parts.append(f"mean={data_stats['y_mean']:.3f}")
    if 'y_var' in data_stats:
        stats_parts.append(f"var={data_stats['y_var']:.3f}")
    if 'y_skew' in data_stats:
        stats_parts.append(f"skew={data_stats['y_skew']:.3f}")

    # Function characteristics
    if 'is_monotonic' in data_stats:
        stats_parts.append(f"monotonic={data_stats['is_monotonic']}")
    if 'zero_crossings' in data_stats:
        stats_parts.append(f"zeros={data_stats['zero_crossings']}")
    if 'complexity' in data_stats:
        stats_parts.append(f"complexity={data_stats['complexity']:.2f}")

    stats_str = " | ".join(stats_parts)
    return f"{basic_context} | STATS: {stats_str}"


def format_superrich_context(basic_context, data_stats):
    """Format superrich context with data statistics and text plot

    Args:
        basic_context: Basic context string
        data_stats: Dict with data statistics and plot info

    Returns:
        Superrich context string with plot
    """
    rich_context = format_rich_context(basic_context, data_stats)

    if data_stats is None or 'text_plot' not in data_stats:
        return rich_context

    plot_str = data_stats['text_plot']
    return f"{rich_context} | PLOT: {plot_str}"


def compute_data_statistics(X, y):
    """Compute rich statistics about the data for context

    Args:
        X: Input data array (N, num_vars)
        y: Target data array (N,)

    Returns:
        Dict with various statistics about the data
    """
    stats = {}

    # Basic statistics
    stats['y_min'] = float(np.min(y))
    stats['y_max'] = float(np.max(y))
    stats['y_mean'] = float(np.mean(y))
    stats['y_var'] = float(np.var(y))

    # Higher moments
    try:
        from scipy import stats as scipy_stats
        stats['y_skew'] = float(scipy_stats.skew(y))
    except ImportError:
        # Fallback skewness calculation
        y_centered = y - stats['y_mean']
        m3 = np.mean(y_centered**3)
        m2 = np.mean(y_centered**2)
        stats['y_skew'] = float(m3 / (m2**(3/2)) if m2 > 1e-10 else 0.0)

    # Function characteristics
    stats['is_monotonic'] = is_monotonic_1d(X, y) if X.shape[1] == 1 else False
    stats['zero_crossings'] = count_zero_crossings(y)
    stats['complexity'] = estimate_complexity(y)

    # Add input variable statistics
    for i in range(X.shape[1]):
        var_name = f'x{i}'
        stats[f'{var_name}_min'] = float(np.min(X[:, i]))
        stats[f'{var_name}_max'] = float(np.max(X[:, i]))
        stats[f'{var_name}_mean'] = float(np.mean(X[:, i]))

    return stats


def is_monotonic_1d(X, y):
    """Check if function is monotonic for 1D input"""
    if X.shape[1] != 1:
        return False

    # Sort by x values
    sort_idx = np.argsort(X[:, 0])
    y_sorted = y[sort_idx]

    # Check if monotonic increasing or decreasing
    diff = np.diff(y_sorted)
    return np.all(diff >= 0) or np.all(diff <= 0)


def count_zero_crossings(y):
    """Count approximate zero crossings in the data"""
    return int(np.sum(np.diff(np.sign(y)) != 0))


def estimate_complexity(y):
    """Estimate complexity of the function based on variation"""
    # Simple measure: normalized total variation
    total_variation = np.sum(np.abs(np.diff(y)))
    y_range = np.max(y) - np.min(y)
    if y_range < 1e-10:
        return 0.0
    return float(total_variation / (y_range * len(y)))


def create_text_plot(X, y, width=40, height=10):
    """Create a simple ASCII plot suitable for inline text context.

    - For 1D inputs, renders a small line plot over `width` columns and `height` rows.
    - For multi-D inputs, returns a compact summary string.

    The returned value may contain newlines; when JSON-serialized it will be escaped.

    Args:
        X: np.ndarray of shape (N, D)
        y: np.ndarray of shape (N,)
        width: plot width in characters
        height: plot height in characters

    Returns:
        str: ASCII plot or compact summary.
    """
    N, D = X.shape[0], X.shape[1]
    if D != 1 or N == 0:
        return f"{D}D(n={N},yr=[{float(np.min(y)):.2f},{float(np.max(y)):.2f}])"

    # Sort by x to produce a left-to-right curve
    idx = np.argsort(X[:, 0])
    x_sorted = X[idx, 0]
    y_sorted = y[idx]

    y_min, y_max = float(np.min(y_sorted)), float(np.max(y_sorted))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or abs(y_max - y_min) < 1e-12:
        y_min, y_max = -1.0, 1.0

    # Bin into `width` columns; take median y per bin for robustness
    bins = np.linspace(x_sorted[0], x_sorted[-1], num=width+1)
    cols_y = np.full(width, np.nan)
    j = 0
    for c in range(width):
        x_lo, x_hi = bins[c], bins[c+1]
        # Collect points in bin [x_lo, x_hi)
        ys = []
        while j < N and (x_sorted[j] < x_hi or (c == width - 1 and j == N - 1)):
            if x_sorted[j] >= x_lo:
                ys.append(y_sorted[j])
            j += 1
        if ys:
            cols_y[c] = float(np.median(ys))

    # Fill gaps by linear interpolation
    if np.all(np.isnan(cols_y)):
        return f"1D(n={N},yr=[{y_min:.2f},{y_max:.2f}])"
    not_nan = np.where(~np.isnan(cols_y))[0]
    for c in range(width):
        if np.isnan(cols_y[c]):
            # nearest neighbors for interpolation
            left = not_nan[not_nan < c]
            right = not_nan[not_nan > c]
            if left.size and right.size:
                l, r = left[-1], right[0]
                t = (c - l) / (r - l) if r != l else 0.0
                cols_y[c] = (1 - t) * cols_y[l] + t * cols_y[r]
            elif left.size:
                cols_y[c] = cols_y[left[-1]]
            elif right.size:
                cols_y[c] = cols_y[right[0]]
            else:
                cols_y[c] = y_min

    # Map y values to row indices (0 at top)
    def y_to_row(val: float) -> int:
        if y_max == y_min:
            return height // 2
        t = (val - y_min) / (y_max - y_min)
        t = min(max(t, 0.0), 1.0)
        return int(round((1 - t) * (height - 1)))

    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Optional: draw horizontal axis at y=0 if within range
    if y_min <= 0.0 <= y_max:
        axis_row = y_to_row(0.0)
        for c in range(width):
            grid[axis_row][c] = '-'

    # Plot the curve
    prev_row = None
    for c in range(width):
        r = y_to_row(cols_y[c])
        # Draw point
        grid[r][c] = '*'
        # Connect to previous point for smoother appearance
        if prev_row is not None and r != prev_row:
            step = 1 if r > prev_row else -1
            for rr in range(prev_row + step, r, step):
                grid[rr][c] = '*'
        prev_row = r

    lines = [''.join(row) for row in grid]
    # Add y-range footer for readability
    footer = f"yr=[{y_min:.2f},{y_max:.2f}]"
    return '\n'.join(lines + [footer])


def format_population_with_fitness(expressions, fitnesses):
    """Format population with fitness values for neural model input"""
    pop_items = []
    for expr, fitness in zip(expressions, fitnesses):
        # Keep more precision to preserve signal in fitness ordering
        pop_items.append(f"{expr} <FITNESS>{float(fitness):.5f}")
    return ' '.join(pop_items)


def format_sigfigs(value: float, sigfigs: int = 3) -> str:
    """Format a number to a given number of significant figures.

    Uses Python's general format which trims trailing zeros and switches
    to scientific notation when appropriate. Returns a compact string
    representation to minimize token/character footprint.

    Args:
        value: Numeric value to format
        sigfigs: Number of significant figures (default: 3)

    Returns:
        String representation with `sigfigs` significant digits.
    """
    try:
        x = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(x):
        return str(x)
    # Using general format ensures minimal length (no trailing zeros)
    return f"{x:.{sigfigs}g}"


def format_population_with_fitness_sigfigs(expressions, fitnesses, sigfigs: int = 3):
    """Format population with fitness values to N significant figures.

    This is a compact alternative to `format_population_with_fitness` that
    reduces token/character usage while preserving fitness ordering fidelity.

    Args:
        expressions: Iterable of expression strings
        fitnesses: Iterable of numeric fitness values
        sigfigs: Number of significant figures for fitness values

    Returns:
        Single string: "expr <FITNESS>value ..." items joined by spaces
    """
    pop_items = []
    for expr, fitness in zip(expressions, fitnesses):
        pop_items.append(f"{expr} <FITNESS>{format_sigfigs(fitness, sigfigs)}")
    return ' '.join(pop_items)


def extract_variables_operators_constants(trajectory_data):
    """
    Extract variables, operators, and constants from trajectory data.

    Args:
        trajectory_data: List of generation data with populations and expressions

    Returns:
        tuple: (variables, operators, constants)
    """
    variables = set()
    operators = set(['+', '-', '*', '/'])  # Standard operators
    constants = set()

    # Helper to detect numeric constants robustly (handles negatives/decimals/exponents)
    def _is_number(tok: str) -> bool:
        try:
            float(tok)
            return True
        except Exception:
            return False

    # Extract from all expressions in the trajectory
    for gen_data in trajectory_data:
        if "expressions" in gen_data:
            for expr in gen_data["expressions"]:
                # Simple parsing to extract variables and constants
                tokens = expr.replace('(', ' ').replace(')', ' ').replace('+', ' + ').replace('-', ' - ').replace('*', ' * ').replace('/', ' / ').split()
                for token in tokens:
                    token = token.strip()
                    if token.startswith('x') and token[1:].isdigit():
                        variables.add(token)
                    elif _is_number(token):
                        constants.add(token)

    return sorted(variables), sorted(operators), sorted(constants)


def format_input_part(bos_token, context, population):
    """Format the input part with special tokens for training/inference"""
    bos = bos_token or ""
    return (
        bos +
        "<CONTEXT>" + context +
        "<POPULATION>" + population
    )


def format_target_part(target, eos_token):
    """Format the target part with special tokens for training"""
    return "<TARGET>" + target + eos_token


def format_inference_input(bos_token, context, population):
    """Format input for model inference (includes TARGET prompt)"""
    bos = bos_token or ""
    return (
        bos +
        "<CONTEXT>" + context +
        "<POPULATION>" + population +
        "<TARGET>"
    )
