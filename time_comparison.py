import numpy as np
import random
import time
import io
from contextlib import redirect_stdout
from sr import BasicSR
from problems import HARDER_PROBLEMS


class ProgressCapture:
    """Captures progress output from BasicSR to extract improvement trajectory"""
    def __init__(self):
        self.improvements = []
        self.start_time = None

    def parse_progress_line(self, line):
        """Parse a line like 'Gen 15: MSE=7.232738, Size=13, Expr=...'"""
        if not line.startswith("Gen "):
            return None

        try:
            # Parse the line
            parts = line.split(": MSE=")
            if len(parts) != 2:
                return None

            gen_part = parts[0]
            rest = parts[1]

            generation = int(gen_part.split("Gen ")[1])

            # Split on ", Size=" to get MSE and size
            mse_size_parts = rest.split(", Size=")
            if len(mse_size_parts) != 2:
                return None

            mse = float(mse_size_parts[0])

            # Split on ", Expr=" to get size
            size_expr_parts = mse_size_parts[1].split(", Expr=")
            if len(size_expr_parts) != 2:
                return None

            size = int(size_expr_parts[0])

            # Estimate time (this is approximate since we don't have exact timing)
            current_time = time.time()
            elapsed = current_time - self.start_time if self.start_time else 0

            return {
                'generation': generation,
                'time': elapsed,
                'mse': mse,
                'size': size
            }
        except:
            return None


def run_timed_experiment(problem, time_limit_seconds, seed=42):
    """Run BasicSR for a specific time limit using the built-in time limit functionality"""

    random.seed(seed)
    np.random.seed(seed)

    # Generate data
    X, y = problem(seed=seed)

    print(f"Running {problem.__name__} for {time_limit_seconds}s...")

    # Create model
    model = BasicSR(
        population_size=30,
        num_generations=10000,  # Will be stopped by time limit or MSE early stopping
        max_depth=5,
        max_size=20,
        tournament_size=5,
        collect_trajectory=False,
        time_limit=time_limit_seconds
    )

    # Capture output to parse improvements
    capture = ProgressCapture()
    output_buffer = io.StringIO()

    # Fit the model while capturing output
    start_time = time.time()
    capture.start_time = start_time

    with redirect_stdout(output_buffer):
        model.fit(X, y)

    actual_time = time.time() - start_time

    # Parse the captured output for improvements
    output_lines = output_buffer.getvalue().split('\n')
    improvements = []

    for line in output_lines:
        line = line.strip()
        if line.startswith("Gen "):
            improvement = capture.parse_progress_line(line)
            if improvement:
                improvements.append(improvement)

    # Get final results
    if model.best_model_:
        y_pred = model.predict(X)
        final_mse = np.mean((y - y_pred)**2)
        final_expression = str(model.best_model_)
        final_size = model.best_model_.size()
    else:
        final_mse = float('inf')
        final_expression = "None"
        final_size = 0

    # Print the captured output so user can see progress
    print(output_buffer.getvalue(), end='')

    return {
        'problem': problem.__name__,
        'time_limit': time_limit_seconds,
        'actual_time': actual_time,
        'generations': model.generation_count if hasattr(model, 'generation_count') else len(improvements),
        'final_mse': final_mse,
        'final_expression': final_expression,
        'final_size': final_size,
        'improvements': improvements
    }


def compare_times_single_problem(problem, time_limits=[60, 300, 600]):
    """Compare different time limits on a single problem"""

    print(f"\n{'='*60}")
    print(f"TESTING: {problem.__name__}")
    print(f"Ground truth: {problem.__doc__}")
    print('='*60)

    results = []

    for time_limit in time_limits:
        print(f"\n--- {time_limit} seconds ---")
        result = run_timed_experiment(problem, time_limit)
        results.append(result)

        print(f"Final: {result['generations']} gens, MSE={result['final_mse']:.6f}")
        print(f"Expression: {result['final_expression']}")
        print(f"Improvements made: {len(result['improvements'])}")

    # Quick comparison
    print(f"\n--- COMPARISON ---")
    print(f"{'Time':<6} {'Gens':<6} {'MSE':<12} {'Improvements':<12} {'Expression':<30}")
    print("-" * 70)

    for result in results:
        print(f"{result['time_limit']:<6} {result['generations']:<6} {result['final_mse']:<12.2e} "
              f"{len(result['improvements']):<12} {result['final_expression'][:29]:<30}")

    return results


def quick_test():
    """Quick test on just the first harder problem"""

    print("QUICK TIME COMPARISON TEST")
    print("Testing just the first harder problem with 1min, 3min, 5min")

    # Test just one problem with shorter times for demo
    problem = HARDER_PROBLEMS[0]  # pythagorean_3d
    results = compare_times_single_problem(problem, time_limits=[60, 180, 300])

    return results


def full_comparison():
    """Run each harder problem for 60 seconds and track improvement trajectory"""

    print("IMPROVEMENT TRAJECTORY ANALYSIS")
    print("Running each problem for 60 seconds and tracking progress...")
    print(f"Testing {len(HARDER_PROBLEMS)} problems × 60 seconds each = ~300 seconds total")

    all_results = []

    for i, problem in enumerate(HARDER_PROBLEMS):
        print(f"\nProblem {i+1}/{len(HARDER_PROBLEMS)}: {problem.__name__}")
        print(f"Ground truth: {problem.__doc__}")

        # Run for 60 seconds and track trajectory
        result = run_timed_experiment(problem, time_limit_seconds=60, seed=42)
        all_results.append(result)

        # Show trajectory summary
        improvements = result['improvements']
        print(f"\nTrajectory Summary:")
        print(f"  Total improvements: {len(improvements)}")
        print(f"  Final MSE: {result['final_mse']:.2e}")
        print(f"  Final expression: {result['final_expression']}")

        if len(improvements) >= 2:
            first_mse = improvements[0]['mse']
            last_mse = improvements[-1]['mse']
            total_improvement = (first_mse - last_mse) / first_mse * 100 if first_mse > 0 else 0
            print(f"  Total improvement: {total_improvement:.1f}% MSE reduction")

            # Show key milestones
            print(f"  Key milestones:")
            for imp in improvements[:5]:  # Show first 5 improvements
                print(f"    {imp['time']:5.1f}s (Gen {imp['generation']:3d}): MSE={imp['mse']:.2e}")
            if len(improvements) > 5:
                print(f"    ... ({len(improvements)-5} more improvements)")

        print("-" * 60)

    # Save results
    import json
    with open('improvement_trajectories.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to improvement_trajectories.json")
    return all_results


def create_markdown_table(results):
    """Create markdown table focusing on improvement trajectories"""

    markdown = "# Improvement Trajectory Analysis\n\n"
    markdown += "Each problem was run for 60 seconds using BasicSR with built-in time limits and MSE early stopping (≤ 3e-16).\n\n"

    for result in results:
        problem_name = result['problem']
        improvements = result['improvements']

        markdown += f"## {problem_name}\n\n"

        # Add problem description if available
        # Note: We'd need to modify the script to pass this through

        markdown += f"**Final Result**: {result['final_mse']:.2e} MSE in {result['generations']} generations  \n"
        markdown += f"**Actual Time**: {result['actual_time']:.1f}s (limit: {result['time_limit']}s)  \n"
        markdown += f"**Expression**: `{result['final_expression']}`\n\n"

        if len(improvements) > 0:
            markdown += f"**Improvement Trajectory** ({len(improvements)} improvements):\n\n"
            markdown += "| Time | Generation | MSE | Size | Improvement |\n"
            markdown += "|------|------------|-----|------|-------------|\n"

            prev_mse = None
            for imp in improvements:  # Show ALL improvements
                improvement_pct = ""
                if prev_mse is not None and prev_mse > 0:
                    pct = (prev_mse - imp['mse']) / prev_mse * 100
                    improvement_pct = f"{pct:.1f}%"

                markdown += f"| {imp['time']:.1f}s | {imp['generation']} | {imp['mse']:.2e} | {imp['size']} | {improvement_pct} |\n"
                prev_mse = imp['mse']
        else:
            markdown += "**No improvements recorded** (likely found perfect solution immediately)\n"

        # Add analysis
        if len(improvements) >= 2:
            first_mse = improvements[0]['mse']
            last_mse = improvements[-1]['mse']
            total_improvement = (first_mse - last_mse) / first_mse * 100 if first_mse > 0 else 0

            markdown += f"\n**Analysis**:\n"
            markdown += f"- Total improvement: {total_improvement:.1f}% MSE reduction\n"
            markdown += f"- First improvement at {improvements[0]['time']:.1f}s (Gen {improvements[0]['generation']})\n"
            markdown += f"- Last improvement at {improvements[-1]['time']:.1f}s (Gen {improvements[-1]['generation']})\n"

            # Check if stopped early due to MSE
            if result['final_mse'] <= 3e-16:
                markdown += f"- **Early stopping**: MSE reached near-zero ({result['final_mse']:.2e})\n"
            elif result['actual_time'] >= result['time_limit'] - 1:
                markdown += f"- **Time limit**: Stopped due to {result['time_limit']}s time limit\n"

            # Improvement rate analysis
            early_improvements = [imp for imp in improvements if imp['time'] <= 20]  # First 20 seconds
            late_improvements = [imp for imp in improvements if imp['time'] > 40]    # Last 20 seconds

            markdown += f"- Early phase (0-20s): {len(early_improvements)} improvements\n"
            markdown += f"- Late phase (40-60s): {len(late_improvements)} improvements\n"

            if len(early_improvements) > len(late_improvements) * 2:
                markdown += "- **Pattern**: Front-loaded discovery (most progress early)\n"
            elif len(late_improvements) > len(early_improvements):
                markdown += "- **Pattern**: Late-stage breakthroughs (progress throughout)\n"
            else:
                markdown += "- **Pattern**: Steady progress throughout run\n"

        markdown += "\n---\n\n"

    return markdown


if __name__ == "__main__":
    print("Improvement Trajectory Analysis")
    print("===============================")
    print()
    print("Running full analysis on all 5 harder problems (60 seconds each)...")

    # Run full comparison automatically
    results = full_comparison()

    # Generate markdown
    if results:
        markdown = create_markdown_table(results)
        with open('improvement_trajectories.md', 'w') as f:
            f.write(markdown)
        print(f"\nMarkdown analysis saved to improvement_trajectories.md")
