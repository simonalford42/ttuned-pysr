"""
Demo for the 'superrich' context plotting.

Generates ASCII plots and superrich context strings for simple functions:
- y = x^2 on x in [-3, 3]
- y = sin(x) on x in [-pi, pi]
"""
import numpy as np
from training.format_utils import (
    compute_data_statistics,
    create_text_plot,
    format_context,
    format_inference_input,
    format_population_with_fitness,
)
from basic_sr import BasicSR
from problems import HARDER_PROBLEMS


def make_dataset(func, x_min, x_max, n=80):
    x = np.linspace(x_min, x_max, num=n)
    y = func(x)
    X = x.reshape(-1, 1)
    return X, y


def show_case(name, func, x_min, x_max):
    print(f"=== {name} ===")
    X, y = make_dataset(func, x_min, x_max)
    stats = compute_data_statistics(X, y)
    stats['text_plot'] = create_text_plot(X, y, width=48, height=12)
    plot = stats['text_plot']
    print(plot)
    variables = ['x0']
    operators = ['*', '+', '-', '/']
    constants = [1.0, 2.0]
    ctx = format_context(0, variables, operators, constants, context_type='superrich', data_stats=stats)
    print("--- superrich context ---")
    print(ctx)
    print()


def main():
    show_case("y = x^2", lambda x: x**2, -3.0, 3.0)
    show_case("y = sin(x)", np.sin, -np.pi, np.pi)

    # Also print full model input strings for first generation across context types
    def first_gen_inputs(X, y, context_type: str):
        # Build initial population and fitnesses
        sr = BasicSR(population_size=20, num_generations=1, max_depth=3, max_size=10)
        num_vars = X.shape[1]
        population = sr.create_initial_population(num_vars)
        fitnesses = [sr.fitness(ind, X, y) for ind in population]

        # Context parts
        variables = [f"x{i}" for i in range(num_vars)]
        operators = sr.operators
        constants = sr.constants

        data_stats = None
        if context_type in ("rich", "superrich"):
            data_stats = compute_data_statistics(X, y)
            if context_type == "superrich":
                data_stats["text_plot"] = create_text_plot(X, y, width=48, height=12)

        context = format_context(0, variables, operators, constants, context_type=context_type, data_stats=data_stats)
        pop_line = format_population_with_fitness([str(ind) for ind in population], fitnesses)
        full_input = format_inference_input(bos_token="", context=context, population=pop_line)
        return full_input

    def demo_full_inputs(label: str, X, y):
        print(f"=== Full Inputs: {label} ===")
        for ctx in ("basic", "rich", "superrich"):
            print(f"-- {ctx} --")
            print(first_gen_inputs(X, y, ctx))
            print()

    # Use same datasets as above for full input demos
    Xq, yq = make_dataset(lambda x: x**2, -3.0, 3.0)
    demo_full_inputs("y = x^2", Xq, yq)

    Xs, ys = make_dataset(np.sin, -np.pi, np.pi)
    demo_full_inputs("y = sin(x)", Xs, ys)

    # Harder problems: print first-gen inputs for each
    print("=== Full Inputs: Harder Problems ===")
    for prob in HARDER_PROBLEMS:
        Xh, yh = prob(seed=42)
        print(f"Problem: {prob.__name__} -- {prob.__doc__}")
        for ctx in ("basic", "rich", "superrich"):
            print(f"-- {ctx} --")
            print(first_gen_inputs(Xh, yh, ctx))
            print()


if __name__ == "__main__":
    main()
