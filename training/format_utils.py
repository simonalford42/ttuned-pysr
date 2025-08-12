"""
Shared formatting utilities for neural SR training and inference.
Ensures consistent formatting between trajectory conversion and model usage.
"""


def format_context(variables, operators, constants):
    """Format context string for neural model input"""
    var_str = ",".join(variables)
    op_str = ",".join(operators)
    const_str = ",".join([str(c) for c in constants])
    return f"{var_str} | {op_str} | {const_str}"


def format_population_with_fitness(expressions, fitnesses):
    """Format population with fitness values for neural model input"""
    pop_items = []
    for expr, fitness in zip(expressions, fitnesses):
        pop_items.append(f"{expr} <FITNESS>{fitness:.2f}")
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
                    elif token.replace('.', '').isdigit():
                        constants.add(token)

    return sorted(variables), sorted(operators), sorted(constants)


def format_input_part(bos_token, context, population):
    """Format the input part with special tokens for training/inference"""
    return (
        bos_token +
        "<CONTEXT>" + context +
        "<POPULATION>" + population
    )


def format_target_part(target, eos_token):
    """Format the target part with special tokens for training"""
    return "<TARGET>" + target + eos_token


def format_inference_input(bos_token, context, population):
    """Format input for model inference (includes TARGET prompt)"""
    return (
        bos_token +
        "<CONTEXT>" + context +
        "<POPULATION>" + population +
        "<TARGET>"
    )