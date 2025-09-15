"""
Generate synthetic expressions for training data.
Builds on existing problems.py infrastructure.
"""
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
import random
import sympy as sp
from typing import List, Tuple, Dict, Any


def generate_simple_polynomials(n_expressions: int, max_degree: int = 3, 
                               max_vars: int = 2, constants: List[float] = [1.0, 2.0],
                               seed: int = 42) -> List[Dict[str, Any]]:
    """Generate simple polynomial expressions"""
    random.seed(seed)
    np.random.seed(seed)
    
    expressions = []
    
    # Variable names
    var_names = [f'x{i}' for i in range(max_vars)]
    
    for i in range(n_expressions):
        # Random degree and number of variables for this expression
        degree = random.randint(1, max_degree)
        n_vars = random.randint(1, max_vars)
        selected_vars = var_names[:n_vars]
        
        # Generate polynomial terms
        terms = []
        n_terms = random.randint(1, min(5, 2**n_vars))  # Limit complexity
        
        for _ in range(n_terms):
            # Create a term
            coeff = random.choice(constants + [-1.0])  # Include negative
            var_powers = {}
            
            # Assign powers to variables
            for var in selected_vars:
                if random.random() < 0.7:  # Not all vars in every term
                    power = random.randint(0, degree)
                    if power > 0:
                        var_powers[var] = power
            
            # Build term string
            if not var_powers:  # Constant term
                # Always render numeric coefficient (avoid bare '-')
                term = str(coeff) if coeff != 1 else "1.0"
            else:
                term_parts = []
                if coeff != 1.0:
                    # Use numeric literal for negatives to avoid strings like '- * x0'
                    term_parts.append(str(coeff))  # e.g., '-1.0'
                
                for var, power in var_powers.items():
                    if power == 1:
                        term_parts.append(var)
                    else:
                        term_parts.append(f"({var}**{power})")
                
                term = " * ".join(term_parts) if term_parts else "1.0"
            
            terms.append(term)
        
        # Combine terms
        if len(terms) == 1:
            expr_str = terms[0]
        else:
            expr_str = " + ".join(f"({term})" for term in terms)

        # Ensure expression is not purely-constant (helps avoid trivial traces)
        if 'x' not in expr_str:
            # Guarantee at least one variable term
            first_var = selected_vars[0]
            expr_str = f"({expr_str}) + {first_var}"
        
        # Create data generation function
        def create_data_func(expr_str=expr_str, n_vars=n_vars):
            def data_func(seed):
                rstate = np.random.RandomState(seed)
                X = rstate.uniform(-3, 3, size=(50, n_vars))
                
                # Evaluate expression safely
                try:
                    # Build evaluation context
                    context = {}
                    for j in range(n_vars):
                        context[f'x{j}'] = X[:, j]
                    
                    # Evaluate expression
                    y = eval(expr_str, {"__builtins__": {}}, context)
                    return X, np.array(y)
                except:
                    # Fallback to simple expression if evaluation fails
                    return X, X[:, 0] ** 2
            
            return data_func
        
        expressions.append({
            'id': i,
            'expression': expr_str,
            'variables': selected_vars,
            'degree': degree,
            'data_func': create_data_func(),
            'description': f"Polynomial: {expr_str}"
        })
    
    return expressions


def save_expressions(expressions: List[Dict], output_file: str, metadata: Dict):
    """Save expressions to file with metadata"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert data_func to None for JSON serialization
    serializable_exprs = []
    for expr in expressions:
        expr_copy = expr.copy()
        expr_copy.pop('data_func', None)  # Remove non-serializable function
        serializable_exprs.append(expr_copy)
    
    data = {
        'metadata': metadata,
        'expressions': serializable_exprs,
        'count': len(expressions)
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {len(expressions)} expressions to {output_file}")
    return serializable_exprs


def generate_training_expressions(n_expressions: int = 100, 
                                 max_degree: int = 3,
                                 max_vars: int = 2,
                                 constants: List[float] = [1.0, 2.0],
                                 seed: int = 42,
                                 output_dir: str = "datasets/expressions") -> str:
    """Generate training expressions and save to file"""
    
    print(f"Generating {n_expressions} training expressions...")
    print(f"Parameters: max_degree={max_degree}, max_vars={max_vars}")
    print(f"Constants: {constants}")
    
    # Generate expressions
    expressions = generate_simple_polynomials(
        n_expressions=n_expressions,
        max_degree=max_degree,
        max_vars=max_vars,
        constants=constants,
        seed=seed
    )
    
    # Create metadata
    metadata = {
        'generation_command': f"python generate_expressions.py --n_expressions={n_expressions} --max_degree={max_degree} --max_vars={max_vars} --seed={seed}",
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'n_expressions': n_expressions,
            'max_degree': max_degree,
            'max_vars': max_vars,
            'constants': constants,
            'seed': seed
        },
        'generator_version': '1.0'
    }
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/train_expressions_{timestamp}.json"
    
    # Save expressions
    save_expressions(expressions, filename, metadata)
    
    print(f"✓ Generated {len(expressions)} expressions")
    print(f"✓ Saved to {filename}")
    
    return filename


def create_test_expressions(n_expressions: int = 10,
                           seed: int = 42,
                           output_dir: str = "datasets/expressions") -> str:
    """Create a small test set of expressions"""
    print(f"Creating test set with {n_expressions} expressions...")
    
    return generate_training_expressions(
        n_expressions=n_expressions,
        max_degree=2,  # Simpler for testing
        max_vars=2,
        constants=[1.0, 2.0],
        seed=seed,
        output_dir=output_dir
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic expressions")
    parser.add_argument("--n_expressions", type=int, default=100, help="Number of expressions to generate")
    parser.add_argument("--max_degree", type=int, default=3, help="Maximum polynomial degree")
    parser.add_argument("--max_vars", type=int, default=2, help="Maximum number of variables")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", default="datasets/expressions", help="Output directory")
    parser.add_argument("--test", action="store_true", help="Generate small test set")
    
    args = parser.parse_args()
    
    if args.test:
        filename = create_test_expressions(args.n_expressions, args.seed, args.output_dir)
    else:
        filename = generate_training_expressions(
            args.n_expressions, args.max_degree, args.max_vars, 
            [1.0, 2.0], args.seed, args.output_dir
        )
    
    print(f"\n✓ Expression generation complete: {filename}")
