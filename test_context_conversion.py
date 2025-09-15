#!/usr/bin/env python3
"""
Test context conversion with our existing test dataset.
"""

import sys
sys.path.append('training')

import json
import numpy as np
from convert_trajectories import convert_basicsr_to_one_step_format
from format_utils import compute_data_statistics, create_text_plot

def create_data_context_for_test_dataset():
    """Create data context for our test dataset"""
    
    # Load the expressions to recreate the data
    with open('datasets/expressions/train_expressions_20250912_142600.json', 'r') as f:
        expr_data = json.load(f)
    
    expressions = expr_data['expressions']
    data_context = {}
    
    for expr_info in expressions:
        expr_id = expr_info['id']
        expr_str = expr_info['expression']
        variables = expr_info['variables']
        n_vars = len(variables)
        
        # Recreate the same data that was used during trace generation
        np.random.seed(42)  # Same seed as used in trace generation
        X = np.random.uniform(-3, 3, size=(50, n_vars))
        
        # Evaluate the expression (simplified approach)
        try:
            context = {}
            for i, var in enumerate(variables):
                if i < X.shape[1]:
                    context[var] = X[:, i]
            
            context.update({'np': np, 'sqrt': np.sqrt, 'exp': np.exp, 
                           'log': np.log, 'sin': np.sin, 'cos': np.cos})
            
            y = eval(expr_str, {"__builtins__": {}}, context)
            y = np.array(y, dtype=float)
            
            # Ensure y is 1D
            if y.ndim == 0:
                y = np.full(X.shape[0], float(y))
            elif y.ndim > 1:
                y = y.flatten()[:X.shape[0]]
            
            # Handle edge cases
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                y = X[:, 0] ** 2  # Fallback
                
        except Exception as e:
            print(f"Warning: Failed to evaluate {expr_str}, using fallback")
            y = X[:, 0] ** 2  # Fallback
        
        problem_name = f"expr_{expr_id}"
        data_context[problem_name] = {
            'X': X.tolist(),  # Convert to list for JSON serialization
            'y': y.tolist(),
            'expression': expr_str,
            'variables': variables
        }
        
        print(f"Created data context for {problem_name}: {expr_str}")
    
    return data_context

def test_context_types():
    """Test different context types on our test dataset"""
    
    print("=== Testing Context Types with Real Dataset ===")
    
    # Create data context
    data_context = create_data_context_for_test_dataset()
    
    # Save data context to file
    with open('datasets/test_data_context.json', 'w') as f:
        json.dump(data_context, f, indent=2)
    print("Saved data context to datasets/test_data_context.json")
    
    # Load trajectories and extract the trajectories part (not wrapped)
    with open('datasets/traces/traces_20250912_142600.json', 'r') as f:
        full_data = json.load(f)
    
    # Extract just the trajectories
    trajectories_only = full_data['trajectories']
    
    # Save trajectories only for conversion
    with open('datasets/test_trajectories_only.json', 'w') as f:
        json.dump(trajectories_only, f, indent=2)
    
    # Convert X, y back to numpy arrays for statistics computation
    for problem_name in data_context:
        data_context[problem_name]['X'] = np.array(data_context[problem_name]['X'])
        data_context[problem_name]['y'] = np.array(data_context[problem_name]['y'])
    
    # Test each context type
    for context_type in ['basic', 'rich', 'superrich']:
        print(f"\n--- Testing {context_type.upper()} context ---")
        
        output_file = f'datasets/test_{context_type}_context.jsonl'
        
        try:
            converted_data = convert_basicsr_to_one_step_format(
                'datasets/test_trajectories_only.json', 
                output_file,
                context_type=context_type,
                data_context=data_context if context_type != 'basic' else None
            )
            
            print(f"✓ Converted {len(converted_data)} examples with {context_type} context")
            
            # Show a sample context
            if converted_data:
                sample_context = converted_data[0]['context']
                print(f"Sample context: {sample_context[:200]}..." if len(sample_context) > 200 else f"Sample context: {sample_context}")
                
        except Exception as e:
            print(f"❌ Failed to convert with {context_type} context: {e}")
    
    # Compare context lengths
    print(f"\n--- Context Length Comparison ---")
    for context_type in ['basic', 'rich', 'superrich']:
        try:
            with open(f'datasets/test_{context_type}_context.jsonl', 'r') as f:
                first_line = f.readline()
                if first_line:
                    data = json.loads(first_line)
                    context = data['context']
                    print(f"{context_type.upper():>10}: {len(context)} characters")
        except Exception as e:
            print(f"{context_type.upper():>10}: Error - {e}")

if __name__ == "__main__":
    test_context_types()
    print(f"\n✓ Context conversion testing completed!")