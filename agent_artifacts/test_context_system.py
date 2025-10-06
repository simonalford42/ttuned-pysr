#!/usr/bin/env python3
"""
Test the new context system with different context types.
"""

import sys
sys.path.append('training')

import numpy as np
from format_utils import (
    format_context, compute_data_statistics, 
    format_rich_context, format_superrich_context,
    create_text_plot
)

def test_basic_context():
    """Test basic context formatting"""
    print("=== Testing Basic Context ===")
    
    variables = ['x0', 'x1']
    operators = ['+', '-', '*', '/']
    constants = [1.0, 2.0]
    
    context = format_context(0, variables, operators, constants, context_type='basic')
    print(f"Basic context: {context}")
    
    return context

def test_rich_context():
    """Test rich context with data statistics"""
    print("\n=== Testing Rich Context ===")
    
    # Generate sample data - quadratic function
    np.random.seed(42)
    X = np.random.uniform(-3, 3, size=(50, 2))
    y = X[:, 0]**2 + 2*X[:, 1] - 1  # x0^2 + 2*x1 - 1
    
    variables = ['x0', 'x1']
    operators = ['+', '-', '*', '/']
    constants = [1.0, 2.0]
    
    # Compute data statistics
    data_stats = compute_data_statistics(X, y)
    print(f"Computed data stats: {data_stats}")
    
    context = format_context(0, variables, operators, constants, context_type='rich', data_stats=data_stats)
    print(f"Rich context: {context}")
    
    return context, data_stats

def test_superrich_context():
    """Test superrich context with text plot"""
    print("\n=== Testing SuperRich Context ===")
    
    # Generate 1D data for better plotting
    np.random.seed(42)
    X = np.random.uniform(-2, 2, size=(30, 1))
    X = np.sort(X, axis=0)  # Sort for better plotting
    y = X[:, 0]**3 - X[:, 0]  # x^3 - x (cubic with interesting shape)
    
    variables = ['x0']
    operators = ['+', '-', '*', '/']
    constants = [1.0, 2.0]
    
    # Compute data statistics
    data_stats = compute_data_statistics(X, y)
    
    # Add text plot
    data_stats['text_plot'] = create_text_plot(X, y)
    print(f"Text plot: {data_stats['text_plot']}")
    
    context = format_context(0, variables, operators, constants, context_type='superrich', data_stats=data_stats)
    print(f"SuperRich context: {context}")
    
    return context

def test_context_differences():
    """Compare all three context types side by side"""
    print("\n" + "="*60)
    print("CONTEXT COMPARISON")
    print("="*60)
    
    # Same data for all tests
    np.random.seed(42)
    X = np.random.uniform(-2, 2, size=(30, 2))
    y = X[:, 0]**2 + X[:, 1]**2  # Simple quadratic
    
    variables = ['x0', 'x1']
    operators = ['+', '-', '*', '/']
    constants = [1.0, 2.0]
    
    data_stats = compute_data_statistics(X, y)
    data_stats['text_plot'] = create_text_plot(X, y)
    
    print("Variables:", variables)
    print("Target function: x0^2 + x1^2")
    print("Data range:", f"X: [{X.min():.2f}, {X.max():.2f}], y: [{y.min():.2f}, {y.max():.2f}]")
    print()
    
    for context_type in ['basic', 'rich', 'superrich']:
        context = format_context(0, variables, operators, constants, context_type, data_stats)
        print(f"{context_type.upper():>10}: {context}")
    
    print("\nContext length comparison:")
    for context_type in ['basic', 'rich', 'superrich']:
        context = format_context(0, variables, operators, constants, context_type, data_stats)
        print(f"{context_type.upper():>10}: {len(context)} characters")

if __name__ == "__main__":
    print("Testing Enhanced Context System for Neural SR")
    print("="*50)
    
    # Test each context type
    basic_context = test_basic_context()
    rich_context, stats = test_rich_context()
    superrich_context = test_superrich_context()
    
    # Compare all types
    test_context_differences()
    
    print(f"\n✓ All context system tests completed successfully!")
    print("\nNext steps:")
    print("- Use rich context when training models on diverse datasets")  
    print("- Use superrich context for complex function approximation")
    print("- Basic context remains default for compatibility")