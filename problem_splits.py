"""
Problem splits for training/validation/test sets for transformer training
"""

import numpy as np
from simple_problems import (
    ULTRA_SIMPLE_PROBLEMS, 
    SIMPLE_PROBLEMS, 
    HARDER_PROBLEMS,
    ALL_SIMPLE_PROBLEMS,
    ALL_PROBLEMS
)


def create_problem_splits(problems, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Split problems into train/validation/test sets
    
    Args:
        problems: List of problem functions
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set  
        test_ratio: Fraction for test set
        seed: Random seed for reproducible splits
    
    Returns:
        dict with 'train', 'val', 'test' keys containing problem lists
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    rng = np.random.RandomState(seed)
    shuffled_problems = problems.copy()
    rng.shuffle(shuffled_problems)
    
    n_total = len(problems)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val  # Ensure all problems are used
    
    splits = {
        'train': shuffled_problems[:n_train],
        'val': shuffled_problems[n_train:n_train + n_val],
        'test': shuffled_problems[n_train + n_val:n_train + n_val + n_test]
    }
    
    return splits


def get_ultra_simple_splits():
    """Get train/val/test splits for ultra-simple problems"""
    return create_problem_splits(ULTRA_SIMPLE_PROBLEMS)


def get_simple_splits():
    """Get train/val/test splits for simple problems"""
    return create_problem_splits(SIMPLE_PROBLEMS)


def get_all_simple_splits():
    """Get train/val/test splits for all simple problems (ultra + regular)"""
    return create_problem_splits(ALL_SIMPLE_PROBLEMS)


def get_comprehensive_splits():
    """Get train/val/test splits for all problems including harder ones"""
    return create_problem_splits(ALL_PROBLEMS)


def print_split_summary(splits, split_name):
    """Print summary of a problem split"""
    print(f"\n=== {split_name} Problem Splits ===")
    
    for split_type, problems in splits.items():
        print(f"\n{split_type.upper()} ({len(problems)} problems):")
        for i, problem in enumerate(problems, 1):
            print(f"  {i}. {problem.__name__}: {problem.__doc__}")
    
    total = sum(len(problems) for problems in splits.values())
    print(f"\nTotal problems: {total}")
    print(f"Split ratios: Train={len(splits['train'])/total:.1%}, "
          f"Val={len(splits['val'])/total:.1%}, Test={len(splits['test'])/total:.1%}")


def save_split_info(splits, filename):
    """Save split information to a text file"""
    import os
    os.makedirs('data', exist_ok=True)
    
    filepath = os.path.join('data', filename)
    with open(filepath, 'w') as f:
        f.write("Problem Splits for Transformer Training\n")
        f.write("=" * 50 + "\n\n")
        
        for split_type, problems in splits.items():
            f.write(f"{split_type.upper()} SET ({len(problems)} problems):\n")
            for i, problem in enumerate(problems, 1):
                f.write(f"  {i}. {problem.__name__}: {problem.__doc__}\n")
            f.write("\n")
        
        total = sum(len(problems) for problems in splits.values())
        f.write(f"Total problems: {total}\n")
        f.write(f"Split ratios: Train={len(splits['train'])/total:.1%}, "
                f"Val={len(splits['val'])/total:.1%}, Test={len(splits['test'])/total:.1%}\n")
    
    print(f"Split information saved to {filepath}")
    return filepath


def main():
    """Create and display different problem splits"""
    
    print("Creating problem splits for transformer training...")
    
    # Create different splits
    ultra_splits = get_ultra_simple_splits()
    simple_splits = get_simple_splits() 
    all_simple_splits = get_all_simple_splits()
    comprehensive_splits = get_comprehensive_splits()
    
    # Print summaries
    print_split_summary(ultra_splits, "Ultra-Simple")
    print_split_summary(simple_splits, "Simple")
    print_split_summary(all_simple_splits, "All Simple")
    print_split_summary(comprehensive_splits, "Comprehensive")
    
    # Save split information
    save_split_info(ultra_splits, "ultra_simple_splits.txt")
    save_split_info(simple_splits, "simple_splits.txt") 
    save_split_info(all_simple_splits, "all_simple_splits.txt")
    save_split_info(comprehensive_splits, "comprehensive_splits.txt")
    
    # Return the splits for use in other scripts
    return {
        'ultra_simple': ultra_splits,
        'simple': simple_splits,
        'all_simple': all_simple_splits,
        'comprehensive': comprehensive_splits
    }


# Export the splits for easy import
ULTRA_SIMPLE_SPLITS = get_ultra_simple_splits()
SIMPLE_SPLITS = get_simple_splits()
ALL_SIMPLE_SPLITS = get_all_simple_splits()
COMPREHENSIVE_SPLITS = get_comprehensive_splits()


if __name__ == "__main__":
    splits = main()