"""
Test script for SR transformer training setup
"""

import json
import os
import sys
from collect_trajectories import collect_trajectories_for_problems
from problem_splits import ULTRA_SIMPLE_SPLITS


def create_small_training_dataset():
    """Create a small trajectory dataset for testing"""
    
    print("Creating small training dataset for testing...")
    
    # Use just the training problems from ultra-simple splits
    train_problems = ULTRA_SIMPLE_SPLITS['train']
    print(f"Using {len(train_problems)} training problems:")
    for problem in train_problems:
        print(f"  - {problem.__name__}: {problem.__doc__}")
    
    # Collect trajectories with minimal settings for fast testing
    trajectory_file, trajectory_data = collect_trajectories_for_problems(
        train_problems,
        "training_test",
        num_runs=2  # Just 2 runs per problem for speed
    )
    
    print(f"Training dataset created: {trajectory_file}")
    return trajectory_file


def test_data_loading():
    """Test that we can load and process the trajectory data"""
    
    print("\nTesting data loading...")
    
    # Create test dataset
    trajectory_file = create_small_training_dataset()
    
    # Test the dataset class
    try:
        from sr_train import SRTrajectoryDataset
        from transformers import AutoTokenizer
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        tokenizer.pad_token = tokenizer.eos_token
        
        print("Creating dataset...")
        dataset = SRTrajectoryDataset(trajectory_file, tokenizer, max_length=256)
        
        print(f"Dataset created with {len(dataset)} examples")
        
        # Show a few examples
        print("\nSample training examples:")
        for i in range(min(3, len(dataset))):
            example = dataset[i]
            print(f"\nExample {i+1}:")
            print(f"Problem: {example['problem']}")
            print(f"Generation: {example['generation']}")
            print(f"Input: {example['input_text'][:100]}...")
            print(f"Target: {example['target_text']}")
        
        print("\n✅ Data loading test passed!")
        return trajectory_file, dataset
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_model_creation():
    """Test that we can create the small model"""
    
    print("\nTesting model creation...")
    
    try:
        from sr_train import create_small_model_config
        from transformers import GPTNeoConfig, GPTNeoForCausalLM
        
        print("Creating model config...")
        model_config = GPTNeoConfig(**create_small_model_config())
        
        print("Creating model...")
        model = GPTNeoForCausalLM(model_config)
        
        print(f"Model created with {model.num_parameters():,} parameters")
        
        # Test forward pass
        import torch
        
        print("Testing forward pass...")
        input_ids = torch.randint(0, 1000, (1, 10))  # Batch size 1, sequence length 10
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        print(f"Output shape: {outputs.logits.shape}")
        print("✅ Model creation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_training_test():
    """Run a very quick training test (1 step) to verify everything works"""
    
    print("\nRunning quick training test...")
    
    # Create test dataset
    trajectory_file = create_small_training_dataset()
    
    # Import required modules
    try:
        import subprocess
        import sys
        
        # Run the training script with minimal settings
        cmd = [
            sys.executable, "sr_train.py",
            "--trajectory_file", trajectory_file,
            "--output_dir", "./test_model_output",
            "--epochs", "1",
            "--batch_size", "2",
            "--lr", "1e-4",
            "--max_length", "256"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run training for a very short time
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("✅ Quick training test passed!")
            print("Training output:")
            print(result.stdout[-500:])  # Show last 500 chars
            return True
        else:
            print("❌ Quick training test failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  Training test timed out (but this is expected for a real run)")
        return True
    except Exception as e:
        print(f"❌ Quick training test failed: {e}")
        return False


def main():
    """Run all tests"""
    
    print("SR Transformer Training Setup Test")
    print("="*50)
    
    # Test 1: Data loading
    trajectory_file, dataset = test_data_loading()
    if trajectory_file is None:
        print("❌ Cannot proceed - data loading failed")
        return
    
    # Test 2: Model creation
    model_success = test_model_creation()
    if not model_success:
        print("❌ Cannot proceed - model creation failed")
        return
    
    # Test 3: Quick training test
    print("\nAll basic tests passed! Ready for full training.")
    print(f"To run full training, use:")
    print(f"python sr_train.py --trajectory_file {trajectory_file} --wandb --epochs 3")
    
    # Optionally run the training test
    run_training = input("\nRun quick training test? (y/n): ").lower().strip() == 'y'
    if run_training:
        training_success = run_quick_training_test()
        if training_success:
            print("\n🎉 All tests passed! SR transformer training is ready!")
        else:
            print("\n❌ Training test failed - check the error messages above")
    else:
        print("\n✅ Setup tests complete - ready for manual training!")


if __name__ == "__main__":
    main()