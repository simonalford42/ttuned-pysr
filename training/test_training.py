"""
Test the training pipeline with a minimal setup.
"""
import sys
import os
sys.path.append('.')

# Test data loading
from datasets import load_dataset
import json

def test_data_loading():
    """Test that the converted data can be loaded properly."""
    print("Testing data loading...")
    
    # Load the datasets
    try:
        hf_datasets = load_dataset(
            "json",
            data_files={
                "train": "data/train_trajectories.jsonl",
                "val": "data/val_trajectories.jsonl",
                "val_target": "data/val_trajectories.jsonl",
            },
        )
        print(f"✓ Loaded datasets successfully")
        print(f"  - Train samples: {len(hf_datasets['train'])}")
        print(f"  - Val samples: {len(hf_datasets['val'])}")
        
        # Print a sample
        sample = hf_datasets['train'][0]
        print(f"  - Sample keys: {list(sample.keys())}")
        print(f"  - Search path length: {len(sample['search_path'])}")
        print(f"  - First search step: {sample['search_path'][0][:100]}...")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load datasets: {e}")
        return False

def test_tokenization():
    """Test tokenization of the data."""
    print("\nTesting tokenization...")
    
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load sample data
        hf_datasets = load_dataset(
            "json",
            data_files={"train": "data/train_trajectories.jsonl"},
        )
        
        sample = hf_datasets['train'][0]
        text = tokenizer.bos_token + " ".join(sample["search_path"]).strip() + tokenizer.eos_token
        
        tokens = tokenizer(text, truncation=True, max_length=512, padding="max_length")
        print(f"✓ Tokenization successful")
        print(f"  - Text length: {len(text)} chars")
        print(f"  - Token count: {len(tokens['input_ids'])}")
        print(f"  - First few tokens: {tokens['input_ids'][:10]}")
        
        return True
    except Exception as e:
        print(f"✗ Tokenization failed: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        from transformers import GPTNeoConfig, GPTNeoForCausalLM
        
        with open('training/gpt-neo-tiny.json', 'r') as f:
            model_config = json.load(f)
        
        config = GPTNeoConfig(**model_config)
        model = GPTNeoForCausalLM(config)
        
        print(f"✓ Model created successfully")
        print(f"  - Parameters: {model.num_parameters():,}")
        print(f"  - Hidden size: {config.hidden_size}")
        print(f"  - Num layers: {config.num_layers}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing MinimalSR training pipeline...")
    print("=" * 50)
    
    success = True
    success &= test_data_loading()
    success &= test_tokenization() 
    success &= test_model_creation()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! Ready to train.")
    else:
        print("✗ Some tests failed. Check the errors above.")