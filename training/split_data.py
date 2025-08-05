"""
Split trajectory data into train/validation sets.
"""
import json
import random

def split_trajectories(input_file, train_file, val_file, val_ratio=0.2, seed=42):
    """Split trajectories into train and validation sets."""
    random.seed(seed)
    
    # Read all trajectories
    trajectories = []
    with open(input_file, 'r') as f:
        for line in f:
            trajectories.append(json.loads(line.strip()))
    
    # Shuffle and split
    random.shuffle(trajectories)
    split_idx = int(len(trajectories) * (1 - val_ratio))
    
    train_data = trajectories[:split_idx]
    val_data = trajectories[split_idx:]
    
    # Write train set
    with open(train_file, 'w') as f:
        for trajectory in train_data:
            f.write(json.dumps(trajectory) + '\n')
    
    # Write validation set
    with open(val_file, 'w') as f:
        for trajectory in val_data:
            f.write(json.dumps(trajectory) + '\n')
    
    print(f"Split {len(trajectories)} trajectories into {len(train_data)} train and {len(val_data)} validation")

if __name__ == "__main__":
    split_trajectories(
        "../data/all_trajectories.jsonl",
        "../data/train_trajectories.jsonl", 
        "../data/val_trajectories.jsonl"
    )