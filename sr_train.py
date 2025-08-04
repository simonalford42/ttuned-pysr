"""
Symbolic Regression Transformer Training
Adapted from Stream of Search training code for our trajectory data
"""

import argparse
import json
import os
import random
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, GPTNeoConfig, GPTNeoForCausalLM
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import wandb


class SRTrajectoryDataset(Dataset):
    """Dataset for symbolic regression trajectories"""
    
    def __init__(self, trajectory_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load trajectory data
        with open(trajectory_file, 'r') as f:
            data = json.load(f)
        
        print(f"Converting {data['metadata']['total_trajectories']} trajectories to training format...")
        
        # Convert each trajectory to training sequences
        for problem_name, trajectories in data['trajectories'].items():
            for trajectory in trajectories:
                self._process_trajectory(trajectory, problem_name)
        
        print(f"Created {len(self.data)} training examples")
    
    def _process_trajectory(self, trajectory, problem_name):
        """Convert a single trajectory to training sequences"""
        trajectory_data = trajectory['trajectory_data']
        
        for i, generation in enumerate(trajectory_data):
            # Create input sequence: problem description + current population state
            input_text = self._format_generation_input(generation, problem_name, trajectory)
            
            # Create target sequence: next generation's best expression (if available)
            if i + 1 < len(trajectory_data):
                next_gen = trajectory_data[i + 1]
                target_text = self._format_generation_target(next_gen)
            else:
                # For last generation, target is the final solution
                target_text = generation['best_expression']
            
            # Combine input and target for causal language modeling
            full_text = input_text + " -> " + target_text
            
            self.data.append({
                'text': full_text,
                'input_text': input_text,
                'target_text': target_text,
                'problem': problem_name,
                'generation': i
            })
    
    def _format_generation_input(self, generation, problem_name, trajectory):
        """Format generation state as input text"""
        # Include problem context
        text = f"Problem: {problem_name} | "
        text += f"Generation {generation['generation']} | "
        text += f"Population size: {generation['population_size']} | "
        text += f"Best fitness: {generation['best_fitness']:.6f} | "
        text += f"Diversity: {generation['population_diversity']} | "
        text += f"Current best: {generation['best_expression']}"
        return text
    
    def _format_generation_target(self, next_generation):
        """Format next generation's best as target"""
        return next_generation['best_expression']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def create_small_model_config():
    """Create a very small GPT-Neo config for fast training"""
    return {
        "activation_function": "gelu_new",
        "attention_dropout": 0.1,
        "attention_types": [[["global"], 4]],  # Much smaller
        "embed_dropout": 0.1,
        "hidden_size": 256,  # Much smaller than 1024
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "max_position_embeddings": 1024,  # Smaller context
        "model_type": "gpt_neo",
        "num_heads": 8,  # Smaller
        "num_layers": 4,  # Much smaller than 16
        "resid_dropout": 0.1,
        "vocab_size": 50257,
        "window_size": 1024,
        "bos_token_id": 50256,
        "eos_token_id": 50256
    }


def tokenize_dataset(dataset, tokenizer, max_length):
    """Tokenize the dataset for training"""
    
    def tokenize_function(examples):
        # Tokenize the full text for causal language modeling
        texts = [example['text'] for example in examples]
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].clone()
        
        return tokenized
    
    # Convert to tokenized format
    tokenized_data = []
    for i in range(0, len(dataset), 32):  # Process in batches
        batch = dataset[i:i+32]
        tokenized_batch = tokenize_function(batch)
        
        for j in range(len(batch)):
            tokenized_data.append({
                'input_ids': tokenized_batch['input_ids'][j],
                'attention_mask': tokenized_batch['attention_mask'][j],
                'labels': tokenized_batch['labels'][j]
            })
    
    return tokenized_data


class TokenizedDataset(Dataset):
    """Wrapper for tokenized data"""
    
    def __init__(self, tokenized_data):
        self.data = tokenized_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory_file', type=str, required=True,
                       help='Path to trajectory JSON file')
    parser.add_argument('--output_dir', type=str, default='./sr_model_output',
                       help='Output directory for model')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(42)
    torch.manual_seed(42)
    
    print("Setting up Symbolic Regression Transformer Training...")
    print(f"Trajectory file: {args.trajectory_file}")
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(
            project="sr-transformer",
            name=f"sr-train-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                'trajectory_file': args.trajectory_file,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'max_length': args.max_length
            }
        )
    
    # Initialize tokenizer (using GPT-Neo tokenizer)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create small model
    print("Creating small model...")
    model_config = GPTNeoConfig(**create_small_model_config())
    model = GPTNeoForCausalLM(model_config)
    
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Load and process dataset
    print("Loading trajectory dataset...")
    dataset = SRTrajectoryDataset(args.trajectory_file, tokenizer, args.max_length)
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_data = tokenize_dataset(dataset, tokenizer, args.max_length)
    tokenized_dataset = TokenizedDataset(tokenized_data)
    
    # Split into train/val (90/10)
    train_size = int(0.9 * len(tokenized_dataset))
    val_size = len(tokenized_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        tokenized_dataset, [train_size, val_size]
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        evaluation_strategy='steps',
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to='wandb' if args.wandb else None,
        learning_rate=args.lr,
        lr_scheduler_type='cosine',
        dataloader_pin_memory=False,  # Disable for compatibility
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling, not masked
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"Training complete! Model saved to {args.output_dir}")
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()