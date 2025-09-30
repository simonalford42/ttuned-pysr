"""
Dataset management for training data generation system.
Handles train/val splits, dataset loading, and conversion to training format.
"""
import json
import os
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import sys
sys.path.append('training')
from convert_trajectories import convert_basicsr_to_one_step_format, split_train_val


class DatasetManager:
    """Manages expression and trace datasets for training"""

    def __init__(self, base_dir: str = "datasets"):
        self.base_dir = Path(base_dir)
        self.expressions_dir = self.base_dir / "expressions"
        self.traces_dir = self.base_dir / "traces"
        self.training_dir = self.base_dir / "training"
        self.metadata_dir = self.base_dir / "metadata"

        # Create directories
        for dir_path in [self.expressions_dir, self.traces_dir, self.training_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def create_dataset_version(self, name: str,
                             expressions_file: str,
                             traces_file: str,
                             description: str = "",
                             val_split: float = 0.2,
                             seed: int = 42) -> str:
        """Create a versioned dataset with train/val split"""

        print(f"=== Creating Dataset Version: {name} ===")
        print(f"Expressions file: {expressions_file}")
        print(f"Traces file: {traces_file}")
        print(f"Validation split: {val_split:.1%}")

        # Convert traces to one-step format
        temp_file = self.training_dir / f"{name}_temp.jsonl"
        print("Converting traces to one-step format...")

        # Load traces file and extract just the trajectories part
        with open(traces_file, 'r') as f:
            trace_data = json.load(f)

        # Extract trajectories from the wrapped format
        if 'trajectories' in trace_data:
            trajectories_only = trace_data['trajectories']
        else:
            trajectories_only = trace_data

        # Save temporary file with just trajectories for conversion
        temp_trace_file = self.training_dir / f"{name}_traces_temp.json"
        with open(temp_trace_file, 'w') as f:
            json.dump(trajectories_only, f, indent=2)

        # For now, use basic context - could be extended later
        converted_data = convert_basicsr_to_one_step_format(str(temp_trace_file), str(temp_file), context_type='basic')

        # Clean up temp trace file
        temp_trace_file.unlink()

        # Generate train/val filenames
        train_file = self.training_dir / f"{name}_train.jsonl"
        val_file = self.training_dir / f"{name}_val.jsonl"

        # Split into train/val
        print("Splitting into train/validation sets...")
        n_train, n_val = split_train_val(str(temp_file), str(train_file), str(val_file), val_split, seed)

        # Remove temporary file
        temp_file.unlink()

        # Create dataset metadata
        metadata = {
            'dataset_name': name,
            'version': '1.0',
            'creation_date': datetime.now().isoformat(),
            'description': description,
            'files': {
                'expressions_source': expressions_file,
                'traces_source': traces_file,
                'train_data': str(train_file),
                'val_data': str(val_file)
            },
            'statistics': {
                'total_examples': len(converted_data),
                'train_examples': n_train,
                'val_examples': n_val,
                'val_split': val_split
            },
            'parameters': {
                'seed': seed,
                'conversion_format': 'one_step_prediction'
            }
        }

        # Save metadata
        metadata_file = self.metadata_dir / f"{name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Dataset created successfully!")
        print(f"  - Train examples: {n_train}")
        print(f"  - Val examples: {n_val}")
        print(f"  - Train file: {train_file}")
        print(f"  - Val file: {val_file}")
        print(f"  - Metadata: {metadata_file}")

        return str(metadata_file)

    def load_dataset_info(self, name: str) -> Dict[str, Any]:
        """Load dataset metadata"""
        metadata_file = self.metadata_dir / f"{name}_metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(f"Dataset metadata not found: {metadata_file}")

        with open(metadata_file, 'r') as f:
            return json.load(f)

    def list_datasets(self) -> List[str]:
        """List available datasets"""
        metadata_files = list(self.metadata_dir.glob("*_metadata.json"))
        dataset_names = [f.stem.replace("_metadata", "") for f in metadata_files]
        return sorted(dataset_names)

    def get_dataset_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a dataset"""
        metadata = self.load_dataset_info(name)

        stats = {
            'name': name,
            'creation_date': metadata['creation_date'],
            'description': metadata.get('description', ''),
            'total_examples': metadata['statistics']['total_examples'],
            'train_examples': metadata['statistics']['train_examples'],
            'val_examples': metadata['statistics']['val_examples'],
            'val_split': metadata['statistics']['val_split']
        }

        # Add file sizes if files exist
        for split in ['train_data', 'val_data']:
            filepath = metadata['files'][split]
            if os.path.exists(filepath):
                stats[f'{split}_size_mb'] = os.path.getsize(filepath) / (1024 * 1024)

        return stats

    def validate_dataset(self, name: str) -> bool:
        """Validate that all dataset files exist and are readable"""
        try:
            metadata = self.load_dataset_info(name)

            # Check that all files exist
            required_files = ['train_data', 'val_data']
            for file_key in required_files:
                filepath = metadata['files'][file_key]
                if not os.path.exists(filepath):
                    print(f"❌ Missing file: {filepath}")
                    return False

                # Try to read a few lines
                with open(filepath, 'r') as f:
                    for i, line in enumerate(f):
                        if i >= 3:  # Check first 3 lines
                            break
                        json.loads(line)  # Will raise if invalid JSON

            print(f"✓ Dataset '{name}' validation passed")
            return True

        except Exception as e:
            print(f"❌ Dataset validation failed: {e}")
            return False

    def create_mini_dataset(self, source_name: str, mini_name: str, n_examples: int = 100) -> str:
        """Create a smaller version of existing dataset for testing"""
        print(f"Creating mini dataset '{mini_name}' from '{source_name}' with {n_examples} examples")

        # Load source dataset
        source_metadata = self.load_dataset_info(source_name)
        source_train_file = source_metadata['files']['train_data']

        # Read source training data
        source_data = []
        with open(source_train_file, 'r') as f:
            for line in f:
                if line.strip():
                    source_data.append(json.loads(line))

        # Sample subset
        random.seed(42)
        if len(source_data) <= n_examples:
            mini_data = source_data
        else:
            mini_data = random.sample(source_data, n_examples)

        # Create mini files
        mini_train_file = self.training_dir / f"{mini_name}_train.jsonl"
        mini_val_file = self.training_dir / f"{mini_name}_val.jsonl"

        # Split mini data (80/20)
        val_size = max(1, int(len(mini_data) * 0.2))
        mini_val = mini_data[:val_size]
        mini_train = mini_data[val_size:]

        # Write files
        with open(mini_train_file, 'w') as f:
            for item in mini_train:
                f.write(json.dumps(item) + '\n')

        with open(mini_val_file, 'w') as f:
            for item in mini_val:
                f.write(json.dumps(item) + '\n')

        # Create metadata
        mini_metadata = {
            'dataset_name': mini_name,
            'version': '1.0',
            'creation_date': datetime.now().isoformat(),
            'description': f"Mini version of {source_name} with {len(mini_data)} examples",
            'parent_dataset': source_name,
            'files': {
                'train_data': str(mini_train_file),
                'val_data': str(mini_val_file)
            },
            'statistics': {
                'total_examples': len(mini_data),
                'train_examples': len(mini_train),
                'val_examples': len(mini_val),
                'val_split': val_size / len(mini_data)
            },
            'parameters': {
                'source_dataset': source_name,
                'sample_size': n_examples,
                'seed': 42
            }
        }

        # Save metadata
        mini_metadata_file = self.metadata_dir / f"{mini_name}_metadata.json"
        with open(mini_metadata_file, 'w') as f:
            json.dump(mini_metadata, f, indent=2)

        print(f"✓ Mini dataset created: {len(mini_train)} train, {len(mini_val)} val examples")
        return str(mini_metadata_file)


def create_training_dataset_pipeline(name: str,
                                   n_expressions: int = 100,
                                   basicsr_generations: int = 10,
                                   seed: int = 42) -> str:
    """End-to-end pipeline: generate expressions -> traces -> training dataset"""

    print(f"=== Full Training Dataset Pipeline: {name} ===")
    print(f"Expressions: {n_expressions}, Generations: {basicsr_generations}")

    from generate_expressions import generate_training_expressions
    from generate_traces import generate_traces_from_expressions

    # Step 1: Generate expressions
    print("\nStep 1: Generating expressions...")
    expressions_file = generate_training_expressions(
        n_expressions=n_expressions,
        max_degree=2,  # Keep simple for training
        max_vars=2,
        seed=seed
    )

    # Step 2: Generate traces
    print("\nStep 2: Generating BasicSR traces...")
    basicsr_params = {
        'population_size': 20,
        'num_generations': basicsr_generations,
        'max_depth': 4,
        'max_size': 15,
        'tournament_size': 3,
        'time_limit': 30
    }

    traces_file = generate_traces_from_expressions(
        expressions_file=expressions_file,
        basicsr_params=basicsr_params,
        max_expressions=n_expressions
    )

    # Step 3: Create training dataset
    print("\nStep 3: Creating training dataset...")
    dm = DatasetManager()
    metadata_file = dm.create_dataset_version(
        name=name,
        expressions_file=expressions_file,
        traces_file=traces_file,
        description=f"Generated dataset with {n_expressions} expressions, {basicsr_generations} generations each",
        val_split=0.2,
        seed=seed
    )

    print(f"\n✓ Complete pipeline finished!")
    print(f"Dataset metadata: {metadata_file}")

    return metadata_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset management utilities")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--info", help="Show info for dataset")
    parser.add_argument("--validate", help="Validate dataset")
    parser.add_argument("--create_mini", nargs=2, metavar=('SOURCE', 'NAME'),
                       help="Create mini dataset from source")
    parser.add_argument("--pipeline", help="Run full pipeline to create dataset")
    parser.add_argument("--n_expressions", type=int, default=100, help="Number of expressions")
    parser.add_argument("--generations", type=int, default=10, help="BasicSR generations")

    args = parser.parse_args()

    dm = DatasetManager()

    if args.list:
        datasets = dm.list_datasets()
        print("Available datasets:")
        for name in datasets:
            stats = dm.get_dataset_stats(name)
            print(f"  - {name}: {stats['train_examples']} train, {stats['val_examples']} val")

    elif args.info:
        stats = dm.get_dataset_stats(args.info)
        print(f"Dataset: {stats['name']}")
        print(f"Created: {stats['creation_date']}")
        print(f"Description: {stats['description']}")
        print(f"Examples: {stats['train_examples']} train, {stats['val_examples']} val")

    elif args.validate:
        dm.validate_dataset(args.validate)

    elif args.create_mini:
        source, name = args.create_mini
        dm.create_mini_dataset(source, name, n_examples=100)

    elif args.pipeline:
        create_training_dataset_pipeline(
            args.pipeline,
            args.n_expressions,
            args.generations
        )
