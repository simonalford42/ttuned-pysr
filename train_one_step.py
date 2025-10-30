"""
Training script for one-step symbolic regression prediction.
Trains a transformer to predict the next generation given current population.
"""
import argparse
import json
import os
import random
from typing import Dict, Any
import pickle
import gzip

import torch
import numpy as np
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import GPTNeoConfig, GPTNeoForCausalLM
from transformers import Trainer
from transformers import TrainingArguments

import wandb
from format_utils import format_input_part, format_target_part
from datetime import datetime
from point_embedder import E2EPointEmbedder


class ModelWithInputEmbedder(torch.nn.Module):
    """Wrapper that combines a language model with an input embedder"""
    def __init__(self, base_model, input_embedder):
        super().__init__()
        self.base_model = base_model
        self.input_embedder = input_embedder
        # Expose config for Trainer
        self.config = base_model.config

    def forward(self, inputs_embeds=None, labels=None, **kwargs):
        """Forward pass - expects inputs_embeds from collator"""
        return self.base_model(inputs_embeds=inputs_embeds, labels=labels, **kwargs)

    def generate(self, **kwargs):
        return self.base_model.generate(**kwargs)


def main(config, checkpoint=None, resume=False, reset=False):
    # read config from a json config file
    with open(config, "r") as f:
        config = json.load(f)

    # set seeds
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # set up accelerator
    accelerator = Accelerator()

    # Enable wandb if config contains wandb settings
    if "wandb" in config and accelerator.is_main_process:
        wandb_kwargs = config["wandb"]
        wandb.init(
            project=wandb_kwargs["project"],
            entity=wandb_kwargs["entity"],
            name=config["name"],
            config=config,
            dir=wandb_kwargs.get("dir", "./wandb"),
        )

        # Log SLURM metadata (job id and stdout file path)
        # SBATCH often uses: -o out/%A.out, where %A expands to the SLURM job ID
        slurm_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
        slurm_out = f"out/{slurm_id}.out" if slurm_id else None
        wandb.config.update({
            "slurm_job_id": slurm_id,
            "slurm_out": slurm_out,
        }, allow_val_change=True)

    # Build model/tokenizer: either from pretrained (reset) or from config (fresh)
    default_pretrained = config.get("model_name_or_path", "cerebras/Cerebras-GPT-256M")

    if reset:
        # Fine-tune from a pretrained checkpoint or HF Hub model
        source = checkpoint or default_pretrained
        print(f"Loading pretrained model/tokenizer from: {source}")
        tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(source)
    else:
        # Fresh init from config (kept for backward compatibility)
        with open(config["model_config"], "r") as f:
            model_config = json.load(f)
        # Default to GPTNeo for existing configs; for other families, use AutoConfig.for_model with model_type in config
        if isinstance(model_config, dict) and model_config.get("model_type", "gpt_neo") != "gpt_neo":
            model_type = model_config.pop("model_type")
            cfg = AutoConfig.for_model(model_type, **model_config)
            model = AutoModelForCausalLM.from_config(cfg)
        else:
            cfg = GPTNeoConfig(**model_config)
            model = GPTNeoForCausalLM(cfg)
        # Tokenizer fallback for fresh training
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

    # Add special tokens for our format
    special_tokens = {
        "additional_special_tokens": [
            "<CONTEXT>", "<POPULATION>", "<FITNESS>", "<TARGET>"
        ]
    }
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    model.base_model.resize_token_embeddings(len(tokenizer))

    # Precision will be controlled by TrainingArguments/Accelerate; avoid manual casting here

    print(f"Number of parameters: {model.num_parameters()}")
    print(f"Added {num_added_tokens} special tokens")

    # Initialize input embedder if specified
    input_embedder_type = config.get("input_embedder", None)
    input_embedder = None
    expressions_data_cache = {}

    if input_embedder_type == "e2e":
        print("Initializing E2E input embedder...")
        hidden_size = model.config.hidden_size
        input_embedder = E2EPointEmbedder(
            max_points=64,
            max_input_dim=10,
            hidden_size=hidden_size,
            mlp_hidden_size=512,
        )
        num_embedder_params = sum(p.numel() for p in input_embedder.parameters())
        print(f"Input embedder parameters: {num_embedder_params:,}")

        # Wrap model with embedder
        base_model = model
        model = ModelWithInputEmbedder(base_model, input_embedder)
        print(f"Wrapped model with input embedder")
        print(f"Total parameters (model + embedder): {model.base_model.num_parameters() + num_embedder_params:,}")
    elif input_embedder_type is not None:
        raise ValueError(f"Unknown input_embedder type: {input_embedder_type}")

    # load dataset - our training-ready format
    train_file = os.path.join(config["data_dir"], config["train_file"])
    val_file = os.path.join(config["data_dir"], config["val_file"])

    hf_datasets = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "val": val_file,
        },
    )

    # Optionally limit number of examples for faster debug runs
    num_train = int(config.get("num_train", -1))
    num_val = int(config.get("num_val", -1))

    if num_train > 0:
        hf_datasets["train"] = hf_datasets["train"].select(
            range(min(num_train, len(hf_datasets["train"])))
        )

    if num_val > 0 and "val" in hf_datasets:
        hf_datasets["val"] = hf_datasets["val"].select(
            range(min(num_val, len(hf_datasets["val"])))
        )

    context_length = config["context_length"]
    # Respect model's maximum context window if known
    max_ctx = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(max_ctx, int) and max_ctx > 0 and context_length > max_ctx:
        print(f"Requested context_length {context_length} exceeds model max {max_ctx}; clamping to {max_ctx}.")
        context_length = max_ctx
    tokenizer.model_max_length = context_length
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model, "config", None) is not None and getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    def tokenize(element):
        """
        Tokenize context/population/target triplets for supervised learning.
        Only trains on target tokens, not context/population.
        """
        input_ids_list = []
        labels_list = []
        expression_ids = []
        source_files = []

        for i in range(len(element["context"])):
            context = element["context"][i]
            population = element["population"][i]
            target = element["target"][i]

            # Create input part (context + population + next prompt)
            input_part = format_input_part(tokenizer.bos_token, context, population)

            # Create target part
            target_part = format_target_part(target, tokenizer.eos_token)

            # Tokenize parts separately to get lengths
            input_tokens = tokenizer(input_part, add_special_tokens=False)
            target_tokens = tokenizer(target_part, add_special_tokens=False)

            # Create full sequence
            full_input_ids = input_tokens["input_ids"] + target_tokens["input_ids"]

            # Truncate if necessary
            if len(full_input_ids) > context_length:
                full_input_ids = full_input_ids[:context_length]

            # Pad to max length (no attention_mask; rely on causal mask)
            full_input_ids = full_input_ids + [tokenizer.pad_token_id] * (context_length - len(full_input_ids))

            # Create labels: -100 for input part, actual tokens for target part
            labels = [-100] * len(input_tokens["input_ids"])
            labels.extend(target_tokens["input_ids"])

            # Truncate labels if necessary
            if len(labels) > context_length:
                print(f'truncating! wish we had length {len(labels)}')
                labels = labels[:context_length]

            # Pad labels
            labels = labels + [-100] * (context_length - len(labels))

            input_ids_list.append(full_input_ids)
            labels_list.append(labels)

            # Keep expression_id and source file if using embedder
            if input_embedder is not None:
                expression_ids.append(element["expression_id"][i] if "expression_id" in element else -1)
                source_files.append(element["metadata"][i].get("source_expressions_file", "") if "metadata" in element else "")

        result = {
            "input_ids": torch.tensor(input_ids_list),
            "labels": torch.tensor(labels_list),
        }

        # Add expression metadata if using embedder
        if input_embedder is not None:
            result["expression_id"] = expression_ids
            result["source_expressions_file"] = source_files

        return result

    # tokenize dataset
    map_num_proc = config.get("map_num_proc", None)
    tokenized_datasets = hf_datasets.map(
        tokenize,
        batched=True,
        remove_columns=hf_datasets["train"].column_names,
        num_proc=map_num_proc if map_num_proc else None,
        desc="Tokenizing datasets"
    )

    # Length distribution debug code removed per request.

    # Create data collator
    if input_embedder is not None:
        # Custom collator that handles input embeddings
        class InputEmbeddingCollator:
            def __init__(self, model_wrapper, tokenizer, expressions_cache):
                self.model_wrapper = model_wrapper  # ModelWithInputEmbedder instance
                self.tokenizer = tokenizer
                self.expressions_cache = expressions_cache  # Cache loaded expression files

            def load_expressions_file(self, filepath):
                """Load and cache expressions file"""
                if filepath not in self.expressions_cache:
                    with gzip.open(filepath, 'rb') as f:
                        data = pickle.load(f)
                    # Index by expression ID for fast lookup
                    self.expressions_cache[filepath] = {
                        expr['id']: (expr['X_data'], expr['y_data'])
                        for expr in data['expressions']
                    }
                return self.expressions_cache[filepath]

            def __call__(self, features):
                # Collate regular features
                batch = {
                    "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
                    "labels": torch.stack([torch.tensor(f["labels"]) for f in features]),
                }

                # Load X, y data and compute embeddings
                if "expression_id" in features[0]:
                    X_batch = []
                    y_batch = []

                    for f in features:
                        expr_id = f["expression_id"]
                        source_file = f["source_expressions_file"]

                        # Load expressions data
                        expressions_data = self.load_expressions_file(source_file)
                        X, y = expressions_data[expr_id]

                        X_batch.append(torch.from_numpy(X))
                        y_batch.append(torch.from_numpy(y))

                    # Pad X, y to same length within batch
                    max_points = max(X.shape[0] for X in X_batch)
                    max_vars = max(X.shape[1] for X in X_batch)

                    X_padded = torch.zeros(len(X_batch), max_points, max_vars)
                    y_padded = torch.zeros(len(y_batch), max_points)

                    for i, (X, y) in enumerate(zip(X_batch, y_batch)):
                        n_points, n_vars = X.shape
                        X_padded[i, :n_points, :n_vars] = X
                        y_padded[i, :n_points] = y

                    # Compute embeddings: (batch_size, 64, hidden_size)
                    # NOTE: No torch.no_grad() here - we want gradients to flow!
                    prefix_embeds = self.model_wrapper.input_embedder(X_padded, y_padded)

                    # Get token embeddings: (batch_size, seq_len, hidden_size)
                    token_embeds = self.model_wrapper.base_model.get_input_embeddings()(batch["input_ids"])

                    # Concatenate: [prefix_embeds, token_embeds]
                    # Result: (batch_size, 64 + seq_len, hidden_size)
                    batch["inputs_embeds"] = torch.cat([prefix_embeds, token_embeds], dim=1)

                    # Adjust labels to account for prefix (add 64 positions of -100)
                    prefix_labels = torch.full((len(features), 64), -100, dtype=torch.long)
                    batch["labels"] = torch.cat([prefix_labels, batch["labels"]], dim=1)

                    # Remove input_ids since we're using inputs_embeds
                    del batch["input_ids"]

                return batch

        data_collator = InputEmbeddingCollator(model, tokenizer, expressions_data_cache)
        print("Using custom InputEmbeddingCollator")
    else:
        # Use simple collator that preserves our precomputed labels
        from transformers import default_data_collator
        data_collator = default_data_collator
        print("Using default_data_collator")

    print("tokenized dataset", tokenized_datasets)

    # prepare training arguments from config
    training_config = config["training_args"]

    # Check if output directory exists and create a shared, deterministic suffix across ranks
    output_dir = training_config["output_dir"]
    if os.path.exists(output_dir) and not resume:
        # Prefer SLURM_JOB_ID to ensure same suffix across all ranks
        shared_id = os.environ.get("SLURM_JOB_ID", None)
        run_id_file = f"{output_dir}.run_id"

        if shared_id is None:
            # Only main process generates a timestamp and writes it; others read it
            if accelerator.is_main_process:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                try:
                    with open(run_id_file, "w") as f:
                        f.write(timestamp)
                except Exception:
                    # Fallback to timestamp without file coordination
                    shared_id = timestamp
                else:
                    shared_id = timestamp
            accelerator.wait_for_everyone()
            if shared_id is None and os.path.exists(run_id_file):
                try:
                    with open(run_id_file, "r") as f:
                        shared_id = f.read().strip()
                except Exception:
                    pass
        # Final fallback
        if shared_id is None:
            shared_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir = f"{output_dir}_{shared_id}"
        if accelerator.is_main_process:
            print(f"Output directory exists, using: {output_dir}")

    # Update training config with the potentially modified output_dir
    training_config = training_config.copy()
    training_config["output_dir"] = output_dir
    training_args = TrainingArguments(
        output_dir=training_config["output_dir"],
        overwrite_output_dir=training_config.get("overwrite_output_dir", True),
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        warmup_steps=training_config["warmup_steps"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.0),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "linear"),

        logging_steps=training_config["logging_steps"],
        save_steps=training_config["save_steps"],
        eval_steps=training_config["eval_steps"],
        eval_strategy=training_config["evaluation_strategy"],
        save_strategy=training_config["save_strategy"],
        save_total_limit=training_config.get("save_total_limit", None),

        load_best_model_at_end=training_config["load_best_model_at_end"],
        metric_for_best_model=training_config["metric_for_best_model"],
        greater_is_better=training_config["greater_is_better"],
        dataloader_drop_last=training_config["dataloader_drop_last"],
        remove_unused_columns=training_config["remove_unused_columns"],
        bf16=training_config["bf16"],
        fp16=training_config.get("fp16", False),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),

        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        report_to="wandb" if "wandb" in config else "none",
        run_name=config["name"],
        seed=config["seed"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        tokenizer=tokenizer,
    )

    print("Starting training...")
    if resume:
        if not checkpoint:
            print("--resume was set but --checkpoint not provided; starting fresh training.")
            trainer.train()
        else:
            print(f"Resuming from checkpoint: {checkpoint}")
            trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save the final model
    final_dir = os.path.join(training_args.output_dir, "final_model")
    trainer.save_model(final_dir)
    # Save tokenizer with special tokens so inference uses identical mapping
    tokenizer.save_pretrained(final_dir)
    print(f"Model and tokenizer saved to {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file", default='training/onestep-s.json')
    parser.add_argument("--resume", action="store_true", help="Resume training from a checkpoint path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint directory to resume from")
    parser.add_argument("--reset", action="store_true", help="Initialize model weights from a pretrained checkpoint for fine-tuning")

    args = parser.parse_args()

    main(config=args.config, checkpoint=args.checkpoint, resume=args.resume, reset=args.reset)
