"""
Training script for one-step symbolic regression prediction.
Trains a transformer to predict the next generation given current population.
"""
import argparse
import json
import os
import random
from typing import Dict, Any

import torch
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import GPTNeoConfig
from transformers import GPTNeoForCausalLM
from transformers import Trainer
from transformers import TrainingArguments

import wandb
from format_utils import format_input_part, format_target_part
from datetime import datetime


def main(args):
    # read config from a json config file
    with open(args.config, "r") as f:
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

    with open(config["model_config"], "r") as f:
        model_config = json.load(f)

    # Create model
    model_config = GPTNeoConfig(**model_config)
    model = GPTNeoForCausalLM(model_config)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

    # Add special tokens for our format
    special_tokens = {
        "additional_special_tokens": [
            "<CONTEXT>", "<POPULATION>", "<FITNESS>", "<TARGET>"
        ]
    }
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    print(f"Number of parameters: {model.num_parameters()}")
    print(f"Added {num_added_tokens} special tokens")

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

    if config["num_train"] > 0:
        hf_datasets["train"] = hf_datasets["train"].select(
            range(min(int(config["num_train"]), len(hf_datasets["train"])))
        )

    context_length = config["context_length"]
    tokenizer.model_max_length = context_length
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(element):
        """
        Tokenize context/population/target triplets for supervised learning.
        Only trains on target tokens, not context/population.
        """
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

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

            # Pad to max length
            attention_mask = [1] * len(full_input_ids) + [0] * (context_length - len(full_input_ids))
            full_input_ids = full_input_ids + [tokenizer.pad_token_id] * (context_length - len(full_input_ids))

            # Create labels: -100 for input part, actual tokens for target part
            labels = [-100] * len(input_tokens["input_ids"])
            labels.extend(target_tokens["input_ids"])

            # Truncate labels if necessary
            if len(labels) > context_length:
                labels = labels[:context_length]

            # Pad labels
            labels = labels + [-100] * (context_length - len(labels))

            input_ids_list.append(full_input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": torch.tensor(input_ids_list),
            "attention_mask": torch.tensor(attention_mask_list),
            "labels": torch.tensor(labels_list),
        }

    # tokenize dataset
    tokenized_datasets = hf_datasets.map(
        tokenize,
        batched=True,
        remove_columns=hf_datasets["train"].column_names
    )

    # Use standard causal LM data collator
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    print("tokenized dataset", tokenized_datasets)

    # prepare training arguments from config
    training_config = config["training_args"]
    
    # Check if output directory exists and add datetime suffix if needed
    output_dir = training_config["output_dir"]
    if os.path.exists(output_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{timestamp}"
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
        warmup_steps=training_config["warmup_steps"],
        learning_rate=training_config["learning_rate"],
        logging_steps=training_config["logging_steps"],
        save_steps=training_config["save_steps"],
        eval_steps=training_config["eval_steps"],
        eval_strategy=training_config["evaluation_strategy"],
        save_strategy=training_config["save_strategy"],
        load_best_model_at_end=training_config["load_best_model_at_end"],
        metric_for_best_model=training_config["metric_for_best_model"],
        greater_is_better=training_config["greater_is_better"],
        dataloader_drop_last=training_config["dataloader_drop_last"],
        remove_unused_columns=training_config["remove_unused_columns"],
        bf16=training_config["bf16"],
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
    trainer.train()

    # Save the final model
    trainer.save_model(os.path.join(training_args.output_dir, "final_model"))
    print(f"Model saved to {training_args.output_dir}/final_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    main(args)
