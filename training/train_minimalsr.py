"""
Training script for MinimalSR transformer adapted from stream-of-search.
"""
import argparse
import json
import os
import random

import torch
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import GPTNeoConfig
from transformers import GPTNeoForCausalLM
from transformers import Trainer
from transformers import TrainingArguments

import wandb


def main(args):
    # read config from a json config file
    with open(args.config, "r") as f:
        config = json.load(f)

    # set seeds
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # set up accelerator
    accelerator = Accelerator()

    if args.wandb and accelerator.is_main_process:
        wandb_kwargs = config.get("wandb", {"project": "", "entity": "", "dir": ""})
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

    print(f"Number of parameters: {model.num_parameters()}")

    # load dataset
    train_file = os.path.join(config["data_dir"], config["train_file"])
    val_file = os.path.join(config["data_dir"], config["val_file"])
    val_target_file = os.path.join(config["data_dir"], config["val_target_file"])
    
    hf_datasets = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "val": val_file,
            "val_target": val_target_file,
        },
    )
    
    if config["num_train"] > 0:
        hf_datasets["train"] = hf_datasets["train"].select(range(min(int(config["num_train"]), len(hf_datasets["train"]))))

    context_length = config["context_length"]
    tokenizer.model_max_length = context_length

    def tokenize(element):
        if config["train_type"] == "sft":
            # Join all search path steps into one sequence
            text = [
                tokenizer.bos_token
                + " ".join(element["search_path"]).strip()
                + tokenizer.eos_token
            ]
        else:
            raise ValueError(f"Invalid train type: {config['train_type']}")
        
        outputs = tokenizer(
            text,
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
            stride=0,
            padding="max_length",
        )
        return {"input_ids": outputs["input_ids"]}

    # tokenize dataset for causal LM
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_datasets = hf_datasets.map(
        tokenize, batched=True, remove_columns=hf_datasets["train"].column_names
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    print("tokenized dataset", tokenized_datasets)

    # prepare training arguments from config
    training_config = config["training_args"]
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
        evaluation_strategy=training_config["evaluation_strategy"],
        save_strategy=training_config["save_strategy"],
        load_best_model_at_end=training_config["load_best_model_at_end"],
        metric_for_best_model=training_config["metric_for_best_model"],
        greater_is_better=training_config["greater_is_better"],
        dataloader_drop_last=training_config["dataloader_drop_last"],
        remove_unused_columns=training_config["remove_unused_columns"],
        bf16=training_config["bf16"],
        gradient_checkpointing=True,
        report_to="wandb" if args.wandb else "none",
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
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    args = parser.parse_args()
    
    main(args)