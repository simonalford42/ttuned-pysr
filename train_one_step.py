"""
Training script for one-step symbolic regression prediction.
Trains a transformer to predict the next generation given current population.
"""
import argparse
import json
import os
import random
from typing import Dict, Any, Optional
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
from expression_parser import ExpressionParser


class ModelWithInputEmbedder(torch.nn.Module):
    """Wrapper that combines a language model with an input embedder.

    Expects token `input_ids` and optional `attention_mask`, along with
    point data tensors `points_X` and `points_y` to build a learned prefix.
    """
    def __init__(self, base_model, input_embedder):
        super().__init__()
        self.base_model = base_model
        self.input_embedder = input_embedder
        # Expose config for Trainer
        self.config = base_model.config

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        points_X=None,
        points_y=None,
        **kwargs,
    ):
        # Compute prefix embeddings on the correct device/dtype
        if self.input_embedder is None:
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

        # Resolve dtype/device from base model's embeddings
        tok_embed = self.base_model.get_input_embeddings()
        model_dtype = tok_embed.weight.dtype
        model_device = tok_embed.weight.device

        # Token embeddings for the sequence
        token_embeds = tok_embed(input_ids.to(model_device))

        # Points to dtype/device
        if points_X is None or points_y is None:
            raise ValueError("points_X and points_y must be provided when using input embedder")
        X = points_X.to(model_device, dtype=model_dtype)
        y = points_y.to(model_device, dtype=model_dtype)

        # Build prefix from embedder (batch, prefix_len, hidden)
        prefix_embeds = self.input_embedder(X, y)

        # Concatenate prefix + token embeddings
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        # Build attention mask if provided; otherwise assume all ones for tokens
        if attention_mask is None:
            attn_tokens = torch.ones(input_ids.shape, dtype=torch.long, device=model_device)
        else:
            attn_tokens = attention_mask.to(model_device)
        prefix_len = prefix_embeds.shape[1]

        # Adjust labels to match concatenated length by prepending -100 over prefix
        labels_full = None
        if labels is not None:
            labels_tokens = labels.to(model_device)
            ignore = torch.full((labels_tokens.shape[0], prefix_len), -100, dtype=labels_tokens.dtype, device=model_device)
            labels_full = torch.cat([ignore, labels_tokens], dim=1)
        else:
            labels_tokens = None

        # Prefix-LM attention masking: bidirectional over (prefix + input tokens), causal over target tokens
        B = token_embeds.shape[0]
        T_tokens = token_embeds.shape[1]
        T_total = prefix_len + T_tokens

        # Determine input length within tokens: positions that are real tokens (attn=1) and label-suppressed (-100)
        # Shape: (B,)
        if labels_tokens is not None:
            is_input_token = (labels_tokens == -100) & (attn_tokens == 1)
            input_lens = is_input_token.sum(dim=1)
        else:
            # Fallback: assume no input tokens (all target)
            input_lens = torch.zeros(B, dtype=torch.long, device=model_device)

        # Build a base causal lower-tri mask (allowed positions True)
        causal = torch.tril(torch.ones((T_total, T_total), dtype=torch.bool, device=model_device))
        allowed = causal.unsqueeze(0).unsqueeze(1).expand(B, 1, T_total, T_total).clone()

        # Promote the prefix+input block to full bidirectional for each batch
        # Compute per-batch boundary index in the concatenated sequence
        input_end_full = prefix_len + input_lens  # (B,)
        for b in range(B):
            end = int(input_end_full[b].item())
            if end > 0:
                allowed[b, :, :end, :end] = True

        # Mask out padded token keys entirely (they shouldn't be attended to)
        key_valid_tokens = attn_tokens.bool()  # (B, T_tokens)
        key_valid_prefix = torch.ones(B, prefix_len, dtype=torch.bool, device=model_device)
        key_valid = torch.cat([key_valid_prefix, key_valid_tokens], dim=1)  # (B, T_total)
        allowed = allowed & key_valid.view(B, 1, 1, T_total)

        # Convert to additive mask: 0 for allowed, -inf for disallowed
        neg_inf = torch.finfo(token_embeds.dtype).min if token_embeds.dtype.is_floating_point else -1e9
        extended_attention_mask = torch.zeros_like(allowed, dtype=token_embeds.dtype)
        extended_attention_mask = extended_attention_mask.masked_fill(~allowed, neg_inf)

        # Also keep a 2D mask in case the backend ignores extended masks (used by some models)
        attention_mask_full_2d = torch.cat([
            torch.ones(B, prefix_len, dtype=attn_tokens.dtype, device=model_device),
            attn_tokens
        ], dim=1)

        # Try passing the extended mask; some architectures (e.g., GPTNeo) accept 4D additive masks
        try:
            return self.base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=extended_attention_mask,
                labels=labels_full,
                **kwargs,
            )
        except Exception:
            # Fallback to 2D mask with default causal behavior if the model rejects 4D masks
            return self.base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask_full_2d,
                labels=labels_full,
                **kwargs,
            )

    def generate(self, **kwargs):
        return self.base_model.generate(**kwargs)

    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing on the base model."""
        return self.base_model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing on the base model."""
        return self.base_model.gradient_checkpointing_disable()


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

    # Reserve room for learned prefix if using embedder
    reserved_prefix = input_embedder.max_points if input_embedder is not None else 0

    # Training mode (onestep or direct)
    training_mode = config.get("training_mode", "onestep")
    print(f"Training mode: {training_mode}")
    if training_mode == "direct" and input_embedder is None:
        raise ValueError("Direct training mode requires an input_embedder (e.g., set 'input_embedder': 'e2e' in config)")

    def tokenize(element):
        """Tokenize examples for onestep or direct modes."""
        input_ids_list = []
        labels_list = []
        attention_masks = []
        expression_ids = []
        source_files = []
        X_data_list = []
        y_data_list = []

        if training_mode == "direct":
            for i in range(len(element["target"])):
                target = element["target"][i]
                input_part = tokenizer.bos_token or ""
                target_part = format_target_part(target, tokenizer.eos_token)
                input_tokens = tokenizer(input_part, add_special_tokens=False)
                target_tokens = tokenizer(target_part, add_special_tokens=False)
                full_input_ids = input_tokens["input_ids"] + target_tokens["input_ids"]
                effective_max_len = max(0, context_length - reserved_prefix)
                if len(full_input_ids) > effective_max_len:
                    full_input_ids = full_input_ids[:effective_max_len]
                attn_len = len(full_input_ids)
                attention_mask = [1] * attn_len + [0] * (effective_max_len - attn_len)
                full_input_ids = full_input_ids + [tokenizer.pad_token_id] * (effective_max_len - len(full_input_ids))
                labels = [-100] * len(input_tokens["input_ids"])
                labels.extend(target_tokens["input_ids"])
                if len(labels) > effective_max_len:
                    labels = labels[:effective_max_len]
                labels = labels + [-100] * (effective_max_len - len(labels))
                input_ids_list.append(full_input_ids)
                labels_list.append(labels)
                attention_masks.append(attention_mask)
                X_data_list.append(element["X_data"][i])
                y_data_list.append(element["y_data"][i])
            result = {
                "input_ids": torch.tensor(input_ids_list),
                "labels": torch.tensor(labels_list),
                "attention_mask": torch.tensor(attention_masks),
                "X_data": X_data_list,
                "y_data": y_data_list,
            }
        else:
            for i in range(len(element["context"])):
                context = element["context"][i]
                population = element["population"][i]
                target = element["target"][i]
                input_part = format_input_part(tokenizer.bos_token, context, population)
                target_part = format_target_part(target, tokenizer.eos_token)
                input_tokens = tokenizer(input_part, add_special_tokens=False)
                target_tokens = tokenizer(target_part, add_special_tokens=False)
                full_input_ids = input_tokens["input_ids"] + target_tokens["input_ids"]
                effective_max_len = max(0, context_length - reserved_prefix)
                if len(full_input_ids) > effective_max_len:
                    full_input_ids = full_input_ids[:effective_max_len]
                attn_len = len(full_input_ids)
                attention_mask = [1] * attn_len + [0] * (effective_max_len - attn_len)
                full_input_ids = full_input_ids + [tokenizer.pad_token_id] * (effective_max_len - len(full_input_ids))
                labels = [-100] * len(input_tokens["input_ids"])
                labels.extend(target_tokens["input_ids"])
                if len(labels) > effective_max_len:
                    labels = labels[:effective_max_len]
                labels = labels + [-100] * (effective_max_len - len(labels))
                input_ids_list.append(full_input_ids)
                labels_list.append(labels)
                attention_masks.append(attention_mask)
                if input_embedder is not None:
                    expression_ids.append(element["expression_id"][i] if "expression_id" in element else -1)
                    source_files.append(element["metadata"][i].get("source_expressions_file", "") if "metadata" in element else "")
            result = {
                "input_ids": torch.tensor(input_ids_list),
                "labels": torch.tensor(labels_list),
                "attention_mask": torch.tensor(attention_masks),
            }
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
        # Custom collator that loads per-example X/y and pads them, but leaves
        # all embedding computation to the model forward so gradients/device are correct.
        class InputEmbeddingCollator:
            def __init__(self, tokenizer, expressions_cache):
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
                batch = {
                    "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
                    "labels": torch.stack([torch.tensor(f["labels"]) for f in features]),
                    "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in features]),
                }

                X_batch = []
                y_batch = []

                # Direct mode: features carry X_data/y_data
                if "X_data" in features[0] and "y_data" in features[0]:
                    for f in features:
                        X_batch.append(torch.tensor(f["X_data"], dtype=torch.float32))
                        y_batch.append(torch.tensor(f["y_data"], dtype=torch.float32))
                # Onestep mode: lookup by expression_id
                elif "expression_id" in features[0]:
                    for f in features:
                        expr_id = f["expression_id"]
                        source_file = f["source_expressions_file"]
                        expressions_data = self.load_expressions_file(source_file)
                        X, y = expressions_data[expr_id]
                        X_batch.append(torch.from_numpy(X))
                        y_batch.append(torch.from_numpy(y))

                if X_batch:
                    max_points = max(X.shape[0] for X in X_batch)
                    max_vars = max(X.shape[1] for X in X_batch)
                    X_padded = torch.zeros(len(X_batch), max_points, max_vars, dtype=torch.float32)
                    y_padded = torch.zeros(len(y_batch), max_points, dtype=torch.float32)
                    for i, (X, y) in enumerate(zip(X_batch, y_batch)):
                        n_points, n_vars = X.shape
                        X_padded[i, :n_points, :n_vars] = X
                        y_padded[i, :n_points] = y
                    batch["points_X"] = X_padded
                    batch["points_y"] = y_padded

                return batch

        data_collator = InputEmbeddingCollator(tokenizer, expressions_data_cache)
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
        # Avoid safetensors shared-storage error with tied weights (e.g., lm_head <-> wte)
        # Use standard PyTorch .bin checkpointing instead.
        save_safetensors=False,

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

    # Inject direct-mode evaluation metrics via a custom Trainer subclass
    expr_parser = ExpressionParser()

    def _translate_e2e_expr(expr_text: str) -> str:
        s = expr_text
        s = s.replace(' add ', ' + ').replace(' sub ', ' - ').replace(' mul ', ' * ').replace(' div ', ' / ')
        s = s.replace(' add', ' +').replace(' sub', ' -').replace(' mul', ' *').replace(' div', ' /')
        s = s.replace('add ', '+ ').replace('sub ', '- ').replace('mul ', '* ').replace('div ', '/ ')
        s = s.replace('x_', 'x')
        return s

    def _safe_parse(expr_text: str):
        try:
            return expr_parser.parse(_translate_e2e_expr(expr_text))
        except Exception:
            return None

    def _r2_and_acc(y_true: np.ndarray, y_pred: np.ndarray, tol_mse: float = 1e-12):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        if y_true.shape != y_pred.shape or y_true.size == 0:
            return 0.0, 0.0, np.inf
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(mask):
            return 0.0, 0.0, np.inf
        yt = y_true[mask]
        yp = y_pred[mask]
        mse = float(np.mean((yt - yp) ** 2))
        denom = float(np.sum((yt - yt.mean()) ** 2))
        if denom <= 1e-20:
            r2 = 1.0 if mse < tol_mse else 0.0
        else:
            r2 = 1.0 - float(np.sum((yt - yp) ** 2)) / denom
        acc = 1.0 if mse < tol_mse else 0.0
        return r2, acc, mse

    class DirectTrainer(Trainer):
        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
            metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
            if training_mode != "direct":
                return metrics
            ds = eval_dataset if eval_dataset is not None else self.eval_dataset
            if ds is None or len(ds) == 0:
                return metrics
            device = self.model.base_model.device if hasattr(self.model, "base_model") else next(self.model.parameters()).device
            tok = self.tokenizer
            n = min(int(config.get("direct_eval_max_examples", 128)), len(ds))
            indices = list(range(n))
            r2s = []
            accs = []
            mses = []
            for idx in indices:
                ex = ds[idx]
                try:
                    X = torch.tensor([ex["X_data"]], dtype=torch.float32, device=device)
                    y = torch.tensor([ex["y_data"]], dtype=torch.float32, device=device)
                except KeyError:
                    continue
                with torch.no_grad():
                    prefix = self.model.input_embedder(X, y)
                    bos_id = torch.tensor([[tok.bos_token_id]] if tok.bos_token_id is not None else [[tok.eos_token_id]],
                                          dtype=torch.long, device=device)
                    tok_embed = self.model.base_model.get_input_embeddings()
                    bos_embed = tok_embed(bos_id)
                    inputs_embeds = torch.cat([prefix, bos_embed], dim=1)
                    attn_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
                    out_ids = self.model.base_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attn_mask,
                        max_new_tokens=96,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                    )
                    pred_text = tok.decode(out_ids[0], skip_special_tokens=True).strip()
                node = _safe_parse(pred_text)
                if node is None:
                    r2, acc, mse = 0.0, 0.0, float("inf")
                else:
                    X_np = X.squeeze(0).detach().cpu().numpy()
                    y_true = y.squeeze(0).detach().cpu().numpy()
                    y_pred = node.evaluate(X_np)
                    r2, acc, mse = _r2_and_acc(y_true, y_pred)
                r2s.append(r2)
                accs.append(acc)
                mses.append(mse)
            if r2s:
                logs = {
                    "eval_direct_r2_mean": float(np.mean(r2s)),
                    "eval_direct_acc_tol": float(np.mean(accs)),
                    "eval_direct_mse_mean": float(np.mean(mses)),
                    "eval_samples": len(r2s),
                }
                self.log(logs)
                metrics.update(logs)

            # Ensure eval_loss exists for metric_for_best_model logic, even when drop_last removes all batches
            if "eval_loss" not in metrics and len(ds) > 0:
                try:
                    # Build a small batch manually via the collator to avoid DataLoader drop_last
                    take = min(8, len(ds))
                    examples = [ds[i] for i in range(take)]
                    batch = self.data_collator(examples)
                    batch = self._prepare_inputs(batch)
                    with torch.no_grad():
                        loss = self.compute_loss(self.model, batch)
                        if isinstance(loss, tuple):
                            loss = loss[0]
                        loss_val = float(loss.detach().cpu().item())
                    metrics["eval_loss"] = loss_val
                    self.log({"eval_loss": loss_val})
                except Exception:
                    # If anything goes wrong, leave eval_loss absent
                    pass
            return metrics

    trainer = DirectTrainer(
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

    # Save the final model (base model) and tokenizer; include embedder weights if present
    final_dir = os.path.join(training_args.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    if hasattr(model, "base_model") and hasattr(model.base_model, "save_pretrained"):
        # Also disable safetensors for final save to handle tied weights safely
        model.base_model.save_pretrained(final_dir, safe_serialization=False)
    else:
        trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    if input_embedder is not None:
        torch.save(input_embedder.state_dict(), os.path.join(final_dir, "input_embedder.pt"))
    print(f"Model and tokenizer saved to {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file", default='training/onestep-s.json')
    parser.add_argument("--resume", action="store_true", help="Resume training from a checkpoint path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint directory to resume from")
    parser.add_argument("--reset", action="store_true", help="Initialize model weights from a pretrained checkpoint for fine-tuning")

    args = parser.parse_args()

    main(config=args.config, checkpoint=args.checkpoint, resume=args.resume, reset=args.reset)
