"""
Evaluate direct prediction model by generating expressions, parsing them,
and evaluating on X to compare against true y.

Reports:
  - R^2 (mean over examples)
  - Accuracy (fraction with MSE < tolerance)
  - Mean MSE
"""
import argparse
import json
import torch
import numpy as np
from transformers import AutoTokenizer
from utils import load_jsonl
from modules import WithEmbedderForCausalLM
from tqdm import tqdm
from expression_parser import ExpressionParser, safe_parse_e2e, translate_e2e_expr


def r2_acc_mse(y_true: np.ndarray, y_pred: np.ndarray, tol_mse: float = 1e-12):
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


def evaluate_accuracy(model, embedder, tokenizer, val_data, device='cuda', num_samples=None, tol_mse: float = 1e-12):
    """Evaluate R^2 and within-tolerance accuracy on validation data."""
    model = model.to(device)
    model.eval()
    # Ensure embedder is on the same device as model
    if embedder is not None:
        embedder = embedder.to(device)
        embedder.eval()
    parser = ExpressionParser()



    r2s, accs, mses = [], [], []

    if num_samples is not None:
        val_data = val_data[:num_samples]

    with torch.no_grad():
        # for example in tqdm(val_data, desc="Evaluating"):
        for example in val_data:
            print(translate_e2e_expr(example['target']))
            # Prepare input embeddings
            X = torch.tensor([example["X_data"]], dtype=torch.float32, device=device)
            y = torch.tensor([example["y_data"]], dtype=torch.float32, device=device)

            # Get input embeddings
            prefix_embeds = embedder(X, y)

            # BOS token embedding
            bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
            bos_id = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
            tok_embed = model.base_model.get_input_embeddings()
            bos_embed = tok_embed(bos_id)

            # Concatenate prefix + bos and simple attention mask
            inputs_embeds = torch.cat([prefix_embeds, bos_embed], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

            # Generate
            try:
                outputs = model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=100,
                    do_sample=False,  # Greedy decoding for evaluation
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                # Decode and parse
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                # print(generated_text)
                node = safe_parse_e2e(parser, generated_text)
                print(str(node))
                if node is None:
                    r2, acc, mse = 0.0, 0.0, float('inf')
                else:
                    X_np = np.asarray(example["X_data"], dtype=np.float32)
                    y_np = np.asarray(example["y_data"], dtype=np.float32)
                    y_pred = node.evaluate(X_np)
                    r2, acc, mse = r2_acc_mse(y_np, y_pred)

                print(r2)
                print()
                r2s.append(r2)
                accs.append(acc)
                mses.append(mse)

            except Exception as e:
                print(f"Generation/parsing error: {e}")
                r2s.append(0.0)
                accs.append(0.0)
                mses.append(float('inf'))

    r2_mean = float(np.mean(r2s)) if r2s else 0.0
    acc_mean = float(np.mean(accs)) if accs else 0.0
    mse_mean = float(np.mean(mses)) if mses else float('inf')
    return r2_mean, acc_mean, mse_mean, len(r2s)


def main():
    parser = argparse.ArgumentParser(description="Evaluate direct prediction model")
    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--val_file", "-v", type=str, default='datasets/training/arith_200_c05_20251103_213559_direct.jsonl', help="Path to validation JSONL file")
    parser.add_argument("--num_samples", "-n", type=int, default=20, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="Device to use for evaluation")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = WithEmbedderForCausalLM.from_pretrained(args.checkpoint, trust_remote_code=True)
    embedder = model.input_embedder

    print(f"Loading validation data from {args.val_file}...")
    val_data = load_jsonl(args.val_file)
    print(f"Loaded {len(val_data)} validation examples")

    print(f"\nEvaluating accuracy...")
    r2_mean, acc_mean, mse_mean, n = evaluate_accuracy(
        model, embedder, tokenizer, val_data,
        device=args.device, num_samples=args.num_samples
    )

    print(f"\n{'='*60}")
    print("Results:")
    print(f"  R^2 (mean): {r2_mean:.6f}")
    print(f"  Acc (MSE < tol): {acc_mean:.2%}")
    print(f"  MSE (mean): {mse_mean:.6e}")
    print(f"  Samples: {n}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
