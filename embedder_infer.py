import torch
from typing import Callable, Tuple, Dict, Any


def _to_device_dtype(t: torch.Tensor, ref: torch.nn.Module) -> torch.Tensor:
    emb = ref.get_input_embeddings()
    return t.to(device=emb.weight.device, dtype=emb.weight.dtype)


def make_onestep_inputs(base_model,
                        tokenizer,
                        prompt_text: str,
                        embedder=None,
                        X=None,
                        y=None,
                        device=None) -> Tuple[Dict[str, Any], Callable[[torch.Tensor], str]]:
    """Prepare inputs for generation. Returns (inputs_dict, decode_fn).

    - If embedder is None: tokenizes `prompt_text` and uses input_ids; decode_fn slices off prompt length.
    - If embedder is provided: builds prefix from (X, y) and concatenates with token embeddings; decode_fn decodes full ids.
    """
    if device is None:
        device = next(base_model.parameters()).device

    if embedder is None:
        tok = tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_len = tok["input_ids"].shape[1]

        def decode_fn(out_ids: torch.Tensor) -> str:
            gen_ids = out_ids[0, prompt_len:]
            return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        return tok, decode_fn

    # With embedder: build prefix from (X, y) and concatenate with token embeddings
    assert X is not None and y is not None, "X and y are required when using an input embedder"
    with torch.no_grad():
        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
        if X_t.dim() == 2:
            X_t = X_t.unsqueeze(0)
        if y_t.dim() == 1:
            y_t = y_t.unsqueeze(0)
        prefix = embedder(X_t, y_t)  # (B, P, H)
        tok = tokenizer(prompt_text, return_tensors="pt").to(device)
        tok_emb = base_model.get_input_embeddings()(tok["input_ids"])  # (B, T, H)
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)  # (B, P+T, H)
        attn_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

    def decode_fn_full(out_ids: torch.Tensor) -> str:
        # When using inputs_embeds, out_ids already contain only generated ids
        return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

    return {"inputs_embeds": inputs_embeds, "attention_mask": attn_mask}, decode_fn_full


def make_direct_inputs(base_model,
                       tokenizer,
                       embedder,
                       X,
                       y,
                       device=None) -> Tuple[Dict[str, Any], Callable[[torch.Tensor], str]]:
    """Prepare inputs for direct generation (prefix + BOS only). Returns (inputs_dict, decode_fn)."""
    if device is None:
        device = next(base_model.parameters()).device
    with torch.no_grad():
        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
        if X_t.dim() == 2:
            X_t = X_t.unsqueeze(0)
        if y_t.dim() == 1:
            y_t = y_t.unsqueeze(0)
        prefix = embedder(X_t, y_t)
        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        bos = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        bos_embed = base_model.get_input_embeddings()(bos)
        inputs_embeds = torch.cat([prefix, bos_embed], dim=1)
        attn_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

    def decode_fn(out_ids: torch.Tensor) -> str:
        return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

    return {"inputs_embeds": inputs_embeds, "attention_mask": attn_mask}, decode_fn

