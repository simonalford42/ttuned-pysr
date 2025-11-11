import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig


class E2EPointEmbedder(nn.Module):
    """
    Simple end-to-end point embedder for symbolic regression.

    Takes X, y data and embeds them into fixed-size vectors using a simple 2-layer MLP.
    This is a simpler alternative to PointEmbedder for baseline experiments.

    Args:
        max_points: Maximum number of data points (default: 64)
        max_input_dim: Maximum input dimension (default: 10)
        hidden_size: Output embedding size (should match model hidden size)
        mlp_hidden_size: Hidden layer size for the MLP (default: 512)
    """
    def __init__(
        self,
        max_points: int = 64,
        max_input_dim: int = 10,
        hidden_size: int = 512,
        mlp_hidden_size: int = 512,
    ):
        super().__init__()
        self.max_points = max_points
        self.max_input_dim = max_input_dim
        self.hidden_size = hidden_size
        self.mlp_hidden_size = mlp_hidden_size
        # Backward-compat reserved prefix length equals max_points
        self.prefix_len = max_points

        # Input size: each point has (max_input_dim + 1) values (X + y)
        input_size = max_input_dim + 1

        # 2-layer MLP with ReLU
        self.mlp = nn.Sequential(
            nn.Linear(input_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, hidden_size)
        )

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Embed (X, y) data points into prefix vectors.

        Args:
            X: Input features of shape (batch_size, num_points, num_vars)
            y: Target values of shape (batch_size, num_points)

        Returns:
            embeddings: (batch_size, max_points, hidden_size)
        """
        batch_size, num_points, num_vars = X.shape

        # Pad/truncate X to max_input_dim
        if num_vars < self.max_input_dim:
            # Pad with zeros
            padding = torch.zeros(
                batch_size, num_points, self.max_input_dim - num_vars,
                device=X.device, dtype=X.dtype
            )
            X_padded = torch.cat([X, padding], dim=-1)
        elif num_vars > self.max_input_dim:
            # Truncate
            X_padded = X[:, :, :self.max_input_dim]
        else:
            X_padded = X

        # Concatenate X and y: (batch_size, num_points, max_input_dim + 1)
        y_expanded = y.unsqueeze(-1)  # (batch_size, num_points, 1)
        points = torch.cat([X_padded, y_expanded], dim=-1)

        # Pad/truncate to max_points
        if num_points < self.max_points:
            # Pad with zeros
            padding = torch.zeros(
                batch_size, self.max_points - num_points, self.max_input_dim + 1,
                device=points.device, dtype=points.dtype
            )
            points = torch.cat([points, padding], dim=1)
        elif num_points > self.max_points:
            # Truncate
            points = points[:, :self.max_points, :]

        # Apply MLP to each point: (batch_size, max_points, hidden_size)
        embeddings = self.mlp(points)

        return embeddings


def test_e2e_point_embedder():
    """Test the E2E point embedder."""
    print("Testing E2EPointEmbedder...")

    # Create embedder
    embedder = E2EPointEmbedder(
        max_points=64,
        max_input_dim=10,
        hidden_size=512,
        mlp_hidden_size=512,
    )

    # Test with some data
    batch_size = 2
    num_points = 50
    num_vars = 3

    X = torch.randn(batch_size, num_points, num_vars)
    y = torch.randn(batch_size, num_points)

    # Embed
    embeddings = embedder(X, y)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Expected output shape: ({batch_size}, 64, 512)")

    assert embeddings.shape == (batch_size, 64, 512), "Output shape mismatch!"
    print("✓ Shape test passed")

    # Test with different input sizes
    X_small = torch.randn(batch_size, 10, 2)
    y_small = torch.randn(batch_size, 10)
    embeddings_small = embedder(X_small, y_small)
    assert embeddings_small.shape == (batch_size, 64, 512)
    print("✓ Small input test passed")

    # Test with max points
    X_large = torch.randn(batch_size, 100, 5)
    y_large = torch.randn(batch_size, 100)
    embeddings_large = embedder(X_large, y_large)
    assert embeddings_large.shape == (batch_size, 64, 512)
    print("✓ Large input test (truncation) passed")

    print("\nAll E2E tests passed! ✓")

    # Print model size
    num_params = sum(p.numel() for p in embedder.parameters())
    print(f"\nNumber of parameters: {num_params:,}")


if __name__ == "__main__":
    test_e2e_point_embedder()


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
        # Provide a minimal compatibility surface expected by HF Trainer
        # Delegate ignore lists to the underlying PreTrainedModel when available
        try:
            self._keys_to_ignore_on_save = getattr(base_model, "_keys_to_ignore_on_save", [])
        except Exception:
            self._keys_to_ignore_on_save = []

    def __getattr__(self, name):
        """Delegate missing attributes to the underlying base_model.

        Use nn.Module.__getattr__ first to preserve parameter/buffer/submodule
        lookup. If still missing, forward to `base_model` for transformers-specific
        attributes Trainer expects.
        """
        try:
            # Preserve nn.Module behavior (finds submodules/parameters/buffers)
            return super().__getattr__(name)
        except AttributeError as e:
            # Try to delegate to base_model; if not yet set, re-raise original
            try:
                base = super().__getattr__("base_model")
            except AttributeError:
                raise e
            return getattr(base, name)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load state dicts flexibly.

        Supports three layouts:
        - Wrapper-format with keys prefixed by 'base_model.' and/or 'input_embedder.'
        - Plain base-model state dict without prefixes (as saved by save_pretrained)
        In the second case, remap keys to 'base_model.<key>' so super().load_state_dict
        loads directly into the wrapped base model.
        """
        # Fast path: if it already looks like a wrapper sd, defer to default
        has_wrapper_keys = any(k.startswith("wrapped.") or k.startswith("input_embedder.") for k in state_dict.keys())
        if not has_wrapper_keys:
            # Heuristically detect base-model-only sd by comparing a subset of keys
            base_keys = set(self.base_model.state_dict().keys())
            sample = 0
            match = 0
            for k in state_dict.keys():
                sample += 1
                if k in base_keys:
                    match += 1
                if sample >= 16:  # small sample to decide
                    break
            if match > 0:
                remapped = {f"base_model.{k}": v for k, v in state_dict.items()}
                state_dict = remapped
                # Strict=False helps tolerate extra missing embedder keys when not present
                strict = False
        return super().load_state_dict(state_dict, strict=strict)


class WithEmbedderForCausalLM(PreTrainedModel):
    """PreTrainedModel wrapper that combines a causal LM with an input embedder.

    This class follows HF expectations so Trainer/Accelerate/DDP interact cleanly.
    It delegates embeddings, resize, and forward to the underlying base model,
    while injecting a learned prefix computed from (points_X, points_y).
    """

    # So PreTrainedModel.base_model property returns `self.wrapped`
    base_model_prefix = "wrapped"
    # Let HF know how to load config for this class
    config_class = AutoConfig

    def __init__(self, config, base_model: torch.nn.Module | None = None, input_embedder: torch.nn.Module | None = None):
        # `config` is a PretrainedConfig (reuse base model config)
        super().__init__(config)
        # Underlying LM
        if base_model is None:
            self.wrapped = AutoModelForCausalLM.from_config(config)
        else:
            self.wrapped = base_model
        # Input embedder
        if input_embedder is not None:
            self.input_embedder = input_embedder
        else:
            # Try to instantiate from config metadata if present
            cls = getattr(config, "embedder_class", None)
            hidden = getattr(config, "hidden_size", getattr(config, "n_embd", 512))
            if cls == "SetPointEmbedder":
                self.input_embedder = SetPointEmbedder(
                    max_input_dim=getattr(config, "embedder_max_input_dim", 10),
                    hidden_size=hidden,
                    d_model=getattr(config, "embedder_d_model", hidden // 2),
                    num_layers=getattr(config, "embedder_num_layers", 2),
                    num_heads=getattr(config, "embedder_num_heads", 8),
                    prefix_len=getattr(config, "embedder_prefix_len", 16),
                    fourier_features=getattr(config, "embedder_fourier_features", 0),
                    normalize=getattr(config, "embedder_normalize", True),
                    append_stats=getattr(config, "embedder_append_stats", False),
                    norm_center_only=getattr(config, "embedder_norm_center_only", False),
                    norm_scale_only=getattr(config, "embedder_norm_scale_only", False),
                )
            elif cls == "E2EPointEmbedder":
                self.input_embedder = E2EPointEmbedder(
                    max_points=getattr(config, "embedder_max_points", 64),
                    max_input_dim=getattr(config, "embedder_max_input_dim", 10),
                    hidden_size=hidden,
                    mlp_hidden_size=getattr(config, "embedder_mlp_hidden_size", 512),
                )
            elif cls == "FeatureSetEmbedder":
                self.input_embedder = FeatureSetEmbedder(
                    max_input_dim=getattr(config, "embedder_max_input_dim", 10),
                    hidden_size=hidden,
                    d_model=getattr(config, "embedder_d_model", hidden // 2),
                    num_layers=getattr(config, "embedder_num_layers", 2),
                    num_heads=getattr(config, "embedder_num_heads", 8),
                    prefix_len=getattr(config, "embedder_prefix_len", 16),
                    use_raw_xy=getattr(config, "embedder_use_raw_xy", True),
                    use_x_phases=getattr(config, "embedder_use_x_phases", False),
                    use_logx_phases=getattr(config, "embedder_use_logx_phases", False),
                    fourier_x_num=getattr(config, "embedder_fourier_x_num", 32),
                    fourier_logx_num=getattr(config, "embedder_fourier_logx_num", 32),
                    log_eps=getattr(config, "embedder_log_eps", 1e-6),
                    include_poly=getattr(config, "embedder_include_poly", False),
                    pool_type=getattr(config, "embedder_pool_type", "encoder"),
                    normalize=getattr(config, "embedder_normalize", False),
                )
            else:
                self.input_embedder = None
        # Align pad/eos tokens if needed
        if getattr(self.wrapped, "config", None) is not None:
            self.config.pad_token_id = getattr(self.wrapped.config, "pad_token_id", getattr(self.config, "pad_token_id", None))
            self.config.eos_token_id = getattr(self.wrapped.config, "eos_token_id", getattr(self.config, "eos_token_id", None))

    # Required by PreTrainedModel
    def get_input_embeddings(self):
        return self.wrapped.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.wrapped.set_input_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens: int):
        return self.wrapped.resize_token_embeddings(new_num_tokens)

    def tie_weights(self):
        if hasattr(self.wrapped, "tie_weights"):
            return self.wrapped.tie_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        points_X=None,
        points_y=None,
        points_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # If no embedder, pass through
        if self.input_embedder is None and inputs_embeds is None:
            return self.wrapped(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        # Resolve dtype/device from base model's token embeddings
        tok_embed = self.wrapped.get_input_embeddings()
        model_dtype = tok_embed.weight.dtype
        model_device = tok_embed.weight.device

        # Build token embeddings if not provided
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
            token_embeds = tok_embed(input_ids.to(model_device))
        else:
            token_embeds = inputs_embeds.to(model_device)

        prefix_len = 0
        if self.input_embedder is not None:
            if points_X is None or points_y is None:
                raise ValueError("points_X and points_y must be provided when using input embedder")
            X = points_X.to(model_device, dtype=model_dtype)
            y = points_y.to(model_device, dtype=model_dtype)
            try:
                prefix_embeds = self.input_embedder(X, y, points_mask=points_mask)
            except TypeError:
                prefix_embeds = self.input_embedder(X, y)
            inputs_embeds_full = torch.cat([prefix_embeds, token_embeds], dim=1)
            prefix_len = prefix_embeds.shape[1]
        else:
            inputs_embeds_full = token_embeds

        # Build/extend attention mask
        if attention_mask is None:
            attn_tokens = torch.ones(token_embeds.shape[:2], dtype=torch.long, device=model_device)
        else:
            attn_tokens = attention_mask.to(model_device)

        if prefix_len > 0:
            attn_prefix = torch.ones(attn_tokens.shape[0], prefix_len, dtype=attn_tokens.dtype, device=model_device)
            attention_mask_full = torch.cat([attn_prefix, attn_tokens], dim=1)
        else:
            attention_mask_full = attn_tokens

        # Adjust labels by prepending ignore tokens for prefix
        if labels is not None and prefix_len > 0:
            labels_tokens = labels.to(model_device)
            ignore = torch.full((labels_tokens.shape[0], prefix_len), -100, dtype=labels_tokens.dtype, device=model_device)
            labels_full = torch.cat([ignore, labels_tokens], dim=1)
        else:
            labels_full = labels

        return self.wrapped(
            inputs_embeds=inputs_embeds_full,
            attention_mask=attention_mask_full,
            labels=labels_full,
            **kwargs,
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        """Allow loading either full-wrapper weights or plain base-model weights.

        If the checkpoint only contains base-model keys (no 'base_model.' prefix),
        remap them to 'base_model.<key>' so loading into the wrapper succeeds.
        """
        has_wrapper_keys = any(k.startswith("base_model.") or k.startswith("input_embedder.") for k in state_dict.keys())
        if not has_wrapper_keys:
            # Heuristic: treat as base-only
            base_keys = set(self.wrapped.state_dict().keys())
            sample = 0
            match = 0
            for k in state_dict.keys():
                sample += 1
                if k in base_keys:
                    match += 1
                if sample >= 16:
                    break
            if match > 0:
                state_dict = {f"wrapped.{k}": v for k, v in state_dict.items()}
                strict = False
        return super().load_state_dict(state_dict, strict=strict)

    def update_embedder_config(self):
        """Persist embedder metadata into config so save_pretrained stores it."""
        if self.input_embedder is None:
            return
        cls = type(self.input_embedder).__name__
        setattr(self.config, "embedder_class", cls)
        if hasattr(self.input_embedder, "max_points"):
            setattr(self.config, "embedder_max_points", int(self.input_embedder.max_points))
        if hasattr(self.input_embedder, "max_input_dim"):
            setattr(self.config, "embedder_max_input_dim", int(self.input_embedder.max_input_dim))
        if hasattr(self.input_embedder, "mlp_hidden_size"):
            setattr(self.config, "embedder_mlp_hidden_size", int(self.input_embedder.mlp_hidden_size))
        if hasattr(self.input_embedder, "d_model"):
            setattr(self.config, "embedder_d_model", int(self.input_embedder.d_model))
        if hasattr(self.input_embedder, "num_layers"):
            setattr(self.config, "embedder_num_layers", int(self.input_embedder.num_layers))
        if hasattr(self.input_embedder, "num_heads"):
            setattr(self.config, "embedder_num_heads", int(self.input_embedder.num_heads))
        if hasattr(self.input_embedder, "prefix_len"):
            setattr(self.config, "embedder_prefix_len", int(self.input_embedder.prefix_len))
        if hasattr(self.input_embedder, "fourier_features"):
            setattr(self.config, "embedder_fourier_features", int(self.input_embedder.fourier_features))
        if hasattr(self.input_embedder, "normalize"):
            setattr(self.config, "embedder_normalize", bool(self.input_embedder.normalize))
        # FeatureSetEmbedder-specific
        if hasattr(self.input_embedder, "use_raw_xy"):
            setattr(self.config, "embedder_use_raw_xy", bool(self.input_embedder.use_raw_xy))
        if hasattr(self.input_embedder, "use_x_phases"):
            setattr(self.config, "embedder_use_x_phases", bool(self.input_embedder.use_x_phases))
        if hasattr(self.input_embedder, "use_logx_phases"):
            setattr(self.config, "embedder_use_logx_phases", bool(self.input_embedder.use_logx_phases))
        if hasattr(self.input_embedder, "fourier_x_num"):
            setattr(self.config, "embedder_fourier_x_num", int(self.input_embedder.fourier_x_num))
        if hasattr(self.input_embedder, "fourier_logx_num"):
            setattr(self.config, "embedder_fourier_logx_num", int(self.input_embedder.fourier_logx_num))
        if hasattr(self.input_embedder, "log_eps"):
            setattr(self.config, "embedder_log_eps", float(self.input_embedder.log_eps))
        if hasattr(self.input_embedder, "include_poly"):
            setattr(self.config, "embedder_include_poly", bool(self.input_embedder.include_poly))
        if hasattr(self.input_embedder, "pool_type"):
            setattr(self.config, "embedder_pool_type", str(self.input_embedder.pool_type))
        if hasattr(self.input_embedder, "append_stats"):
            setattr(self.config, "embedder_append_stats", bool(self.input_embedder.append_stats))
        if hasattr(self.input_embedder, "norm_center_only"):
            setattr(self.config, "embedder_norm_center_only", bool(self.input_embedder.norm_center_only))
        if hasattr(self.input_embedder, "norm_scale_only"):
            setattr(self.config, "embedder_norm_scale_only", bool(self.input_embedder.norm_scale_only))

    def generate(self, **kwargs):
        return self.wrapped.generate(**kwargs)

    def gradient_checkpointing_enable(self, **kwargs):
        return self.wrapped.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        return self.wrapped.gradient_checkpointing_disable()


class SetPointEmbedder(nn.Module):
    """
    Permutation-invariant set encoder for (X, y) points that produces a small
    number of summary prefix tokens via cross-attention (Set/Perceiver style).

    - Per-point MLP to d_model
    - Transformer encoder over points with padding mask support
    - Learnable queries attend to point encodings to yield `prefix_len` tokens

    Args:
        max_input_dim: Maximum variable count supported
        hidden_size: Output embedding size (must match LM hidden size)
        d_model: Internal model width for set encoder
        num_layers: TransformerEncoder layers over points
        num_heads: Attention heads
        prefix_len: Number of summary tokens to emit (reserved prefix length)
        fourier_features: If >0, append random Fourier features of inputs
        normalize: Whether to normalize X and y per-sample (zero mean, unit var)
    """
    def __init__(
        self,
        max_input_dim: int = 10,
        hidden_size: int = 512,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        prefix_len: int = 16,
        fourier_features: int = 0,
        normalize: bool = True,
        append_stats: bool = False,
        norm_center_only: bool = False,
        norm_scale_only: bool = False,
    ):
        super().__init__()
        self.max_input_dim = max_input_dim
        self.hidden_size = hidden_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prefix_len = prefix_len
        self.fourier_features = fourier_features
        self.normalize = normalize
        self.append_stats = append_stats
        self.norm_center_only = norm_center_only
        self.norm_scale_only = norm_scale_only

        input_size = max_input_dim + 1  # X vars + y
        feat_size = input_size
        if self.append_stats:
            feat_size += 2 * input_size
        if fourier_features and fourier_features > 0:
            # Cos/Sin per input dim (D dims × F features per dim)
            feat_size = feat_size + 2 * input_size * fourier_features
            self.register_buffer(
                "fourier_B",
                torch.randn(input_size, fourier_features) * 3.0,
                persistent=False,
            )

        self.point_proj = nn.Sequential(
            nn.Linear(feat_size, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, batch_first=True, norm_first=True,
            dim_feedforward=d_model * 4, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.query = nn.Parameter(torch.randn(prefix_len, d_model) / (d_model ** 0.5))
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(d_model, hidden_size)
        self.ln_out = nn.LayerNorm(hidden_size)

    def _build_features(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # X: (B, N, V), y: (B, N)
        B, N, V = X.shape
        # Pad/truncate X to max_input_dim
        if V < self.max_input_dim:
            pad = torch.zeros(B, N, self.max_input_dim - V, device=X.device, dtype=X.dtype)
            Xp = torch.cat([X, pad], dim=-1)
        else:
            Xp = X[:, :, : self.max_input_dim]

        yp = y.unsqueeze(-1)
        pts = torch.cat([Xp, yp], dim=-1)  # (B, N, D)

        # Per-sample stats on original scale
        orig_mean = pts.mean(dim=1, keepdim=True)
        orig_std = pts.std(dim=1, keepdim=True).clamp_min(1e-6)

        if self.normalize:
            if self.norm_center_only and not self.norm_scale_only:
                pts = pts - orig_mean
            elif self.norm_scale_only and not self.norm_center_only:
                pts = pts / orig_std
            else:
                pts = (pts - orig_mean) / orig_std

        if self.append_stats:
            stats = torch.cat([orig_mean, orig_std], dim=-1)  # (B,1,2D)
            stats = stats.expand(-1, N, -1)
            pts = torch.cat([pts, stats], dim=-1)

        if self.fourier_features and self.fourier_features > 0:
            # Apply random Fourier features per-dimension
            # Compute Fourier features on original dims (first input_size channels)
            base = pts[:, :, : self.max_input_dim + 1]
            proj = base.unsqueeze(-1) * self.fourier_B.unsqueeze(0).unsqueeze(0)
            sin = torch.sin(proj)
            cos = torch.cos(proj)
            # Flatten last two dims: (B, N, D*F)
            sin = sin.reshape(pts.shape[0], pts.shape[1], -1)
            cos = cos.reshape(pts.shape[0], pts.shape[1], -1)
            pts = torch.cat([pts, sin, cos], dim=-1)

        return pts

    def forward(self, X: torch.Tensor, y: torch.Tensor, points_mask: torch.Tensor = None) -> torch.Tensor:
        # Build per-point features
        B, N, _ = X.shape
        feats = self._build_features(X, y)  # (B, N, F)
        h = self.point_proj(feats)  # (B, N, d_model)

        # Build key padding mask: True masks PAD positions
        key_padding_mask = None
        if points_mask is not None:
            # points_mask: (B, N) with 1 for valid, 0 for pad
            key_padding_mask = ~(points_mask.to(torch.bool))

        # Encode with TransformerEncoder
        h_enc = self.encoder(h, src_key_padding_mask=key_padding_mask)  # (B, N, d_model)

        # Cross-attention from learnable queries to encoded points
        q = self.query.unsqueeze(0).expand(B, -1, -1)  # (B, P, d_model)
        # MultiheadAttention expects (B, S, E) with batch_first=True
        attn_out, _ = self.cross_attn(
            query=q,
            key=h_enc,
            value=h_enc,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )  # (B, P, d_model)

        out = self.out_proj(attn_out)
        out = self.ln_out(out)
        return out  # (B, prefix_len, hidden_size)


class FeatureSetEmbedder(nn.Module):
    """
    Modular point-embedder with pluggable feature sets and pooling mechanisms.

    Supports three feature modes:
      - raw: per-point concat of padded X and y
      - phases: sin/cos features on x and on log(|x|+eps)
      - combined: raw + phases

    Supports three pooling types from set of points to `prefix_len` tokens:
      - encoder: per-point MLP -> TransformerEncoder -> cross-attn queries (like SetPointEmbedder)
      - dsum: per-point MLP -> masked sum -> broadcast to K tokens with learned token embeddings
      - xattn: learned K queries attend once to per-point features (no self-attn)

    Notes:
      - No normalization by default (preserve scale)
      - Deterministic, log-spaced frequency banks for x and log|x|
      - Optional polynomial block can be appended for quick sanity tests
    """
    def __init__(
        self,
        max_input_dim: int = 10,
        hidden_size: int = 512,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        prefix_len: int = 16,
        # Feature controls
        use_raw_xy: bool = True,
        use_x_phases: bool = False,
        use_logx_phases: bool = False,
        fourier_x_num: int = 32,
        fourier_logx_num: int = 32,
        log_eps: float = 1e-6,
        include_poly: bool = False,  # optional x^2, y^2, xy block
        # Pooling control
        pool_type: str = "encoder",  # one of {"encoder","dsum","xattn"}
        # Misc
        normalize: bool = False,
    ):
        super().__init__()
        self.max_input_dim = max_input_dim
        self.hidden_size = hidden_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prefix_len = prefix_len

        self.use_raw_xy = use_raw_xy
        self.use_x_phases = use_x_phases
        self.use_logx_phases = use_logx_phases
        self.fourier_x_num = fourier_x_num
        self.fourier_logx_num = fourier_logx_num
        self.log_eps = log_eps
        self.include_poly = include_poly
        self.pool_type = pool_type
        self.normalize = normalize

        # Build feature size and any frequency buffers
        feat_size = 0
        input_size = max_input_dim + 1  # X vars + y

        if self.use_raw_xy:
            feat_size += input_size

        if self.include_poly:
            # x^2, y^2, xy (xy computed from first x-dim) — minimal poly sanity toggle
            feat_size += 3

        if self.use_x_phases and self.fourier_x_num > 0:
            # sin/cos per X-dimension only (exclude y)
            feat_size += 2 * max_input_dim * self.fourier_x_num
            # Frequencies log-spaced in [w_min, w_max]
            w_min, w_max = 0.5, 64.0
            w = torch.logspace(torch.log10(torch.tensor(w_min)), torch.log10(torch.tensor(w_max)), steps=self.fourier_x_num)
            self.register_buffer("omega_x", w, persistent=False)

        if self.use_logx_phases and self.fourier_logx_num > 0:
            feat_size += 2 * max_input_dim * self.fourier_logx_num
            v_min, v_max = 0.5, 64.0
            v = torch.logspace(torch.log10(torch.tensor(v_min)), torch.log10(torch.tensor(v_max)), steps=self.fourier_logx_num)
            self.register_buffer("nu_logx", v, persistent=False)

        # Per-point projection into d_model
        self.point_proj = nn.Sequential(
            nn.Linear(feat_size, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Pool-specific modules
        if self.pool_type == "encoder":
            enc_lyr = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads, batch_first=True, norm_first=True,
                dim_feedforward=d_model * 4, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_lyr, num_layers=num_layers)
            self.query = nn.Parameter(torch.randn(prefix_len, d_model) / (d_model ** 0.5))
            self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        elif self.pool_type == "xattn":
            self.query = nn.Parameter(torch.randn(prefix_len, d_model) / (d_model ** 0.5))
            self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        elif self.pool_type == "dsum":
            # Global pooled vector -> broadcast with learned token embeddings
            self.token_embed = nn.Parameter(torch.randn(prefix_len, hidden_size) / (hidden_size ** 0.5))
            self.global_proj = nn.Linear(d_model, hidden_size)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        # Common output projection and norm when needed
        if self.pool_type in ("encoder", "xattn"):
            self.out_proj = nn.Linear(d_model, hidden_size)
        self.ln_out = nn.LayerNorm(hidden_size)

    def _pad_truncate_X(self, X: torch.Tensor) -> torch.Tensor:
        B, N, V = X.shape
        if V < self.max_input_dim:
            pad = torch.zeros(B, N, self.max_input_dim - V, device=X.device, dtype=X.dtype)
            Xp = torch.cat([X, pad], dim=-1)
        else:
            Xp = X[:, :, : self.max_input_dim]
        return Xp

    def _build_features(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # X: (B,N,V), y: (B,N)
        B, N, V = X.shape
        Xp = self._pad_truncate_X(X)
        feats = []

        if self.use_raw_xy:
            feats.append(torch.cat([Xp, y.unsqueeze(-1)], dim=-1))

        if self.include_poly:
            # Use first x-dim for xy product for simplicity; pad if absent
            x0 = Xp[..., 0:1] if self.max_input_dim > 0 else torch.zeros(B, N, 1, device=X.device, dtype=X.dtype)
            y1 = y.unsqueeze(-1)
            x2 = x0 * x0
            y2 = y1 * y1
            xy = x0 * y1
            feats.append(torch.cat([x2, y2, xy], dim=-1))

        if self.use_x_phases and self.fourier_x_num > 0:
            # Only apply to X dims (exclude y)
            # proj: (B,N,Vx,F)
            Vx = self.max_input_dim
            Xx = Xp[:, :, :Vx]
            proj = Xx.unsqueeze(-1) * self.omega_x.view(1, 1, 1, -1)
            feats.append(torch.sin(proj).reshape(B, N, -1))
            feats.append(torch.cos(proj).reshape(B, N, -1))

        if self.use_logx_phases and self.fourier_logx_num > 0:
            Vx = self.max_input_dim
            Xx = Xp[:, :, :Vx]
            logx = torch.log(torch.clamp(Xx.abs(), min=self.log_eps))
            proj = logx.unsqueeze(-1) * self.nu_logx.view(1, 1, 1, -1)
            feats.append(torch.sin(proj).reshape(B, N, -1))
            feats.append(torch.cos(proj).reshape(B, N, -1))

        if not feats:
            raise ValueError("FeatureSetEmbedder constructed with no active feature blocks")
        pts = torch.cat(feats, dim=-1)

        # Optional per-sample normalization (off by default)
        if self.normalize:
            mean = pts.mean(dim=1, keepdim=True)
            std = pts.std(dim=1, keepdim=True).clamp_min(1e-6)
            pts = (pts - mean) / std
        return pts

    def _masked_sum(self, h: torch.Tensor, points_mask: torch.Tensor | None) -> torch.Tensor:
        # h: (B,N,D)
        if points_mask is None:
            return h.sum(dim=1)
        m = points_mask.to(h.dtype).unsqueeze(-1)  # (B,N,1)
        return (h * m).sum(dim=1)

    def forward(self, X: torch.Tensor, y: torch.Tensor, points_mask: torch.Tensor = None) -> torch.Tensor:
        B, N, _ = X.shape
        feats = self._build_features(X, y)
        h = self.point_proj(feats)  # (B,N,d_model)

        key_padding_mask = None
        if points_mask is not None:
            key_padding_mask = ~(points_mask.to(torch.bool))

        if self.pool_type == "encoder":
            h_enc = self.encoder(h, src_key_padding_mask=key_padding_mask)
            q = self.query.unsqueeze(0).expand(B, -1, -1)
            attn_out, _ = self.cross_attn(query=q, key=h_enc, value=h_enc, key_padding_mask=key_padding_mask, need_weights=False)
            out = self.out_proj(attn_out)
            out = self.ln_out(out)
            return out  # (B,K,H)

        if self.pool_type == "xattn":
            q = self.query.unsqueeze(0).expand(B, -1, -1)
            attn_out, _ = self.cross_attn(query=q, key=h, value=h, key_padding_mask=key_padding_mask, need_weights=False)
            out = self.out_proj(attn_out)
            out = self.ln_out(out)
            return out

        # dsum
        g = self._masked_sum(h, points_mask)  # (B,d_model)
        g_proj = self.global_proj(g)  # (B,H)
        out = g_proj.unsqueeze(1) + self.token_embed.unsqueeze(0)  # (B,K,H)
        out = self.ln_out(out)
        return out
