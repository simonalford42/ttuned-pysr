import torch
import torch.nn as nn
import torch.nn.functional as F


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
    test_point_embedder()
    print("\n" + "="*50 + "\n")
    test_e2e_point_embedder()
