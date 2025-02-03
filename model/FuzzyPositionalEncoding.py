import torch
from torch import nn
from torch.nn import functional as F

def interpolate_positional_encoding(positional_encoding, perturbed_coords):
    """
    Interpolates a 3D positional encoding tensor (D, H, W) at perturbed spatial coordinates.

    Args:
        positional_encoding (torch.Tensor): A tensor of shape (D, H, W) where:
                                             - D: Embedding dimension
                                             - H, W: Spatial grid dimensions
        perturbed_coords (torch.Tensor): A tensor of perturbed coordinates of shape (N, 2),
                                         where each row is a pair of (x, y) spatial coordinates
                                         in the range [-0.5, 0.5].

    Returns:
        torch.Tensor: Interpolated positional encodings at the perturbed coordinates, of shape (N, D).
    """
    # Get the dimensions of the positional encoding
    D, H, W = positional_encoding.shape

    # Reshape positional_encoding to (1, D, H, W) for grid_sample
    positional_encoding = positional_encoding.unsqueeze(0)  # Shape: (1, D, H, W)

    # Reshape perturbed_coords to match grid_sample input
    # Expected shape: (1, N, 1, 2), where N is the number of perturbed coordinates
    perturbed_coords = perturbed_coords.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, N, 2)

    # Perform grid sampling
    interpolated = F.grid_sample(
        positional_encoding,  # Shape: (1, D, H, W)
        perturbed_coords,     # Shape: (1, 1, N, 2)
        mode="bilinear",
        align_corners=True
    )

    # Reshape to grid
    return interpolated.squeeze().reshape(D, H, W)


class FuzzyPositionalEncoding(nn.Module):
    def __init__(self, H, W, D):
        """
        Fuzzy Positional Encoding (FPE) layer.
        
        Args:
            H (int): Height of the Positional Encoding Latent Grid.
            W (int): Width of the Positional Encoding Latent Grid.
            D (int): Embedding dimension.
        """
        super().__init__()
        self.H = H
        self.W = W
        self.D = D
        self.inference = False
        self.pos_encoding = nn.Parameter(torch.zeros(D, H, W))

    def perturb(self):
        """
        Perturbs the positional encoding grid with random perturbations and calculates the positional encoding values at these perturbed points.
        """
        perturbation_x = torch.rand(self.H, self.W) - 0.5  # Shape: (H, W)
        perturbation_y = torch.rand(self.H, self.W) - 0.5  # Shape: (H, W)
        perturbation_coords = torch.stack([perturbation_x.flatten(), perturbation_y.flatten()], dim=-1)  # Shape: (H*W, 2)
        perturbed_pos_encoding = interpolate_positional_encoding(self.pos_encoding, perturbation_coords.to(self.pos_encoding.device))
        return perturbed_pos_encoding


    def forward(self, x):
        """
        Forward pass of the Fuzzy Positional Encoding layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W) with the positional encoding added.
        """
        _, _, H, W = x.shape
        if not self.inference:
            pos_encoding = self.perturb()
        else:
            pos_encoding = self.pos_encoding.detach()
        resized_pos_encoding = F.interpolate(pos_encoding.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=True)
        return x + resized_pos_encoding.squeeze()


    def train(self, mode=True):
        """
        Overrides the train method to ensure that the FPE layer enters training mode
        when the parent model switches to training mode.
        """
        self.inference = not mode  # In training mode, enable perturbations
        return super().train(mode)

    def eval(self):
        """
        Overrides the eval method to ensure that the FPE layer enters inference mode
        when the parent model switches to evaluation mode.
        """
        self.inference = True  # In evaluation mode, disable perturbations
        return super().eval()