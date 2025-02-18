import torch
from torch import nn

def grid_pad(img, Gh, Gw):
    '''
    Grid Padding for Adaptive Token Merging (ATM).

    Args:
        img (torch.Tensor): Input tensor of shape (B, C, H, W).
        Gh (int): Target height of the grid.
        Gw (int): Target width of the grid.
    '''
    B, C, H, W = img.shape
    
    if H > 2 * Gh:
        # If the dimensions are greater than twice the target dimension size we only pad it to the nearest even size
        if H % 2 != 0:
            h_pad = 1
        else:
            h_pad = 0
    else:
        # Otherwise we pad it to twice the target dimension size Gh
        h_pad = 2 * Gh - H

    if W > 2 * Gw:
        # If the dimensions are greater than twice the target dimension size we only pad it to the nearest even size
        if W % 2 != 0:
            w_pad = 1
        else:
            w_pad = 0
    else:
        # Otherwise we pad it to twice the target dimension size Gw
        w_pad = 2 * Gw - W

    new_H, new_W = H + h_pad, W + w_pad

    if h_pad == 0 and w_pad == 0:
        return img

    padded_tensor = torch.zeros(B, C, new_H, new_W).to(img.device)

    if h_pad < H and w_pad < W:
        # If the padding is less than the image dimension size

        # Grid pad the top left section
        padded_tensor[:, :, :2 * h_pad:2, :2 * w_pad:2] = img[:, :, :h_pad, :w_pad]
        # Grid pad the bottom left section
        padded_tensor[:, :, 2 * h_pad:, :2 * w_pad:2] = img[:, :, h_pad:, :w_pad]
        # Grid pad the top right section
        padded_tensor[:, :, :2 * h_pad:2, 2 * w_pad:] = img[:, :, :h_pad, w_pad:]
        # Grid pad the bottom right section
        padded_tensor[:, :, 2 * h_pad:, 2 * w_pad:] = img[:, :, h_pad:, w_pad:]
    elif h_pad < H:
        # Grid pad the top section
        padded_tensor[:, :, :2 * h_pad:2, :2 * W:2] = img[:, :, :h_pad, :]
        # Grid pad the bottom section
        padded_tensor[:, :, 2 * h_pad:, :2 * W:2] = img[:, :, h_pad:, :]
    elif w_pad < W:
        # Grid pad the left section
        padded_tensor[:, :, :2 * H:2, :2 * w_pad:2] = img[:, :, :, :w_pad]
        # Grid pad the right section
        padded_tensor[:, :, :2 * H:2, 2 * w_pad:] = img[:, :, :, w_pad:]
    else:
        padded_tensor[:, :, :2 * H:2, :2 * W:2] = img[:, :, :, :]

    return padded_tensor


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, dropout=0.1):
        """
        FeedForward Network for Adaptive Token Merging (ATM).

        Args:
            embed_dim (int): Dimension of the input and output embeddings (D).
            hidden_dim (int, optional): Dimension of the intermediate hidden layer. Defaults to 4 * embed_dim.
            dropout (float): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim * 4  # Default to 4x expansion.

        self.fc1 = nn.Linear(embed_dim, hidden_dim)  # First linear layer
        self.activation = nn.GELU()  # Non-linear activation
        self.fc2 = nn.Linear(hidden_dim, embed_dim)  # Second linear layer
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization

    def forward(self, x):
        """
        Forward pass of the FFN.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, D).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, D).
        """
        x = self.fc1(x)  # (B, N, hidden_dim)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (B, N, embed_dim)
        x = self.dropout(x)
        return x


class AdaptiveTokenMerger(nn.Module):
    def __init__(self, adaptive_ratio=2, Gh=14, Gt=14, embedding_dim=768, num_heads=12):
        """
        Adaptive Token Merger (ATM) module.

        Args:
            adaptive_ratio (int, optional): Ratio for adaptive token merging. Defaults to 2.
            Gh (int, optional): Target height of the grid. Defaults to 14.
            Gt (int, optional): Target width of the grid. Defaults to 14.
            embedding_dim (int, optional): Dimension of the input and output embeddings (D). Defaults to 768.
            num_heads (int, optional): Number of attention heads. Defaults to 12.
        Returns:
            torch.Tensor: Output tensor of shape (B, D, Gh, Gt).
        """
        super().__init__()
        self.adaptive_ratio = adaptive_ratio
        self.Gh = Gh
        self.Gt = Gt
        self.pooling = nn.AvgPool2d(kernel_size=adaptive_ratio, stride=adaptive_ratio)
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        self.ffn = FeedForward(embed_dim=embedding_dim)

    def ae_iteration(self, x):
        """
        A single Adaptive Token Merging (ATM) iteration.

        Args:
            x (torch.Tensor): Input tensor of shape (B, D, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (B, D, H_new, G_new).
        """
        B, D, _, _ = x.shape
        y = grid_pad(x, self.Gh, self.Gt)
        y = self.pooling(y)
        B_, D_, H_, W_ = y.shape
        x = x.reshape(B, -1, D)
        y = y.reshape(B, -1, D)

        x, _ = self.attention(y, x, x)
        x = y + x
        x = self.ffn(x)
        x = x.reshape(B_, H_, W_, D_)
        return x.permute(0, 3, 1, 2)

    def forward(self, x):
        """
        Forward pass of the ATM module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, D, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (B, D, Gh, Gt).
        """
        _, _, H, W = x.shape
        while H != self.Gh or W != self.Gt:
            x = self.ae_iteration(x)
            _, _, H, W = x.shape
        return x