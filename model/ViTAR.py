import torch
from torch import nn
from torch.nn import functional as F
from model.FuzzyPositionalEncoding import FuzzyPositionalEncoding
from model.AdaptiveTokenMerger import AdaptiveTokenMerger

class PatchEmbed(nn.Module):
    def __init__(self,patch_size=16,in_chans=3,embed_dim=768):
        """
        Generates the patch embeddings from the input image.
        Args:
            patch_size (int): The size of the patch to be extracted.
            in_chans (int): Number of input channels.
            embed_dim (int): The dimension of the embedding
        """
        super().__init__()
        self.patch_size=patch_size
        self.embedding_layer = nn.Conv2d(in_chans,embed_dim,kernel_size=patch_size,stride=patch_size, padding=0)

    def calculate_padding(self, img_dim, patch_dim):
        """
        Calculate padding to ensure that the image dimensions are divisible by the patch dimensions.
        Args:
            img_dim (int): The dimension of the image.
            patch_dim (int): The dimension of the patch.
        Returns:
            Tuple[int, int]: Padding values for the given image dimension."""
        H = img_dim
        H_pad = [0, 0]
        if H % patch_dim != 0:
            if H % 2 == 1:
                H_pad[0] += 1
            H_pad[0] += (patch_dim - H % patch_dim) // 2
            H_pad[1] += (patch_dim - H % patch_dim) // 2
        return H_pad

    def forward(self,x):
        """
        Forward pass of the PatchEmbed module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (B, D, H_new, W_new).
        """
        img_size = x.shape[-2:]
        padding_height = self.calculate_padding(img_size[0], self.patch_size)
        padding_width = self.calculate_padding(img_size[1], self.patch_size)
        padding = (padding_height, padding_width)
        x = F.pad(x, (padding[1][0], padding[1][1], padding[0][0], padding[0][1]))
        x = self.embedding_layer(x)
        return x

class ViTAR(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=4 * 768, Gh=14, Gt=14, pe_H=64, pe_W=64, num_heads=12, num_layers=12, num_classes=10, atm_num_heads=12):
        """
        Vision Transformer for Any Resolution (ViTAR) model.

        Args:
            embedding_dim (int, optional): Dimension of the patch embeddings. Defaults to 768.
            hidden_dim (int, optional): Dimension of the hidden layers. Defaults to 4*768.
            Gh (int, optional): Target height of the grid. Defaults to 14.
            Gt (int, optional): Target width of the grid. Defaults to 14.
            pe_H (int, optional): Height of the Positional Encoding Latent Grid. Defaults to 64.
            pe_W (int, optional): Width of the Positional Encoding Latent Grid. Defaults to 64.
            num_heads (int, optional): Number of attention heads. Defaults to 12.
            num_layers (int, optional): Number of transformer layers. Defaults to 12.
            num_classes (int, optional): Number of output classes. Defaults to 10.
            atm_num_heads (int, optional): Number of attention heads in the Adaptive Token Merger. Defaults to 12.

            Default configuration is the ViTAR-Base model.
        """
        super().__init__()
        self.embedding_layer = PatchEmbed(patch_size=16, embed_dim=embedding_dim)
        self.pe = FuzzyPositionalEncoding(pe_H, pe_W, embedding_dim)
        self.atm = AdaptiveTokenMerger(Gh=Gh, Gt=Gt, embedding_dim=embedding_dim, num_heads=atm_num_heads)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True), num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

    def forward(self, images):
        """
        Forward pass of the ViTAR model.
        
        Args:
            images (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes).
        """
        outputs = []
        for image in images:
            image = image.unsqueeze(0)
            output = self.embedding_layer(image)
            output = self.pe(output)
            output = self.atm(output)
            outputs.append(output)
        x = torch.cat(outputs, dim=0)
        B, E, _, _ = x.shape
        x = x.reshape(B, -1, E)
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.transformer_encoder(x)
        x = self.fc(x[:, 0, :])
        return x