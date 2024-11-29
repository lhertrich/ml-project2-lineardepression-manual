import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    
    def __init__(self, img_size=256, patch_size=16, num_classes=1, embed_dim=768, depth=12, num_heads=8):
        super(VisionTransformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch Embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional Encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(0.1)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        B, C, H, W = x.size()
        x = self.patch_embed(x)  # Shape: [B, embed_dim, num_patches_sqrt, num_patches_sqrt]
        x = x.flatten(2).transpose(1, 2)  # Shape: [B, num_patches, embed_dim]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Shape: [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # Add CLS token
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer(x)  # Apply transformer encoder
        cls_output = x[:, 0]  # CLS token output

        out = self.mlp_head(cls_output)  # Final classification/segmentation
        return out
