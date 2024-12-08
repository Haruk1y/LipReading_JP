import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
import math

class Conv3D(nn.Module):
    """3D Convolution for initial feature extraction"""
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(5, 7, 7),
                     stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
    def forward(self, x):
        # x shape: (B, C, T, H, W)
        return self.conv(x)

class ResNetBlock(nn.Module):
    """Basic ResNet block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ImageEncoder(nn.Module):
    """Image Encoder with 3D Conv and ResNet34"""
    def __init__(self, in_channels=3, hidden_dim=512):
        super().__init__()
        self.conv3d = Conv3D(in_channels, 64)
        
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, hidden_dim, 3, stride=2)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = [ResNetBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: (B, C, T, H, W)
        B, C, T, H, W = x.size()
        x = self.conv3d(x)
        
        # Reshape and process each frame independently
        _, C, T, H, W = x.size()
        x = x.transpose(1, 2).contiguous()
        x = x.view(B * T, C, H, W)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Reshape back
        _, C, H, W = x.size()
        x = x.view(B, T, C, H, W)
        x = x.mean(dim=(3, 4))  # Global average pooling
        return x  # Shape: (B, T, C)

class ConformerBlock(nn.Module):
    """Improved Conformer block with expansion factors"""
    def __init__(self, dim, num_heads=8, ff_dim=2048, kernel_size=31, conv_expansion_factor=2):
        super().__init__()
        
        # First Feed Forward Module
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, dim),
            nn.Dropout(0.1),
        )
        
        # Multi-Head Self Attention
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=0.1)
        self.attn_norm = nn.LayerNorm(dim)
        
        # Convolution Module
        conv_dim = dim * conv_expansion_factor
        # LayerNormを独立させる
        self.conv_norm = nn.LayerNorm(dim)
        self.conv_module = nn.Sequential(
            nn.Conv1d(dim, conv_dim, 1),
            nn.GELU(),
            nn.Conv1d(conv_dim, conv_dim, kernel_size, padding=kernel_size//2, groups=conv_dim),
            nn.BatchNorm1d(conv_dim),
            nn.GELU(),
            nn.Conv1d(conv_dim, dim, 1),
            nn.Dropout(0.1),
        )
        
        # Second Feed Forward Module
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, dim),
            nn.Dropout(0.1),
        )
        
        self.final_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Feed Forward 1
        x = x + 0.5 * self.ff1(x)
        
        # Self Attention
        attn_out = self.attn_norm(x)
        attn_out, _ = self.self_attn(attn_out.transpose(0,1), attn_out.transpose(0,1), attn_out.transpose(0,1))
        x = x + attn_out.transpose(0,1)
        
        # Convolution Module
        # 正規化を先に適用
        conv_in = self.conv_norm(x)
        conv_in = conv_in.transpose(1, 2)
        x = x + self.conv_module(conv_in).transpose(1, 2)

        # Feed Forward 2
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)

class ConformerEncoder(nn.Module):
    """Conformer Encoder with improved architecture"""
    def __init__(self, dim=512, num_layers=4, ff_expansion_factor=4, conv_expansion_factor=2):
        super().__init__()
        
        self.layers = nn.ModuleList([
            ConformerBlock(
                dim=dim,
                ff_dim=dim * ff_expansion_factor,
                conv_expansion_factor=conv_expansion_factor
            ) for _ in range(num_layers)
        ])
        
        # Final normalization layer
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class TransformerDecoder(nn.Module):
    """Transformer Decoder for sequence generation"""
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)


        # TransformerDecoderLayerにdropoutを追加
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout  # dropoutを設定
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Additional dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask=None):
        # マスクのサイズチェックと調整
        if tgt_mask is not None:
            seq_len = tgt.size(1)
            tgt_mask = tgt_mask[:seq_len, :seq_len]

        # 型をFloat32に統一
        tgt = self.embedding(tgt).to(memory.dtype)
        tgt = self.dropout(tgt)  # Apply dropout to embeddings

        # 次元の順序を調整
        tgt = tgt.transpose(0, 1)     # (B, T, E) -> (T, B, E)
        memory = memory.transpose(0, 1)  # (B, T, E) -> (T, B, E)

        try:
            output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            output = output.transpose(0, 1)  # (T, B, E) -> (B, T, E)
            return self.output_projection(output)
        except RuntimeError as e:
            print(f"Error in TransformerDecoder: {str(e)}")
            print(f"Shapes - tgt: {tgt.shape}, memory: {memory.shape}, mask: {tgt_mask.shape if tgt_mask is not None else None}")
            raise


class LipReadingModel(nn.Module):
    """Complete Lip Reading Model"""
    def __init__(self, vocab_size, hidden_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder(hidden_dim=hidden_dim)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Improved Conformer
        self.conformer = ConformerEncoder(
            dim=hidden_dim,
            num_layers=6,
            ff_expansion_factor=4,
            conv_expansion_factor=2
        )
        
        # Improved Decoder
        self.decoder = TransformerDecoder(
            vocab_size,
            d_model=hidden_dim,
            nhead=8,
            num_layers=6,
            dropout=0.2
        )
        
        # Initialize weights
        self.apply(self._init_weights)

        # CUDAメモリの使用量を削減
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, tgt=None, tgt_mask=None):
        """
        Args:
            x: input frames (B, C, T, H, W)
            tgt: target sequence (B, L)
            tgt_mask: target mask for decoder
        """
        try:
            # Encode visual features
            visual_features = self.image_encoder(x)  # (B, T, C)
            visual_features = self.dropout(visual_features)
            
            # Apply Conformer
            encoded_features = self.conformer(visual_features)  # (B, T, C)
            encoded_features = self.dropout(encoded_features)
            
            # Decoding
            if tgt is not None:
                # Training phase
                output = self.decoder(tgt, encoded_features, tgt_mask)
            else:
                # Inference phase
                batch_size = x.size(0)
                init_token = torch.zeros((batch_size, 1), dtype=torch.long, device=x.device)
                output = self.decoder(init_token, encoded_features)
                
            return output

        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Input shapes:")
            print(f"x: {x.shape}")
            print(f"tgt: {tgt.shape if tgt is not None else 'None'}")
            print(f"tgt_mask: {tgt_mask.shape if tgt_mask is not None else 'None'}")
            raise e

    def generate_square_subsequent_mask(self, sz):
        """Generate mask for decoder's self-attention with size check"""
        if sz <= 0:
            return None
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(next(self.parameters()).dtype)