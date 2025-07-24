import torch
import torch.nn as nn
from timm import create_model

# --- SE Attention Block ---
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.se(x)  # shape: [B, C]
        return x * weights


# --- 單一 Swin 分支（可複用） ---
class SwinBranch(nn.Module):
    def __init__(self, backbone_name='swin_base_patch4_window12_384'):
        super().__init__()
        self.encoder = create_model(backbone_name, pretrained=True, features_only=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.output_dim = self.encoder.feature_info.channels()[-1]

    def forward(self, x):
        features = self.encoder(x)
        last_feat = features[-1]  # shape: [B, H, W, C]
        #print("last_feat.shape before permute:", last_feat.shape)

        # 轉成 NCHW 格式
        last_feat = last_feat.permute(0, 3, 1, 2)
        #print("last_feat.shape after permute:", last_feat.shape)

        pooled = self.pool(last_feat)  # shape: [B, C, 1, 1]
        out = self.flatten(pooled)     # shape: [B, C]
        #print("pooled shape:", pooled.shape)
        #print("flattened shape:", out.shape)
        return out


# --- 多分支 Swin 模型 + Attention 融合 ---
class SwinClassifierWithBranches(nn.Module):
    def __init__(self, num_classes=8, backbone='swin_base_patch4_window12_384'):
        super().__init__()
        self.branch_whole = SwinBranch(backbone)
        self.branch_body = SwinBranch(backbone)
        self.branch_edge = SwinBranch(backbone)

        # Attention blocks
        self.se_whole = SEBlock(self.branch_whole.output_dim)
        self.se_body = SEBlock(self.branch_body.output_dim)
        self.se_edge = SEBlock(self.branch_edge.output_dim)

        total_dim = self.branch_whole.output_dim * 3  # 拼接三支輸出
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_whole, x_body, x_edge):
        f_whole = self.branch_whole(x_whole)
        f_body = self.branch_body(x_body)
        f_edge = self.branch_edge(x_edge)

        # Apply SE attention
        f_whole = self.se_whole(f_whole)
        f_body = self.se_body(f_body)
        f_edge = self.se_edge(f_edge)

        combined = torch.cat([f_whole, f_body, f_edge], dim=1)
        out = self.classifier(combined)
        return out
