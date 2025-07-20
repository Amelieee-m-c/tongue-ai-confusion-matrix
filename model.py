import torch.nn as nn
from timm import create_model

class SwinClassifier(nn.Module):
    def __init__(self, num_classes=13, backbone='swin_base_patch4_window12_384'):
        super().__init__()
        self.backbone = create_model(backbone, pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)
