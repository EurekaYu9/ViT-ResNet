import warnings
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

# 忽略警告
warnings.filterwarnings("ignore", message="These pretrained weights were trained with QuickGELU activation.*")
warnings.filterwarnings("ignore", message="Converting mask without torch.bool dtype to bool.*")


class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练的ResNet34
        resnet34 = models.resnet34(pretrained=True)
        # 移除最后的全连接层
        self.backbone = nn.Sequential(*list(resnet34.children())[:-1])
        # 设置输出维度与RN50一致，方便与VIT交互
        self.output_dim = 1024
        self.projection = nn.Linear(512, self.output_dim)

    def encode_image(self, x):
        # 提取特征
        features = self.backbone(x)
        # 调整形状
        features = features.squeeze(-1).squeeze(-1)
        # 投影到更高维度
        features = self.projection(features)
        # 归一化
        features = F.normalize(features, dim=1)
        return features

    @property
    def visual(self):
        return self