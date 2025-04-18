import warnings

import torch
import torch.nn as nn
import open_clip
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from resnet18 import ResNet18
from resnet34 import ResNet34

# 忽略警告
warnings.filterwarnings("ignore", message="These pretrained weights were trained with QuickGELU activation.*")
warnings.filterwarnings("ignore", message="Converting mask without torch.bool dtype to bool.*")


class HybridVisionEncoder(nn.Module):
    def __init__(self, vit_model, resnet_model, vit_model_name="ViT-B-32"):
        super().__init__()
        self.vit_model = vit_model
        self.resnet_model = resnet_model
        self.vit_model_name = vit_model_name
        
        # 获取模型输出维度
        self.vit_dim = self.vit_model.visual.output_dim
        self.resnet_dim = self.resnet_model.visual.output_dim
        
        # 创建交叉注意力层
        self.cross_attention = CrossAttention(
            dim=self.vit_dim,
            context_dim=self.resnet_dim,
            heads=8,
            dim_head=64,
            dropout=0.1
        )
        
        # 创建输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(self.vit_dim, 512),
            nn.LayerNorm(512)
        )

        # 冻结参数
        self._freeze_parameters()
        
    def _freeze_parameters(self):
        # 冻结VIT的大部分参数，只保留最后几层可训练
        for name, param in self.vit_model.visual.named_parameters():
            if 'transformer.resblocks.11' not in name and 'ln_post' not in name:
                param.requires_grad = False
                
        # 冻结ResNet的大部分参数，只保留最后几层可训练
        if self.resnet_model == "RN50":
            for name, param in self.resnet_model.visual.named_parameters():
                if 'layer4' not in name:
                    param.requires_grad = False
    
    def forward(self, x, return_features=False):
        # 获取VIT特征
        vit_features = self.vit_model.encode_image(x)  # [batch_size, vit_dim]
        
        # 获取ResNet特征
        resnet_features = self.resnet_model.encode_image(x)  # [batch_size, resnet_dim]
        
        # 使用VIT特征作为查询，ResNet特征作为键和值
        vit_features_unsqueezed = vit_features.unsqueeze(1)  # [batch_size, 1, vit_dim]
        resnet_features_unsqueezed = resnet_features.unsqueeze(1)  # [batch_size, 1, resnet_dim]
        
        # 应用交叉注意力
        attended_features = self.cross_attention(vit_features_unsqueezed, resnet_features_unsqueezed)  # [batch_size, 1, vit_dim]
        attended_features = attended_features.squeeze(1)  # [batch_size, vit_dim]
        
        # 投影到512维输出空间
        output = self.output_projection(attended_features)  # [batch_size, 512]

        # 如果需要返回中间特征，则返回ViT特征、ResNet特征和交叉注意力后的特征
        if return_features:
            return vit_features, resnet_features, output  # 最后一个output替代了原来的logits
        
        return output  # 直接返回特征，不进行分类
    
    # 添加获取文本特征的方法，用于零样本分类
    def get_text_features(self, text_descriptions, device):
        """获取文本特征用于零样本分类"""
        # 使用原始CLIP模型的文本编码器
        text_tokens = []
        
        # 获取tokenizer
        import open_clip
        tokenizer = open_clip.get_tokenizer(self.vit_model_name)
        
        for desc in text_descriptions:
            text_tokens.append(tokenizer([desc]).to(device))
        
        text_tokens = torch.cat(text_tokens)
        
        with torch.no_grad():
            text_features = self.vit_model.encode_text(text_tokens)
        
        return text_features


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else dim
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, context=None, mask=None):
        h = self.heads
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        
        # 计算注意力分数
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        # 如果提供了掩码，确保它是布尔类型
        if mask is not None:
            # 确保掩码是布尔类型
            mask = mask.bool() if not mask.dtype == torch.bool else mask
            # 扩展掩码维度以匹配注意力分数
            mask = mask.unsqueeze(1).unsqueeze(1)  # [b, 1, 1, seq_len]
            # 应用掩码
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        
        # 注意力权重
        attn = sim.softmax(dim=-1)
        
        # 加权聚合
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


# 创建混合模型
def create_hybrid_model(vit_model="ViT-B-32", resnet_model="RN50", pretrained="openai"):
    # 加载预训练的ViT模型
    vit, _, _ = open_clip.create_model_and_transforms(vit_model, pretrained=pretrained)

    # 加载ResNet模型
    if resnet_model == "RN18":
        resnet = ResNet18()
    elif resnet_model == "RN34":
        resnet = ResNet34()
    else:
        resnet, _, _ = open_clip.create_model_and_transforms(resnet_model, pretrained=pretrained)

    # 创建混合视觉编码器
    model = HybridVisionEncoder(vit_model=vit, resnet_model=resnet, vit_model_name=vit_model)

    return model
