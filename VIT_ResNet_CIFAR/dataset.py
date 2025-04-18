from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
import open_clip
import numpy as np
from PIL import Image
import torch
import os


class CIFARDataset(Dataset):
    def __init__(self, root="../data", train=True, vit_model_name="ViT-B-32",
                  pretrained="openai", cifar100=False, download=True):
        """
        CIFAR数据集加载器
        Args:
            root: 数据集根目录
            train: 是否为训练集
            vit_model_name: ViT模型名称
            resnet_model_name: ResNet模型名称
            pretrained: 预训练权重来源
            cifar100: 是否使用CIFAR100数据集，默认为CIFAR10
            download: 是否自动下载数据集
        """
        # 选择CIFAR10或CIFAR100
        if cifar100:
            self.cifar = CIFAR100(root=root, train=train, download=download)
            self.num_classes = 100
        else:
            self.cifar = CIFAR10(root=root, train=train, download=download)
            self.num_classes = 10
        
        # 获取VIT和ResNet的预处理转换
        _, _, self.vit_preprocess = open_clip.create_model_and_transforms(
            vit_model_name, pretrained=pretrained
        )

        # 使用VIT的预处理作为默认预处理
        self.preprocess = self.vit_preprocess

        # CIFAR10类别名称
        self.cifar10_classes = [
            "airplane", "automobile", "bird", "cat", "deer", 
            "dog", "frog", "horse", "ship", "truck"
        ]
        
        # 如果是CIFAR100，加载类别名称
        if cifar100:
            self.classes = self.cifar.classes
        else:
            self.classes = self.cifar10_classes
        
        # 为每个类别创建文本描述
        self.label_texts = [f"A photo of a {cls}" for cls in self.classes]

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        image, label = self.cifar[idx]
        
        # 应用预处理
        image = self.preprocess(image)
        
        return {
            'image': image,
            'label': label,
            'text': self.label_texts[label]
        }


def get_dataloader(batch_size=32, train=True, root="../data",
                   vit_model_name="ViT-B-32", resnet_model_name="RN50", 
                   pretrained="openai", cifar100=False, download=True):
    """
    获取CIFAR数据加载器
    Args:
        batch_size: 批次大小
        train: 是否为训练集
        root: 数据集根目录
        vit_model_name: ViT模型名称
        resnet_model_name: ResNet模型名称
        pretrained: 预训练权重来源
        cifar100: 是否使用CIFAR100数据集
        download: 是否自动下载数据集
    """
    dataset = CIFARDataset(
        root=root, 
        train=train, 
        vit_model_name=vit_model_name,
        pretrained=pretrained,
        cifar100=cifar100,
        download=download
    )
    
    # 返回数据集和数据加载器，与train.py中的调用方式保持一致
    return dataset, DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=train, 
        num_workers=4, 
        pin_memory=True
    )