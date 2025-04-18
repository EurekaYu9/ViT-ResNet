from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
import open_clip
import numpy as np
from PIL import Image


class HybridMNISTDataset(Dataset):
    def __init__(self, root="../data", train=True, vit_model_name="ViT-B-32",
                 resnet_model_name="RN50", pretrained="openai"):
        self.mnist = MNIST(root=root, train=train, download=True)
        
        # 获取VIT和ResNet的预处理转换
        _, _, self.vit_preprocess = open_clip.create_model_and_transforms(
            vit_model_name, pretrained=pretrained
        )
        
        _, _, self.resnet_preprocess = open_clip.create_model_and_transforms(
            resnet_model_name, pretrained=pretrained
        )
        
        # 使用VIT的预处理作为默认预处理
        self.preprocess = self.vit_preprocess

        self.label_texts = [
            "A handwritten digit 0",
            "A handwritten digit 1",
            "A handwritten digit 2",
            "A handwritten digit 3",
            "A handwritten digit 4",
            "A handwritten digit 5",
            "A handwritten digit 6",
            "A handwritten digit 7",
            "A handwritten digit 8",
            "A handwritten digit 9"
        ]

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]

        # 将灰度图像转换为RGB
        image = Image.fromarray(np.repeat(np.array(image)[:, :, np.newaxis], 3, axis=2).astype(np.uint8))
        
        # 应用预处理
        image = self.preprocess(image)

        return {
            'image': image,
            'label': label,
            'text': self.label_texts[label]
        }


def get_dataloader(batch_size=32, train=True, root="../data",
                   vit_model_name="ViT-B-32", resnet_model_name="RN50", pretrained="openai"):
    dataset = HybridMNISTDataset(
        root=root, 
        train=train, 
        vit_model_name=vit_model_name,
        resnet_model_name=resnet_model_name,
        pretrained=pretrained
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4, pin_memory=True)
