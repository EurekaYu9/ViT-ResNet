import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class LeavesDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        """
        叶子数据集加载器
        Args:
            root_dir: 数据集根目录
            transform: 图像变换
            train: 是否为训练集
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

        # 设置训练集或测试集路径
        data_dir = os.path.join(root_dir, 'train' if train else 'val')

        # 获取所有类别
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 获取所有图像路径和标签
        self.samples = []
        for target_class in self.classes:
            class_dir = os.path.join(data_dir, target_class)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png', ".JPG", ".JEPG", ".PNG")):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[target_class]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label}


def get_transforms(train=True):
    """获取图像变换"""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataloader(
        root_dir=r'D:\projects\python\open_clip-main-improve\data\Plant_leave_diseases_dataset_with_augmentation',
        batch_size=32, train=True, shuffle=None):
    """
    获取数据加载器
    Args:
        root_dir: 数据集根目录
        batch_size: 批次大小
        train: 是否为训练集
        shuffle: 是否打乱数据，默认训练集打乱，测试集不打乱
    """
    if shuffle is None:
        shuffle = train

    transform = get_transforms(train)
    dataset = LeavesDataset(root_dir=root_dir, transform=transform, train=train)

    return dataset, DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
