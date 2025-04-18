import warnings

import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from train import parse_class_name
from model import create_hybrid_model
from dataset import LeavesDataset, get_dataloader, get_transforms
from tqdm import tqdm
import os

# python predict.py --model_path ./saved_models_RN-18/best_vit_model.pth --visualize
# python predict.py --model_path ./saved_models/best_vit_model.pth --image_path D:\projects\python\open_clip-main-improve\data\Plant_leave_diseases_dataset_with_augmentation\train\Blueberry___healthy\image.jpg


# 忽略警告
warnings.filterwarnings("ignore", message="These pretrained weights were trained with QuickGELU activation.*")
warnings.filterwarnings("ignore", message="Converting mask without torch.bool dtype to bool.*")

# 设置环境变量，避免某些Windows系统上的OMP错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def zero_shot_evaluation(model, class_names, data_loader, device):
    model.eval()

    text_descriptions = []
    # 定义分类标签文本
    for leaf_class in class_names:
        text_descriptions += [f"a leaf photo of a {leaf_class}"]

    # 获取文本特征
    with torch.no_grad():
        text_features = model.get_text_features(text_descriptions, device)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Zero-shot Evaluation"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # 获取图像特征
            image_features = model(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 计算相似度
            similarity = (100.0 * image_features @ text_features.T)

            # 获取预测
            _, predicted = similarity.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    print(f"零样本分类准确率: {accuracy:.2f}%")
    return accuracy


def predict_single_image(model, image_path, device):
    """预测单个图像的类别"""
    # 使用测试变换而不是训练变换，避免随机增强
    transforms = get_transforms(train=False)
    dataset = LeavesDataset(
        root_dir=r'D:\projects\python\open_clip-main-improve\data\Plant_leave_diseases_dataset_with_augmentation',
        transform=transforms, train=False)

    # 加载图像
    image = Image.open(image_path).convert('RGB')
    # 转换图像并添加批次维度
    image_tensor = transforms(image).unsqueeze(0).to(device)

    class_names = dataset.classes
    text_descriptions = []
    for leaf_class in class_names:
        text_descriptions += [parse_class_name(leaf_class)]

    text_features = model.get_text_features(text_descriptions, device)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        # 使用批次维度的图像张量
        image_features = model(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # 计算相似度
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # 获取预测
    values, indices = similarity[0].topk(3)

    # 返回预测结果和类别名称
    return [(class_names[indices[i].item()], values[i].item()) for i in range(3)]


def visualize_predictions(model, class_names, test_loader, device, num_images=10, random_seed=None):
    model.eval()

    # 设置随机种子（如果提供）
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # 从数据加载器中获取所有数据
    all_images = []
    all_labels = []

    # 只获取有限数量的批次以避免内存问题
    max_batches = 500
    batch_count = 0

    with torch.no_grad():
        for batch in test_loader:
            all_images.append(batch['image'])
            all_labels.append(batch['label'])
            batch_count += 1
            if batch_count >= max_batches:
                break

    # 合并批次
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 随机选择样本
    total_samples = len(all_images)
    indices = torch.randperm(total_samples)[:num_images]

    images = all_images[indices]
    labels = all_labels[indices]

    text_descriptions = []
    for leaf_class in class_names:
        text_descriptions += [parse_class_name(leaf_class)]

    # 文本特征
    text_features = model.get_text_features(text_descriptions, device)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 获取图像特征
    with torch.no_grad():
        image_features = model(images.to(device))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # 计算相似度
    similarity = (100.0 * image_features @ text_features.T)
    _, predicted = similarity.max(1)

    # 可视化
    plt.figure(figsize=(15, 8))
    for i in range(min(num_images, len(images))):
        plt.subplot(2, 5, i + 1)
        img = images[i].permute(1, 2, 0).cpu().numpy()
        # 归一化图像数据到 [0, 1] 范围
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        plt.imshow(img)
        plt.title(f"true result: {class_names[labels[i].item()]} \n predict: {class_names[predicted[i].item()]}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('vit_predictions.png')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用Hybrid模型进行PLANT预测")
    parser.add_argument("--model_path", type=str, default="./saved_models/best_ViT_model.pth", help="模型路径")
    parser.add_argument("--vit_model", type=str, default="ViT-B-32", help="ViT模型名称")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda 或 cpu)")
    parser.add_argument("--image_path", type=str, default=None, help="单张图像的路径")
    parser.add_argument("--batch_size", type=int, default=32, help="评估的批量大小")
    parser.add_argument("--visualize", action="store_true", help="可视化一些预测结果")
    parser.add_argument("--random_seed", type=int, default=None, help="随机种子")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    # 创建模型
    model = create_hybrid_model(vit_model=args.vit_model,resnet_model="RN18")

    # 加载训练好的模型权重
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    if args.image_path:
        if not os.path.exists(args.image_path):
            print(f"错误: 图像文件 {args.image_path} 未找到")
        else:
            # 预测单张图像
            predictions = predict_single_image(
                model,
                args.image_path,
                device
            )
            print("预测结果 (类别, 置信度):")
            for i, (cls, prob) in enumerate(predictions):
                print(f"{i + 1}. 类别 {cls} ({prob:.4f})")

    else:
        # 加载测试数据集
        dataset, test_loader = get_dataloader(
            batch_size=args.batch_size,
            train=False,
        )

        # 评估模型
        zero_shot_evaluation(model, dataset.classes, test_loader, device)

        # 如果需要可视化
        if args.visualize:
            visualize_predictions(model, dataset.classes, test_loader, device, random_seed=args.random_seed)
