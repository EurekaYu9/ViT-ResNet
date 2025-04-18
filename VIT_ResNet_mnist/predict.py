import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from model import create_hybrid_model
from dataset import HybridMNISTDataset, get_dataloader
from tqdm import tqdm
import os

# python predict.py --model_path ./saved_models/best_hybrid_model.pth --visualize
# python predict.py --model_path ./saved_models/best_hybrid_model.pth --image_path D:\projects\python\open_clip-main-improve\VIT_ResNet_mnist\data\test_img\img.png  --visualize

# 设置环境变量，避免某些Windows系统上的OMP错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def zero_shot_evaluation(model, data_loader, device):
    """零样本分类评估"""
    model.eval()
    
    # 文本描述
    text_descriptions = [
        "a photo of number 0",
        "a photo of number 1",
        "a photo of number 2",
        "a photo of number 3",
        "a photo of number 4",
        "a photo of number 5",
        "a photo of number 6",
        "a photo of number 7",
        "a photo of number 8",
        "a photo of number 9"
    ]
    
    # 获取文本特征
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


def predict_single_image(model, image_path, device, vit_model_name="ViT-B-32", resnet_model_name="RN50"):
    """预测单个图像的类别"""
    dataset = HybridMNISTDataset(train=False, vit_model_name=vit_model_name, resnet_model_name=resnet_model_name)
    
    # 按照MNIST数据集的方法预处理图片
    image = Image.open(image_path).convert('L')
    image = Image.fromarray(np.repeat(np.array(image)[:, :, np.newaxis], 3, axis=2).astype(np.uint8))
    image = dataset.preprocess(image)
    image = image.unsqueeze(0).to(device)

    # 文本描述
    text_descriptions = [
        "a photo of number 0",
        "a photo of number 1",
        "a photo of number 2",
        "a photo of number 3",
        "a photo of number 4",
        "a photo of number 5",
        "a photo of number 6",
        "a photo of number 7",
        "a photo of number 8",
        "a photo of number 9"
    ]
    
    # 文本特征
    text_features = model.get_text_features(text_descriptions, device)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 图像特征
    with torch.no_grad():
        image_features = model(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # 计算相似度
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # 获取预测
    values, indices = similarity[0].topk(3)
    
    return [(indices[i].item(), values[i].item()) for i in range(3)]


def visualize_predictions(model, test_loader, device, num_images=10, random_seed=None):
    """可视化预测结果"""
    model.eval()

    # 设置随机种子（如果提供）
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # 从数据加载器中获取所有数据
    all_images = []
    all_labels = []
    
    # 只获取有限数量的批次以避免内存问题
    max_batches = 5
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

    # 文本描述
    text_descriptions = [f"A handwritten digit {i}" for i in range(10)]
    
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
        plt.title(f"true result: {labels[i].item()}, predict: {predicted[i].item()}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('hybrid_predictions.png')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用混合视觉编码器模型进行MNIST预测")
    parser.add_argument("--model_path", type=str, default="./saved_models/best_hybrid_model.pth", help="模型路径")
    parser.add_argument("--vit_model", type=str, default="ViT-B-32", help="ViT模型名称")
    parser.add_argument("--resnet_model", type=str, default="RN50", help="ResNet模型名称")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda 或 cpu)")
    parser.add_argument("--image_path", type=str, default=None, help="单张图像的路径")
    parser.add_argument("--batch_size", type=int, default=32, help="评估的批量大小")
    parser.add_argument("--visualize", action="store_true", help="可视化一些预测结果")
    parser.add_argument("--random_seed", type=int, default=None, help="随机种子")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    # 创建模型
    model = create_hybrid_model(vit_model=args.vit_model, resnet_model=args.resnet_model)
    
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
                device,
                vit_model_name=args.vit_model,
                resnet_model_name=args.resnet_model
            )
            print("预测结果 (类别, 置信度):")
            for i, (cls, prob) in enumerate(predictions):
                print(f"{i+1}. 数字 {cls} ({prob:.4f})")

    else:
        # 加载测试数据集
        test_loader = get_dataloader(
            batch_size=args.batch_size,
            train=False,
            vit_model_name=args.vit_model,
            resnet_model_name=args.resnet_model
        )

        # 评估模型
        zero_shot_evaluation(model, test_loader, device)

        # 如果需要可视化预测结果
        if args.visualize:
            visualize_predictions(model, test_loader, device, random_seed=args.random_seed)
