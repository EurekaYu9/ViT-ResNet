import warnings
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from model import create_hybrid_model
from dataset import LeavesDataset, get_dataloader, get_transforms
from tqdm import tqdm
import os

# 忽略警告
warnings.filterwarnings("ignore", message="These pretrained weights were trained with QuickGELU activation.*")
warnings.filterwarnings("ignore", message="Converting mask without torch.bool dtype to bool.*")

# 设置环境变量，避免某些Windows系统上的OMP错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_prompts(prompt_file):
    """从文件加载提示词"""
    prompts = {}
    with open(prompt_file, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                class_name, prompt = line.strip().split(':', 1)
                prompts[class_name] = prompt
    return prompts


def zero_shot_evaluation(model, class_names, data_loader, device, prompts):
    model.eval()

    # 使用自定义提示词
    text_descriptions = []
    class_to_idx = {}
    
    for i, leaf_class in enumerate(class_names):
        if leaf_class in prompts:
            text_descriptions.append(prompts[leaf_class])
            class_to_idx[leaf_class] = i
        else:
            print(f"警告: 类别 '{leaf_class}' 在提示词文件中未找到")
            # 使用默认提示词
            text_descriptions.append(f"a leaf photo of a {leaf_class}")
            class_to_idx[leaf_class] = i

    # 获取文本特征
    with torch.no_grad():
        text_features = model.get_text_features(text_descriptions, device)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    correct = 0
    total = 0
    
    # 创建混淆矩阵
    confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="零样本评估"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # 获取图像特征
            image_features = model(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 计算相似度
            similarity = (100.0 * image_features @ text_features.T)

            # 获取预测
            _, predicted = similarity.max(1)

            # 更新混淆矩阵
            for t, p in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                confusion_matrix[t, p] += 1

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    print(f"零样本分类准确率: {accuracy:.2f}%")
    
    # 计算每个类别的准确率
    per_class_accuracy = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)

    # 打印每个类别的准确率
    print("\n每个类别的准确率:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {per_class_accuracy[i]*100:.2f}%")
    
    return accuracy, confusion_matrix, per_class_accuracy


def predict_single_image(model, image_path, device, prompts, class_names):
    """使用自定义提示词预测单个图像的类别"""
    # 使用测试变换而不是训练变换，避免随机增强
    transforms = get_transforms(train=False)

    # 加载图像
    image = Image.open(image_path).convert('RGB')
    # 转换图像并添加批次维度
    image_tensor = transforms(image).unsqueeze(0).to(device)

    # 使用自定义提示词
    text_descriptions = []
    for leaf_class in class_names:
        if leaf_class in prompts:
            text_descriptions.append(prompts[leaf_class])
        else:
            # 使用默认提示词
            text_descriptions.append(f"a leaf photo of a {leaf_class}")

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


def plot_confusion_matrix(confusion_matrix, class_names, figsize=(12, 10)):
    """绘制混淆矩阵"""
    plt.figure(figsize=figsize)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    
    # 设置刻度标记
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    
    # 添加数值标签
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用自定义提示词进行零样本分类")
    parser.add_argument("--model_path", type=str, default="./saved_models/best_vit_model.pth", help="模型路径")
    parser.add_argument("--vit_model", type=str, default="ViT-B-32", help="ViT模型名称")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda 或 cpu)")
    parser.add_argument("--image_path", type=str, default=None, help="单张图像的路径")
    parser.add_argument("--batch_size", type=int, default=32, help="评估的批量大小")
    parser.add_argument("--visualize", action="store_true", help="可视化一些预测结果")
    parser.add_argument("--random_seed", type=int, default=None, help="随机种子")
    parser.add_argument("--prompt_file", type=str, default="disease_prompts.txt", help="提示词文件路径")
    parser.add_argument("--confusion_matrix", action="store_true", help="是否绘制混淆矩阵")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    # 加载提示词
    prompt_file = os.path.join(os.path.dirname(__file__), args.prompt_file)
    prompts = load_prompts(prompt_file)
    print(f"已加载 {len(prompts)} 个类别的提示词")

    # 创建模型
    model = create_hybrid_model(vit_model=args.vit_model, resnet_model="RN18")

    # 加载训练好的模型权重
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 加载数据集以获取类别名称
    dataset, _ = get_dataloader(batch_size=1, train=False)
    class_names = dataset.classes

    if args.image_path:
        if not os.path.exists(args.image_path):
            print(f"错误: 图像文件 {args.image_path} 未找到")
        else:
            # 预测单张图像
            predictions = predict_single_image(
                model,
                args.image_path,
                device,
                prompts,
                class_names
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
        accuracy, confusion_matrix, per_class_accuracy = zero_shot_evaluation(
            model, dataset.classes, test_loader, device, prompts
        )

        # 如果需要绘制混淆矩阵
        if args.confusion_matrix:
            plot_confusion_matrix(confusion_matrix, dataset.classes)