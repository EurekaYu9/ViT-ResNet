import argparse
import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

from dataset import get_dataloader
from model import create_hybrid_model

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

# python train.py --vit_model ViT-B-32 --batch_size 32 --learning_rate 1e-4 --epochs 2 --save_dir "./saved_models" --visualize_features --vis_interval 1
# tensorboard --logdir="./saved_models/logs" --samples_per_plugin images=99999

# 忽略警告 Mask BOOL
warnings.filterwarnings("ignore", message="Converting mask without torch.bool dtype to bool.*")


def visualize_features(features, labels, class_names, epoch, save_dir, prefix=""):
    """使用t-SNE可视化特征"""
    tsne = TSNE(n_components=2, random_state=42)
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # 降维到2D
    features_2d = tsne.fit_transform(features_np)

    # 绘制散点图
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels_np)
    for i in unique_labels:
        plt.scatter(
            features_2d[labels_np == i, 0],
            features_2d[labels_np == i, 1],
            label=class_names[i],
            alpha=0.7,
            edgecolors='w'
        )

    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.title(f"{prefix} Feature Space Visualization (Epoch {epoch})")

    # 保存图像
    os.makedirs(os.path.join(save_dir, "feature_vis"), exist_ok=True)
    save_path = os.path.join(save_dir, "feature_vis", f"{prefix}_epoch_{epoch}.png")
    plt.savefig(save_path, bbox_inches='tight')  # 保持图例完整
    plt.close()


def zero_shot_evaluation(model, class_names, data_loader, device):
    """零样本分类评估"""
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
            # 归一化特征
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 计算相似度
            similarity = image_features @ text_features.T

            # 获取预测
            _, predicted = similarity.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    print(f"零样本评估准确率：{accuracy}%")

    return accuracy


def parse_class_name(class_name):
    if class_name == 'Background_without_leaves':
        return "a background photo without leaves"

    if '___' in class_name:
        plant_part, condition_part = class_name.split('___', 1)
        plant = plant_part.split('_')[0]
        if condition_part == 'healthy':
            return f"a healthy {plant} leaf photo"
        else:
            disease = condition_part.replace('_', ' ')
            return f"a {plant} leaf photo with {disease}"
    return f"a leaf photo of {class_name}"


def train(args):
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    # 创建TensorBoard日志
    writer = SummaryWriter(os.path.join(args.save_dir, "logs"))

    # 获取数据加载器
    dataset, train_loader = get_dataloader(
        batch_size=args.batch_size,
        train=True,
    )

    _, val_loader = get_dataloader(
        batch_size=args.batch_size,
        train=False,
    )

    # 创建混合视觉编码器模型
    model = create_hybrid_model(
        vit_model=args.vit_model,
        resnet_model=args.resnet_model,
        pretrained=args.pretrained
    )
    model = model.to(device)

    class_names = dataset.classes
    text_descriptions = []
    # 定义分类标签文本
    for leaf_class in class_names:
        text_descriptions += [parse_class_name(leaf_class)]
    # print(f"数据集长度{len(dataset)}")
    # print(f"训练数据长度{len(train_loader)}")
    # print(f"测试数据长度{len(val_loader)}")
    # print(f"种类数目{len(class_names)}")
    # print(text_descriptions)

    # 定义优化器
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )
    # 定义学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # 定义损失函数，使用交叉熵 - 对比学习
    criterion = nn.CrossEntropyLoss()

    # 训练前进行零样本分类评估
    print("训练前进行零样本分类评估...")
    initial_accuracy = zero_shot_evaluation(model, class_names, val_loader, device)
    # 记录最佳零样本准确率
    best_zero_shot_acc = initial_accuracy

    print("\n========训练开始========")
    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # 获取文本特征
            with torch.no_grad():
                text_features = model.get_text_features(text_descriptions, device)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 获取图像特征
            image_features = model(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 计算图像-文本相似度
            logits = image_features @ text_features.T * 100.0  # 缩放因子

            # 计算损失
            loss = criterion(logits, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新统计信息
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': 100. * train_correct / train_total
            })

        # 更新学习率
        scheduler.step()

        # 记录训练指标
        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

        # 特征可视化部分
        if epoch % args.vis_interval == 0 and args.visualize_features:
            # 使用验证集的一小部分进行可视化
            model.eval()
            all_vit_features = []
            all_resnet_features = []
            all_cross_attn_features = []
            all_labels = []

            with torch.no_grad():  # 不计算梯度，节省内存
                # 只使用少量样本
                sample_count = 0
                max_samples = 1500  # 最大样本数，GPU内存占用2000->3.3G

                for batch in val_loader:
                    if sample_count >= max_samples:
                        break

                    images = batch['image'].to(device)
                    labels = batch['label'].to(device)

                    # 获取特征
                    vit_feat, resnet_feat, cross_feat = model(images, return_features=True)

                    # 立即转移到CPU
                    all_vit_features.append(vit_feat.cpu())
                    all_resnet_features.append(resnet_feat.cpu())
                    all_cross_attn_features.append(cross_feat.cpu())
                    all_labels.append(labels.cpu())

                    sample_count += images.size(0)

                    # 清理GPU内存
                    del vit_feat, resnet_feat, cross_feat
                    torch.cuda.empty_cache()

            # 合并特征
            all_vit_features = torch.cat(all_vit_features, dim=0)
            all_resnet_features = torch.cat(all_resnet_features, dim=0)
            all_cross_attn_features = torch.cat(all_cross_attn_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # 可视化
            visualize_features(all_vit_features, all_labels, class_names, epoch + 1, args.save_dir, "ViT")
            visualize_features(all_resnet_features, all_labels, class_names, epoch + 1, args.save_dir, "ResNet")
            visualize_features(all_cross_attn_features, all_labels, class_names, epoch + 1, args.save_dir, "CrossAttn")

            # 清理内存
            del all_vit_features, all_resnet_features, all_cross_attn_features, all_labels
            torch.cuda.empty_cache()

            # 恢复训练模式
            model.train()

        # 进行每轮训练后的零样本分类评估
        print(f"进行第{epoch + 1}轮训练后的零样本分类评估...")
        zero_shot_acc= zero_shot_evaluation(model, class_names, val_loader, device)
        # 记录每轮的评估效果到TensorBoard
        writer.add_scalar('Accuracy/val_after_train', zero_shot_acc, epoch + 1)

        # 保存最佳模型
        if zero_shot_acc > best_zero_shot_acc:
            best_zero_shot_acc = zero_shot_acc
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_ViT_model.pth"))
            print(f"!!!最佳ViT模型已保存，准确率: {best_zero_shot_acc:.2f}%!!!")

    # 训练结束后的最终评估
    print("\n进行最终零样本分类评估...")
    final_accuracy = zero_shot_evaluation(model, class_names, val_loader, device)
    print(f"最终零样本准确率: {final_accuracy:.2f}%")
    print(f"最佳零样本准确率: {best_zero_shot_acc:.2f}%")

    # 关闭TensorBoard写入器
    writer.close()
    print("========训练结束========")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="训练混合注意力模型-PLANT")
    parser.add_argument("--vit_model", type=str, default="ViT-B-32", help="ViT模型名称")
    parser.add_argument("--resnet_model", type=str, default="RN50", help="ResNet模型名称")
    parser.add_argument("--pretrained", type=str, default="openai", help="使用的预训练模型")
    parser.add_argument("--batch_size", type=int, default=64, help="批量大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备 (cuda 或 cpu)")
    parser.add_argument("--save_dir", type=str, default="./saved_models", help="模型保存路径")
    parser.add_argument("--visualize_features", action="store_true", help="是否可视化特征")
    parser.add_argument("--vis_interval", type=int, default=5, help="特征可视化间隔")

    args = parser.parse_args()
    train(args)
