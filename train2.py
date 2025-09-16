import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from MedMamba import VSSM


def set_seed(seed=42):
    random.seed(seed)  # Python的随机种子
    np.random.seed(seed)  # NumPy的随机种子
    torch.manual_seed(seed)  # PyTorch的CPU随机种子
    torch.cuda.manual_seed(seed)  # PyTorch的GPU随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True  # 确保CUDA卷积操作 deterministic
    torch.backends.cudnn.benchmark = False  # 关闭benchmark优化以保证可复现性


set_seed(42)  # 设置随机种子为42（可以改为其他你喜欢的数字）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建保存图表和模型的目录
os.makedirs("model_pth", exist_ok=True)
os.makedirs("plots", exist_ok=True)

train_transformer = transforms.Compose([  # 如果数据少就添加数据增强
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转 - 大脑是左右对称的，适合
    transforms.RandomVerticalFlip(p=0.3),  # 垂直翻转 - 谨慎使用，概率较低
    transforms.RandomRotation(degrees=(-10, 10)),  # 小角度旋转
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 轻微调整亮度和对比度
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.ImageFolder(root=os.path.join(r"data_brain_tumor", "Training"),
                                transform=train_transformer)
valset = datasets.ImageFolder(root=os.path.join(r"data_brain_tumor", "Validation"),
                              transform=val_transformer)

train_loader = DataLoader(trainset, batch_size=16, num_workers=0, shuffle=True)
val_loader = DataLoader(valset, batch_size=16, num_workers=0, shuffle=False)


def train(model, train_loader, criterion, optimizer, num_epochs):
    # 初始化记录器
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_acc = 0.0

    # 早停参数
    patience = 20  # 允许性能没有提升的epoch数
    early_stop_counter = 0  # 计数器
    best_epoch = 0  # 记录最佳epoch
    stop_training = False  # 停止训练标志

    for epoch in range(num_epochs):
        if stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}. Best epoch was {best_epoch + 1}")
            break

        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch:{epoch + 1}/{num_epochs}", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        # 计算并记录训练损失
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # 评估验证集
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            save_model(model, save_path)
            early_stop_counter = 0  # 重置计数器
            print(f"Model saved with best accuracy: {best_acc:.2f}%")
        else:
            early_stop_counter += 1
            print(f"No improvement for {early_stop_counter}/{patience} epochs")

            # 检查是否达到早停条件
            if early_stop_counter >= patience:
                stop_training = True
                print(f"Validation accuracy hasn't improved for {patience} consecutive epochs.")

    # 训练结束后绘制图表
    plot_metrics(train_losses, val_losses, val_accuracies, len(train_losses))  # 使用实际训练的epoch数

    return train_losses, val_losses, val_accuracies, best_epoch


def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = val_loss / len(val_loader.dataset)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def plot_metrics(train_losses, val_losses, val_accuracies, num_epochs):
    epochs = range(1, num_epochs + 1)

    # 创建图表
    plt.figure(figsize=(15, 5))

    # 损失曲线图
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 准确度曲线图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # 保存图表
    plt.tight_layout()
    plt.savefig('plots/training_metrics.png')  # 曲线图的名字改成自己的
    plt.close()

    print("Training metrics plot saved to 'plots/training_metrics.png'")  # 曲线图的名字改成自己的


if __name__ == "__main__":
    num_epochs = 35  # 训练轮次  我一般用25轮，如果模型训练效果不好就50轮
    lr = 0.0005
    num_class = 4  # 类别数
    save_path = r"medmamba_b.pth"  # 添加文件扩展名 .pth
    model = VSSM(depths=[2, 2, 12, 2],dims=[128,256,512,1024],num_class=num_class).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练模型并获取指标
    train_losses, val_losses, val_accuracies, best_epoch = train(model, train_loader, criterion, optimizer, num_epochs)

    # 最终评估
    final_val_loss, final_val_acc = evaluate(model, val_loader, criterion)
    print(f"\nFinal Validation Loss: {final_val_loss:.4f}, Accuracy: {final_val_acc:.2f}%")
    print(f"Best validation accuracy: {max(val_accuracies):.2f}% at epoch {best_epoch + 1}")