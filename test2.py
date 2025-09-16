import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from MedMamba import VSSM
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import os

# 设置随机种子保证可复现性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 测试集预处理（需与训练时一致）
test_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载测试集
testset = datasets.ImageFolder(root=os.path.join("data_brain_tumor", "Testing"),
                               transform=test_transformer)
test_loader = DataLoader(testset, batch_size=16, shuffle=False)

# 类别名称（根据实际修改）
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']  # 要按照文件夹的顺序

# 加载训练好的模型
model = VSSM(depths=[2, 2, 12, 2],dims=[128,256,512,1024],num_class=4).to(device)
model.load_state_dict(torch.load("medmamba_b.pth"))
model.eval()

# 存储所有预测结果
all_preds = []
all_labels = []
all_probs = []

# 禁用梯度计算以加速推理
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

all_probs = np.concatenate(all_probs, axis=0)


# 1. 混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 8))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.savefig('plots/confusion_matrix.png')
    plt.close()


cm = confusion_matrix(all_labels, all_preds)
plot_confusion_matrix(cm, class_names, normalize=True)

# 2. 分类报告（Precision/Recall/F1）
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print("Classification Report:\n", report)

# 保存分类报告
with open('plots/classification_report.txt', 'w') as f:
    f.write(report)


# 3. ROC曲线和AUC值（多分类）
def plot_multiclass_roc(y_true, y_prob, classes):
    plt.figure(figsize=(10, 8))

    # 二值化标签
    y_true_bin = label_binarize(y_true, classes=np.arange(len(classes)))

    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2,
                 label='{0} (AUC = {1:0.2f})'
                       ''.format(classes[i], roc_auc[i]))

    # 计算微平均ROC曲线和AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average (AUC = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('plots/multiclass_roc.png')
    plt.close()

    return roc_auc


roc_auc = plot_multiclass_roc(all_labels, all_probs, class_names)

# 保存AUC值
auc_df = pd.DataFrame.from_dict(roc_auc, orient='index', columns=['AUC'])
auc_df.to_csv('plots/auc_values.csv')
print("AUC values saved to plots/auc_values.csv")


# 4. 各类别关键指标可视化
def plot_metrics_per_class(report_dict):
    metrics = ['precision', 'recall', 'f1-score']
    classes = class_names
    data = {m: [] for m in metrics}

    for cls in classes:
        for m in metrics:
            data[m].append(report_dict[cls][m])

    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(12, 6))
    for i, (m, vals) in enumerate(data.items()):
        plt.bar(x + i * width, vals, width, label=m)

    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Metrics by Class')
    plt.xticks(x + width, classes, rotation=45)
    plt.legend(loc='lower right')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('plots/metrics_per_class.png')
    plt.close()


# 将分类报告转为字典
report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
plot_metrics_per_class(report_dict)

print("所有评估结果已保存到plots目录")