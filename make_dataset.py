import os
import shutil
import random
from sklearn.model_selection import train_test_split


def split_dataset(dataset_path, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    划分数据集为训练集、验证集和测试集

    参数:
    dataset_path: 原始数据集路径
    output_path: 划分后数据集的输出路径
    train_ratio: 训练集比例
    val_ratio: 验证集比例
    test_ratio: 测试集比例
    seed: 随机种子，确保可重复性
    """
    # 设置随机种子
    random.seed(seed)

    # 创建输出目录
    train_dir = os.path.join(output_path, 'train')
    val_dir = os.path.join(output_path, 'val')
    test_dir = os.path.join(output_path, 'test')

    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
        for class_name in ['ageDegeneration','cataract','diabetes','glaucoma','hypertension','myopia','normal','others']:
            os.makedirs(os.path.join(dir_path, class_name), exist_ok=True)

    # 遍历每个类别
    for class_name in ['ageDegeneration','cataract','diabetes','glaucoma','hypertension','myopia','normal','others']:
        class_path = os.path.join(dataset_path, class_name)

        # 获取该类别的所有图像文件
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        random.shuffle(images)  # 随机打乱

        # 计算各集合的数量
        total_count = len(images)
        train_count = int(total_count * train_ratio)
        val_count = int(total_count * val_ratio)
        test_count = total_count - train_count - val_count

        print(f"类别 {class_name}: 共 {total_count} 张图像")
        print(f"  训练集: {train_count} 张, 验证集: {val_count} 张, 测试集: {test_count} 张")

        # 划分数据集
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]

        # 复制图像到对应目录
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_dir, class_name, img)
            shutil.copy2(src, dst)

        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_dir, class_name, img)
            shutil.copy2(src, dst)

        for img in test_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(test_dir, class_name, img)
            shutil.copy2(src, dst)

    print("\n数据集划分完成！")
    print(f"训练集路径: {train_dir}")
    print(f"验证集路径: {val_dir}")
    print(f"测试集路径: {test_dir}")


# 使用示例
if __name__ == "__main__":
    # 设置路径
    dataset_path = "Data/ODIR5K/datasets"  # 替换为您的数据集路径
    output_path = "Data/ODIR5K_spilt"  # 替换为您希望保存划分后数据的路径

    # 划分数据集 (70% 训练, 15% 验证, 15% 测试)
    split_dataset(dataset_path, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)