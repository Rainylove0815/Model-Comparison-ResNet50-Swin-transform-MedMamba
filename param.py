import torch
import time
from thop import profile
from model.MobileNetV2 import MobileNetV2
from model.ShuffleNetV2 import shufflenet_v2_x2_0
from model.ResNet import resnet50
from model.Swin_Transformer import swin_tiny_patch4_window7_224,swin_small_patch4_window7_224,swin_base_patch4_window7_224


def load_model(num_classes=4):
    model = resnet50(num_classes=4)
    # in_features = model.classifier[1].in_features
    # model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load('model_pth/brain_tumor_pth/resnet50.pth', map_location='cpu'))
    return model.eval()


def get_test_image():
    # 尝试加载真实图像或创建随机张量
    try:
        # 这里可以替换为你的真实图像加载代码
        # 例如: from PIL import Image; image = Image.open('test.jpg')
        image_tensor = torch.randn(1, 3, 224, 224)
    except:
        # 备用方案：创建随机张量
        image_tensor = torch.randn(1, 3, 224, 224)

    return image_tensor


def calculate_metrics(model, input_tensor, device='cuda', repeats=100):
    # 自动修复可能的形状问题
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)  # 添加批次维度
    if input_tensor.shape[1] == 1:
        input_tensor = input_tensor.repeat(1, 3, 1, 1)  # 灰度转RGB

    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # 计算参数量
    params = sum(p.numel() for p in model.parameters())

    # 计算FLOPs
    flops, _ = profile(model, inputs=(input_tensor,), verbose=False)

    # 预热
    for _ in range(10):
        _ = model(input_tensor)

    # 计算推理时间
    start = time.time()
    for _ in range(repeats):
        _ = model(input_tensor)
        if device == 'cuda':
            torch.cuda.synchronize()  # 确保CUDA操作完成

    latency = (time.time() - start) / repeats * 1000  # 毫秒

    return params, flops, latency


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    model = load_model(num_classes=4)
    test_image = get_test_image()

    print(f"输入张量形状: {test_image.shape}")

    params, flops, latency = calculate_metrics(model, test_image, device)

    print(f"\n参数量: {params:,} ({params / 1e6:.2f}M)")
    print(f"FLOPs: {flops:,.0f} ({flops / 1e9:.2f}G)")
    print(f"单张图像推理延迟: {latency:.2f} ms")
    print(f"推理速度: {1000 / latency:.1f} FPS")


