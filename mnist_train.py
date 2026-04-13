import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 设备
device = torch.device("cpu")

# 2. 超参数
batch_size = 64
learning_rate = 0.001
epochs = 5

# 3. 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转成张量，并归一化到 [0,1]
])

# 4. 下载 MNIST 数据集
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

# 5. DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 6. 定义模型
class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),          # (1, 28, 28) -> 784
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)      # 输出 10 类
        )

    def forward(self, x):
        return self.model(x)

model = DigitNet().to(device)

# 7. 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 8. 训练函数
def train():
    model.train()
    total_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Train Loss: {avg_loss:.4f}")

# 9. 测试函数
def test():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# 10. 开始训练
for epoch in range(epochs):
    print(f"\nEpoch [{epoch + 1}/{epochs}]")
    train()
    test()

# 11. 保存模型
torch.save(model.state_dict(), "mnist_model.pth")
print("\nModel saved as mnist_model.pth")
