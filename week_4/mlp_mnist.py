import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Data Preparation
# Define transformations for the training and test sets
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to PyTorch Tensor 将像素值从 [0, 255] 映射到 [0.0, 1.0] 的浮点数
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the dataset 𝑥norm=𝑥−𝜇/𝜎标准化为0均值 单位方差
    ])

# Download and load the training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True,
                               transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False,
                              transform=transform, download=True)

# Create data loaders
# train_loader 每调用一次返回一个新的迭代对象  迭代随机分成的多个batch
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# 2. Model Construction
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(28*28, 32)  # Input layer to hidden layer
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, 10)     # Hidden layer to output layer
        self.softmax = nn.LogSoftmax(dim=1)  # Use LogSoftmax for numerical stability

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

model = MLP()
#the number of parameters in the shallow network is 32*784+32+10*32+10=25,450
#the number of parameters in the deep network is 30*784+30+28*30+28+26*28+26+10*26+10=25442
#deep network
class DMLP(nn.Module):
    def __init__(self):
        super(DMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(784, 30)
        self.l2 = nn.Linear(30, 28)
        self.l3 = nn.Linear(28, 26)
        self.output = nn.Linear(26, 10)     # Hidden layer to output layer
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)  # Use LogSoftmax for numerical stability

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.output(x)
        x = self.softmax(x)
        return x

deep_model = DMLP()
# 3. Model Compilation
#它内部会自动对 output 做 log_softmax,然后用 target 的整数索引来选取对应类别的概率
criterion = nn.NLLLoss()  # Negative Log Likelihood Loss (used with LogSoftmax)
optimizer = optim.SGD(model.parameters(), lr=0.01)# lr learning rate
deep_optimizer = optim.SGD(deep_model.parameters(), lr=0.01)# lr learning rate

# 4. Model Training
epochs = 20
train_losses = []
valid_losses = []
deep_train_losses = []
train_accuracy = []
deep_train_accuracy = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0

    for data, target in train_loader:
        optimizer.zero_grad()# 清除旧梯度 上一个batch
        output = model(data)
        loss = criterion(output, target)  # target is not one-hot encoded in PyTorch one-hot用一个长度为类别总数的向量表示每个类别，其中只有一个位置是 1，其余都是 0。 output: logits
        loss.backward()# 反向传播，计算损失对参数的梯度
        optimizer.step()# 更新参数

        epoch_loss += loss.item()
        #沿着第 1 维（即每行）找出最大值的索引,也就是每个样本预测得分最高的类别索引 keepdim保留原来的维度结构
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()#当前训练的正确样本数

    train_losses.append(epoch_loss / len(train_loader))#平均训练损失
    train_accuracy.append(100. * correct / len(train_loader.dataset)) 

    print(f'Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracy[-1]:.2f}%')


for epoch in range(epochs):
    deep_model.train()
    epoch_loss = 0
    correct = 0

    for data, target in train_loader:
        deep_optimizer.zero_grad()# 清除旧梯度 上一个batch
        output = deep_model(data)
        loss = criterion(output, target)  # target is not one-hot encoded in PyTorch one-hot用一个长度为类别总数的向量表示每个类别，其中只有一个位置是 1，其余都是 0。 output: logits
        loss.backward()# 反向传播，计算损失对参数的梯度
        deep_optimizer.step()# 更新参数

        epoch_loss += loss.item()
        #沿着第 1 维（即每行）找出最大值的索引,也就是每个样本预测得分最高的类别索引 keepdim保留原来的维度结构
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()#当前训练的正确样本数

    deep_train_losses.append(epoch_loss / len(train_loader))#平均训练损失
    deep_train_accuracy.append(100. * correct / len(train_loader.dataset))

    print(f'Epoch {epoch+1}/{epochs}, Loss: {deep_train_losses[-1]:.4f}, Accuracy: {deep_train_accuracy[-1]:.2f}%')

#compare the train process between shallow and deep network
# 创建图形
plt.figure(figsize=(10, 6))
plt.plot(list(range(1, epochs+1)), train_losses, label='Shallow MLP', marker='o')
plt.plot(list(range(1, epochs+1)), deep_train_losses, label='Deep MLP', marker='s')
plt.xticks(range(1, epochs+1))  # 强制 x 轴显示整数刻度
# 添加标题和标签
plt.title('Loss vs Epoch for Two Models')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 显示图形
plt.show()
# 5. Model Evaluation
model.eval()
test_loss = 0
correct = 0
#with 是 Python 中的上下文管理器（context manager）语法，它的作用是：
#在一段代码块执行前后自动处理资源的初始化和清理工作。
#在这个代码块中，不要追踪梯度计算（no gradient tracking）
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader)
test_accuracy = 100. * correct / len(test_loader.dataset)

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')





