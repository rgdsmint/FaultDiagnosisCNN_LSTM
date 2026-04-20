import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from model import CNN_LSTM_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")



class BearingDataset(Dataset):
    def __init__(self, x_path, y_path):
        """
        x_path: .npy 文件的路径 (N, 64, 64)
        y_path: 标签文件的路径 (N,)
        """
        # 加载数据
        print(f"正在从 {x_path} 加载数据...")
        self.x_data = np.load(x_path)
        self.y_data = np.load(y_path)
        
        # 确保数据类型正确 (float32)
        self.x_data = self.x_data.astype(np.float32)
        # 标签通常需要是 long 类型 (int64)
        self.y_data = self.y_data.astype(np.int64)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        # 1. 提取单条样本
        img = self.x_data[idx]
        label = self.y_data[idx]
        
        # 2. 增加通道轴: (64, 64) -> (1, 64, 64)
        img = img[np.newaxis, :, :]
        
        # 3. 归一化: 从 0-255 映射到 0.0-1.0
        img = img / 255.0
        
        # 4. 转换为 Tensor
        img_tensor = torch.from_numpy(img)
        label_tensor = torch.from_numpy(np.array(label)) # 标量转 Tensor
        
        return img_tensor, label_tensor


def get_dataloaders(data_dir, batch_size=32):
    # 实例化三个 Dataset
    train_dataset = BearingDataset(
        os.path.join(data_dir, "2d/X_train_2d_mapped.npy"),
        os.path.join(data_dir, "label/y_train.npy")
    )
    val_dataset = BearingDataset(
        os.path.join(data_dir, "2d/X_val_2d_mapped.npy"),
        os.path.join(data_dir, "label/y_val.npy")
    )
    test_dataset = BearingDataset(
        os.path.join(data_dir, "2d/X_test_2d_mapped.npy"),
        os.path.join(data_dir, "label/y_test.npy")
    )

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, save_path="best_model.pth"):
    # 2. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss() # 多分类标准选择
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 学习率衰减：如果 5 个 epoch 验证集 loss 不降，则 lr 减半
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    model.to(device)
    best_val_acc = 0.0

    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 使用 tqdm 显示进度条
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()      # 梯度清零
            outputs = model(inputs)     # 前向传播
            loss = criterion(outputs, labels)
            loss.backward()            # 反向传播
            optimizer.step()           # 权重更新
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = 100.0 * train_correct / train_total

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad(): # 验证阶段不计算梯度
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = 100.0 * val_correct / val_total
        
        # 更新学习率
        scheduler.step(epoch_val_loss)

        # 打印统计数据
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")

        # 保存表现最好的模型
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), save_path)
            print(f"⭐ 发现更好的模型，已保存至 {save_path}")

    print(f"训练完成！最佳验证集准确率: {best_val_acc:.2f}%")



