import torch.nn as nn

class CNN_LSTM_Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. CNN 特征提取层
        # 输入形状: (Batch, 1, 64, 64)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 输出: (32, 32, 32)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 输出: (64, 16, 16)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 输出: (128, 8, 8)
        )
        
        # 2. 桥接与 LSTM 层
        # CNN 最后的特征图是 (128, 8, 8)。
        # 我们将图像的“宽度” 8 看作序列长度（对应时间轴上的演化）。
        # 将 “通道数(128) * 高度(8)” 作为每个时间步的特征。
        self.lstm = nn.LSTM(
            input_size=128 * 8,  # 即 1024 个特征点
            hidden_size=256,     # 隐藏层维度，可以根据计算力调整
            num_layers=2,        # 堆叠两层 LSTM 增强非线性表达
            batch_first=True,
            dropout=0.3          # 防止过拟合
        )
        
        # 3. 分类层
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, 1, 64, 64)
        
        # --- CNN 阶段 ---
        features = self.cnn(x)  # (batch, 128, 8, 8)
        
        # --- 转换阶段 (Reshape for LSTM) ---
        # 目标形状: (batch, seq_len, input_size) -> (batch, 8, 128*8)
        batch_size, c, h, w = features.size()
        
        # 将宽度 w 设为序列长度，合并通道和高度
        # permute(0, 3, 1, 2) 之后形状为 (batch, 8, 128, 8)
        # view 之后形状为 (batch, 8, 1024)
        x = features.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, w, c * h)
        
        # --- LSTM 阶段 ---
        # out 形状: (batch, 8, 256)
        out, _ = self.lstm(x)
        
        # 取最后一个时间步的结果进行分类
        out = out[:, -1, :] # (batch, 256)
        
        # --- 分类阶段 ---
        logits = self.classifier(out)
        return logits



