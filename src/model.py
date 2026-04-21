import torch
import torch.nn as nn

class MotorNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64)
        )
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2) # (B, 1024, 4) -> (B, 4, 1024)
        x = self.conv_block(x)
        x = x.transpose(1, 2) # (B, 128, 64) -> (B, 64, 128)
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])