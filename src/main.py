import torch
from model import CNN_LSTM_Model
from train import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def main(): 
    # 1. 路径与配置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasetDir = "dataset/processed"   # 数据存放路径
    modelSavePath = "best_cnn_lstm.pth"
    batchSize = 64
    lr = 0.001
    epoch = 50
    numClasses = 10  # 假设为 10 类故障

    # 2. 获取数据加载器
    # 确保 data/processed 下已经有 X_train_2d_mapped.npy 等文件
    print("--- 正在初始化数据加载器 ---")
    train_loader, val_loader, test_loader = get_dataloaders(datasetDir, batch_size=batchSize)

    # 3. 初始化模型
    print("--- 正在构建 CNN-LSTM 网络 ---")
    model = CNN_LSTM_Model(num_classes=numClasses)
    model.to(device)

    # 4. 开始训练
    print("--- 开始训练流程 ---")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epoch,
        lr=lr,
        save_path=modelSavePath
    )

    validate(model, modelSavePath, test_loader)
    class_names = ['Normal', 'B_007', 'B_014', 'B_021', 'IR_007', 'IR_014', 'IR_021', 'OR_007', 'OR_014', 'OR_021']
    plot_confusion_matrix(model, test_loader, device, class_names)

def validate(model, modelSavePath, test_loader):
    # 5. 训练完成后，加载最优模型在【测试集】上做最终评估
    print("\n--- 正在进行最终测试集评估 ---")
    model.load_state_dict(torch.load(modelSavePath))
    model.eval()
    
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    final_acc = 100.0 * test_correct / test_total
    print(f"测试集最终准确率 (Test Accuracy): {final_acc:.2f}%")


def plot_confusion_matrix(model, test_loader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - Bearing Fault Diagnosis')
    plt.show()

if __name__ == "__main__":
    main()
    
