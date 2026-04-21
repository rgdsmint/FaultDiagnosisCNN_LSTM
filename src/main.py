import torch
import torch.nn as nn
import numpy as np
import os
import swanlab
from torch.utils.data import DataLoader, TensorDataset
from model import MotorNet
from train import train_one_epoch, validate, evaluate_test_set
import glob
import re
from sklearn.metrics import classification_report

def load_data(data_path):
    loaders = {}
    for s in ['train', 'val', 'test']:
        x = np.load(os.path.join(data_path, f'x_{s}.npy'))
        y = np.load(os.path.join(data_path, f'y_{s}.npy'))
        ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
        loaders[s] = DataLoader(ds, batch_size=config["batch_size"], shuffle=(s=='train'))
    return loaders

def main(save_dir='models', round=1):
    save_dir = os.path.join(save_dir, f'round{round}')
    os.makedirs(save_dir, exist_ok=True)
    # 加载带噪声的数据
    data_dir = 'dataset/processed/1d/withNoise'
    loaders = load_data(data_dir)
    
    model = MotorNet(num_classes=config["num_classes"]).to(config["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # 定义调度器：监控 val_acc，如果 10 轮不升，学习率乘以 0.5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max',       # 监控的是准确率，所以是 max
    factor=0.5,       # 衰减倍数
    patience=10,       # 容忍 10 轮不提升
)
    best_val_acc = 0.0
    saveRound = 0
    model_save_path = ""
    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_one_epoch(model, loaders['train'], criterion, optimizer, config["device"])
        val_loss, val_acc = validate(model, loaders['val'], criterion, config["device"])
        scheduler.step(val_acc)
        # 记录每轮指标
        swanlab.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
        })
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            saveRound += 1
            model_save_path = os.path.join(save_dir, f'best_model_{saveRound}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Epoch {epoch+1}: 验证集准确率提升至 {val_acc:.4f}，已保存最优模型。")

    print(f"训练完成，正在进行测试集最终评估...")

    # 加载最优权重进行测试
    model.load_state_dict(torch.load(model_save_path))
    test_acc, test_report, cm_path = evaluate_test_set(
        model, loaders['test'], config["device"], config["label_names"],save_path=os.path.join(save_dir, f'confusion_matrix_round{round}.png')
    )

    # 打印测试报告
    print(f"测试集正确率: {test_acc:.4f}")
    print(test_report)

    # 将测试结果和混淆矩阵上传至 SwanLab
    swanlab.log({
        "test_accuracy": test_acc,
        "confusion_matrix": swanlab.Image(cm_path, caption="Test Set Confusion Matrix")
    })
    
def model_evaluation(model_dir='models',round=1):
    # 1. 确定目标文件夹：例如 models/round1
    target_dir = os.path.join(model_dir, f'round{round}')
    
    if not os.path.exists(target_dir):
        print(f"❌ 找不到目录: {os.path.abspath(target_dir)}")
        return

    # 2. 获取该文件夹下的文件名（注意：这里必须用 target_dir 而不是 '.'）
    files = [f for f in os.listdir(target_dir) if f.startswith('best_model') and f.endswith('.pth')]

    if not files:
        print(f"❌ 在 {target_dir} 下未找到任何 best_model*.pth 文件")
        return

    # 3. 自然排序：提取文件名中的数字进行整数排序
    files.sort(key=lambda f: int(re.findall(r'\d+', f)[0]))
    
    print(f"✅ 成功加载 {len(files)} 个模型，顺序如下：\n{files}")

    # 4. 初始化模型和数据（只需一次）
    model = MotorNet(num_classes=config["num_classes"]).to(config["device"])
    test_loader = load_data('dataset/processed/1d/withNoise')['test']

    for f in files:
        # ⚠️ 加载时必须拼接路径：target_dir + 文件名
        model_path = os.path.join(target_dir, f)
        model.load_state_dict(torch.load(model_path, map_location=config["device"]))
        model.eval()
        
        all_true, all_pred = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs.to(config["device"]))
                _, predicted = torch.max(outputs, 1)
                all_true.extend(labels.numpy())
                all_pred.extend(predicted.cpu().numpy())

        acc = np.mean(np.array(all_true) == np.array(all_pred))
        report = classification_report(all_true, all_pred, target_names=config["label_names"], digits=4)
        
        print(f"\n▶️ 模型: {f} | 测试集准确率: {acc:.4f}")
        print(report)
        print("-" * 50)
   
if __name__ == "__main__":
    is_final_eval = True  # 设置为 True 以运行模型评估

    config = {
            "dataset": "HUSTmotor_Noisy",
            "snr": "0dB",
            "batch_size": 64,
            "learning_rate": 0.001,
            "epochs": 200,
            "num_classes": 6,
            "label_names": ['H', 'BF', 'BOW', 'BROKEN', 'MISAL', 'UNBAL'],
            "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    if is_final_eval:
        model_evaluation('models',round=2)
    else:
        round = int(open('round_config.txt').read().strip()) # 从配置文件读取当前轮次
        # 初始化 SwanLab
        swanlab.init(
            project="FaultDiagnosisCNN_LSTM",
            experiment_name=f"round{round}",
            config=config
        )
        print("使用设备:", config["device"])
        main(save_dir='models', round=round)
        swanlab.finish()
        open("round_config.txt", "w").write(str(int(round) + 1))