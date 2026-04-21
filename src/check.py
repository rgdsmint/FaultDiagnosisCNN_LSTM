import numpy as np
import os

# 配置路径
SAVE_DIR = 'dataset/processed/1d'
EXPORT_DIR = 'dataset/processed/check_txt'
LABEL_NAMES = ['Healthy', 'Bearing_Fault', 'Bowed_Rotor', 'Broken_Bars', 'Misalignment', 'Unbalance']
CHANNELS = ['Vib_X', 'Vib_Y', 'Vib_Z', 'Sound']

os.makedirs(EXPORT_DIR, exist_ok=True)

def inspect_data():
    # 1. 加载训练集
    try:
        x_train = np.load(os.path.join(SAVE_DIR, 'x_train.npy'))
        y_train = np.load(os.path.join(SAVE_DIR, 'y_train.npy'))
    except FileNotFoundError:
        print("错误：找不到 .npy 文件，请确认 preprocess.py 运行成功且路径正确。")
        return

    print(f"✅ 成功加载训练集，形状为: {x_train.shape}")
    print("-" * 50)

    # 2. 打印第一个样本的片段
    sample_0 = x_train[0]
    print(f"样本 #0 (类别: {LABEL_NAMES[y_train[0]]}) 前 10 行数值预览：")
    print(f"{'Vib_X':>10} {'Vib_Y':>10} {'Vib_Z':>10} {'Sound':>10}")
    for row in sample_0[:10]:
        print(f"{row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f} {row[3]:10.6f}")
    print("...")
    print("-" * 50)

    # 3. 为每一类导出一个 TXT 文件
    print("正在为每一类故障导出一个样本到 TXT 文件...")
    for i in range(6):
        # 找到属于该类别的第一个样本
        idx = np.where(y_train == i)[0][0]
        sample = x_train[idx]
        
        file_name = f"Sample_{i}_{LABEL_NAMES[i]}.txt"
        save_path = os.path.join(EXPORT_DIR, file_name)
        
        # 使用制表符分隔保存，保留 6 位小数
        header = "\t".join(CHANNELS)
        np.savetxt(save_path, sample, fmt='%.6f', delimiter='\t', header=header)
        print(f" - 已保存: {file_name}")

    print(f"\n🚀 所有预览文件已保存在: {EXPORT_DIR}")

if __name__ == '__main__':
    inspect_data()