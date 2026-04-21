import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# 配置
RAW_DIR = 'dataset/raw_data'
SAVE_DIR = 'dataset/processed/1d'
WINDOW_SIZE = 1024  # 如果觉得 5Hz 效果不好可改为 2048
LABEL_MAP = {'H': 0, 'BF': 1, 'BOW': 2, 'BROKEN': 3, 'MISAL': 4, 'UNBAL': 5}

os.makedirs(SAVE_DIR, exist_ok=True)

def preprocess():
    sets = {'train': [[], []], 'val': [[], []], 'test': [[], []]}
    files = [f for f in os.listdir(RAW_DIR) if f.endswith('.txt')]
    
    print(f"开始处理 {len(files)} 个文件...")

    for file_name in tqdm(files):
        label = LABEL_MAP[file_name.split('_')[0]]
        df = pd.read_csv(os.path.join(RAW_DIR, file_name), skiprows=18, sep='\s+', header=None)
        
        time_col = df.iloc[:, 0].values
        data_cols = df.iloc[:, 1:5].values # X, Y, Z, Sound
        
        # 自动检测 5 段数据的重置点
        resets = np.where(np.diff(time_col) < 0)[0] + 1
        boundaries = np.concatenate(([0], resets, [len(df)]))
        
        for i in range(len(boundaries) - 1):
            # 提取单段 (Trial)
            seg = data_cols[boundaries[i]:boundaries[i+1]]
            
            # 分通道归一化 (Z-score)
            seg = (seg - seg.mean(axis=0)) / (seg.std(axis=0) + 1e-9)
            
            # 0 重叠切片
            num_samples = len(seg) // WINDOW_SIZE
            samples = [seg[s*WINDOW_SIZE : (s+1)*WINDOW_SIZE] for s in range(num_samples)]
            
            # 按照 Group 划分集合
            if i < 3:    target = 'train' # Group 1,2,3
            elif i == 3: target = 'val'   # Group 4
            else:        target = 'test'  # Group 5
            
            sets[target][0].extend(samples)
            sets[target][1].extend([label] * len(samples))

    # 保存为 npy
    for s in ['train', 'val', 'test']:
        np.save(f'{SAVE_DIR}/x_{s}.npy', np.array(sets[s][0], dtype=np.float32))
        np.save(f'{SAVE_DIR}/y_{s}.npy', np.array(sets[s][1], dtype=np.int64))
    
    print(f"预处理完成！训练集: {len(sets['train'][1])}, 验证集: {len(sets['val'][1])}, 测试集: {len(sets['test'][1])}")

if __name__ == '__main__':
    preprocess()