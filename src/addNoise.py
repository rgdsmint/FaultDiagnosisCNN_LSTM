import numpy as np
import os
import matplotlib.pyplot as plt

# 配置路径
INPUT_DIR = 'dataset/processed/1d'
OUTPUT_DIR = 'dataset/processed/1d/withNoise'
SNR_DB = 0  # 噪声强度，0dB 代表噪声功率等于信号功率。数值越小噪声越大。

os.makedirs(OUTPUT_DIR, exist_ok=True)

def add_white_gaussian_noise(signal, snr_db):
    """
    为信号添加高斯白噪声
    """
    # 计算信号功率: Ps = sum(x^2) / N
    # 对 (1024, 4) 这种多通道数据，我们按通道计算功率
    shape = signal.shape
    signal_power = np.mean(signal**2, axis=0) # 4个通道的功率
    
    # 根据 SNR 公式计算噪声功率: Pn = Ps / 10^(SNR/10)
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # 生成噪声: 均值为 0，标准差为 sqrt(Pn)
    noise = np.random.normal(0, np.sqrt(noise_power), size=shape)
    
    return signal + noise

def process():
    print(f"开始添加噪声 (SNR = {SNR_DB}dB)...")
    
    for s in ['train', 'val', 'test']:
        x_path = os.path.join(INPUT_DIR, f'x_{s}.npy')
        y_path = os.path.join(INPUT_DIR, f'y_{s}.npy')
        
        if not os.path.exists(x_path):
            continue
            
        x_data = np.load(x_path)
        y_data = np.load(y_path)
        
        # 对每一个样本添加噪声
        x_noisy = np.array([add_white_gaussian_noise(sample, SNR_DB) for sample in x_data])
        
        # 保存到新目录
        np.save(os.path.join(OUTPUT_DIR, f'x_{s}.npy'), x_noisy.astype(np.float32))
        np.save(os.path.join(OUTPUT_DIR, f'y_{s}.npy'), y_data) # 标签保持不变
        
        print(f" - {s} 集已完成，保存至 {OUTPUT_DIR}")

    # --- 可视化对比 ---
    raw_sample = np.load(os.path.join(INPUT_DIR, 'x_train.npy'))[0]
    noisy_sample = np.load(os.path.join(OUTPUT_DIR, 'x_train.npy'))[0]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(raw_sample[:, 0], label='Raw X-Axis')
    plt.title("Original Signal (Clean)")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(noisy_sample[:, 0], label='Noisy X-Axis', color='orange')
    plt.title(f"Noisy Signal (SNR = {SNR_DB}dB)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    process()