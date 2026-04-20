import numpy as np
import os

def add_gaussian_noise_vectorized(x_data, snr_db=0):
    """
    对整个 numpy 数组添加高斯白噪声
    :param x_data: 形状为 (N, 1, 1024) 的 1D 信号数组
    :param snr_db: 信噪比 (dB)，0 表示噪声和信号等强，10 表示信号强，-5 表示噪声极强
    """
    # 转换为 float64 进行计算以防溢出，计算完再转回 float32
    x_data = x_data.astype(np.float64)
    
    # 计算每个样本的信号功率 (针对 1024 个点)
    # axis=(1,2) 对应 (1, 1024) 部分
    sig_power = np.mean(x_data**2, axis=(1, 2), keepdims=True)
    
    # 根据 SNR 公式计算噪声功率: P_noise = P_signal / 10^(SNR/10)
    noise_power = sig_power / (10**(snr_db / 10.0))
    
    # 生成高斯噪声，形状与 x_data 一致
    noise = np.random.normal(0, np.sqrt(noise_power), x_data.shape)
    
    return (x_data + noise).astype(np.float32)

def process_noise_task(input_dir, output_dir, snr_db=0):
    # 1. 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建新目录: {output_dir}")

    # 2. 定义要处理的文件
    files_to_process = ["X_train.npy", "X_val.npy", "X_test.npy"]

    print(f"--- 开始添加噪声 (SNR: {snr_db}dB) ---")
    
    for filename in files_to_process:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if not os.path.exists(input_path):
            print(f"跳过：找不到文件 {input_path}")
            continue

        # 加载无噪原始数据
        print(f"正在读取: {filename}...")
        x_clean = np.load(input_path)

        # 添加噪声
        x_noised = add_gaussian_noise_vectorized(x_clean, snr_db)

        # 保存到新路径 (不会覆盖原文件，因为目录不同)
        np.save(output_path, x_noised)
        print(f"✅ 已保存带噪文件至: {output_path}")

if __name__ == "__main__":
    # 路径配置
    INPUT_DIR = "dataset/processed/1d/withoutNoise"
    OUTPUT_DIR = "dataset/processed/1d/withNoise"
    
    # 设置信噪比
    # 0dB: 强噪声环境（适合展示模型强大的鲁棒性）
    # 6dB: 中等噪声
    # -5dB: 极端恶劣环境
    target_snr = 0 
    
    process_noise_task(INPUT_DIR, OUTPUT_DIR, snr_db=target_snr)
    print("\n所有 1D 数据加噪任务已完成！")