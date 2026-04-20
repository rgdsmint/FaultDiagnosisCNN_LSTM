import numpy as np
import pywt
import cv2
import os

# --- 1. 底层核心转换函数 ---
def _core_converter(input_dir, output_dir, img_size=(64, 64)):
    """
    底层核心逻辑：遍历任务并转换。
    该函数会被下面的无噪/有噪函数调用。
    """
    # 自动创建输出目录（在函数内部实现，满足你的要求）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"检测到目录缺失，已自动创建: {output_dir}")

    tasks = [
        ("X_train.npy", "X_train_2d_mapped.npy"),
        ("X_val.npy", "X_val_2d_mapped.npy"),
        ("X_test.npy", "X_test_2d_mapped.npy")
    ]

    scales = np.arange(1, 65)
    wavelet = 'cmor3-1.0'

    for in_name, out_name in tasks:
        input_file = os.path.join(input_dir, in_name)
        output_file = os.path.join(output_dir, out_name)

        if not os.path.exists(input_file):
            print(f"⚠️ 跳过：未找到输入文件 {input_file}")
            continue

        x_data = np.load(input_file)
        num_samples = x_data.shape[0]
        x_2d = np.zeros((num_samples, img_size[0], img_size[1]), dtype=np.float32)

        print(f"🚀 开始处理: {input_file} -> {output_file}")
        
        for i in range(num_samples):
            sig = x_data[i].flatten()
            coef, _ = pywt.cwt(sig, scales, wavelet)
            amp = np.abs(coef)
            
            # 归一化映射到 0-255
            amp_min, amp_max = amp.min(), amp.max()
            if amp_max > amp_min:
                amp = (amp - amp_min) / (amp_max - amp_min) * 255.0
            else:
                amp = np.zeros_like(amp)

            img = cv2.resize(amp, img_size, interpolation=cv2.INTER_CUBIC)
            x_2d[i] = img.astype(np.float32)

            if (i + 1) % 500 == 0:
                print(f"   进度: {i+1}/{num_samples}")

        np.save(output_file, x_2d)
        print(f"✅ 完成！保存至: {output_file}\n")


# --- 2. 专门处理无噪数据的函数 ---
def process_without_noise():
    """
    处理不带噪声的 1D -> 2D 转换
    """
    print("=== [任务] 开始处理：无噪数据 ===")
    in_dir = "dataset/processed/1d/withoutNoise"
    out_dir = "dataset/processed/2d/withoutNoise"
    _core_converter(in_dir, out_dir)


# --- 3. 专门处理有噪数据的函数 ---
def process_with_noise():
    """
    处理带噪声的 1D -> 2D 转换
    """
    print("=== [任务] 开始处理：有噪数据 ===")
    in_dir = "dataset/processed/1d/withNoise"
    out_dir = "dataset/processed/2d/withNoise"
    _core_converter(in_dir, out_dir)


# --- 4. 运行入口 ---
if __name__ == "__main__":
    # 你可以根据需要决定运行哪个，或者两个都运行
    
    # 运行无噪转换
    process_without_noise()
    process_with_noise()
    # 运行有噪转换（前提是你已经运行了 addNoise.py 生成了 1d/withNoise 文件）
    # process_with_noise()