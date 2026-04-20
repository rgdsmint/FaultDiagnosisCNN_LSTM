import numpy as np
import pywt
import cv2
import os

def convert_1d_to_2d_mapped_float(input_file, output_file, img_size=(64, 64)):
    """
    将1D信号转换为2D时频图，映射到 0-255 范围，但保持 float32 精度
    """
    if not os.path.exists(input_file):
        print(f"跳过：未找到文件 {input_file}")
        return

    # 1. 加载 1D 数据
    x_data = np.load(input_file)
    num_samples = x_data.shape[0]
    
    scales = np.arange(1, 65)
    wavelet = 'cmor3-1.0'
    
    # 初始化输出矩阵 [N, 64, 64]
    x_2d = np.zeros((num_samples, img_size[0], img_size[1]), dtype=np.float32)

    print(f"开始转换并映射至 0-255 (float32): {input_file} ...")
    
    for i in range(num_samples):
        sig = x_data[i].flatten()
        
        # 连续小波变换 (CWT)
        coef, _ = pywt.cwt(sig, scales, wavelet)
        amp = np.abs(coef) # 取模
        
        # 2. 映射到 0-255 范围 (Min-Max Scaling)
        amp_min, amp_max = amp.min(), amp.max()
        if amp_max > amp_min:
            # 这里的计算结果依然是 float32
            amp = (amp - amp_min) / (amp_max - amp_min) * 255.0
        else:
            amp = np.zeros_like(amp) # 防止全 0 信号导致除零错误

        # 3. 调整尺寸
        img = cv2.resize(amp, img_size, interpolation=cv2.INTER_CUBIC)
        x_2d[i] = img.astype(np.float32)

        if (i + 1) % 500 == 0:
            print(f"进度: {i+1}/{num_samples}")

    # 4. 保存文件
    np.save(output_file, x_2d)
    print(f"✅ 处理完成！已保存高精度映射数据至: {output_file}")
    print(f"📊 检查：Max={x_2d.max():.2f}, Min={x_2d.min():.2f}, Dtype={x_2d.dtype}")

if __name__ == "__main__":
    processed_dir = "dataset/processed"
    tasks = [
        ("X_train.npy", "X_train_2d_mapped.npy"),
        ("X_val.npy", "X_val_2d_mapped.npy"),
        ("X_test.npy", "X_test_2d_mapped.npy")
    ]
    
    for in_name, out_name in tasks:
        convert_1d_to_2d_mapped_float(
            os.path.join(processed_dir, in_name),
            os.path.join(processed_dir, out_name)
        )