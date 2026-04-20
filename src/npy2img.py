import numpy as np
import cv2
import os
from tqdm import tqdm

def save_npy_to_img(base_dir, output_dir):
    # 1. 确认输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建目录: {output_dir}")

    # 定义要处理的数据集类型
    sets = ['train', 'val', 'test']

    print("开始生成图片...")

    for s in sets:
        # 根据你指定的路径拼接
        x_path = os.path.join(base_dir, "2d", f"X_{s}_2d_mapped.npy")
        y_path = os.path.join(base_dir, "label", f"y_{s}.npy")

        if not os.path.exists(x_path):
            print(f"跳过 {s}：找不到文件 {x_path}")
            continue

        # 2. 加载数据
        x_data = np.load(x_path)  # (N, 64, 64)
        y_data = np.load(y_path)  # (N,)

        print(f"正在处理 {s} 集，共 {len(x_data)} 个样本...")

        # 3. 循环保存
        for i in tqdm(range(len(x_data))):
            img = x_data[i]
            label = y_data[i]

            # 转换为 uint8 格式（OpenCV 保存图片的要求）
            # 因为数据已经是 0-255 的 float32，直接 clip 然后转换即可
            img_uint8 = np.clip(img, 0, 255).astype(np.uint8)

            # 文件名格式：集合_序号_标签.png
            # 例如：train_00001_class3.png
            file_name = f"{s}_{i:05d}_class{label}.png"
            file_path = os.path.join(output_dir, file_name)

            # 保存图片
            cv2.imwrite(file_path, img_uint8)

    print(f"所有图片已保存至: {output_dir}")

if __name__ == "__main__":
    BASE_DIR = "dataset/processed"
    OUT_DIR = "dataset/processed/img"
    
    save_npy_to_img(BASE_DIR, OUT_DIR)