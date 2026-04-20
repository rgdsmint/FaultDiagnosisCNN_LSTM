import scipy.io
import numpy as np
import os
from sklearn.model_selection import train_test_split

def prepare_data_flexible(root_dir, save_dir):
    X_all, y_all = [], []
    # 显式初始化 Normal 为 0
    label_map = {"Normal": 0}
    current_label = 1

    print("--- 开始扫描文件夹 ---")
    
    # 1. 第一次遍历：确定所有包含 .mat 的路径并建立标签映射
    for root, dirs, files in os.walk(root_dir):
        mat_files = [f for f in files if f.endswith('.mat')]
        if not mat_files:
            continue
            
        # 获取相对路径，例如 "Normal" 或 "B/007"
        rel_path = os.path.relpath(root, root_dir)
        
        # 统一命名逻辑
        if rel_path == "." or "Normal" in rel_path:
            cat_name = "Normal"
        else:
            # 将路径分隔符替换为下划线，如 B/007 -> B_007
            cat_name = rel_path.replace(os.sep, "_")
            
        if cat_name not in label_map:
            label_map[cat_name] = current_label
            current_label += 1
            
        target_label = label_map[cat_name]
        print(f"检测到类别: {cat_name.ljust(15)} | 标签: {target_label} | 文件数: {len(mat_files)}")

        # 2. 读取数据并切片
        for mat_name in mat_files:
            file_path = os.path.join(root, mat_name)
            try:
                data = scipy.io.loadmat(file_path)
                signals = None
                for key in data.keys():
                    if 'DE_time' in key:
                        signals = data[key].flatten()
                        break
                
                if signals is not None:
                    # 长度 1024，步长 512 (50% 重叠)
                    for i in range(0, len(signals) - 1024, 512):
                        X_all.append(signals[i : i + 1024])
                        y_all.append(target_label)
            except Exception as e:
                print(f"无法读取 {mat_name}: {e}")

    # 3. 转换
    X_all = np.array(X_all, dtype=np.float32)[:, np.newaxis, :]
    y_all = np.array(y_all, dtype=np.int64)

    print(f"\n--- 准备划分数据集 (7:2:1) ---")
    
    # 第一步：分出 70% 的训练集 和 30% 的临时集(包含验证和测试)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.3, random_state=42, shuffle=True, stratify=y_all
    )
    
    # 第二步：将 30% 的临时集按 2:1 的比例划分为验证集和测试集
    # 这里 test_size 设为 1/3，即占临时集的 33.33% (也就是整体的 10%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1/3, random_state=42, shuffle=True, stratify=y_temp
    )

    # 4. 保存文件
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "X_val.npy"), X_val)
    np.save(os.path.join(save_dir, "y_val.npy"), y_val)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)
    
    # 保存标签映射
    with open(os.path.join(save_dir, "labels.txt"), "w") as f:
        for name, lab in sorted(label_map.items(), key=lambda x: x[1]):
            f.write(f"{lab}:{name}\n")

    print(f"\n✅ 处理完成！共 {len(label_map)} 类。")
    print(f"总样本数: {len(X_all)}")
    print(f"训练集 (70%): {len(X_train)} 样本")
    print(f"验证集 (20%): {len(X_val)} 样本")
    print(f"测试集 (10%): {len(X_test)} 样本")
    print(f"数据已保存至: {save_dir} 目录下")

if __name__ == "__main__":
    # 路径确保正确
    prepare_data_flexible("dataset/12k_Drive_End_Bearing_Fault_Data", "data/processed")