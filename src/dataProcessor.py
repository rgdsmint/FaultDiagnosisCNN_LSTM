import pandas as pd
import numpy as np
import torch

class SignalProcessor:
    def __init__(self):
        self.window_size = 1024 
        self.start_idx = 0
        self.fill_mode = '插值补全'

    def update_params(self, start_idx, fill_mode):
        self.start_idx = start_idx
        self.fill_mode = fill_mode

    def load_and_clean(self, file_path):
        df = pd.read_csv(file_path, sep=r'\s+', header=None)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        time_col = df.iloc[:, 0].values
        data_cols = df.iloc[:, 1:5].values 
        
        resets = np.where(np.diff(time_col) < 0)[0] + 1
        boundaries = np.concatenate(([0], resets, [len(df)]))
        return data_cols[boundaries[0]:boundaries[1]]

    def get_raw_and_padded(self, raw_signal):
        total_len = len(raw_signal)
        end_idx = self.start_idx + self.window_size
        
        if self.start_idx >= total_len:
            return None, None

        actual_end = min(end_idx, total_len)
        raw_seg = raw_signal[self.start_idx : actual_end]
        current_len = len(raw_seg)

        if current_len < self.window_size:
            if self.fill_mode == '插值补全':
                x_old = np.linspace(0, 1, current_len)
                x_new = np.linspace(0, 1, self.window_size)
                padded_seg = np.zeros((self.window_size, 4))
                for i in range(4):
                    padded_seg[:, i] = np.interp(x_new, x_old, raw_seg[:, i])
            elif self.fill_mode == '重复补全':
                padded_seg = np.pad(raw_seg, ((0, self.window_size - current_len), (0, 0)), mode='wrap')
            else: # 补零
                padding = np.zeros((self.window_size - current_len, 4))
                padded_seg = np.vstack([raw_seg, padding])
        else:
            padded_seg = raw_seg

        return raw_seg, padded_seg

    def get_model_input(self, padded_segment):
        seg_norm = (padded_segment - padded_segment.mean(axis=0)) / (padded_segment.std(axis=0) + 1e-9)
        return torch.tensor(seg_norm, dtype=torch.float32).unsqueeze(0)