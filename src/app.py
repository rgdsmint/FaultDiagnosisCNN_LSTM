import sys
import torch
import numpy as np
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QHBoxLayout, 
                             QFrame, QSpinBox, QComboBox, QFormLayout, QTextEdit, QSlider)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

from model import MotorNet
from dataProcessor import SignalProcessor

LABEL_MAP = {0: '正常 (H)', 1: '轴承故障 (BF)', 2: '转子弯曲 (BOW)', 
             3: '转子断条 (BROKEN)', 4: '对中不良 (MISAL)', 5: '电压不稳 (UNBAL)'}

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("电机故障智能诊断系统 - MotorNet 控制台")
        self.resize(1400, 950)

        self.model = None
        self.raw_full_data = None 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SignalProcessor()

        self.init_ui()
        self.log("<b>系统就绪</b>，请加载模型及数据...")

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_widget = QWidget()
        left_panel = QVBoxLayout(left_widget)
        
        self.btn_load_model = QPushButton("📂 1. 加载 MotorNet 权重")
        self.btn_load_model.setFixedHeight(40)
        self.btn_load_model.clicked.connect(self.on_load_model)
        
        self.btn_load_data = QPushButton("📄 2. 加载全量原始数据")
        self.btn_load_data.setFixedHeight(40)
        self.btn_load_data.clicked.connect(self.on_load_data)
        
        params_group = QFrame()
        params_group.setFrameShape(QFrame.StyledPanel)
        params_layout = QFormLayout(params_group)
        self.spin_start = QSpinBox()
        self.spin_start.setRange(0, 1000000)
        self.spin_start.valueChanged.connect(self.on_param_changed)
        self.combo_fill = QComboBox()
        self.combo_fill.addItems(["插值补全", "重复补全", "补零填充"])
        params_layout.addRow("起始索引:", self.spin_start)
        params_layout.addRow("补全逻辑:", self.combo_fill)
        
        self.btn_diagnose = QPushButton("🚀 3. 运行算法诊断")
        self.btn_diagnose.setFixedHeight(50)
        self.btn_diagnose.setStyleSheet("background-color: #ffffff; color: #2d3436; font-weight: bold; border: 2px solid #00b894;")
        self.btn_diagnose.clicked.connect(self.on_diagnose)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("""
            background-color: #ffffff; 
            color: #2d3436; 
            font-family: 'Microsoft YaHei'; 
            font-size: 13px;
            border: 1px solid #dfe6e9;
        """)

        left_panel.addWidget(self.btn_load_model)
        left_panel.addWidget(self.btn_load_data)
        left_panel.addWidget(QLabel("🔧 处理配置"))
        left_panel.addWidget(params_group)
        left_panel.addWidget(self.btn_diagnose)
        left_panel.addWidget(QLabel("📜 诊断日志"))
        left_panel.addWidget(self.log_output)

        right_panel = QVBoxLayout()
        self.fig = Figure(figsize=(10, 8), facecolor='#ffffff')
        self.canvas = FigureCanvas(self.fig)
        self.ax_vib = self.fig.add_subplot(211) 
        self.ax_snd = self.fig.add_subplot(212) 
        self.fig.subplots_adjust(hspace=0.5, top=0.9, bottom=0.1)
        right_panel.addWidget(self.canvas)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.on_slider_moved)
        right_panel.insertWidget(1, self.slider)

        main_layout.addWidget(left_widget, 1)
        main_layout.addLayout(right_panel, 5)

    def log(self, message, mode="normal"):
        """
        mode: "normal" (灰色), "result" (青蓝色), "warning" (红色)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        if mode == "result":
            msg = f"<br><span style='color:#0984e3; font-size:14px; font-weight:bold;'>[{timestamp}] ➔ {message}</span><br>"
        elif mode == "warning":
            # 数据不足时的红字输出
            msg = f"<span style='color:#e84118; font-weight:bold;'>[{timestamp}] ⚠️ {message}</span>"
        else:
            msg = f"<span style='color:#636e72;'>[{timestamp}]</span> {message}"
            
        self.log_output.append(msg)
        self.log_output.ensureCursorVisible()

    def on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "加载权重", "", "Weights (*.pth)")
        if path:
            try:
                self.model = MotorNet(num_classes=6)
                self.model.load_state_dict(torch.load(path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.log("✅ MotorNet 权重载入成功")
            except Exception as e:
                self.log(f"❌ 模型错误: {str(e)}")

    def on_load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择数据", "", "Data (*.txt)")
        if path:
            try:
                self.raw_full_data = self.processor.load_and_clean(path)
                max_pos = len(self.raw_full_data) - 1
                self.slider.setRange(0, max_pos)
                self.slider.setEnabled(True)
                self.spin_start.setRange(0, max_pos)
                self.refresh_plots()
                self.log(f"✅ 数据载入: {path.split('/')[-1]}")
            except Exception as e:
                self.log(f"❌ 读取错误: {str(e)}")

    def on_slider_moved(self, value):
        self.spin_start.blockSignals(True)
        self.spin_start.setValue(value)
        self.spin_start.blockSignals(False)
        self.refresh_plots()

    def on_param_changed(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self.refresh_plots()

    def refresh_plots(self):
        if self.raw_full_data is None: return
        start = self.spin_start.value()
        self.processor.update_params(start, self.combo_fill.currentText())
        raw_seg, _ = self.processor.get_raw_and_padded(self.raw_full_data)
        
        if raw_seg is not None:
            actual_len = len(raw_seg)
            time_axis = np.arange(start, start + actual_len)
            self.ax_vib.clear()
            self.ax_vib.plot(time_axis, raw_seg[:, 0], label='X轴', color='#0984e3')
            self.ax_vib.plot(time_axis, raw_seg[:, 1], label='Y轴', color='#00b894')
            self.ax_vib.plot(time_axis, raw_seg[:, 2], label='Z轴', color='#fdcb6e')
            self.ax_vib.set_title("三轴振动原始信号 (Vibration)", fontsize=18, fontweight='bold', loc='center')
            self.ax_vib.set_xlim(start, start + 1024) 
            self.ax_vib.legend(loc='upper right', fontsize='x-small')
            self.ax_vib.grid(True, alpha=0.3)

            self.ax_snd.clear()
            self.ax_snd.plot(time_axis, raw_seg[:, 3], label='Sound', color='#d63031')
            self.ax_snd.set_title("声学原始信号 (Acoustic)", fontsize=18, fontweight='bold', loc='center')
            self.ax_snd.set_xlim(start, start + 1024)
            self.ax_snd.legend(loc='upper right', fontsize='x-small')
            self.ax_snd.grid(True, alpha=0.3)
            self.canvas.draw()

    def on_diagnose(self):
        if self.model is None or self.raw_full_data is None:
            self.log("⚠️ 无法诊断：数据或模型缺失")
            return

        try:
            raw_seg, padded_seg = self.processor.get_raw_and_padded(self.raw_full_data)
            
            # 计算数据占比
            actual_len = len(raw_seg)
            ratio = (actual_len / 1024) * 100
            
            input_tensor = self.processor.get_model_input(padded_seg).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred = np.argmax(probs)

            # 正常输出结论
            res = f"结果：{LABEL_MAP[pred]} (置信度 {probs[pred]*100:.2f}%)"
            self.log(res, mode="result")

            # 如果数据不足 1024，输出红色警告
            if actual_len < 1024:
                warn_msg = f"注意：当前数据仅 {actual_len} 点(占比 {ratio:.2f}%)，不足 1024。已执行[{self.combo_fill.currentText()}]补全。"
                self.log(warn_msg, mode="warning")
                
        except Exception as e:
            self.log(f"❌ 诊断失败: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())