import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit)
from PySide6.QtCore import Signal, Slot, QThread
from qt_material import apply_stylesheet

# 模拟后台任务：加噪或训练
class WorkThread(QThread):
    # 定义一个信号，用于将日志发回主界面
    log_signal = Signal(str)

    def __init__(self, snr, lr):
        super().__init__()
        self.snr = snr
        self.lr = lr

    def run(self):
        self.log_signal.emit(f"🚀 [状态] 启动实验配置：SNR={self.snr}dB, Learning Rate={self.lr}")
        
        # --- 这里放置你真实的逻辑 ---
        # 1. 调用 addNoise.py 里的函数
        # 2. 调用 train.py 里的模型训练函数
        
        # 模拟运行耗时
        import time
        time.sleep(2) 
        
        self.log_signal.emit("✅ [完成] 训练任务已启动，请在 SwanLab 云端查看实时指标。")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CWRU 轴承故障诊断 - 实验管理控制台")
        self.setMinimumSize(500, 400)

        # 整体布局
        main_layout = QVBoxLayout()

        # 1. 参数配置区
        param_group = QVBoxLayout()
        
        # 信噪比
        snr_layout = QHBoxLayout()
        snr_layout.addWidget(QLabel("信噪比 SNR (dB):"))
        self.snr_input = QLineEdit("0")
        snr_layout.addWidget(self.snr_input)
        param_group.addLayout(snr_layout)

        # 学习率
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("学习率 (LR):"))
        self.lr_input = QLineEdit("0.001")
        lr_layout.addWidget(self.lr_input)
        param_group.addLayout(lr_layout)

        main_layout.addLayout(param_group)

        # 2. 按钮区
        self.btn_run = QPushButton("🚀 开始运行 (加噪+训练)")
        self.btn_run.setFixedHeight(40)
        self.btn_run.clicked.connect(self.run_experiment)
        main_layout.addWidget(self.btn_run) 

        # 3. 日志显示区
        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setPlaceholderText("系统运行日志将显示在这里...")
        main_layout.addWidget(self.log_window)

        # 设置中央控件
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def run_experiment(self):
        try:
            snr = float(self.snr_input.text())
            lr = float(self.lr_input.text())
            
            # 禁用按钮防止重复点击
            self.btn_run.setEnabled(False)
            self.log_window.append("------------------------------------")
            
            # 开启新线程运行实验
            self.worker = WorkThread(snr, lr)
            self.worker.log_signal.connect(self.log_window.append)
            self.worker.finished.connect(lambda: self.btn_run.setEnabled(True))
            self.worker.start()
        except ValueError:
            self.log_window.append("❌ 错误：请输入有效的数字参数！")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 应用主题 (推荐：dark_teal.xml, light_blue.xml, dark_amber.xml)
    apply_stylesheet(app, theme='dark_teal.xml')
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())