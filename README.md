# FaultDiagnosisCNN_LSTM - 基于 CNN-LSTM 的机械故障诊断系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

本项目是一个基于深度学习的混合模型，结合了卷积神经网络（CNN）的空间特征提取能力和长短期记忆网络（LSTM）的时序建模能力，主要用于旋转机械（如电机）在动态环境下的智能故障诊断。

本项目的数据集基于开源的 [HUSTmotor-multi-modal-dataset](https://github.com/CHAOZHAO-1/HUSTmotor-multi-modal-dataset)。

## 📌 项目简介

针对机械设备在复杂工况下的故障特征提取难题，本项目设计了如下混合模型架构：
1. **CNN 层**：自动提取振动等多模态信号中的空间与局部频率特征。
2. **LSTM 层**：捕捉时序数据中的时间依赖性、长期趋势及动态演变。
3. **分类层**：实现对电机不同工况及故障类型的高精度识别。

## 🚀 主要特性

- **多模态数据适配**：针对 HUSTmotor-multi-modal-dataset 数据集进行了适配，支持多维度数据的处理。
- **混合架构**：采用 CNN 与 LSTM 级联的深度网络结构，最大化特征提取效率。
- **数据预处理**：内置针对长时序信号的切片、打标签以及添加高斯噪声等预处理功能。
- **可视化与评估**：支持输出混淆矩阵（Confusion Matrix）、计算召回率（Recall）等多项关键性能指标。

## 📁 目录结构

```text
FaultDiagnosisCNN_LSTM/
├── dataset/            # 数据集存放目录 (需手动下载 HUSTmotor 数据集)
├── models/             # 模型定义文件
├── src/              
│   ├── preprocess.py   # 数据预处理
    └── model.py        # 核心 CNN-LSTM 模型架构
    └── train.py        # 训练核心
    └── main.py         # 主流程
    └── app .py         # ui界面
├── pyproject.toml      # 项目依赖
└── README.md           # 项目说明文档