# EIS -> SOH Predictor

基于电化学阻抗谱（EIS）预测锂离子电池健康状态（State-of-Health, SOH），支持模型训练、测试与单样本预测。

---

## 项目目录结构

```
├── data/                      # 数据集目录
│   ├── processed_data_Capacity_*.csv  # Zhang 数据集原始命名
│   ├── EIS_state_V_*.csv              # Zhang 数据集原始命名
│   ├── Cell*_*SOH_*degC_95SOC_*.xls   # Rashid 数据集原始命名
│   ├── mycell_j*.csv                  # 自测电池样本
│   └── mycell_j*_lite.csv             # 自测电池样本（选定特征频率点）
├── src/                       # 所有 Python 模块
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model.py
│   ├── util.py
│   ├── train.py
│   ├── test.py
│   └── inference.py
└── README.md
```

---

## 环境安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## 使用教程

### 1. 模型训练

```powershell
python src/train.py --soh_pattern data/processed_data_Capacity_*.csv --eis_pattern data/EIS_state_V_*.csv --excel_pattern data/Cell*_*SOH_*degC_95SOC_*.xls --epochs 100 --batch_size 32 --model_path src/merged_model.pth
```

输出：

- 模型权重：`src/merged_model.pth`
- 训练损失曲线：`train_loss.png`

---

### 2. 模型评估

```powershell
python src/eval.py --model_path src/merged_model.pth --soh_pattern data/processed_data_Capacity_*.csv --eis_pattern data/EIS_state_V_*.csv --excel_pattern data/Cell*_*SOH_*degC_95SOC_*.xls
```

输出示例：

```
MAE  : 0.0123
MSE  : 0.0004
RMSE : 0.0189
```

---

### 3. 单样本推理（自测数据）

假设数据在 `data/mycell_j2.csv`，插值点数为 60 → 特征维度为 120：

```powershell
python src/inference.py --model_path src/merged_model.pth --eis_path data/mycell_j2.csv --input_dim 120
```

输出：

```
预测 SOH ≈ 93.45%
```

需要注意的是，输入的自测数据的维数是没有要求的，都会先经过归一化再输入模型，因此可以实现少量频率点预测

---

## 📚 数据来源

本项目使用了以下两个公开数据集进行训练与验证：

[1] Zhang, Y., et al. (2020). *Identifying degradation patterns of lithium ion batteries from impedance spectroscopy using machine learning*. Zenodo. [https://doi.org/10.5281/zenodo.3633835](https://doi.org/10.5281/zenodo.3633835)

[2] Rashid, M., et al. (2023). *Dataset for rapid state of health estimation of lithium batteries using EIS and machine learning: Training and validation*. Data in Brief, 48, 109157. [https://doi.org/10.1016/j.dib.2023.109157](https://doi.org/10.1016/j.dib.2023.109157)

---

## ✨ 欢迎贡献

如有问题或建议，欢迎提出:D
