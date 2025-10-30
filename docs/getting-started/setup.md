---
sidebar_position: 1
---

# 環境設定

在開始訓練 AI 模型之前，我們需要先建立適合的開發環境。本指南將幫助你選擇合適的平台並完成初始設定。

## 🖥️ 選擇訓練平台

你可以根據自己的資源和需求選擇以下任一平台：

### 選項 1：Google Colab（推薦新手）

**優點：**
- ✅ 完全免費（有限制）
- ✅ 無需本地硬體
- ✅ 預裝常用套件
- ✅ 可升級到 Colab Pro 獲得更好的資源

**缺點：**
- ❌ 每次執行時間有限制
- ❌ 需要網路連線
- ❌ 免費版 GPU 資源有限

**適合對象：** 初學者、預算有限、想快速開始

### 選項 2：本地 GPU

**優點：**
- ✅ 無使用時間限制
- ✅ 可離線工作
- ✅ 更好的隱私保護
- ✅ 可自由安裝套件

**缺點：**
- ❌ 需要投資硬體
- ❌ 需要自行設定環境
- ❌ 電費成本

**最低建議配置：**
- GPU：NVIDIA RTX 3060（12GB VRAM）或以上
- RAM：16GB 或以上
- 硬碟空間：100GB 可用空間

**適合對象：** 有預算、需要長時間訓練、重視隱私

### 選項 3：雲端服務（AWS、GCP、Azure）

**優點：**
- ✅ 彈性擴充資源
- ✅ 專業級硬體
- ✅ 按需付費

**缺點：**
- ❌ 需要付費
- ❌ 設定相對複雜
- ❌ 需要管理成本

**適合對象：** 企業使用、大規模訓練、專業開發

## 🚀 Google Colab 快速設定

### 步驟 1：開啟 Colab

1. 前往 [Google Colab](https://colab.research.google.com/)
2. 使用 Google 帳號登入
3. 點選「新增筆記本」

### 步驟 2：啟用 GPU

1. 點選上方選單：**執行階段** → **變更執行階段類型**
2. 在「硬體加速器」選單中選擇 **GPU** 或 **TPU**
3. 點選「儲存」

### 步驟 3：驗證 GPU 可用性

在 Colab 筆記本中執行以下程式碼：

```python
import torch

# 檢查 CUDA 是否可用
print(f"CUDA 可用: {torch.cuda.is_available()}")

# 如果可用，顯示 GPU 資訊
if torch.cuda.is_available():
    print(f"GPU 型號: {torch.cuda.get_device_name(0)}")
    print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

### 步驟 4：安裝必要套件

```python
# 安裝 transformers 和相關套件
!pip install -q transformers datasets accelerate peft bitsandbytes

# 驗證安裝
import transformers
print(f"Transformers 版本: {transformers.__version__}")
```

## 🖥️ 本地環境設定

### 前置需求

1. **安裝 Python 3.10+**
   - 下載：https://www.python.org/downloads/
   - 建議使用 Anaconda：https://www.anaconda.com/

2. **安裝 CUDA Toolkit**（NVIDIA GPU 使用者）
   - 下載：https://developer.nvidia.com/cuda-downloads
   - 確認版本相容性：https://pytorch.org/get-started/locally/

### 建立虛擬環境

```bash
# 使用 conda
conda create -n ai-training python=3.10
conda activate ai-training

# 或使用 venv
python -m venv ai-training
source ai-training/bin/activate  # Linux/Mac
# 或
ai-training\Scripts\activate  # Windows
```

### 安裝 PyTorch

訪問 [PyTorch 官網](https://pytorch.org/get-started/locally/) 選擇適合你系統的版本。

```bash
# CUDA 11.8 範例
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU 版本
pip install torch torchvision torchaudio
```

### 驗證 PyTorch 安裝

```python
import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 型號: {torch.cuda.get_device_name(0)}")
```

### 安裝必要套件

```bash
pip install transformers datasets accelerate peft bitsandbytes
pip install sentencepiece protobuf
pip install jupyter notebook  # 可選：如果想使用 Jupyter
```

## 📦 套件說明

- **transformers**: Hugging Face 的模型庫
- **datasets**: 資料集載入和處理
- **accelerate**: 分散式訓練和混合精度
- **peft**: Parameter-Efficient Fine-Tuning（如 LoRA）
- **bitsandbytes**: 量化和記憶體優化

## ✅ 驗證設定

建立測試腳本 `test_setup.py`：

```python
import torch
import transformers
from transformers import AutoTokenizer, AutoModel

print("=== 環境檢查 ===")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print("\n=== 測試模型載入 ===")
try:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModel.from_pretrained("bert-base-chinese")
    print("✅ 模型載入成功")
except Exception as e:
    print(f"❌ 模型載入失敗: {e}")

print("\n環境設定完成！")
```

執行測試：

```bash
python test_setup.py
```

## 🔧 常見問題

### CUDA 無法使用

1. 確認 NVIDIA 驅動程式已安裝
2. 檢查 CUDA 版本與 PyTorch 版本相容性
3. 重新安裝對應版本的 PyTorch

### 記憶體不足

1. 減少批次大小（batch size）
2. 使用梯度累積
3. 啟用混合精度訓練
4. 使用量化（quantization）

### 套件安裝失敗

1. 確認 Python 版本正確
2. 更新 pip：`pip install --upgrade pip`
3. 使用清華鏡像（中國地區）：
   ```bash
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package-name>
   ```

## 下一步

環境設定完成後，接下來：

- 📖 [了解基礎概念](./concepts)
- 🤖 [選擇合適的模型](./model-selection)
- 🚀 [開始第一個專案](./first-project)
