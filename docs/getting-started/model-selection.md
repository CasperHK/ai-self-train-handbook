---
sidebar_position: 3
---

# 選擇合適的模型

選擇正確的基礎模型是成功訓練的第一步。本指南將幫助你根據需求和資源選擇最適合的模型。

## 🎯 選擇考量因素

### 1. 任務類型

不同的任務需要不同類型的模型：

#### 文字生成（Text Generation）
- **用途**：聊天機器人、文章撰寫、程式碼生成
- **推薦模型**：GPT 系列、LLaMA、Mistral

#### 文字理解（Text Understanding）
- **用途**：文本分類、情感分析、命名實體識別
- **推薦模型**：BERT、RoBERTa

#### 問答系統（Question Answering）
- **用途**：客服機器人、知識問答
- **推薦模型**：GPT 系列、T5

#### 翻譯（Translation）
- **用途**：多語言翻譯
- **推薦模型**：MarianMT、T5

### 2. 語言支援

#### 多語言模型
- **XLM-RoBERTa**：支援 100+ 語言
- **mBERT**：多語言 BERT
- **BLOOM**：支援 46 種語言

#### 中文專用模型
- **BERT-base-chinese**：中文 BERT
- **Chinese-LLaMA**：中文 LLaMA
- **ChatGLM**：清華大學開源對話模型
- **Baichuan**：百川智能開源模型
- **Qwen**：阿里雲通義千問

### 3. 模型規模與資源需求

| 模型類型 | 參數量 | GPU 記憶體需求 | 適用平台 |
|---------|--------|----------------|----------|
| 極小型 | < 500M | 2-4GB | Colab 免費版 |
| 小型 | 500M-1B | 4-8GB | Colab 免費版 |
| 中型 | 1B-7B | 8-16GB | Colab Pro / RTX 3060 |
| 大型 | 7B-13B | 16-24GB | RTX 3090 / 4090 |
| 超大型 | 13B+ | 24GB+ | A100 / 多 GPU |

:::tip 記憶體計算
模型記憶體需求（FP16）≈ 參數量 × 2 bytes
例如：7B 模型 ≈ 14GB（僅載入模型，不含訓練）
:::

## 🤖 推薦模型列表

### 入門推薦（適合 Colab 免費版）

#### 1. BERT-base-chinese
```python
model_name = "bert-base-chinese"
```
- **參數量**：110M
- **記憶體需求**：約 2GB
- **適用任務**：文本分類、命名實體識別、問答
- **優點**：輕量、快速、中文支援好
- **缺點**：不適合長文本生成

#### 2. GPT2-chinese
```python
model_name = "uer/gpt2-chinese-cluecorpussmall"
```
- **參數量**：117M
- **記憶體需求**：約 2GB
- **適用任務**：文本生成、對話
- **優點**：輕量、適合初學者
- **缺點**：能力相對有限

### 中階推薦（適合 Colab Pro / RTX 3060）

#### 1. Chinese-LLaMA-2-7B
```python
model_name = "hfl/chinese-llama-2-7b"
```
- **參數量**：7B
- **記憶體需求**：約 14GB（可用 QLoRA 降至 6GB）
- **適用任務**：通用對話、指令跟隨
- **優點**：中文能力強、社群支援好
- **缺點**：需要較大記憶體

#### 2. ChatGLM3-6B
```python
model_name = "THUDM/chatglm3-6b"
```
- **參數量**：6B
- **記憶體需求**：約 12GB（可用 QLoRA 降至 5GB）
- **適用任務**：對話、多輪對話
- **優點**：中文原生、效能好
- **缺點**：特殊架構，需額外適配

#### 3. Qwen-7B
```python
model_name = "Qwen/Qwen-7B"
```
- **參數量**：7B
- **記憶體需求**：約 14GB（可用 QLoRA 降至 6GB）
- **適用任務**：通用對話、程式碼生成
- **優點**：多語言支援、效能優秀
- **缺點**：模型較新，文件較少

### 進階推薦（適合 RTX 3090 / 4090）

#### 1. LLaMA-2-13B
```python
model_name = "meta-llama/Llama-2-13b-hf"
```
- **參數量**：13B
- **記憶體需求**：約 26GB（可用 QLoRA 降至 10GB）
- **適用任務**：複雜對話、推理
- **優點**：效能強大、開源友善
- **缺點**：中文能力需強化

#### 2. Baichuan2-13B
```python
model_name = "baichuan-inc/Baichuan2-13B-Base"
```
- **參數量**：13B
- **記憶體需求**：約 26GB（可用 QLoRA 降至 10GB）
- **適用任務**：通用對話
- **優點**：中文能力強
- **缺點**：需要較大資源

## 🔍 模型來源

### Hugging Face Model Hub
最大的開源模型平台，提供數萬個預訓練模型。

**使用方式：**
```python
from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

**瀏覽模型：** https://huggingface.co/models

### ModelScope（魔搭）
阿里雲的模型平台，中國地區訪問更快。

**使用方式：**
```python
from modelscope import AutoModel, AutoTokenizer

model_name = "damo/nlp_structbert_backbone_base_std"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

**瀏覽模型：** https://modelscope.cn/models

## 🎯 決策流程圖

```
開始
  ↓
你的 GPU 記憶體是？
  ├─ < 8GB
  │   └─ 使用 QLoRA + 小型模型（1-3B）
  │       推薦：BERT-chinese, GPT2-chinese
  │
  ├─ 8-16GB
  │   └─ 使用 QLoRA + 中型模型（3-7B）
  │       推薦：ChatGLM3-6B, LLaMA-2-7B
  │
  └─ > 16GB
      └─ 可選擇大型模型（7-13B）
          推薦：LLaMA-2-13B, Baichuan2-13B
```

## 📊 模型效能比較

### 中文能力

| 模型 | 參數量 | 中文流暢度 | 指令理解 | 推薦度 |
|------|--------|-----------|---------|--------|
| ChatGLM3-6B | 6B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Qwen-7B | 7B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Chinese-LLaMA-2-7B | 7B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Baichuan2-13B | 13B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| BERT-chinese | 110M | ⭐⭐⭐ | N/A | ⭐⭐⭐ |

### 資源需求

| 模型 | 最低 GPU | 推薦 GPU | 訓練速度 |
|------|---------|---------|---------|
| BERT-chinese | 4GB | 8GB | 快 |
| GPT2-chinese | 4GB | 8GB | 快 |
| ChatGLM3-6B | 8GB | 16GB | 中等 |
| Qwen-7B | 8GB | 16GB | 中等 |
| LLaMA-2-7B | 10GB | 16GB | 中等 |
| Baichuan2-13B | 12GB | 24GB | 慢 |

## 🚀 快速載入模型

### 基本載入

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen-7B"

# 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# 載入模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)
```

### 使用量化載入

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
```

## ⚠️ 注意事項

### 授權許可

某些模型有使用限制，請務必檢查：
- 商業使用是否允許
- 是否需要申請授權
- 資料使用限制

### 模型快取

模型下載後會快取在本地：
- 預設位置：`~/.cache/huggingface/`
- 可設定環境變數：`export HF_HOME=/path/to/cache`
- 大型模型可能佔用 10-30GB 空間

### 網路問題

中國地區可能需要：
- 使用鏡像站（ModelScope）
- 設定代理
- 手動下載模型檔案

## 下一步

選擇好模型後，接下來：

- 🚀 [開始第一個微調專案](./first-project)
- 📊 [準備訓練資料](../data-preparation/data-collection)
- 🔧 [了解 LoRA 微調](../fine-tuning/lora)

## 參考資源

- [Hugging Face Model Hub](https://huggingface.co/models)
- [ModelScope](https://modelscope.cn/models)
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
