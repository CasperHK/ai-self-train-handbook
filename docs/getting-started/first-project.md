---
sidebar_position: 4
---

# 第一個微調專案

這是一個完整的實戰教學，將帶你從零開始完成第一個 AI 模型微調專案。我們將使用 Google Colab 和 QLoRA 技術，訓練一個中文客服機器人。

## 🎯 專案目標

訓練一個能回答特定產品問題的客服機器人，例如：

**輸入：** 你們的產品保固期是多久？
**輸出：** 我們的產品提供一年免費保固服務。

## 📋 準備工作

### 1. 開啟 Google Colab

1. 前往 [Google Colab](https://colab.research.google.com/)
2. 建立新筆記本
3. 啟用 GPU：**執行階段** → **變更執行階段類型** → 選擇 **T4 GPU**

### 2. 安裝必要套件

```python
# 安裝所需套件
!pip install -q transformers datasets accelerate peft bitsandbytes trl
```

## 🔧 步驟 1：載入模型和 Tokenizer

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# 選擇基礎模型
model_name = "Qwen/Qwen-7B"

# 配置 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 載入模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
```

## 📊 步驟 2：準備訓練資料

```python
from datasets import Dataset

# 範例資料：客服問答對
data = [
    {
        "instruction": "你們的產品保固期是多久？",
        "output": "我們的產品提供一年免費保固服務，在保固期內如有任何品質問題都可以免費維修或更換。"
    },
    {
        "instruction": "如何聯繫客服？",
        "output": "您可以透過以下方式聯繫我們：\n1. 客服電話：0800-123-456（週一至週五 9:00-18:00）\n2. 電子郵件：service@example.com\n3. 線上客服：官網右下角的聊天按鈕"
    },
    {
        "instruction": "可以退貨嗎？",
        "output": "購買後7天內，如商品未使用且包裝完整，可申請退貨。請聯繫客服取得退貨授權碼。"
    },
    {
        "instruction": "運送需要多久時間？",
        "output": "訂單確認後，一般在3-5個工作天內送達。偏遠地區可能需要額外1-2天。"
    },
    {
        "instruction": "付款方式有哪些？",
        "output": "我們接受以下付款方式：\n1. 信用卡（Visa、MasterCard、JCB）\n2. ATM轉帳\n3. 貨到付款（限台灣本島）\n4. 超商代收"
    },
]

# 轉換為 Dataset 格式
dataset = Dataset.from_list(data)

# 格式化資料
def format_instruction(sample):
    return f"""### 指令：
{sample['instruction']}

### 回應：
{sample['output']}"""

# 應用格式化
dataset = dataset.map(lambda x: {"text": format_instruction(x)})
```

:::tip 實際應用
在實際專案中，你需要準備 100-1000 筆以上的高品質問答對。資料越多越好，但品質比數量更重要！
:::

## ⚙️ 步驟 3：配置 LoRA

```python
# LoRA 配置
lora_config = LoraConfig(
    r=16,                      # LoRA rank
    lora_alpha=32,             # LoRA alpha
    target_modules=[           # 要訓練的模組
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    lora_dropout=0.05,         # Dropout
    bias="none",
    task_type="CAUSAL_LM"
)

# 應用 LoRA 到模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

## 🏃 步驟 4：訓練模型

```python
# 訓練參數
training_args = TrainingArguments(
    output_dir="./results",              # 輸出目錄
    num_train_epochs=3,                  # 訓練輪數
    per_device_train_batch_size=1,       # 批次大小
    gradient_accumulation_steps=4,       # 梯度累積
    learning_rate=2e-4,                  # 學習率
    logging_steps=10,                    # 日誌頻率
    save_steps=50,                       # 儲存頻率
    save_total_limit=2,                  # 最多保留檢查點數
    fp16=True,                           # 混合精度訓練
    report_to="none",                    # 不使用外部日誌工具
)

# 建立訓練器
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=512,
)

# 開始訓練
trainer.train()
```

## 💾 步驟 5：儲存模型

```python
# 儲存 LoRA 權重
trainer.model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")

print("模型已儲存至 ./fine-tuned-model")
```

## 🧪 步驟 6：測試模型

```python
# 載入微調後的模型進行測試
from peft import PeftModel

# 重新載入基礎模型
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 載入 LoRA 權重
model = PeftModel.from_pretrained(base_model, "./fine-tuned-model")

# 測試函數
def generate_response(instruction):
    prompt = f"""### 指令：
{instruction}

### 回應：
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 只返回回應部分
    return response.split("### 回應：")[-1].strip()

# 測試範例
test_questions = [
    "產品保固多久？",
    "如何退貨？",
    "客服電話是什麼？"
]

for question in test_questions:
    print(f"問題：{question}")
    print(f"回答：{generate_response(question)}")
    print("-" * 50)
```

## 📈 監控訓練進度

訓練過程中你會看到類似的輸出：

```
Step 10: loss=2.345
Step 20: loss=1.987
Step 30: loss=1.654
...
```

**Loss 下降** = 模型在學習！

理想情況：
- Loss 持續下降
- 最終收斂到較低值（< 1.0）

## 🎓 理解輸出

### 訓練參數說明

```
trainable params: 8,388,608 || all params: 7,241,728,000 || trainable%: 0.1158
```

這表示：
- 只有 0.12% 的參數被訓練（LoRA 的威力！）
- 大幅減少記憶體和運算需求

### 訓練時間估計

在 Colab T4 GPU 上：
- 5 筆資料：約 5-10 分鐘
- 100 筆資料：約 1-2 小時
- 1000 筆資料：約 10-20 小時

## 🔧 常見問題與解決

### 記憶體不足（OOM）

```python
# 減少批次大小
per_device_train_batch_size=1

# 增加梯度累積
gradient_accumulation_steps=8

# 減少序列長度
max_seq_length=256
```

### 訓練不穩定

```python
# 降低學習率
learning_rate=1e-4

# 增加 warmup
warmup_steps=100
```

### 效果不理想

1. **增加訓練資料**：更多高品質資料
2. **增加訓練輪數**：從 3 增加到 5-10
3. **調整 LoRA rank**：增加 r 值（如 32）

## 📦 下載模型到本地

```python
from google.colab import files

# 壓縮模型檔案
!zip -r fine-tuned-model.zip ./fine-tuned-model

# 下載
files.download('fine-tuned-model.zip')
```

## ✅ 完整程式碼範本

完整的可執行 Colab Notebook：

👉 [點此開啟 Colab Notebook](https://colab.research.google.com/)

## 🎯 下一步

恭喜完成第一個微調專案！接下來你可以：

1. **準備更多資料**：[資料準備指南](../data-preparation/data-collection)
2. **深入學習 LoRA**：[LoRA 進階技巧](../fine-tuning/lora)
3. **部署模型**：[模型部署](../deployment/local)
4. **優化效能**：[效能優化](../evaluation/optimization)

## 💡 實戰技巧

### 資料品質最重要
- 寧少勿濫：10 筆高品質 > 100 筆低品質
- 涵蓋多樣性：確保資料涵蓋各種情況
- 格式一致：統一問答格式

### 迭代優化
1. 從小規模開始（10-20 筆）
2. 快速訓練和測試
3. 發現問題並改進資料
4. 逐步擴大規模

### 版本管理
- 記錄每次實驗的參數
- 保存不同版本的模型
- 比較不同版本的效果

## 參考資源

- [Hugging Face Transformers 文件](https://huggingface.co/docs/transformers)
- [PEFT 文件](https://huggingface.co/docs/peft)
- [TRL 文件](https://huggingface.co/docs/trl)
