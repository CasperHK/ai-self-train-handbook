---
sidebar_position: 1
---

# 資料收集

高品質的訓練資料是成功微調 AI 模型的關鍵。本指南將教你如何有效收集和組織訓練資料。

## 📊 資料需求評估

### 資料量建議

| 任務複雜度 | 最少資料量 | 推薦資料量 | 理想資料量 |
|-----------|-----------|-----------|-----------|
| 簡單（FAQ） | 50 筆 | 200-500 筆 | 1000+ 筆 |
| 中等（客服對話） | 100 筆 | 500-1000 筆 | 5000+ 筆 |
| 複雜（專業領域） | 500 筆 | 2000-5000 筆 | 10000+ 筆 |

:::tip 品質 > 數量
10 筆高品質資料的效果 > 100 筆低品質資料
:::

### 資料多樣性

確保資料涵蓋：
- ✅ 不同的問題形式
- ✅ 各種回答風格
- ✅ 邊緣案例
- ✅ 常見錯誤
- ✅ 專業術語和通俗說法

## 🔍 資料來源

### 1. 內部資料

#### 客服記錄
- 歷史對話記錄
- 常見問題清單
- 客服工單系統

**優點：**
- 真實場景
- 貼合需求
- 高度相關

**取得方式：**
```python
# 範例：從 CSV 讀取客服記錄
import pandas as pd

df = pd.read_csv('customer_service_logs.csv')
conversations = df[['question', 'answer']].to_dict('records')
```

#### 產品文件
- 使用手冊
- FAQ 文件
- 技術規格
- 操作指南

#### 內部知識庫
- Wiki 頁面
- 訓練文件
- 標準作業流程

### 2. 公開資料集

#### 中文對話資料集

**LCCC（Large-scale Chinese Conversation Collection）**
```python
from datasets import load_dataset

dataset = load_dataset("lccc", "base")
```
- 規模：1200 萬對話
- 來源：社群媒體
- 適用：通用對話

**Chinese-Chitchat-Corpus**
- GitHub: https://github.com/codemayq/chinese_chatbot_corpus
- 規模：550 萬對話
- 適用：閒聊機器人

**WebQA**
```python
from datasets import load_dataset

dataset = load_dataset("suolyer/webqa")
```
- 規模：42k 問答對
- 適用：問答系統

#### 英文資料集（可翻譯）

**Alpaca**
- 52k 指令資料
- 涵蓋多種任務
- 適用：指令微調

**Dolly**
- 15k 人工標註資料
- 開放授權
- 適用：通用任務

### 3. 合成資料

使用大型語言模型生成訓練資料。

#### 使用 GPT 生成

```python
import openai

def generate_qa_pairs(topic, num_pairs=10):
    prompt = f"""請生成 {num_pairs} 個關於「{topic}」的問答對。
    格式：
    Q: 問題
    A: 詳細回答
    
    要求：
    - 問題要有變化性
    - 答案要專業準確
    - 涵蓋不同角度
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

# 使用範例
qa_pairs = generate_qa_pairs("Python 程式設計", 20)
print(qa_pairs)
```

#### 使用開源模型生成

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

def generate_synthetic_data(seed_examples, count=100):
    """根據種子範例生成更多資料"""
    prompt = f"""基於以下範例，生成類似的問答對：

{seed_examples}

請生成 {count} 個新的問答對。"""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=2000)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### 4. 網路爬蟲

#### 合法爬取公開資訊

```python
import requests
from bs4 import BeautifulSoup

def scrape_faq(url):
    """爬取 FAQ 頁面"""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    qa_pairs = []
    # 根據網站結構調整選擇器
    questions = soup.select('.question')
    answers = soup.select('.answer')
    
    for q, a in zip(questions, answers):
        qa_pairs.append({
            'question': q.text.strip(),
            'answer': a.text.strip()
        })
    
    return qa_pairs

# 使用範例
faqs = scrape_faq('https://example.com/faq')
```

:::warning 注意事項
- 遵守網站使用條款
- 尊重 robots.txt
- 避免過度請求
- 注意版權問題
:::

### 5. 眾包標註

使用平台尋求幫助：
- Amazon Mechanical Turk
- Figure Eight
- 自建標註平台

## 📝 資料收集最佳實踐

### 1. 定義明確的範圍

```markdown
## 專案範圍定義

**領域：** 電商客服
**語言：** 繁體中文
**主題：**
- 產品諮詢
- 訂單查詢
- 退換貨政策
- 付款問題
- 配送資訊

**不包含：**
- 技術支援
- 法律諮詢
- 醫療建議
```

### 2. 建立資料收集檢查清單

```markdown
□ 資料來源已確認
□ 授權許可已檢查
□ 隱私問題已處理
□ 資料格式已統一
□ 品質標準已定義
□ 樣本已抽查驗證
```

### 3. 維護資料目錄

```python
# 資料組織結構範例
data/
├── raw/                    # 原始資料
│   ├── customer_logs.csv
│   ├── faq_scraped.json
│   └── synthetic_data.jsonl
├── processed/              # 處理後資料
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
└── metadata/               # 資料描述
    ├── sources.md
    ├── statistics.json
    └── changelog.md
```

## 🔧 資料收集工具

### Python 腳本範例

```python
import json
import pandas as pd
from pathlib import Path

class DataCollector:
    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collected_data = []
    
    def add_from_csv(self, filepath, question_col, answer_col):
        """從 CSV 新增資料"""
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            self.collected_data.append({
                'question': row[question_col],
                'answer': row[answer_col],
                'source': 'csv',
                'source_file': filepath
            })
    
    def add_from_json(self, filepath):
        """從 JSON 新增資料"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                self.collected_data.append({
                    **item,
                    'source': 'json',
                    'source_file': filepath
                })
    
    def add_manual(self, question, answer, metadata=None):
        """手動新增資料"""
        self.collected_data.append({
            'question': question,
            'answer': answer,
            'source': 'manual',
            'metadata': metadata or {}
        })
    
    def save(self, filename="collected_data.jsonl"):
        """儲存收集的資料"""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in self.collected_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"已儲存 {len(self.collected_data)} 筆資料至 {output_path}")
    
    def get_statistics(self):
        """取得資料統計"""
        return {
            'total_count': len(self.collected_data),
            'sources': pd.Series([d['source'] for d in self.collected_data]).value_counts().to_dict(),
            'avg_question_length': sum(len(d['question']) for d in self.collected_data) / len(self.collected_data),
            'avg_answer_length': sum(len(d['answer']) for d in self.collected_data) / len(self.collected_data),
        }

# 使用範例
collector = DataCollector()
collector.add_from_csv('customer_logs.csv', 'question', 'answer')
collector.add_manual("你們的營業時間？", "我們的營業時間是週一至週五 9:00-18:00")
print(collector.get_statistics())
collector.save()
```

## ✅ 資料收集檢查表

### 法律與倫理

- [ ] 確認資料使用權限
- [ ] 遵守隱私保護法規（GDPR、個資法等）
- [ ] 移除個人識別資訊（PII）
- [ ] 取得必要授權

### 品質控制

- [ ] 定義品質標準
- [ ] 執行樣本抽查
- [ ] 去除重複資料
- [ ] 驗證資料完整性

### 文件記錄

- [ ] 記錄資料來源
- [ ] 說明收集方法
- [ ] 註明收集日期
- [ ] 記錄資料統計

## 🎯 常見挑戰與解決

### 挑戰 1：資料量不足

**解決方案：**
1. 使用資料增強技術
2. 合成資料生成
3. 主動學習（選擇性標註）
4. 少樣本學習技術

### 挑戰 2：資料品質參差

**解決方案：**
1. 建立明確的品質標準
2. 多輪人工審核
3. 自動化品質檢查
4. 迭代改進流程

### 挑戰 3：資料不平衡

**解決方案：**
1. 過採樣少數類別
2. 欠採樣多數類別
3. 合成少數類別資料
4. 調整損失函數權重

## 下一步

資料收集完成後：

- 📝 [資料清理與處理](./data-cleaning)
- 🏷️ [資料標註](./data-annotation)
- 📋 [資料集格式](./dataset-format)

## 參考資源

- [Hugging Face Datasets](https://huggingface.co/datasets)
- [中文 NLP 資源](https://github.com/crownpku/Awesome-Chinese-NLP)
- [資料增強技術](https://github.com/makcedward/nlpaug)
