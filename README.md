# AI 自訓實戰手冊

建立一個開源、實務導向的 **Docusaurus 知識庫**，讓任何人都能**從零開始，用最少資源（Colab / 家用 GPU）訓練出具備專業領域實戰能力的自有 AI 模型**。

## 📚 專案簡介

本專案是一個完整的 AI 模型自訓練教學手冊，涵蓋：

- 🚀 **環境設定**：從 Colab 到本地 GPU 的完整設定指南
- 📊 **資料準備**：資料收集、清理、標註的最佳實踐
- 🔧 **模型微調**：LoRA、QLoRA、全參數微調等技術
- 📈 **效能評估**：評估指標、測試方法、優化技巧
- 🌐 **模型部署**：本地部署、雲端部署、API 服務建置

## 🎯 目標受眾

- 想要建立專屬聊天機器人的開發者
- 需要訓練領域專用模型的研究人員  
- 對 AI 技術感興趣的學習者
- 希望用最少資源訓練 AI 模型的實務工作者

## 🚀 快速開始

### 線上閱讀

訪問我們的文件網站：https://casperhk.github.io/ai-self-train-handbook/

### 本地開發

```bash
# 安裝依賴
npm install

# 啟動開發伺服器
npm start

# 建置靜態網站
npm run build

# 預覽建置結果
npm run serve
```

## 📖 文件結構

```
docs/
├── intro.md                    # 首頁介紹
├── getting-started/            # 入門指南
│   ├── setup.md               # 環境設定
│   ├── concepts.md            # 基礎概念
│   ├── model-selection.md     # 模型選擇
│   └── first-project.md       # 第一個專案
├── data-preparation/           # 資料準備
│   ├── data-collection.md     # 資料收集
│   ├── data-cleaning.md       # 資料清理
│   ├── data-annotation.md     # 資料標註
│   └── dataset-format.md      # 資料集格式
├── fine-tuning/                # 模型微調
│   ├── basics.md              # 微調基礎
│   ├── lora.md                # LoRA 微調
│   ├── qlora.md               # QLoRA 微調
│   ├── full-parameter.md      # 全參數微調
│   └── hyperparameters.md     # 超參數調整
├── evaluation/                 # 效能評估
│   ├── metrics.md             # 評估指標
│   ├── testing.md             # 測試方法
│   └── optimization.md        # 效能優化
└── deployment/                 # 模型部署
    ├── local.md               # 本地部署
    ├── cloud.md               # 雲端部署
    ├── api.md                 # API 服務
    └── optimization.md        # 部署優化
```

## 🤝 參與貢獻

我們歡迎各種形式的貢獻：

- 🐛 回報問題或建議改進
- 📝 改善文件或新增範例
- 💻 貢獻程式碼或工具
- 🌟 分享你的使用經驗

請參考 [GitHub Issues](https://github.com/CasperHK/ai-self-train-handbook/issues) 了解如何參與。

## 📄 授權

本專案採用 MIT License 授權。

## 🙏 致謝

感謝所有貢獻者和開源社群的支持。

本網站使用 [Docusaurus](https://docusaurus.io/) 建置，一個現代化的靜態網站生成器。

---

**立即開始你的 AI 自訓之旅！** 🚀
