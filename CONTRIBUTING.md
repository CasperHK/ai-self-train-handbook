# 貢獻指南

感謝你有興趣為「AI 自訓實戰手冊」做出貢獻！

## 🤝 如何貢獻

我們歡迎各種形式的貢獻，包括但不限於：

- 📝 改善文件內容
- 🐛 回報錯誤或問題
- 💡 提出新功能建議
- 🌟 分享使用經驗和案例
- 💻 貢獻程式碼範例
- 🔧 改進網站功能

## 📋 貢獻流程

### 1. Fork 專案

1. 點擊 GitHub 頁面右上角的 "Fork" 按鈕
2. Clone 你 fork 的專案到本地：

```bash
git clone https://github.com/你的用戶名/ai-self-train-handbook.git
cd ai-self-train-handbook
```

### 2. 建立分支

為你的貢獻建立一個新分支：

```bash
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/your-bug-fix
```

分支命名建議：
- `feature/` - 新增功能
- `fix/` - 修正問題
- `docs/` - 文件改進
- `style/` - 格式調整

### 3. 設定開發環境

安裝依賴：

```bash
npm install
```

啟動開發伺服器：

```bash
npm start
```

這會在 `http://localhost:3000/ai-self-train-handbook/` 啟動本地預覽。

### 4. 進行修改

- 遵循現有的文件風格和結構
- 確保程式碼範例可以正常運行
- 使用繁體中文撰寫文件
- 添加適當的範例和說明

### 5. 測試你的修改

在提交前，請確保：

```bash
# 測試建置
npm run build

# 檢查沒有錯誤
```

### 6. 提交變更

```bash
git add .
git commit -m "簡短描述你的修改"
```

提交訊息建議格式：
- `feat: 新增 XXX 功能`
- `fix: 修正 XXX 問題`
- `docs: 更新 XXX 文件`
- `style: 調整 XXX 格式`

### 7. 推送到 GitHub

```bash
git push origin feature/your-feature-name
```

### 8. 建立 Pull Request

1. 前往你 fork 的專案頁面
2. 點擊 "New Pull Request" 按鈕
3. 選擇你的分支
4. 填寫 PR 描述：
   - 說明你做了什麼改變
   - 為什麼需要這個改變
   - 如何測試這個改變

## 📝 文件撰寫指南

### 文件結構

每個文件應該包含：

1. **Front matter**：包含 `sidebar_position` 等元數據
2. **標題**：清楚描述主題
3. **簡介**：簡短說明本節內容
4. **主要內容**：詳細的說明和範例
5. **範例程式碼**：實際可執行的程式碼
6. **注意事項**：重要提示或警告
7. **下一步**：引導讀者到相關主題

### Markdown 範例

```markdown
---
sidebar_position: 1
---

# 文件標題

簡短介紹這個主題。

## 主要概念

詳細說明...

### 子標題

更多細節...

## 程式碼範例

\```python
# 你的程式碼
def example():
    return "Hello, World!"
\```

:::tip 提示
這是一個有用的提示。
:::

:::warning 注意
這是一個重要的警告。
:::

## 下一步

- [相關主題 1](./related-topic-1)
- [相關主題 2](./related-topic-2)
```

### 程式碼規範

- 使用清楚的變數名稱
- 添加必要的註解
- 確保程式碼可以直接執行
- 包含必要的 import 語句
- 處理可能的錯誤情況

### 中文撰寫規範

- 使用繁體中文
- 專業術語首次出現時提供英文對照
- 保持語氣友善、易懂
- 使用項目符號和編號列表提高可讀性

## 🐛 回報問題

如果你發現問題，請建立一個 Issue：

1. 前往 [Issues 頁面](https://github.com/CasperHK/ai-self-train-handbook/issues)
2. 點擊 "New Issue"
3. 選擇適當的模板（如果有）
4. 提供詳細資訊：
   - 問題描述
   - 重現步驟
   - 預期行為
   - 實際行為
   - 環境資訊（作業系統、瀏覽器等）
   - 螢幕截圖（如果適用）

## 💡 提出建議

我們歡迎任何改進建議：

1. 建立一個 Issue
2. 使用 "Feature Request" 或 "Enhancement" 標籤
3. 清楚描述你的想法
4. 說明為什麼這個改進有價值
5. 提供可能的實作方式（如果有想法）

## ✅ Pull Request 審查流程

提交 PR 後：

1. **自動測試**：GitHub Actions 會自動執行測試
2. **程式碼審查**：維護者會審查你的程式碼
3. **討論**：可能會有一些討論和修改建議
4. **合併**：通過審查後，PR 會被合併

## 📜 授權

貢獻到本專案的內容將採用與專案相同的授權（MIT License）。

## 🙏 感謝

感謝你願意花時間為這個專案做出貢獻！每一個貢獻，無論大小，都很重要。

## 📞 聯繫方式

如果你有任何問題：

- 建立 [GitHub Issue](https://github.com/CasperHK/ai-self-train-handbook/issues)
- 參與 [GitHub Discussions](https://github.com/CasperHK/ai-self-train-handbook/discussions)

---

再次感謝你的貢獻！🎉
