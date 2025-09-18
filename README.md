# RAG 教學示範系統

這是一個基於 RAG（Retrieval-Augmented Generation）架構的文件查詢系統教學專案。本系統展示了如何從網頁爬取資料、建立向量索引，到最後使用 LLM 進行智慧問答的完整流程。

## 🎯 教學目標

1. 理解 RAG 系統的基本概念和運作原理
2. 學習如何處理和結構化網頁資料
3. 掌握向量資料庫的建立和查詢方法
4. 實踐 LLM 在實際應用中的整合

## 📚 學習路徑

### 1️⃣ 資料收集和預處理階段

首先，我們使用 `composite_builder_cli.py` 來爬取和處理網頁資料：

```bash
# 使用 OpenAI 處理網頁內容
python scripts/composite_builder_cli.py https://example.com \
    --llm_provider openai \
    --openai_api_key YOUR_OPENAI_API_KEY \
    --openai_model gpt-3.5-turbo \
    --max_pages 10 \
    --out output/processed_data.json

# 或使用 Ollama 本地模型
python scripts/composite_builder_cli.py https://example.com \
    --llm_provider ollama \
    --ollama_url http://your.ollama.server:11434/v1 \
    --ollama_model qwen2.5vl:7b \
    --max_pages 10
```

這個步驟會：
- 爬取指定網站的內容
- 使用 LLM 分析和結構化網頁內容
- 生成 JSON 格式的資料檔案

### 2️⃣ 建立向量索引

使用處理好的資料建立向量索引，可以選擇直接使用或保存供後續使用：

```bash
# 建立索引並保存
python scripts/rag_cli.py \
    --embeddings output/processed_data.json \
    --query "測試查詢" \
    --save_index output/my_index \
    --llm_provider ollama \
    --ollama_url http://your.ollama.server:11434/v1 \
    --ollama_model qwen3:0.6b
```

這個步驟會：
- 讀取結構化的 JSON 資料
- 計算文本的向量嵌入
- 建立 FAISS 索引
- 將索引保存到指定位置

### 3️⃣ 執行查詢

現在可以開始進行實際的查詢操作：

```bash
# 使用現有索引進行查詢
python scripts/rag_cli.py \
    --load_index output/my_index \
    --query "你的問題" \
    --llm_provider ollama \
    --ollama_url http://your.ollama.server:11434/v1 \
    --ollama_model qwen3:0.6b
```

這個步驟會：
- 載入預建立的索引
- 將查詢轉換為向量
- 在索引中搜尋相關內容
- 使用 LLM 生成最終答案

## 🛠️ 命令行參數說明

### composite_builder_cli.py 參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| url | 起始網址 | 必填 |
| --max_pages | 最大爬取頁數 | 5 |
| --llm_provider | LLM 提供者 (openai/ollama/azure) | openai |
| --out | 輸出文件路徑 | ../output/composite_v2.json |
| --debug | 啟用調試模式 | False |

### rag_cli.py 參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| --embeddings | 嵌入 JSON 文件路徑 | 必填（除非使用 --load_index）|
| --query | 查詢文字 | 必填 |
| --top_k | 返回結果數量 | 5 |
| --save_index | 索引保存路徑 | 無 |
| --load_index | 載入已有索引路徑 | 無 |

## 系統要求

- Python 3.8+
- FAISS
- OpenAI Python SDK
- 其他依賴套件（見 requirements.txt）

## 安裝說明

1. 克隆專案：
```bash
git clone https://github.com/RichTsai1688/RAG_teach_demo.git
cd RAG_teach_demo
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

## 使用方法

### 使用 RAG CLI

RAG CLI 提供了一個便捷的方式來執行文件查詢：

1. 使用 OpenAI：
```bash
python scripts/rag_cli.py \
    --embeddings output/sections_embeddings.json \
    --query "你的問題" \
    --llm_provider openai \
    --openai_api_key YOUR_OPENAI_API_KEY \
    --openai_model gpt-3.5-turbo
```

2. 使用 Ollama：
```bash
python scripts/rag_cli.py \
    --embeddings output/sections_embeddings.json \
    --query "你的問題" \
    --llm_provider ollama \
    --ollama_url http://your.ollama.server:11434/v1 \
    --ollama_model YOUR_MODEL_NAME
```

### 使用網頁爬取工具

使用 Composite Builder CLI 來爬取網頁並生成結構化資料：

```bash
python scripts/composite_builder_cli.py \
    https://example.com \
    --llm_provider openai \
    --openai_api_key YOUR_OPENAI_API_KEY \
    --openai_model gpt-3.5-turbo \
    --max_pages 10
```

## 專案結構

```
docs/               # 文件
lib/                # 核心程式庫
  ├── rag_system.py           # RAG 系統主要實現
  ├── composite_element_builder.py    # 網頁爬取和解析
  └── composite_element_builder_v2.py # 增強版網頁爬取
output/             # 輸出文件
scripts/            # 命令行工具
  ├── rag_cli.py             # RAG 系統 CLI
  ├── composite_builder_cli.py    # 網頁爬取 CLI
  └── build_text_embeddings_json_multi.py  # 文本嵌入生成工具
```

## 環境變數

為了安全性考慮，建議使用環境變數來設置 API 金鑰：

```bash
export OPENAI_API_KEY=your_api_key
export AZURE_API_KEY=your_azure_key
```

## 💡 進階使用技巧

1. **批量處理**：
   - 使用腳本處理多個網站
   - 合併多個索引
   
2. **優化搜索結果**：
   - 調整 top_k 參數
   - 使用不同的 LLM 模型
   
3. **調試模式**：
   ```bash
   # 啟用調試模式查看詳細過程
   python scripts/composite_builder_cli.py https://example.com --debug --max_pages 3
   ```

## 🔒 安全性考慮

為了安全性考慮，建議：

1. 使用環境變數設置 API 金鑰：
```bash
export OPENAI_API_KEY=your_api_key
export AZURE_API_KEY=your_azure_key
```

2. 不要將含有 API 金鑰的命令保存在 shell 歷史記錄中
3. 確保生成的索引文件存放在安全位置

## 📈 效能考慮

- 建議先使用小數據集測試
- 大型網站爬取時注意設置適當的 max_pages
- 考慮使用本地 Ollama 模型降低成本

## 📝 故障排除

常見問題：
1. API 金鑰錯誤：檢查環境變數或參數設置
2. 索引載入失敗：確認檔案路徑和權限
3. 網頁爬取失敗：檢查網址和網路連接

使用順序：
Step 1. 建立資料庫，來自target網頁：
python scripts/composite_builder_cli.py "https://www.3t.org.tw/News2.aspx?n=541&sms=47411" \
                                            --llm_provider ollama \
                                            --ollama_url http://localhost:11434/v1 \
                                            --ollama_model localhost:11434 \
                                            --max_pages 10
Step 2. 建立搜尋索引，與測試搜尋
python scripts/rag_cli.py --embeddings output/composite_v2.json \
                    --save_index output/composite_v2_index \
                    --query "介紹一下T大使" \
                    --llm_provider ollama \
                    --ollama_url http://localhost:11434/v1 \
                    --ollama_model gemma3:1b \
                    --top_k 5

Step 3. 測試索引與測試搜尋
python scripts/rag_cli.py --load_index output/composite_v2_index \
                    --query "介紹一下培育對象" \
                    --llm_provider ollama \
                    --ollama_url http://localhost:11434/v1 \
                    --ollama_model gemma3:1b \
                    --top_k 10


## 📄 授權條款

MIT License

Copyright (c) 2025 RAG_teach_demo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
