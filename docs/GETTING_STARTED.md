# RAG_v2 入門指南

這份指南將幫助您快速開始使用 RAG_v2 系統。

## 環境設置

1. 克隆儲存庫（如果尚未完成）：
   ```bash
   git clone <儲存庫URL>
   cd RAG_v2
   ```

2. 安裝依賴項：
   ```bash
   pip install -r requirements.txt
   ```

3. 確保 `docs/embedder.json` 檔案配置正確，包含您要使用的嵌入模型設定。

## 快速開始

### 網頁抓取和處理

從抓取網頁並生成結構化 JSON 開始：

```bash
python scripts/composite_builder_cli.py https://example.com \
  --llm_provider ollama \
  --ollama_url http://localhost:11434/v1 \
  --ollama_model llama2 \
  --max_pages 5 \
  --debug
```

這將會：
- 抓取 example.com 網站上最多 5 個頁面
- 提取文本、表格和圖片
- 生成嵌入向量
- 將結果保存為 JSON 檔案（預設在 output 目錄）

### 查詢處理後的內容

一旦您有了生成的 JSON 檔案，您可以使用 RAG 系統進行查詢：

```bash
python scripts/rag_cli.py \
  --embeddings output/composite_v2.json \
  --query "您的問題" \
  --llm_provider ollama \
  --ollama_url http://localhost:11434/v1 \
  --ollama_model llama2 \
  --top_k 5
```

這將會：
- 載入您的 JSON 檔案
- 使用您的問題查詢索引
- 找到相關的上下文
- 使用 LLM 生成答案

## 進階使用

查看 `README.md` 文件以獲取更多詳細信息，包括：
- 完整的命令行選項
- API 文檔
- 自定義配置
- 效能調整技巧

## 常見問題排解

1. **錯誤：找不到模組**
   
   確保您已安裝所有依賴項，並且從項目根目錄運行命令。

2. **錯誤：找不到 embedder.json**

   確保 `docs/embedder.json` 文件存在並且可讀。

3. **錯誤：OpenAI API 錯誤**

   檢查您的 API 密鑰和網絡連接。

4. **錯誤：Ollama 連接失敗**

   確保 Ollama 服務正在運行，並且 URL 正確。

5. **查詢結果不精確**

   嘗試調整 `top_k` 參數，或使用更精確的查詢關鍵詞。

## 獲取幫助

如果您遇到任何問題，請：
- 查看項目文檔
- 檢查項目 issues
- 在 issues 中提交新問題

祝您使用愉快！
