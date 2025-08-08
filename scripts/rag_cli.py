#!/usr/bin/env python3
"""
rag_cli.py

RAG 系統的命令行介面，支援建立嵌入索引和執行查詢。
"""
import sys
import argparse
import json
from pathlib import Path
import os

# 添加庫目錄到路徑
sys.path.append(str(Path(__file__).parent.parent / 'lib'))

from rag_system import RAGSystem

def main():
    parser = argparse.ArgumentParser(description='RAGSystem CLI')
    parser.add_argument('--embeddings', type=str, required=False, help='Path to embeddings JSON')
    parser.add_argument('--query', type=str, required=True, help='Query string')
    parser.add_argument('--top_k', type=int, default=5, help='Number of results')
    parser.add_argument('--model_path', type=str, default='sentence-transformers/all-MiniLM-L6-v2', 
                      help='Model path (unused, kept for compatibility)')
    parser.add_argument('--embedding_dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--save_index', type=str, default=None, help='Path to save the FAISS index')
    parser.add_argument('--load_index', type=str, default=None, help='Path to load a pre-built FAISS index')
    parser.add_argument('--llm_provider', choices=['openai','ollama'], default='openai', help='LLM provider')
    # OpenAI parameters
    parser.add_argument('--openai_api_key', required=False, help='OpenAI API key')
    parser.add_argument('--openai_model', required=False, help='OpenAI chat model')
    # Ollama parameters
    parser.add_argument('--ollama_url', required=False, help='Ollama server URL')
    parser.add_argument('--ollama_model', required=False, help='Ollama model name')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # 檢查參數合理性
    if not args.embeddings and not args.load_index:
        print("錯誤: 必須提供 --embeddings 或 --load_index 參數")
        sys.exit(1)
        
    # 建立 RAG 系統，可選擇載入現有索引
    rag = RAGSystem(
        model_path=args.model_path,
        embedding_dimension=args.embedding_dim,
        llm_provider=args.llm_provider,
        openai_api_key=args.openai_api_key,
        openai_model=args.openai_model,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        debug_mode=args.debug,
        index_file_path=args.load_index
    )
    
    # 如果提供了 embeddings 參數且沒有成功載入索引，則處理數據
    if args.embeddings and (not args.load_index or not rag.index.ntotal):
        # 載入資料
        entries = json.load(open(args.embeddings, 'r', encoding='utf-8'))
        print(f"載入 {len(entries)} 個項目...")
        
        # 建立索引並可選擇保存
        try:
            rag.ingest_entries(entries, save_index_path=args.save_index)
            print(f"索引建立完成，包含 {rag.index.ntotal} 個向量")
        except Exception as e:
            print(f"索引建立過程中發生錯誤: {e}")
            if args.save_index:
                print("嘗試保存已處理的部分...")
                try:
                    if rag.save_index(args.save_index):
                        print(f"部分索引已保存到 {args.save_index}")
                    else:
                        print(f"無法保存部分索引")
                except Exception as save_err:
                    print(f"保存部分索引時發生錯誤: {save_err}")
            
            if not args.query:
                sys.exit(1)
    elif rag.index.ntotal > 0:
        print(f"使用已載入的索引，包含 {rag.index.ntotal} 個向量")
    
    # 執行查詢
    if args.query:
        try:
            results = rag.query(args.query, args.top_k)
            rag.display_results(results)
        except Exception as e:
            print(f"查詢過程中發生錯誤: {e}")
            sys.exit(1)

if __name__ == '__main__':
    """
    CLI usage examples:
    
    使用 OpenAI:
    $ python rag_cli.py --embeddings output/sections_embeddings.json --query "你的問題" --llm_provider openai --openai_api_key YOUR_OPENAI_API_KEY --openai_model gpt-3.5-turbo
    
    使用 Ollama:
    $ python rag_cli.py --embeddings output/gear_full_output.json --query "幫我找系統表格" --llm_provider ollama --ollama_url http://your.ollama.server:11434/v1 --ollama_model qwen3:30b --top_k 20
    
    啟用調試模式:
    $ python rag_cli.py --embeddings output/sections_embeddings.json --query "你的問題" --llm_provider ollama --ollama_url http://your.ollama.server:11434/v1 --ollama_model qwen3:8b --debug
    
    保存索引:
    $ python scripts/rag_cli.py --embeddings output/gear_composite.json --query "你是什麼公司" --save_index output/gear_index --llm_provider ollama --ollama_url http://your.ollama.server:11434/v1 --ollama_model qwen3:0.6b
    
    載入現有索引:
    $ python scripts/rag_cli.py --load_index output/gear_index --query "機械設備有哪些資訊" --llm_provider ollama --ollama_url http://your.ollama.server:11434/v1 --ollama_model qwen3:0.6b
    """
    main()
