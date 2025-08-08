#!/usr/bin/env python3
"""
composite_builder_cli.py

CompositeElementBuilder 的命令行介面，用於爬取網頁並生成結構化 JSON 資料。
"""
import sys
import argparse
from pathlib import Path

# 添加庫目錄到路徑
sys.path.append(str(Path(__file__).parent.parent / 'lib'))

from composite_element_builder_v2 import CompositeElementBuilder

def main():
    parser = argparse.ArgumentParser(description='Enhanced CompositeElementBuilder CLI')
    parser.add_argument('url', help='Start URL')
    parser.add_argument('-o', '--out', type=Path, default=Path('../output/composite_v2.json'))
    parser.add_argument('--max_pages', type=int, default=5)
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # LLM 提供者選擇
    parser.add_argument('--llm_provider', choices=['openai', 'ollama', 'azure'], 
                        default='openai', help='LLM provider')
    
    # OpenAI 參數
    parser.add_argument('--openai_api_key', help='OpenAI API key')
    parser.add_argument('--openai_model', help='OpenAI model name')
    
    # Ollama 參數
    parser.add_argument('--ollama_url', help='Ollama server URL')
    parser.add_argument('--ollama_model', help='Ollama model name')
    
    # Azure OpenAI 參數
    parser.add_argument('--azure_api_key', help='Azure OpenAI API key')
    parser.add_argument('--azure_endpoint', help='Azure OpenAI endpoint')
    parser.add_argument('--azure_deployment', help='Azure OpenAI deployment name')
    
    args = parser.parse_args()

    # 確保輸出目錄存在
    out_path = args.out
    if not out_path.is_absolute():
        out_path = Path(__file__).parent.parent / 'output' / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    builder = CompositeElementBuilder(
        start_url=args.url,
        out_path=out_path,
        max_pages=args.max_pages,
        llm_provider=args.llm_provider,
        openai_api_key=args.openai_api_key,
        openai_model=args.openai_model,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        azure_api_key=args.azure_api_key,
        azure_endpoint=args.azure_endpoint,
        azure_deployment=args.azure_deployment,
        debug_mode=args.debug
    )
    builder.build()

if __name__ == '__main__':
    """
    CLI usage examples:
    
    使用 OpenAI:
    $ python composite_builder_cli.py https://example.com --llm_provider openai --openai_api_key YOUR_OPENAI_API_KEY --openai_model gpt-3.5-turbo --max_pages 10
    
    使用 Ollama:
    $ python composite_builder_cli.py https://example.com --llm_provider ollama --ollama_url http://your.ollama.server:11434/v1 --ollama_model qwen2.5vl:7b --max_pages 10
    
    啟用調試模式:
    $ python composite_builder_cli.py https://example.com --debug --max_pages 3
    """
    main()
