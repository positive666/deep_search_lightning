import asyncio
import logging
import json
import time
import os
from typing import List, Dict, Any, Optional
from src.deep_search_lightning import WebEnhancedLLM, SearchEngineConfig
import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import httpx
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 配置日志记录
logging.basicConfig(level=logging.ERROR)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

async def search_tavily(query: str, max_results: int = 5, chunks_per_source: int = 3) -> dict:
    """Performs a Tavily web search and returns specified number of results."""
    if not TAVILY_API_KEY:
        return {"error": "Tavily API key is missing. Set it in your .env file."}
    if not TAVILY_SEARCH_URL:
        return {"error": "Tavily search URL is missing."}

    payload = {
        "query": query,
        "topic": "general",
        "search_depth": "basic",
        "chunks_per_source": chunks_per_source,
        "max_results": max_results,
        "time_range": None,
        "days": 3,
        "include_answer": True,
        "include_raw_content": False,
        "include_images": False,
        "include_image_descriptions": False,
        "include_domains": [],
        "exclude_domains": []
    }

    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(TAVILY_SEARCH_URL, json=payload, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        return {"error": f"HTTP error: {http_err}"}
    except httpx.RequestError as req_err:
        logging.error(f"Request error occurred: {req_err}")
        return {"error": f"Request error: {req_err}"}
    except Exception as err:
        logging.error(f"An unexpected error occurred: {err}")
        return {"error": f"Unexpected error: {err}"}



def bocha_parse_response(response):
    # ... (与之前 server 版本相同) ...
    result = {}
    if "data" in response:
        data = response["data"]
        if "webPages" in data:
            webPages = data["webPages"]
            if "value" in webPages:
                result["webpage"] = [
                    {
                        "id": item.get("id", ""),
                        "name": item.get("name", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("snippet", ""),
                        "summary": item.get("summary", ""),
                        "siteName": item.get("siteName", ""),
                        "siteIcon": item.get("siteIcon", ""),
                        "datePublished": item.get("datePublished", "") or item.get("dateLastCrawled", ""),
                    }
                    for item in webPages["value"]
                ]
    return result

def bocha_search(
    query: str, count: int, api_key: str, filter_list: Optional[list[str]] = None
) -> list:
    # ... (与之前 server 版本相同, 确保 api_key 正确处理) ...
    url = "https://api.bochaai.com/v1/web-search" # 修正 URL
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = json.dumps({
        "query": query,
        # "summary": True, # Bocha API 可能不支持 summary，根据实际情况调整
        "freshness": "noLimit",
        "count": count
    })
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        results_data = bocha_parse_response(response.json())
        # 提取所需字段，优先用 summary，其次用 snippet
        return [
            {
                "title": result.get("name", ""),
                "href": result.get("url", ""),
                "body": result.get("summary", result.get("snippet", "")) # 使用 snippet 作为备选
            } for result in results_data.get("webpage", []) if result.get("url") # 确保有 URL
        ]
    except requests.exceptions.RequestException as e:
        logger.error(f"Bocha search error for query '{query}': {e}")
        return [] # 返回空列表表示错误
    except Exception as e:
        logger.error(f"Error processing Bocha response for query '{query}': {e}")
        return [] # 返回空列表表示错误



# --- FastMCP 服务器设置 ---
mcp = FastMCP("web_search_tools", description="聚合多个搜索引擎的搜索结果")

# Bocha API Key
BOCHA_API_KEY = os.getenv("BOCHA_API_KEY", "") # 替换为你的有效 Key

# --- 定义独立的搜索引擎工具 ---



@mcp.tool()
async def web_search_tools(
    messages: List[Dict[str, Any]]=[],
    search_engines = ['baidu', 'duckduckgo', 'bocha',"tavily"],
    #max_results_per_engine: int = 3,
   # max_final_results: int = 6
) -> Dict[str, Any]:
    """
    聚合多个搜索引擎 (Baidu, DuckDuckGo, Bocha) 的搜索结果。
    它会为每个关键词调用指定的搜索引擎，并将结果聚合在一起。

    Args:
        messages (List[Dict[str, Any]]): 对话列表，如[{"role": "system", "content": '你是个网络检索助手'},{"role": "user", "content": '你好，请帮我搜索一下 "Python"'}, {"role": "assistant", "content": '好的，完成了'}]】
        search_engines (List[str]): 根据问题类型自动适配合适的搜索源，从默认搜索源中确定(至少2个,最多5个)，如 ["baidu", "duckduckgo"]。
      
    Returns:
        Dict[str, Any]: 包含 'results' (聚合后的结果列表) 或 'error' (字符串)的字典。
                        结果包含 'title', 'href', 'body', 和 'relevance_score'。
    """
    load_dotenv()
    engine_config = {
    'baidu': SearchEngineConfig('baidu', max_results=int(os.getenv('BAIDU_MAX_RESULTS')), api_key=os.getenv('BAIDU_API_KEY'), enabled=os.getenv('BAIDU_ENABLED')== 'true'),
    'duckduckgo': SearchEngineConfig('duckduckgo', max_results=int(os.getenv('DUCKDUCKGO_MAX_RESULTS')),enabled=os.getenv('DUCKDUCKGO_ENABLED')== 'true'),
    'bocha': SearchEngineConfig('bocha', max_results=int(os.getenv('BOCHA_MAX_RESULTS')), api_key=os.getenv('BOCHA_API_KEY'), enabled=os.getenv('BOCHA_ENABLED')== 'true'),
    'tavily': SearchEngineConfig('tavily', max_results=int(os.getenv('TAVILY_MAX_RESULTS')), api_key=os.getenv('TAVILY_API_KEY'), enabled=os.getenv('TAVILY_ENABLED')== 'true'),
}
    if search_engines is None or len(search_engines) < 2:
       # return {"error": "至少需要两个搜索引擎进行聚合搜索。"}
       search_engines = ['baidu', 'duckduckgo']
    logger.info(f"use rearch_engines: {search_engines}")
            
    web_llm = WebEnhancedLLM(model=os.getenv('MODEL_NAME'),
                                 search_engine=search_engines ,language='zh',
                                 api_key=os.getenv('MODEL_API_KEY'),
                                 api_base_url=os.getenv('API_BASE_URL'),
                                
                                depth=1,engine_config=engine_config,width=3,search_max_results=8)

    context = web_llm.format_context(messages[:-1]) #获取历史提问
    print("context:",context)
    sub_tasks,_=web_llm.rewire_query(context,messages[-1]['content'],width=3)
   # print('rewrite_query Done')
    search_results,ref= await web_llm.search_with_context(sub_tasks,enable_reflection=True) #默认聚合搜索&深度反思
    logger.info("Search results: %s", search_results)
    return {"results": search_results, "references": ref}
    
   


if __name__ == "__main__":
    logger.info("Starting search aggregator server...")
    # 运行在 8000 端口，使用 SSE
    mcp.run(transport="sse")
    logger.info("Search aggregator server stopped.")