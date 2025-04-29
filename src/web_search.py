from typing import Optional
from baidusearch.baidusearch import search as search_baidu
from duckduckgo_search import DDGS

import requests
import json
import os



def baidu_search(query,max_results=5):
    
    response= search_baidu(query)[:max_results]
    results = [
                    {"title": r["title"], "href": r["url"], "body": r.get("abstract", "")}
                    for r in response
                ]
    
    return results

def duckduckgo_search(query: str, max_results: int = 5) -> list:
    with DDGS() as ddgs:
         results = [r for r in ddgs.text(query, max_results=max_results)]
         return results
def bocha_parse_response(response):
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
    query: str, count: int, api_key:str=os.environ.get("BOCHA_API_KEY"),filter_list: Optional[list[str]] = None
) -> list:
    """Search using Bocha's Search API and return the results as a list of SearchResult objects.

    Args:
        api_key (str): A Bocha Search API key
        query (str): The query to search for
    """
    url = "https://api.bochaai.com/v1/web-search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = json.dumps({
        "query": query,
        "summary": True,
        "freshness": "noLimit",
        "count": count
    })

    response = requests.post(url, headers=headers, data=payload, timeout=5)
    response.raise_for_status()
    results = bocha_parse_response(response.json())

    return [
        {
            "title": result.get("name", ""),
            "href": result.get("url", ""),
            "body": result.get("summary", "")
        } for result in results.get("webpage", [])
    ]
    
import httpx
import logging

# 配置日志记录
logging.basicConfig(level=logging.ERROR)

#
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

def search_tavily_sync(query: str, max_results: int = 5, chunks_per_source: int = 3, api_key: str = os.environ.get("TAVILY_API_KEY")) -> dict:
    """同步版本的Tavily搜索"""
    if not api_key:
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
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        with httpx.Client() as client:
            response = client.post(TAVILY_SEARCH_URL, json=payload, headers=headers, timeout=10.0)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        return {"results": []}
    except httpx.RequestError as req_err:
        logging.error(f"Request error occurred: {req_err}")
        return {"results": []}
    except Exception as err:
        logging.error(f"An unexpected error occurred: {err}")
        return {"results": []}

def get_tavily_results_sync(query: str, max_results: int = 3, chunks_per_source: int = 3):
    """同步版本的获取Tavily搜索结果"""
    results = search_tavily_sync(query, max_results, chunks_per_source)

    if isinstance(results, dict):
        return {"results": results.get("results", [])}
    else:
        return {"error": "Unexpected Tavily response format"}

# 保留原有的异步函数
def search_tavily(query: str, max_results: int = 5, chunks_per_source: int = 3, api_key: str =os.environ.get("TAVILY_API_KEY")) -> dict:
    """Performs a Tavily web search and returns specified number of results."""
    if not api_key:
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
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    print ("Searching Tavily for: ", query)
    try:
        with httpx.Client(timeout=30.0)  as client:
            response = client.post(TAVILY_SEARCH_URL, json=payload, headers=headers, timeout=30.0)
           
            response.raise_for_status()
            print( response)
            return [
            {
                "title": result.get("title", ""),
                "href": result.get("url", ""),
                "body": result.get("content", "") or result.get("description", "")
            }
            for result in response.json().get("results", [])
        ]
            
    except httpx.HTTPStatusError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        return {"error": f"HTTP error: {http_err}"}
    except httpx.RequestError as req_err:
        logging.error(f"Request error occurred: {req_err}")
        return {"error": f"Request error: {req_err}"}
    except Exception as err:
        logging.error(f"An unexpected error occurred: {err}")
        return {"error": f"Unexpected error: {err}"}

