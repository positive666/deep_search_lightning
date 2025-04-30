import asyncio
import os
from typing import AsyncGenerator
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json
import uuid


# 加载环境变量
load_dotenv()
## Memory Saver
#memory = MemorySaver()
# --- 配置 ---
LLM_MODEL =  os.getenv("MODEL_NAME")
LLM_API_KEY = os.getenv("MODEL_API_KEY")
LLM_BASE_URL = os.getenv("API_BASE_URL") # 可选

# 新的搜索服务器地址和端口
SEARCH_SERVER_URL = "http://localhost:8000/sse" # 端口改为 8002

async def run_agent(query: str):

    if not LLM_API_KEY:
        raise ValueError("LLM API Key is not set. Please set OPENAI_API_KEY environment variable.")

    model_kwargs = {
        "model": LLM_MODEL,
        "api_key": LLM_API_KEY,
        "base_url": LLM_BASE_URL,
        "temperature": 0.1,
    }

    model = ChatOpenAI(**model_kwargs)

    # connect to the search server using SSE (Server-Sent Events)
    async with MultiServerMCPClient(
        {
            "web_search_tools": { # 服务名称与服务器匹配
                "url": SEARCH_SERVER_URL,
                "transport": "sse",
            },
            
        }
    ) as client:
        # load tools from server
        try:
            tools = client.get_tools()
            print("Available tools from server:", [tool.name for tool in tools])
            if not tools:
                 print(f"Warning: No tools loaded from the server at {SEARCH_SERVER_URL}. Check server status.")
                 return "Error: Could not load search tools."
        except Exception as e:
            print(f"Error loading tools from {SEARCH_SERVER_URL}: {e}")
            return f"Error loading tools: {e}"

        # React Agent
        try:
            agent_executor = create_react_agent(model, tools)
            
        except Exception as e:
            print(f"Error creating agent: {e}")
            return f"Error creating agent: {e}"

        system_message = SystemMessage(content=(
            "You are an AI assistant with web search capabilities. Consider using the web_search_tools when answering user questions.\n"
            "Available tools:\n"
            "- `web_search_tools`: Retrieves web information by selecting 3-5 most suitable search engines from ['baidu', 'duckduckgo','bocha','tavily'] based on the user's query.\n"
        ))
        # Process query
        #search_results = []
        try:
       
            input_messages = [system_message, HumanMessage(content=query)]
            #agent_response = await agent_executor.ainvoke({"messages": input_messages})
            inputs = { "messages": input_messages }
            print(f"Inputs: {inputs}")
            print("\n--- Agent Execution Steps ---")
            
            #agent_response=''
            async for event in agent_executor.astream_events(inputs, version="v2"):
               
                if event["event"]== "on_tool_start":
                        print(
                            f"\n### Start Tool: `{event['name']}`, Tool Input: `{event['data'].get('input')}`\n"
                        )
                elif event["event"] == "on_tool_end":
                         print(
                 f"\n### Finished Tool: `{event['name']}`, Tool Results: \n"
             )
                         print(f"Done tool: {event['name']}")
                         print(f"Tool output was: {event['data'].get('output')}")
                         print("----")
                elif event["event"] == "on_chat_model_stream":
                    print(event["data"]["chunk"].content, end="", flush=True) 
           
            print("--- End Agent Execution Steps ---\n")
      
        except Exception as e:
            import traceback
            print(f"Error invoking agent: {e}")
            traceback.print_exc() # 打印详细的回溯信息
            return f"Agent execution failed: {e}"

# run
if __name__ == "__main__":
    # user_query = input("请输入您的问题: ")
    user_query = " 搜一下 中国最好的大学是？" # 示例查询，适合聚合工具
    # user_query = "特斯拉今天的股价" # 示例查询，可能适合单个工具
    if user_query:
        print(f"Running agent for query: '{user_query}'")
        final_response = asyncio.run(run_agent(user_query))
        print("\nFinal Response from Agent:")
        print(final_response)
    else:
        print("No query provided.")