import asyncio
import json
import os
import platform
from typing import Dict, List, Any, Optional
import nest_asyncio

# 处理Windows异步IO策略
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# 允许嵌套事件循环
nest_asyncio.apply()

from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# 加载环境变量
load_dotenv(override=True)

## Memory Saver
#memory = MemorySaver()
# --- 配置 ---
LLM_MODEL =  os.getenv("MODEL_NAME")
LLM_API_KEY = os.getenv("MODEL_API_KEY")
LLM_BASE_URL = os.getenv("API_BASE_URL") # 可选

# 配置文件路径
CONFIG_FILE_PATH = "tools_config/config.json"

# 系统提示词
SYSTEM_PROMPT = """<ROLE>
You are a smart agent with an ability to use tools. 
You will be given a question and you will use the tools to answer the question.
Pick the most relevant tool to answer the question. 
If you are failed to answer the question, try different tools to get context.
Your answer should be very polite and professional.
</ROLE>

----

<INSTRUCTIONS>
Step 1: Analyze the question
- Analyze user's question and final goal.
- If the user's question is consist of multiple sub-questions, split them into smaller sub-questions.

Step 2: Pick the most relevant tool
- Pick the most relevant tool to answer the question.
- If you are failed to answer the question, try different tools to get context.

Step 3: Answer the question
- Answer the question in the same language as the question.
- Your answer should be very polite and professional.

Step 4: Provide the source of the answer(if applicable)
- If you've used the tool, provide the source of the answer.
- Valid sources are either a website(URL) or a document(PDF, etc).

Guidelines:
- If you've used the tool, your answer should be based on the tool's output(tool's output is more important than your own knowledge).
- If you've used the tool, and the source is valid URL, provide the source(URL) of the answer.
- Skip providing the source if the source is not URL.
- Answer in the same language as the question.
- Answer should be concise and to the point.
- Avoid response your output with any other information than the answer and the source.  
</INSTRUCTIONS>

----

<OUTPUT_FORMAT>
(concise answer to the question)

**Source**(if applicable)
- (source1: valid URL)
- (source2: valid URL)
- ...
</OUTPUT_FORMAT>
"""

# 模型输出token信息
OUTPUT_TOKEN_INFO = {
    "claude-3-5-sonnet-latest": {"max_tokens": 8192},
    "claude-3-5-haiku-latest": {"max_tokens": 8192},
    "claude-3-7-sonnet-latest": {"max_tokens": 64000},
    "gpt-4o": {"max_tokens": 16000},
    "gpt-4o-mini": {"max_tokens": 16000},
}

def load_config_from_json(config_path: str = CONFIG_FILE_PATH) -> Dict:
    """
    从JSON文件加载配置。如果文件不存在，则创建默认配置文件。

    Args:
        config_path: 配置文件路径

    Returns:
        dict: 加载的配置
    """
    default_config = {
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # 如果文件不存在，创建默认配置文件
            save_config_to_json(default_config, config_path)
            return default_config
    except Exception as e:
        print(f"Error loading settings file: {str(e)}")
        return default_config

def save_config_to_json(config: Dict, config_path: str = CONFIG_FILE_PATH) -> bool:
    """
    保存配置到JSON文件。

    Args:
        config: 要保存的配置
        config_path: 配置文件路径
    
    Returns:
        bool: 保存成功状态
    """
    try:
        # 读取现有配置
        existing_config = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                existing_config = json.load(f)
        
        # 合并配置(新配置会覆盖旧配置)
        existing_config.update(config)
        
        # 写入合并后的配置
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(existing_config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
    
   
        print(f"Error saving settings file: {str(e)}")
        return False

def generate_random_uuid() -> str:
    """生成随机UUID"""
    import uuid
    return str(uuid.uuid4())

class MCPToolsManager:
    """MCP工具管理器"""
    
    def __init__(self):
        self.mcp_client = None
        self.agent = None
        self.thread_id = generate_random_uuid()
        self.recursion_limit = 100
        self.model_name = "claude-3-7-sonnet-latest"
        self.tools = []
        self.tool_count = 0
        
    async def cleanup_mcp_client(self):
        """安全终止现有MCP客户端"""
        if self.mcp_client is not None:
            try:
                await self.mcp_client.__aexit__(None, None, None)
                self.mcp_client = None
            except Exception as e:
                import traceback
                print(f"Error while terminating MCP client: {str(e)}")
                print(traceback.format_exc())
    
    async def initialize_session(self, mcp_config=None, model_name=None):
        """
        初始化MCP会话和Agent

        Args:
            mcp_config: MCP工具配置信息(JSON)。如果为None则使用默认设置
            model_name: 要使用的模型名称

        Returns:
            bool: 初始化成功状态
        """
        print("🔄 Connecting to MCP server...")
        
        # 首先安全清理现有客户端
        await self.cleanup_mcp_client()

        if mcp_config is None:
            # 从配置文件加载设置
            mcp_config = load_config_from_json()
            
        if model_name:
            self.model_name = model_name
            
        print("初始化MCP客户端",mcp_config)
        client = MultiServerMCPClient(mcp_config)
        await client.__aenter__()
        self.tools = client.get_tools()
        print("Available tools from server:", [tool.name for tool in self.tools ])
       # print(self.tools)
        self.tool_count = len(self.tools)
        self.mcp_client = client

        # 根据选择初始化适当的模型
       
     
            # 默认使用自定义模型配置
        model_kwargs = {
        "model": LLM_MODEL,
        "api_key": LLM_API_KEY,
        "base_url": LLM_BASE_URL,
        "temperature": 0.1,
    }
        print(model_kwargs)
        model = ChatOpenAI(**model_kwargs)
        
        # 创建Agent
        agent = create_react_agent(
            model,
            self.tools,
            checkpointer=MemorySaver(),
            #prompt=SYSTEM_PROMPT,
        )
        self.agent = agent
        self.config=RunnableConfig(thread_id=self.thread_id)
        print(f"✅ MCP session initialized with {self.tool_count} tools")
        return True
    
    async def process_query(self, query: str, timeout_seconds: int = 60) -> Dict:
        """
        处理用户问题并生成响应

        Args:
            query: 用户输入的问题文本
            timeout_seconds: 响应生成时间限制(秒)

        Returns:
            Dict: 包含响应内容的字典
        """
        try:
            if not self.agent:
                return {"error": "🚫 Agent has not been initialized."}
                
            # 确保返回字典
            result = {}
            inputs = {"messages": [SYSTEM_PROMPT, HumanMessage(content=query)]}
            
            async for event in self.agent.astream_events(inputs, self.config, version="v2"):
                if event["event"] == "on_tool_start":
                    print(f"\n### Start Tool: `{event['name']}`, Tool Input: `{event['data'].get('input')}`\n")
                elif event["event"] == "on_tool_end":
                    print(f"\n### Finished Tool: `{event['name']}`")
                    print(f"Tool output: {event['data'].get('output')}\n----")
                elif event["event"] == "on_chat_model_stream":
                    # 收集模型输出
                    if "content" not in result:
                        result["content"] = ""
                    result["content"] += event["data"]["chunk"].content
            
            print("--- End Agent Execution Steps ---\n")
            return result if result else {"content": "✅ 请求已处理，但未返回有效内容"}
            
        except asyncio.TimeoutError:
            return {"error": f"⏱️ Request time exceeded {timeout_seconds} seconds"}
        except Exception as e:
            return {"error": f"❌ Error occurred: {str(e)}"}
    
    async def chat(self, query: str, timeout_seconds: int = 60) -> str:
        """
        高级聊天接口，处理用户输入并返回响应
        """
        response = await self.process_query(query, timeout_seconds)
        if not response or "error" in response:
            return f"❌ 错误: {response.get('error', '未知错误')}"
        return response.get("content", "✅ 请求已处理，但未返回有效内容")

    def add_tool(self, tool_json: str) -> Dict[str, Any]:
        """
        添加工具配置
        
        Args:
            tool_json: 工具配置的JSON字符串
            
        Returns:
            Dict: 包含操作结果的字典
        """
        try:
            # 解析JSON
            parsed_tool = json.loads(tool_json)
            print(f"Parsed tool configuration: {parsed_tool}")
            # 检查是否是mcpServers格式
            if "mcpServers" in parsed_tool:
                parsed_tool = parsed_tool["mcpServers"]
                
            # 检查工具数量
            if len(parsed_tool) == 0:
                return {"success": False, "message": "Please enter at least one tool."}
                
            # 加载当前配置
            current_config = load_config_from_json()
            
            # 处理所有工具
            success_tools = []
            for tool_name, tool_config in parsed_tool.items():
                # 检查URL字段并设置传输方式
                if "url" in tool_config:
                    tool_config["transport"] = "sse"
                elif "transport" not in tool_config:
                    tool_config["transport"] = "stdio"
                    
                # 检查必要字段
                if "command" not in tool_config and "url" not in tool_config:
                    return {
                        "success": False, 
                        "message": f"'{tool_name}' tool configuration requires either 'command' or 'url' field."
                    }
                elif "command" in tool_config and "args" not in tool_config:
                    return {
                        "success": False, 
                        "message": f"'{tool_name}' tool configuration requires 'args' field."
                    }
                elif "command" in tool_config and not isinstance(tool_config["args"], list):
                    return {
                        "success": False, 
                        "message": f"'args' field in '{tool_name}' tool must be an array ([]) format."
                    }
                else:
                    # 添加工具到配置
                    current_config[tool_name] = tool_config
                    success_tools.append(tool_name)
                    
            # 保存配置
            save_result = save_config_to_json(current_config)
            if not save_result:
                return {"success": False, "message": "Failed to save configuration."}
                
            # 成功消息
            if success_tools:
                if len(success_tools) == 1:
                    return {
                        "success": True, 
                        "message": f"{success_tools[0]} tool has been added.",
                        "tools": success_tools
                    }
                else:
                    tool_names = ", ".join(success_tools)
                    return {
                        "success": True, 
                        "message": f"Total {len(success_tools)} tools ({tool_names}) have been added.",
                        "tools": success_tools
                    }
            else:
                return {"success": False, "message": "No tools were added."}
                
        except json.JSONDecodeError as e:
            return {"success": False, "message": f"JSON parsing error: {e}"}
        except Exception as e:
            return {"success": False, "message": f"Error occurred: {e}"}
    
    def delete_tool(self, tool_name: str) -> Dict[str, Any]:
        """
        删除工具配置
        
        Args:
            tool_name: 要删除的工具名称
            
        Returns:
            Dict: 包含操作结果的字典
        """
        try:
            # 加载当前配置
            current_config = load_config_from_json()
            
            # 检查工具是否存在
            if tool_name not in current_config:
                return {"success": False, "message": f"Tool '{tool_name}' not found."}
                
            # 删除工具
            del current_config[tool_name]
            
            # 保存配置
            save_result = save_config_to_json(current_config)
            if not save_result:
                return {"success": False, "message": "Failed to save configuration."}
                
            return {"success": True, "message": f"{tool_name} tool has been deleted."}
            
        except Exception as e:
            return {"success": False, "message": f"Error occurred: {e}"}
    
    def list_tools(self) -> Dict[str, Any]:
        """
        列出所有已注册的工具
        
        Returns:
            Dict: 包含工具列表的字典
        """
        try:
            # 加载当前配置
            current_config = load_config_from_json()
            
            return {
                "success": True, 
                "tools": list(current_config.keys()),
                "count": len(current_config),
                "config": current_config
            }
            
        except Exception as e:
            return {"success": False, "message": f"Error occurred: {e}"}
    
    def reset_conversation(self):
        """重置对话"""
        self.thread_id = generate_random_uuid()
        print("✅ Conversation has been reset.")

   
def generate_mcp_tool_file(tool_name: str, tool_code: str, output_dir: str = "tools") -> str:
    """
    将传入的代码转换为带有 @mcp.tool() 装饰器的 Python 文件
    
    Args:
        tool_name: 工具名称(作为文件名)
        tool_code: 要转换的代码字符串
        output_dir: 输出目录
        
    Returns:
        str: 生成的完整文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件头部
    file_content = f'''import json
from mcp.server.fastmcp import FastMCP
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 MCP 实例
mcp = FastMCP("{tool_name}", description="自动生成的工具")

@mcp.tool()
{tool_code}

if __name__ == "__main__":
    logger.info("Starting {tool_name} server...")
    # 默认使用 stdio 传输
    mcp.run(transport="stdio")
    logger.info("{tool_name} server stopped.")
'''
    
    # 写入文件
    file_path = os.path.join(output_dir, f"{tool_name}.py")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(file_content)
    
    return file_path


async def main():

    
    os.makedirs("tools_config", exist_ok=True)
     # 1. 首先列出当前所有工具
  
    # 示例1: 添加/更新工具
    
    input_data = {
    "name": "AWP",
    "instructions": "对服务器执行巡检。",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "xxxxxi",
                "description": "xxxx。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ip": {
                            "type": "string",
                            "description": "服务器IP地址"
                        }
                    },
                    "required": ["ip"]
                }
            },
            "code": "your code "
        }
    ]
}

# 生成工具文件
    script_path = generate_mcp_tool_file("awp_inspect", input_data["tools"][0]["code"])
    print(f"工具文件已生成: {script_path}")
   # exit()
    tool_config = json.dumps({
    "awp_inspect": {
        "command": "python",
        "args": [f"./tools/{os.path.basename(script_path)}"],
        "transport": "stdio",

    }
})
    
     # 添加工具
    manager = MCPToolsManager()
    await manager.initialize_session()
    print('开始添加')
    result = manager.add_tool(tool_config)
    print("添加工具结果:", json.dumps(result, indent=2))
    config_dict = json.loads(tool_config)
    tool_name = next(iter(config_dict.keys()))
    # 示例2: 删除工具
    result = manager.delete_tool(tool_name)
    print(json.dumps(result, indent=2))
    print("\n删除工具结果:", json.dumps(result, indent=2))
    
    # 再次列出工具确认已删除
    tools = manager.list_tools()
    print("\n工具列表:", json.dumps(tools, indent=2))
    # 示例3: 聊天交互
    #response = await manager.chat("现在几点了?")
    # print(response)
    
    await manager.cleanup_mcp_client()

if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())