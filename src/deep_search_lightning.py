import requests
import asyncio
import json

import time
from datetime import date, datetime, timezone
from typing import List, Dict, Any, Optional, AsyncGenerator
import logging
import re
from .llm_base import LLMBase,AsyncOpenAI,OpenAI
from .web_search import bocha_search, baidu_search,search_tavily,duckduckgo_search
from bs4 import BeautifulSoup
from .prompt import PromptManager
# 定义Web搜索工具
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 确保有控制台输出处理器
    ]
)
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)  # 显式设置日志级别



class SearchEngineConfig:
    def __init__(self, engine_name: str, max_results: int = 5, enabled: bool = True, api_key: str = None):
        self.engine_name = engine_name
        self.max_results = max_results
        self.enabled = enabled
        self.api_key = api_key

    def to_dict(self):
        return {
            'max_results': self.max_results,
            'enabled': self.enabled,
            'api_key': self.api_key
        }
class WebSearchTool:
    
    def __init__(self, search_engine:str='duckduckgo', engine_config: Dict[str, SearchEngineConfig] = None):
        
        self.logger = logging.getLogger(f"{__name__}.WebSearchTool")
        self.logger.setLevel(logging.INFO)  # 设置工具类日志级别
        self.SUPPORTED_ENGINES = {
            'duckduckgo': SearchEngineConfig('duckduckgo', max_results=5),
            'bocha': SearchEngineConfig('bocha', max_results=5,api_key="default_bocha_key"), 
            'baidu': SearchEngineConfig('baidu', max_results=5),
            'tavily': SearchEngineConfig('tavily', max_results=5,api_key="default_tavily_key")  
        }
        #  更新引擎配置
        if engine_config:
            for engine, config in engine_config.items():
                if engine in self.SUPPORTED_ENGINES:
                    self.SUPPORTED_ENGINES[engine] = SearchEngineConfig(
                        engine,
                        max_results=config.max_results,
                        api_key=config.api_key,
                        enabled=config.enabled
                    )
       
         # 验证搜索引擎
        if search_engine not in self.SUPPORTED_ENGINES:
            raise ValueError(f"Unsupported search engine: {search_engine}")
        self.search_engine = search_engine
        self.max_results = self.SUPPORTED_ENGINES[search_engine].max_results
        self.last_request_time = 0
        self.min_request_interval = 2  # 最小请求间隔(秒)
    
    async def _rate_limit(self):
        """请求限速控制"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
        
    def get_supported_engines(self) -> List[str]:
        """获取当前启用的搜索引擎列表"""
        return [engine for engine in self.SUPPORTED_ENGINES if self.SUPPORTED_ENGINES[engine].enabled]  # 直接访问enabled属性
    def _is_high_quality_result(self, result: dict) -> bool:
        """判断结果是否高质量"""
        # 过滤掉标题或内容过短的结果
        if len(result.get("title", "")) < 5 or len(result.get("body", "")) < 10:
            return False
        return True
    def _calculate_relevance_score(self, result: dict, keywords: List[str]) -> float:
        """计算搜索结果的相关性评分"""
        score = 0.0
        title = result.get("title", "").lower()
        body = result.get("body", "").lower()
        
        # 关键词匹配评分
        for keyword in keywords:
            kw = keyword.lower()
            # 标题中的关键词权重更高
            if kw in title:
                score += 2.0
            if kw in body:
                score += 1.0
        return score
    async def search(self, query: str):
        search_engine = self.search_engine
        if not search_engine:
            raise ValueError("No search engine specified")
        
        engine_config = self.SUPPORTED_ENGINES.get(search_engine)
        print(f"engine_config: {search_engine}___{engine_config.enabled}")
        if not engine_config or not engine_config.enabled:
            
            logger.warning(f"Search engine {search_engine} is disabled or not found")
            return self._generate_result_dict(
                "error", 
                query, 
                [], 
                time.time(), 
                f"Search engine {search_engine} is disabled"
            )
        
        try:
            
            if search_engine == 'duckduckgo':
                    results = duckduckgo_search(query, max_results=engine_config.max_results)
            elif search_engine == 'bocha':
                api_key = engine_config.api_key
                if not api_key:
                    logger.error("Bocha API key is required but not provided")
                    return self._generate_result_dict("error", query, [], time.time(), "Bocha API key is required")
                results = bocha_search(query, engine_config.max_results, api_key=api_key)
            elif search_engine == 'baidu':
                results=  baidu_search(query, engine_config.max_results)
            elif search_engine == 'tavily':
                api_key = engine_config.api_key
                if not api_key:
                    logger.error("Tavily API key is required but not provided")
                    return self._generate_result_dict("error", query, [], time.time(), "Tavily API key is required")
                results =  search_tavily(query, max_results=engine_config.max_results,api_key=api_key)
            else:
                raise ValueError(f"Unsupported search engine: {search_engine}")
            
            return self._generate_result_dict("success", query, results, time.time(), search_engine=search_engine)
        except Exception as e:
            logger.error(f"{search_engine}---Search error: {e}")
            return self._generate_result_dict("error", query, [], time.time(), str(e), search_engine=search_engine)
                                       
    async def aggregate_search_with_keywords(self, keywords: List[str], engines: List[str] = None) -> dict:
        """
        多关键词多引擎聚合搜索
        :param keywords: 关键词列表
        :param engines: 要使用的搜索引擎列表，如果为None则使用所有支持的引擎
        :return: 包含聚合结果的字典
        """
        engines = engines or self.get_supported_engines()
        search_start_time = time.time()
        try:
            # 为每个关键词创建搜索任务
            tasks = []
            for keyword in keywords:
                for engine in engines:
                    engine_max_results = self.SUPPORTED_ENGINES[engine].max_results
                    tasks.append(self._search_with_engine(
                        engine, 
                        keyword,
                        engine_max_results))
            # 并发执行所有搜索任务
            results = await asyncio.gather(*tasks)
            
            # 合并结果并去重
            merged_results = []
            seen_urls = set()
            seen_titles = set()  # 新增标题去重
            for engine_result in results:
                if engine_result["status"] == "success":
                    for item in engine_result["results"]:
                        if item["href"] not in seen_urls and item["title"] not in seen_titles:
                            seen_urls.add(item["href"])
                            seen_titles.add(item["title"])
                             # 质量过滤
                            if self._is_high_quality_result(item):
                                merged_results.append(item)        
            return self._generate_result_dict("success", keywords, merged_results, search_start_time)            
        except Exception as e:
            logger.error(f"Aggregate search with keywords error: {e}")
            return self._generate_result_dict("error", keywords, [], search_start_time, str(e))
    async def _search_with_engine(self, engine: str, query: str, max_results: int) -> dict:
        """使用指定引擎进行搜索"""
        try:
            logger.info(f"Searching {engine} for query: {query}")
            original_engine = self.search_engine
            original_max_results = self.max_results
            
            # 临时修改引擎和结果数限制
            self.search_engine = engine
            self.max_results = max_results
            
            result = await self.search(query)
            
            # 恢复原始配置
            self.search_engine = original_engine
            self.max_results = original_max_results
            
            return result
        except Exception as e:
            # 确保异常时也恢复配置
            self.search_engine = original_engine
            self.max_results = original_max_results
            raise e
   
    def _generate_result_dict(self, status: str, query: Any, results: List[Dict[str, Any]], search_start_time: float, message: str = "",search_engine: str = None):
        """
            生成统一的搜索结果字典。
            
            :param status: 搜索状态（"success" 或 "error"）
            :param query: 搜索关键词或关键词列表
            :param results: 搜索结果列表
            :param search_start_time: 搜索开始时间
            :param message: 错误信息（可选）
            :return: 包含搜索结果的字典
        """
       #logger.info(f"\nSearch engine: {self.search_engine}, results: {results}")
        search_end_time = time.time()
        search_duration = search_end_time - search_start_time
        return {
            "status": status,
            "query": query,
            "results": results,
            "references": self._generate_references(results),
            "search_engine": search_engine or self.search_engine,
            "search_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(search_start_time)),
            "search_duration": search_duration,
            "message": message
        }

    def _generate_references(self, results: list) -> list:
        references = []
        for result in results:
            if isinstance(result, dict):
                reference = {
                        "title": result.get("title", ""),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", "")
                    }
                references.append(reference)
            else:
                logger.warning(f"Invalid search result format: {result}")    
        #logger.info(f'Generated references: {references}')
        return references
    
    def format_results(self, results: list,limit_numbers=50) -> str:
        """
        格式化搜索结果为字符串"""
        formatted = []
        for i, result in enumerate(results, 1):
            if isinstance(result, dict):
                # 添加字段验证
                title = result.get("title", "")[:limit_numbers] if len(result.get("title", "")) > limit_numbers else result.get("title", "")
                body = result.get("body", "")[:limit_numbers] + "..." if len(result.get("body", "")) > limit_numbers else result.get("body", "")
                href = result.get("href", "")
                formatted.append(f"{i}. {title}\n{body}\n链接: {href}\n")
            else:
                logger.warning(f"Invalid search result format: {result}")
        return "\n".join(formatted)
TOOLS='联网搜索'


class WebEnhancedLLM(LLMBase):
    def __init__(self,  language='zh', api_base_url='',api_key='',search_engine=['baidu','bocha','duckduckgo','tavily'],depth=3,width=5,search_max_results=5,engine_config=None,**kwargs):
        super().__init__(**kwargs)
        default_configs = {
            'baidu': SearchEngineConfig('baidu', max_results=5),
            'bocha': SearchEngineConfig('bocha', max_results=5, api_key="your_bocha_key"),
            'duckduckgo': SearchEngineConfig('duckduckgo', max_results=5),
            'tavily': SearchEngineConfig('tavily', max_results=5, api_key="default_tavily_key")
        }
        # 合并用户配置
        
        self.engine_configs = {**default_configs, **(engine_config or {})}
         # 确保search_engine只包含启用的引擎
        self.search_engine = [engine for engine in search_engine 
                           if self.engine_configs.get(engine, {}).enabled]
       
        #print(f"search_engine: {self.search_engine}")
        self.search_tool =WebSearchTool(search_engine=self.search_engine[0],
        engine_config=self.engine_configs)
        self.language = "中文" if language in ['zh', 'zh-CN', 'zh-TW', '中文','chinese'] else '英文'
        self.supported_engines = self.search_engine if self.search_engine else self.search_tool.get_supported_engines()
        self.depth=depth
        self.width=width
        
         # 添加LOGO配置
        self.ENGINE_LOGOS = {
            engine: f'assets/{engine}.png'
            for engine in ['baidu', 'duckduckgo', 'bocha', 'tavily']
        }
        
       
        
         # 初始化客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base_url,
            max_retries=0,
            timeout=120
        )
        
        self.client_stream = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base_url,
            max_retries=0,
            timeout=60
        )
         # 初始化PromptManager
        self.prompt_manager = PromptManager(model_name=self.model)
        self.intent_prompt = """作为专业的网络检索助手，首先对用户输入进行分析并考虑是否启动联网搜索进行回答：
### 输入分析

1. 当前时间：{current_time}  
2. 历史对话（最近3轮）：{context}  
3. 当前问题："{query}" 

### 决策规则
• 结合历史对话和当前问题和时间分析输入后，参考是否存在禁止联网情况：如果命中，直接禁止联网；否则，默认开启联网搜索  
### 禁止联网搜索启用的情况
<禁止联网>
• **闲聊/问候语**  
  - 如"你好"、"在吗"、"谢谢"、"你会啥"、"你是谁"  
• **主观/观点类问题**  
  - 如"你觉得...？"、"你喜欢...吗？"
• **重复问题（历史已回答）**  
  - 检查最近3轮对话是否已回答  
• **简单计算/单位换算**  
  - 如"3+5等于多少？"、"1英里是多少公里？"  
• **常识性问题（无需最新数据）**  
  - 如"水的沸点是多少？"、"中国首都是哪里？"  
• **命令/控制类请求**  
  - 如"清空聊天记录"、"切换到英文模式"  
• **能力/功能询问**  
  - 如"你能做什么？"、"你有什么功能？"
</禁止联网>  

### 任务生成规范
1. 必须生成{generate_num}个多样化搜索的查询任务，丰富检索信息
2. 每个任务必须:
    * 保留原始问题核心实体
    * 明确限定时间/空间/类型范围
    * 时间递进(如过去/现在/未来)
    * 或空间递进(如全球/国家/地区)
    * 或概念细化(如整体/部分)
### 搜索引擎选择指南
为了匹配检索任务，请根据以下指南选择最合适的搜索引擎：
{engine_guide}
##  引擎选择
{search_engine}
### 输出格式（必须严格遵循）
[动作]
使用联网搜索工具|不使用联网搜索工具
[选择引擎]（仅联网时生成）
[引擎1,引擎2]  # 必须使用此格式
[搜索任务]（仅联网时生成）  
• 任务1: "标准化查询任务1" #[类型:范围]
• 任务2: "标准化查询任务2" #[类型:范围]
• 任务3: "标准化查询任务3" #[类型:范围]

请用{language}回答，注意：不要添加任何额外解释，严格按照格式输出
"""
#         self.intent_prompt = """作为专业的网络检索助手，首先对用户请求进行理解分析并考虑是否启动联网搜索：
# ### 输入分析

# 1. 现在时间：
# {current_time}
# 2. 历史对话:
# {context}
# 3. 当前问题:
# "{query}"

# ### 任务处理流程
# 1. 工具调用规则:
#    - 当确定调用联网搜索时候，结合历史对话和当前问题和时间分析输入后，把用户需求改成高质量查询。
#    - 禁止使用联网搜索的情况：
#     * 问候语(如"你好"、"早上好"、"再见")
#     * 闲聊内容(如"你叫什么"、"你会什么")
#     * 指令不明确(如"帮个忙"、"查一下")

# 2. 搜索任务生成要求:
#    - 结合上下文分析问题，拆解生成{generate_num}个任务
#    - 每个任务必须:
#      * 保留原始问题核心实体
#      * 明确限定时间/空间/类型范围
#    - 每个任务之间必须存在:
#      * 时间递进(如过去/现在/未来)
#      * 或空间递进(如全球/国家/地区)
#      * 或概念细化(如整体/部分)
#    - 选择工具执行完成任务

# ### 输出格式规范
# <!--
# tools_type: "search" | "no select search "
# sub_tasks: ["改写的查询任务1", "查询任务2", "查询任务3"]
# -->
# 请用{language}回答，严格按照规范输出，不要额外的解释。
# """
#         self.intent_prompt =  """作为一个专业的人工智能体，擅长利用各种工具解决问题，当前时间:{current_time}，用户当前问题："{query}"，历史上下文："{context}"，可用工具：{tools}，为了更好回答用户问题：
# ### 工具选择规则：
#     1. 基于用户的查询，使用"联网搜索"获取信息
#     2. 当问题要求生成图像(如"画"、"生成"、"图片")时，使用"图片生成"
#     3. 当问题属于常规对话如问候语、自我介绍、个性回答时，则不需要使用工具
# ### 思考过程：       
#     1. 结合全文理解用户的核心需求是什么，需要哪些具体信息
#     2. 将用户当前问题拓展成{generate_num}个问题，每个问题必须保留原问题主实体和完整语义
#     3. 简述表达使用哪种工具来解决当前任务
#         示例1：接下来,我将使用联网搜索工具，查找用户需要查询的信息
#         示例2：接下来我将使用图片生成工具，生成用户需要的图像
# ### 输出格式：
#     <!--
#     tools_type: "search"|"draw"|"chat" 
#     sub_tasks: [object-query, object-sub_task1, object-sub_task2, object-sub_task3] 
#     --> 
#     按此格式输出，无需其他自然语言回答，必须完全遵守输出格式，否则会造成严重的系统错误！用{language}回答。
#     回复示例：
#         输入："特斯拉最新财报"
#         输出：<!--
#         tools_type: "search"
#         sub_tasks: ["特斯拉最新财报","特斯拉2024Q3财报摘要", "特斯拉最新股价", "特斯拉CEO财报说明会"] -->
#         """
        self.rewrite_prompt= """作为查询优化助手，你需要根据以下对话历史和当前问题，生成{generate_num}个语义完整的子问题。
        
### 对话历史:
{history}

### 当前问题:
{query}

### 生成要求:
1. 每个子问题必须包含原始问题中的关键实体
2. 子问题之间应有逻辑递进关系
3. 保持专业语气，避免口语化表达
4. 确保每个子问题都能独立回答

### 输出格式:
<!--
sub_tasks: ["问题1", "问题2", "问题3"]
-->
"""
        self.summary_prompt = """As a professional web research analyst, you are summarizing search results for the following sub-tasks: {sub_tasks}. Current date: {current_time}.

### Search Results:
{search_results}

### Summary Requirements:
1. **Relevance Filtering**:
   - Rank information by relevance to the subtasks
   - Filter out low-quality sources (blogs without citations, outdated data, etc.)

2. **Key Information Extraction**:
   - Extract verifiable facts, statistics, and quantitative data
   - Highlight conflicting information across sources

3. **Consistency Analysis**:
   - Identify agreement/disagreement between authoritative sources
   - Note potential biases in controversial topics

4. **Structured Output Format**:
---
1. Key Findings: [2-3 bullet points of most important conclusions]
2. Supporting Data: 
   - [Metric 1]: [Value] ([Source])
   - [Metric 2]: [Value] ([Source])
3. Source Consistency: 
   - Agreed Points: [list]
   - Disputed Points: [list]
---

Note: Use neutral language and maintain academic tone. All claims must be source-attributed.Please use {language} to answer the following question:"""
    
#         self.summary_prompt="""
# 请根据用户提问{sub_tasks}和之前对话得到的搜索参考结果，进行回答。具体要求如下：

# 1. 内容要求：
# - 优先参考信息相关度高、来源可信的内容
# - 遇到矛盾信息时，需分析分歧点并给出客观解释
# - 关键信息需标明具体来源
# - 不要展示参考资料内容

# 2. 格式要求：
# - 使用{language}语言回复
# - 保持回答简洁、逻辑清晰

# 3. 其他说明：
# - 当前时间：{current_time}
# - 可根据问题性质调整表述方式

# 参考资料：
# {search_results}
# """
    def _generate_engine_guide(self):
        """获取当前可用搜索引擎的选择指南"""
        engine_descriptions = {
            'baidu': '适合中文内容、国内信息查询',
            'bocha': '适合学术论文、技术文档、生活娱乐等搜索',
            'duckduckgo': '国外的搜索引擎，适合英文内容、国际信息查询',
            'tavily': '适合实时新闻、最新资讯查询',
#'dazhong':"餐饮美食、旅游、购物等生活服务查询", 
        }
        
        des="\n".join(
            f"• **{engine.capitalize()}**: {engine_descriptions[engine]}"
            for engine in self.search_engine
            if engine in engine_descriptions
        )
        #print('des :',des)
        return des
    def get_engine_logo(self, engine_name: str) -> str:
            """获取搜索引擎的Logo路径"""
            return self.ENGINE_LOGOS.get(engine_name.lower(), "")
    def _get_tool_description(self, tool):
        """多语言工具描述"""
        
        descriptions = {
            '联网搜索': {'中文': '获取实时信息', 'en': 'Fetch real-time info','type': 'search'},
            '图片生成': {'中文': '生成视觉内容', 'en': 'Create visual content','type': 'draw'}
        }
        return descriptions.get(tool, {}).get(self.language, '')
    def _build_llm_context(self, results: List[Dict[str, Any]]) -> str:
        """构建LLM上下文"""
        context_parts = []
        for result in results:
            if "content_chunks" in result:
                for chunk in result["content_chunks"]:
                    context_parts.append(
                        f"来源: {result.get('title', '')}\n"
                        f"URL: {result.get('href', '')}\n"
                        f"内容: {chunk}\n"
                    )
        return "\n".join(context_parts)[:8000]  # 限制总长度
    def _clean_content(self, html: str) -> str:
        """内容清洗"""
        import re
        soup = BeautifulSoup(html, 'html.parser')
        
        # 移除不需要的元素
        for element in soup(['script', 'style', 'iframe', 'nav', 'footer']):
            element.decompose()
            
        # 提取正文（可根据网站结构调整）
        main_content = soup.find('article') or soup.find('div', class_=re.compile('content|main'))
        text = main_content.get_text() if main_content else soup.get_text()
        
        # 清理空白和特殊字符
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:10000]  # 限制长度

    def _chunk_text(self, text: str) -> List[str]:
        """文本分块"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        return splitter.split_text(text)
    async def _fetch_with_strategy(self, url: str, strategy: str = "default") -> Optional[str]:
        """多策略爬取"""
        import  aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                
                # 策略1：普通请求
                if strategy == "default":
                    async with session.get(url, headers=headers, timeout=10) as resp:
                        return await resp.text()
                        
                # 策略2：模拟浏览器
                elif strategy == "browser":
                    headers.update({
                        "Accept": "text/html,application/xhtml+xml",
                        "Accept-Language": "zh-CN,zh;q=0.9"
                    })
                    async with session.get(url, headers=headers, timeout=15) as resp:
                        return await resp.text()
                        
                # 策略3：API请求
                elif strategy == "api":
                    if "news" in url:
                        api_url = f"https://api.example.com/extract?url={url}"
                        async with session.get(api_url, timeout=8) as resp:
                            data = await resp.json()
                            return data.get("content", "")
        except Exception as e:
            logger.warning(f"Strategy {strategy} failed for {url}: {e}")
            return None

    async def enrich_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """异步增强搜索结果，获取页面内容"""
            
        if not results:
            return results
            
        # 尝试不同爬取策略
        strategies = ["default", "browser", "api"]
        for result in results:
            content = None
            for strategy in strategies:
                if content := await self._fetch_with_strategy(result["href"], strategy):
                    break
                    
            if content:
                cleaned = self._clean_content(content)
                chunks = self._chunk_text(cleaned)
                result.update({
                    "full_content": cleaned,
                    "content_chunks": chunks,
                    "strategy_used": strategy
                })
                
        return results
    
    def contains_web_search_instruction(self, text: str) -> tuple[bool, list]:
        """检查文本中是否包含联网搜索指令
        
        Args:
            text: 要检查的文本内容
            
        Returns:
            tuple: (是否使用搜索工具, 查询任务列表)
        """
        import re
        # 匹配是否使用搜索工具
        positive_pattern = r'\[动作\]\s*使用联网搜索工具'
        negative_patterns = [
            r'\[动作\]\s*不使用联网搜索工具',
            r'不使用联网搜索',
            r'不要用联网搜索',
            r'禁止联网搜索',
            r'无需联网搜索',
            r'不必联网搜索',
            r'不用联网搜索',
            r'禁止使用联网搜索',
            r'无需使用联网搜索'
        ]
    
    # 检查是否有任何否定表达
        # 检查是否有明确的"使用"标记
        use_tool = bool(re.search(positive_pattern, text))
        # 如果有任何否定表达，则强制返回False
        if any(re.search(pattern, text) for pattern in negative_patterns):
            use_tool = False
        else:
            use_tool = True
        engine_pattern = r'\[选择引擎\]\s*(?:\[([^\]]+)\]|([^\n]+))'
        engine_match = re.search(engine_pattern, text)
        engines = []
        if engine_match:
        # 优先尝试第一种格式
            engines_str = engine_match.group(1) or engine_match.group(2)
            try:
                engines = json.loads(engines_str.replace("'", '"')) if engine_match.group(1) else [e.strip() for e in engines_str.split(',')]
            except json.JSONDecodeError:
                # 处理非JSON格式的引擎列表
                engines = [e.strip().strip("'\"") for e in engines_str.split(',')]
        # 提取查询任务
        tasks = self.extract_query_tasks(text)  # 现在作为实例方法调用
       
        return use_tool, tasks, engines

    def extract_query_tasks(self, text: str) -> list:
        """从文本中提取查询任务"""
        import re
        #pattern = r'(?:\d+\.\s*|-\s*)查询任务\d+:\s*"?([^"\n]+)"?'
        pattern =r'•\s*(?:任务\d+:\s*)?([^#\n]+?)\s*#\[[^\]]+\]'
        return re.findall(pattern, text)
    def get_prompt(self, name: str, variables: Dict[str, Any]) -> str:
        """统一获取提示词的方法"""
        return self.prompt_manager.get_prompt(name, variables)
    def get_intent_prompt(self, query, context, tools,width_num):
        """获取意图识别提示"""
        tools = "、".join([f"{t}({self._get_tool_description(t)})" for t in tools])
        logger.info(f"tools: {tools}")
          # 动态生成引擎选择部分
        engine_guide = self._generate_engine_guide()
        return self.intent_prompt.format(
            current_time=datetime.now(timezone.utc).strftime('%Y年%m月%d日 %H:%M'),
            query=query,
            context=context,
            tools=tools,generate_num=width_num,
            engine_guide=engine_guide , # 新增参数
            search_engine=self.search_engine,
            language=self.language
        )
    def get_rewrite_prompt(self, history, query,width_num):
        """rewrite提示"""
       
        return self.rewrite_prompt.format(
            history=history,query=query,generate_num=width_num ,language=self.language
        )
    def rewire_query(self,history,query,width):
        self.rewrite_prompt=self.get_rewrite_prompt( history,query,width)
        #logger.info(f"rewire_query: {self.rewrite_prompt}")
        sub_tasks,_= self.call_llm(self.rewrite_prompt)
        sub_tasks,action_name=self.extract_keywords_response(sub_tasks)
        logger.info(f"rewire_query--sub_tasks: {sub_tasks}")
        return sub_tasks,action_name
    
        return sub_tasks
    def get_summary_prompt(self, sub_tasks, search_results):
        """获取总结提示"""
        return self.summary_prompt.format(
            sub_tasks=sub_tasks,
            current_time=datetime.now(timezone.utc).strftime('%Y年%m月%d日 %H:%M'),
            search_results=search_results,
            language=self.language
        )
    async def _evaluate_search_quality_and_sufficiency(self, subtask: str, search_results: list, eval_report: dict) -> dict:
        """评估搜索结果质量并判断是否足够回答子任务"""
        logger.info(f"subtask: {subtask} && search_results: {search_results}")
        prompt = f"""作为一个信息分析专家，你将汇总搜索到的内容，评估现有信息能否满足任务{subtask}需求，检索报告记录{eval_report if eval_report else'无'}：当前时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}，严格按以下JSON格式输出，
            1. 以左花括号 `{{` 开头，右花括号 `}}` 结尾
            2. 所有字符串值使用双引号（非单引号）
            3. 确保特殊字符（如换行符、引号）已转义
            不要包含任何额外解释或文本，保证输出是可解析的JSON对象：

{{
    "search_results_count": {len(search_results)},
    "evaluation": {{
        
        "quality": {{
            "score": <0-1的小数>,
            "reason": "<不超过20字的简短理由>"
        }},
        "is_sufficient": <true/false>,
        "suggested_query": "<如果不满足则提供优化后的搜索词，否则留空>"
    }},
    "overall_assessment": "<50字内的总结建议>"
}}

        质量评估要求：
        1. 内容相关性：根据内容与子任务匹配度评分
        2. 生成质量：根据信息权威性、时效性评分
        3. 内容完整度：判断现有信息是否能直接解决子任务
        4. 若评分<0.5,则is_sufficient=false，必须提供suggested_query

        搜索内容摘要：
        {search_results}
"""
        
        evaluation,reason_content= self.call_llm(prompt)
        logger.info(f"evaluation: {evaluation}")
        # 先尝试直接解析（兼容纯JSON）
        try:
            return json.loads(evaluation)
        except json.JSONDecodeError:
            pass
        
        # 若直接解析失败，尝试清理常见包裹模式
        patterns_to_remove = [
            r'```json(.*?)```',  # Markdown代码块
            r'```(.*?)```',      # 无语言标记的代码块
            r"'''json(.*?)'''",  # Python多行字符串
            r'\"\"\"json(.*?)\"\"\"'  # Python多行字符串(双引号)
        ]
        
        for pattern in patterns_to_remove:
            match = re.search(pattern, evaluation, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue
         # 最终尝试去除所有非JSON符号（激进清理）
        cleaned = re.sub(r'^[\s\'\"]+|[\s\'\"]+$', '', evaluation)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    async def _deep_search_with_reflection(self,subtasks: list) -> list:
        """带反思的深度搜索"""
        total_searches = 0
        total_evaluations = 0
        query_revisions = 0
        #chunks_score=[]
        
        async def process_subtask(subtask):
            current_query = subtask
            nonlocal total_searches, total_evaluations, query_revisions
            best_results = []
            best_refs = []
            subtask_searches = 0
            subtask_evaluations = 0
            evaluation_history =None #
        
            for iteration in range(0, self.depth):
                logger.info(f"深度搜索迭代 {iteration}/{self.depth}，查询任务: {current_query}")
                subtask_searches += 1
                total_searches += 1
                # 执行搜索并评估
                search_results, ref = await self.search_with_context([current_query], False, False)
                if self.depth>1:
                    try:
                        logger.info(f"评估搜索结果：")
                        evaluation = await self._evaluate_search_quality_and_sufficiency([current_query], search_results,evaluation_history)
                       # chunks_score.append(evaluation['evaluation']['quality']['score'])
                        subtask_evaluations += 1
                        total_evaluations += 1
                        
                        logger.info(f"质量分数: {evaluation['evaluation']['quality']['score']}\n")
                        
                        if evaluation['evaluation']["is_sufficient"]:
                            logger.info(f"子任务 '{current_query}' 搜索结果已足够")
                            break
                        else:
                            logger.info(f"子任务 '{current_query}' 搜索结果不足，继续搜索")
                        # 更新查询词继续搜索
                            old_query=current_query
                            current_query = evaluation['evaluation']["suggested_query"]
                            query_revisions += 1
                            logger.info(f"查询优化 - 从 '{old_query}' 修改为 '{current_query}'")
                            logger.info(f"搜索结果不足，生成新查询: {current_query}")
                            
                            # 记录完整评估上下文
                            evaluation_history = {
                                "query_revisions": query_revisions,
                                "query": current_query,
                                "evaluation": evaluation['overall_assessment'],
                            }
                            
                    except Exception as e:
                        logger.error(f"评估子任务 '{current_query}' 时发生错误: {str(e)}")
                        logger.warning("跳过评估，保留当前结果")
            best_results = search_results
            best_refs = ref 

            return best_results, best_refs

        tasks = [process_subtask(subtask) for subtask in subtasks]
        results = await asyncio.gather(*tasks)
        
        # 合并所有结果
        all_results = []
        all_refs = []
        for result, ref in results:
            if result:
                all_results.extend(result)
                all_refs.extend(ref)
        
        return all_results, all_refs
    async def search_with_context(self, search_keywords,enable_reflection=True,format=True,max_final_results=10,enabled_engines=[])-> str: 
        """
        Perform search with context using multiple keywords
        """
        search_results=[]
        if not enabled_engines:
           # enabled_engines = engines
            enabled_engines = [e for e in self.supported_engines 
                            if self.search_tool.SUPPORTED_ENGINES[e].enabled]
        
        logger.info(f"Searching with multiple engines: {self.supported_engines},deep_search: {enable_reflection}")
        if not enabled_engines:
            logger.error("No enabled search engines available")
            return [], []
        logger.info(f"Searching with multiple engines: {enabled_engines},deep_search: {enable_reflection}")
        if enable_reflection:   
                logger.info(f"深度聚合思考模式——Searching with reflection: {enable_reflection}")
                search_results,ref = await self._deep_search_with_reflection(search_keywords)
                
                if len(search_results)>max_final_results:
                     search_results=search_results[:max_final_results]
                     ref=ref[:max_final_results]
                return search_results,ref
        else:
                logger.info(f"多引擎聚合搜索模式——Searching with multiple engines: {enabled_engines}")
                search_results= await self.search_tool.aggregate_search_with_keywords(search_keywords,enabled_engines)
    
        raw_results = search_results.get("results", []) if search_results["status"] == "success" else []
        # 在外层统一进行URL检查
        valid_results = []
        for item in raw_results:
            if isinstance(item, dict) and item.get("href"):
                try:
                    response = requests.head(item["href"], timeout=5, allow_redirects=True)
                    if response.status_code < 400:  # 只保留有效的URL
                        valid_results.append(item)
                except:
                    continue
        formatted_results = self.search_tool.format_results(valid_results) if format else valid_results
        ref = self.search_tool._generate_references(valid_results)  # 使用过滤后的结果生成引用
        return formatted_results, ref #返回搜索结果，参考内容
    
    def extract_keywords_response(self, model_response: str):
        """
        双模式关键词提取函数
        :param model_response: 大模型返回的原始文本
        :return: 提取后的关键词列表（可能为空）和工具名称
        """
        # 模式1：从自然语言描述中提取
        try:
            import re
            # 匹配HTML注释
            comment_pattern = r'<!--\s*(.*?)\s*-->'
            comment_match = re.search(comment_pattern, model_response, re.DOTALL)
            if comment_match:
                comment_content = comment_match.group(1)

                # 提取action
                action_pattern = r'tools_type:\s*"([^"]+)"'
                action_match = re.search(action_pattern, comment_content)
                tools_name = action_match.group(1) if action_match else ""

                # 提取keywords
                keywords_pattern = r'sub_tasks:\s*\[([^\]]+)\]'
                keywords_match = re.search(keywords_pattern, comment_content)
                if keywords_match:
                    keywords_str = keywords_match.group(1)
                    keywords = [k.strip().strip('"') for k in keywords_str.split(',') if k.strip()]
                    return keywords, tools_name
            else:
                # 提取action
                tool_match = re.search(r'tools_type:\s*"([^"]+)"', model_response)
                tools_name = tool_match.group(1) if tool_match else "search"

                # 提取关键词列表
                keywords_match = re.search(r'子任务|sub_tasks:\s*\[([^\]]+)\]', model_response)
                if keywords_match:
                    keywords_str = keywords_match.group(1)
                    keywords = [k.strip().strip('"\'') for k in keywords_str.split(',') if k.strip()]
                    return keywords, tools_name

                # 提取纯文本关键词
                text_match = re.search(r'(?:子任务|sub_tasks):\s*([^\n]+)', model_response)
                if text_match:
                    keywords_str = text_match.group(1)
                    keywords = [k.strip().strip('"\'') for k in keywords_str.split(',') if k.strip()]
                    return keywords, tools_name
        except Exception as e:
            logger.error(f"HTML注释解析失败: {e}")
            pass
        try:
            import json
            json_match = re.search(r'```json(.*?)```', model_response, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
                result = json.loads(json_content)
                return result.get("sub_tasks", []), result.get("tools_type", "")
        except json.JSONDecodeError:
            pass
        # 模式3：正则表达式-兜底
        try:
            # 提取tools_type
            tools_type_match = re.search(r'tools_type:\s*"([^"]+)"',  model_response)
            tools_name = tools_type_match.group(1) if tools_type_match else ""
            # 提取sub_tasks
            sub_tasks_match = re.search(r'sub_tasks:\s*\[([^\]]+)\]',  model_response)
            if sub_tasks_match:
                    sub_tasks_str = sub_tasks_match.group(1)
                    keywords = [task.strip().strip('"\'') for task in sub_tasks_str.split(',')]
                    return keywords, tools_name
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")

        return [], ''
 
    def format_context(self, messages, keep_last_k=6):
        """格式化对话历史，保留最近的K个用户提问以保持连续性
        :param messages: 完整的对话记录
        :param keep_last_k: 保留最近K个问题（默认6）
        """
        context = ""
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        # 取最近的K个query
        recent_messages = user_messages[-keep_last_k:] if len(user_messages) > keep_last_k else user_messages
        if recent_messages:
            # 保持原始顺序，仅调整编号
            for i, msg in enumerate(recent_messages, start=1):
                context += f"最近问题 {i}: {msg['content']}\n"
        return context.strip()

    async def web_search(self, query: str,context:str,sub_nums:int=3,tools={"联网搜索"})-> AsyncGenerator[str, None]:
        # 根据语言选择提示词模板
        self.intent_prompt=self.get_intent_prompt(query, context, tools,sub_nums)    
        
        mes = [{"role": "user", "content": self.intent_prompt}]
        
        async for chunk in self.call_llm_stream(messages=mes, temperature=0):
            yield chunk
        
    def call_llm(self, prompt: str) -> tuple[str, str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],temperature=0.6,
                stream=False
            )
            reasoning_content = response.choices[0].message.reasoning_content if hasattr(response.choices[0].message, 'reasoning_content') else ""
            return response.choices[0].message.content, reasoning_content
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            return None, None
        
    async def call_llm_stream(self,
        messages: List[Dict[str, str]],
        temperature: float=0.6,
        max_length: int=4096,
        id: str='user',
        stop: Optional[List[str]] = None,
    ): 
        """
        异步stream out function for calling the LLM

        :param messages: 聊天消息列表，每个消息是一个字典，包含 "role" 和 "content"。
        :param temperature: 控制生成文本的随机性。
        :param max_length: 生成文本的最大长度。
        :param id: 用于标识生成结果的 ID。
        :param stop: 自定义的停止词列表。
        :return: 异步生成器，每次迭代返回一个 JSON 格式的字符串。
        """
        
        async for chunk in await self.client_stream.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
            max_tokens=max_length,
            #stop=stop if stop else None,
        ):
            yield chunk

