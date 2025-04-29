import streamlit as st
from src import WebEnhancedLLM, SearchEngineConfig
from dotenv import load_dotenv
import os
import asyncio
from typing import AsyncGenerator, Dict, List

# 初始化配置
load_dotenv()
st.set_page_config(page_title="Web search with Any LLM ", layout="wide")

class ChatApp:
    def __init__(self):
        self.config = self._load_config()
        self.llm = self._init_llm()
        self.setup_ui()

    def _load_config(self):
        """加载或初始化配置"""
        def safe_int(value: str, default: int) -> int:
            try:
                return int(value.split('#')[0].strip())
            except (ValueError, AttributeError):
                return default

        if "config" not in st.session_state:
            st.session_state.config = {
                "model": os.getenv('MODEL_NAME', ''),
                "api_key": os.getenv("MODEL_API_KEY", ""),
                "api_base": os.getenv('API_BASE_URL', ''),
                "depth": safe_int(os.getenv('DEPTH'), 1),
                "width": safe_int(os.getenv('WIDTH'), 3),
                "max_final_results": safe_int(os.getenv('MAX_SEARCH_RESULTS'), 15),
                "engines": {
                    'baidu': {
                        'enabled': os.getenv('BAIDU_ENABLED', 'true').lower() == 'true',
                        'max_results': safe_int(os.getenv('BAIDU_MAX_RESULTS'), 5)
                    },
                    'duckduckgo': {
                        'enabled': os.getenv('DUCKDUCKGO_ENABLED', 'false').lower() == 'true',
                        'max_results': safe_int(os.getenv('DUCKDUCKGO_MAX_RESULTS'), 5)
                    },
                    'bocha': {
                        'enabled': os.getenv('BOCHA_ENABLED', 'false').lower() == 'true',
                        'max_results': safe_int(os.getenv('BOCHA_MAX_RESULTS'), 3),
                        'api_key': os.getenv('BOCHA_API_KEY', '')
                    },
                    'tavily': {
                        'enabled': os.getenv('TAVILY_ENABLED', 'false').lower() == 'true',
                        'max_results': safe_int(os.getenv('TAVILY_MAX_RESULTS'), 5),
                        'api_key': os.getenv('TAVILY_API_KEY', '')
                    }
                }
            }
        return st.session_state.config

    def _show_config_ui(self):
        """显示配置界面"""
        with st.sidebar.expander("⚙️ 引擎配置"):
            # 模型配置
            self.config['model'] = st.text_input("模型名称", value=self.config['model'])
            self.config['api_key'] = st.text_input("API Key", value=self.config['api_key'], type="password")
            self.config['api_base'] = st.text_input("API地址", value=self.config['api_base'])
            
            # 新增搜索参数配置
            st.subheader("搜索参数")
            self.config['depth'] = st.number_input("搜索深度", min_value=1, max_value=5, value=self.config['depth'])
            self.config['width'] = st.number_input("搜索宽度", min_value=1, max_value=5, value=self.config['width'])
            self.config['max_final_results'] = st.number_input("最大结果数", min_value=1, max_value=20, value=self.config['max_final_results'])
            
            # 引擎配置
            for engine, config in self.config['engines'].items():
                st.subheader(f"{engine}配置")
                config['enabled'] = st.checkbox(f"启用{engine}", value=config['enabled'])
                if config['enabled']:
                    config['max_results'] = st.number_input(
                        f"{engine}最大结果数", 
                        min_value=1, max_value=10, value=config['max_results']
                    )
                    if engine in ['tavily', 'bocha']:  # 需要API Key的引擎
                        config['api_key'] = st.text_input(
                            f"{engine} API Key", 
                            value=config.get('api_key', ''), 
                            type="password"
                        )
        
            if st.button("保存配置"):
                self.llm = self._init_llm()
                st.success("配置已更新！")

    def _init_llm(self) -> WebEnhancedLLM:
        """初始化LLM实例"""
        engine_config = {
            name: SearchEngineConfig(
                name,
                max_results=config['max_results'],
                api_key=config.get('api_key', ''),
                enabled=config['enabled']
            )
            for name, config in self.config['engines'].items()
        }
        
        return WebEnhancedLLM(
            model=self.config['model'],
            search_engine=[name for name, cfg in self.config['engines'].items() if cfg['enabled']],
            language='zh',
            api_key=self.config['api_key'],
            api_base_url=self.config['api_base'],
            engine_config=engine_config,
            depth=self.config['depth'],
            width=self.config['width']
        )

    def setup_ui(self):
        """设置用户界面"""
        st.title("🔍 Web search with Any LLM")
        self._show_config_ui()  # 添加配置界面
        st.caption("轻量的深度联网搜索")
        
        # 添加重新对话按钮
        if st.button("🔄 重新对话"):
            st.session_state.messages = [
                {"role": "assistant", "content": "你好！请问有什么可以帮您？"}
            ]
            st.rerun()
        
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "你好！请问有什么可以帮您？"}
            ]

        # 显示历史消息
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

    async def process_query(self, query: str) -> AsyncGenerator[str, None]:
        """处理用户查询并生成响应"""
        
        # 初始化变量
        import json
        intent_content = ""
        reasoning_content = ""
        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": query})
        reasoning_emitted = False
        # 准备对话上下文
        messages = [{"role": "system", "content": "你是一个有帮助的AI助手"}]
        messages.extend([{"role": msg["role"], "content": msg["content"]} 
                        for msg in st.session_state.messages[:-1]])
        print(messages)
        think_mode=False
        # 流式处理响应
        async for chunk in self.llm.web_search(query, context=messages, tools={'联网搜索'}):
            reasoning_content=''
            # 检查是否有 stop_reason 字段
            if not hasattr(chunk, 'choices') or not chunk.choices:
                continue
         
            if hasattr(chunk.choices[0].delta, 'reasoning_content')and chunk.choices[0].delta.reasoning_content:
                        reasoning_content += chunk.choices[0].delta.reasoning_content
                        if not reasoning_emitted :
                            start_label = "<--------思考与行动过程------>\n"
                            yield f"{start_label}"
                            reasoning_emitted= True
                        md_content = f"{chunk.choices[0].delta.reasoning_content}"
                        yield md_content
            if chunk.choices[0].delta.content:
                        if chunk.choices[0].delta.content=="<think>":
                                think_mode=True
                                start_label = "<-------思考与行动过程----->\n"
                                yield f"{start_label}"
                        elif chunk.choices[0].delta.content == "</think>":
                                think_mode = False
                        elif think_mode:  # 只有在think模式下的非标签内容才特殊处理
                                yield f"{chunk.choices[0].delta.content}"
                        intent_content+= chunk.choices[0].delta.content
        keywords,action_name=self.llm.extract_keywords_response(intent_content)
        
        web_search_mes=messages.copy()
        if keywords and action_name =='search':
            # 2.搜索
            use_label=f"\n<----使用工具: 联网搜索----->"
            yield f"{use_label}"
            search_results,ref= await self.llm.search_with_context(keywords,enable_reflection=True)
            
            # 3.生成搜索结果的总结提示        
            summary_prompt = self.llm.get_summary_prompt(keywords,search_results)
            
            # 简化搜索结果展示
            search_results_display = ""
            for i, result in enumerate(search_results, 1):
                search_results_display += f"""
{i}. **{result.get('title', '无标题')}**  
🔗 {result.get('href', '')}\n
"""
            yield search_results_display
            print('搜索结果:',search_results)
            web_search_mes=messages[-2:].copy() if len(messages) >2 else messages.copy()
            web_search_mes[-1]={
                    "role": "user",
                    "content": summary_prompt
                }
        think_mode=False
        reasoning_emitted=False
        async for chunk in self.llm.call_llm_stream(messages=web_search_mes): 
                        if not hasattr(chunk, 'choices') or not chunk.choices:
                                    continue
                        elif hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content :
                            reasoning_emitted=True
                            html_content = chunk.choices[0].delta.reasoning_content 
                            yield f"{json.dumps(html_content, ensure_ascii=False)}\n"
                            
                        elif hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content :
                                if reasoning_emitted:
                                    reasoning_emitted=False
                                    think_mode = False
                                    label="\n-----思考结束:准备输出结果-----"
                                    yield f"{label}"
                               
                            
                      
                                elif chunk.choices[0].delta.content=="<think>":
                                    think_mode=True
                                    labelx="\n<-----根据收集到的信息，进行如下分析和梳理----->"
                                    yield f"{labelx}\n"
                            
                                elif chunk.choices[0].delta.content == "</think>":
                                    label="\n<-----思考结束:准备输出结果----->"
                                    yield f"{label}"
                                    think_mode = False
                                elif  not think_mode  or not  reasoning_emitted:   
                                     yield {"type": "full_results", "data": chunk.choices[0].delta.content}
                     
                        

    def run(self):
        """运行主循环"""
        if query := st.chat_input("输入您的问题..."):
            # 显示用户消息
            st.chat_message("user").write(query)
            
            # 创建占位符用于流式输出
            response_placeholder = st.empty()
            full_response = ""
            save_content=""
            # 处理并显示响应
            async def stream_response():
                nonlocal full_response
                nonlocal save_content
                async for chunk in self.process_query(query):
                    #full_response += chunk
                    if isinstance(chunk, dict):
                        if chunk.get("type") == "full_results":
                            print(chunk["data"])
                            save_content += chunk["data"]  # 保存完整结果
                            full_response += chunk["data"] 
                       
                    else:
                        #print(chunk)
                        full_response += chunk
                    response_placeholder.markdown(full_response)
                print("save_content:", save_content)
                # 添加完整响应到历史
                st.session_state.messages.append({"role": "assistant", "content":  save_content})
            
            asyncio.run(stream_response())

if __name__ == "__main__":
    app = ChatApp()
    app.run()