import streamlit as st
from src import WebEnhancedLLM, SearchEngineConfig
from dotenv import load_dotenv
import os
import asyncio
from typing import AsyncGenerator, Dict, List

# 初始化配置
load_dotenv(override=True)
st.set_page_config(page_title="Web search with Any LLM ", layout="wide")
# # 临时测试用绝对路径
# 在process_query方法开头临时测试

#logo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets', f'baidu.png'))
# print(f"Absolute logo path: {logo_path}")  # 检查路径是否正确
class ChatApp:
    def __init__(self):
        #assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
    #     self.ENGINE_LOGOS = {
    #     engine: f'assets/{engine}.png'
    #     for engine in ['baidu', 'duckduckgo', 'bocha', 'tavily']
    # }
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
        think_mode = False  # 确保每次新查询都重置
        reasoning_emitted = False  # 确保每次新查询都重置
        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": query})
        
        # 准备对话上下文
        messages = [{"role": "system", "content": "你是一个有帮助的AI助手"}]
        messages.extend([{"role": msg["role"], "content": msg["content"]} 
                        for msg in st.session_state.messages[:-1]])
        
#         test_img = """
# <div style="padding: 10px;">
#     <img src="https://www.baidu.com/img/flexible/logo/pc/result.png" width="100">
#     <p>测试图片显示</p>
# </div>
# """
#         yield test_img  # 如果这个能显示，说明是路径问题
        # 流式处理响应
        
        async for chunk in self.llm.web_search(query, context=messages, tools={'联网搜索'}):
            reasoning_content=''
            # 检查是否有 stop_reason 字段
            if not hasattr(chunk, 'choices') or not chunk.choices:
                continue
         
            if hasattr(chunk.choices[0].delta, 'reasoning_content')and chunk.choices[0].delta.reasoning_content:
                        reasoning_content += chunk.choices[0].delta.reasoning_content
                        if not reasoning_emitted :
                            start_label =  """<div style='color: #666; background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin: 10px 0;'>
<b>思考与行动过程</b>
</div>"""
                            yield f"{start_label}"
                            reasoning_emitted= True
                        md_content = f"{chunk.choices[0].delta.reasoning_content}"
                        yield md_content
            if chunk.choices[0].delta.content:
                        if chunk.choices[0].delta.content=="<think>":
                                think_mode=True
                                start_label =  """<div style='color: #666; background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin: 10px 0;'>
<b>思考与行动过程</b>
</div>"""                       
                                yield f"{start_label}"
                        elif chunk.choices[0].delta.content == "</think>":
                            
                                think_mode = False
                        elif think_mode:  
                                yield f"{chunk.choices[0].delta.content}"
                        else:      
                             intent_content+= chunk.choices[0].delta.content
          
        action_name,keywords,engines=self.llm.contains_web_search_instruction(intent_content)
        print(f"action_name: {action_name}",f"keywords: {keywords}")
        web_search_mes=messages.copy()
        if keywords and action_name:
            # 2.搜索
            engine_logos_html = ""
            #print(  f"searching for {keywords} using {engines}")
            if not isinstance(engines, list):
                engines = [engines]
            for engine in engines:
                logo_path = self.llm.get_engine_logo(engine)
                with open(logo_path, "rb") as f:
                    logo_bytes = f.read()
                import base64
                logo_base64 = base64.b64encode(logo_bytes).decode()
                engine_logos_html += f"<img src='data:image/png;base64,{logo_base64}' width='40' style='margin-right: 8px;'>"
             
                
            use_label=f"""<div style='color: #666; background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin: 10px 0;'>
<b>使用工具:联网搜索</b>{engine_logos_html}
</div>"""
            yield use_label
             # 单独渲染logo部分
    #         if engine_logos_html:
    #             yield f"""<div style='margin: 10px 0;'>
    # {engine_logos_html}
    # </div>"""
            search_results,ref= await self.llm.search_with_context(keywords,enable_reflection=True,enabled_engines=engines)
           # yield f'{engine_logos_html if engine_logos_html else ""}'
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
                                        label="""<div style='color: #666; background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin: 10px 0;'>
    <b>思考结束：准备输出结果</b>
    </div>"""
                                        yield f"{label}"
                                
                                    elif chunk.choices[0].delta.content=="<think>":

                                        think_mode=True
                                        labelx="""<div style='color: #666; background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin: 10px 0;'>
    <b>根据收集到的信息，进行如下分析和整理</b>
    </div>"""
                                        yield f"{labelx}\n"
                                
                                    elif chunk.choices[0].delta.content == "</think>":
                                      
                                        label="""<div style='color: #666; background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin: 10px 0;'>
    <b>思考结束：准备输出结果</b>
    </div>"""
                                        yield f"{label}"
                                        think_mode = False
                                    elif  not think_mode  or not  reasoning_emitted:   
                                        yield {"type": "full_results", "data": chunk.choices[0].delta.content}
        else:
                think_mode=False
                reasoning_emitted=False
                messages.append({"role": "user", "content": query})
             
                async for chunk in self.llm.call_llm_stream(messages=messages): 
                        
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
                                        label="""<div style='color: #666; background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin: 10px 0;'>
    <b>思考结束：准备输出结果</b>
    </div>"""
                                        yield f"{label}"
                           
                                    elif chunk.choices[0].delta.content == "<think>":
                                        yield "\n"
                                    elif chunk.choices[0].delta.content == "</think>":
                                        label="""<div style='color: #666; background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin: 10px 0;'>
    <b>思考结束：准备输出结果</b>
    </div>"""
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
                            
                            save_content += chunk["data"]  # 保存完整结果
                            full_response += chunk["data"] 
                       
                    else:
                        
                        full_response += chunk
                    response_placeholder.markdown(full_response,unsafe_allow_html=True)
                #print("save_content:", save_content)
                # 添加完整响应到历史
                st.session_state.messages.append({"role": "assistant", "content":  save_content})
            
            asyncio.run(stream_response())

if __name__ == "__main__":
    app = ChatApp()
    app.run()