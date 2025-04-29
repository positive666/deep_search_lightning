from src import WebEnhancedLLM, SearchEngineConfig
from dotenv import load_dotenv
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    async def main():
        """主函数示例
            所需依赖：
            # 基础库
            # pip install asyncio python-dateutil typing loguru
            # 搜索相关
            # pip install duckduckgo-search requests
            # OpenAI SDK
            # pip install openai
            # 其他工具
            
            """
        load_dotenv() # load        
        engine_config = {
    'baidu': SearchEngineConfig('baidu', max_results=int(os.getenv('BAIDU_MAX_RESULTS')), api_key=os.getenv('BAIDU_API_KEY'), enabled=os.getenv('BAIDU_ENABLED')== 'true'),
    'duckduckgo': SearchEngineConfig('duckduckgo', max_results=int(os.getenv('DUCKDUCKGO_MAX_RESULTS')),enabled=os.getenv('DUCKDUCKGO_ENABLED')== 'true'),
    'bocha': SearchEngineConfig('bocha', max_results=int(os.getenv('BOCHA_MAX_RESULTS')), api_key=os.getenv('BOCHA_API_KEY'), enabled=os.getenv('BOCHA_ENABLED')== 'true'),
    'tavily': SearchEngineConfig('tavily', max_results=int(os.getenv('TAVILY_MAX_RESULTS')), api_key=os.getenv('TAVILY_API_KEY'), enabled=os.getenv('TAVILY_ENABLED') == 'true'),
}
        print(engine_config)
         # 只选择被启用的引擎
        search_engines = [engine for engine in engine_config 
                         if engine_config[engine].enabled]
        logger.info("使用的搜索引擎: %s", search_engines)
        web_llm = WebEnhancedLLM(model=os.getenv('MODEL_NAME'),
                                 search_engine=search_engines ,language='zh',
                                 api_key=os.getenv("MODEL_API_KEY") ,
                                 api_base_url=os.getenv('API_BASE_URL'),
                                depth=int(os.getenv('DEPTH')),engine_config=engine_config,width=int(os.getenv('WIDTH')),search_max_results=os.getenv('SEARCH_MAX_RESULTS'))
        tools={'联网搜索'}
        query='中国最好的大学是哪个？'
        buffer='' #历史记录缓存
        messages = [{"role": "system", "content": '你是一个人工智能助手'},{"role": "user", "content": '画个粉色猫咪'}, {"role": "assistant", "content": '好的，完成了'}, {"role": "user", "content": query}]
        context = web_llm.format_context(messages[:-1]) #获取历史提问
        intent_content=''
        #1.意图识别和生成搜索任务
        async for chunk in  web_llm.web_search(query,context,sub_nums=3,tools=tools ):
            reasoning_content=''
            # 检查是否有 stop_reason 字段
            if hasattr(chunk.choices[0], 'stop_reason'):                   
                    print("Stream finished without a specific stop reason.")
                    break
           
            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content is not None:
                reasoning_content += chunk.choices[0].delta.reasoning_content
                print(chunk.choices[0].delta.reasoning_content,end="",flush=True)     
            elif hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content is not None:
                            intent_content+= chunk.choices[0].delta.content
                            
        
        keywords,action_name=web_llm.extract_keywords_response(intent_content)
        print(keywords,action_name)
        
        if keywords and action_name =='search':
                     
                    # 2.搜索
                    search_results,ref= await web_llm.search_with_context(keywords,enable_reflection=True) #默认聚合搜索&深度反思
                    
                    # 3.生成搜索结果的总结提示        
                    summary_prompt = web_llm.get_summary_prompt(keywords,search_results)
                 
                    print('搜索结果:',search_results)
                    print('类目数量',len(search_results))
                    
                    web_search_mes=messages[-2:].copy() if len(messages) >2 else messages.copy()
                    web_search_mes[-1]={
                            "role": "user",
                            "content": summary_prompt
                        }
                    
                    async for chunk in web_llm.call_llm_stream(
                            messages=web_search_mes):
                            if hasattr(chunk.choices[0], 'stop_reason'):                   
                                print("\n Stream finished without a specific stop reason.")
                                break
                            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content is not None:
                                
                                print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
                            elif hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content is not None:
                              
                                buffer+= chunk.choices[0].delta.content #总缓存
                                print(chunk.choices[0].delta.content, end="", flush=True)
                                
       
        else: 
            print('general chat')
            
    import asyncio
    asyncio.run(main())
