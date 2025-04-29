from typing import AsyncGenerator, Dict, Any, Optional, Union, List
import openai
from openai import AsyncOpenAI, OpenAI
import logging
from datetime import datetime, timezone
logger = logging.getLogger(__name__)

class LLMBase:
    """LLM基础类，提供核心LLM功能和客户端管理"""
    
    def __init__(
        self,
        model: str = "",
        api_key: str = "",
        api_base: str = "",
        max_retries: int = 3,
        **kwargs
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.max_retries = max_retries
        self._init_clients()

    def _init_clients(self) -> None:
        """统一初始化同步/异步客户端"""
        common_params = {
            'api_key': self.api_key,
            'base_url': self.api_base,
            'max_retries': self.max_retries,
            'timeout': 120
        }
        
        self.sync_client = OpenAI(**common_params)
        self.async_client = AsyncOpenAI(**common_params)

    @property
    def current_time(self) -> str:
        """获取标准化当前时间"""
        return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    async def async_generate(
        self,
        messages: Union[str, List[Dict[str, str]]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """增强版异步生成，支持多种消息格式"""
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
                
            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                yield chunk
                    
        except Exception as e:
            logger.exception(f"Async generation failed: {e}")
            yield f"[ERROR] {str(e)}"

    def generate(
        self,
        prompt,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """同步生成文本"""
        try:
            response = self.sync_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.exception(f"Generation failed: {e}")
            return f"[ERROR] {str(e)}"
