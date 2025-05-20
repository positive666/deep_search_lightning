# 在顶部添加 List 导入
from typing import Dict, Any, List  # 修改这里

import os
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)
class PromptManager:
    def __init__(self, prompt_dir: str = None, language: str = 'zh', model_name: str = None):
        self.prompts = {}
        self.language = language
        self.model_name = model_name
        self.prompt_dir = Path(prompt_dir or str(Path(__file__).parent / "prompts"))
        self._load_prompts()
    
    def _load_prompts(self):
        """从JSON文件加载提示词模板"""
        if self.model_name:
            model_file = self.prompt_dir / f"prompts_{self.model_name}_{self.language}.json"
            if model_file.exists():
                with open(model_file, 'r', encoding='utf-8') as f:
                    model_prompts = json.load(f)
                    self.prompts.update(model_prompts)
                    logger.info(f"Loaded model-specific prompts from {model_file}")
        
        prompt_file = Path(self.prompt_dir) / f"prompts_{self.language}.json"
        try:
            if prompt_file.exists():
                # 增加文件可读性检查
                if not os.access(prompt_file, os.R_OK):
                    raise PermissionError(f"无法读取文件: {prompt_file}")
                    
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    self.prompts = json.load(f)
                    logger.info(f"成功加载提示词文件: {prompt_file}")
            else:
                logger.warning(f"提示词文件不存在，使用默认提示词: {prompt_file}")
                # 这里应该填充完整的默认提示词内容
                self.prompts = {
                    "intent": {
                        "template": """作为专业的网络检索助手，首先对用户输入进行分析并考虑是否启动联网搜索进行回答：
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

### 输出格式
[动作]
使用联网搜索工具|不使用联网搜索工具
[搜索任务]（仅联网时生成）  
• 任务1: "标准化查询任务1" #[类型:范围]
• 任务2: "标准化查询任务2" #[类型:范围]
• 任务3: "标准化查询任务3" #[类型:范围]
</任务>

请用{language}回答，严格按照规范输出，不要额外的解释。
""",
                        "variables": ["current_time", "context", "query", "tools", "generate_num", "language"]
                    },
                    "rewrite": {
                        "template": """作为查询优化助手，你需要根据以下对话历史和当前问题，生成{generate_num}个语义完整的子问题。
        
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
""",
                        "variables": ["history", "query", "generate_num", "language"]
                    },
                    "summary": {
                        "template": """As a professional web research analyst, you are summarizing search results for the following sub-tasks: {sub_tasks}. Current date: {current_time}.

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

Note: Use neutral language and maintain academic tone. All claims must be source-attributed.Please use {language} to answer the following question:""",
                        "variables": ["sub_tasks", "current_time", "search_results", "language"]
                    }
                }
        except Exception as e:
            logger.error(f"加载提示词失败: {e}", exc_info=True)
            self.prompts = {}

    def get_prompt(self, name: str, variables: Dict[str, Any]) -> str:
        """
        获取格式化后的提示词
        
        :param name: 提示词名称 (intent/rewrite/summary)
        :param variables: 变量字典
        :return: 格式化后的提示词字符串
        """
        if name not in self.prompts:
            raise ValueError(f"未知提示词模板: {name}")
            
        template = self.prompts[name]["template"]
        required_vars = set(self.prompts[name]["variables"])
        missing_vars = required_vars - set(variables.keys())
        
        if missing_vars:
            raise ValueError(f"缺少必要变量: {missing_vars}")
            
        return template.format(**variables)
    
    def add_prompt(self, name: str, template: str, variables: List[str]):
        """添加或更新提示词模板"""
        self.prompts[name] = {
            "template": template,
            "variables": variables
        }
    
    def save_prompts(self):
        """保存提示词到文件"""
        prompt_file = Path(self.prompt_dir) / f"prompts_{self.language}.json"
        try:
            os.makedirs(self.prompt_dir, exist_ok=True)
            with open(prompt_file, 'w', encoding='utf-8') as f:
                json.dump(self.prompts, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存提示词失败: {e}")

