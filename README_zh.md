# Deep Search Lighting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

一个轻量的、纯净的大模型联网深度搜索方案，支持多种搜索引擎聚合搜索、深度反思和结果评估。介于联网搜索和DeepResearch之间的平衡方案,提供无三方框架的实现，便于开发人员快速集成。

## ✨ Features

- 支持多个搜索引擎的聚合搜索: 
✅ Baidu (免费)
✅ DuckDuckGo (免费但需要VPN)
✅ Bocha (需要API Key)
✅ Tavily (需要注册KEY)
- 反思策略和可控的评估,通过参数灵活调节流程复杂度
- 定制的管道适应所有的LLM模型,Openai-style的调用方式，兼容思考推理模型的输出调整
- 提供纯净源码便于集成
- 提供MCP服务
## 📺 演示视频
[点击查看演示视频](assets/demo.mp4)  
## 🔄 流程图
[Piepline](assets/piepline.png) 


## ✨ Why deepsearch_lightning?

     
    联网搜索是大模型的通用功能，但是很多开源的联网搜索在大模型交互平台效果都有明显的局限性，且需要强大的模型和付费的引擎支持。因为不是每一个模型都能很好支持工具调用模式，不同尺寸的和模式的模型在工具调用中的效果并不稳定，特别是小模型在指代、和上下文的意图中容易发生混淆，而deepsearch_lightning可以支持任意的模型。无任何三方框架依赖的实现，没有任何限制，你可以使用免费的API，能够保证不错的查询质量，并通过调整深度参数，去用推理时间换取更好的结果。即使能力相对较弱的小模型也能够适应。多引擎提供了召回源，采用了反思机制，让模型自己去思考和评估。
    额外提供了MCP servers和langgraph的调用参考，方便开发人员快速集成。你可以运行streamlit demo，体验一下。

    [实验规划]:参考了一些深度研究的webresearch方案，没有考虑采用爬虫或者网页解析Content的方式，考虑到一些小尺寸、上下文长度受限的情况；降低复杂度,没有引入文本Chunk 召回的策略，联网搜索是属于聊天交互的场景，因此考虑实时性，直接将搜索结果作为输入，让模型自己去思考和推理。 正在考虑加入RL训练的小尺寸召回模型。



## 🚀 快速开始

### 1. 安装
```bash
    conda create -n deepsearch_lightning python==3.11
    conda activate deepsearch_lightning
    pip install -r requirements.txt
    note: if you want to use langchain, you should install requirements_langchain.txt
```
### 2.🔧配置
    重命名  .env.examples to .env，填写你的模型信息，目前仅支持Openai-style，默认百度搜索是打开的，其他你可以根据需要关填写。
    
### 🚀 运行
```bash
    1. Test case
        python test_demo.py
    2. Streamlit demo
        streamlit run streamlit_app.py
    3. Run mcp server
        python mcp_server.py 
        python langgraph_mcp_client.py
```
###  计划
    
    🧪RL训练小尺寸的召回问答模型验证
    🧪策略的改进
    🧪多智能体框架的实现

🙌 欢迎贡献你的想法！通过[Issues]或[Pull Requests]参与项目。


###  许可
This repository is licensed under the [Apache-2.0 License](./LICENSE).
