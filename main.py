"""
╔══════════════════════════════════════════════════════════════╗
║  main.py — 项目入口                                          ║
║                                                             ║
║  作用：                                                      ║
║    1. 初始化所有模块（Agent、工具、RAG、评估器）                 ║
║    2. 提供命令行交互界面（终端对话）                             ║
║    3. 串联完整流程：用户输入 → Agent处理 → 评估 → 返回          ║
║                                                             ║
║  调用链：                                                     ║
║    main.py                                                   ║
║      ├── utils/config.py          ← 加载 API Key、模型配置    ║
║      ├── rag/embedder.py          ← 初始化 Embedding 模型     ║
║      ├── rag/retriever.py         ← 加载知识库、建索引        ║
║      ├── agent/tools.py           ← 注册所有工具              ║
║      ├── agent/memory.py          ← 初始化对话记忆            ║
║      ├── agent/core.py            ← 创建 Agent 实例           ║
║      └── eval/evaluator.py        ← 初始化评估器              ║
║                                                             ║
║  面试要点：main.py 只做"组装"，不负责任何业务逻辑。              ║
║  这是面试官想看到的：关注点分离（Separation of Concerns）。      ║
╚══════════════════════════════════════════════════════════════╝
"""
from openai import OpenAI

from utils.config import (LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
                          EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL,
                          SYSTEM_PROMPT)
from rag.embedder import Embedder
from rag.retriever import Retriever
from agent.tools import create_tools
from agent.memory import Memory
from agent.core import Agent
from eval.evaluator import Evaluator

if __name__ == "__main__":
    print("AI Agent 项目启动中...")
    #两个客户端
    llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    emb_client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
    #RAG
    embedder = Embedder(emb_client)
    retriever = Retriever(embedder)
    retriever.load_documents("data/company_docs.txt")
    #工具
    schemas, execute_tool = create_tools(retriever)
    #记忆
    memory = Memory(system_prompt=SYSTEM_PROMPT)
    #评估器
    evaluator = Evaluator()
    #Agent
    agent = Agent(llm_client, schemas, execute_tool, memory, evaluator)
    print("各模块加载完成，已启动Agent！输入\'exit\'or\'quit\'退出\n")
    while True:
        user_input = input("请输入问题：")
        if user_input.lower() in ("exit","quit"):
            break
        if user_input.lower() in ("reset", "clear"):
            agent.reset()
            print("对话已重置\n")
            continue
        try:
            answer = agent.run(user_input)
            print(f"回答: {answer}")
        except Exception as e:
            print(f"出错了：{e}")
    print("对话结束，评分：", evaluator.score())