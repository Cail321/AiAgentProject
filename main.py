"""项目入口：组装 Agent 依赖并提供命令行交互。"""

from pathlib import Path

from openai import OpenAI

from agent.core import Agent
from agent.memory import Memory
from agent.tools import create_tools
from eval.evaluator import Evaluator
from rag.embedder import Embedder
from rag.retriever import Retriever
from utils.config import (
    DATA_DIR,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LOG_DIR,
    MAX_MEMORY_TOKENS,
    SYSTEM_PROMPT,
)


def build_agent() -> tuple[Agent, Evaluator]:
    """初始化模型、知识库、工具、记忆和评估器。"""
    llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    embedding_client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)

    embedder = Embedder(embedding_client, model=EMBEDDING_MODEL)
    retriever = Retriever(embedder)
    retriever.load_documents(str(Path(DATA_DIR) / "company_docs.txt"))

    schemas, execute_tool = create_tools(retriever)
    memory = Memory(system_prompt=SYSTEM_PROMPT, max_tokens=MAX_MEMORY_TOKENS)
    evaluator = Evaluator(log_dir=LOG_DIR)
    agent = Agent(llm_client, schemas, execute_tool, memory, evaluator)
    return agent, evaluator


def main() -> None:
    """启动命令行对话，并在退出时保存评估日志。"""
    print("AI Agent 项目启动中...")
    try:
        agent, evaluator = build_agent()
    except Exception as exc:
        print(f"初始化失败：{exc}")
        return

    print("各模块加载完成，已启动 Agent！输入 exit 或 quit 退出。")
    while True:
        try:
            user_input = input("请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() in ("reset", "clear"):
            agent.reset()
            print("对话已重置。")
            continue
        if not user_input:
            print("请输入有效的问题。")
            continue

        print(f"回答：{agent.run(user_input)}")

    if evaluator.all_sessions:
        log_path = evaluator.save()
        print("对话结束，评分：", evaluator.score())
        print(f"运行日志已保存：{log_path}")


if __name__ == "__main__":
    main()
