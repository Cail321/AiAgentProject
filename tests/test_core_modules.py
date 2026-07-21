"""验证工具、记忆、RAG 切分和评估器的基础行为。"""

import os
import tempfile
import unittest

os.environ.setdefault("LLM_API_KEY", "test-llm-key")
os.environ.setdefault("EMBEDDING_API_KEY", "test-embedding-key")

from agent.memory import Memory
from agent.tools import create_tools
from eval.evaluator import Evaluator
from rag.retriever import Retriever


class FakeEmbedder:
    """为测试提供不依赖网络的向量化实现。"""

    def embed_batch(self, texts):
        return [[float("产品" in text), float("物流" in text)] for text in texts]

    def embed(self, text):
        return [float("产品" in text), float("物流" in text)]


class CoreModuleTests(unittest.TestCase):
    def test_calculator_rejects_code_execution(self):
        """计算器应拒绝函数调用和属性访问。"""
        _, execute_tool = create_tools()
        self.assertEqual(execute_tool("calculator", {"expression": "(3+5)*2"}), "16")
        result = execute_tool("calculator", {"expression": "__import__('os').getcwd()"})
        self.assertIn("错误", result)

    def test_memory_removes_complete_old_turn(self):
        """上下文超限时不能留下孤立的工具消息。"""
        memory = Memory("系统提示词", max_tokens=80)
        memory.add("user", "旧问题" * 10)
        memory.add("assistant", None, tool_calls=[{"id": "1", "type": "function"}])
        memory.add("tool", "旧工具结果" * 10, tool_call_id="1", name="demo")
        memory.add("user", "新问题")
        roles = [message["role"] for message in memory.get_messages()]
        self.assertEqual(roles[0], "system")
        self.assertNotIn("tool", roles)

    def test_retriever_chunk_and_search(self):
        """检索器应正确切分文本并返回排序结果。"""
        retriever = Retriever(FakeEmbedder())
        retriever.chunks = retriever.chunk_text("产品价格为 99 元。\n\n物流支持全国配送。", 20)
        retriever.build_index()
        results = retriever.search("产品价格", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("产品价格", results[0]["content"])

    def test_evaluator_save_and_score(self):
        """评估器应生成 JSON 日志并返回基本指标。"""
        with tempfile.TemporaryDirectory() as directory:
            evaluator = Evaluator(directory)
            evaluator.log_tool_call("search_knowledge", {"query": "价格"}, "找到相关信息", 0.01)
            evaluator.log_final_answer("价格是多少", "99 元", 2, 0.02)
            score = evaluator.score()
            output_path = evaluator.save()
            self.assertEqual(score["total_sessions"], 1)
            self.assertEqual(score["rag_hit_rate"], 1.0)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
