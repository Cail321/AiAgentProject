"""
╔══════════════════════════════════════════════════════════════╗
║  rag/embedder.py — 文本向量化                                 ║
║                                                             ║
║  作用：把一段文字转换成一个浮点数列表（向量）。                     ║
║  为什么需要向量？因为计算机没法直接比较两段话"意思像不像"，         ║
║  但可以比较两个向量的余弦距离——语义越近，向量越近。                ║
║                                                             ║
║  依赖关系：                                                    ║
║    embedder.py 被 rag/retriever.py 调用                       ║
║    embedder.py 调用 → 通义千问 Embedding API                   ║
║                                                             ║
║  你需要实现：                                                  ║
║    class Embedder:                                           ║
║      def __init__(self, client, model="text-embedding-v2")   ║
║      def embed(self, text: str) -> list[float]               ║
║      def embed_batch(self, texts: list) -> list[list[float]] ║
║                                                             ║
║  面试要点：                                                    ║
║    Q: text-embedding-v2 输出多少维向量？                       ║
║    A: 1536 维（和 OpenAI text-embedding-ada-002 一样）        ║
║                                                             ║
║    Q: 为什么需要 embed_batch？                                ║
║    A: 一条一条调 API 太慢。批量调用可以减少网络往返次数，          ║
║       构建知识库索引用。                                       ║
╚══════════════════════════════════════════════════════════════╝
"""
from openai import OpenAI

class Embedder:
    def __init__(self, client, model="text-embedding-v2"):
        self.client = client   # OpenAI 兼容客户端
        self.model = model

    def embed(self, text: str) -> list[float]:
        """单条文本 → 向量"""
        response = self.client.embeddings.create(
            model=self.model,
            input = text,
            encoding_format="float")
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量文本 → 向量列表（建索引时用，效率更高）"""
        response = self.client.embeddings.create(
            model=self.model,
            input = texts,
            encoding_format="float")
        return [item.embedding for item in response.data]

