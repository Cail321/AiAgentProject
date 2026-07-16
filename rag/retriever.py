"""
╔══════════════════════════════════════════════════════════════╗
║  rag/retriever.py — 知识库检索器                              ║
║                                                             ║
║  作用：                                                       ║
║    1. 加载文档 → 切片（chunking）                              ║
║    2. 每个切片生成向量 → 存起来（索引）                          ║
║    3. 用户提问时，把问题也转成向量                               ║
║    4. 用余弦相似度找到最相关的 top_k 个切片                     ║
║                                                             ║
║  依赖关系：                                                    ║
║    retriever.py 被 agent/tools.py 调用（作为 search_knowledge 工具的后端）  ║
║    retriever.py 调用 → rag/embedder.py（生成向量）              ║
║    retriever.py 被 main.py 调用（初始化时加载文档）              ║
║                                                             ║
║  你需要实现：                                                  ║
║    class Retriever:                                          ║
║      def __init__(self, embedder: Embedder)                  ║
║      def load_documents(self, file_path: str)                ║
║      def chunk_text(self, text: str, chunk_size=300)         ║
║      def build_index(self)                                   ║
║      def search(self, query: str, top_k=3) -> list[dict]     ║
║                                                             ║
║  面试要点：                                                    ║
║    Q: 切片大小怎么选？                                          ║
║    A: 太小（<100字）语义不完整；太大（>500字）检索不精确。         ║
║       300字左右是经验值。中文按句号/段落切，英文按句子切。         ║
║                                                             ║
║    Q: 余弦相似度是什么？                                       ║
║    A: 衡量两个向量在方向上有多接近。=1 表示完全同向（语义相同），   ║
║       =0 表示正交（语义无关）。公式：cos = A·B / (|A|·|B|)      ║
║                                                             ║
║    Q: 如果知识库有 10000 篇文档，每次搜索要遍历全部向量吗？       ║
║    A: 暴力遍历 O(n) 在小规模（<1万）够用。生产环境用向量数据库    ║
║       （如 ChromaDB、Milvus、Pinecone），它们用 ANN 近似算法    ║
║       把复杂度降到 O(log n)。                                  ║
╚══════════════════════════════════════════════════════════════╝
"""
import numpy as np
import re
from utils.config import CHUNK_SIZE, TOP_K_RETRIEVAL
class Retriever:
    def __init__(self, embedder):
        self.embedder = embedder         # Embedder 实例
        self.chunks = []                 # 文档片段列表
        self.embeddings = []             # 每个片段对应的向量

    def load_documents(self, file_path: str):
        """从文件加载文档"""
        # 读取 data/ 目录下的 .txt 文件
        # 调用 chunk_text 切片
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.chunks = self.chunk_text(content, chunk_size=300)
        self.build_index()

    def chunk_text(self,
                   text: str,
                   chunk_size: int = CHUNK_SIZE) -> list[str]:
        """文档切片 — 按段落+句号切，保证语义完整"""
        # 策略：优先按 \n\n 切（段落），然后按。！？切（句子）
        # 每个 chunk 尽量接近 chunk_size，但不在句子中间切断
        split_text = text.split('\n\n')
        split_chunk = []
        for chunk in split_text:
            if len(chunk) > chunk_size:
                short_chunk = re.split(r'[。！？]',chunk)
                current_chunk = ""
                for i in short_chunk:
                    if not i:
                        continue
                    elif len(current_chunk) + len(i) > chunk_size:
                        split_chunk.append(current_chunk)
                        current_chunk = i
                    else:
                        current_chunk += i
                if current_chunk:
                    split_chunk.append(current_chunk)
            elif len(chunk) <= chunk_size:
                split_chunk.append(chunk)
        split_chunk = [c for c in split_chunk if c.strip()]
        return split_chunk

    def build_index(self):
        """为所有切片生成向量索引"""
        self.embeddings = self.embedder.embed_batch(self.chunks)

    def search(self,
               query: str,
               top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        """语义搜索 — 输入问题，返回最相关的文档片段"""
        # 1. query 转向量
        # 2. 计算与每个 chunk 的余弦相似度
        # 3. 排序取 top_k
        # 4. 返回 [{"score": float, "content": str}, ...]
        query_vec = self.embedder.embed(query)
        result = []
        # 使用 zip 同时遍历向量和文本
        for chunk_vec, chunk_text in zip(self.embeddings, self.chunks):
            # 计算余弦相似度
            score = np.dot(query_vec, chunk_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec))
            # 直接拿到 chunk_text，原本使用chunks[i]索引用来取文本。
            result.append((score, chunk_text))

        # 按分数降序排序
        result.sort(key=lambda x: x[0], reverse=True)
        return [{"score": r[0], "content": r[1]} for r in result[:top_k]]

