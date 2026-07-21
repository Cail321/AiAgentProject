"""实现轻量级的文档切分、向量索引和相似度检索。"""

import re
from pathlib import Path

import numpy as np

from utils.config import CHUNK_SIZE, TOP_K_RETRIEVAL


class Retriever:
    """在内存中保存文档片段及其向量，并使用余弦相似度检索。"""

    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks: list[str] = []
        self.embeddings: list[list[float]] = []

    def load_documents(self, file_path: str) -> None:
        """读取文本文件、切分文档并构建向量索引。"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"知识库文件不存在：{path}")
        content = path.read_text(encoding="utf-8")
        self.chunks = self.chunk_text(content)
        self.build_index()

    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
        """优先按段落和句子切分，过长句子再按长度切分。"""
        if not text or chunk_size <= 0:
            return []

        chunks: list[str] = []
        paragraphs = re.split(r"\n\s*\n", text.strip())
        for paragraph in paragraphs:
            sentences = [
                sentence.strip()
                for sentence in re.split(r"(?<=[。！？!?；;])", paragraph)
                if sentence.strip()
            ]
            current = ""
            for sentence in sentences:
                if len(sentence) > chunk_size:
                    if current:
                        chunks.append(current.strip())
                        current = ""
                    chunks.extend(
                        sentence[index:index + chunk_size].strip()
                        for index in range(0, len(sentence), chunk_size)
                    )
                elif len(current) + len(sentence) > chunk_size:
                    if current:
                        chunks.append(current.strip())
                    current = sentence
                else:
                    current += sentence
            if current:
                chunks.append(current.strip())

        return [chunk for chunk in chunks if chunk]

    def build_index(self) -> None:
        """批量调用 Embedding 服务生成索引。"""
        self.embeddings = self.embedder.embed_batch(self.chunks) if self.chunks else []
        if len(self.embeddings) != len(self.chunks):
            raise ValueError("向量数量与文档片段数量不一致")

    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        """返回按余弦相似度排序的文档片段及分数。"""
        if not query or not query.strip() or not self.chunks or top_k <= 0:
            return []

        query_vector = np.asarray(self.embedder.embed(query), dtype=float)
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []

        results = []
        for chunk, embedding in zip(self.chunks, self.embeddings):
            chunk_vector = np.asarray(embedding, dtype=float)
            chunk_norm = np.linalg.norm(chunk_vector)
            if chunk_norm == 0:
                continue
            score = float(np.dot(query_vector, chunk_vector) / (query_norm * chunk_norm))
            results.append({"score": score, "content": chunk})

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]
