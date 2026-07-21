"""封装 OpenAI 兼容的文本向量化接口。"""


class Embedder:
    """提供单条和批量文本向量化能力。"""

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def embed(self, text: str) -> list[float]:
        """将单条文本转换为向量。"""
        if not text or not text.strip():
            raise ValueError("待向量化文本不能为空")
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float",
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量将文本转换为向量，减少网络请求次数。"""
        if not texts:
            return []
        if any(not text or not text.strip() for text in texts):
            raise ValueError("批量向量化文本中不能包含空字符串")
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]
