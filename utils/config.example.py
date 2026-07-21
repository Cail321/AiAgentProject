"""安全配置模板。

将本文件复制为 ``config.py``。密钥从环境变量读取，禁止提交到仓库。
"""

import os


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少必要的环境变量：{name}")
    return value


LLM_API_KEY = _required("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

EMBEDDING_API_KEY = _required("EMBEDDING_API_KEY")
EMBEDDING_BASE_URL = os.getenv(
    "EMBEDDING_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v2")

MAX_ITERATIONS = 5
MAX_MEMORY_TOKENS = 8000
CHUNK_SIZE = 300
TOP_K_RETRIEVAL = 5
DATA_DIR = "data"
LOG_DIR = "logs"

SYSTEM_PROMPT = """
你是一个智能客服助手，服务于某电商平台。

行为规则：
- 使用简洁、友好的中文回答。
- 产品信息、物流和退换货政策必须先调用 search_knowledge。
- 订单查询必须先确认用户提供了订单号。
- 不要编造知识库中没有的信息；不确定时建议转人工。
- 只有在当前信息足够时才直接回答，避免重复调用工具。
""".strip()
