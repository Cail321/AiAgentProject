"""
╔══════════════════════════════════════════════════════════════╗
║  agent/memory.py — 对话记忆管理                               ║
║                                                             ║
║  作用：管理 Agent 的对话历史。                                  ║
║    1. 存储每条消息（system/user/assistant/tool）               ║
║    2. 控制上下文长度（token 预算管理）                          ║
║    3. 支持"遗忘"旧的对话轮次（滑动窗口）                         ║
║                                                             ║
║  为什么需要专门的 memory 模块？                                ║
║    - LLM 有 context window 限制（比如 Qwen 最大 32K tokens）   ║
║    - 对话太长会超出限制，API 报错                                ║
║    - 需要策略来压缩或丢弃旧消息                                  ║
║                                                             ║
║  依赖关系：                                                    ║
║    memory.py 被 agent/core.py 调用                            ║
║    memory.py 不依赖其他项目模块（独立工具）                      ║
║                                                             ║
║  你需要实现：                                                  ║
║    class Memory:                                             ║
║      def __init__(self, system_prompt, max_tokens=8000)      ║
║      def add(self, role, content, tool_calls=None)           ║
║      def get_messages(self) -> list[dict]                    ║
║      def count_tokens(self) -> int     ← 估算当前 tokens 数   ║
║      def compress(self)               ← 超出限制时压缩/遗忘   ║
║      def clear(self)                  ← 重置对话             ║
║                                                             ║
║  面试要点：                                                    ║
║    Q: 怎么估算 token 数？                                      ║
║    A: 粗略法：中文 1 字≈1-2 token，英文 1 词≈1.3 token。        ║
║       精确法：用 tiktoken 库计算。这里用粗略法就够了。           ║
║                                                             ║
║    Q: 有哪些记忆管理策略？                                     ║
║    A: 1) 滑动窗口：只保留最近 N 轮                              ║
║       2) 摘要压缩：把旧对话让 LLM 总结成一段，替换原始消息        ║
║       3) 向量检索：把所有历史存向量库，每次检索相关部分           ║
║       我们先用滑动窗口，这是最实用的基线。                        ║
╚══════════════════════════════════════════════════════════════╝
"""

class Memory:
    def __init__(self, system_prompt: str, max_tokens: int = 8000):
        self.messages = [{"role": "system", "content": system_prompt}]
        self.max_tokens = max_tokens

    def add(self, role: str, content: str = None, tool_calls = None, tool_call_id=None, name = None):
        """添加一条消息到记忆"""
        # 构建消息 dict，追加到 self.messages
        msg = {"role": role, "content": content}
        #判断tool_calls是否为空,防止null值报错
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls
        if tool_call_id is not None:
            msg["tool_call_id"] = tool_call_id
        if name is not None:
            msg["name"] = name
        self.messages.append(msg)
        # 调用 self.compress() 检查是否超出限制
        self.compress()

    def get_messages(self) -> list:
        """返回当前完整对话历史（传给 LLM）"""
        return self.messages

    def count_tokens(self) -> int:
        """粗略估算当前 messages 的总 token 数"""
        # 中文约 1 字 ≈ 1.5 tokens，英文约 1 词 ≈ 1.3 tokens(精准计算token时间复杂度O(n**2))
        # 改为len()粗略计算
        total = 0
        for msg in self.messages:
            text = str(msg.get("content", "") or "")
            total += len(text)
        return total

    def compress(self):
        """滑动窗口策略：保留 system_prompt + 最近 N 轮对话"""
        # 如果 token 数不超限，不处理
        # 如果超限，从最早的非 system 消息开始删除
        # 确保 system_prompt 永远在第一位
        while self.count_tokens() > self.max_tokens and len(self.messages) > 1:
            self.messages.pop(1)


    def clear(self):
        """重置对话（只保留 system_prompt）"""
        self.messages = [self.messages[0]]