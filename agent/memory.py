"""管理 Agent 的对话消息和上下文长度。"""

import json
from typing import Any


class Memory:
    """使用滑动窗口保留系统提示词和最近的完整对话轮次。"""

    def __init__(self, system_prompt: str, max_tokens: int = 8000):
        if max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")
        self.messages = [{"role": "system", "content": system_prompt}]
        self.max_tokens = max_tokens

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        """将 OpenAI 对象或普通对象转换成可传给 API 的字典。"""
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True)
        if isinstance(value, list):
            return [Memory._to_jsonable(item) for item in value]
        if isinstance(value, dict):
            return {key: Memory._to_jsonable(item) for key, item in value.items()}
        return value

    def add(
        self,
        role: str,
        content: str | None = None,
        tool_calls: list[Any] | None = None,
        tool_call_id: str | None = None,
        name: str | None = None,
    ) -> None:
        """追加消息，并在超出预算时裁剪旧对话。"""
        message: dict[str, Any] = {"role": role, "content": content}
        if tool_calls is not None:
            message["tool_calls"] = self._to_jsonable(tool_calls)
        if tool_call_id is not None:
            message["tool_call_id"] = tool_call_id
        if name is not None:
            message["name"] = name
        self.messages.append(message)
        self.compress()

    def get_messages(self) -> list[dict[str, Any]]:
        """返回可直接传入聊天接口的消息列表。"""
        return self.messages

    def count_tokens(self) -> int:
        """粗略估算上下文长度，生产环境可替换为模型对应的 tokenizer。"""
        return sum(
            len(json.dumps(message, ensure_ascii=False, default=str))
            for message in self.messages
        )

    def _drop_oldest_turn(self) -> bool:
        """删除最早的一轮用户对话及其后续工具调用消息。"""
        user_indices = [
            index for index, message in enumerate(self.messages[1:], start=1)
            if message.get("role") == "user"
        ]
        # 只有一轮当前对话时不能删除用户问题，否则下一次请求会失去上下文。
        if len(user_indices) < 2:
            return False

        first_user, next_user = user_indices[0], user_indices[1]
        del self.messages[first_user:next_user]
        return True

    def compress(self) -> None:
        """按完整对话轮次裁剪，避免留下孤立的工具调用结果。"""
        while self.count_tokens() > self.max_tokens and self._drop_oldest_turn():
            pass

    def clear(self) -> None:
        """重置对话，只保留系统提示词。"""
        self.messages = [self.messages[0]]
