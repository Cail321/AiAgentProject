"""实现基于 Function Calling 的手写 ReAct Agent。"""

import json
import time

from utils.config import LLM_MODEL, MAX_ITERATIONS


class Agent:
    """负责组织 LLM 调用、工具执行、记忆更新和运行轨迹记录。"""

    def __init__(self, llm_client, tools, execute_tool, memory, evaluator=None):
        self.client = llm_client
        self.model = LLM_MODEL
        self.tools = tools
        self.execute_tool = execute_tool
        self.memory = memory
        self.evaluator = evaluator
        self.max_iterations = MAX_ITERATIONS

    def reset(self) -> None:
        """清空当前对话，但保留系统提示词。"""
        self.memory.clear()
        if self.evaluator:
            self.evaluator.reset()

    def _record_final(self, question: str, answer: str, iterations: int, started: float) -> None:
        """统一记录一轮对话的最终结果。"""
        if self.evaluator:
            self.evaluator.log_final_answer(
                question,
                answer,
                iterations,
                time.perf_counter() - started,
            )

    def run(self, user_input: str) -> str:
        """执行一轮 ReAct 循环，直到得到答案或达到迭代上限。"""
        if not user_input or not user_input.strip():
            return "请输入有效的问题。"

        started = time.perf_counter()
        question = user_input.strip()
        self.memory.add("user", question)
        seen_calls: set[tuple[str, str]] = set()

        for iteration in range(1, self.max_iterations + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.memory.get_messages(),
                    tools=self.tools,
                )
                message = response.choices[0].message
            except Exception as exc:
                answer = f"调用模型失败：{exc}"
                self._record_final(question, answer, iteration, started)
                return answer

            tool_calls = message.tool_calls or []
            if not tool_calls:
                answer = message.content or "抱歉，模型没有生成有效回答。"
                self.memory.add("assistant", answer)
                self._record_final(question, answer, iteration, started)
                return answer

            self.memory.add(
                "assistant",
                message.content,
                tool_calls=tool_calls,
            )
            for tool_call in tool_calls:
                name = tool_call.function.name
                raw_arguments = tool_call.function.arguments or "{}"
                started_tool = time.perf_counter()
                try:
                    args = json.loads(raw_arguments)
                    call_key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
                    if call_key in seen_calls:
                        result = "检测到重复工具调用，请根据已有结果直接回答。"
                    else:
                        seen_calls.add(call_key)
                        result = self.execute_tool(name, args)
                except (json.JSONDecodeError, TypeError) as exc:
                    args = {}
                    result = f"工具参数不是合法 JSON：{exc}"
                except Exception as exc:
                    args = {}
                    result = f"工具调用失败：{exc}"

                self.memory.add(
                    "tool",
                    result,
                    tool_call_id=tool_call.id,
                    name=name,
                )
                if self.evaluator:
                    self.evaluator.log_tool_call(
                        name,
                        args,
                        result,
                        time.perf_counter() - started_tool,
                    )

        answer = "抱歉，处理步骤超过上限，请稍后重试。"
        self._record_final(question, answer, self.max_iterations, started)
        return answer
