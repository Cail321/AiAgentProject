"""
╔══════════════════════════════════════════════════════════════╗
║  agent/core.py — Agent 核心循环（ReAct 模式）                 ║
║                                                             ║
║  作用：这是整个项目的大脑。                                      ║
║    1. 接收用户输入                                             ║
║    2. 调用 LLM，判断是直接回答还是调用工具                        ║
║    3. 如果要调工具 → 执行工具 → 结果喂回 LLM → 重复              ║
║    4. 如果能回答 → 返回最终答案                                 ║
║                                                             ║
║  核心数据结构：messages（对话历史列表）                          ║
║    格式: [{"role": "system/user/assistant/tool",             ║
║             "content": "...",                                ║
║             "tool_calls": [...]}]                            ║
║                                                             ║
║  依赖关系：                                                    ║
║    core.py 被 main.py 调用                                    ║
║    core.py 调用 → agent/tools.py（执行工具）                    ║
║    core.py 调用 → agent/memory.py（读写记忆）                   ║
║    core.py 调用 → eval/evaluator.py（记录日志）                ║
║                                                             ║
║  你需要实现：                                                  ║
║    class Agent:                                              ║
║      def __init__(self, llm, tools, memory, evaluator)       ║
║      def _call_llm(self) -> 调用 LLM 并返回响应               ║
║      def _execute_tools(self, tool_calls) -> 执行工具列表     ║
║      def run(self, user_input) -> str   ← 核心循环在这里     ║
║      def reset(self) -> 重置对话                               ║
║                                                             ║
║  关键设计决策（面试会问）：                                     ║
║    Q: 为什么用 while 循环而不是递归？                           ║
║    A: 递归有调用栈深度限制；while 循环更直观、更好控制上限        ║
║                                                             ║
║    Q: 怎么防止死循环？                                         ║
║    A: MAX_ITERATIONS 限制 + 重复工具调用检测                   ║
║                                                             ║
║    Q: 和 LangChain 的 AgentExecutor 有什么区别？               ║
║    A: LangChain 封装了错误重试、中间步骤记录、流式输出等，        ║
║       但核心逻辑就是这个 while 循环                             ║
╚══════════════════════════════════════════════════════════════╝
"""

# 参考你今天写的 agent.py 里的 Agent 类
# 实现时注意：
# 1. 把 execute_tool 的调用改成从 agent/tools.py 导入
# 2. 每次 LLM 调用后，把中间步骤传给 evaluator 记录
# 3. 记忆管理交给 memory.py，不要直接在 core.py 里操作 messages
from utils.config import LLM_MODEL,MAX_ITERATIONS
import json

class Agent:
    def __init__(self, llm_client, tools, execute_tool, memory, evaluator=None):
        self.client = llm_client       # OpenAI 兼容客户端
        self.model = LLM_MODEL         # 模型名
        self.tools = tools             # 工具列表（从 tools.py 传入）
        self.memory = memory           # 记忆管理器
        self.evaluator = evaluator     # 评估器（可选）
        self.max_iterations = MAX_ITERATIONS   # 防止死循环
        self.execute_tool = execute_tool # 使用工具

    def reset(self):
        self.memory.clear()
        if self.evaluator:
            self.evaluator.reset()

    def run(self, user_input: str) -> str:
        """核心 ReAct 循环 — 面试必问"""
        # 1. 用户输入加入记忆
        self.memory.add("user", user_input)
        # 2. while 循环（最多 max_iterations 轮）：
        #    a. 调用 LLM
        #    b. 如果没有 tool_calls → 返回答案，退出循环
        #    c. 如果有 tool_calls → 逐个执行 → 结果加入记忆 → 继续循环
        for iteration in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.memory.get_messages(),
                tools=self.tools,
            )
            msg = response.choices[0].message
            if msg.tool_calls is None:
                self.memory.add("assistant", msg.content)
                if self.evaluator:
                    self.evaluator.log_final_answer(user_input, msg.content, iteration + 1, 0.0)
                return msg.content
            else:
                self.memory.add("assistant", msg.content, tool_calls=msg.tool_calls)
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    result = self.execute_tool(name, args)
                    self.memory.add("tool", result, tool_call_id=tc.id, name=name)
                    if self.evaluator:
                        self.evaluator.log_tool_call(name, args, result, duration=0.0)
        if self.evaluator:
            self.evaluator.log_final_answer(user_input, "抱歉，请求超时", self.max_iterations, 0.0)
        return "抱歉，请求超时，请稍后重试。"


