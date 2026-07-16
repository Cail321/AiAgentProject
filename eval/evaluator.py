"""
╔══════════════════════════════════════════════════════════════╗
║  eval/evaluator.py — 对话质量评估                             ║
║                                                             ║
║  作用：                                                       ║
║    1. 记录 Agent 的每一步操作（哪个工具被调用、耗时多久）         ║
║    2. 对话结束后自动打分                                       ║
║    3. 保存日志到文件（方便面试时展示"你会做评估"）                ║
║                                                             ║
║  为什么需要这个模块？                                          ║
║    截图JD明确要求："跟踪 Agent 的运行轨迹，定位逻辑缺陷，         ║
║    进行端到端的性能调优"。这个模块就是为此设计的。                ║
║                                                             ║
║  依赖关系：                                                    ║
║    evaluator.py 被 agent/core.py 调用（每轮对话后记录）        ║
║    evaluator.py 被 main.py 调用（初始化时传入 Agent）          ║
║                                                             ║
║  你需要实现：                                                  ║
║    class Evaluator:                                          ║
║      def __init__(self, log_dir="logs")                      ║
║      def log_turn(self, user_input, agent_output,            ║
║                    tool_calls, iterations, duration)          ║
║      def score(self) -> dict         ← 自动评分              ║
║      def save_log(self)              ← 保存到 JSON 文件      ║
║                                                             ║
║  评分维度（面试时可以说的）：                                   ║
║    1. 工具调用次数：少 = 高效                                   ║
║    2. 是否找到相关文档：RAG 命中率                              ║
║    3. 是否超最大轮次：死循环检测                                ║
║    4. 用户问题复杂度：用于分类统计                               ║
╚══════════════════════════════════════════════════════════════╝
"""

class Evaluator:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.current_trace = []  # 当前对话的每一步
        self.all_sessions = []  # 所有对话记录

    def log_tool_call(self, tool_name: str, args: dict, result: str, duration: float):
        """记录一次工具调用"""
        # 记录：工具名、参数、结果、耗时
        self.current_trace.append((tool_name, args, result, duration))

    def log_final_answer(self, question: str, answer: str, total_iterations: int, total_duration: float):
        """记录一轮对话的最终结果"""
        session = {
            "question": question,
            "answer": answer,
            "total_iterations": total_iterations,
            "total_duration": total_duration,
            "tool_calls": self.current_trace.copy()
        }
        self.all_sessions.append(session)
        self.current_trace = []

    def score(self) -> dict:
        """对当前对话自动评分"""
        # 返回 {"efficiency": 0-10, "tool_usage": int, "hit_rag": bool, "timeout": bool}
        if not self.all_sessions:
            return {"total": 0}
        else:
            total = len(self.all_sessions)
            avg_iter = sum(s["total_iterations"] for s in self.all_sessions) / total
            efficiency = round(max(0.0, 10.0 - avg_iter * 2), 1)
        return {
            "total_sessions": total,
            "avg_iterations": avg_iter,
            "efficiency_score": efficiency,
        }

    def save(self):
        """保存本轮对话日志到 logs/ 目录（JSON 格式）"""
        # 文件名：logs/session_20260715_001500.json

    def reset(self):
        self.current_trace = []
