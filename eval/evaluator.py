"""记录 Agent 运行轨迹，并提供基础的离线统计指标。"""

import json
from datetime import datetime
from pathlib import Path


class Evaluator:
    """保存单轮轨迹，并汇总效率、工具调用和超时指标。"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.current_trace: list[dict] = []
        self.all_sessions: list[dict] = []

    def log_tool_call(self, tool_name: str, args: dict, result: str, duration: float) -> None:
        """记录一次工具调用及其耗时。"""
        self.current_trace.append(
            {
                "tool_name": tool_name,
                "args": args,
                "result": result,
                "duration": round(duration, 6),
            }
        )

    def log_final_answer(
        self,
        question: str,
        answer: str,
        total_iterations: int,
        total_duration: float,
    ) -> None:
        """结束一轮对话并保存最终回答。"""
        self.all_sessions.append(
            {
                "question": question,
                "answer": answer,
                "total_iterations": total_iterations,
                "total_duration": round(total_duration, 6),
                "tool_calls": self.current_trace.copy(),
            }
        )
        self.current_trace = []

    def score(self) -> dict:
        """汇总当前评估器中的基础运行指标。"""
        if not self.all_sessions:
            return {"total_sessions": 0}

        total = len(self.all_sessions)
        tool_calls = [call for session in self.all_sessions for call in session["tool_calls"]]
        rag_calls = [call for call in tool_calls if call["tool_name"] == "search_knowledge"]
        rag_hits = [
            call for call in rag_calls
            if "没有找到足够相关" not in call["result"]
            and "未初始化" not in call["result"]
        ]
        timeout_sessions = [
            session for session in self.all_sessions
            if "超过上限" in session["answer"] or "超时" in session["answer"]
        ]
        avg_iterations = sum(item["total_iterations"] for item in self.all_sessions) / total
        efficiency = round(max(0.0, 10.0 - avg_iterations * 2), 2)
        return {
            "total_sessions": total,
            "total_tool_calls": len(tool_calls),
            "avg_iterations": round(avg_iterations, 2),
            "avg_duration_seconds": round(
                sum(item["total_duration"] for item in self.all_sessions) / total,
                4,
            ),
            "rag_calls": len(rag_calls),
            "rag_hit_rate": round(len(rag_hits) / len(rag_calls), 2) if rag_calls else 0.0,
            "timeout_rate": round(len(timeout_sessions) / total, 2),
            "efficiency_score": efficiency,
        }

    def save(self) -> Path:
        """将当前会话和汇总指标保存为 JSON 文件。"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.log_dir / f"session_{timestamp}.json"
        payload = {"score": self.score(), "sessions": self.all_sessions}
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    def reset(self) -> None:
        """清空尚未结束的当前轨迹，不删除已完成会话。"""
        self.current_trace = []
