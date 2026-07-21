"""定义并执行 Agent 可调用的工具。"""

import ast
import datetime
import operator
from typing import Callable


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_calculate(expression: str) -> float:
    """只允许数字和白名单运算符，避免直接执行任意 Python 代码。"""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("表达式不能为空")
    if len(expression) > 100:
        raise ValueError("表达式过长")

    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("幂运算指数过大")
            return _BINARY_OPERATORS[type(node.op)](left, right)
        raise ValueError("表达式包含不支持的语法")

    return evaluate(tree)


def create_tools(retriever=None, order_db=None):
    """创建工具描述和执行函数，并注入检索器、订单数据等外部依赖。"""
    if order_db is None:
        order_db = {"ORD001": "已发货", "ORD002": "处理中", "ORD003": "待付款"}

    schemas = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "执行基础数学计算，支持加减乘除、取模、整除和幂运算。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "例如：'(3+5)*2'"}
                    },
                    "required": ["expression"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "获取当前日期和时间。",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": "检索电商产品、物流、会员和退换货知识库。",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "需要检索的问题"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_order_status",
                "description": "根据订单编号查询订单状态。",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string", "description": "订单编号"}},
                    "required": ["order_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "transfer_to_human",
                "description": "将复杂、敏感或无法解决的问题转交人工客服。",
                "parameters": {
                    "type": "object",
                    "properties": {"reason": {"type": "string", "description": "转人工原因"}},
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    def execute_tool(name: str, args: dict) -> str:
        """根据工具名路由执行，并将异常转换为可供 Agent 理解的结果。"""
        try:
            args = args or {}
            if name == "calculator":
                return str(_safe_calculate(args["expression"]))
            if name == "get_current_time":
                return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if name == "search_knowledge":
                if retriever is None:
                    return "知识库未初始化，请联系管理员。"
                results = retriever.search(args["query"])
                if not results:
                    return "知识库中没有找到足够相关的信息。"
                return "找到以下相关信息：\n" + "\n".join(
                    f"[{item['score']:.3f}] {item['content']}" for item in results
                )
            if name == "get_order_status":
                order_id = str(args["order_id"]).strip().upper()
                return f"订单{order_id}状态：{order_db.get(order_id, '未找到该订单')}"
            if name == "transfer_to_human":
                reason = str(args.get("reason", "未说明")).strip()
                return f"已转接人工客服，原因：{reason}"
            return f"未知工具：{name}"
        except KeyError as exc:
            return f"缺少必要参数：{exc.args[0]}"
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
            return f"工具参数或计算错误：{exc}"
        except Exception as exc:
            return f"工具执行出错：{exc}"

    return schemas, execute_tool
