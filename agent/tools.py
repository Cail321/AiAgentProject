"""
╔══════════════════════════════════════════════════════════════╗
║  agent/tools.py — 工具定义与执行                              ║
║                                                             ║
║  作用：Agent 的"手脚"。每个工具就是一个 Python 函数，            ║
║  外加一份 JSON Schema 描述，让 LLM 知道这个工具能干什么。        ║
║                                                             ║
║  依赖关系：                                                    ║
║    tools.py 被 agent/core.py 调用（执行工具）                   ║
║    tools.py 被 main.py 调用（注册工具列表传给 Agent）           ║
║    tools.py 调用 → rag/retriever.py（RAG_search 工具）        ║
║                                                             ║
║  你需要实现 4-5 个工具：                                       ║
║    1. calculator           — 数学计算                         ║
║    2. get_current_time     — 查询当前时间                     ║
║    3. search_knowledge     — RAG 知识库检索（核心！）          ║
║    4. get_order_status     — 查询订单状态（模拟数据库查询）     ║
║    5. transfer_to_human    — 转人工客服（模拟）                ║
║                                                             ║
║  每个工具分两部分：                                              ║
║    A. TOOL_SCHEMAS: 工具描述列表（JSON Schema），传给 LLM       ║
║    B. execute_tool():  根据工具名执行对应的 Python 函数         ║
║                                                             ║
║  面试要点：                                                    ║
║    Q: 为什么工具描述用 JSON Schema？                            ║
║    A: 这是 OpenAI Function Calling 的标准格式。LLM 通过         ║
║       description 字段理解工具用途，通过 parameters 知道        ║
║       需要传什么参数、什么类型。                                  ║
║                                                             ║
║    Q: execute_tool 里为什么要用 try/except？                   ║
║    A: LLM 可能传错误的参数格式，工具执行不能崩，要把错误          ║
║       信息作为正常结果返回给 LLM，让它自己修正。                   ║
╚══════════════════════════════════════════════════════════════╝
"""
import datetime

def create_tools(retriever=None, order_db=None):
    """工厂函数：接收外部依赖，返回 (工具描述列表, 执行函数)"""
    if order_db is None:
        order_db = {"ORD001": "已发货", "ORD002": "处理中", "ORD003": "待付款"}
    # 1. 构建 TOOL_SCHEMAS（一个 list，包含上面4-5
    schemas = [
        # 工具1: calculator — 数学表达式的计算器
        {
            "type": "function",
            "function":{
                "name": "calculator",
                "description": "执行数学计算。支持加减乘除、幂运算等。",
                "parameters": {
                    "type": "object",
                    "properties":{
                        "expression":{
                            "type": "string",
                            "description": "数学表达式，例如'(3+5)*2'"
                        }
                    },
                    "required": ["expression"]
                }
            }
        },
        # 工具2: get_current_time — 获取当前日期时间，无需参数
        {
            "type": "function",
            "function":{
                "name": "get_current_time",
                "description": "获取当前时间和日期。",
                "parameters": {
                    "type": "object",
                    "properties":{},
                    "required": []
                },
            }
        },
        # 工具3: search_knowledge — RAG知识库检索，参数 query: str
        # 这个工具需要从外部传入 retriever 对象，所以用工厂函数 create_tools(retriever)
        {
            "type": "function",
            "function":{
                "name": "search_knowledge",
                "description": "进行RAG知识库检索。",
                "parameters": {
                    "type": "object",
                    "properties":{
                        "query":{
                            "type": "string",
                            "description": "用户提出的问题。"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        # 工具4: get_order_status — 查询订单状态，参数 order_id: str
        # 模拟数据库：用一个 dict 存储 {"ORD001": "已发货", "ORD002": "处理中"}
        {
            "type": "function",
            "function":{
                "name": "get_order_status",
                "description": "查询订单状态。",
                "parameters": {
                    "type": "object",
                    "properties":{
                        "order_id":{
                            "type": "string",
                            "description": "查询的订单编号。"
                        }
                    },
                    "required": ["order_id"]
                }
            }
        },
        # 工具5: transfer_to_human — 转接人工客服，参数 reason: str
        {
            "type": "function",
            "function":{
                "name": "transfer_to_human",
                "description": "转接人工客服",
                "parameters": {
                    "type": "object",
                    "properties":{
                        "reason":{
                            "type": "string",
                            "description": "转接人工客服。"
                        }
                    },
                    "required": ["reason"]
                }
            }
        }
    ]
    # 2. 定义 execute_tool(name, args) 函数
    #    - 根据 name 路由到具体工具
    #    - search_knowledge 调用 retriever 对象
    #    - get_order_status 查询 order_db
    def execute_tool(name, args):
        try:
            if name == "calculator":
                result = eval(args["expression"])
                return str(result)

            elif name == "get_current_time":
                return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            elif name == "search_knowledge":
                if retriever is None:
                    return "知识库未初始化，请联系管理员"
                results = retriever.search(args["query"])
                return "找到以下相关信息：\n" + "\n".join([r["content"] for r in results])

            elif name == "get_order_status":
                order_id = args["order_id"]
                if order_id in order_db:
                    return f"订单{order_id}状态：{order_db[order_id]}"
                else:
                    return f"未找到订单{order_id}"

            elif name == "transfer_to_human":
                return f"转接人工客服，原因：{args.get('reason','未说明')}"

            else:
                return f"未知工具{name}"
        except KeyError as e:
            return f"缺少必要参数{e}"

        except ZeroDivisionError as e:
            return f"除以零错误"

        except Exception as e:
            return f"工具‘{name}’执行出错:{e}"
    # 3. 返回 (TOOL_SCHEMAS, execute_tool)
    return schemas, execute_tool
