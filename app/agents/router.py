from app.llm import chat_complete


SYSTEM = ("你是意图分类器。只回答一个词：rag（需查资料才能答）、chat（闲聊或常识）、"
          "unknown（无法判断）。不要解释。")
def route(question: str) -> str:
    """Router Agent：判断问题类型 rag / chat / unknown。"""
    resp = chat_complete([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ])
    label = (resp.choices[0].message.content or "").strip().lower()   # content 可能是 None
    return label if label in ("rag", "chat", "unknown") else "unknown"