from app.llm import chat_complete

PROMPT = ("请只根据下面【资料】回答问题。资料中没有就如实说“资料中未提及”。\n\n"
          "【资料】\n{context}\n\n【问题】{question}")

def answer(question: str, chunks: list[str], retry: bool = False) -> str:
    """Answerer Agent：结合检索资料作答。retry=True 用于被 Critic 打回后重答。"""
    context = "\n\n".join(chunks) if chunks else "（无相关资料）"
    prefix = "上次回答不够准确，请更严谨地" if retry else ""
    resp = chat_complete([{"role": "user", "content": prefix + PROMPT.format(
        context=context, question=question)}])
    return resp.choices[0].message.content

def chat_only(question: str) -> str:
    """Answerer 的闲聊模式：不检索，直接对话。"""
    resp = chat_complete([{"role": "user", "content": question}])
    return resp.choices[0].message.content