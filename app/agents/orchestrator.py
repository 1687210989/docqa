from app.agents.router import route
from app.agents.retriever import retrieve_chunks
from app.agents.answerer import answer, chat_only
from app.agents.critic import pass_check

def handle(question: str, top_k: int = 3) -> dict:
    """手写编排：Router → (Retriever → Answerer → Critic) / 直接闲聊。"""
    intent = route(question)
    if intent != "rag":                                  # chat / unknown：不检索
        return {"intent": intent, "answer": chat_only(question), "sources": []}

    chunks = retrieve_chunks(question, top_k=top_k)      # Retriever 上
    ans = answer(question, chunks)                       # Answerer 答
    passed = pass_check(question, ans)                   # Critic 验
    if not passed:
        ans = answer(question, chunks, retry=True)       # 不过关打回重答一次
    return {"intent": intent, "answer": ans, "sources": chunks, "critic_passed": passed}