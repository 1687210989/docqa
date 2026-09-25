from app.rag.pipeline import retrieve


def retrieve_chunks(question: str, top_k: int = 3) -> list[str]:
    """Retriever Agent：复用第三阶段 retrieve()，返回文档块文本列表。"""
    return retrieve(question, top_k=top_k)