import chromadb
from app.config import settings

# 【P0 修复】绝对路径："./chroma_db" 会跟着启动目录跑
chroma = chromadb.PersistentClient(path=settings.chroma_dir)
COLLECTION = "documents"

_local_model = None
def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    return _local_model


def embed(texts: list[str]) -> list[list[float]]:
    """把若干段文本转成向量。"""
    if settings.embedding_provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key, base_url="https://api.openai.com/v1")
        resp = client.embeddings.create(model=settings.embedding_model, input=texts)
        return [d.embedding for d in resp.data]
    # 默认走本地模型（免费、离线）
    vecs = _get_local_model().encode(texts, normalize_embeddings=True)
    return vecs.tolist()


def index_document(doc_id: int, chunks: list[str]) -> None:
    """把一篇文档的切块存进向量库。"""
    if not chunks:
        return
    col = chroma.get_or_create_collection(COLLECTION)
    col.add(
        ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks),
        metadatas=[{"doc_id": doc_id} for _ in chunks],
    )

def delete_document(doc_id: int) -> None:
    """删掉某篇文档在向量库里的全部切片。"""
    col = chroma.get_or_create_collection(COLLECTION)
    col.delete(where={"doc_id": doc_id})

def retrieve(query: str, top_k: int = 3) -> list[str]:
    """按问题检索最相关的 top_k 个文档块。"""
    col = chroma.get_or_create_collection(COLLECTION)
    vectors = embed([query])
    res = col.query(query_embeddings=vectors, n_results=top_k)
    return res["documents"][0]