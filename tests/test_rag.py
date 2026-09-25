"""向量检索链路：嵌入 → 索引 → 检索 → 删除。"""
import pytest

from app.rag import pipeline
from app.rag.chunking import split_text


def test_embed_shape_and_normalized():
    vecs = pipeline.embed(["第一段文本", "第二段文本"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 512, f"bge-small-zh-v1.5 应为 512 维，实得 {len(vecs[0])}"
    norm = sum(x * x for x in vecs[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-3, "向量未归一化"


def test_embed_semantic_order():
    """语义相近的句子，余弦相似度应高于无关句子。"""
    a, b, c = pipeline.embed(["公司报销流程", "差旅费报销规定", "今天天气不错"])
    sim = lambda x, y: sum(i * j for i, j in zip(x, y))
    assert sim(a, b) > sim(a, c)


def test_index_and_retrieve():
    content = "密码找回流程：在登录页点击忘记密码，输入注册邮箱，查收重置链接。"
    pipeline.index_document(1, split_text(content))
    hits = pipeline.retrieve("忘记密码怎么办", top_k=2)
    assert hits, "检索结果为空"
    assert "密码" in "".join(hits)


def test_delete_only_target_document():
    pipeline.index_document(1, split_text("文档一的内容"))
    pipeline.index_document(2, split_text("文档二的内容"))
    pipeline.delete_document(1)

    col = pipeline.chroma.get_or_create_collection(pipeline.COLLECTION)
    left = {m["doc_id"] for m in col.get()["metadatas"]}
    assert left == {2}, f"删除 doc_id=1 后残留: {left}"


def test_delete_nonexistent_document_is_noop():
    pipeline.index_document(3, split_text("内容"))
    pipeline.delete_document(9999)                     # 不应抛异常
    col = pipeline.chroma.get_or_create_collection(pipeline.COLLECTION)
    assert {m["doc_id"] for m in col.get()["metadatas"]} == {3}


def test_retrieve_on_empty_collection():
    """空库检索必须返回空列表而不是抛异常，否则 /ask 在没文档时会直接 500。"""
    assert pipeline.retrieve("任意问题") == []
