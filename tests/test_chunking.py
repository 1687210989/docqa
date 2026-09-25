"""切块逻辑纯函数测试，不涉及数据库与模型。"""
from app.rag.chunking import split_text


def test_empty_text():
    assert split_text("") == []
    assert split_text("   ") == []


def test_short_text_single_chunk():
    assert split_text("很短的一句话") == ["很短的一句话"]


def test_long_text_multiple_chunks():
    text = "中文内容" * 500          # 2000 字
    chunks = split_text(text)
    assert len(chunks) > 1
    assert all(chunks)                # 不应出现空块


def test_chunks_have_overlap():
    """相邻块要有重叠，否则跨块的语义会被切断。"""
    text = "".join(str(i % 10) for i in range(1200))
    chunks = split_text(text, chunk_size=400, overlap=80)
    assert len(chunks) >= 3
    a, b = chunks[0], chunks[1]
    assert a[-80:] == b[:80], "相邻块之间没有重叠"


def test_whitespace_only_chunk_dropped():
    text = "A" * 400 + "     " + "B" * 400
    chunks = split_text(text)
    assert "" not in chunks
