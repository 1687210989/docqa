"""测试前置：必须在导入 app 之前搭好隔离环境。

原因：app.config 在导入那一刻就把 settings 实例化了，
app.rag.pipeline 也在导入时就建好了 chroma 客户端（路径写死 "./chroma_db"）。
所以切换数据库、切换工作目录的动作必须发生在本文件顶部，早于任何 app 导入。
"""
import os
import pathlib
import shutil
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 固定路径 + 每次重建：既方便排查，又不会像 mkdtemp 那样每跑一次留一个垃圾目录
_TMP = pathlib.Path(tempfile.gettempdir()) / "docqa_test"
shutil.rmtree(_TMP, ignore_errors=True)
_TMP.mkdir(parents=True, exist_ok=True)

os.environ["HF_HUB_OFFLINE"] = "1"                          # 只用本地缓存模型，避免联网卡死
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"  # 优先级高于 .env，指向临时库
os.chdir(_TMP)                                              # 让 "./chroma_db" 落在临时目录

import pytest
from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.main import app
from app.models import Document
from app.rag import pipeline

TEST_TMP = _TMP


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


def _reset_db():
    db = SessionLocal()
    try:
        db.query(Document).delete()
        db.commit()
    finally:
        db.close()


def _reset_vectors():
    col = pipeline.chroma.get_or_create_collection(pipeline.COLLECTION)
    ids = col.get()["ids"]
    if ids:
        col.delete(ids=ids)


@pytest.fixture(autouse=True)
def clean_state():
    """每个用例跑完清空文档表与向量库，用例之间互不干扰。"""
    yield
    _reset_db()
    _reset_vectors()


@pytest.fixture(scope="session")
def vector_count():
    """返回当前向量库里属于某 doc_id 的切片数量。"""

    def _count(doc_id=None):
        col = pipeline.chroma.get_or_create_collection(pipeline.COLLECTION)
        got = col.get(where={"doc_id": doc_id} if doc_id is not None else None)
        return len(got["ids"])

    return _count


@pytest.fixture(scope="session", autouse=True)
def warmup_model():
    """提前把本地嵌入模型加载好，避免第一个用例的耗时被算进 RAG 用例里。"""
    pipeline.embed(["预热"])
