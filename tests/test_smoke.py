"""启动冒烟：应用能不能起来、路由有没有挂上、路径有没有被 cwd 带跑。"""
import os
from pathlib import Path

from app.config import BASE_DIR, settings


def test_root_ok(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_offline_mode_built_in():
    """离线开关必须由代码内置，不能靠运维记得设环境变量。"""
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_paths_are_absolute():
    """数据库和向量库必须是绝对路径，否则换个目录启动就像数据丢了。

    注意断言的是默认值：测试环境里 DATABASE_URL 已被 conftest 指向临时库。
    """
    from app.config import Settings

    assert Path(settings.chroma_dir).is_absolute()
    assert settings.chroma_dir == str(BASE_DIR / "chroma_db")

    default_db = Settings.model_fields["database_url"].default
    assert ".docqa.db" in default_db
    assert not default_db.startswith("sqlite:///.docqa.db"), "默认值仍是相对路径"


def test_app_metadata(client):
    assert client.app.title == "智能文档问答助手"
    assert client.app.version == "0.1.0"


def test_routes_registered(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/api/documents/", "/api/documents/{doc_id}", "/api/chat/", "/api/chat/stream", "/api/ask/"):
        assert p in paths, f"路由缺失: {p}"


def test_openapi_schema_buildable(client):
    """OpenAPI 能生成，说明没有无法序列化的响应模型。"""
    assert client.get("/openapi.json").status_code == 200
