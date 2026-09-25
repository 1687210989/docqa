"""LLM 接口测试：默认用假客户端，不花钱、不依赖网络。

想真连 DeepSeek 跑一次，加环境变量：RUN_LIVE_LLM=1
"""
import os

import pytest

from app import llm as llm_mod
from app.config import settings

LIVE = os.getenv("RUN_LIVE_LLM") == "1"


class _Delta:
    def __init__(self, t):
        self.content = t


class _Message:
    def __init__(self, t):
        self.content = t


class _Choice:
    def __init__(self, t):
        self.message = _Message(t)
        self.delta = _Delta(t)


class _Completion:
    def __init__(self, t):
        self.choices = [_Choice(t)]


class _Completions:
    def __init__(self, sink):
        self._sink = sink

    def create(self, **kw):
        self._sink.append(kw)
        if kw.get("stream"):
            return iter([_Completion("你好"), _Completion("，这是流式")])
        return _Completion("MOCK_ANSWER")


class _Chat:
    def __init__(self, sink):
        self.completions = _Completions(sink)


class FakeClient:
    def __init__(self):
        self.calls = []
        self.chat = _Chat(self.calls)


@pytest.fixture()
def fake_client(monkeypatch):
    """chat 与 ask 现在共用 app.llm.client，换一次就够。"""
    c = FakeClient()
    monkeypatch.setattr(llm_mod, "client", c)
    return c


def test_chat_ok(client, fake_client):
    r = client.post("/api/chat/", json={"message": "你好"})
    assert r.status_code == 200, r.text
    assert r.json() == {"answer": "MOCK_ANSWER"}


def test_chat_disables_thinking(client, fake_client):
    """关思考的参数必须真的发出去，否则会返回一大段思维链、白烧 token。"""
    client.post("/api/chat/", json={"message": "你好"})
    assert fake_client.calls[-1]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_chat_uses_configured_model(client, fake_client, monkeypatch):
    monkeypatch.setattr(settings, "model_name", "deepseek-flash")
    client.post("/api/chat/", json={"message": "你好"})
    assert fake_client.calls[-1]["model"] == "deepseek-flash"


def test_client_built_with_timeout(monkeypatch):
    """超时必须真的传给 SDK，否则一个挂死的上游请求就能拖垮服务。"""
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(llm_mod, "client", None)
    monkeypatch.setattr(llm_mod, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(settings, "llm_timeout", 12.5)
    llm_mod.get_client()
    assert captured["timeout"] == 12.5


def test_missing_api_key_returns_503(client, monkeypatch):
    """key 没配时不能让整个应用起不来，只在真正调用时才报 503。"""
    monkeypatch.setattr(llm_mod, "client", None)
    monkeypatch.setattr(settings, "openai_api_key", "")
    r = client.post("/api/chat/", json={"message": "你好"})
    assert r.status_code == 503
    assert "OPENAI_API_KEY" in r.json()["detail"]


def test_upstream_error_translated(client, monkeypatch):
    """上游 401/429 不能被吞成 500，应转成 502 并把上游状态码带出来。"""
    import httpx
    from openai import APIStatusError

    http_resp = httpx.Response(
        401, request=httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
    )

    class Boom:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise APIStatusError("上游炸了", response=http_resp, body=None)

    monkeypatch.setattr(llm_mod, "client", Boom())
    r = client.post("/api/chat/", json={"message": "你好"})
    assert r.status_code == 502
    assert "401" in r.json()["detail"]


def test_chat_stream(client, fake_client):
    r = client.post("/api/chat/stream", json={"message": "你好"})
    assert r.status_code == 200
    assert r.text == "你好，这是流式"


def test_chat_missing_field_422(client, fake_client):
    assert client.post("/api/chat/", json={}).status_code == 422


def test_ask_returns_answer_and_sources(client, fake_client, vector_count):
    doc = client.post(
        "/api/documents/",
        json={"title": "考勤", "content": "公司规定上班时间为早上九点，迟到超过三十分钟按旷工半天处理。"},
    ).json()
    assert vector_count(doc["id"]) >= 1

    r = client.post("/api/ask/", json={"question": "几点上班"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "MOCK_ANSWER"
    assert body["sources"], "sources 不应为空"
    assert "九点" in "".join(body["sources"])


def test_ask_prompt_contains_context(client, fake_client):
    client.post("/api/documents/", json={"title": "t", "content": "唯一的知识点是ABCXYZ。"})
    client.post("/api/ask/", json={"question": "知识点是什么"})
    prompt = fake_client.calls[-1]["messages"][0]["content"]
    assert "ABCXYZ" in prompt
    assert "资料中未提及" in prompt


def test_ask_disables_thinking(client, fake_client):
    """/api/ask 以前没关思考，与 /api/chat 行为不一致。"""
    client.post("/api/ask/", json={"question": "随便"})
    assert fake_client.calls[-1]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_ask_on_empty_library(client, fake_client):
    """库里一篇文档都没有时，/api/ask 不应 500，应正常走“资料中未提及”。"""
    r = client.post("/api/ask/", json={"question": "公司几点上班"})
    assert r.status_code == 200, r.text
    assert r.json()["sources"] == []


@pytest.mark.live
@pytest.mark.skipif(not LIVE, reason="设置 RUN_LIVE_LLM=1 才会真实调用大模型")
def test_live_chat(client):
    r = client.post("/api/chat/", json={"message": "用一句话介绍你自己"})
    assert r.status_code == 200, r.text
    assert r.json()["answer"]


@pytest.mark.live
@pytest.mark.skipif(not LIVE, reason="设置 RUN_LIVE_LLM=1 才会真实调用大模型")
def test_live_ask(client):
    client.post(
        "/api/documents/",
        json={"title": "休假", "content": "正式员工每年享有十天年假，入职满一年后开始计算。"},
    )
    r = client.post("/api/ask/", json={"question": "年假有几天"})
    assert r.status_code == 200, r.text
    assert "十天" in r.json()["answer"]
