"""第四阶段多 Agent 编排测试：全程假客户端，按脚本应答，不花钱、不联网。

想真连 DeepSeek 跑一次：RUN_LIVE_LLM=1 pytest -q -m live
"""
import os

import pytest

from app import llm as llm_mod


class _Msg:
    def __init__(self, t): self.content = t


class _Choice:
    def __init__(self, t): self.message = _Msg(t); self.delta = _Msg(t)


class _Completion:
    def __init__(self, t): self.choices = [_Choice(t)]


class _Completions:
    def __init__(self, owner): self.owner = owner

    def create(self, **kw):
        self.owner.calls.append(kw)
        return _Completion(self.owner.replies.pop(0))


class _Chat:
    def __init__(self, owner): self.completions = _Completions(owner)


class ScriptedClient:
    """按脚本依次应答，模拟 Router → Answerer → Critic 的多次调用。"""
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = _Chat(self)


@pytest.fixture()
def scripted(monkeypatch):
    def _make(*replies):
        c = ScriptedClient(*replies)
        monkeypatch.setattr(llm_mod, "client", c)   # 换成假 client，get_client() 直接复用
        return c
    return _make


def test_agent_route_registered(client):
    """/api/agent/ 必须出现在路由表里。

    注意不能直接用 r.path：新版 FastAPI 的 app.routes 里混有 _IncludedRouter 等非 APIRoute 对象。
    """
    paths = client.app.openapi()["paths"]      # 比遍历 app.routes 可靠：子路由能展开
    assert "/api/agent/" in paths


def test_rag_path_returns_sources(client, scripted):
    """rag 意图：Router→Retriever→Answerer→Critic，返回来源且恰好 3 次调用。"""
    client.post("/api/documents/", json={"title": "考勤", "content": "公司上班时间是早上九点。"})
    c = scripted("rag", "九点上班", "pass")
    r = client.post("/api/agent/", json={"question": "几点上班"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["intent"] == "rag"
    assert body["sources"], "sources 不应为空"
    assert "九点" in "".join(body["sources"])
    assert len(c.calls) == 3                          # Router + Answerer + Critic


def test_chat_path_skips_retrieve(client, scripted):
    """chat 意图：不检索，只有 Router + Answerer 两次调用，无 sources。"""
    c = scripted("chat", "你好呀")
    r = client.post("/api/agent/", json={"question": "讲个笑话"})
    assert r.status_code == 200, r.text
    assert r.json()["intent"] == "chat"
    assert r.json()["sources"] == []
    assert len(c.calls) == 2


def test_critic_reject_triggers_retry(client, scripted):
    """Critic 判 fail 时应打回重答一次，共 4 次调用。"""
    client.post("/api/documents/", json={"title": "休假", "content": "正式员工每年十天年假。"})
    c = scripted("rag", "第一版答案", "fail", "第二版更严谨")
    r = client.post("/api/agent/", json={"question": "年假几天"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "第二版更严谨"
    assert body["critic_passed"] is False
    assert len(c.calls) == 4


def test_critic_pass_flag_true(client, scripted):
    """Critic 判 pass 时 critic_passed 为 True，不重答。"""
    client.post("/api/documents/", json={"title": "休假", "content": "正式员工每年十天年假。"})
    scripted("rag", "十天", "pass")
    r = client.post("/api/agent/", json={"question": "年假几天"})
    assert r.json()["critic_passed"] is True


def test_router_dirty_output_falls_back(client, scripted):
    """模型多说话时兜底成 unknown，不能崩。"""
    scripted("这应该算 RAG 吧（仅供参考）", "闲聊答案")
    r = client.post("/api/agent/", json={"question": "讲个笑话"})
    assert r.status_code == 200, r.text
    assert r.json()["intent"] == "unknown"
    assert len(r.json()["sources"]) == 0


def test_agent_on_empty_library(client, scripted):
    """库里没文档时 rag 路径也不能 500，sources 为空列表。"""
    scripted("rag", "资料中未提及", "pass")
    r = client.post("/api/agent/", json={"question": "几点上班"})
    assert r.status_code == 200, r.text
    assert r.json()["sources"] == []


def test_agent_top_k_respected(client, scripted):
    """top_k 参数要传到检索层。"""
    client.post("/api/documents/", json={"title": "考勤", "content": "上班九点，下班六点，午休一小时。"})
    scripted("rag", "九点", "pass")
    r = client.post("/api/agent/", json={"question": "上班时间", "top_k": 1})
    assert r.status_code == 200, r.text
    assert len(r.json()["sources"]) <= 1


def test_agent_disables_thinking(client, scripted):
    """Agent 的每次调用都必须关思考，否则返回思维链、白烧 token。"""
    c = scripted("chat", "答案")
    client.post("/api/agent/", json={"question": "讲个笑话"})
    assert c.calls[-1]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_agent_missing_field_422(client, scripted):
    scripted()
    assert client.post("/api/agent/", json={}).status_code == 422


def test_agent_upstream_error_translated(client, monkeypatch):
    """上游 401 不能被吞成 500，应转成 502 并带出上游状态码。"""
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
    r = client.post("/api/agent/", json={"question": "你好"})
    assert r.status_code == 502
    assert "401" in r.json()["detail"]


def test_agent_missing_api_key_returns_503(client, monkeypatch):
    """没配 key 时只在真正调用时才报 503。"""
    monkeypatch.setattr(llm_mod, "client", None)
    from app.config import settings
    monkeypatch.setattr(settings, "openai_api_key", "")
    r = client.post("/api/agent/", json={"question": "你好"})
    assert r.status_code == 503
    assert "OPENAI_API_KEY" in r.json()["detail"]


@pytest.mark.live
@pytest.mark.skipif(os.getenv("RUN_LIVE_LLM") != "1", reason="设置 RUN_LIVE_LLM=1 才会真实调用大模型")
def test_live_agent(client):
    client.post("/api/documents/", json={"title": "休假", "content": "正式员工每年十天年假。"})
    r = client.post("/api/agent/", json={"question": "年假有几天"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["intent"] in ("rag", "chat", "unknown")
    assert body["answer"]
