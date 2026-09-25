"""大模型调用的统一入口：惰性建 client + 统一关思考 + 超时 + 异常翻译。"""
from fastapi import HTTPException
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from app.config import settings

# 【P0 修复】不在模块级建 client：key 为空时 openai 会直接抛错，
# 那样 import 路由就失败，整个应用（包括文档接口）都起不来。
client = None


def get_client() -> OpenAI:
    global client
    if client is None:
        settings.require_llm_config()          # 没配 key 时给明确提示
        client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout,      # 不加超时，请求会一直挂住打满线程池
        )
    return client


def chat_complete(messages: list[dict], stream: bool = False):
    return get_client().chat.completions.create(
        model=settings.model_name,
        extra_body={"thinking": {"type": "disabled"}},   # 关思考，省 token 也避免混进思维链
        messages=messages,
        stream=stream,
    )


def http_error(e: Exception) -> None:
    """把上游异常翻译成对调用方有意义的 HTTP 状态码。"""
    if isinstance(e, RuntimeError):                      # 未配置 API key
        raise HTTPException(status_code=503, detail=str(e))
    if isinstance(e, APITimeoutError):
        raise HTTPException(status_code=504, detail=f"大模型响应超时（>{settings.llm_timeout}s）")
    if isinstance(e, APIConnectionError):
        raise HTTPException(status_code=502, detail="无法连接大模型服务")
    if isinstance(e, APIStatusError):                    # 401 / 429 / 上游 5xx
        raise HTTPException(status_code=502, detail=f"大模型返回错误: {e.status_code}")
    raise HTTPException(status_code=500, detail=f"调用大模型失败: {e}")
