from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from app.llm import chat_complete, http_error

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str


@router.post("/")
def chat(req: ChatRequest):
    try:
        resp = chat_complete([{"role": "user", "content": req.message}])
    except HTTPException:
        raise
    except Exception as e:
        http_error(e)
    return {"answer": resp.choices[0].message.content}


@router.post("/stream")
def chat_stream(req: ChatRequest):
    def event_generator():
        try:
            stream = chat_complete([{"role": "user", "content": req.message}], stream=True)
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            # 响应头已经发出去了，不能再抛 HTTPException，只能把错误写进流里
            yield f"\n[出错了] {e}"

    return StreamingResponse(event_generator(), media_type="text/plain")
