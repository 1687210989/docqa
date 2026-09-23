from fastapi import APIRouter
from pydantic import BaseModel
from openai import OpenAI
from app.config import settings
from fastapi.responses import StreamingResponse

router=APIRouter(prefix="/api/chat",tags=["chat"])

client=OpenAI(api_key=settings.openai_api_key,base_url=settings.openai_base_url)
class ChatRequest(BaseModel):
    message: str

@router.post("/")
def chat(req: ChatRequest):
    resp = client.chat.completions.create(
        model=settings.model_name,
        extra_body={"thinking": {"type": "disabled"}},
        messages=[{"role": "user", "content": req.message},],
    )
    return {"answer": resp.choices[0].message.content}

@router.post("/stream")
def chat_stream(req:ChatRequest):
    def event_generator():
        stream=client.chat.completions.create(
            model=settings.model_name,
            extra_body={"thinking": {"type": "disabled"}},
            messages=[{"role": "user", "content": req.message},],
            stream=True,
        )
        for chunk in stream:
            delta= chunk.choices[0].delta.content
            if delta:
                yield delta
    return StreamingResponse(event_generator(),media_type="text/plain")
