from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agents.orchestrator import handle
from app.llm import http_error

router = APIRouter(prefix="/api/agent", tags=["multi-agent"]) 

class AgentRequest(BaseModel):
    question: str
    top_k: int = 3


@router.post("/")
def agent(req: AgentRequest):
    try:
        return handle(req.question, top_k=req.top_k)
    except HTTPException:
        raise                       # 翻译过的状态码
    except Exception as e:
        http_error(e)   