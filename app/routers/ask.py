from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.llm import chat_complete, http_error
from app.rag.pipeline import retrieve

router = APIRouter(prefix="/api/ask", tags=["rag"])   # 原为 "/ask"，与其它 /api/* 前缀不一致


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


PROMPT_TEMPLATE = (
    "请只根据下面【资料】回答问题。如果资料里没有答案，就如实说“资料中未提及”。\n\n"
    "【资料】\n{context}\n\n"
    "【问题】{question}"
)


@router.post("/")
def ask(req: AskRequest):
    try:
        sources = retrieve(req.question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {e}")

    context = "\n\n".join(sources)
    prompt = PROMPT_TEMPLATE.format(context=context, question=req.question)

    try:
        resp = chat_complete([{"role": "user", "content": prompt}])   # 内部已关思考 + 带超时
    except HTTPException:
        raise
    except Exception as e:
        http_error(e)

    return {
        "answer": resp.choices[0].message.content,
        "sources": sources,          # 把引用来源一并返回，方便前端展示
    }
