from fastapi import FastAPI
from app.database import engine,Base
from app import models
from app.routers import documents
from app.routers import chat
from app.routers import documents, chat, ask

Base.metadata.create_all(bind=engine)
app=FastAPI(title="智能文档问答助手",version="0.1.0")
app.include_router(documents.router)
app.include_router(chat.router) 
app.include_router(ask.router) 
@app.get("/")
def read_root():
    return {"status":"ok","msg":"ok"}