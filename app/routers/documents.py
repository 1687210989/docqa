from fastapi import APIRouter,Depends,HTTPException
from sqlalchemy.orm import Session
from app import crud,schemas,models
from app.database import get_db
from app.rag.chunking import split_text
from app.rag.pipeline import index_document,delete_document

router=APIRouter(prefix="/api/documents",tags=["documents"])

@router.post("/",response_model=schemas.DocumentOut,status_code=201)
def create_document(payload: schemas.DocumentCreate,db: Session = Depends(get_db)):
    doc = crud.create_document(db, payload, commit=False)   # 先拿 id，先不落盘
    try:
        chunks = split_text(payload.content)    #  切块
        index_document(doc.id, chunks)          # 建索引
        db.commit()                             # 两边都成功才真正入库
    except Exception:
        db.rollback()                           # 回滚数据库，不留孤儿记录
        delete_document(doc.id)                 # 清掉可能已写进去的向量
        raise
    db.refresh(doc)
    return doc

@router.get("/",response_model=list[schemas.DocumentOut])
def get_documents(skip: int = 0, limit: int = 100,db: Session = Depends(get_db)):
    return crud.get_document(db,skip,limit)

@router.get("/{doc_id}",response_model=schemas.DocumentOut)
def get_doc(doc_id: int,db: Session = Depends(get_db)):
    doc=crud.get_document_by_id(db,doc_id)
    if not doc:
        raise HTTPException(status_code=404,detail="文档不存在")
    return doc

@router.delete("/{doc_id}",status_code=204)
def remove_doc(doc_id: int,db: Session = Depends(get_db)):
    if not crud.get_document_by_id(db,doc_id):
        raise HTTPException(status_code=404,detail="文档不存在")
    delete_document(doc_id)         # 先清向量：宁可搜不到，也不能搜到已删的内容
    crud.delete_document(db,doc_id) # 再删库记录
    return {"deleted":doc_id}
    