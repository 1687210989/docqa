from fastapi import APIRouter,Depends,HTTPException
from sqlalchemy.orm import Session
from app import crud,schemas,models
from app.database import get_db

router=APIRouter(prefix="/api/documents",tags=["documents"])

@router.post("/",response_model=schemas.DocumentOut,status_code=201)
def create_document(payload: schemas.DocumentCreate,db: Session = Depends(get_db)):
    return crud.create_document(db,payload)

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
    if not crud.delete_document(db,doc_id):
        raise HTTPException(status_code=404,detail="文档不存在")
    return {"deleted":doc_id}
    