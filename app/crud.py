from sqlalchemy.orm import Session
from app import models,schemas

def create_document(db: Session, payload: schemas.DocumentCreate) -> models.Document:
    doc=models.Document(title=payload.title, content=payload.content)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def get_document(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Document).offset(skip).limit(limit).all()


def get_document_by_id(db: Session, doc_id: int) :
    return db.query(models.Document).filter(models.Document.id == doc_id).first()

def delete_document(db: Session, doc_id: int) -> bool:
    doc = get_document_by_id(db, doc_id)
    if not doc:
        return False      
    db.delete(doc)
    db.commit()
    return True 