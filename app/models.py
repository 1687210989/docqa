from sqlalchemy import Column, Integer, String,Text,DateTime
from datetime import datetime
from app.database import Base
from datetime import timezone

def utc_now():
    return datetime.now(timezone.utc)

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True) 
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)
