from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, JSON, func

from db.base import Base


class SummaryChunkModel(Base):
    __tablename__ = 'summary_chunk'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='ID')
    kb_name = Column(String(50), comment='')
    summary_context = Column(String(255), comment='')
    summary_id = Column(String(255), comment='')
    doc_ids = Column(String(1024), comment="")
    meta_data = Column(JSON, default={})

    def __repr__(self):
        return (f"<SummaryChunk(id='{self.id}', kb_name='{self.kb_name}', summary_context='{self.summary_context}',"
                f" doc_ids='{self.doc_ids}', metadata='{self.metadata}')>")
