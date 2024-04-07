from sqlalchemy import Column, Integer, String, DateTime, func

from db.base import Base


class KnowledgeBaseModel(Base):
    """
    knowledge base
    """
    __tablename__ = 'knowledge_base'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='knowledge base ID')
    kb_name = Column(String(50), comment='knowledge base name')
    kb_info = Column(String(200), comment='knowledge base description')
    vs_type = Column(String(50), comment='vector type')
    embed_model = Column(String(50), comment='embedding model')
    file_count = Column(Integer, default=0, comment='files count')
    create_time = Column(DateTime, default=func.now(), comment='create time')

    def __repr__(self):
        return f"<KnowledgeBase(id='{self.id}', kb_name='{self.kb_name}',kb_intro='{self.kb_info} vs_type='{self.vs_type}', embed_model='{self.embed_model}', file_count='{self.file_count}', create_time='{self.create_time}')>"
