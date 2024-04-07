from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, JSON, func

from db.base import Base


class KnowledgeFileModel(Base):
    __tablename__ = 'knowledge_file'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='')
    file_name = Column(String(255), comment='')
    file_ext = Column(String(10), comment='')
    kb_name = Column(String(50), comment='')
    document_loader_name = Column(String(50), comment='')
    text_splitter_name = Column(String(50), comment='')
    file_version = Column(Integer, default=1, comment='')
    file_mtime = Column(Float, default=0.0, comment="")
    file_size = Column(Integer, default=0, comment="")
    custom_docs = Column(Boolean, default=False, comment="")
    docs_count = Column(Integer, default=0, comment="")
    create_time = Column(DateTime, default=func.now(), comment='')

    def __repr__(self):
        return f"<KnowledgeFile(id='{self.id}', file_name='{self.file_name}', file_ext='{self.file_ext}', kb_name='{self.kb_name}', document_loader_name='{self.document_loader_name}', text_splitter_name='{self.text_splitter_name}', file_version='{self.file_version}', create_time='{self.create_time}')>"


class FileDocModel(Base):
    __tablename__ = 'file_doc'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='')
    kb_name = Column(String(50), comment='')
    file_name = Column(String(255), comment='')
    doc_id = Column(String(50), comment="")
    meta_data = Column(JSON, default={})

    def __repr__(self):
        return f"<FileDoc(id='{self.id}', kb_name='{self.kb_name}', file_name='{self.file_name}', doc_id='{self.doc_id}', metadata='{self.meta_data}')>"
