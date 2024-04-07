from datetime import datetime
from sqlalchemy import Column, DateTime, String, Integer


class BaseModel:
    """
    base model
    """
    id = Column(Integer, primary_key=True, index=True, comment="primary key")
    create_time = Column(DateTime, default=datetime.utcnow, comment="create time")
    update_time = Column(DateTime, default=None, onupdate=datetime.utcnow, comment="update time")
    create_by = Column(String, default=None, comment="creator")
    update_by = Column(String, default=None, comment="updater")
