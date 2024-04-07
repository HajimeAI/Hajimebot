from functools import wraps
from contextlib import contextmanager
from db.base import SessionLocal
from sqlalchemy.orm import Session


@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def with_session(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        with session_scope() as session:
            try:
                result = f(session, *args, **kwargs)
                session.commit()
                return result
            except:
                session.rollback()
                raise

    return wrapper


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db0() -> Session:
    db = SessionLocal()
    return db
