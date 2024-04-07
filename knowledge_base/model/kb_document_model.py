
from langchain.docstore.document import Document


class DocumentWithVSId(Document):
    """
    Documnet after vectorize
    """
    id: str = None
    score: float = 3.0
