from pydantic import BaseModel
from typing import Dict

class Document(BaseModel):
    url: str
    title: str
    summary: str
    content: str
    metadata: Dict[str, str]
