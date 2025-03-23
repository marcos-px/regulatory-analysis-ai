from pydantic import BaseModel
from typing import Optional

class Regulation(BaseModel):
    id: str
    date: str
    text: str
    previous_id: Optional[str] = None