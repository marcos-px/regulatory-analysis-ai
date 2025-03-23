from pydantic import BaseModel
from typing import List, Dict, Any

class SimilarityResponse(BaseModel):
    similarity: float
    key_differences: List[str]
    raw_changes: Dict[str, Any]