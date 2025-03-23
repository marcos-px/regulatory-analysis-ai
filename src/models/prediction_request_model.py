from pydantic import BaseModel
from typing import Optional

class PredictionRequest(BaseModel):
    num_predictions: int = 1
    text: Optional[str] = None