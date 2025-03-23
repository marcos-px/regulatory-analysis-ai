from pydantic import BaseModel

class PredictionRequest(BaseModel):
    num_predictions: int = 1
