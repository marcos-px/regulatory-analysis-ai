from pydantic import BaseModel

class RegulationComparison(BaseModel):
    text1: str
    text2: str