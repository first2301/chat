from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    mode: str = "lcel"  # 'manual' | 'lcel'

