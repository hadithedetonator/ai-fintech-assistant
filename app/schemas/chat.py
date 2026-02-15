from pydantic import BaseModel, Field
from typing import List, Optional

class Transaction(BaseModel):
    id: str
    amount: float
    currency: str
    description: str
    date: str
    category: Optional[str] = None

class ChatRequest(BaseModel):
    msg: str = Field(..., description="User message")
    transactions: List[Transaction] = Field(default_factory=list)
    analyst_type: str = Field(default="advisor", description="Type of analyst: advisor, tax, coach")

class ChatResponse(BaseModel):
    answer: str
