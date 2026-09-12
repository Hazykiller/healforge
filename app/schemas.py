from pydantic import BaseModel, Field

class InspectRequest(BaseModel):
    pr_url: str = Field(min_length=10)

class AnalyzeRequest(BaseModel):
    session_id: str = Field(min_length=8)

class RepairRequest(BaseModel):
    session_id: str = Field(min_length=8)
    attempt: int = Field(default=1, ge=1, le=2)

class PublishRequest(BaseModel):
    session_id: str = Field(min_length=8)
    title: str = Field(min_length=3, max_length=120)
    body: str = Field(min_length=1, max_length=10000)
