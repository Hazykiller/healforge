from pydantic import BaseModel, Field


class InspectRequest(BaseModel):
    pr_url: str = Field(min_length=10, max_length=500)
    base_sha: str | None = Field(default=None, max_length=100)


class AnalyzeRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=100)


class RepairRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=100)
    attempt: int = Field(default=1, ge=1, le=3)


class PublishRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=100)
    title: str = Field(min_length=3, max_length=120)
    body: str = Field(min_length=1, max_length=10000)
