from fastapi import APIRouter
from models.schemas import QARequest, QAResponse
from agents.qa_agent import run_qa_agent

router = APIRouter()


@router.post("/qa", response_model=QAResponse)
def qa(request: QARequest) -> QAResponse:
    """Run agent 4: answer a user question about their recommendations."""
    return run_qa_agent(request)
