import concurrent.futures
from fastapi import APIRouter
from models.schemas import UserRequest, RecommendationAgentOutput
from agents.events_agent import run_events_agent
from agents.weather_agent import run_weather_agent
from agents.recommendation_agent import run_recommendation_agent

router = APIRouter()


@router.post("/recommend", response_model=RecommendationAgentOutput)
def recommend(request: UserRequest, top_n: int = 6) -> RecommendationAgentOutput:
    """Run agents 1-3: fetch events + weather in parallel, then score & rank."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_events  = pool.submit(run_events_agent,  request)
        f_weather = pool.submit(run_weather_agent, request)
        events_out  = f_events.result()
        weather_out = f_weather.result()

    if events_out.error:
        return RecommendationAgentOutput(request=request, recommendations=[])

    return run_recommendation_agent(request, events_out, weather_out, top_n=top_n)
