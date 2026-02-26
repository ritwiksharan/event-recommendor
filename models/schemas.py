from datetime import date
from typing import Optional
from pydantic import BaseModel, Field


class UserRequest(BaseModel):
    city: str
    state_code: Optional[str] = None
    country_code: str = "US"
    start_date: date
    end_date: date
    event_description: str = Field(..., description="What kind of event the user wants")
    budget_max: Optional[float] = None


class EventResult(BaseModel):
    event_id: str
    event_name: str
    date: str
    time: str
    venue_name: str
    venue_address: str
    venue_city: str
    venue_state: str
    venue_latitude: float
    venue_longitude: float
    price_min: float
    price_max: float
    category: str
    genre: str
    url: str
    image_url: str
    is_weekend: bool
    is_outdoor: bool


class EventAgentOutput(BaseModel):
    request: UserRequest
    events: list[EventResult] = []
    total_found: int = 0
    error: Optional[str] = None


class DailyForecast(BaseModel):
    date: str
    temp_min_f: float
    temp_max_f: float
    description: str
    precipitation_chance: float
    wind_speed_mph: float
    is_suitable_outdoor: bool


class WeatherAgentOutput(BaseModel):
    city: str
    forecasts: dict[str, DailyForecast] = {}
    error: Optional[str] = None


class ScoredEvent(BaseModel):
    event: EventResult
    weather: Optional[DailyForecast] = None
    relevance_score: float
    score_reason: str


class RecommendationAgentOutput(BaseModel):
    request: UserRequest
    recommendations: list[ScoredEvent] = []


class QAMessage(BaseModel):
    role: str
    content: str


class QARequest(BaseModel):
    recommendations: RecommendationAgentOutput
    conversation_history: list[QAMessage] = []
    user_question: str


class QAResponse(BaseModel):
    answer: str
    updated_history: list[QAMessage]
