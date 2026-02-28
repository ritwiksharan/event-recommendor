import json
import os
from typing import Optional
from litellm import completion
from config import CLAUDE_MODEL
from models.schemas import (
    UserRequest, EventResult, DailyForecast,
    EventAgentOutput, WeatherAgentOutput,
    ScoredEvent, RecommendationAgentOutput,
)


def _build_event_summary(event: EventResult, weather: Optional[DailyForecast]) -> str:
    price_str = (
        f"${event.price_min:.0f}-${event.price_max:.0f}"
        if event.price_min or event.price_max else "Free/Unknown"
    )
    weather_str = (
        f"{weather.description}, {weather.temp_min_f:.0f}-{weather.temp_max_f:.0f}F, "
        f"rain {weather.precipitation_chance:.0f}%, outdoor_ok={weather.is_suitable_outdoor}"
        if weather else "No forecast"
    )
    description_str = event.description.strip() if event.description else "No description available"
    return (
        f"ID: {event.event_id}\n"
        f"Name: {event.event_name}\n"
        f"Description: {description_str}\n"
        f"Date: {event.date} ({'Weekend' if event.is_weekend else 'Weekday'}) @ {event.time}\n"
        f"Venue: {event.venue_name} ({'Outdoor' if event.is_outdoor else 'Indoor'})\n"
        f"Category: {event.category} / {event.genre}\n"
        f"Price: {price_str}\n"
        f"Weather: {weather_str}"
    )


def run_recommendation_agent(
    request: UserRequest,
    events_out: EventAgentOutput,
    weather_out: WeatherAgentOutput,
    top_n: int = 6,
    anthropic_api_key: Optional[str] = None,
) -> RecommendationAgentOutput:
    """Agent 3 — use Claude LLM to score and rank events."""
    if not events_out.events:
        return RecommendationAgentOutput(request=request, recommendations=[])

    if anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    event_blocks = []
    weather_map: dict[str, Optional[DailyForecast]] = {}
    for event in events_out.events:
        weather = weather_out.forecasts.get(event.date)
        weather_map[event.event_id] = weather
        event_blocks.append(_build_event_summary(event, weather))

    events_text = "\n\n---\n\n".join(event_blocks)
    budget_str  = f"${request.budget_max}" if request.budget_max else "No limit"

    system_msg = (
        "You are an expert event recommendation engine. "
        "Your primary job is to semantically match what the user is looking for against each event's name and description. "
        "Score each event 0-100 using this priority order:\n"
        "1. SEMANTIC MATCH (most important): Does the event name/description align with what the user asked for? "
        "Read the description carefully — an event called 'Jazz Night' with a description about a rock band should score low for a jazz request.\n"
        "2. PRACTICAL FIT: Does the price fit the budget? Is the venue type (indoor/outdoor) appropriate given the weather?\n"
        "3. TIMING: Weekend events score slightly higher for leisure requests.\n"
        "Give a 'reason' that explains specifically how the event description matches or mismatches the user's request. "
        "Respond with ONLY a valid JSON array. No prose, no markdown, no code fences."
    )
    user_msg = (
        f"User is looking for: \"{request.event_description}\"\n"
        f"Budget max: {budget_str}\n"
        f"Date range: {request.start_date} to {request.end_date}\n\n"
        f"Score each of the following {len(events_out.events)} events based on how well they match what the user described. "
        f"Pay close attention to the Description field of each event.\n\n"
        f"{events_text}\n\n"
        f"Respond with ONLY this JSON array:\n"
        f'[{{"event_id": "...", "score": <0-100>, "reason": "one sentence explaining the semantic match"}}, ...]'
    )

    try:
        response = completion(
            model       = CLAUDE_MODEL,
            messages    = [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature = 0.2,
            max_tokens  = 2000,
        )
        raw_json    = response.choices[0].message.content.strip()
        raw_json    = raw_json.strip("```json").strip("```").strip()
        scores_list = json.loads(raw_json)

    except Exception as exc:
        print(f"[Recommendation Agent] LLM error: {exc}. Returning unscored events.")
        scored = [
            ScoredEvent(
                event           = e,
                weather         = weather_map.get(e.event_id),
                relevance_score = 50.0,
                score_reason    = "LLM unavailable",
            )
            for e in events_out.events
        ]
        return RecommendationAgentOutput(request=request, recommendations=scored[:top_n])

    event_lookup = {e.event_id: e for e in events_out.events}
    scores_map   = {item["event_id"]: item for item in scores_list}

    scored = []
    for event_id, event in event_lookup.items():
        score_data = scores_map.get(event_id, {"score": 0, "reason": "Not scored by LLM"})
        scored.append(ScoredEvent(
            event           = event,
            weather         = weather_map.get(event_id),
            relevance_score = float(score_data.get("score", 0)),
            score_reason    = score_data.get("reason", ""),
        ))

    scored.sort(key=lambda x: x.relevance_score, reverse=True)
    return RecommendationAgentOutput(request=request, recommendations=scored[:top_n])
