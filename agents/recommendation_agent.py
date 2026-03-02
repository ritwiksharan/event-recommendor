import json
import os
import re
from typing import Optional
from litellm import completion
from config import CLAUDE_MODEL

LLM_EVENT_LIMIT = 50  # max events sent to the LLM at once
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

    # Cap the number of events sent to the LLM to avoid token limit issues
    candidate_events = events_out.events[:LLM_EVENT_LIMIT]

    event_blocks = []
    weather_map: dict[str, Optional[DailyForecast]] = {}
    for event in candidate_events:
        weather = weather_out.forecasts.get(event.date)
        weather_map[event.event_id] = weather
        event_blocks.append(_build_event_summary(event, weather))

    events_text = "\n\n---\n\n".join(event_blocks)
    budget_str  = f"${request.budget_max}" if request.budget_max else "No limit"

    system_msg = (
        "You are an expert event recommendation engine for EventScout. "
        "Your job is to score how well each event matches what the user is looking for.\n\n"

        "WHAT YOU SCORE:\n"
        "- Semantic match between user's description and event name/description\n"
        "- Price fit within the user's budget\n"
        "- Venue type match (indoor/outdoor preference)\n"
        "- Weather suitability for outdoor events\n"
        "- Timing preference (weekday vs weekend)\n\n"

        "SCORING EXAMPLES:\n\n"

        "EXAMPLE 1 — Perfect match:\n"
        "User wants: 'jazz music indoor weekend'\n"
        "Event: 'Birdland Jazz Club - Friday Night Jazz' | Category: Music | Genre: Jazz | Indoor | Friday\n"
        "Score: 92 | Reason: Jazz genre matches exactly, indoor venue as requested, weekend date.\n\n"

        "EXAMPLE 2 — Wrong category:\n"
        "User wants: 'jazz music indoor weekend'\n"
        "Event: 'Yankees vs Red Sox' | Category: Sports | Genre: N/A | Outdoor | Saturday\n"
        "Score: 8 | Reason: Sports event with outdoor stadium, completely unrelated to jazz music request.\n\n"

        "EXAMPLE 3 — Partial match:\n"
        "User wants: 'jazz music indoor weekend'\n"
        "Event: 'Classical Piano Recital' | Category: Music | Genre: Classical | Indoor | Saturday\n"
        "Score: 45 | Reason: Music category and indoor venue match, but classical genre differs from jazz.\n\n"

        "EXAMPLE 4 — Budget mismatch:\n"
        "User wants: 'live music', Budget: $30\n"
        "Event: 'Coldplay World Tour' | Category: Music | Genre: Rock | Indoor | Saturday | Price: $150-$300\n"
        "Score: 20 | Reason: Music genre matches but price far exceeds the $30 budget.\n\n"

        "EXAMPLE 5 — Weather penalty:\n"
        "User wants: 'outdoor festival'\n"
        "Event: 'Summer Music Festival' | Category: Music | Outdoor | Saturday | Weather: Heavy rain, outdoor_ok=False\n"
        "Score: 30 | Reason: Outdoor festival matches request but heavy rain makes attendance unsuitable.\n\n"

        "ESCAPE HATCH: If event data is missing or ambiguous, score 50 and state 'Insufficient data to score accurately'.\n\n"

        "Respond with ONLY a valid JSON array. No prose, no markdown, no code fences."
    )
    user_msg = (
        f"User is looking for: \"{request.event_description}\"\n"
        f"Budget max: {budget_str}\n"
        f"Date range: {request.start_date} to {request.end_date}\n\n"
        f"Score each of the following {len(candidate_events)} events based on how well they match what the user described. "
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
            max_tokens  = 8000,
        )
        raw_json    = response.choices[0].message.content.strip()
        raw_json    = re.sub(r"^```(?:json)?\s*", "", raw_json)
        raw_json    = re.sub(r"\s*```$", "", raw_json).strip()
        scores_list = json.loads(raw_json)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[Recommendation Agent] LLM error: {exc}")
        scored = [
            ScoredEvent(
                event           = e,
                weather         = weather_map.get(e.event_id),
                relevance_score = 50.0,
                score_reason    = f"Scoring error: {exc}",
            )
            for e in candidate_events
        ]
        return RecommendationAgentOutput(request=request, recommendations=scored[:top_n])

    event_lookup = {e.event_id: e for e in candidate_events}
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
