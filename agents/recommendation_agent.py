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

def _format_markdown(scored: list[ScoredEvent]) -> str:
    """Generate a markdown summary of recommendations."""
    lines = ["# Recommendations", ""]
    for s in scored:
        ev = s.event
        lines.append(f"- **{ev.event_name}** ({ev.date})")
        lines.append(f"  - Venue: {ev.venue_name} ({'Outdoor' if ev.is_outdoor else 'Indoor'})")
        lines.append(f"  - Score: {s.relevance_score:.1f}")
        lines.append(f"  - Reason: {s.score_reason}")
        lines.append("")
    return "\n".join(lines)
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


def _parse_scores_json(raw: str) -> list[dict]:
    """Robustly parse the LLM's JSON array, handling common Gemini quirks."""
    # Strip markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw).strip()

    # Extract the JSON array — find first '[' and last ']'
    start = raw.find("[")
    end   = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]
    elif start != -1:
        # Truncated response — close off any open object and the array
        raw = raw[start:]
        raw = re.sub(r",\s*\{[^}]*$", "", raw)  # drop incomplete last object
        raw = raw.rstrip(", \n") + "]"

    # Remove trailing commas before ] or }  (common Gemini output)
    raw = re.sub(r",\s*([}\]])", r"\1", raw)

    return json.loads(raw)


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

        "SCORING FACTORS (in order of importance):\n"
        "1. Semantic match between user's request and event name/description/genre — this is by far the most important factor\n"
        "2. Price fit within budget\n"
        "3. Timing (weekday vs weekend)\n"
        "4. Venue type (indoor/outdoor) — minor factor only, max -5 points even if user prefers outdoor\n\n"

        "VENUE/WEATHER GUIDANCE:\n"
        "- Venue type (indoor vs outdoor) is a MINOR preference, not a requirement. "
        "Never subtract more than 5 points for a venue type mismatch.\n"
        "- A highly relevant indoor event should still score 80+ even if the user mentioned outdoor vibes.\n"
        "- Only apply a weather penalty (-5 to -10) if the event is outdoor AND the forecast is clearly bad (heavy rain, storm).\n"
        "- Never let venue type or weather alone dominate the score.\n\n"

        "SCORING EXAMPLES:\n\n"

        "EXAMPLE 1 — Perfect match:\n"
        "User wants: 'jazz music weekend', Vibe: 'chill date night'\n"
        "Event: 'Birdland Jazz Club - Friday Night Jazz' | Music/Jazz | Indoor | Friday\n"
        "Score: 92 | Reason: Jazz genre matches exactly, chill indoor venue suits date night vibe.\n\n"

        "EXAMPLE 2 — Good match, minor venue mismatch:\n"
        "User wants: 'live rock concert', Vibe: 'outdoor vibes'\n"
        "Event: 'Beauty School Dropout' | Music/Rock | Indoor | Saturday\n"
        "Score: 82 | Reason: Rock concert matches the request well; indoor venue is a slight mismatch with outdoor preference (-5).\n\n"

        "EXAMPLE 3 — Wrong category:\n"
        "User wants: 'jazz music', Vibe: 'casual'\n"
        "Event: 'Yankees vs Red Sox' | Sports | Outdoor | Saturday\n"
        "Score: 8 | Reason: Sports event completely unrelated to jazz music request.\n\n"

        "EXAMPLE 4 — Outdoor event + bad weather (gentle penalty):\n"
        "User wants: 'outdoor festival', Vibe: 'outdoor vibes'\n"
        "Event: 'Summer Music Festival' | Music | Outdoor | Saturday | Weather: Heavy rain, outdoor_ok=False\n"
        "Score: 62 | Reason: Festival matches request well but heavy rain is a concern for outdoor attendance (-8).\n\n"

        "EXAMPLE 5 — Budget mismatch:\n"
        "User wants: 'live music', Budget: $30\n"
        "Event: 'Coldplay World Tour' | Music/Rock | Indoor | Saturday | Price: $150-$300\n"
        "Score: 20 | Reason: Music matches but price ($150-$300) far exceeds $30 budget.\n\n"

        "SCORING GUIDANCE FOR SPARSE DATA:\n"
        "If an event's description is missing or minimal, score based on its name, category, and genre alone. "
        "Do NOT default to 50 just because data is sparse — make your best inference. "
        "Only score 50 when the event is genuinely ambiguous and could plausibly be a partial match.\n\n"

        "Respond with ONLY a valid JSON array. No prose, no markdown, no code fences."
    )
    venue_pref  = request.venue_preference  # "Indoor" / "Outdoor" / "No preference"
    vibe_str    = request.vibe_notes if request.vibe_notes else "(none)"
    user_msg = (
        f"User is looking for: \"{request.event_description}\"\n"
        f"Venue preference: {venue_pref}\n"
        f"Vibe & extra preferences: {vibe_str}\n"
        f"Budget max: {budget_str}\n"
        f"Date range: {request.start_date} to {request.end_date}\n\n"
        f"Score each of the following {len(candidate_events)} events based on how well they match. "
        f"Apply the WEATHER SCORING RULES strictly based on the venue preference '{venue_pref}'.\n\n"
        f"{events_text}\n\n"
        f"Respond with ONLY this JSON array:\n"
        f'[{{"event_id": "...", "score": <0-100>, "reason": "one sentence explaining score including weather impact if relevant"}}, ...]'
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
        raw_json = response.choices[0].message.content.strip()
        scores_list = _parse_scores_json(raw_json)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[Recommendation Agent] LLM error: {exc}")
        scored = [
            ScoredEvent(
                event           = e,
                weather         = weather_map.get(e.event_id),
                relevance_score = 0.0,
                score_reason    = f"Scoring error: {exc}",
            )
            for e in candidate_events
        ]
        return RecommendationAgentOutput(request=request, recommendations=scored[:top_n], formatted_output="")

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
