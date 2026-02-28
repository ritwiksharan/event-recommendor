import os
from typing import Optional
from litellm import completion
from config import CLAUDE_MODEL
from models.schemas import RecommendationAgentOutput, QAMessage, QARequest, QAResponse


def _build_context(recs: RecommendationAgentOutput) -> str:
    lines = [
        f"Top {len(recs.recommendations)} recommended events for the user:\n"
        f"(City: {recs.request.city}, "
        f"Dates: {recs.request.start_date} to {recs.request.end_date})\n"
    ]
    for i, r in enumerate(recs.recommendations, 1):
        e, w = r.event, r.weather
        weather_str = (
            f"{w.description}, {w.temp_min_f:.0f}-{w.temp_max_f:.0f}F, "
            f"rain {w.precipitation_chance:.0f}%, suitable_outdoor={w.is_suitable_outdoor}"
            if w else "No forecast available"
        )
        price_str = (
            f"${e.price_min:.0f}-${e.price_max:.0f}"
            if e.price_min or e.price_max else "Free/Unknown"
        )
        lines.append(
            f"#{i} {e.event_name} [Score: {r.relevance_score}/100]\n"
            f"  Date   : {e.date} ({'Weekend' if e.is_weekend else 'Weekday'}) @ {e.time}\n"
            f"  Venue  : {e.venue_name} ({'Outdoor' if e.is_outdoor else 'Indoor'})\n"
            f"  Genre  : {e.category} / {e.genre}\n"
            f"  Price  : {price_str}\n"
            f"  Weather: {weather_str}\n"
            f"  Tickets: {e.url}\n"
            f"  Why recommended: {r.score_reason}\n"
        )
    return "\n".join(lines)


def run_qa_agent(
    qa: QARequest,
    anthropic_api_key: Optional[str] = None,
) -> QAResponse:
    """Agent 4 — answer user questions about recommendations using Claude."""
    if anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    system_prompt = (
        "You are a helpful event recommendation assistant. "
        "Answer the user's questions accurately based on the event data below. "
        "Be concise, friendly, and specific — reference actual event names, dates, "
        "prices, and venues. If asked for ticket links, provide the actual URLs.\n\n"
        + _build_context(qa.recommendations)
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in qa.conversation_history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": qa.user_question})

    try:
        response = completion(
            model       = CLAUDE_MODEL,
            messages    = messages,
            temperature = 0.7,
            max_tokens  = 1000,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as exc:
        answer = f"Sorry, I encountered an error: {exc}. Please try again."

    updated_history = qa.conversation_history + [
        QAMessage(role="user",      content=qa.user_question),
        QAMessage(role="assistant", content=answer),
    ]
    return QAResponse(answer=answer, updated_history=updated_history)
