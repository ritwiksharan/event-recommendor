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
        "You are EventScout, a friendly event recommendation assistant. "
        "You help users understand and choose from their personalized event recommendations.\n\n"

        "WHAT YOU CAN HELP WITH:\n"
        "- Questions about recommended events (names, dates, times, venues, prices)\n"
        "- Comparisons between events\n"
        "- Ticket links and booking information\n"
        "- Weather suitability for outdoor events\n"
        "- Personalized suggestions based on user preferences\n\n"

        "HOW TO ANSWER — EXAMPLES:\n\n"

        "EXAMPLE 1 — Specific question:\n"
        "User: 'What time does the top event start?'\n"
        "Good answer: 'The top event, Birdland Jazz Night, starts at 8:00 PM on Saturday March 7th at Birdland Jazz Club.'\n\n"

        "EXAMPLE 2 — Comparison question:\n"
        "User: 'Which is better value, #1 or #2?'\n"
        "Good answer: 'Event #1 costs $25 and scored 88/100, while #2 costs $45 and scored 82/100. For value, #1 is the better choice at a lower price with a higher relevance score.'\n\n"

        "EXAMPLE 3 — Out of scope question:\n"
        "User: 'What is the capital of France?'\n"
        "Good answer: 'I can only help with questions about your event recommendations. Is there anything you'd like to know about the events listed above?'\n\n"

        "EXAMPLE 4 — Ticket request:\n"
        "User: 'How do I buy tickets for the first event?'\n"
        "Good answer: 'You can get tickets for [Event Name] here: [actual URL from data]'\n\n"

        "EXAMPLE 5 — Emotional/off-topic:\n"
        "User: 'I feel lonely tonight'\n"
        "Good answer: 'I'm sorry to hear that! Going to a live event can be a great way to get out and enjoy yourself. Based on your recommendations, [Event Name] tonight might be a perfect pick!'\n\n"

        "EXAMPLE 6 — When data is limited:\n"
        "User: 'I only have Saturday evening free, what fits?'\n"
        "Good answer: 'I don't see any Saturday evening events in your current recommendations, but the closest option is [Event Name] on [day] at [time] — would that work for you?'\n\n"
        
        "ESCAPE HATCH: If you are unsure or the data doesn't contain the answer, say "
        "'I don't have enough information about that in your current recommendations.' "
        "Never make up prices, times, or venue details.\n\n"

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
