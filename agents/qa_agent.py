import os
import re
from typing import Optional
from litellm import completion
from ddgs import DDGS
from config import LLM_MODEL
from models.schemas import RecommendationAgentOutput, QAMessage, QARequest, QAResponse

DIRECTION_KEYWORDS = [
    "how to get", "directions", "direction", "transit", "subway", "bus", "train",
    "walk", "walking", "commute", "travel to", "get there", "get to", "from ",
    "nearest station", "how far", "distance",
]


def _is_directions_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in DIRECTION_KEYWORDS)


def _fetch_directions(venues: list[str], city: str, question: str) -> str:
    """Search DuckDuckGo for transit info from the user's stated origin to each venue."""
    results = []
    try:
        with DDGS() as ddgs:
            for venue in venues[:4]:  # cap at 4 to keep latency reasonable
                query = f"{question} {venue} {city} public transit"
                hits = list(ddgs.text(query, max_results=2))
                if hits:
                    snippet = hits[0].get("body", "")[:400]
                    results.append(f"**{venue}**: {snippet}")
    except Exception:
        pass
    return "\n\n".join(results)


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

# ── Python Backstop Classifier ────────────────────────────────────────────────
# Runs AFTER the LLM generates an answer.
# Catches cases where the LLM failed to use the escape hatch properly.

DISTRESS_KEYWORDS = [
    "suicide", "kill myself", "self harm", "hurt myself",
    "want to die", "end my life", "hopeless", "no reason to live"
]

OFFTOPIC_PATTERNS = [
    r"\b(capital of|president of|prime minister of)\b",
    r"\b(stock price|bitcoin|crypto)\b",
    r"\b(recipe for|how to cook|ingredients)\b",
    r"\b(who won|world cup|super bowl|oscar)\b",
    r"\b(weather in|temperature in)\s+(?!.*event)",  # weather not about events
]

EMPTY_ANSWER_THRESHOLD = 10  # characters


def backstop_classifier(question: str, answer: str) -> str:
    """
    Post-generation safety classifier.
    Runs after LLM generates answer to catch 3 out-of-scope categories:
    1. Distress/safety keywords → override with support message
    2. Off-topic patterns → redirect to events
    3. Empty/too-short answer → fallback message
    """
    q = question.lower()

    # Category 1: Distress detection → safety override
    if any(kw in q for kw in DISTRESS_KEYWORDS):
        return (
            "I'm sorry you're feeling this way. Please consider reaching out to "
            "a support line if you need help. In the meantime, sometimes getting "
            "out to a live event can lift your spirits — I'm here to help you find "
            "something enjoyable if you'd like."
        )

    # Category 2: Off-topic pattern detection → redirect
    if any(re.search(p, q) for p in OFFTOPIC_PATTERNS):
        return (
            "I'm EventScout, so I can only help with your event recommendations! "
            "Is there anything you'd like to know about the events listed — "
            "like prices, venues, or ticket links?"
        )

    # Category 3: Empty or error answer → fallback
    if len(answer.strip()) < EMPTY_ANSWER_THRESHOLD:
        return (
            "I wasn't able to generate a response. Try asking about event names, "
            "prices, dates, venues, or ticket links."
        )

    return answer  # answer passed all checks — return as-is


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
        "- Personalized suggestions based on user preferences\n"
        "- Directions and transit to venues (subway, bus, walking from any location)\n\n"

        "OUT-OF-SCOPE CATEGORIES (redirect politely):\n"
        "1. General knowledge unrelated to events — not geography trivia, history, or sports scores\n"
        "2. Personal/emotional advice — not medical, financial, or relationship advice\n"
        "3. Events outside the current recommendations — only the events shown above\n\n"

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

        "EXAMPLE 7 — Family/suitability question:\n"
        "User: 'Is there anything suitable for children?'\n"
        "Good answer: 'Based on the recommendations, [Event Name] at [Venue] could work for families — it's an indoor music event which tends to be more family-friendly. [Event Name 2] is a jazz performance which may suit older children. None are explicitly marketed as children's events, but these are your best options.'\n\n"

        "EXAMPLE 8 — Directions question:\n"
        "User: 'How do I get to these venues from Columbia University?'\n"
        "Good answer: 'Here's how to reach each venue from Columbia University:\\n"
        "**Birdland Jazz Club** (315 W 44th St): Take the 1 train from 116th St–Columbia to Times Sq–42nd St (~20 min), then walk 5 min west.\\n"
        "**Madison Square Garden**: Take the 1/2/3 train to 34th St–Penn Station (~25 min), it's right above the station.'\n\n"

        "ESCAPE HATCH: Only say 'I don't have enough information' if the specific fact "
        "(e.g. an exact price or time) is truly missing from the data. "
        "For questions about venues, suitability, directions, or general advice — always use the "
        "event names, venue names, and details already provided above to give a helpful answer. "
        "Never refuse a question that can be answered using the event data above.\n\n"

        + _build_context(qa.recommendations)
    )

    # Enrich with web search results for directions questions
    if _is_directions_question(qa.user_question):
        venues = [r.event.venue_name for r in qa.recommendations.recommendations]
        city = qa.recommendations.request.city
        web_results = _fetch_directions(venues, city, qa.user_question)
        if web_results:
            system_prompt += (
                "\n\nWEB SEARCH RESULTS (use these to answer the directions question):\n"
                + web_results
            )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in qa.conversation_history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": qa.user_question})

    try:
        response = completion(
            model       = LLM_MODEL,
            messages    = messages,
            temperature = 0.7,
            max_tokens  = 1000,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as exc:
        answer = f"Sorry, I encountered an error: {exc}. Please try again."

    safe_answer = backstop_classifier(qa.user_question, answer)
    updated_history = qa.conversation_history + [
        QAMessage(role="user",      content=qa.user_question),
        QAMessage(role="assistant", content=safe_answer),
    ]
    return QAResponse(answer=safe_answer, updated_history=updated_history)
