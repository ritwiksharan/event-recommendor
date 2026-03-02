import os
from typing import Optional
from litellm import completion
from ddgs import DDGS
from config import LLM_MODEL
from models.schemas import RecommendationAgentOutput, QAMessage, QARequest, QAResponse


# ── Web search helper ──────────────────────────────────────────────────────────

def _web_search(query: str, max_results: int = 3) -> str:
    """Run a DuckDuckGo search and return a compact summary of results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return ""
        lines = []
        for r in results:
            title = r.get("title", "")
            body  = r.get("body", "")
            href  = r.get("href", "")
            lines.append(f"- **{title}**: {body} ({href})")
        return "\n".join(lines)
    except Exception:
        return ""


# ── Context builder ────────────────────────────────────────────────────────────

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
        desc_str = e.description.strip() if e.description.strip() else "No description available from Ticketmaster."
        lines.append(
            f"#{i} {e.event_name} [Score: {r.relevance_score}/100]\n"
            f"  Date        : {e.date} ({'Weekend' if e.is_weekend else 'Weekday'}) @ {e.time}\n"
            f"  Venue       : {e.venue_name}, {e.venue_address}, {e.venue_city}, {e.venue_state} "
            f"({'Outdoor' if e.is_outdoor else 'Indoor'})\n"
            f"  Genre       : {e.category} / {e.genre}\n"
            f"  Price       : {price_str}\n"
            f"  Weather     : {weather_str}\n"
            f"  Tickets     : {e.url}\n"
            f"  Description : {desc_str}\n"
            f"  Why recommended: {r.score_reason}\n"
        )
    return "\n".join(lines)


def _enrich_with_search(recs: RecommendationAgentOutput) -> str:
    """
    Pre-search any events that have minimal or no descriptions so the LLM
    has richer context baked into the system prompt.
    """
    enrichments = []
    for r in recs.recommendations:
        e = r.event
        # Search if description is absent/short or the name looks like a ticket package
        needs_search = (
            not e.description.strip()
            or len(e.description.strip()) < 60
        )
        if needs_search:
            query = f"{e.event_name} {e.venue_name} {e.venue_city}"
            results = _web_search(query, max_results=3)
            if results:
                enrichments.append(
                    f"\n--- Web search results for \"{e.event_name}\" ---\n{results}"
                )
    return "\n".join(enrichments)


# ── Main agent ─────────────────────────────────────────────────────────────────

def run_qa_agent(
    qa: QARequest,
    anthropic_api_key: Optional[str] = None,
) -> QAResponse:
    """Agent 4 — answer user questions about recommendations using Gemini + web search."""
    if anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    event_context   = _build_context(qa.recommendations)
    search_context  = _enrich_with_search(qa.recommendations)

    system_prompt = (
        "You are EventScout, a knowledgeable and friendly event assistant. "
        "Your job is to help users understand and choose from their personalized event recommendations — "
        "and to answer any related questions using both the event data you have AND your broad world knowledge.\n\n"

        "## Event Data\n"
        "You have been given structured data for each recommended event below. "
        "Always reference specific event names, times, venues, and prices from this data when relevant.\n\n"

        "## Web Search Results\n"
        "For events whose Ticketmaster data was sparse, web search results have been pre-fetched and are "
        "included below the event list. Use these freely to give richer, more informative answers.\n\n"

        "## How to Answer\n"
        "- **Use your own knowledge**: If a user asks about a venue, artist, team, or event concept "
        "that you know about (e.g. 'What is Madison Square Garden?', 'Who is Drake?', "
        "'What is SJU?'), answer from your knowledge — don't pretend you don't know.\n"
        "- **Combine sources**: Merge the structured event data with your general knowledge and the web "
        "search results to give the most complete answer possible.\n"
        "- **Directions & maps**: For location questions, describe how to get there (subway, transit, parking) "
        "based on your knowledge of the city.\n"
        "- **Ticket packages / add-ons**: If an event listing appears to be a ticket add-on or package "
        "(e.g. food vouchers, VIP upgrades, parking passes), explain what it likely is and how it relates "
        "to the main event at that venue.\n"
        "- **Comparisons**: Compare events using score, price, weather suitability, and genre.\n"
        "- **Off-topic redirects**: For questions completely unrelated to events or the city "
        "(e.g. 'What is the capital of France?'), gently redirect: "
        "'I'm best at helping with your event recommendations — is there anything about the events listed above I can help with?'\n"
        "- **Never fabricate**: Don't invent prices, times, or URLs. If the data has it, use it. "
        "If not, say 'the listing doesn't include that detail' and offer to help with what you do know.\n\n"

        "EXAMPLE — Ticket add-on question:\n"
        "User: 'What is SJU Food & Bev Vouchers?'\n"
        "Good answer: 'SJU Food & Bev Vouchers is a food and beverage package offered alongside "
        "St. John's University (SJU) basketball games at Madison Square Garden. "
        "These are add-on tickets that include a pre-loaded credit you can spend on food and drinks "
        "at MSG concession stands during the event. If you purchase them alongside a game ticket, "
        "it's a convenient way to budget for in-arena dining.'\n\n"

        "EXAMPLE — Directions question:\n"
        "User: 'How do I get to Madison Square Garden?'\n"
        "Good answer: 'Madison Square Garden is located at 4 Pennsylvania Plaza, NYC. "
        "By subway take the A/C/E or 1/2/3 lines to 34th Street–Penn Station — it's directly connected. "
        "PATH trains also stop at 33rd Street a block away. Parking garages are nearby but expensive; "
        "public transit is strongly recommended.'\n\n"

        "---\n"
        "## Recommended Events\n\n"
        + event_context
        + ("\n\n## Web Search Enrichment\n" + search_context if search_context else "")
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
            max_tokens  = 1200,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as exc:
        answer = f"Sorry, I encountered an error: {exc}. Please try again."

    updated_history = qa.conversation_history + [
        QAMessage(role="user",      content=qa.user_question),
        QAMessage(role="assistant", content=answer),
    ]
    return QAResponse(answer=answer, updated_history=updated_history)
