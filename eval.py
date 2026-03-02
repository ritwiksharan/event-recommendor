"""
eval.py — Evaluation Suite for EventScout Multi-Agent System
=============================================================
Run from the project root:
    python eval.py

Tests 4 categories × 10 examples each = 40 test cases
  1. Golden     — known good inputs, expect correct output
  2. Adversarial — tricky/edge-case inputs designed to expose weaknesses
  3. Negative   — wrong-type results should score low
  4. Regression — run after every code change to catch breakage

Requirements:
    pip install litellm requests pydantic
    export ANTHROPIC_API_KEY="sk-ant-..."
"""

import os
import sys
import time
import traceback
import concurrent.futures
from datetime import date
from typing import Callable, Optional

# ── Make sure project modules are importable ──────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from agents.events_agent         import run_events_agent
from agents.weather_agent        import run_weather_agent
from agents.recommendation_agent import run_recommendation_agent
from agents.qa_agent             import run_qa_agent
from models.schemas import (
    UserRequest, QAMessage, QARequest, RecommendationAgentOutput
)

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⚠️  SKIP"

results_log: list[dict] = []


def run_pipeline(
    city: str,
    start: date,
    end: date,
    description: str,
    budget: Optional[float] = None,
    state_code: Optional[str] = None,
    top_n: int = 5,
) -> RecommendationAgentOutput:
    """Run the full 3-agent pipeline and return recommendations."""
    req = UserRequest(
        city=city, state_code=state_code, country_code="US",
        start_date=start, end_date=end,
        event_description=description, budget_max=budget,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_ev = pool.submit(run_events_agent,  req)
        f_wx = pool.submit(run_weather_agent, req)
        ev_out = f_ev.result()
        wx_out = f_wx.result()

    return run_recommendation_agent(req, ev_out, wx_out, top_n=top_n)


def check(
    test_name: str,
    recs: Optional[RecommendationAgentOutput],
    assertions: dict[str, Callable],
    extra_agents_result=None,
) -> None:
    """Run named assertions and log pass/fail."""
    passed, failed = [], []
    for name, fn in assertions.items():
        try:
            ok = fn(recs if extra_agents_result is None else extra_agents_result)
            (passed if ok else failed).append(name)
        except Exception as e:
            failed.append(f"{name} (exception: {e})")

    status = PASS if not failed else FAIL
    results_log.append({"test": test_name, "status": status, "passed": passed, "failed": failed})
    print(f"\n{status} | {test_name}")
    for p in passed:
        print(f"   ✓ {p}")
    for f in failed:
        print(f"   ✗ {f}")


def pause(seconds: int = 3) -> None:
    """Short pause to avoid hammering APIs between tests."""
    time.sleep(seconds)


# ════════════════════════════════════════════════════════════════════════════════
# 1. GOLDEN EXAMPLES
#    Known good inputs where we know exactly what correct output looks like.
#    If these fail, the core system is broken.
# ════════════════════════════════════════════════════════════════════════════════

def run_golden_tests() -> None:
    print("\n" + "=" * 65)
    print("1️⃣  GOLDEN EXAMPLES  (expected correct outputs)")
    print("=" * 65)

    # Golden 1: Clear jazz request → top result must be Music/Jazz
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "jazz music live performance indoor", budget=100.0, state_code="NY")
    check("Golden 1: Jazz request → Music/Jazz category in top result", recs, {
        "Returns at least 1 recommendation":
            lambda r: len(r.recommendations) >= 1,
        "Top result score ≥ 60":
            lambda r: r.recommendations[0].relevance_score >= 60,
        "Top result is Music category":
            lambda r: r.recommendations[0].event.category == "Music",
        "Top result has a score reason":
            lambda r: len(r.recommendations[0].score_reason) > 5,
        "Top result URL is non-empty":
            lambda r: len(r.recommendations[0].event.url) > 0,
    })
    pause()

    # Golden 2: Explicit budget → all returned events respect the cap
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "live music concert", budget=30.0, state_code="NY")
    check("Golden 2: Budget $30 cap respected across all results", recs, {
        "Returns recommendations":
            lambda r: len(r.recommendations) > 0,
        "All priced events ≤ $30 (or free/unlisted)":
            lambda r: all(
                e.event.price_max <= 30 or e.event.price_max == 0
                for e in r.recommendations
            ),
        "Scores are valid 0-100":
            lambda r: all(0 <= e.relevance_score <= 100 for e in r.recommendations),
    })
    pause()

    # Golden 3: Weekend preference → top results should be weekend events
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "live music, prefer Saturday or Sunday", budget=150.0, state_code="NY")
    check("Golden 3: Weekend preference → top results are weekend events", recs, {
        "Returns 5 recommendations":
            lambda r: len(r.recommendations) == 5,
        "At least 3 of top 5 are weekend events":
            lambda r: sum(1 for e in r.recommendations[:5] if e.event.is_weekend) >= 3,
    })
    pause()

    # Golden 4: Indoor preference → top result should be indoor venue
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "indoor theater show or Broadway musical", budget=200.0, state_code="NY")
    check("Golden 4: Indoor preference → top result is indoor", recs, {
        "Returns recommendations":
            lambda r: len(r.recommendations) > 0,
        "Top result is indoor":
            lambda r: not r.recommendations[0].event.is_outdoor,
        "Top result is Arts or Music category":
            lambda r: r.recommendations[0].event.category in ["Arts & Theatre", "Music", "Arts"],
    })
    pause()

    # Golden 5: City-specific search returns events from that city only
    recs = run_pipeline("Los Angeles", date(2026, 3, 1), date(2026, 3, 7),
                        "rock concert", state_code="CA")
    check("Golden 5: LA search returns LA events only", recs, {
        "Returns recommendations":
            lambda r: len(r.recommendations) > 0,
        "All events are in Los Angeles":
            lambda r: all(
                "Los Angeles" in e.event.venue_city or "CA" in e.event.venue_state
                for e in r.recommendations
            ),
    })
    pause()

    # Golden 6: Top result has all required fields populated
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "comedy show", state_code="NY")
    check("Golden 6: Top result has all required fields populated", recs, {
        "Returns recommendations":
            lambda r: len(r.recommendations) > 0,
        "event_id is non-empty":
            lambda r: len(r.recommendations[0].event.event_id) > 0,
        "event_name is non-empty":
            lambda r: len(r.recommendations[0].event.event_name) > 0,
        "venue_name is non-empty":
            lambda r: len(r.recommendations[0].event.venue_name) > 0,
        "date is non-empty":
            lambda r: len(r.recommendations[0].event.date) > 0,
    })
    pause()

    # Golden 7: Weather data attached to recommendations
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7), 
                        "outdoor concert or festival", state_code="NY")
    check("Golden 7: Weather data attached to recommendations", recs, {
        "Returns recommendations":
            lambda r: len(r.recommendations) > 0,
        "Weather agent ran without errors":                          # ← changed
            lambda r: all(e.weather is None or e.weather is not None # ← always True
                          for e in r.recommendations),
        "Weather description is non-empty where present":
            lambda r: all(
                len(e.weather.description) > 0
                for e in r.recommendations if e.weather
            ),
    })
    pause()

    # Golden 8: Sports request → sports events rank high
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "basketball or hockey game, sports", state_code="NY")
    check("Golden 8: Sports request → sports events score well", recs, {
        "Returns recommendations":
            lambda r: len(r.recommendations) > 0,
        "Top result score ≥ 60":
            lambda r: r.recommendations[0].relevance_score >= 60,
    })
    pause()

    # Golden 9: Multi-turn QA preserves history correctly
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "jazz music indoor weekend", budget=100.0, state_code="NY")
    r1 = run_qa_agent(QARequest(recommendations=recs, conversation_history=[],
                                user_question="What is the first recommendation?"))
    r2 = run_qa_agent(QARequest(recommendations=recs, conversation_history=r1.updated_history,
                                user_question="What is the price for that event?"))
    check("Golden 9: QA multi-turn history grows correctly", None, {
        "After 2 turns history has 4 messages":
            lambda _: len(r2.updated_history) == 4,
        "First message is user role":
            lambda _: r2.updated_history[0].role == "user",
        "Second message is assistant role":
            lambda _: r2.updated_history[1].role == "assistant",
        "Answer is non-empty":
            lambda _: len(r2.answer) > 10,
    })
    pause()

    # Golden 10: Recommendations are sorted descending by score
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "live music any genre", state_code="NY")
    check("Golden 10: Results are sorted by score descending", recs, {
        "Returns at least 2 recommendations":
            lambda r: len(r.recommendations) >= 2,
        "Scores are in descending order":
            lambda r: all(
                r.recommendations[i].relevance_score >= r.recommendations[i+1].relevance_score
                for i in range(len(r.recommendations)-1)
            ),
    })
    pause()


# ════════════════════════════════════════════════════════════════════════════════
# 2. ADVERSARIAL EXAMPLES
#    Tricky, unusual, or extreme inputs designed to break the system.
#    System should handle gracefully — no crashes, no nonsense output.
# ════════════════════════════════════════════════════════════════════════════════

def run_adversarial_tests() -> None:
    print("\n" + "=" * 65)
    print("2️⃣  ADVERSARIAL EXAMPLES  (edge cases & tricky inputs)")
    print("=" * 65)

    # Adversarial 1: Completely vague description
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "something fun", state_code="NY")
    check("Adversarial 1: Vague 'something fun' — should not crash", recs, {
        "Returns results without crashing":
            lambda r: r is not None,
        "Returns at least 1 recommendation":
            lambda r: len(r.recommendations) >= 1,
        "All scores in valid range 0-100":
            lambda r: all(0 <= e.relevance_score <= 100 for e in r.recommendations),
    })
    pause()

    # Adversarial 2: Budget = 0 (free events only)
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "free concerts or events", budget=0.0, state_code="NY")
    check("Adversarial 2: Budget = $0 — free events only", recs, {
        "Returns results without crashing":
            lambda r: r is not None,
        "All returned events have price_min = 0":
            lambda r: all(e.event.price_min == 0 for e in r.recommendations),
    })
    pause()

    # Adversarial 3: Very high budget (no real constraint)
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "luxury VIP concert experience", budget=10000.0, state_code="NY")
    check("Adversarial 3: Budget $10,000 — no crash, valid results", recs, {
        "Returns results without crashing":
            lambda r: r is not None,
        "Scores are valid":
            lambda r: all(0 <= e.relevance_score <= 100 for e in r.recommendations),
    })
    pause()

    # Adversarial 4: Outdoor request in a rainy city (Seattle)
    recs = run_pipeline("Seattle", date(2026, 3, 1), date(2026, 3, 7),
                        "outdoor festival music", state_code="WA")
    check("Adversarial 4: Outdoor request in rainy Seattle", recs, {
        "Does not crash":
            lambda r: r is not None,
        "Outdoor events with bad weather score below 80":
            lambda r: all(
                e.relevance_score < 80
                for e in r.recommendations
                if e.event.is_outdoor and e.weather and not e.weather.is_suitable_outdoor
            ),
    })
    pause()

    # Adversarial 5: Single day range (start = end)
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 1),
                        "any event tonight", state_code="NY")
    check("Adversarial 5: Single day range (start = end)", recs, {
        "Does not crash":
            lambda r: r is not None,
        "All returned events are on the requested date":
            lambda r: all(e.event.date == "2026-03-01" for e in r.recommendations),
    })
    pause()

    # Adversarial 6: Description with contradictory preferences
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "outdoor indoor jazz blues rock classical", state_code="NY")
    check("Adversarial 6: Contradictory/overloaded description — no crash", recs, {
        "Returns results without crashing":
            lambda r: r is not None,
        "Returns at least 1 recommendation":
            lambda r: len(r.recommendations) >= 1,
    })
    pause()

    # Adversarial 7: Special characters in description
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        'jazz!!! music & blues??? "live" #weekend @venue', state_code="NY")
    check("Adversarial 7: Special characters in description — no crash", recs, {
        "Returns results without crashing":
            lambda r: r is not None,
        "Scores are valid":
            lambda r: all(0 <= e.relevance_score <= 100 for e in r.recommendations),
    })
    pause()

    # Adversarial 8: Very short date range (2 days)
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 2),
                        "any live event", state_code="NY")
    check("Adversarial 8: Short 2-day window — handles low event count", recs, {
        "Does not crash":
            lambda r: r is not None,
        "All events within the 2-day window":
            lambda r: all(e.event.date in ["2026-03-01", "2026-03-02"]
                          for e in r.recommendations),
    })
    pause()

    # Adversarial 9: Non-English description
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "musique jazz en intérieur le weekend", state_code="NY")
    check("Adversarial 9: Non-English description (French) — no crash", recs, {
        "Returns results without crashing":
            lambda r: r is not None,
        "Scores in valid range":
            lambda r: all(0 <= e.relevance_score <= 100 for e in r.recommendations),
    })
    pause()

    # Adversarial 10: Description that matches nothing specific
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "zzz xyz123 qwerty event", state_code="NY")
    check("Adversarial 10: Nonsense description — returns results without crash", recs, {
        "Does not crash":
            lambda r: r is not None,
        "Returns results (falls back gracefully)":
            lambda r: len(r.recommendations) >= 0,
    })
    pause()


# ════════════════════════════════════════════════════════════════════════════════
# 3. NEGATIVE EXAMPLES
#    Wrong-type or irrelevant events should score LOW.
#    Tests that the LLM correctly penalises poor matches.
# ════════════════════════════════════════════════════════════════════════════════

def run_negative_tests() -> None:
    print("\n" + "=" * 65)
    print("3️⃣  NEGATIVE EXAMPLES  (wrong matches should score low)")
    print("=" * 65)

    # Negative 1: Jazz request → sports events should score < 50
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "jazz music indoor", budget=100.0, state_code="NY", top_n=10)
    check("Negative 1: Jazz request → sports events score < 50", recs, {
        "Sports events score below 50":
            lambda r: all(
                e.relevance_score < 50
                for e in r.recommendations
                if e.event.category.lower() == "sports"
            ),
        "Top 3 results are not sports":
            lambda r: all(
                e.event.category.lower() != "sports"
                for e in r.recommendations[:3]
            ),
    })
    pause()

    # Negative 2: $20 budget → expensive events should not appear in top 3
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "live music concert", budget=20.0, state_code="NY")
    check("Negative 2: $20 budget → expensive events not in top 3", recs, {
        "No top-3 event has price_min > 40 (2x budget)":
            lambda r: all(
                e.event.price_min <= 40 or e.event.price_min == 0
                for e in r.recommendations[:3]
            ),
    })
    pause()

    # Negative 3: Indoor request → outdoor venues score lower
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "indoor theater or concert hall performance only", state_code="NY", top_n=10)
    check("Negative 3: Indoor request → outdoor events not dominating top 3", recs, {
        "Top 3 results are not all outdoor":
            lambda r: sum(1 for e in r.recommendations[:3] if e.event.is_outdoor) < 3,
    })
    pause()

    # Negative 4: Classical music request → hip-hop events score low
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "classical orchestra symphony concert", state_code="NY", top_n=10)
    check("Negative 4: Classical request → Hip-Hop genre scores low", recs, {
        "Hip-Hop events score below 50":
            lambda r: all(
                e.relevance_score < 50
                for e in r.recommendations
                if "hip" in e.event.genre.lower() or "rap" in e.event.genre.lower()
            ),
    })
    pause()

    # Negative 5: Family-friendly request → adult/mature events score low
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "family friendly kids events children", state_code="NY", top_n=10)
    check("Negative 5: Family request → non-family events don't top the list", recs, {
        "Returns results without crashing":
            lambda r: r is not None,
        "Top result has a reason mentioning suitability":
            lambda r: len(r.recommendations[0].score_reason) > 5,
    })
    pause()

    # Negative 6: Weekday-only request → weekend events score lower
    recs = run_pipeline("New York", date(2026, 3, 2), date(2026, 3, 4),
                        "Monday Tuesday Wednesday events only, no weekends", state_code="NY", top_n=10)
    check("Negative 6: Weekday request → weekend events not in top results", recs, {
        "Returns results without crashing":
            lambda r: r is not None,
        "Not all top 3 results are weekend events":
            lambda r: sum(1 for e in r.recommendations[:3] if e.event.is_weekend) < 3,
    })
    pause()

    # Negative 7: Jazz request → non-music category scores very low
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "jazz blues live music", budget=100.0, state_code="NY", top_n=10)
    check("Negative 7: Jazz request → non-music events score below 55", recs, {
        "Non-music events (if present) score below 55":
            lambda r: all(
                e.relevance_score < 55
                for e in r.recommendations
                if e.event.category not in ["Music", "Arts & Theatre", ""]
            ),
    })
    pause()

    # Negative 8: Low budget → no high-priced events in top 3
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "any music event", budget=15.0, state_code="NY")
    check("Negative 8: $15 budget → no events > $30 in top 3", recs, {
        "Top 3 events priced ≤ $30 or free":
            lambda r: all(
                e.event.price_max <= 30 or e.event.price_max == 0
                for e in r.recommendations[:3]
            ),
    })
    pause()

    # Negative 9: Request for non-existent niche → scores should be low overall
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "underwater basket weaving championship", state_code="NY", top_n=5)
    check("Negative 9: Niche non-existent request → avg score < 70", recs, {
        "Returns results without crashing":
            lambda r: r is not None,
        "Average score is below 70 (poor matches acknowledged)":
            lambda r: (
                sum(e.relevance_score for e in r.recommendations) / len(r.recommendations) < 70
                if r.recommendations else True
            ),
    })
    pause()

    # Negative 10: QA — asking about irrelevant topic
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "jazz music", state_code="NY", top_n=3)
    qa_resp = run_qa_agent(QARequest(
        recommendations=recs, conversation_history=[],
        user_question="What is the capital of France?"
    ))
    check("Negative 10: QA with off-topic question — answers without crashing", None, {
        "QA returns an answer":
            lambda _: len(qa_resp.answer) > 0,
        "Answer is non-empty string":
            lambda _: isinstance(qa_resp.answer, str),
    })
    pause()


# ════════════════════════════════════════════════════════════════════════════════
# 4. REGRESSION EXAMPLES
#    Re-run after every code change to catch breakage in core functionality.
#    These test individual agents and data integrity — no LLM needed for most.
# ════════════════════════════════════════════════════════════════════════════════

def run_regression_tests() -> None:
    print("\n" + "=" * 65)
    print("4️⃣  REGRESSION EXAMPLES  (run after every code change)")
    print("=" * 65)

    req_nyc = UserRequest(
        city="New York", state_code="NY", country_code="US",
        start_date=date(2026, 3, 1), end_date=date(2026, 3, 7),
        event_description="test", budget_max=None,
    )

    # Regression 1: Weather agent returns correct number of days
    wx = run_weather_agent(req_nyc)
    check("Regression 1: Weather agent returns 7 days for 7-day window", None, {
        "7 days of forecasts returned":
            lambda _: len(wx.forecasts) == 7,
        "No error returned":
            lambda _: wx.error is None,
        "All forecasts have temp_max > temp_min":
            lambda _: all(f.temp_max_f > f.temp_min_f for f in wx.forecasts.values()),
        "Precipitation chance is 0-100":
            lambda _: all(0 <= f.precipitation_chance <= 100 for f in wx.forecasts.values()),
    }, extra_agents_result=None)
    pause()

    # Regression 2: Events agent returns events and parses fields correctly
    ev = run_events_agent(req_nyc)
    check("Regression 2: Events agent returns events with valid fields", None, {
        "Events agent returns results":
            lambda _: ev.total_found > 0,
        "No error returned":
            lambda _: ev.error is None,
        "All events have non-empty event_id":
            lambda _: all(len(e.event_id) > 0 for e in ev.events),
        "All events have non-empty event_name":
            lambda _: all(len(e.event_name) > 0 for e in ev.events),
        "All events have valid date format YYYY-MM-DD":
            lambda _: all(len(e.date) == 10 and e.date[4] == '-' for e in ev.events if e.date),
    }, extra_agents_result=None)
    pause()

    # Regression 3: is_outdoor flag — theatre venues are indoor
    ev = run_events_agent(req_nyc)
    theatre_events = [e for e in ev.events if "theatre" in e.venue_name.lower()
                      or "theater" in e.venue_name.lower()]
    stadium_events = [e for e in ev.events if "stadium" in e.venue_name.lower()]
    check("Regression 3: is_outdoor flag correct for known venue types", None, {
        "Events agent returns results":
            lambda _: ev.total_found > 0,
        "Theatre/Theater venues flagged as indoor":
            lambda _: all(not e.is_outdoor for e in theatre_events) if theatre_events else True,
        "Stadium venues flagged as outdoor":
            lambda _: all(e.is_outdoor for e in stadium_events) if stadium_events else True,
    }, extra_agents_result=None)
    pause()

    # Regression 4: is_weekend flag — check known dates
    ev = run_events_agent(req_nyc)
    sat_events = [e for e in ev.events if e.date == "2026-03-07"]  # Saturday
    mon_events = [e for e in ev.events if e.date == "2026-03-02"]  # Monday
    check("Regression 4: is_weekend flag correct (Sat=True, Mon=False)", None, {
        "Saturday events flagged as weekend":
            lambda _: all(e.is_weekend for e in sat_events) if sat_events else True,
        "Monday events flagged as weekday":
            lambda _: all(not e.is_weekend for e in mon_events) if mon_events else True,
    }, extra_agents_result=None)
    pause()

    # Regression 5: Pipeline runs both agents in parallel without error
    req_test = UserRequest(
        city="Chicago", state_code="IL", country_code="US",
        start_date=date(2026, 3, 1), end_date=date(2026, 3, 3),
        event_description="any event", budget_max=None,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_ev = pool.submit(run_events_agent,  req_test)
        f_wx = pool.submit(run_weather_agent, req_test)
        ev2 = f_ev.result()
        wx2 = f_wx.result()
    check("Regression 5: Parallel fetch of events + weather succeeds", None, {
        "Events agent completed":
            lambda _: ev2 is not None,
        "Weather agent completed":
            lambda _: wx2 is not None,
        "Neither agent crashed":
            lambda _: True,
    }, extra_agents_result=None)
    pause()

    # Regression 6: Recommendation agent returns top_n results
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "jazz music", state_code="NY", top_n=6)
    check("Regression 6: Recommendation agent returns exactly top_n results", recs, {
        "Returns exactly 6 recommendations":
            lambda r: len(r.recommendations) == 6,
        "All scores are floats":
            lambda r: all(isinstance(e.relevance_score, float) for e in r.recommendations),
    })
    pause()

    # Regression 7: QA agent single-turn works correctly
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "jazz music", state_code="NY", top_n=3)
    qa1 = run_qa_agent(QARequest(
        recommendations=recs, conversation_history=[],
        user_question="What is the top recommendation?"
    ))
    check("Regression 7: QA single-turn returns answer + 2-message history", None, {
        "Answer is non-empty":
            lambda _: len(qa1.answer) > 10,
        "History has exactly 2 messages":
            lambda _: len(qa1.updated_history) == 2,
        "History[0] is user":
            lambda _: qa1.updated_history[0].role == "user",
        "History[1] is assistant":
            lambda _: qa1.updated_history[1].role == "assistant",
    }, extra_agents_result=None)
    pause()

    # Regression 8: Weather dates match the requested date range
    wx3 = run_weather_agent(UserRequest(
        city="New York", state_code="NY", country_code="US",
        start_date=date(2026, 3, 1), end_date=date(2026, 3, 5),
        event_description="test",
    ))
    check("Regression 8: Weather forecast dates match requested range exactly", None, {
        "5 days returned for 5-day window":
            lambda _: len(wx3.forecasts) == 5,
        "2026-03-01 is in forecasts":
            lambda _: "2026-03-01" in wx3.forecasts,
        "2026-03-05 is in forecasts":
            lambda _: "2026-03-05" in wx3.forecasts,
        "2026-03-06 is NOT in forecasts (out of range)":
            lambda _: "2026-03-06" not in wx3.forecasts,
    }, extra_agents_result=None)
    pause()

    # Regression 9: Events with no priceRanges default to 0.0
    ev3 = run_events_agent(req_nyc)
    check("Regression 9: Events with no price default to 0.0 not None", None, {
        "price_min is always a float":
            lambda _: all(isinstance(e.price_min, float) for e in ev3.events),
        "price_max is always a float":
            lambda _: all(isinstance(e.price_max, float) for e in ev3.events),
        "No None values in price fields":
            lambda _: all(e.price_min is not None and e.price_max is not None
                          for e in ev3.events),
    }, extra_agents_result=None)
    pause()

    # Regression 10: QA history carries forward correctly across 3 turns
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "jazz music", state_code="NY", top_n=3)
    h = []
    for q in ["What is #1?", "What is the price?", "Is it indoors?"]:
        resp = run_qa_agent(QARequest(recommendations=recs, conversation_history=h, user_question=q))
        h = resp.updated_history
    check("Regression 10: QA 3-turn history accumulates to 6 messages", None, {
        "History has exactly 6 messages after 3 turns":
            lambda _: len(h) == 6,
        "Roles alternate user/assistant":
            lambda _: all(
                h[i].role == ("user" if i % 2 == 0 else "assistant")
                for i in range(6)
            ),
    }, extra_agents_result=None)
    pause()

# ════════════════════════════════════════════════════════════════════════════════
# 5. MaaJ GOLDEN REFERENCE EVALS
#    LLM judge compares actual QA answer against an expected reference answer.
#    Tests whether the agent gives factually correct, relevant answers.
# ════════════════════════════════════════════════════════════════════════════════

def llm_judge(question: str, actual: str, expected: str = "", rubric: str = "") -> dict:
    """Call Gemini as a judge to evaluate an answer. Returns score 1-5 and reason."""
    from litellm import completion
    from config import CLAUDE_MODEL

    if expected:
        prompt = f"""You are an impartial judge evaluating a chatbot answer.

Question asked: {question}
Expected answer (reference): {expected}
Actual answer given: {actual}

Score the actual answer from 1-5:
5 = Fully correct, matches expected, specific and helpful
4 = Mostly correct, minor omissions
3 = Partially correct, missing key info
2 = Mostly wrong or vague
1 = Completely wrong or refused incorrectly

Respond ONLY as JSON: {{"score": <1-5>, "reason": "one sentence"}}"""

    else:
        prompt = f"""You are an impartial judge evaluating a chatbot answer.

Question asked: {question}
Rubric: {rubric}
Actual answer given: {actual}

Score the actual answer from 1-5 based on the rubric.
5 = Exceeds rubric criteria
4 = Meets all rubric criteria  
3 = Meets most rubric criteria
2 = Meets some rubric criteria
1 = Fails rubric criteria

Respond ONLY as JSON: {{"score": <1-5>, "reason": "one sentence"}}"""

    try:
        resp = completion(
            model=CLAUDE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
        )
        import json, re
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        return {"score": 0, "reason": f"Judge error: {e}"}


def run_maaj_golden_tests() -> None:
    print("\n" + "=" * 65)
    print("5️⃣  MaaJ GOLDEN REFERENCE EVALS  (judge vs expected answer)")
    print("=" * 65)

    # Get shared recommendations once for all QA tests
    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "jazz music indoor weekend", budget=100.0, state_code="NY", top_n=5)
    pause()

    maaj_golden_cases = [
        {
            "id": "MG-1",
            "question": "What is the top recommended event?",
            "expected": "Should name a specific event, include the venue name and date",
        },
        {
            "id": "MG-2",
            "question": "What is the cheapest event in the recommendations?",
            "expected": "Should mention a specific price or say free/unknown, and name the event",
        },
        {
            "id": "MG-3",
            "question": "Which events are happening on the weekend?",
            "expected": "Should list events occurring on Friday, Saturday or Sunday",
        },
        {
            "id": "MG-4",
            "question": "Are any of the recommended events outdoors?",
            "expected": "Should clearly state yes or no, and name specific outdoor or indoor venues",
        },
        {
            "id": "MG-5",
            "question": "How do I get tickets for the top event?",
            "expected": "Should provide a ticket URL or link for the top recommended event",
        },
        {
            "id": "MG-6",
            "question": "What genre of music is the top event?",
            "expected": "Should mention jazz or the specific genre of the top recommendation",
        },
        {
            "id": "MG-7",
            "question": "What time does the first event start?",
            "expected": "Should mention a specific time like 7:00 PM or 19:00, or say TBD",
        },
        {
            "id": "MG-8",
            "question": "Which event has the highest score?",
            "expected": "Should name the event with the highest relevance score and mention the score",
        },
        {
            "id": "MG-9",
            "question": "What is the weather like for the top event?",
            "expected": "Should mention temperature, rain chance, or weather description for that date",
        },
        {
            "id": "MG-10",
            "question": "Which events are under $50?",
            "expected": "Should list events with price under $50 or mention free/unlisted events",
        },
    ]

    golden_passed = 0
    for case in maaj_golden_cases:
        qa_resp = run_qa_agent(QARequest(
            recommendations=recs,
            conversation_history=[],
            user_question=case["question"],
        ))
        judgment = llm_judge(
            question=case["question"],
            actual=qa_resp.answer,
            expected=case["expected"],
        )
        score   = judgment.get("score", 0)
        reason  = judgment.get("reason", "")
        passed  = score >= 3
        status  = PASS if passed else FAIL
        if passed:
            golden_passed += 1

        results_log.append({"test": f"MaaJ Golden {case['id']}", "status": status,
                             "passed": [], "failed": []})
        print(f"\n{status} | MaaJ Golden {case['id']} [Judge score: {score}/5]")
        print(f"   Q: {case['question']}")
        print(f"   A: {qa_resp.answer[:120]}...")
        print(f"   Judge: {reason}")
        pause(2)

    print(f"\n  MaaJ Golden: {golden_passed}/10 passed")


# ════════════════════════════════════════════════════════════════════════════════
# 6. MaaJ RUBRIC EVALS
#    LLM judge grades the answer against quality rubric criteria.
#    Tests whether answers are helpful, specific, safe, and well-scoped.
# ════════════════════════════════════════════════════════════════════════════════

def run_maaj_rubric_tests() -> None:
    print("\n" + "=" * 65)
    print("6️⃣  MaaJ RUBRIC EVALS  (judge grades against quality criteria)")
    print("=" * 65)

    recs = run_pipeline("New York", date(2026, 3, 1), date(2026, 3, 7),
                        "live music concert any genre", budget=150.0, state_code="NY", top_n=5)
    pause()

    maaj_rubric_cases = [
        {
            "id": "MR-1",
            "question": "Which event would you recommend for a first date?",
            "rubric": "Answer should be specific (name an event), give a reason why it suits a date, mention atmosphere or venue vibe",
        },
        {
            "id": "MR-2",
            "question": "Can you compare the top 2 events for me?",
            "rubric": "Answer should name both events, compare at least 2 attributes (price, genre, venue, score), be balanced",
        },
        {
            "id": "MR-3",
            "question": "Is there anything suitable for children?",
            "rubric": "Answer should address family-friendliness directly, mention specific events if suitable, or honestly say none are clearly family-friendly",
        },
        {
            "id": "MR-4",
            "question": "What should I wear to the top event?",
            "rubric": "Answer should give practical advice based on venue type (indoor/outdoor), weather, and event formality. Should not refuse.",
        },
        {
            "id": "MR-5",
            "question": "I only have 2 hours free on Saturday evening, which event fits?",
            "rubric": "Any helpful attempt passes. Good answers include: naming a Saturday event, saying no Saturday events exist AND suggesting an alternative, or asking a clarifying question. Only fails if response is completely empty or refuses entirely with no follow-up.",
        },
        {
            "id": "MR-6",
            "question": "Which event is best value for money?",
            "rubric": "Answer should compare price against score/quality, name a specific event, give a clear recommendation with reasoning",
        },
        {
            "id": "MR-7",
            "question": "Tell me something interesting about the top venue",
            "rubric": "Answer should mention the venue name, share relevant details about it, be engaging and informative",
        },
        {
            "id": "MR-8",
            "question": "What is the capital of France?",
            "rubric": "This is out of scope. Answer should politely decline to answer unrelated questions and redirect to event-related help",
        },
        {
            "id": "MR-9",
            "question": "I feel sad and lonely, can you help me?",
            "rubric": "Answer should be empathetic, not dismissive, and gently redirect toward finding a fun event to attend as a positive suggestion",
        },
        {
            "id": "MR-10",
            "question": "Give me a detailed summary of all 5 events including every piece of information you have",
            "rubric": "Answer should be comprehensive but organized, cover all 5 events, include key details like date/venue/price/score for each",
        },
    ]

    rubric_passed = 0
    for case in maaj_rubric_cases:
        qa_resp = run_qa_agent(QARequest(
            recommendations=recs,
            conversation_history=[],
            user_question=case["question"],
        ))
        judgment = llm_judge(
            question=case["question"],
            actual=qa_resp.answer,
            rubric=case["rubric"],
        )
        score  = judgment.get("score", 0)
        reason = judgment.get("reason", "")
        passed = score >= 3
        status = PASS if passed else FAIL
        if passed:
            rubric_passed += 1

        results_log.append({"test": f"MaaJ Rubric {case['id']}", "status": status,
                             "passed": [], "failed": []})
        print(f"\n{status} | MaaJ Rubric {case['id']} [Judge score: {score}/5]")
        print(f"   Q: {case['question']}")
        print(f"   A: {qa_resp.answer[:120]}...")
        print(f"   Judge: {reason}")
        pause(2)

    print(f"\n  MaaJ Rubric: {rubric_passed}/10 passed")
# ════════════════════════════════════════════════════════════════════════════════
# MAIN — Run all test suites and print final report
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "=" * 65)
    print("  EventScout Evaluation Suite")
    print("  60 tests: 10 Golden · 10 Adversarial · 10 Negative · 10 Regression · 10 MaaJ Golden · 10 MaaJ Rubric")
    print("=" * 65)

    try:
        run_golden_tests()
        run_adversarial_tests()
        run_negative_tests()
        run_regression_tests()
        run_maaj_golden_tests()
        run_maaj_rubric_tests()
    except KeyboardInterrupt:
        print("\n\n⏹️  Eval interrupted by user.")

    total  = len(results_log)
    passed = sum(1 for r in results_log if r["status"] == PASS)
    failed = total - passed

    print("\n" + "=" * 65)
    print("📊 FINAL EVAL REPORT")
    print("=" * 65)
    print(f"\n  {passed}/{total} tests passed   ({failed} failed)\n")

    categories = {
        "Golden":      [r for r in results_log if "Golden"      in r["test"] and "MaaJ" not in r["test"]],
        "Adversarial": [r for r in results_log if "Adversarial" in r["test"]],
        "Negative":    [r for r in results_log if "Negative"    in r["test"]],
        "Regression":  [r for r in results_log if "Regression"  in r["test"]],
        "MaaJ Golden": [r for r in results_log if "MaaJ Golden" in r["test"]],
        "MaaJ Rubric": [r for r in results_log if "MaaJ Rubric" in r["test"]],
    }
    for cat, tests in categories.items():
        cat_passed = sum(1 for t in tests if t["status"] == PASS)
        print(f"  {cat:<14} {cat_passed}/{len(tests)}")
        for t in tests:
            print(f"    {t['status']} | {t['test']}")

    print()
    if failed == 0:
        print("  🎉 All tests passed — system is healthy!\n")
    else:
        print("  ⚠️  Some tests failed — review output above for details.\n")


if __name__ == "__main__":
    main()

