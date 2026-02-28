import concurrent.futures
from datetime import date, timedelta

import streamlit as st

from agents.events_agent import run_events_agent
from agents.weather_agent import run_weather_agent
from agents.recommendation_agent import run_recommendation_agent
from agents.qa_agent import run_qa_agent
from models.schemas import UserRequest, QAMessage, QARequest

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EventScout",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ──────────────────────────────────────────────────────────────
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "events_out" not in st.session_state:
    st.session_state.events_out = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[QAMessage] = []

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 EventScout")
    st.markdown("Powered by **Ticketmaster**, **Open-Meteo** & **Claude AI**")
    st.divider()

    st.subheader("Search Settings")

    city = st.text_input("City", value="New York", placeholder="e.g. Los Angeles")

    col1, col2 = st.columns(2)
    with col1:
        state_code = st.text_input("State Code", value="NY", max_chars=2)
    with col2:
        country_code = st.text_input("Country", value="US", max_chars=2)

    today = date.today()
    col3, col4 = st.columns(2)
    with col3:
        start_date = st.date_input("Start Date", value=today)
    with col4:
        end_date = st.date_input("End Date", value=today + timedelta(days=7))

    event_description = st.text_area(
        "What are you looking for?",
        value="fun weekend events, concerts or sports",
        placeholder="e.g. outdoor jazz festival, family-friendly, under $50",
        height=80,
    )

    budget_max = st.number_input(
        "Max Budget ($)", min_value=0.0, value=0.0, step=10.0,
        help="Set to 0 for no budget limit",
    )

    top_n = st.slider("Recommendations to show", min_value=3, max_value=10, value=6)

    st.divider()
    search_btn = st.button("🔍 Find Events", use_container_width=True, type="primary")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔍 EventScout")
st.markdown(
    "Discover the best events in your city — scored by AI and matched to your vibe."
)

# ── Run pipeline ───────────────────────────────────────────────────────────────
if search_btn:
    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()

    # Reset state on new search
    st.session_state.chat_history     = []
    st.session_state.recommendations  = None
    st.session_state.events_out       = None

    request = UserRequest(
        city              = city.strip(),
        state_code        = state_code.strip() or None,
        country_code      = country_code.strip() or "US",
        start_date        = start_date,
        end_date          = end_date,
        event_description = event_description.strip(),
        budget_max        = budget_max if budget_max > 0 else None,
    )

    status = st.status("Running pipeline...", expanded=True)
    with status:
        # Stage 1 & 2: parallel fetch
        st.write("📡 Fetching events & weather in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f_events  = pool.submit(run_events_agent,  request)
            f_weather = pool.submit(run_weather_agent, request)
            events_out  = f_events.result()
            weather_out = f_weather.result()

        if events_out.error:
            st.error(f"Events API error: {events_out.error}")
        else:
            st.write(f"✅ Found **{events_out.total_found}** events")

        if weather_out.error:
            st.warning(f"Weather API warning: {weather_out.error}")
        else:
            st.write(f"✅ Weather data for **{len(weather_out.forecasts)}** days")

        if events_out.total_found == 0:
            status.update(label="No events found — try different filters.", state="error")
            st.stop()

        # Stage 3: LLM scoring
        st.write(f"🤖 Scoring {events_out.total_found} events with Claude...")
        recs = run_recommendation_agent(
            request, events_out, weather_out,
            top_n=top_n,
        )
        st.write(f"✅ Top **{len(recs.recommendations)}** recommendations ready")
        status.update(label="Done!", state="complete", expanded=False)

    st.session_state.recommendations = recs
    st.session_state.events_out      = events_out

# ── Recommendations ────────────────────────────────────────────────────────────
if st.session_state.recommendations:
    recs = st.session_state.recommendations
    req  = recs.request

    st.divider()
    st.subheader(
        f"Top {len(recs.recommendations)} Events in {req.city} "
        f"({req.start_date} → {req.end_date})"
    )

    for i, r in enumerate(recs.recommendations, 1):
        e, w = r.event, r.weather

        score_color = (
            "🟢" if r.relevance_score >= 75
            else "🟡" if r.relevance_score >= 50
            else "🔴"
        )
        price_str = (
            f"${e.price_min:.0f} – ${e.price_max:.0f}"
            if e.price_min or e.price_max else "Free / Not listed"
        )
        weather_str = (
            f"{w.description} · {w.temp_min_f:.0f}–{w.temp_max_f:.0f}°F · "
            f"Rain {w.precipitation_chance:.0f}%"
            if w else "No forecast"
        )

        with st.container(border=True):
            img_col, info_col = st.columns([1, 3])

            with img_col:
                if e.image_url:
                    st.image(e.image_url, use_container_width=True)
                else:
                    st.markdown("### 🔍")

            with info_col:
                title_col, score_col = st.columns([3, 1])
                with title_col:
                    st.markdown(f"### {i}. {e.event_name}")
                    badges = []
                    if e.category: badges.append(f"`{e.category}`")
                    if e.genre:    badges.append(f"`{e.genre}`")
                    badges.append("🌳 Outdoor" if e.is_outdoor else "🏢 Indoor")
                    badges.append("🗓️ Weekend" if e.is_weekend else "📅 Weekday")
                    st.markdown("  ".join(badges))
                with score_col:
                    st.metric(f"{score_color} Score", f"{r.relevance_score:.0f} / 100")

                d1, d2, d3 = st.columns(3)
                with d1:
                    st.markdown("**📅 Date**")
                    time_str = e.time[:5] if e.time not in ("TBD", "") else "TBD"
                    st.markdown(f"{e.date} @ {time_str}")
                with d2:
                    st.markdown("**📍 Venue**")
                    st.markdown(e.venue_name)
                    if e.venue_address:
                        st.caption(e.venue_address)
                with d3:
                    st.markdown("**💵 Price**")
                    st.markdown(price_str)

                st.markdown(f"**🌤️ Weather:** {weather_str}")
                st.caption(f"**Why recommended:** {r.score_reason}")

                if e.url:
                    st.link_button("🎫 Get Tickets", e.url)

    # ── Stats row ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Summary")
    m1, m2, m3, m4 = st.columns(4)
    events_out = st.session_state.events_out
    m1.metric("Total Events Found",   events_out.total_found)
    m2.metric("Recommendations",      len(recs.recommendations))
    m3.metric("Weekend Events",       sum(1 for r in recs.recommendations if r.event.is_weekend))
    avg = sum(r.relevance_score for r in recs.recommendations) / len(recs.recommendations)
    m4.metric("Avg Score",            f"{avg:.1f} / 100")

    # ── Q&A ────────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💬 Ask About These Events")
    st.markdown(
        "Ask anything — prices, directions, weather, which to pick, ticket links, etc."
    )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg.role):
            st.markdown(msg.content)

    user_question = st.chat_input("e.g. Which events are best for families?")
    if user_question:
        with st.chat_message("user"):
                st.markdown(user_question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                qa_resp = run_qa_agent(
                    QARequest(
                        recommendations      = recs,
                        conversation_history = st.session_state.chat_history,
                        user_question        = user_question,
                    ),
                )
            st.markdown(qa_resp.answer)
        st.session_state.chat_history = qa_resp.updated_history

else:
    # Empty state
    st.info("Configure your search in the sidebar and click **Find Events** to get started.", icon="👈")
    st.markdown("""
    ### How it works
    1. **Agent 1** fetches events from Ticketmaster API
    2. **Agent 2** fetches weather forecasts from Open-Meteo (free, no key needed)
    3. **Agent 3** uses Claude to score & rank events based on your preferences
    4. **Agent 4** answers your follow-up questions via AI chat
    """)
