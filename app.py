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

    st.markdown("**What are you looking for?**")
    EVENT_CATEGORIES = [
        "🎵 Concerts & Live Music",
        "🏀 Sports",
        "🎭 Theater & Broadway",
        "😂 Comedy",
        "🎨 Arts & Exhibitions",
        "👨‍👩‍👧 Family & Kids",
        "🎉 Festivals & Fairs",
        "🍷 Food & Drink",
        "🎤 Hip-Hop & R&B",
        "🎸 Rock & Alternative",
        "🎷 Jazz & Blues",
        "💃 Dance & EDM",
        "🏛️ Cultural & Community",
        "🌿 Outdoor & Adventure",
    ]
    selected_categories = st.multiselect(
        "Categories (pick one or more)",
        options=EVENT_CATEGORIES,
        default=["🎵 Concerts & Live Music", "🏀 Sports"],
        placeholder="Choose categories...",
    )

    vibe_notes = st.text_area(
        "✏️ Vibe & preferences *",
        placeholder="e.g. date night, outdoor vibes, family-friendly, high energy, chill atmosphere…",
        height=75,
        help="Required — describe the atmosphere, indoor/outdoor preference, or anything else.",
    )
    venue_preference_clean = "No preference"  # inferred from vibe_notes by the LLM

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
    if not vibe_notes.strip():
        st.error("Please fill in your vibe & preferences field before searching.")
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
        event_description = (
            ", ".join(c.split(" ", 1)[-1] for c in selected_categories)
            + (". " + vibe_notes.strip())
        ) or "any events",
        venue_preference    = venue_preference_clean,
        vibe_notes          = vibe_notes.strip(),
        budget_max          = budget_max if budget_max > 0 else None,
        selected_categories = selected_categories,

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
    st.markdown("""
<style>
.hero-wrap {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.6rem;
}
.hero-sub {
    font-size: 1.2rem;
    color: #94a3b8;
    max-width: 600px;
    margin: 0 auto 2.5rem;
    line-height: 1.7;
}
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.2rem;
    max-width: 860px;
    margin: 0 auto 3rem;
}
.feature-card {
    background: linear-gradient(145deg, #1e1b4b22, #1e1b4b44);
    border: 1px solid #6366f133;
    border-radius: 16px;
    padding: 1.4rem 1.2rem;
    transition: transform 0.2s, border-color 0.2s;
}
.feature-card:hover {
    transform: translateY(-4px);
    border-color: #a855f766;
}
.feature-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.feature-title { font-size: 1rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.3rem; }
.feature-desc { font-size: 0.85rem; color: #94a3b8; line-height: 1.5; }

.steps-wrap {
    max-width: 680px;
    margin: 0 auto 2rem;
    text-align: left;
}
.steps-title {
    text-align: center;
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 1.2rem;
}
.step-row {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 1rem;
}
.step-num {
    width: 32px;
    height: 32px;
    min-width: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    font-weight: 700;
    color: white;
}
.step-text { font-size: 0.95rem; color: #cbd5e1; line-height: 1.5; padding-top: 4px; }
.step-text strong { color: #e2e8f0; }

.cta-hint {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    color: white;
    font-size: 0.95rem;
    font-weight: 600;
    padding: 0.65rem 1.8rem;
    border-radius: 999px;
    margin-top: 0.5rem;
    letter-spacing: 0.02em;
}
</style>

<div class="hero-wrap">
  <div class="hero-title">Discover Events<br>Made for You</div>
  <div class="hero-sub">
    Tell us your city, your vibe, and your budget —<br>
    EventScout finds and ranks the best live events happening near you,<br>scored by AI to match exactly what you&apos;re in the mood for.
  </div>

  <div class="feature-grid">
    <div class="feature-card">
      <div class="feature-icon">🎯</div>
      <div class="feature-title">Personalised Rankings</div>
      <div class="feature-desc">Every event scored against your taste, vibe, and budget — not generic popularity.</div>
    </div>
    <div class="feature-card">
      <div class="feature-icon">🌤️</div>
      <div class="feature-title">Weather-Aware</div>
      <div class="feature-desc">Live forecasts baked into your results so outdoor events are ranked based on real conditions.</div>
    </div>
    <div class="feature-card">
      <div class="feature-icon">💬</div>
      <div class="feature-title">Ask Anything</div>
      <div class="feature-desc">Chat with AI about your results — directions, ticket prices, which to pick, what to wear.</div>
    </div>
  </div>

  <div class="steps-wrap">
    <div class="steps-title">How to get started</div>
    <div class="step-row">
      <div class="step-num">1</div>
      <div class="step-text"><strong>Set your city & dates</strong> in the sidebar on the left</div>
    </div>
    <div class="step-row">
      <div class="step-num">2</div>
      <div class="step-text"><strong>Pick your event categories</strong> — concerts, sports, theater, comedy &amp; more</div>
    </div>
    <div class="step-row">
      <div class="step-num">3</div>
      <div class="step-text"><strong>Describe your vibe</strong> — e.g. <em>"date night, something lively, under $80"</em></div>
    </div>
    <div class="step-row">
      <div class="step-num">4</div>
      <div class="step-text"><strong>Hit Find Events</strong> and get your AI-ranked shortlist with reasons</div>
    </div>
    <div class="step-row">
      <div class="step-num">5</div>
      <div class="step-text"><strong>Ask follow-up questions</strong> in the chat — we&apos;ll help you decide</div>
    </div>
  </div>

  <span class="cta-hint">👈 Start by filling in the sidebar</span>
</div>
""", unsafe_allow_html=True)

