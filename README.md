# EventScout 🎭

> AI-powered event discovery — find the best events in your city, scored by Claude and matched with live weather forecasts.

Built with a **multi-agent architecture**: four specialised agents collaborate to fetch events, check the weather, rank recommendations with an LLM, and answer follow-up questions in a chat interface.

---

## Demo

![EventScout UI](https://i.imgur.com/placeholder.png)

---

## Architecture

### High-Level System

```mermaid
flowchart TD
    User["👤 User\n(Streamlit UI)"]

    subgraph Agents["Multi-Agent Pipeline"]
        A1["🎟 Agent 1\nEvents Agent\n(Ticketmaster API)"]
        A2["🌤 Agent 2\nWeather Agent\n(Open-Meteo API)"]
        A3["🤖 Agent 3\nRecommendation Agent\n(Claude LLM)"]
        A4["💬 Agent 4\nQA Agent\n(Claude LLM)"]
    end

    subgraph External["External APIs"]
        TM["Ticketmaster\nDiscovery API"]
        OM["Open-Meteo\nForecast API"]
        GEO["Open-Meteo\nGeocoding API"]
        CL["Anthropic\nClaude API"]
    end

    User -->|"Search request"| A1
    User -->|"Search request"| A2
    A1 -->|"paginate events"| TM
    A2 -->|"geocode city"| GEO
    A2 -->|"fetch forecast"| OM
    A1 -->|"EventAgentOutput"| A3
    A2 -->|"WeatherAgentOutput"| A3
    A3 -->|"score & rank"| CL
    A3 -->|"RecommendationAgentOutput"| User
    User -->|"follow-up question"| A4
    A4 -->|"answer with context"| CL
    A4 -->|"QAResponse"| User
```

---

### Agent Pipeline (Sequential + Parallel)

```mermaid
sequenceDiagram
    participant U as User
    participant A1 as Events Agent
    participant A2 as Weather Agent
    participant A3 as Recommendation Agent
    participant A4 as QA Agent
    participant TM as Ticketmaster
    participant OM as Open-Meteo
    participant CL as Claude

    U->>+A1: UserRequest (city, dates, description, budget)
    U->>+A2: UserRequest (city, dates)

    Note over A1,A2: Agents 1 & 2 run in parallel

    A1->>TM: GET /events.json (paginated)
    TM-->>A1: raw event JSON
    A1-->>-U: EventAgentOutput (up to 1000 events)

    A2->>OM: GET /geocoding + /forecast
    OM-->>A2: daily forecasts
    A2-->>-U: WeatherAgentOutput (per-day forecasts)

    U->>+A3: EventAgentOutput + WeatherAgentOutput
    A3->>CL: Score top 50 events (0-100) with reasons
    CL-->>A3: JSON scores array
    A3-->>-U: RecommendationAgentOutput (top N ranked)

    loop Chat Q&A
        U->>+A4: question + conversation history
        A4->>CL: system context + history + question
        CL-->>A4: answer
        A4-->>-U: QAResponse (answer + updated history)
    end
```

---

### Data Model

```mermaid
classDiagram
    class UserRequest {
        +str city
        +str state_code
        +str country_code
        +date start_date
        +date end_date
        +str event_description
        +float budget_max
    }

    class EventResult {
        +str event_id
        +str event_name
        +str description
        +str date
        +str time
        +str venue_name
        +float price_min
        +float price_max
        +str category
        +str genre
        +str url
        +str image_url
        +bool is_weekend
        +bool is_outdoor
    }

    class DailyForecast {
        +str date
        +float temp_min_f
        +float temp_max_f
        +str description
        +float precipitation_chance
        +float wind_speed_mph
        +bool is_suitable_outdoor
    }

    class ScoredEvent {
        +float relevance_score
        +str score_reason
    }

    class EventAgentOutput {
        +list~EventResult~ events
        +int total_found
        +str error
    }

    class WeatherAgentOutput {
        +str city
        +dict~str,DailyForecast~ forecasts
        +str error
    }

    class RecommendationAgentOutput {
        +list~ScoredEvent~ recommendations
    }

    class QARequest {
        +str user_question
        +list~QAMessage~ conversation_history
    }

    class QAResponse {
        +str answer
        +list~QAMessage~ updated_history
    }

    UserRequest --o EventAgentOutput
    UserRequest --o WeatherAgentOutput
    UserRequest --o RecommendationAgentOutput
    EventResult --o EventAgentOutput
    EventResult --o ScoredEvent
    DailyForecast --o WeatherAgentOutput
    DailyForecast --o ScoredEvent
    ScoredEvent --o RecommendationAgentOutput
    RecommendationAgentOutput --o QARequest
```

---

## Project Structure

```
event-recommendor/
├── app.py                      # Streamlit UI
├── config.py                   # API keys, model name, constants
├── pyproject.toml              # Python deps (uv)
├── agents/
│   ├── events_agent.py         # Agent 1 — Ticketmaster event fetcher
│   ├── weather_agent.py        # Agent 2 — Open-Meteo weather fetcher
│   ├── recommendation_agent.py # Agent 3 — Claude LLM scorer & ranker
│   └── qa_agent.py             # Agent 4 — Claude LLM chat assistant
└── models/
    └── schemas.py              # Pydantic models shared across all agents
```

---

## Quickstart

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | |
| [uv](https://docs.astral.sh/uv/) | Fast Python package manager |
| Anthropic API key | [console.anthropic.com](https://console.anthropic.com) |
| Ticketmaster API key | [developer.ticketmaster.com](https://developer.ticketmaster.com) — free tier available |

### Run

```bash
# 1. Clone
git clone https://github.com/ritwiksharan/event-recommendor.git
cd event-recommendor

# 2. Install dependencies
uv sync

# 3. Run
ANTHROPIC_API_KEY="sk-ant-..." uv run streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### Optional env vars

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export TICKETMASTER_API_KEY="your-key"  # defaults to a demo key in config.py
```

---

## How It Works

### Agent 1 — Events Agent
Queries the **Ticketmaster Discovery API** with the user's city, date range, and optional budget filter. Paginates up to 1,000 results, parses venue/price/category data, and flags each event as `is_weekend` or `is_outdoor`.

### Agent 2 — Weather Agent
Geocodes the city via **Open-Meteo's geocoding API**, then fetches a daily forecast (temperature, precipitation chance, wind speed, WMO weather code) for each day in the date range. Marks each day `is_suitable_outdoor` based on weather codes + precipitation + wind.

### Agent 3 — Recommendation Agent
Sends the top 50 events (with their weather context) to **Claude** with a structured prompt. Claude returns a JSON array of scores (0–100) and one-sentence reasons, prioritising semantic match → budget fit → timing. Results are sorted and the top N are returned.

### Agent 4 — QA Agent
A stateless chat agent that receives the full recommendation context + conversation history on every call. Powered by **Claude**, it answers questions about specific events, prices, venues, weather, or ticket links.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| LLM | [Claude](https://anthropic.com) via [LiteLLM](https://github.com/BerriAI/litellm) |
| Events data | [Ticketmaster Discovery API](https://developer.ticketmaster.com/products-and-docs/apis/discovery-api/v2/) |
| Weather data | [Open-Meteo](https://open-meteo.com) (no API key required) |
| Data validation | [Pydantic v2](https://docs.pydantic.dev) |
| Package management | [uv](https://docs.astral.sh/uv/) |

---

## License

MIT
