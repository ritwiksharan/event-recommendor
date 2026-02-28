import requests
from datetime import datetime
from config import TICKETMASTER_API_KEY, TICKETMASTER_BASE, OUTDOOR_KEYWORDS
from models.schemas import UserRequest, EventResult, EventAgentOutput


def _is_weekend(date_str: str) -> bool:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").weekday() in {4, 5, 6}
    except ValueError:
        return False


def _is_outdoor(venue_name: str) -> bool:
    return any(kw in venue_name.lower() for kw in OUTDOOR_KEYWORDS)


def _parse_event(raw: dict) -> EventResult:
    venue_name = venue_address = venue_city = venue_state = ""
    venue_lat = venue_lon = 0.0

    if "_embedded" in raw and "venues" in raw["_embedded"]:
        v             = raw["_embedded"]["venues"][0]
        venue_name    = v.get("name", "")
        venue_address = v.get("address", {}).get("line1", "")
        venue_city    = v.get("city", {}).get("name", "")
        venue_state   = v.get("state", {}).get("stateCode", "")
        venue_lat     = float(v.get("location", {}).get("latitude",  0) or 0)
        venue_lon     = float(v.get("location", {}).get("longitude", 0) or 0)

    price_min = price_max = 0.0
    if raw.get("priceRanges"):
        price_min = float(raw["priceRanges"][0].get("min", 0) or 0)
        price_max = float(raw["priceRanges"][0].get("max", 0) or 0)

    category = genre = ""
    if raw.get("classifications"):
        c        = raw["classifications"][0]
        category = c.get("segment", {}).get("name", "")
        genre    = c.get("genre",   {}).get("name", "")

    dates      = raw.get("dates", {}).get("start", {})
    event_date = dates.get("localDate", "")
    event_time = dates.get("localTime", "TBD")
    image_url  = raw["images"][0]["url"] if raw.get("images") else ""

    description = (
        raw.get("description", "")
        or raw.get("info", "")
        or raw.get("pleaseNote", "")
        or ""
    )

    return EventResult(
        event_id        = raw.get("id", ""),
        event_name      = raw.get("name", ""),
        description     = description,
        date            = event_date,
        time            = event_time,
        venue_name      = venue_name,
        venue_address   = venue_address,
        venue_city      = venue_city,
        venue_state     = venue_state,
        venue_latitude  = venue_lat,
        venue_longitude = venue_lon,
        price_min       = price_min,
        price_max       = price_max,
        category        = category,
        genre           = genre,
        url             = raw.get("url", ""),
        image_url       = image_url,
        is_weekend      = _is_weekend(event_date),
        is_outdoor      = _is_outdoor(venue_name),
    )


def run_events_agent(request: UserRequest) -> EventAgentOutput:
    """Agent 1 — paginate Ticketmaster and return all matching events."""
    all_raw, page, size = [], 0, 200

    params = {
        "apikey"       : TICKETMASTER_API_KEY,
        "city"         : request.city,
        "countryCode"  : request.country_code,
        "startDateTime": f"{request.start_date}T00:00:00Z",
        "endDateTime"  : f"{request.end_date}T23:59:59Z",
        "size"         : size,
        "sort"         : "date,asc",
    }
    if request.state_code: params["stateCode"] = request.state_code
    if request.budget_max: params["priceMax"]  = request.budget_max

    try:
        while True:
            params["page"] = page
            resp = requests.get(f"{TICKETMASTER_BASE}/events.json", params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if "fault" in data:
                raise RuntimeError(data["fault"]["faultstring"])
            if "_embedded" not in data or "events" not in data["_embedded"]:
                break

            all_raw.extend(data["_embedded"]["events"])
            total_pages = data.get("page", {}).get("totalPages", 1)
            page += 1

            if page >= total_pages or page * size >= 1000:
                break

        events = [_parse_event(e) for e in all_raw]
        return EventAgentOutput(request=request, events=events, total_found=len(events))

    except Exception as exc:
        return EventAgentOutput(request=request, error=str(exc))
