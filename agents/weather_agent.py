import requests
from config import WMO_CODES, BAD_CODES
from models.schemas import UserRequest, DailyForecast, WeatherAgentOutput


def _geocode(city: str) -> tuple[float, float]:
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=8,
    )
    r.raise_for_status()
    results = r.json().get("results")
    if not results:
        raise ValueError(f"Cannot geocode city: {city!r}")
    return results[0]["latitude"], results[0]["longitude"]


def _c_to_f(c: float) -> float:
    return round(c * 9 / 5 + 32, 1)


def _kmh_to_mph(k: float) -> float:
    return round(k * 0.621371, 1)


def run_weather_agent(request: UserRequest) -> WeatherAgentOutput:
    """Agent 2 — fetch daily weather forecasts from Open-Meteo (no API key needed)."""
    try:
        lat, lon = _geocode(request.city)

        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude" : lat,
                "longitude": lon,
                "daily": [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "weathercode",
                    "precipitation_probability_max",
                    "windspeed_10m_max",
                ],
                "start_date"      : request.start_date.isoformat(),
                "end_date"        : request.end_date.isoformat(),
                "timezone"        : "auto",
                "temperature_unit": "celsius",
                "windspeed_unit"  : "kmh",
            },
            timeout=10,
        )
        resp.raise_for_status()
        d = resp.json()["daily"]

        forecasts = {}
        for i, dt in enumerate(d["time"]):
            wcode    = int(d["weathercode"][i] or 0)
            precip   = float(d["precipitation_probability_max"][i] or 0)
            wind_mph = _kmh_to_mph(float(d["windspeed_10m_max"][i] or 0))
            forecasts[dt] = DailyForecast(
                date                 = dt,
                temp_min_f           = _c_to_f(float(d["temperature_2m_min"][i] or 0)),
                temp_max_f           = _c_to_f(float(d["temperature_2m_max"][i] or 0)),
                description          = WMO_CODES.get(wcode, "Unknown"),
                precipitation_chance = precip,
                wind_speed_mph       = wind_mph,
                is_suitable_outdoor  = wcode not in BAD_CODES and precip < 50 and wind_mph < 25,
            )

        return WeatherAgentOutput(city=request.city, forecasts=forecasts)

    except Exception as exc:
        return WeatherAgentOutput(city=request.city, error=str(exc))
