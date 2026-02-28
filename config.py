import os

TICKETMASTER_API_KEY = os.environ.get("TICKETMASTER_API_KEY", "wh6AHfUFqPPnFjBFEYVGA7VX3jWVR1Jx")
TICKETMASTER_BASE    = "https://app.ticketmaster.com/discovery/v2"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
CLAUDE_MODEL      = "claude-haiku-4-5-20251001"

OUTDOOR_KEYWORDS = {"stadium", "park", "amphitheater", "field", "grounds", "pavilion"}

WMO_CODES = {
    0: "Clear sky",       1: "Mainly clear",       2: "Partly cloudy",  3: "Overcast",
    45: "Fog",            48: "Rime fog",
    51: "Light drizzle",  53: "Moderate drizzle",  55: "Dense drizzle",
    61: "Slight rain",    63: "Moderate rain",      65: "Heavy rain",
    71: "Slight snow",    73: "Moderate snow",      75: "Heavy snow",
    80: "Slight showers", 81: "Moderate showers",  82: "Violent showers",
    95: "Thunderstorm",   96: "Thunderstorm+hail",  99: "Thunderstorm+heavy hail",
}

BAD_CODES = {45, 48, 51, 53, 55, 61, 63, 65, 71, 73, 75, 80, 81, 82, 95, 96, 99}
