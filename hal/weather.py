"""`bin/weather [PLACE] [DAY]`: a forecast in one call, for quick spoken answers.

Asking Claude about the weather used to take several steps (load a web tool,
search, fetch, read). This prints everything needed for a spoken answer in one
call, from Open-Meteo (no key; only the place name and its coordinates are
sent).

    bin/weather                       today in London
    bin/weather tomorrow              tomorrow in London
    bin/weather Paris friday          Friday in Paris
    bin/weather "New York" weekend    Saturday and Sunday
    bin/weather Edinburgh week        the next seven days, a line each
    bin/weather London 2026-11-21     a given date (up to 15 days ahead)

Output: a heading line, then for each day a one-line summary and, for one or
two days, the night, morning, afternoon and evening. Temperatures in °C, wind
in mph, times local to the place. Exit status 1 with a message on failure.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta

DEFAULT_PLACE = "London"
_KNOWN = {"london": ("London, England", 51.5074, -0.1278, "Europe/London")}
_TIMEOUT = 8

# WMO weather interpretation codes, as Open-Meteo reports them; higher is worse,
# which picks the word for a period with mixed weather.
_CODES = {
    0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "violent showers", 85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail",
}
_PERIODS = (("night", 0, 6), ("morning", 6, 12), ("afternoon", 12, 18), ("evening", 18, 24))
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_DAY_WORDS = {"today", "tonight", "tomorrow", "weekend", "week"} | set(_WEEKDAYS) | {d[:3] for d in _WEEKDAYS}


class WeatherError(Exception):
    pass


def _get(url: str, params: dict) -> dict:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(full, headers={"User-Agent": "hal-weather/1"})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise WeatherError(f"could not reach Open-Meteo ({exc})") from exc


def parse_args(args: list[str]) -> tuple[str, str]:
    """(place, day word or ISO date) from the command line."""
    words = [a.strip() for a in args if a.strip()]
    day = "today"
    if words:
        last = words[-1].lower()
        if last in _DAY_WORDS or _iso_date(last):
            day = last
            words = words[:-1]
            while words and words[-1].lower() in ("this", "next", "on", "for", "in"):  # "Paris this weekend"
                words = words[:-1]
    place = " ".join(words).strip(" ,") or DEFAULT_PLACE
    return place, day


def _iso_date(text: str) -> date | None:
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def wanted_days(day: str, today: date) -> list[date]:
    """The dates a day word or ISO date stands for, counted from today (the place's)."""
    if day in ("today", "tonight"):
        return [today]
    if day == "tomorrow":
        return [today + timedelta(days=1)]
    if day == "week":
        return [today + timedelta(days=i) for i in range(7)]
    if day == "weekend":
        saturday = today + timedelta(days=(5 - today.weekday()) % 7)
        if today.weekday() == 6:  # on a Sunday, "the weekend" is today
            return [today]
        return [saturday, saturday + timedelta(days=1)]
    named = next((i for i, d in enumerate(_WEEKDAYS) if day in (d, d[:3])), None)
    if named is not None:
        return [today + timedelta(days=(named - today.weekday()) % 7)]
    iso = _iso_date(day)
    if iso:
        return [iso]
    raise WeatherError(f"unknown day {day!r}")


def locate(place: str) -> tuple[str, float, float, str]:
    """(name, latitude, longitude, time zone) for a place name."""
    known = _KNOWN.get(place.lower())
    if known:
        return known
    data = _get("https://geocoding-api.open-meteo.com/v1/search",
                {"name": place, "count": 1, "language": "en", "format": "json"})
    results = data.get("results") or []
    if not results:
        raise WeatherError(f"no place called {place!r} found")
    r = results[0]
    name = ", ".join(str(p) for p in (r.get("name"), r.get("admin1") if r.get("country_code") in ("GB", "US") else None,
                                      r.get("country")) if p)
    return name, float(r["latitude"]), float(r["longitude"]), str(r.get("timezone") or "auto")


def fetch(lat: float, lon: float, tz: str, days: int) -> dict:
    return _get("https://api.open-meteo.com/v1/forecast", {
        "latitude": lat, "longitude": lon, "timezone": tz, "forecast_days": min(16, max(1, days)),
        "wind_speed_unit": "mph",
        "current": "temperature_2m,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,"
                 "precipitation_probability_max,wind_speed_10m_max,wind_gusts_10m_max,sunrise,sunset",
        "hourly": "temperature_2m,precipitation_probability,precipitation,weather_code",
    })


def _word(code) -> str:
    try:
        return _CODES.get(int(code), f"code {code}")
    except (TypeError, ValueError):
        return "unknown"


def _num(x, digits=0) -> str:
    if x is None:
        return "?"
    value = round(float(x), digits)
    return f"{value:.{digits}f}" if digits else str(int(value))


def _wind(mph) -> str:
    if mph is None:
        return "?"
    mph = float(mph)
    return ("calm" if mph < 3 else "light" if mph < 12 else "moderate" if mph < 20
            else "strong" if mph < 32 else "gale-force")


def _clock(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%H:%M")
    except (TypeError, ValueError):
        return "?"


def _day_name(d: date, today: date) -> str:
    label = "today" if d == today else "tomorrow" if d == today + timedelta(days=1) else d.strftime("%A")
    return f"{label}, {d.strftime('%A')} {d.day} {d.strftime('%B')}" if label in ("today", "tomorrow") \
        else f"{label} {d.day} {d.strftime('%B')}"


def periods(forecast: dict, d: date) -> list[str]:
    """One line for each quarter of the day: the worst weather in it, the
    temperature range, and the rain."""
    hourly = forecast.get("hourly") or {}
    times = hourly.get("time") or []
    lines = []
    for name, start, end in _PERIODS:
        idx = [i for i, t in enumerate(times) if t.startswith(d.isoformat()) and start <= int(t[11:13]) < end]
        if not idx:
            continue

        def col(key):
            values = hourly.get(key) or []
            return [values[i] for i in idx if i < len(values) and values[i] is not None]

        temps, codes = col("temperature_2m"), col("weather_code")
        chance, rain = col("precipitation_probability"), col("precipitation")
        worst = max(codes) if codes else None
        wet = [times[i][11:16] for i in idx if (hourly.get("precipitation") or [0] * (i + 1))[i]]
        when = (wet[0] if wet and wet[0] == wet[-1] else f"{wet[0]}–{wet[-1]}") if wet else ""
        rain_text = f"rain {sum(rain):.1f} mm ({when})" if rain and sum(rain) >= 0.1 and wet else "dry"
        low, high = (_num(min(temps)), _num(max(temps))) if temps else ("?", "?")
        span = low if low == high else f"{low}–{high}"
        lines.append(f"  {name:<9} {_word(worst):<20} {span + ' °C':<10} chance of rain up to "
                     f"{_num(max(chance)) if chance else '?'}%   {rain_text}")
    return lines


def report(name: str, forecast: dict, day_word: str, today: date | None = None) -> str:
    daily = forecast.get("daily") or {}
    dates = [date.fromisoformat(t) for t in daily.get("time") or []]
    if today is None:
        current = (forecast.get("current") or {}).get("time")
        today = datetime.fromisoformat(current).date() if current else (dates[0] if dates else date.today())
    want = wanted_days(day_word, today)
    out = [f"Weather for {name} (Open-Meteo; times local; °C, mph)"]
    current = forecast.get("current") or {}
    if today in want and current.get("temperature_2m") is not None:
        out.append(f"Now ({_clock(current.get('time'))}): {_word(current.get('weather_code'))}, "
                   f"{_num(current['temperature_2m'], 1)} °C, wind {_num(current.get('wind_speed_10m'))} mph")
    detailed = len(want) <= 2
    found = False
    for d in want:
        if d not in dates:
            continue
        found = True
        i = dates.index(d)

        def day(key):
            values = daily.get(key) or []
            return values[i] if i < len(values) else None

        out.append(f"{_day_name(d, today)}: {_word(day('weather_code'))}; high {_num(day('temperature_2m_max'))} °C, "
                   f"low {_num(day('temperature_2m_min'))} °C; rain {_num(day('precipitation_sum'), 1)} mm, "
                   f"chance up to {_num(day('precipitation_probability_max'))}%; "
                   f"wind {_wind(day('wind_speed_10m_max'))} ({_num(day('wind_speed_10m_max'))} mph, "
                   f"gusts {_num(day('wind_gusts_10m_max'))}); "
                   f"sunrise {_clock(day('sunrise'))}, sunset {_clock(day('sunset'))}")
        if detailed:
            out.extend(periods(forecast, d))
    if not found:
        raise WeatherError(f"no forecast for {', '.join(d.isoformat() for d in want)} (Open-Meteo covers 16 days)")
    return "\n".join(out)


def main(args: list[str]) -> int:
    if args and args[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    try:
        place, day = parse_args(args)
        name, lat, lon, tz = locate(place)
        horizon = 7
        iso = _iso_date(day)
        if iso:
            horizon = (iso - date.today()).days + 2
        elif day in ("week", "weekend") or day in _WEEKDAYS or day in {d[:3] for d in _WEEKDAYS}:
            horizon = 8
        print(report(name, fetch(lat, lon, tz, horizon), day))
        return 0
    except WeatherError as exc:
        print(f"weather: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
