"""Tests of bin/weather (hal/weather.py) without the network: argument
parsing, which days a word means, and the report from a canned forecast.

Run: .venv/bin/python -m unittest tests.test_weather -v
"""

import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hal import weather  # noqa: E402

WEDNESDAY = date(2026, 11, 18)


def canned() -> dict:
    hours = [f"2026-11-19T{h:02d}:00" for h in range(24)]
    return {
        "current": {"time": "2026-11-18T23:30", "temperature_2m": 9.4, "weather_code": 3, "wind_speed_10m": 5.2},
        "daily": {
            "time": ["2026-11-18", "2026-11-19"],
            "weather_code": [3, 63],
            "temperature_2m_max": [12.1, 10.7], "temperature_2m_min": [8.0, 7.2],
            "precipitation_sum": [0.0, 5.4], "precipitation_probability_max": [10, 100],
            "wind_speed_10m_max": [9.0, 7.6], "wind_gusts_10m_max": [20.0, 16.6],
            "sunrise": ["2026-11-18T07:23", "2026-11-19T07:25"], "sunset": ["2026-11-18T16:07", "2026-11-19T16:06"],
        },
        "hourly": {
            "time": hours,
            "temperature_2m": [9.0, 9.2, 9.3, 9.4, 8.8, 8.0, 7.5, 7.6, 7.8, 8.0, 8.5, 9.2,
                               9.8, 10.4, 10.7, 10.5, 10.3, 9.7, 9.0, 8.3, 7.4, 7.2, 7.9, 8.4],
            "precipitation_probability": [96, 100, 100, 92, 70, 41, 18, 9, 6, 6, 7, 10,
                                          12, 11, 9, 10, 18, 29, 37, 40, 41, 41, 41, 41],
            "precipitation": [2.0, 1.5, 1.2, 0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "weather_code": [63, 53, 61, 51, 0, 1, 2, 3, 2, 1, 2, 2, 1, 1, 3, 2, 3, 2, 0, 0, 2, 0, 3, 3],
        },
    }


class ParseTest(unittest.TestCase):
    def test_place_and_day(self):
        self.assertEqual(weather.parse_args([]), ("London", "today"))
        self.assertEqual(weather.parse_args(["tomorrow"]), ("London", "tomorrow"))
        self.assertEqual(weather.parse_args(["Paris", "Friday"]), ("Paris", "friday"))
        self.assertEqual(weather.parse_args(["New", "York", "weekend"]), ("New York", "weekend"))
        self.assertEqual(weather.parse_args(["New York", "this", "weekend"]), ("New York", "weekend"))
        self.assertEqual(weather.parse_args(["Edinburgh", "2026-11-21"]), ("Edinburgh", "2026-11-21"))
        self.assertEqual(weather.parse_args(["St Andrews"]), ("St Andrews", "today"))

    def test_days(self):
        self.assertEqual(weather.wanted_days("tomorrow", WEDNESDAY), [date(2026, 11, 19)])
        self.assertEqual(weather.wanted_days("fri", WEDNESDAY), [date(2026, 11, 20)])
        self.assertEqual(weather.wanted_days("wednesday", WEDNESDAY), [WEDNESDAY])
        self.assertEqual(weather.wanted_days("weekend", WEDNESDAY), [date(2026, 11, 21), date(2026, 11, 22)])
        self.assertEqual(weather.wanted_days("weekend", date(2026, 11, 22)), [date(2026, 11, 22)])
        self.assertEqual(len(weather.wanted_days("week", WEDNESDAY)), 7)
        with self.assertRaises(weather.WeatherError):
            weather.wanted_days("someday", WEDNESDAY)


class ReportTest(unittest.TestCase):
    def test_tomorrow_with_periods(self):
        text = weather.report("London, England", canned(), "tomorrow")
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith("Weather for London, England"))
        self.assertIn("tomorrow, Thursday 19 November: rain; high 11 °C, low 7 °C; rain 5.4 mm", lines[1])
        self.assertIn("sunrise 07:25, sunset 16:06", lines[1])
        self.assertEqual(len(lines), 6, text)
        self.assertRegex(lines[2], r"night\s+rain\s+8–9 °C\s+chance of rain up to 100%\s+rain 5.4 mm \(00:00–03:00\)")
        self.assertRegex(lines[3], r"morning\s+overcast\s+.*dry$")
        self.assertNotIn("Now (", text, "current conditions only for today")

    def test_today_shows_now_and_missing_day_fails(self):
        text = weather.report("London, England", canned(), "today")
        self.assertIn("Now (23:30): overcast, 9.4 °C, wind 5 mph", text)
        with self.assertRaises(weather.WeatherError):
            weather.report("London, England", canned(), "2026-12-25")


if __name__ == "__main__":
    unittest.main()
