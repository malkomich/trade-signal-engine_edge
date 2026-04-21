from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from zoneinfo import ZoneInfo
import os


def _parse_date(value: str) -> date:
    return date.fromisoformat(value.strip())


def _parse_time(value: str) -> time:
    hour, minute = value.strip().split(":", maxsplit=1)
    return time(hour=int(hour), minute=int(minute))


def _parse_early_close_item(value: str) -> tuple[date, time]:
    raw_date, raw_time = value.split("=", maxsplit=1)
    return _parse_date(raw_date), _parse_time(raw_time)


@dataclass(slots=True)
class SessionCalendar:
    timezone_name: str = "America/New_York"
    regular_open: time = time(9, 30)
    regular_close: time = time(16, 0)
    holiday_dates: frozenset[date] = field(default_factory=frozenset)
    early_closes: dict[date, time] = field(default_factory=dict)

    def is_open(self, instant: datetime) -> bool:
        local_time = instant.astimezone(ZoneInfo(self.timezone_name))
        session_day = local_time.date()
        if local_time.weekday() >= 5 or session_day in self.holiday_dates:
            return False
        close_time = self.early_closes.get(session_day, self.regular_close)
        return self.regular_open <= local_time.time() < close_time


def load_session_calendar() -> SessionCalendar:
    holidays = {
        _parse_date(item)
        for item in os.getenv("MARKET_HOLIDAYS", "").split(",")
        if item.strip()
    }
    early_closes = {
        session_date: close_time
        for item in os.getenv("MARKET_EARLY_CLOSES", "").split(",")
        if item.strip()
        for session_date, close_time in [_parse_early_close_item(item)]
    }
    return SessionCalendar(
        holiday_dates=frozenset(holidays),
        early_closes=early_closes,
    )
