from datetime import datetime, timezone

from trade_signal_edge.session_calendar import SessionCalendar


def test_session_calendar_allows_regular_session_hours() -> None:
    calendar = SessionCalendar()
    instant = datetime(2026, 4, 20, 14, 0, tzinfo=timezone.utc)

    assert calendar.is_open(instant)


def test_session_calendar_blocks_weekends() -> None:
    calendar = SessionCalendar()
    instant = datetime(2026, 4, 18, 14, 0, tzinfo=timezone.utc)

    assert not calendar.is_open(instant)


def test_session_calendar_blocks_early_close_after_cutoff() -> None:
    calendar = SessionCalendar(
        early_closes={
            datetime(2026, 12, 24, tzinfo=timezone.utc).date(): datetime(2026, 12, 24, 13, 0, tzinfo=timezone.utc).time()
        }
    )
    instant = datetime(2026, 12, 24, 19, 30, tzinfo=timezone.utc)

    assert not calendar.is_open(instant)
