# Trade Signal Engine Edge

Python edge worker that ingests market data, normalizes bars, computes indicators, and emits signal events.

## Stack

- Python 3.12+
- `uv` for dependency management
- pytest for tests

## Run

```bash
make run
```

## Test

```bash
make test
```

## Environment

- `MARKET_TZ`: market timezone, default `America/New_York`
- `MARKET_HOLIDAYS`: comma-separated `YYYY-MM-DD` holiday dates
- `MARKET_EARLY_CLOSES`: comma-separated `YYYY-MM-DD=HH:MM` early close rules
- `DATA_PROVIDER`: provider name selected by configuration

## Notes

- The session calendar short-circuits execution when the market is closed.
- The provider contract is intentionally narrow so data vendors can be swapped without changing the engine.
