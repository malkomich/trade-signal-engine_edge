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

- `EDGE_SYMBOL`: ticker symbol to monitor, default `AAPL`
- `EDGE_BARS`: number of bars to fetch, default `60`
- `EDGE_PROVIDER`: provider selected by configuration, `synthetic` or `alpaca`
- `API_BASE_URL`: optional API endpoint used by the decision publisher
- `MARKET_HOLIDAYS`: comma-separated `YYYY-MM-DD` holiday dates
- `MARKET_EARLY_CLOSES`: comma-separated `YYYY-MM-DD=HH:MM` early close rules
- `ALPACA_API_KEY_ID`: required when `EDGE_PROVIDER=alpaca`
- `ALPACA_API_SECRET_KEY`: required when `EDGE_PROVIDER=alpaca`
- `ALPACA_DATA_FEED`: Alpaca data feed, default `iex`

## Notes

- The session calendar short-circuits execution when the market is closed.
- The calendar uses `America/New_York` by default and currently reads holiday and early-close overrides from environment variables.
- The provider contract is intentionally narrow so data vendors can be swapped without changing the engine.
