# Trade Signal Engine Edge

Python edge worker that ingests market data, normalizes bars, computes indicators, and emits signal events.

## Stack

- Python 3.12+
- `uv` for dependency management
- pytest for tests

## External integrations

- `Alpaca` provides the live market-data feed used to monitor stocks in near real time and to fetch the raw bars that feed the ingestion pipeline.
- `synthetic` is the local fallback provider for development and tests when Alpaca credentials are not available.

## Provider matrix

| Provider | Access model | Cost tier | Redistribution | Commercial use |
| --- | --- | --- | --- | --- |
| `synthetic` | Local-only generated data | Free | Safe for local development and tests because the bars are generated in-process | Development and test only |
| `alpaca` | External market-data API | Account-plan dependent | Review Alpaca terms before sharing raw bars or derived market data outside the app boundary | Subject to the Alpaca account plan and market-data terms |

## Run

```bash
make run
```

## Run in Docker

```bash
docker compose up -d --build
```

The compose file sets the project name to `trade-signal-engine-server`, so Dozzle groups the
edge container with the API service on the Raspberry Pi.

Inside that shared Compose project, the edge worker talks to the API container through the
service name `api`, so `API_BASE_URL` defaults to `http://api:8080` in Docker.

The worker also exposes a lightweight status page on `http://localhost:18081/edge` so the
Raspberry Pi proxy can route `https://tradesignalengine.backend.synapsesea.com/edge` to a
visible runtime snapshot while Dozzle streams the container logs.

By default the runtime watches a Nasdaq universe of `AAPL`, `AMZN`, `GOOGL`, `META`, `MSFT`,
`NVDA`, `PLTR`, and `TSLA` against the `IXIC` benchmark. Override the universe with
`EDGE_SYMBOLS` and the benchmark with `EDGE_BENCHMARK_SYMBOL` when needed.
The merge-to-`main` workflow now runs on the Raspberry Pi self-hosted runner, checks out the
repository on the Pi itself, and recreates the container with the same compose project name.

## Test

```bash
make test
```

## Environment

- `EDGE_SYMBOL`: primary ticker symbol to monitor, default `AAPL`
- `EDGE_SYMBOLS`: comma-separated symbol universe to monitor, default `AAPL,AMZN,GOOGL,META,MSFT,NVDA,PLTR,TSLA`
- `EDGE_BENCHMARK_SYMBOL`: benchmark symbol used as context for scoring, default `IXIC`
- `EDGE_SESSION_ID`: session identifier used when reading and writing window state, default `nasdaq-live`
- `EDGE_BARS`: number of bars to fetch, default `60`
- `EDGE_PROVIDER`: provider selected by configuration, `synthetic` or `alpaca`
- `API_BASE_URL`: optional API endpoint used by the decision publisher and session reader
- `EDGE_DEPLOYMENT_PROFILE`: deployment target label, default `pi`
- `EDGE_LOG_LEVEL`: runtime log level surfaced in the CLI report, default `INFO`
- `EDGE_METRICS_ENABLED`: enable the lightweight observability flag in the runtime report, default `false`
- `EDGE_SECRET_SOURCE`: where runtime secrets come from, default `environment`
- `EDGE_HTTP_PORT`: local HTTP status port exposed by the worker, default `8081`
- `MARKET_HOLIDAYS`: comma-separated `YYYY-MM-DD` holiday dates
- `MARKET_EARLY_CLOSES`: comma-separated `YYYY-MM-DD=HH:MM` early close rules
- `ALPACA_API_KEY_ID` or `ALPACA_API_KEY_ID_FILE`: required when `EDGE_PROVIDER=alpaca`
- `ALPACA_API_SECRET_KEY` or `ALPACA_API_SECRET_KEY_FILE`: required when `EDGE_PROVIDER=alpaca`
- `ALPACA_DATA_FEED`: Alpaca data feed, default `iex`

When running in Docker on the Raspberry Pi, point `ALPACA_API_KEY_ID_FILE` and
`ALPACA_API_SECRET_KEY_FILE` at root-owned files outside the repository so the credential values
stay out of the rendered compose config and the container inspect output.

## Notes

- The `alpaca` provider talks to the Alpaca market-data API. Its bars are normalized later by the ingestion pipeline before they reach the signal engine.
- The session calendar short-circuits execution when the market is closed.
- The calendar uses `America/New_York` by default and currently reads holiday and early-close overrides from environment variables.
- The provider contract is intentionally narrow so data vendors can be swapped without changing the engine.
- The selected provider is visible in the CLI output together with the full provider policy matrix so the tradeoff stays explicit.
- Runtime output includes the deployment profile, log level, metrics flag, and secret source so the Pi path stays explicit without checking secrets into the repository.
- `--watch` keeps the worker alive and prints a fresh evaluation every interval so Dozzle can stream logs continuously.
- The worker reads open windows from the API so entry and exit decisions stay stateful per symbol.
- The `IXIC` benchmark is loaded once per cycle and influences the entry and exit scoring for each stock in the universe.
- The `/edge` route can be proxied to the status page, while `/status` returns the same runtime snapshot as JSON for operators.
