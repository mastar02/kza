# Nexa Web

## Local browser fixtures

```sh
cargo run -p nexa-web -- --test-mode --bind 127.0.0.1:9500 \
  --allow-insecure-cookies --static-dir frontend/dist
```

Omit `--static-dir` to run the API and WebSocket fixtures before the frontend
bundle exists. The listening address is logged at startup. Port `0` assigns
an ephemeral port and uses it in the default allowed Origin.

Send `{"token":"test-operator"}` to `POST /api/admin/auth/login` with the browser
Origin. The test-only tokens are `test-operator`, `test-home`, and
`test-evaluator`. Each process creates a temporary authority DB, private
fixtures, and in-process daemon substitutes. The home account ID for a
retirement test is `test-home-user`.

The browser can send `live_hello` version 3 and echo the `live_reset` ID and
nonce in `live_reset_ack`. After `rest_ready`, the connection ID identifies
the same lease for protected projections in `X-KZA-Live-Connection`.
Views must clear on reset and policy close.

## Runtime configuration

`--config` accepts the configuration directory or `nexa.toml`. The configured
data directory must exist and the authority file must be owned by the runtime
user with mode 0600. `KZA_DASHBOARD_TOKEN` can bootstrap an operator credential
from the private process environment. The first Rust open clears Python-era
sessions. The three RPC sockets must belong to the runtime user and have mode
0600. Set `NEXA_WEB_ORIGIN` or `--origin` to the externally visible HTTPS
origin when the proxy uses a different hostname from the local listener.

At base SHA `0fe6f0db`, the daemons lack serving implementations and several
Python operations lack sufficient RPC fields. The affected routes return 503
when a required operation or scope cannot be represented. These include
routine history and proposals, list sync, reminder status and notice listings,
memory pagination, personal voice verification and revisioned consent, the
revisioned global voice switch, voice profile writes and grouping, and
evaluation grant preparation. The v3 receipt state is connection-local.
Durable privacy lineage, account migration, and projections involving fields
absent from Protobuf still require daemon or migration work.
