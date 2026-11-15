# Free Model Aggregator

Original Python standard-library server for free-model-aggregator-01.

Run `python3 main.py` with MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE supplied by
the trial. `python3 main.py --list` probes the catalog and prints exactly the
catalog JSON registered by a normal run. Without a brief it prints an empty
catalog. No packages, account setup, gateway calls, or paid model calls are
needed. Run local unit and HTTP integration tests with
`python3 -m unittest -v test_main.py`.

The server probes each provider using its declared auth scheme and API wire.
It ranks listed free models by the brief's quality rating, filters capabilities
and context length, and uses stable provider/model ordering for ties. Requests
follow manifest order, augmented by requirements visible in the actual body.
A JSON-format request cannot be sent through an upstream wire that cannot
represent it. A provider is tried at most once per request. Refusals trigger
cross-provider failover and an exponential cooldown measured in request counts,
not wall time. Cooling providers remain emergency fallbacks if others fail.
No hidden provider budgets, failure schedules, response fixtures or practice
answers are embedded in the entry.

Both client APIs support JSON and SSE. All three upstream wires support JSON
and SSE. Translation preserves system text, images, tools, tool results, token
limits, stop sequences, JSON mode and usage. Tool-call argument deltas are
assembled before cross-wire rendering. The env is polled until /work is done,
then the server shuts down and exits zero.

Limitations: SSE is buffered before downstream emission, so this is not a
low-latency live token proxy. Catalog discovery expects the trial's single-page
listings. The HTTP timeout is two seconds per operation, appropriate to this
simulated environment rather than slow real model services. Nonstandard vendor
extensions and reasoning blocks are not generally translated. The two APIs
cannot express every vendor-specific feature identically.

The companion verification report and official-env process harness are outside
the entry. Docker was unavailable locally; local Python is 3.11, while the
competition uses Python 3.13. Native container grading and transcript gate
checks remain the parent's responsibility. This repository has not been
published or submitted and has no remote.
