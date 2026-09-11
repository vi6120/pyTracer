# Licensing

pyTracer is split-licensed by component. Most of it is permissive so you can
embed and run it freely; the hosted-server components are copyleft so a
competing hosted service has to share its changes.

## Which package is under which license

| Package | License | Why |
| --- | --- | --- |
| `pytracer-sdk` | Apache-2.0 | You embed it in your own (often proprietary) agent, so it must be permissive. |
| `pytracer-langgraph` | Apache-2.0 | Instrumentation adapter that runs inside your agent. |
| `pytracer-openai-agents` | Apache-2.0 | Instrumentation adapter that runs inside your agent. |
| `pytracer-crewai` | Apache-2.0 | Instrumentation adapter that runs inside your agent. |
| `pytracer-autogen` | Apache-2.0 | Instrumentation adapter that runs inside your agent. |
| `pytracer-replay` | Apache-2.0 | The test/replay engine you run locally and in CI. |
| `pytracer-runner` | Apache-2.0 | The cross-branch runner you run locally and in CI. |
| `pytracer-cli` | Apache-2.0 | You run it locally and in CI. |
| `pytracer-stores` | Apache-2.0 | Client library for reading traces/mocks. |
| `pytracer` (meta) | Apache-2.0 | Convenience bundle; see the note below. |
| **`pytracer-gateway`** | **AGPL-3.0-or-later** | The ingestion server someone would host as a service. |
| **`pytracer-registry`** | **AGPL-3.0-or-later** | The collaboration platform someone would host as a service. |

The full texts are in [LICENSE](LICENSE) (Apache-2.0, the default) and in each
AGPL package's own `LICENSE` file
([gateway](packages/gateway/LICENSE), [registry](packages/registry/LICENSE)).

## What this means in practice

- **Embedding the SDK or an adapter** in your own agent, even a closed-source or
  commercial one: Apache-2.0, no source obligations.
- **Running the CLI, replay, or runner** locally or in CI: Apache-2.0, no source
  obligations.
- **Using the hosted pyTracer service**: you are a user of the service and take
  on no source obligations.
- **Self-hosting the gateway or registry**: AGPL-3.0 permits this. If you modify
  them and make the modified version available to others over a network, you
  must offer those users the corresponding source of your modified version under
  the AGPL. Running an unmodified copy for your own organization does not
  trigger a public-source obligation.

## The meta-package

`pip install pytracer` is a convenience bundle that pulls in every component,
each under its own license (this is mere aggregation, not a single combined
license). Because the bundle includes `pytracer-gateway` and `pytracer-registry`,
installing it brings in AGPL-3.0 components alongside the Apache-2.0 ones. Install
only `pytracer-sdk` (Apache-2.0) if you want just the client in your agent.

## Contributions

Contributions are accepted under the license of the package they modify. A
Contributor License Agreement may be required before contributions are merged so
the project can keep its relicensing and commercial-licensing options open.
